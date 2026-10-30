"""Classifiers: models that answer typed questions (yes/no, a label, a score) fast.

The shared half of every such answerer: what the model reads of a turn (only what
the lens's subject covers, secrets masked, long texts cut to their start and end),
when a text is too long to judge at all, one request per stage for the questions
about the same state, and the per-source yes/no behind a "taken from" lens. A
subclass says how a lens is asked and how a reply reads:

    class MyClassifier(Classifier):
        external = "My classifier"            # who receives turn content

        def question(self, lens):             # None: this lens is not for me
            return {"kind": lens.kind, "question": lens.question}  # whatever your API wants

        async def classify(self, state, questions, *, timeout=None):
            reply = await my_client.classify(state, questions, timeout=timeout)
            return {name: Answer(value=p, confidence=certainty(p)) for name, p in reply.items()}

`classify` answers each question as the lens reads values: a yes/no as a bool or
a probability of yes (0.5 and up is yes; `certainty` turns it into a confidence),
a label as the label, a score from 0 to 1. A question it leaves out is unanswered,
so the next answerer is asked. TypeSafe (`answerers.typesafe`) is the worked example.

What a classifier reads, per lens (state; characters; a longer text keeps its start
and its end). The limits are class attributes, so a subclass sets its own:

    lens subject      history         user_message  assistant_reply  sources [1]          tool_call
    input             last 6, 500 ea  TEXT_CHARS    -                -                    -
    exchange          last 6, 500 ea  TEXT_CHARS    TEXT_CHARS       -                    -
    output            -               TEXT_CHARS    TEXT_CHARS       yes, earlier too     -
    sources           -               TEXT_CHARS    -                yes, this turn only  -
    tool_call         -               TEXT_CHARS    -                -                    name, args [2]
    per_source [3]    -               TEXT_CHARS    TEXT_CHARS       one per question     -

    [1] each source and tool result: its title (200) and text, SOURCE_CHARS
        together; all of them SOURCES_CHARS together, past which the rest are left
        out (earlier turns' sources first, as they come last).
    [2] the tool's name (200) and its arguments, as sent to the tool; past
        TOOL_CALL_CHARS characters of JSON, the start and end of that JSON.
    [3] an Entities lens with `per_source` (the quality pack's "taken from"): each
        source goes in a yes/no question of its own (`source_question`).

Nothing else: no user id, thread id, sources' urls or metadata. Secrets (the
Secrets patterns) are masked before anything is cut or sent; personal data is not.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import json
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from django_ai_sdk.logger import get_logger
from django_ai_sdk.oversight.answerers.base import TOO_LONG, Answerer, clip, unavailable
from django_ai_sdk.oversight.answerers.llm import DEFAULT_LIMITS, HISTORY_ITEM_CHARS, TITLE_CHARS
from django_ai_sdk.oversight.answerers.pii import mask_secrets
from django_ai_sdk.oversight.lenses import Flag
from django_ai_sdk.oversight.types import Answer, Source, Subject

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Generator

    from django_ai_sdk.oversight.lenses import Lens
    from django_ai_sdk.oversight.types import TurnView

logger = get_logger(__name__)

# How long the first question of a stage waits for the others before the request goes.
BATCH_DELAY = 0.005

# A yes/no's probability reads as yes or no exactly as a Flag lens reads it.
_YES_NO = Flag(name="yes_no")


def says_yes(probability: float) -> bool:
    """Whether a probability of yes means yes: as `Flag.parse` reads it (0.5 and up)."""
    return _YES_NO.parse(float(probability))


def certainty(probability: float) -> float:
    """How sure a probability of yes is: its distance from 0.5, from 0 to 1."""
    return min(1.0, abs(float(probability) - 0.5) * 2)


# Asked when an answer has no source at all: one that states facts is taken from none.
STATES_FACTS = Flag(
    name="facts",
    question="Does `assistant_reply` state facts as its answer to `user_message`?",
    yes="The reply gives a fact, rule, figure, date, amount or step as what is so.",
    no=(
        "The reply only greets, thanks, asks something back, offers help, or says it doesn't know."
    ),
)


class Batch:
    """One stage's classifier questions: those about the same state go out as one call.

    The flush runs as a task of its own, so a lens cancelled at the deadline never
    cancels the call the other lenses wait on. `deadline` is the loop time the
    stage ends at; None for a background stage.
    """

    def __init__(self, deadline: float | None = None) -> None:
        self.deadline = deadline
        self.groups: dict[tuple[str, str], tuple[Any, Any, dict[str, Any]]] = {}
        self._flushes: set[asyncio.Task] = set()

    def left(self) -> float | None:
        """Seconds until the deadline (at least 0.05), or None in the background."""
        if self.deadline is None:
            return None
        return max(0.05, self.deadline - asyncio.get_running_loop().time())

    def ask(
        self,
        key: str,
        classify: Callable[..., Awaitable[dict[str, Answer]]],
        state: Any,
        name: str,
        question: Any,
    ) -> asyncio.Future:
        loop = asyncio.get_running_loop()
        if not self.groups:
            loop.call_later(BATCH_DELAY, self._start_flush)
        group = (key, json.dumps(state, sort_keys=True, default=str))
        future = loop.create_future()
        self.groups.setdefault(group, (classify, state, {}))[2][name] = (question, future)
        return future

    def _start_flush(self) -> None:
        task = asyncio.ensure_future(self.flush())
        self._flushes.add(task)
        task.add_done_callback(self._flushes.discard)

    async def flush(self) -> None:
        groups, self.groups = self.groups, {}
        await asyncio.gather(
            *(self._send(classify, state, waiting) for classify, state, waiting in groups.values())
        )

    async def _send(self, classify: Any, state: Any, waiting: dict[str, Any]) -> None:
        """Send one call for `waiting`; cancelled once no one waits for it any more.

        A lens cancelled at the deadline, or a stage cancelled whole, cancels its
        future: when every future is done, the call is cancelled too, so it never
        outlives the stage that asked.
        """
        questions = {name: question for name, (question, _) in waiting.items()}
        futures = [future for _, future in waiting.values()]
        call = asyncio.ensure_future(classify(state, questions, timeout=self.left()))

        def abandon(_: asyncio.Future) -> None:
            if all(f.done() for f in futures) and not call.done():
                call.cancel()

        for future in futures:
            future.add_done_callback(abandon)
        abandon(call)
        try:
            result: Any = await call
        except asyncio.CancelledError:
            if not call.cancelled():
                raise
            return
        except Exception as exc:
            result = exc
        for future in futures:
            if future.done():
                continue
            if isinstance(result, Exception):
                future.set_exception(result)
            else:
                future.set_result(result)


_batch: contextvars.ContextVar[Batch | None] = contextvars.ContextVar(
    "django_ai_sdk_oversight_classifier_batch", default=None
)


@contextlib.contextmanager
def batched(deadline_ms: float | None = None) -> Generator[Batch]:
    """Collect the classifier questions of lens tasks started inside the block into one Batch."""
    deadline = asyncio.get_running_loop().time() + deadline_ms / 1000 if deadline_ms else None
    batch = Batch(deadline)
    token = _batch.set(batch)
    try:
        yield batch
    finally:
        _batch.reset(token)


class Classifier(Answerer):
    """Base for answerers that classify: yes/no, label and score lenses, and per source.

    Subclasses implement `question` and `classify`, and set `external`. Within one
    stage, questions about the same state go to `classify` together, as one call.
    A call that raises is an `unavailable` error, so FAIL_MODE applies and reflect
    asks again. See the module docstring for what is read.
    """

    metered = False
    external: ClassVar[str] = "classifier"
    timeout = 20.0

    # Characters of the question and the answer; of each source, and of all of them;
    # of a tool call's arguments as JSON; of earlier messages, all together.
    TEXT_CHARS: ClassVar[int] = 30_000
    SOURCE_CHARS: ClassVar[int] = 8_000
    SOURCES_CHARS: ClassVar[int] = 32_000
    TOOL_CALL_CHARS: ClassVar[int] = 4_000
    HISTORY: ClassVar[int] = DEFAULT_LIMITS["history"]

    # One stage's questions go out together: stages.run_lenses enters this around its lenses.
    batched = staticmethod(batched)

    @abstractmethod
    def question(self, lens: Lens) -> Any | None:
        """`lens` as this classifier's question, or None when it cannot answer that kind."""

    @abstractmethod
    async def classify(
        self, state: dict[str, Any], questions: dict[str, Any], *, timeout: float | None = None
    ) -> dict[str, Answer]:
        """Answer each of `questions` (by name) about `state`, in one call.

        `timeout` is set inline: the seconds left before the stage's deadline, so a
        retry would come too late. None in the background.
        """

    def batch_key(self) -> str:
        """What a stage's questions must share to go out in one call: the same configuration.

        Built from `describe()`, so two aliases of one class with different settings
        (a model, a label) never answer each other's questions.
        """
        described = {k: v for k, v in self.describe().items() if k != "name"}
        return f"{type(self).__qualname__}:{json.dumps(described, sort_keys=True, default=str)}"

    async def answer(self, lens: Lens, view: TurnView) -> Answer | None:
        if lens.kind == "entities" and getattr(lens, "per_source", False):
            return await self.per_source(lens, view)
        question = self.question(lens)
        if question is None or self.missing():
            return None
        if self.too_long(lens, view):
            return Answer(error=TOO_LONG)
        try:
            answers = await self._ask(self.state(lens, view), {lens.name: question})
        except Exception as exc:
            return self._failed(lens, exc)
        return answers.get(lens.name)

    async def _ask(
        self, state: dict[str, Any], questions: dict[str, Any], *, alone: bool = False
    ) -> dict[str, Answer]:
        """`classify` these questions, inside the stage's batch when there is one."""
        batch = _batch.get()
        if batch is None:
            return await self.classify(state, questions)
        if alone:
            return await self.classify(state, questions, timeout=batch.left())
        futures = [
            batch.ask(self.batch_key(), self.classify, state, name, question)
            for name, question in questions.items()
        ]
        results = await asyncio.gather(*futures)
        merged: dict[str, Answer] = {}
        for result in results:
            merged.update(result)
        return {name: merged[name] for name in questions if name in merged}

    def _failed(self, lens: Lens, exc: Exception) -> Answer:
        # The type only: a client's error text can carry the response body.
        logger.warning("{} failed on lens {}: {}", self.label(), lens.name, type(exc).__name__)
        return Answer(error=unavailable(exc))

    # What the classifier reads.

    def text(self, text: str, limit: int) -> str:
        """Turn text as the classifier reads it: secrets masked, then its start and end."""
        return clip(mask_secrets(text or ""), limit)

    def labelled_sources(
        self, view: TurnView, *, earlier: bool = False
    ) -> list[tuple[str, Source]]:
        """The turn's evidence, each labelled S1, S2, ... in order, masked and cut.

        Every source, title and text, is kept whole up to SOURCE_CHARS; once all of
        them together pass SOURCES_CHARS, the rest are left out: earlier turns'
        sources first, as they come last.
        """
        labelled: list[tuple[str, Source]] = []
        total = 0
        for source in view.evidence(earlier=earlier):
            title = self.text(source.title, TITLE_CHARS)
            content = self.text(source.content, self.SOURCE_CHARS - len(title))
            if labelled and total + len(title) + len(content) > self.SOURCES_CHARS:
                break
            total += len(title) + len(content)
            kept = source.model_copy(update={"title": title, "content": content})
            labelled.append((f"S{len(labelled) + 1}", kept))
        return labelled

    def state(self, lens: Lens, view: TurnView) -> dict[str, Any]:
        """What the classifier reads for `lens`: only the parts its subject covers."""
        subject = lens.subject
        state: dict[str, Any] = {}
        if view.history and subject in {Subject.INPUT, Subject.EXCHANGE}:
            state["history"] = [
                {"role": h.role, "content": self.text(h.content, HISTORY_ITEM_CHARS)}
                for h in view.history[-self.HISTORY :]
            ]
        # An answer is judged against what was asked, as a model judge reads it too.
        state["user_message"] = self.text(view.input, self.TEXT_CHARS)
        if subject in {Subject.OUTPUT, Subject.EXCHANGE}:
            state["assistant_reply"] = self.text(view.output, self.TEXT_CHARS)
        # An answer is judged against what earlier turns found too; a search on its own.
        if subject in {Subject.SOURCES, Subject.OUTPUT} and (
            labelled := self.labelled_sources(view, earlier=subject is Subject.OUTPUT)
        ):
            state["sources"] = [
                {"id": label, "title": source.title, "content": source.content}
                for label, source in labelled
            ]
        if subject is Subject.TOOL_CALL and view.tool_call:
            arguments = _masked(view.tool_call.arguments)
            compact = json.dumps(arguments, ensure_ascii=False, default=str)
            state["tool_call"] = {
                "name": self.text(view.tool_call.name, TITLE_CHARS),
                "arguments": arguments
                if len(compact) <= self.TOOL_CALL_CHARS
                else clip(compact, self.TOOL_CALL_CHARS),
            }
        return state

    def too_long(self, lens: Lens, view: TurnView) -> bool:
        """Whether the classifier would read only part of the text `lens` judges: then it is not asked.

        The question, the answer, both, or a tool call's arguments; what is read
        beside them (earlier messages, sources, the question beside an answer) is cut
        to fit.
        """
        judged: list[tuple[str, int]] = []
        if lens.subject in {Subject.INPUT, Subject.EXCHANGE}:
            judged.append((view.input, self.TEXT_CHARS))
        if lens.subject in {Subject.OUTPUT, Subject.EXCHANGE} or getattr(lens, "per_source", False):
            judged.append((view.output, self.TEXT_CHARS))
        if lens.subject is Subject.TOOL_CALL and view.tool_call:
            compact = json.dumps(_masked(view.tool_call.arguments), ensure_ascii=False, default=str)
            judged.append((compact, self.TOOL_CALL_CHARS))
        return any(len(text) > limit and len(mask_secrets(text)) > limit for text, limit in judged)

    # Which sources an answer is taken from.

    def source_question(self, source: Source) -> Any | None:
        """The yes/no question "is the answer taken from `source`?", the source inside it."""
        return self.question(
            Flag(
                name="taken_from",
                question="Is `assistant_reply` taken from this source?",
                guidance=f"Source: {source.title}\n{source.content}",
                yes="The reply states at least one fact that is in the source and relies on it.",
                no=(
                    "The reply's facts are not in the source: it only shares the topic, or "
                    "it says something else."
                ),
            )
        )

    async def per_source(self, lens: Lens, view: TurnView) -> Answer | None:
        """Which sources the answer is taken from: one yes/no per source, in one call.

        The value is the titles of the sources that answer yes ("(found earlier)" for
        an earlier turn's), `items` each source's probability by source key, the
        confidence that of the least certain. With no source, the classifier is asked
        whether the reply states facts: one that does is taken from none, as is an
        empty reply.
        """
        if not view.output.strip():
            return Answer(value=lens.parse([]), confidence=1.0, rationale="No answer.")
        if self.missing():
            return None
        if self.too_long(lens, view):
            return Answer(error=TOO_LONG)
        labelled = self.labelled_sources(view, earlier=True)
        state = {
            "user_message": self.text(view.input, self.TEXT_CHARS),
            "assistant_reply": self.text(view.output, self.TEXT_CHARS),
        }
        questions = {label: self.source_question(source) for label, source in labelled}
        questions = {label: q for label, q in questions.items() if q is not None}
        if labelled and not questions:
            return None
        try:
            if not labelled:
                facts = self.question(STATES_FACTS)
                if facts is None:
                    return None
                answers = await self._ask(state, {"facts": facts}, alone=True)
            else:
                answers = await self._ask(state, questions, alone=True)
        except Exception as exc:
            return self._failed(lens, exc)
        if not labelled:
            return _unsourced(lens, answers.get("facts"))

        found: list[str] = []
        sure: list[float] = []
        notes: list[str] = []
        items: dict[str, float] = {}
        model = ""
        for label, source in labelled:
            judged = answers.get(label)
            if judged is None or judged.error or not isinstance(judged.value, int | float):
                continue
            yes = float(judged.value)
            model = model or judged.model
            sure.append(judged.confidence)
            title = source.title or label
            if source.seeded:
                title = f"{title} (found earlier)"
            if says_yes(yes):
                found.append(title)
            notes.append(f"{label} {title}: {yes:.2f}")
            items[source.key or label] = round(yes, 3)
        if not sure:
            return None
        return Answer(
            value=lens.parse(found),
            confidence=min(sure),
            model=model,
            rationale="; ".join(notes),
            items=items,
        )


def _unsourced(lens: Lens, judged: Answer | None) -> Answer | None:
    """An answer with no source to rest on: taken from none when it states facts."""
    if judged is None or judged.error or not isinstance(judged.value, int | float):
        return None
    if not says_yes(judged.value):
        return None
    return Answer(
        value=lens.parse([]),
        confidence=judged.confidence,
        model=judged.model,
        rationale=f"No source found; states facts: {float(judged.value):.2f}",
    )


def _masked(value: Any) -> Any:
    if isinstance(value, str):
        return mask_secrets(value)
    if isinstance(value, dict):
        return {k: _masked(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_masked(v) for v in value]
    return value


__all__ = [
    "STATES_FACTS",
    "Batch",
    "Classifier",
    "batched",
    "certainty",
    "says_yes",
]
