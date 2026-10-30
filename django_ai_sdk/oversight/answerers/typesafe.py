"""TypeSafe as an answerer: typed questions are what it was built for.

TypeSafe is a hosted, commercial classification service; this answerer is optional.
It needs django-ai-sdk[typesafe] and TYPESAFE_API_KEY; without either it abstains
(the ai_sdk.oversight.W005 check says so). Flag, Choice and Score lenses map onto its
yes/no ("noul"), choice and score questions; an Entities lens with `per_source` is
answered source by source. A failed request is an `unavailable` error, so FAIL_MODE
applies and reflect asks again. Within one stage, questions about the same state go
to TypeSafe as one request.

What leaves for TypeSafe, per lens (typesafe_state, source_question; characters; a
longer text keeps its start and its end):

    lens subject      history         user_message  assistant_reply  sources [1]          tool_call
    input             last 6, 500 ea  30,000        -                -                    -
    exchange          last 6, 500 ea  30,000        30,000           -                    -
    output            -               30,000        30,000           yes, earlier too     -
    sources           -               30,000        -                yes, this turn only  -
    tool_call         -               30,000        -                -                    name, args [2]
    per_source [3]    -               30,000        30,000           one per question     -

    [1] each source and tool result: its title (200) and text, 8,000 together; all
        of them 32,000 together, past which the rest are left out (earlier turns'
        sources first, as they come last).
    [2] the tool's name (200) and its arguments, as sent to the tool; past 4,000
        characters of JSON, the start and end of that JSON.
    [3] an Entities lens with `per_source` (the quality pack's "taken from"): each
        source goes in a yes/no question of its own, within the same limits.

With them go each lens's question (its question and guidance, and what yes, no,
each label or each score level means) and the model id. One request carries at
most REQUEST_CHARS characters of the turn. Nothing else: no user id, thread id,
sources' urls or metadata. Secrets (the Secrets patterns) are masked before
anything is cut or sent; personal data is not.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import json
from typing import TYPE_CHECKING, Any, ClassVar

from django_ai_sdk.logger import get_logger
from django_ai_sdk.oversight.answerers.base import TOO_LONG, Answerer, clip, unavailable
from django_ai_sdk.oversight.answerers.llm import DEFAULT_LIMITS, HISTORY_ITEM_CHARS, TITLE_CHARS
from django_ai_sdk.oversight.answerers.pii import mask_secrets
from django_ai_sdk.oversight.lenses import Choice, Flag, Score
from django_ai_sdk.oversight.types import Answer, Source, Subject

if TYPE_CHECKING:
    from collections.abc import Generator

    from django_ai_sdk.oversight.lenses import Lens
    from django_ai_sdk.oversight.types import TurnView

logger = get_logger(__name__)

# How long the first question of a stage waits for the others before the request goes.
BATCH_DELAY = 0.005
# Seconds a request may take, retries included; inline stages give it what they have left.
TYPESAFE_TIMEOUT = 20.0
# The most characters of a turn one request carries. TypeSafe reads at most 32k tokens
# per question (the state and that question) and 64k per request; at three or four
# characters a token, this keeps a question with all its state inside that.
REQUEST_CHARS = 100_000
# The question, and the answer: the start and end of a longer text.
TEXT_CHARS = 30_000
# Sources: each (title and text) in full up to SOURCE_CHARS, all of them up to
# SOURCES_CHARS together.
SOURCE_CHARS = 8_000
SOURCES_CHARS = 32_000
# A tool call's arguments past this many characters (as JSON) are sent as cut JSON text.
TOOL_CALL_CHARS = 4_000
# Earlier messages: as many, and as long, as a model judge reads.
HISTORY_CHARS = DEFAULT_LIMITS["history"] * HISTORY_ITEM_CHARS
assert (
    2 * TEXT_CHARS + SOURCES_CHARS + TITLE_CHARS + TOOL_CALL_CHARS + HISTORY_CHARS <= REQUEST_CHARS
)
# The confidence of a judgment that states none.
UNSTATED_CONFIDENCE = 0.8
# How many of a judgment's likeliest answers its rationale names.
RATIONALE_ANSWERS = 3

# A noul is TypeSafe's probability of yes; it reads as yes or no as a Flag reads it.
_YES_NO = Flag(name="yes_no")


def says_yes(probability: float) -> bool:
    """Whether a noul's probability of yes means yes: as `Flag.parse` reads it."""
    return _YES_NO.parse(float(probability))


def typesafe_question(lens: Lens) -> dict[str, Any] | None:
    """`lens` as a TypeSafe question, or None when TypeSafe has no such type."""
    if isinstance(lens, Flag):
        return {
            "type": "noul",
            "instructions": typesafe_instructions(lens),
            "criteria": {"true": lens.yes, "false": lens.no},
        }
    if isinstance(lens, Choice):
        return {
            "type": "choice",
            "instructions": typesafe_instructions(lens),
            "criteria": {label: lens.criteria.get(label, label) for label in lens.labels},
        }
    if isinstance(lens, Score):
        return {
            "type": "score",
            "instructions": typesafe_instructions(lens),
            "criteria": [lens.low, lens.middle, lens.high],
        }
    return None


def typesafe_instructions(lens: Lens) -> str:
    return f"{lens.question}\n{lens.guidance}" if lens.guidance else lens.question


def labelled_sources(view: TurnView, *, earlier: bool = False) -> list[tuple[str, Source]]:
    """The turn's evidence as TypeSafe reads it, each labelled S1, S2, ... in order.

    Every source, title and text, is kept whole up to SOURCE_CHARS; once all of them
    together pass SOURCES_CHARS, the rest are left out: earlier turns' sources first,
    as they come last. Secrets are masked in each.
    """
    labelled: list[tuple[str, Source]] = []
    total = 0
    for source in view.evidence(earlier=earlier):
        title = _text(source.title, TITLE_CHARS)
        content = _text(source.content, SOURCE_CHARS - len(title))
        if labelled and total + len(title) + len(content) > SOURCES_CHARS:
            break
        total += len(title) + len(content)
        kept = source.model_copy(update={"title": title, "content": content})
        labelled.append((f"S{len(labelled) + 1}", kept))
    return labelled


def _typesafe_source(label: str, source: Source) -> dict[str, str]:
    return {"id": label, "title": source.title, "content": source.content}


def typesafe_state(lens: Lens, view: TurnView) -> dict[str, Any]:
    """The state TypeSafe reads: only the parts of the turn the lens's subject covers.

    Secrets are masked first: a pasted key must not reach TypeSafe even when the gate
    masks it for the chat model.
    """
    state: dict[str, Any] = {}
    if view.history and lens.subject in {Subject.INPUT, Subject.EXCHANGE}:
        state["history"] = [
            {"role": h.role, "content": _text(h.content, HISTORY_ITEM_CHARS)}
            for h in view.history[-DEFAULT_LIMITS["history"] :]
        ]
    # An answer is judged against what was asked, as a model judge reads it too.
    if lens.subject in {
        Subject.INPUT,
        Subject.OUTPUT,
        Subject.EXCHANGE,
        Subject.SOURCES,
        Subject.TOOL_CALL,
    }:
        state["user_message"] = _text(view.input, TEXT_CHARS)
    if lens.subject in {Subject.OUTPUT, Subject.EXCHANGE}:
        state["assistant_reply"] = _text(view.output, TEXT_CHARS)
    # An answer is judged against what earlier turns found too; a search on its own.
    if lens.subject in {Subject.SOURCES, Subject.OUTPUT} and (
        labelled := labelled_sources(view, earlier=lens.subject is Subject.OUTPUT)
    ):
        state["sources"] = [_typesafe_source(label, source) for label, source in labelled]
    if lens.subject is Subject.TOOL_CALL and view.tool_call:
        arguments = _masked(view.tool_call.arguments)
        compact = json.dumps(arguments, ensure_ascii=False, default=str)
        state["tool_call"] = {
            "name": _text(view.tool_call.name, TITLE_CHARS),
            "arguments": arguments
            if len(compact) <= TOOL_CALL_CHARS
            else clip(compact, TOOL_CALL_CHARS),
        }
    return state


def too_long(lens: Lens, view: TurnView) -> bool:
    """Whether TypeSafe would get only part of the text `lens` judges: then it is not asked.

    The question, the answer, both, or a tool call's arguments; what is sent beside
    them (earlier messages, sources, the question beside an answer) is cut to fit.
    """
    judged: list[tuple[str, int]] = []
    if lens.subject in {Subject.INPUT, Subject.EXCHANGE}:
        judged.append((view.input, TEXT_CHARS))
    if lens.subject in {Subject.OUTPUT, Subject.EXCHANGE} or getattr(lens, "per_source", False):
        judged.append((view.output, TEXT_CHARS))
    if lens.subject is Subject.TOOL_CALL and view.tool_call:
        compact = json.dumps(_masked(view.tool_call.arguments), ensure_ascii=False, default=str)
        judged.append((compact, TOOL_CALL_CHARS))
    return any(len(text) > limit and len(mask_secrets(text)) > limit for text, limit in judged)


def _text(text: str, limit: int) -> str:
    """Turn text as TypeSafe reads it: secrets masked, then the start and end of it."""
    return clip(mask_secrets(text or ""), limit)


def _masked(value: Any) -> Any:
    if isinstance(value, str):
        return mask_secrets(value)
    if isinstance(value, dict):
        return {k: _masked(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_masked(v) for v in value]
    return value


class Batch:
    """One stage's TypeSafe questions: those about the same state go out as one request.

    The flush runs as a task of its own, so a lens cancelled at the deadline never
    cancels the request the other lenses wait on. `deadline` is the loop time the
    stage ends at; None for a background stage.
    """

    def __init__(self, deadline: float | None = None) -> None:
        self.deadline = deadline
        self.groups: dict[tuple[str, str], tuple[Any, dict[str, Any]]] = {}
        self._flushes: set[asyncio.Task] = set()

    def ask(self, model: str | None, state: Any, name: str, question: Any) -> asyncio.Future:
        loop = asyncio.get_running_loop()
        if not self.groups:
            loop.call_later(BATCH_DELAY, self._start_flush)
        key = (model or "", json.dumps(state, sort_keys=True, default=str))
        future = loop.create_future()
        self.groups.setdefault(key, (state, {}))[1][name] = (question, future)
        return future

    def _start_flush(self) -> None:
        task = asyncio.ensure_future(self.flush())
        self._flushes.add(task)
        task.add_done_callback(self._flushes.discard)

    async def flush(self) -> None:
        groups, self.groups = self.groups, {}
        await asyncio.gather(
            *(self._send(model, state, waiting) for (model, _), (state, waiting) in groups.items())
        )

    async def _send(self, model: str, state: Any, waiting: dict[str, Any]) -> None:
        """Send one request for `waiting`; cancelled once no one waits for it any more.

        A lens cancelled at the deadline, or a stage cancelled whole, cancels its
        future: when every future is done, the request is cancelled too, so it never
        outlives the stage that asked.
        """
        questions = {name: question for name, (question, _) in waiting.items()}
        futures = [future for _, future in waiting.values()]
        request = asyncio.ensure_future(_request(state, questions, model or None, self.deadline))

        def abandon(_: asyncio.Future) -> None:
            if all(f.done() for f in futures) and not request.done():
                request.cancel()

        for future in futures:
            future.add_done_callback(abandon)
        abandon(request)
        try:
            summary: Any = await request
        except asyncio.CancelledError:
            if not request.cancelled():
                raise
            return
        except Exception as exc:
            summary = exc
        for future in futures:
            if future.done():
                continue
            if isinstance(summary, Exception):
                future.set_exception(summary)
            else:
                future.set_result(summary)


_batch: contextvars.ContextVar[Batch | None] = contextvars.ContextVar(
    "django_ai_sdk_oversight_typesafe_batch", default=None
)


@contextlib.contextmanager
def batched(deadline_ms: float | None = None) -> Generator[Batch]:
    """Collect the TypeSafe questions of lens tasks started inside the block into one Batch."""
    deadline = asyncio.get_running_loop().time() + deadline_ms / 1000 if deadline_ms else None
    batch = Batch(deadline)
    token = _batch.set(batch)
    try:
        yield batch
    finally:
        _batch.reset(token)


async def _request(
    state: Any, questions: dict[str, Any], model: str | None, deadline: float | None
) -> dict[str, Any]:
    from django_ai_sdk.oversight.answerers._typesafe_client import request

    if deadline is None:
        return await request(state, questions, model=model)
    from typesafe_sdk import RetryPolicy

    # Inline, a retry would land after the deadline: one attempt, given the time left.
    left = max(0.05, deadline - asyncio.get_running_loop().time())
    return await request(
        state, questions, model=model, timeout=left, retry=RetryPolicy(max_retries=0)
    )


class TypeSafe(Answerer):
    """Answer with TypeSafe, a hosted commercial classification service.

    Needs the `typesafe` extra (pip install 'django-ai-sdk[typesafe]') and a
    TYPESAFE_API_KEY; without either it abstains. `model` pins a model; the default
    is TYPESAFE_MODEL, else the SDK's.

    Flag, Choice and Score lenses become TypeSafe questions. An Entities lens with
    `per_source = True` (the quality pack's "taken from") is answered by rating each
    source: its value is the titles of the sources that answer yes, `items` each
    source's probability by source key, its confidence that of the least certain.
    Sources earlier turns found count too, marked "(found earlier)". With no source
    at all, TypeSafe is asked whether the reply states facts: one that does is taken
    from none, as is an empty reply. A probability of yes reads as yes or no exactly
    as `Flag.parse` reads it (0.5 and up is yes).

    One request carries at most REQUEST_CHARS characters of the turn: the question
    and the answer TEXT_CHARS each, cut to their start and end; sources SOURCE_CHARS
    each and SOURCES_CHARS together; tool arguments TOOL_CALL_CHARS. The module
    docstring lists what is sent for each lens.
    """

    metered = False
    external: ClassVar[str] = "TypeSafe"
    timeout = TYPESAFE_TIMEOUT

    # One stage's questions go out together: stages.run_lenses enters this around its lenses.
    batched = staticmethod(batched)

    def __init__(self, model: str | None = None, *, name: str = "") -> None:
        self.model = model
        self.name = name or "typesafe"

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "model": self.model or ""}

    def missing(self) -> str:
        from django_ai_sdk.oversight.answerers._typesafe_client import typesafe_missing

        return typesafe_missing()

    async def answer(self, lens: Lens, view: TurnView) -> Answer | None:
        from django_ai_sdk.oversight.answerers._typesafe_client import typesafe_missing

        if lens.kind == "entities" and getattr(lens, "per_source", False):
            return await self.per_source(lens, view)
        question = typesafe_question(lens)
        if question is None or typesafe_missing():
            return None
        if too_long(lens, view):
            return Answer(error=TOO_LONG)
        state = typesafe_state(lens, view)
        batch = _batch.get()
        try:
            if batch is None:
                summary = await _request(state, {lens.name: question}, self.model, None)
            else:
                summary = await batch.ask(self.model, state, lens.name, question)
        except Exception as exc:
            return _failed(lens, exc)
        return judgment_answer(summary, lens.name, question, self.model)

    async def per_source(self, lens: Lens, view: TurnView) -> Answer | None:
        """Which sources the answer is taken from, by asking TypeSafe about each one.

        One request: the state is the question and the answer, and each source is its
        own yes/no question with the source inside it (TypeSafe's pattern for one
        judgment over many items).
        """
        from django_ai_sdk.oversight.answerers._typesafe_client import typesafe_missing

        if not view.output.strip():
            return Answer(value=lens.parse([]), confidence=1.0, rationale="No answer.")
        if typesafe_missing():
            return None
        if too_long(lens, view):
            return Answer(error=TOO_LONG)
        labelled = labelled_sources(view, earlier=True)
        state = {
            "user_message": _text(view.input, TEXT_CHARS),
            "assistant_reply": _text(view.output, TEXT_CHARS),
        }
        questions = {label: source_question(source) for label, source in labelled}
        batch = _batch.get()
        try:
            summary = await _request(
                state,
                questions or {"facts": STATES_FACTS},
                self.model,
                batch.deadline if batch else None,
            )
        except Exception as exc:
            return _failed(lens, exc)
        model = str(summary.get("model") or self.model or "")
        if not labelled:
            return _unsourced(lens, summary, model)

        found: list[str] = []
        certainty: list[float] = []
        notes: list[str] = []
        items: dict[str, float] = {}
        for label, source in labelled:
            judged = judgment_answer(summary, label, questions[label], self.model)
            if judged is None or not isinstance(judged.value, int | float):
                continue
            yes = float(judged.value)
            certainty.append(judged.confidence)
            title = source.title or label
            if source.seeded:
                title = f"{title} (found earlier)"
            if says_yes(yes):
                found.append(title)
            notes.append(f"{label} {title}: {yes:.2f}")
            items[source.key or label] = round(yes, 3)
        if not certainty:
            return None
        return Answer(
            value=lens.parse(found),
            confidence=min(certainty),
            model=model,
            rationale="; ".join(notes),
            items=items,
        )


def _failed(lens: Lens, exc: Exception) -> Answer:
    # The type only: an SDK error's text can carry the response body.
    logger.warning("TypeSafe failed on lens {}: {}", lens.name, type(exc).__name__)
    return Answer(error=unavailable(exc))


STATES_FACTS: dict[str, Any] = {
    "type": "noul",
    "instructions": "Does `assistant_reply` state facts as its answer to `user_message`?",
    "criteria": {
        "true": "The reply gives a fact, rule, figure, date, amount or step as what is so.",
        "false": (
            "The reply only greets, thanks, asks something back, offers help, or says "
            "it doesn't know."
        ),
    },
}


def _unsourced(lens: Lens, summary: dict[str, Any], model: str) -> Answer | None:
    """An answer with no source to rest on: taken from none when it states facts."""
    judged = judgment_answer(summary, "facts", STATES_FACTS, model)
    if judged is None or not isinstance(judged.value, int | float) or not says_yes(judged.value):
        return None
    return Answer(
        value=lens.parse([]),
        confidence=judged.confidence,
        model=model,
        rationale=f"No source found; states facts: {float(judged.value):.2f}",
    )


def source_question(source: Source) -> dict[str, Any]:
    """TypeSafe's yes/no question about one source, the source inside it.

    `source` as labelled_sources gives it, already masked and cut; masked and cut
    again here, so a source from elsewhere is held to the same limits.
    """
    return {
        "type": "noul",
        "instructions": {
            "source": {
                "title": _text(source.title, TITLE_CHARS),
                "content": _text(source.content, SOURCE_CHARS),
            },
            "question": "Is `assistant_reply` taken from `source`?",
        },
        "criteria": {
            "true": "The reply states at least one fact that is in the source and relies on it.",
            "false": (
                "The reply's facts are not in the source: it only shares the topic, or it "
                "says something else."
            ),
        },
    }


def judgment_answer(
    summary: dict[str, Any], name: str, question: dict[str, Any], model: str | None = None
) -> Answer | None:
    """TypeSafe's judgment on question `name` as an Answer; None when it gave none."""
    judgment = next((j for j in summary.get("judgments") or [] if j.get("id") == name), None)
    if judgment is None:
        return None
    value = judgment.get("value")
    confidence = judgment.get("confidence")
    if judgment.get("type") == "noul" and isinstance(value, float):
        # A noul is a probability of yes; its distance from 0.5 is its certainty.
        confidence = abs(value - 0.5) * 2
    elif judgment.get("type") == "score" and isinstance(value, int | float):
        # A score is a level index: 0 to levels - 1, so 1 of 0..2 is the middle.
        levels = len(judgment.get("legend") or {}) or len(question.get("criteria") or [])
        if levels > 1:
            value = min(1.0, max(0.0, value / (levels - 1)))
    probabilities = judgment.get("probabilities") or {}
    if not confidence and probabilities:
        # A score's spread over its levels: the likeliest level's share is how sure.
        confidence = max(float(p or 0) for p in probabilities.values())
    return Answer(
        value=value,
        confidence=float(confidence) if confidence is not None else UNSTATED_CONFIDENCE,
        model=str(summary.get("model") or model or ""),
        rationale=_rationale(judgment, question),
    )


def _rationale(judgment: dict[str, Any], question: dict[str, Any] | None = None) -> str:
    """The likeliest answers with their probabilities, a score's levels by their names."""
    probabilities = judgment.get("probabilities") or {}
    if not probabilities:
        value = judgment.get("value")
        if judgment.get("type") == "noul" and isinstance(value, int | float):
            return f"yes {float(value):.2f}"
        return ""
    criteria = (question or {}).get("criteria")
    names = list(criteria) if isinstance(criteria, list) else []

    def label_name(label: str) -> str:
        if names and str(label).isdigit() and int(label) < len(names):
            return names[int(label)]
        return str(label)

    top = sorted(probabilities.items(), key=lambda item: -float(item[1] or 0))[:RATIONALE_ANSWERS]
    return ", ".join(f"{label_name(label)} {float(p):.2f}" for label, p in top)


__all__ = [
    "REQUEST_CHARS",
    "Batch",
    "TypeSafe",
    "batched",
    "judgment_answer",
    "labelled_sources",
    "says_yes",
    "source_question",
    "typesafe_instructions",
    "typesafe_question",
    "typesafe_state",
]
