"""TypeSafe's Jev as an answerer: typed questions are what it was built for.

Flag, Choice and Score lenses map onto its Noul, Choice and Score questions.
Needs django-ai-sdk[typesafe] and TYPESAFE_API_KEY; without either it abstains
(the oversight.W008 check says so). A failed request is an error, so FAIL_MODE
applies and reflect asks again.

Within one stage, questions about the same state go to Jev as one request.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import json
from typing import TYPE_CHECKING, Any, ClassVar

from django_ai_sdk.logger import get_logger
from django_ai_sdk.oversight.answerers.base import Answerer
from django_ai_sdk.oversight.answerers.llm import DEFAULT_LIMITS, HISTORY_ITEM_CHARS
from django_ai_sdk.oversight.answerers.pii import mask_secrets
from django_ai_sdk.oversight.types import Answer, Subject

if TYPE_CHECKING:
    from collections.abc import Iterator

    from django_ai_sdk.oversight.lenses import Lens
    from django_ai_sdk.oversight.types import TurnView

logger = get_logger(__name__)

# How long the first question of a stage waits for the others before the request goes.
BATCH_DELAY = 0.005


def typesafe_state(lens: Lens, view: TurnView) -> dict[str, Any]:
    """The state Jev reads: only the parts of the turn the lens's subject covers.

    Secrets are masked first: a pasted key must not reach Jev even when the gate
    redacts it for the chat model.
    """
    state: dict[str, Any] = {}
    if view.history and lens.subject in {Subject.INPUT, Subject.EXCHANGE}:
        state["history"] = [
            {"role": h.role, "content": mask_secrets(h.content[:HISTORY_ITEM_CHARS])}
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
        state["user_message"] = mask_secrets(view.input)
    if lens.subject in {Subject.OUTPUT, Subject.EXCHANGE}:
        state["assistant_reply"] = mask_secrets(view.output)
    if lens.subject in {Subject.SOURCES, Subject.OUTPUT} and (evidence := view.evidence()):
        state["sources"] = [
            {"title": mask_secrets(s.title), "content": mask_secrets(s.content[:1500])}
            for s in evidence[:8]
        ]
    if lens.subject is Subject.TOOL_CALL and view.tool_call:
        state["tool_call"] = {
            "name": view.tool_call.name,
            "arguments": _masked(view.tool_call.arguments),
        }
    return state


def _masked(value: Any) -> Any:
    if isinstance(value, str):
        return mask_secrets(value)
    if isinstance(value, dict):
        return {k: _masked(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_masked(v) for v in value]
    return value


class Batch:
    """One stage's Jev questions: those about the same state go out as one request.

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
        questions = {name: question for name, (question, _) in waiting.items()}
        try:
            summary: Any = await _request(state, questions, model or None, self.deadline)
        except Exception as exc:
            summary = exc
        for _, future in waiting.values():
            if future.done():
                continue
            if isinstance(summary, Exception):
                future.set_exception(summary)
            else:
                future.set_result(summary)


_batch: contextvars.ContextVar[Batch | None] = contextvars.ContextVar(
    "django_ai_sdk_oversight_jev_batch", default=None
)


@contextlib.contextmanager
def batched(deadline_ms: float | None = None) -> Iterator[Batch]:
    """Collect the Jev questions of lens tasks started inside the block into one Batch."""
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
    from django_ai_sdk.typesafe.client import request

    if deadline is None:
        return await request(state, questions, model=model)
    from typesafe_sdk import RetryPolicy

    # Inline, a retry would land after the deadline: one attempt, given the time left.
    left = max(0.05, deadline - asyncio.get_running_loop().time())
    return await request(
        state, questions, model=model, timeout=left, retry=RetryPolicy(max_retries=0)
    )


class TypeSafe(Answerer):
    """Answer with Jev. `model` pins a Jev version; default is TYPESAFE_MODEL."""

    metered: ClassVar[bool] = False
    external: ClassVar[str] = "TypeSafe"
    timeout = 20.0

    def __init__(self, model: str | None = None, *, name: str = "") -> None:
        self.model = model
        self.name = name or "typesafe"

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "model": self.model or ""}

    async def answer(self, lens: Lens, view: TurnView) -> Answer | None:
        from django_ai_sdk.typesafe.client import typesafe_missing

        question = lens.typesafe_question()
        if question is None or typesafe_missing():
            return None
        state = typesafe_state(lens, view)
        batch = _batch.get()
        try:
            if batch is None:
                summary = await _request(state, {lens.name: question}, self.model, None)
            else:
                summary = await batch.ask(self.model, state, lens.name, question)
        except Exception as exc:
            logger.warning("Jev failed on lens {}: {!r}", lens.name, exc)
            return Answer(error=f"jev: {type(exc).__name__}")
        return judgment_answer(summary, lens.name, question, self.model)


def judgment_answer(
    summary: dict[str, Any], name: str, question: dict[str, Any], model: str | None = None
) -> Answer | None:
    """Jev's judgment on question `name` as an Answer; None when it gave none."""
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
        confidence=float(confidence) if confidence is not None else 0.8,
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

    top = sorted(probabilities.items(), key=lambda item: -float(item[1] or 0))[:3]
    return ", ".join(f"{label_name(label)} {float(p):.2f}" for label, p in top)


__all__ = ["Batch", "TypeSafe", "batched", "judgment_answer", "typesafe_state"]
