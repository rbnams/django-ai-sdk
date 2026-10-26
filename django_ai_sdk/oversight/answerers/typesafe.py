"""TypeSafe's Jev as an answerer: typed questions are what it was built for.

Flag, Choice and Score lenses map onto its Noul, Choice and Score questions.
Needs django-ai-sdk[typesafe] and TYPESAFE_API_KEY; without either it abstains,
so a Cascade moves on rather than the turn failing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from django_ai_sdk.oversight.answerers.base import Answerer
from django_ai_sdk.oversight.types import Answer, Subject

if TYPE_CHECKING:
    from django_ai_sdk.oversight.lenses import Lens
    from django_ai_sdk.oversight.types import TurnView


def typesafe_state(lens: Lens, view: TurnView) -> dict[str, Any]:
    """The state Jev reads: only the parts of the turn the lens's subject covers."""
    state: dict[str, Any] = {}
    if lens.subject in {Subject.INPUT, Subject.EXCHANGE, Subject.SOURCES, Subject.TOOL_CALL}:
        state["user_message"] = view.input
    if lens.subject in {Subject.OUTPUT, Subject.EXCHANGE}:
        state["assistant_reply"] = view.output
    if lens.subject in {Subject.SOURCES, Subject.OUTPUT} and view.fresh_sources():
        state["sources"] = [
            {"title": s.title, "content": s.content[:1500]} for s in view.fresh_sources()[:8]
        ]
    if lens.subject is Subject.TOOL_CALL and view.tool_call:
        state["tool_call"] = {"name": view.tool_call.name, "arguments": view.tool_call.arguments}
    return state


class TypeSafe(Answerer):
    """Answer with Jev. `model` pins a Jev version; default is TYPESAFE_MODEL."""

    timeout = 20.0

    def __init__(self, model: str | None = None, *, name: str = "") -> None:
        self.model = model
        self.name = name or "typesafe"

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "model": self.model or ""}

    async def answer(self, lens: Lens, view: TurnView) -> Answer | None:
        from django_ai_sdk.typesafe.client import evaluate

        question = lens.typesafe_question()
        if question is None:
            return None
        summary = await evaluate(
            typesafe_state(lens, view), {lens.name: question}, model=self.model
        )
        if not summary:
            return None
        judgment = next(
            (j for j in summary.get("judgments") or [] if j.get("id") == lens.name), None
        )
        if judgment is None:
            return None
        value = judgment.get("value")
        confidence = judgment.get("confidence")
        if judgment.get("type") == "noul" and isinstance(value, float):
            # A noul is a probability of yes; its distance from 0.5 is its certainty.
            confidence = abs(value - 0.5) * 2
        elif judgment.get("type") == "score" and isinstance(value, int | float):
            levels = len(question.get("criteria") or []) or 1
            if value > 1 and levels > 1:
                value = value / (levels - 1)
        probabilities = judgment.get("probabilities") or {}
        if not confidence and probabilities:
            # A score's spread over its levels: the likeliest level's share is how sure.
            confidence = max(float(p or 0) for p in probabilities.values())
        return Answer(
            value=value,
            confidence=float(confidence) if confidence is not None else 0.8,
            model=str(summary.get("model") or self.model or ""),
            rationale=_rationale(judgment, question),
        )


def _rationale(judgment: dict[str, Any], question: dict[str, Any] | None = None) -> str:
    """The likeliest answers with their probabilities, a score's levels by their names."""
    probabilities = judgment.get("probabilities") or {}
    if not probabilities:
        return ""
    criteria = (question or {}).get("criteria")
    names = list(criteria) if isinstance(criteria, list) else []

    def name(label: str) -> str:
        if names and str(label).isdigit() and int(label) < len(names):
            return names[int(label)]
        return str(label)

    top = sorted(probabilities.items(), key=lambda item: -float(item[1] or 0))[:3]
    return ", ".join(f"{name(label)} {float(p):.2f}" for label, p in top)


__all__ = ["TypeSafe", "typesafe_state"]
