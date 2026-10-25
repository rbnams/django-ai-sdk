"""Workflow steps that call TypeSafe and gate on the judgments.

``EvaluateStep`` runs one System One request and stores the raw summary.
``WhenStep`` compares one judgment to a threshold and completes or skips.
Jev does not choose the next step; the author does, with numbers in config.
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Any

from django_ai_sdk.workflows.steps import Step, StepOutcome

if TYPE_CHECKING:
    from django_ai_sdk.workflows.steps import WorkflowContext

_QUESTION_TYPES = frozenset({"noul", "choice", "score"})
_OPS = {
    "lt": operator.lt,
    "lte": operator.le,
    "gt": operator.gt,
    "gte": operator.ge,
    "eq": operator.eq,
}


def _dig(value: Any, path: str) -> Any:
    """Walk ``a.b.c`` through dicts. Missing keys yield None."""
    current = value
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def resolve_binding(ctx: WorkflowContext, binding: str) -> Any:
    """Resolve ``input:name``, ``step:name``, or ``step:name.field``."""
    text = (binding or "").strip()
    if text.startswith("input:"):
        rest = text[len("input:") :]
        if not rest:
            raise ValueError(f"Empty input binding: {binding!r}")
        name, _, path = rest.partition(".")
        value = ctx.input(name)
        return _dig(value, path) if path else value
    if text.startswith("step:"):
        rest = text[len("step:") :]
        if not rest:
            raise ValueError(f"Empty step binding: {binding!r}")
        name, _, path = rest.partition(".")
        value = ctx.step(name)
        return _dig(value, path) if path else value
    raise ValueError(
        f"State binding {binding!r} must start with 'input:' or 'step:'."
    )


def validate_questions(questions: Any) -> dict[str, Any]:
    """Refuse a bad question map before the first run."""
    if not isinstance(questions, dict) or not questions:
        raise ValueError("EvaluateStep config.questions must be a non-empty object.")
    for qid, spec in questions.items():
        if not isinstance(qid, str) or not qid.strip():
            raise ValueError("Each question needs a non-empty id.")
        if not isinstance(spec, dict):
            raise ValueError(f"Question {qid!r} must be an object with a type.")
        kind = spec.get("type")
        if kind not in _QUESTION_TYPES:
            raise ValueError(
                f"Question {qid!r} type must be one of {sorted(_QUESTION_TYPES)}, "
                f"not {kind!r}."
            )
        if not str(spec.get("instructions") or "").strip():
            raise ValueError(f"Question {qid!r} needs instructions.")
        if "criteria" not in spec:
            raise ValueError(f"Question {qid!r} needs criteria.")
    return questions


def validate_state_bindings(state: Any) -> dict[str, str]:
    """Each value is an input: or step: binding."""
    if not isinstance(state, dict) or not state:
        raise ValueError("EvaluateStep config.state must be a non-empty object.")
    bindings: dict[str, str] = {}
    for key, binding in state.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Each state key needs a non-empty name.")
        if not isinstance(binding, str) or not binding.strip():
            raise ValueError(f"State key {key!r} needs an input: or step: binding.")
        prefix = binding.split(":", 1)[0]
        if prefix not in {"input", "step"}:
            raise ValueError(
                f"State key {key!r} binding {binding!r} must start with "
                "'input:' or 'step:'."
            )
        bindings[key] = binding.strip()
    return bindings


def judgment_value(summary: Any, judgment_id: str) -> Any:
    """The raw value for one judgment id in an evaluate output."""
    if not isinstance(summary, dict):
        return None
    judgments = summary.get("judgments")
    if not isinstance(judgments, list):
        return None
    for item in judgments:
        if isinstance(item, dict) and item.get("id") == judgment_id:
            return item.get("value")
    return None


class EvaluateStep(Step):
    """One System One request. Output is ``{model, judgments}``. Fail closed."""

    def validate_config(self) -> None:
        validate_state_bindings(self.config.get("state"))
        validate_questions(self.config.get("questions"))

    async def run(self, ctx: WorkflowContext) -> StepOutcome:
        from django_ai_sdk.typesafe import evaluate

        try:
            bindings = validate_state_bindings(self.config.get("state"))
            questions = validate_questions(self.config.get("questions"))
        except ValueError as exc:
            return StepOutcome(status="failed", detail=str(exc))

        state: dict[str, Any] = {}
        try:
            for key, binding in bindings.items():
                state[key] = resolve_binding(ctx, binding)
        except ValueError as exc:
            return StepOutcome(status="failed", detail=str(exc))

        summary = await evaluate(state=state, questions=questions)
        if summary is None:
            return StepOutcome(
                status="failed",
                detail="TypeSafe evaluate returned nothing.",
            )
        return StepOutcome(
            output=summary,
            detail=f"{len(summary.get('judgments') or [])} judgment(s)",
        )


class WhenStep(Step):
    """Complete when one judgment matches; skip when it does not."""

    def validate_config(self) -> None:
        judgment = self.config.get("judgment")
        if not isinstance(judgment, str) or not judgment.strip():
            raise ValueError("WhenStep config.judgment must be a non-empty string.")
        op = self.config.get("op")
        if op not in _OPS:
            raise ValueError(
                f"WhenStep config.op must be one of {sorted(_OPS)}, not {op!r}."
            )
        if "value" not in self.config:
            raise ValueError("WhenStep config.value is required.")

    def _source_summary(self, ctx: WorkflowContext) -> Any:
        """Prefer the first required step's output; else the named source."""
        if self.requires:
            return ctx.step(self.requires[0])
        source = self.config.get("source")
        if isinstance(source, str) and source.strip():
            return ctx.step(source.strip())
        return None

    async def run(self, ctx: WorkflowContext) -> StepOutcome:
        try:
            self.validate_config()
        except ValueError as exc:
            return StepOutcome(status="failed", detail=str(exc))

        judgment_id = str(self.config["judgment"]).strip()
        op_name = str(self.config["op"])
        expected = self.config["value"]
        summary = self._source_summary(ctx)
        if summary is None:
            return StepOutcome(
                status="failed",
                detail="WhenStep needs a completed evaluate step in requires.",
            )
        actual = judgment_value(summary, judgment_id)
        if actual is None:
            return StepOutcome(
                status="failed",
                detail=f"Judgment {judgment_id!r} is missing.",
            )

        compare = _OPS[op_name]
        try:
            passed = bool(compare(actual, expected))
        except TypeError:
            return StepOutcome(
                status="failed",
                detail=(
                    f"Cannot compare judgment {judgment_id!r} value {actual!r} "
                    f"with {expected!r} using {op_name}."
                ),
            )
        if not passed:
            return StepOutcome(
                status="skipped",
                detail=f"{judgment_id}={actual!r} not {op_name} {expected!r}",
            )
        return StepOutcome(
            output={"judgment": judgment_id, "value": actual, "op": op_name},
            detail=f"{judgment_id}={actual!r} {op_name} {expected!r}",
        )


__all__ = [
    "EvaluateStep",
    "WhenStep",
    "judgment_value",
    "resolve_binding",
    "validate_questions",
    "validate_state_bindings",
]
