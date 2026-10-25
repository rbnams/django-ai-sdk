"""EvaluateStep and WhenStep: TypeSafe judgments gate the graph."""

from __future__ import annotations

from typing import Any

import pytest
from django.core.exceptions import ImproperlyConfigured
from django.test import override_settings

from django_ai_sdk.workflows.definitions import compile_steps
from django_ai_sdk.workflows.schemas import StepDefinition, WorkflowDefinition
from django_ai_sdk.workflows.steps import StepOutcome, WorkflowContext
from django_ai_sdk.workflows.typesafe_steps import (
    EvaluateStep,
    WhenStep,
    judgment_value,
    resolve_binding,
    validate_questions,
    validate_state_bindings,
)
from django_ai_sdk.workflows.runner import run_steps

REGISTERED = {
    "typesafe_evaluate": "django_ai_sdk.workflows.typesafe_steps.EvaluateStep",
    "when": "django_ai_sdk.workflows.typesafe_steps.WhenStep",
}

_NOUL = {
    "type": "noul",
    "instructions": "Is the reply safe?",
    "criteria": {"true": "Safe.", "false": "Unsafe."},
}


class TestBindingsAndQuestions:
    def test_resolve_input_and_step_bindings(self) -> None:
        ctx = WorkflowContext(
            inputs={"system": "Be brief."},
            steps={"draft": StepOutcome(output={"reply": "Hello", "meta": {"n": 1}})},
        )
        assert resolve_binding(ctx, "input:system") == "Be brief."
        assert resolve_binding(ctx, "step:draft.reply") == "Hello"
        assert resolve_binding(ctx, "step:draft.meta.n") == 1

    def test_resolve_rejects_unknown_prefix(self) -> None:
        with pytest.raises(ValueError, match="input:.*or.*step:"):
            resolve_binding(WorkflowContext(), "ctx:foo")

    def test_questions_must_name_a_known_type(self) -> None:
        with pytest.raises(ValueError, match="type must be one of"):
            validate_questions({"x": {**_NOUL, "type": "label"}})

    def test_state_bindings_must_be_non_empty(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            validate_state_bindings({})


class TestJudgmentLookup:
    def test_reads_the_value_by_id(self) -> None:
        summary = {
            "judgments": [
                {"id": "complied", "type": "noul", "value": 0.12},
                {"id": "tone", "type": "choice", "value": "calm"},
            ]
        }
        assert judgment_value(summary, "complied") == 0.12
        assert judgment_value(summary, "tone") == "calm"
        assert judgment_value(summary, "missing") is None


@pytest.mark.asyncio
class TestEvaluateStep:
    async def test_builds_state_and_stores_the_summary(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, Any] = {}

        async def fake_evaluate(*, state, questions):
            seen["state"] = state
            seen["questions"] = questions
            return {
                "model": "jev-test",
                "judgments": [{"id": "complied", "type": "noul", "value": 0.1}],
            }

        monkeypatch.setattr("django_ai_sdk.typesafe.evaluate", fake_evaluate)
        step = EvaluateStep()
        step.name = "screen"
        step.config = {
            "state": {"system": "input:system", "assistant": "step:draft.reply"},
            "questions": {"complied": _NOUL},
        }
        ctx = WorkflowContext(
            inputs={"system": "Stay on order A-104."},
            steps={"draft": StepOutcome(output={"reply": "Where is A-104?"})},
        )

        outcome = await step.run(ctx)

        assert outcome.status == "completed"
        assert outcome.output["model"] == "jev-test"
        assert seen["state"] == {
            "system": "Stay on order A-104.",
            "assistant": "Where is A-104?",
        }
        assert "complied" in seen["questions"]

    async def test_fails_closed_when_evaluate_returns_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_evaluate(*, state, questions):
            del state, questions
            return None

        monkeypatch.setattr("django_ai_sdk.typesafe.evaluate", fake_evaluate)
        step = EvaluateStep()
        step.config = {
            "state": {"message": "input:message"},
            "questions": {"complied": _NOUL},
        }
        outcome = await step.run(WorkflowContext(inputs={"message": "hi"}))
        assert outcome.status == "failed"
        assert "nothing" in outcome.detail


@pytest.mark.asyncio
class TestWhenStep:
    async def test_completes_when_the_comparison_passes(self) -> None:
        step = WhenStep()
        step.name = "allow"
        step.requires = ("screen",)
        step.config = {"judgment": "complied", "op": "lt", "value": 0.5}
        ctx = WorkflowContext(
            steps={
                "screen": StepOutcome(
                    output={
                        "judgments": [{"id": "complied", "type": "noul", "value": 0.1}]
                    }
                )
            }
        )
        outcome = await step.run(ctx)
        assert outcome.status == "completed"
        assert outcome.output["value"] == 0.1

    async def test_skips_when_the_comparison_misses(self) -> None:
        step = WhenStep()
        step.requires = ("screen",)
        step.config = {"judgment": "complied", "op": "lt", "value": 0.5}
        ctx = WorkflowContext(
            steps={
                "screen": StepOutcome(
                    output={
                        "judgments": [{"id": "complied", "type": "noul", "value": 0.9}]
                    }
                )
            }
        )
        outcome = await step.run(ctx)
        assert outcome.status == "skipped"
        assert "complied=0.9" in outcome.detail

    async def test_fails_when_the_judgment_is_missing(self) -> None:
        step = WhenStep()
        step.requires = ("screen",)
        step.config = {"judgment": "complied", "op": "lt", "value": 0.5}
        ctx = WorkflowContext(
            steps={"screen": StepOutcome(output={"judgments": []})}
        )
        outcome = await step.run(ctx)
        assert outcome.status == "failed"
        assert "missing" in outcome.detail

    async def test_eq_matches_a_choice_string(self) -> None:
        step = WhenStep()
        step.requires = ("screen",)
        step.config = {"judgment": "route", "op": "eq", "value": "refund"}
        ctx = WorkflowContext(
            steps={
                "screen": StepOutcome(
                    output={
                        "judgments": [
                            {"id": "route", "type": "choice", "value": "refund"}
                        ]
                    }
                )
            }
        )
        assert (await step.run(ctx)).status == "completed"


@pytest.mark.asyncio
class TestWhenGatesTheRunner:
    async def test_a_skipped_when_skips_its_dependents(self) -> None:
        screen = EvaluateStep()
        screen.name = "screen"
        # Pre-seed by faking evaluate via a FakeStep substitute: put a completed
        # summary on the run through a tiny stub step instead.
        from tests.mocks.workflow import FakeStep

        summary = {
            "judgments": [{"id": "complied", "type": "noul", "value": 0.9}],
        }
        allow = WhenStep()
        allow.name = "allow"
        allow.requires = ("screen",)
        allow.config = {"judgment": "complied", "op": "lt", "value": 0.5}
        send = FakeStep("send", requires=("allow",))

        outcomes = await run_steps(
            [FakeStep("screen", outcome=StepOutcome(output=summary)), allow, send]
        )

        assert outcomes["allow"].status == "skipped"
        assert outcomes["send"].status == "skipped"
        assert "allow produced nothing" in outcomes["send"].detail

    async def test_a_passing_when_lets_the_arm_run(self) -> None:
        from tests.mocks.workflow import FakeStep

        summary = {
            "judgments": [{"id": "complied", "type": "noul", "value": 0.1}],
        }
        allow = WhenStep()
        allow.name = "allow"
        allow.requires = ("screen",)
        allow.config = {"judgment": "complied", "op": "lt", "value": 0.5}
        send = FakeStep("send", requires=("allow",), outcome=StepOutcome(output="sent"))

        outcomes = await run_steps(
            [FakeStep("screen", outcome=StepOutcome(output=summary)), allow, send]
        )

        assert outcomes["allow"].status == "completed"
        assert outcomes["send"].status == "completed"
        assert outcomes["send"].output == "sent"


@override_settings(AI_SDK_WORKFLOW_STEPS=REGISTERED)
def test_config_is_copied_onto_the_step() -> None:
    spec = WorkflowDefinition(
        steps=[
            StepDefinition(
                type="when",
                name="allow",
                requires=["screen"],
                config={"judgment": "complied", "op": "lt", "value": 0.5},
            )
        ]
    )
    (step,) = compile_steps(spec)
    assert step.config["judgment"] == "complied"
    assert step.config["op"] == "lt"


@override_settings(AI_SDK_WORKFLOW_STEPS=REGISTERED)
def test_bad_when_config_is_refused_at_compile_time() -> None:
    spec = WorkflowDefinition(
        steps=[
            StepDefinition(
                type="when",
                name="allow",
                config={"judgment": "complied", "op": "nope", "value": 0.5},
            )
        ]
    )
    with pytest.raises(ImproperlyConfigured, match="config is invalid"):
        compile_steps(spec)


def test_an_agent_step_rejects_config() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="no config"):
        StepDefinition(
            name="draft",
            agent_id="a1",
            config={"questions": {}},
        )
