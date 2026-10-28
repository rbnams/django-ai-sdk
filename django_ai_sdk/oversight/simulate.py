"""Dry runs: what a policy would do with a message, without recording a turn.

The console's test bench. A simulation runs a real TurnSession through gate,
watch (the whole answer as one streamed piece) and review, then answers the
reflect lenses on the spot (every one, whatever its sample rate), so the result
is what a live turn would get. No turn is stored: the session is never closed,
and a Human answerer only says where it would queue. Model calls are charged to
the daily budget like any other.

    result = await simulate("support", input="Where is my refund?", output="It is on its way.")
    result["stages"][0]["observations"]   # what the gate asked, and who answered
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from django_ai_sdk.oversight import budget as budgets
from django_ai_sdk.oversight.stages import (
    Matched,
    Observed,
    asks_again,
    evaluate_rules,
    lens_health,
    run_lenses,
)
from django_ai_sdk.oversight.types import Answer, Stage, Subject, TurnView, worst

if TYPE_CHECKING:
    from django.contrib.auth.base_user import AbstractBaseUser

    from django_ai_sdk.oversight.lenses import Lens
    from django_ai_sdk.oversight.policy import ResolvedPolicy
    from django_ai_sdk.oversight.session import TurnSession


def answer_out(answer: Answer) -> dict[str, Any]:
    """One answer as the console shows it, with every answerer asked on the way."""
    return {
        "answerer": answer.answerer,
        "model": answer.model,
        "value": answer.value,
        "confidence": answer.confidence,
        "rationale": answer.rationale,
        "error": answer.error,
        "queued": answer.queued,
        "latency_ms": answer.latency_ms,
        "cost": answer.cost,
        "spans": [span.model_dump() for span in answer.spans],
        "items": answer.items,
        "trail": [
            {
                "answerer": step.answerer,
                "model": step.model,
                "value": step.value,
                "confidence": step.confidence,
                "rationale": step.rationale,
                "error": step.error,
                "latency_ms": step.latency_ms,
            }
            for step in answer.trail
        ],
    }


def _observed(observed: Observed) -> dict[str, Any]:
    lens = observed.lens
    return {
        "lens": lens.name,
        "label": lens.label or lens.name.replace("_", " "),
        "kind": lens.kind,
        "subject": str(lens.subject),
        "stage": str(observed.stage),
        **answer_out(observed.answer),
    }


def _matched(match: Matched) -> dict[str, Any]:
    return {
        "rule": match.rule.name,
        "description": match.rule.description,
        "stage": str(match.stage),
        "mode": str(match.rule.mode),
        "actions": [action.to_dict() for action in match.rule.then],
        "acted": [action.to_dict() for action in match.acted],
        "would": str(match.outcome(shadow=True)),
        "outcome": str(match.outcome()) if match.acted else "allowed",
        "condition": match.rule.when.to_dict(),
    }


async def simulate(
    policy: str,
    *,
    input: str,
    output: str = "",
    tool_call: dict[str, Any] | None = None,
    user: AbstractBaseUser | None = None,
) -> dict[str, Any] | None:
    """Run `policy` over one exchange as a turn would, and record nothing. None: no policy.

    `tool_call` ({"name", "arguments"}) is checked as the model's one tool call.
    """
    from django_ai_sdk.oversight.policy import aget_policy
    from django_ai_sdk.oversight.session import TurnSession

    resolved = await aget_policy(policy, uncached=True)
    if resolved is None:
        return None
    agent = SimpleNamespace(name="Console test", oversight=policy)
    session = await TurnSession.start(agent, channel="console", input=input, user=user)
    try:
        return await _run(session, resolved, output=output, tool_call=tool_call)
    finally:
        # The session is never closed, so its spend is recorded here.
        if session._budget is not None:
            await session._budget.flush()


async def _run(
    session: TurnSession,
    resolved: ResolvedPolicy,
    *,
    output: str,
    tool_call: dict[str, Any] | None,
) -> dict[str, Any]:
    stages: list[dict[str, Any]] = []

    def stage_out(stage: Stage, before: int, matched_before: int, **extra: Any) -> None:
        observed = session.observed[before:]
        asked = {o.lens.name for o in observed}
        # Shown, not observed: a live turn records nothing for these, and no rule sees them.
        silent = [
            Observed(lens, stage, Answer(error="no answerer", confidence=0.0))
            for lens in resolved.active_lenses(stage)
            if lens.name not in asked and not lens_health(lens)["runnable"]
        ]
        stages.append(
            {
                "stage": str(stage),
                "observations": [_observed(o) for o in [*observed, *silent]],
                "matched": [_matched(m) for m in session.matched[matched_before:]],
                **extra,
            }
        )

    gate = await session.gate()
    stage_out(
        Stage.GATE,
        0,
        0,
        blocked=gate.blocked,
        message=gate.message,
        redacted=gate.redacted,
        text=gate.input,
        restrict=gate.restrict,
    )
    ended_at = "gate" if gate.blocked else ""

    if not gate.blocked and tool_call and tool_call.get("name"):
        from django_ai_sdk.oversight.types import ToolCall

        observed, matched = len(session.observed), len(session.matched)
        call = ToolCall(name=str(tool_call["name"]), arguments=tool_call.get("arguments"))
        refused = await session.check_tool(call)
        session.view.tool_calls.append(call)
        stage_out(
            Stage.WATCH,
            observed,
            matched,
            blocked=bool(refused),
            message=refused or "",
            tool=call.name,
        )

    if not gate.blocked and output:
        observed, matched = len(session.observed), len(session.matched)
        session.view.output = output
        watched = await session.watch()
        stage_out(
            Stage.WATCH,
            observed,
            matched,
            blocked=watched.blocked,
            message=watched.message,
            text=watched.output,
        )
        if watched.blocked:
            ended_at = "watch"

    if not ended_at and output:
        observed, matched = len(session.observed), len(session.matched)
        review = await session.review(output)
        stage_out(
            Stage.REVIEW,
            observed,
            matched,
            blocked=review.blocked,
            message=review.message,
            redacted=review.replaced and not review.blocked,
            text=review.output,
        )
        if review.blocked:
            ended_at = "review"

    if not gate.blocked:
        observed, matched = len(session.observed), len(session.matched)
        await _reflect(session, resolved)
        stage_out(Stage.REFLECT, observed, matched)

    acted = [m for m in session.matched if m.acted]
    return {
        "policy": resolved.name,
        "stages": stages,
        "ended_at": ended_at,
        "outcome": str(worst(*(m.outcome() for m in acted))) if acted else "allowed",
        "shadow_outcome": (
            str(worst(*(m.outcome(shadow=True) for m in session.matched)))
            if session.matched
            else "allowed"
        ),
        "had_output": bool(output),
    }


async def _reflect(session: Any, resolved: Any) -> None:
    """The reflect stage in place: its lenses, plus inline lenses that missed their deadline."""
    late = {
        o.lens.name
        for o in session.observed
        if o.stage in {Stage.GATE, Stage.WATCH, Stage.REVIEW}
        and (
            o.answer.error == "timeout"
            or asks_again(o.answer.error, o.answer.trail[-1].error if o.answer.trail else "")
        )
    }
    # As live: reflect has no tool call to show a tool call lens.
    lenses = [
        lens
        for lens in resolved.lenses.values()
        if lens.enabled
        and lens.subject is not Subject.TOOL_CALL
        and (Stage.REFLECT in lens.stages or lens.name in late)
    ]
    if not session.view.output:
        # Without an answer, only what reads the question can be judged.
        lenses = [lens for lens in lenses if str(lens.subject) == "input"]
    token = budgets.activate(session._budget)
    try:
        observed = await run_lenses(lenses, session.view, Stage.REFLECT, sample=False)
    finally:
        budgets.deactivate(token)
    session.observed.extend(observed)
    for o in observed:
        session.view.answers[o.lens.name] = o.answer
    session.matched.extend(
        evaluate_rules(
            resolved,
            session.view.answers,
            Stage.REFLECT,
            done={m.rule.name for m in session.matched},
        )
    )


async def try_lens(lens: Lens, *, input: str, output: str = "") -> dict[str, Any]:
    """Ask one lens, saved or not, at its first stage: the editor's "test this lens"."""
    from django_ai_sdk.oversight.stages import answer_lens

    view = TurnView(turn_id="console", channel="console", input=input, output=output)
    stage = lens.stages[0] if lens.stages else Stage.REVIEW
    if not lens.applies(view):
        answer = Answer(error="needs sources or tool results", confidence=0.0)
    elif (answer := await answer_lens(lens, view, stage, sample=False)) is None:
        answer = Answer(error="no answerer answered", confidence=0.0)
    return {"lens": lens.name, "kind": lens.kind, "stage": str(stage), **answer_out(answer)}


async def replay(agent: Any, cases: list[dict[str, Any]], *, policy: str = "") -> dict[str, Any]:
    """Replay questions through an agent as it is now, and judge the new answers.

    `agent` is an agent, or its id, name or class name; each case is
    {"input": ..., "expected": {lens: value}}. Per lens, how often the new answer
    still gets the expected verdict. Raises ValueError when no policy applies.
    """
    import hashlib

    from django_ai_sdk.common import ChatMessage
    from django_ai_sdk.oversight.answerers.llm import AgentJudge
    from django_ai_sdk.oversight.metrics import band
    from django_ai_sdk.oversight.policy import aget_policy
    from django_ai_sdk.oversight.session import code_policy, unobserved

    if isinstance(agent, str):
        agent = AgentJudge(agent).resolve_agent()
    resolved = await aget_policy(policy or code_policy(agent)[0])
    if resolved is None:
        raise ValueError("The agent has no oversight policy; pass a policy.")
    lenses: dict[str, Any] = {}
    failed = 0
    for case in cases:
        try:
            with unobserved():
                output = await agent.run(
                    [ChatMessage(role="user", content=case["input"])], tools=True
                )
        except Exception:
            failed += 1
            continue
        digest = hashlib.sha256(case["input"].encode()).hexdigest()[:16]
        view = TurnView(turn_id=f"replay:{digest}", input=case["input"], output=str(output or ""))
        expected = case.get("expected") or {}
        judged = [
            lens
            for name, lens in resolved.lenses.items()
            if name in expected and (Stage.REVIEW in lens.stages or Stage.REFLECT in lens.stages)
        ]
        with unobserved():
            observed = await run_lenses(judged, view, Stage.RETRO, sample=False)
        for o in observed:
            stats = lenses.setdefault(o.lens.name, {"expected": 0, "kept": 0, "answers": []})
            stats["expected"] += 1
            if o.answer.ok and band(o.answer.value) == band(expected[o.lens.name]):
                stats["kept"] += 1
            stats["answers"].append(o.answer.value)
    return {"cases": len(cases), "failed": failed, "lenses": lenses}


__all__ = ["answer_out", "replay", "simulate", "try_lens"]
