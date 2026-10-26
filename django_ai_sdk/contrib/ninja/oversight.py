"""Oversight: the turns feed, review queues, policies, insight and the ledger.

Mount behind staff auth; every endpoint checks the oversight domain's permissions
again, and every read lands on the access log::

    api.add_router("/", get_oversight_router(), auth=staff_auth)
"""

from __future__ import annotations

import json
from collections.abc import Collection
from typing import Any

from django.http import HttpRequest, StreamingHttpResponse
from ninja import Router, Schema

from django_ai_sdk.contrib.ninja.routing import ERRORS, ApiRouter, Limit, Offset
from django_ai_sdk.errors import NotFound
from django_ai_sdk.oversight.schemas import (
    AccessOut,
    AgentPolicyIn,
    AgentRow,
    AnswererIn,
    CheckpointOut,
    ObservationOut,
    OverrideIn,
    OverrideOut,
    Overview,
    PolicyIn,
    PolicySummary,
    RevealOut,
    ReviewOut,
    SimulateIn,
    TryAnswererIn,
    TryLensIn,
    TurnDetail,
    TurnPage,
    VerdictIn,
    VerifyOut,
)
from django_ai_sdk.oversight.services import OversightService

routes = ApiRouter()


class ReasonIn(Schema):
    reason: str


class NoteIn(Schema):
    note: str = ""


class ResolveIn(VerdictIn):
    dismiss: bool = False


def _found(value: Any, what: str) -> Any:
    if value is None:
        raise NotFound(f"{what} not found")
    return value


@routes.get("/oversight/turns/", response=TurnPage)
async def list_oversight_turns(
    request: HttpRequest,
    purpose: str = "security",
    outcome: str = "",
    shadow_outcome: str = "",
    agent_id: str = "",
    channel: str = "",
    policy: str = "",
    severity: str = "",
    label: str = "",
    value: str = "",
    rule: str = "",
    review: str = "",
    thread_id: str = "",
    limit: Limit = 50,
    offset: Offset = 0,
) -> Any:
    return await OversightService.list_turns(
        user=request.user,
        purpose=purpose,
        outcome=outcome,
        shadow_outcome=shadow_outcome,
        agent_id=agent_id,
        channel=channel,
        policy=policy,
        severity=severity,
        label=label,
        value=value,
        rule=rule,
        review=review,
        thread_id=thread_id,
        limit=limit,
        offset=offset,
    )


@routes.get("/oversight/turns/{turn_id}/", response=TurnDetail)
async def get_oversight_turn(request: HttpRequest, turn_id: str, purpose: str = "security") -> Any:
    turn = await OversightService.get_turn(turn_id, user=request.user, purpose=purpose)
    return _found(turn, "Turn")


@routes.post("/oversight/turns/{turn_id}/reveal/", response=RevealOut)
async def reveal_oversight_actor(request: HttpRequest, turn_id: str, payload: ReasonIn) -> Any:
    revealed = await OversightService.reveal(turn_id, user=request.user, reason=payload.reason)
    return _found(revealed, "Turn")


@routes.post("/oversight/turns/{turn_id}/verdicts/", response={201: ObservationOut, **ERRORS})
async def add_oversight_verdict(request: HttpRequest, turn_id: str, payload: VerdictIn) -> Any:
    observation = await OversightService.add_verdict(turn_id, payload, user=request.user)
    return 201, _found(observation, "Turn")


@routes.get("/oversight/reviews/", response=list[ReviewOut])
async def list_oversight_reviews(
    request: HttpRequest,
    queue: str = "",
    status: str = "open",
    purpose: str = "quality",
    limit: Limit = 50,
    offset: Offset = 0,
) -> Any:
    return await OversightService.list_reviews(
        user=request.user, queue=queue, status=status, purpose=purpose, limit=limit, offset=offset
    )


@routes.get("/oversight/queues/", response=dict[str, int])
async def list_oversight_queues(request: HttpRequest) -> Any:
    return await OversightService.queues(user=request.user)


@routes.post("/oversight/reviews/{item_id}/", response=ReviewOut)
async def resolve_oversight_review(request: HttpRequest, item_id: str, payload: ResolveIn) -> Any:
    verdict = VerdictIn(**payload.model_dump(exclude={"dismiss"}))
    item = await OversightService.resolve_review(
        item_id, verdict, user=request.user, dismiss=payload.dismiss
    )
    return _found(item, "Review item")


@routes.get("/oversight/policies/", response=list[PolicySummary])
async def list_oversight_policies(request: HttpRequest) -> Any:
    return await OversightService.list_policies(user=request.user)


@routes.get("/oversight/policies/{name}/", response=dict[str, Any])
async def get_oversight_policy(request: HttpRequest, name: str) -> Any:
    return _found(await OversightService.get_policy(name, user=request.user), "Policy")


@routes.get("/oversight/policies/{name}/overrides/", response=list[OverrideOut])
async def list_oversight_overrides(request: HttpRequest, name: str) -> Any:
    return await OversightService.list_overrides(name, user=request.user)


@routes.post("/oversight/policies/{name}/overrides/", response=dict[str, Any])
async def set_oversight_override(request: HttpRequest, name: str, payload: OverrideIn) -> Any:
    policy = await OversightService.set_override(name, payload, user=request.user)
    return _found(policy, "Policy")


@routes.post("/oversight/policies/", response={201: dict[str, Any], **ERRORS})
async def create_oversight_policy(request: HttpRequest, payload: PolicyIn) -> Any:
    return 201, await OversightService.create_policy(payload, user=request.user)


@routes.post("/oversight/policies/{name}/delete/", response={204: None, **ERRORS})
async def delete_oversight_policy(request: HttpRequest, name: str, payload: NoteIn) -> Any:
    _found(
        await OversightService.delete_policy(name, user=request.user, note=payload.note) or None,
        "Policy",
    )
    return 204, None


@routes.post("/oversight/policies/{name}/simulate/", response=dict[str, Any])
async def simulate_oversight_policy(request: HttpRequest, name: str, payload: SimulateIn) -> Any:
    return _found(await OversightService.simulate(name, payload, user=request.user), "Policy")


@routes.post("/oversight/policies/{name}/try-lens/", response=dict[str, Any])
async def try_oversight_lens(request: HttpRequest, name: str, payload: TryLensIn) -> Any:
    return _found(await OversightService.try_lens(name, payload, user=request.user), "Lens")


@routes.get("/oversight/agents/", response=list[AgentRow])
async def list_oversight_agents(request: HttpRequest) -> Any:
    return await OversightService.list_agents(user=request.user)


@routes.post("/oversight/agents/{agent_id}/", response=AgentRow)
async def set_oversight_agent_policy(
    request: HttpRequest, agent_id: str, payload: AgentPolicyIn
) -> Any:
    row = await OversightService.set_agent_policy(agent_id, payload, user=request.user)
    return _found(row, "Agent")


@routes.get("/oversight/catalog/", response=dict[str, Any])
async def get_oversight_catalog(request: HttpRequest) -> Any:
    return await OversightService.catalog(user=request.user)


@routes.post("/oversight/answerers/{alias}/", response=dict[str, Any])
async def set_oversight_answerer(request: HttpRequest, alias: str, payload: AnswererIn) -> Any:
    return await OversightService.set_answerer(alias, payload, user=request.user)


@routes.post("/oversight/answerers-try/", response=dict[str, Any])
async def try_oversight_answerer(request: HttpRequest, payload: TryAnswererIn) -> Any:
    return await OversightService.try_answerer(payload, user=request.user)


@routes.get("/oversight/changes/", response=list[OverrideOut])
async def list_oversight_changes(request: HttpRequest, policy: str = "", limit: Limit = 100) -> Any:
    return await OversightService.changes(user=request.user, policy=policy, limit=limit)


@routes.get("/oversight/overview/", response=Overview)
async def get_oversight_overview(
    request: HttpRequest, days: int = 14, agent_id: str = "", policy: str = ""
) -> Any:
    return await OversightService.overview(
        user=request.user, days=days, agent_id=agent_id, policy=policy
    )


@routes.get("/oversight/checkpoints/", response=list[CheckpointOut])
async def list_oversight_checkpoints(request: HttpRequest, limit: Limit = 50) -> Any:
    return await OversightService.checkpoints(user=request.user, limit=limit)


@routes.post("/oversight/checkpoints/", response={201: CheckpointOut, 204: None, **ERRORS})
async def seal_oversight_ledger(request: HttpRequest) -> Any:
    checkpoint = await OversightService.seal_now(user=request.user)
    return (201, checkpoint) if checkpoint is not None else (204, None)


@routes.post("/oversight/verify/", response=VerifyOut)
async def verify_oversight_ledger(request: HttpRequest) -> Any:
    return await OversightService.verify(user=request.user)


@routes.get("/oversight/access/", response=list[AccessOut])
async def list_oversight_access(
    request: HttpRequest, turn_id: str = "", limit: Limit = 100, offset: Offset = 0
) -> Any:
    return await OversightService.access_log(
        user=request.user, turn_id=turn_id, limit=limit, offset=offset
    )


@routes.get("/oversight/export/", response={200: None, **ERRORS})
async def export_oversight_turns(
    request: HttpRequest, purpose: str = "audit", days: int = 30
) -> Any:
    rows = await OversightService.export(user=request.user, purpose=purpose, days=days)

    async def lines() -> Any:
        async for row in rows:
            yield json.dumps(row, default=str) + "\n"

    response = StreamingHttpResponse(lines(), content_type="application/x-ndjson")
    response["Content-Disposition"] = f'attachment; filename="oversight-{purpose}.jsonl"'
    return response


@routes.post("/oversight/threads/{thread_id}/forget/", response={204: None, **ERRORS})
async def forget_oversight_thread(request: HttpRequest, thread_id: str, payload: ReasonIn) -> Any:
    await OversightService.forget_thread(thread_id, user=request.user, reason=payload.reason)
    return 204, None


def get_oversight_router(*, exclude: Collection[str] = (), **router_kwargs: Any) -> Router:
    """A new Router with the oversight endpoints; mount with ``api.add_router("/", ...)``."""
    return routes.build(exclude=exclude, **router_kwargs)
