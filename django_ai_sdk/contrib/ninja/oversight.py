"""Oversight: the turns feed, review queues, policies, insight and the ledger.

The router brings no auth of its own; it runs under the API's (e.g.
``NinjaAPI(auth=SessionAuth())``). Every endpoint's service call checks the
``oversight`` permission domain for ``request.user``: by default staff view,
review and audit, and only superusers reveal who asked or change policy
(OversightDefaultPermission; override it in ``AI_SDK_PERMISSIONS``). Reads of
turns and reviews land on the access log::

    api.add_router("/", get_oversight_router())
"""

from __future__ import annotations

import json
from collections.abc import Collection
from typing import Any

from django.http import HttpRequest, StreamingHttpResponse
from ninja import Query, Router, Schema

from django_ai_sdk.contrib.ninja.routing import ERRORS, ApiRouter, Limit, Offset
from django_ai_sdk.errors import NotFound
from django_ai_sdk.oversight.schemas import (
    AccessOut,
    AgentPolicyIn,
    AgentRow,
    AnswererIn,
    AnswerOut,
    CheckpointOut,
    ObservationOut,
    OverrideIn,
    OverrideOut,
    Overview,
    PolicyIn,
    PolicyOut,
    PolicySummary,
    RevealOut,
    ReviewOut,
    SimulateIn,
    SimulateOut,
    TryAnswererIn,
    TryLensIn,
    TryLensOut,
    TurnDetail,
    TurnFilter,
    TurnPage,
    VerdictIn,
    VerifyOut,
)
from django_ai_sdk.oversight.services import OversightService
from django_ai_sdk.views.schemas import ErrorResponse

routes = ApiRouter()


class ReasonIn(Schema):
    reason: str


class ResolveIn(VerdictIn):
    dismiss: bool = False


def _found(value: Any, what: str) -> Any:
    if value is None:
        raise NotFound(f"{what} not found")
    return value


@routes.get("/oversight/turns/", response=TurnPage)
async def list_oversight_turns(
    request: HttpRequest,
    filters: Query[TurnFilter],
    purpose: str = "review",
    limit: Limit = 50,
    offset: Offset = 0,
) -> Any:
    return await OversightService.list_turns(
        filters, user=request.user, purpose=purpose, limit=limit, offset=offset
    )


@routes.get("/oversight/turns/{turn_id}/", response=TurnDetail)
async def get_oversight_turn(request: HttpRequest, turn_id: str, purpose: str = "review") -> Any:
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
    purpose: str = "review",
    limit: Limit = 50,
    offset: Offset = 0,
) -> Any:
    return await OversightService.list_reviews(
        user=request.user, queue=queue, status=status, purpose=purpose, limit=limit, offset=offset
    )


@routes.get("/oversight/queues/", response=dict[str, int])
async def list_oversight_queues(request: HttpRequest) -> Any:
    return await OversightService.queues(user=request.user)


# 409: the item was closed already.
@routes.post(
    "/oversight/reviews/{item_id}/", response={200: ReviewOut, 409: ErrorResponse, **ERRORS}
)
async def resolve_oversight_review(request: HttpRequest, item_id: str, payload: ResolveIn) -> Any:
    verdict = VerdictIn(**payload.model_dump(exclude={"dismiss"}))
    item = await OversightService.resolve_review(
        item_id, verdict, user=request.user, dismiss=payload.dismiss
    )
    return _found(item, "Review item")


@routes.get("/oversight/policies/", response=list[PolicySummary])
async def list_oversight_policies(request: HttpRequest) -> Any:
    return await OversightService.list_policies(user=request.user)


@routes.get("/oversight/policies/{name}/", response=PolicyOut)
async def get_oversight_policy(request: HttpRequest, name: str) -> Any:
    return _found(await OversightService.get_policy(name, user=request.user), "Policy")


@routes.post("/oversight/policies/{name}/overrides/", response=PolicyOut)
async def set_oversight_override(request: HttpRequest, name: str, payload: OverrideIn) -> Any:
    policy = await OversightService.set_override(name, payload, user=request.user)
    return _found(policy, "Policy")


@routes.post("/oversight/policies/", response={201: PolicyOut, **ERRORS})
async def create_oversight_policy(request: HttpRequest, payload: PolicyIn) -> Any:
    return 201, await OversightService.create_policy(payload, user=request.user)


@routes.delete("/oversight/policies/{name}/", response={204: None, **ERRORS})
async def delete_oversight_policy(request: HttpRequest, name: str, note: str = "") -> Any:
    _found(
        await OversightService.delete_policy(name, user=request.user, note=note) or None, "Policy"
    )
    return 204, None


@routes.post("/oversight/policies/{name}/simulate/", response=SimulateOut)
async def simulate_oversight_policy(request: HttpRequest, name: str, payload: SimulateIn) -> Any:
    return _found(await OversightService.simulate(name, payload, user=request.user), "Policy")


@routes.post("/oversight/policies/{name}/try-lens/", response=TryLensOut)
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


@routes.post("/oversight/answerers-try/", response=AnswerOut)
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
