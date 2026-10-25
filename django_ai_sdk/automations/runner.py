"""Due-ness, claiming and dispatch: everything one tick does.

A tick is safe to run on any number of hosts: the claim is one conditional UPDATE
that advances the cursor, so a second tick's UPDATE matches no rows. Each run is a
WorkflowRun tagged with the automation's name, executed by the workflow worker.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import TYPE_CHECKING, NamedTuple

from asgiref.sync import sync_to_async
from django.db import transaction
from django.utils import timezone

from django_ai_sdk.automations.models import AutomationState
from django_ai_sdk.automations.registry import get_automation, get_automations
from django_ai_sdk.utils import resolve_setting

if TYPE_CHECKING:
    from datetime import datetime

    from django_ai_sdk.automations.base import Automation
    from django_ai_sdk.workflows.models import WorkflowRun

logger = logging.getLogger(__name__)


class Dispatched(NamedTuple):
    """What one tick did to one automation."""

    name: str
    run: WorkflowRun | None
    reason: str = ""


def automations_enabled() -> bool:
    """The global kill switch, AI_SDK_AUTOMATIONS_ENABLED."""
    return bool(resolve_setting("AI_SDK_AUTOMATIONS_ENABLED", True))


async def tick(*, now: datetime | None = None) -> list[Dispatched]:
    """Claim and dispatch every automation that is due."""
    if not automations_enabled():
        logger.info("AI_SDK_AUTOMATIONS_ENABLED is False; nothing dispatched.")
        return []

    now = now or timezone.now()
    results: list[Dispatched] = []
    for name, automation in get_automations().items():
        try:
            results.append(await _tick_one(automation, now=now))
        except Exception:
            # One broken automation must not stop the rest of the tick.
            logger.exception("Automation %r failed to dispatch", name)
            results.append(Dispatched(name, None, "dispatch raised; see logs"))
    return results


def is_enabled(automation: Automation, state: AutomationState | None) -> tuple[bool, str]:
    """Whether the schedule fires it, and which layer decided: kill-switch, db or code."""
    if not automations_enabled():
        return False, "kill-switch"
    if state is not None and state.enabled is not None:
        return state.enabled, "db"
    return automation.enabled, "code"


async def _tick_one(automation: Automation, *, now: datetime) -> Dispatched:
    existing = await AutomationState.objects.filter(name=automation.name).afirst()
    if not is_enabled(automation, existing)[0]:
        return Dispatched(automation.name, None, "disabled")

    state, _ = await ensure_state(automation, now=now)
    if state.next_run_at is None or state.next_run_at > now:
        return Dispatched(automation.name, None, "not due")

    if not await claim(automation, state, now=now):
        return Dispatched(automation.name, None, "another tick claimed it")

    return Dispatched(automation.name, await dispatch(automation))


async def ensure_state(automation: Automation, *, now: datetime) -> tuple[AutomationState, bool]:
    """Fetch this automation's state row, creating it on first sight.

    Scheduled forward, so deploying one does not fire it for a past cron match.
    """
    return await AutomationState.objects.aget_or_create(
        name=automation.name,
        defaults={"next_run_at": automation.get_schedule().next_after(now)},
    )


async def claim(automation: Automation, state: AutomationState, *, now: datetime) -> bool:
    """Win the right to dispatch this occurrence.

    Predicate and write are one statement, so a second tick's UPDATE matches no rows.
    """
    next_at = automation.get_schedule().next_after(now)
    won = await AutomationState.objects.filter(id=state.id, next_run_at__lte=now).aupdate(
        next_run_at=next_at, last_dispatched_at=now, updated_at=now
    )
    if won:
        state.next_run_at = next_at
    return bool(won)


async def run_now(name: str) -> WorkflowRun | None:
    """Dispatch an automation immediately, outside its schedule, enabled or not.

    Leaves `next_run_at` alone, so a pending scheduled occurrence still fires.
    """
    automation = get_automation(name)
    if automation is None:
        raise ValueError(f"No automation named {name!r}")
    if not automations_enabled():
        logger.info("AI_SDK_AUTOMATIONS_ENABLED is False; %r not dispatched.", name)
        return None

    now = timezone.now()
    state, _ = await ensure_state(automation, now=now)
    await AutomationState.objects.filter(id=state.id).aupdate(
        last_dispatched_at=now, updated_at=now
    )
    return await dispatch(automation)


async def dispatch(automation: Automation) -> WorkflowRun | None:
    """Open one WorkflowRun as `run_as` and enqueue it on commit, or log why not.

    A skip is a log line rather than a row: nothing ran, so there is no run.
    """
    from django_ai_sdk.utils import aget_active_user
    from django_ai_sdk.workflows.executor import open_run, validate_inputs
    from django_ai_sdk.workflows.registry import aget_workflow, validate_definition

    definition = await aget_workflow(automation.workflow)
    if definition is None:
        logger.warning(
            "Automation %r runs workflow %r, which is not registered; skipped",
            automation.name,
            automation.workflow,
        )
        return None
    validate_definition(definition)

    user = await aget_active_user(automation.run_as)
    # Running with no user instead would quietly drop whatever the account may reach.
    if automation.run_as and user is None:
        logger.error(
            "Automation %r: run_as %r matches no single active user; skipped",
            automation.name,
            automation.run_as,
        )
        return None

    inputs = automation.get_inputs(user=user, last_run_at=await last_run_at(automation))

    # A workflow drops inputs it does not declare, which would be a run that
    # succeeded having asked nothing. Refused here, where the mismatch is ours.
    declared = definition.input_fields
    undeclared = sorted(set(inputs) - set(declared)) if declared else []
    if undeclared:
        logger.warning(
            "Automation %r supplies %s, which workflow %r does not declare; skipped. "
            "Declare them in its input_fields or override get_inputs().",
            automation.name,
            ", ".join(repr(k) for k in undeclared),
            automation.workflow,
        )
        return None
    validate_inputs(definition, inputs)

    run = await open_run(definition, inputs=inputs, user=user, automation=automation.name)
    # From on_commit, so a caller inside a transaction cannot hand a worker an id it
    # would not find.
    await sync_to_async(transaction.on_commit)(partial(_enqueue, run))
    return run


def _enqueue(run: WorkflowRun) -> None:
    """Hand one run to the workflow worker, or fail it if the backend will not take it."""
    from django_ai_sdk.workflows.models import WorkflowRun
    from django_ai_sdk.workflows.tasks import execute_workflow

    try:
        result = execute_workflow.enqueue(str(run.id))
    except Exception:
        logger.exception("Could not enqueue workflow run %s", run.id)
        WorkflowRun.objects.filter(id=run.id).update(
            status=WorkflowRun.Status.FAILED,
            error="Could not enqueue; is a task backend configured?",
            completed_at=timezone.now(),
        )
        return
    WorkflowRun.objects.filter(id=run.id).update(task_id=str(result.id))


async def last_run_at(automation: Automation) -> datetime | None:
    """When this automation's last completed run started."""
    from django_ai_sdk.workflows.models import WorkflowRun

    run = (
        await WorkflowRun.objects.filter(
            automation=automation.name, status=WorkflowRun.Status.COMPLETED
        )
        .order_by("-created_at")
        .afirst()
    )
    if run is None:
        return None
    return run.started_at or run.created_at


__all__ = [
    "Dispatched",
    "automations_enabled",
    "claim",
    "dispatch",
    "ensure_state",
    "is_enabled",
    "last_run_at",
    "run_now",
    "tick",
]
