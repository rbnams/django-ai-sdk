"""Service-layer facade for automations: the seam an HTTP layer sits on.

The SDK ships no router. Schedules stay read-only here: `cron` has no database layer.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from django.core.exceptions import ImproperlyConfigured
from django.db.models import Max, OuterRef, Subquery

from django_ai_sdk.automations.models import AutomationState
from django_ai_sdk.automations.registry import get_automation, get_automations
from django_ai_sdk.automations.runner import is_enabled
from django_ai_sdk.automations.schemas import AutomationOut, AutomationRunOut
from django_ai_sdk.permissions import (
    Operation,
    PermissionDenied,
    PermissionDomain,
    PermissionsMixin,
)

if TYPE_CHECKING:
    from datetime import datetime

    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser

    from django_ai_sdk.automations.base import Automation
    from django_ai_sdk.workflows.models import WorkflowRun

    UserType = AbstractBaseUser | AnonymousUser | None

logger = logging.getLogger(__name__)


class AutomationService(PermissionsMixin):
    """Resolve, permission-check, and act on automations by name."""

    domain = PermissionDomain.AUTOMATIONS

    @classmethod
    async def list_for_user(cls, user: UserType) -> list[AutomationOut]:
        """Every automation, with live state, if `user` may view automations."""
        if not await cls.has_perms(user, Operation.VIEW_AUTOMATION, raise_on_deny=False):
            return []
        automations = get_automations()
        if not automations:
            return []
        states = {
            state.name: state
            async for state in AutomationState.objects.filter(name__in=list(automations))
        }
        latest = await cls._latest_runs(list(automations))
        return [
            cls._describe(automation, states.get(name), latest.get(name))
            for name, automation in automations.items()
        ]

    @classmethod
    async def set_enabled(cls, name: str, *, enabled: bool, user: UserType) -> AutomationOut | None:
        """Turn an automation on or off at the database layer.

        Returns None for an unknown name, so a router answers 404 without catching.
        """
        automation = get_automation(name)
        if automation is None:
            return None
        await cls._check(user, Operation.MANAGE_AUTOMATION, name)

        state = await cls._ensure_state(automation)
        await AutomationState.objects.filter(id=state.id).aupdate(enabled=enabled)
        state.enabled = enabled
        latest = await cls._latest_runs([name])
        return cls._describe(automation, state, latest.get(name))

    @classmethod
    async def run_now(cls, name: str, *, user: UserType = None) -> list[AutomationRunOut] | None:
        """Dispatch an automation outside its schedule, or None if the name is unknown.

        `user` is who asked, not who it runs as; `run_as` decides that. Empty when
        nothing was dispatched (kill switch off, or the run was skipped; see logs).
        """
        automation = get_automation(name)
        if automation is None:
            return None
        await cls._check(user, Operation.MANAGE_AUTOMATION, name)

        from django_ai_sdk.automations.runner import run_now as _run_now

        run = await _run_now(name)
        return [cls._describe_run(run)] if run is not None else []

    @classmethod
    async def list_runs(
        cls, name: str, *, user: UserType = None, limit: int = 50, offset: int = 0
    ) -> list[AutomationRunOut]:
        """This automation's runs, newest first.

        A run's output is its account's content, so only a manager sees every run.
        """
        from django_ai_sdk.workflows.models import WorkflowRun

        await cls._check(user, Operation.VIEW_AUTOMATION, name)
        qs = WorkflowRun.objects.filter(automation=name).order_by("-created_at")
        if not await cls.has_perms(user, Operation.MANAGE_AUTOMATION, raise_on_deny=False):
            qs = qs.filter(user_id=user.pk) if user is not None and user.pk else qs.none()
        return [cls._describe_run(run) async for run in qs[offset : offset + limit]]

    @classmethod
    async def get_run(cls, run_id: str, *, user: UserType = None) -> AutomationRunOut | None:
        """One automation run by id, or None.

        Someone else's run reads as absent: a 403 would confirm the id exists.
        """
        from django.core.exceptions import ValidationError

        from django_ai_sdk.workflows.models import WorkflowRun

        try:
            run = await WorkflowRun.objects.exclude(automation="").filter(id=run_id).afirst()
        except ValidationError:
            return None
        if run is None:
            return None
        await cls._check(user, Operation.VIEW_AUTOMATION, run.automation)
        mine = user is not None and user.pk is not None and run.user_id == user.pk
        if not mine and not await cls.has_perms(
            user, Operation.MANAGE_AUTOMATION, raise_on_deny=False
        ):
            return None
        return cls._describe_run(run)

    @classmethod
    async def last_tick_at(cls) -> datetime | None:
        """When any automation was last dispatched, across the whole deployment.

        A dead-man's switch: the scheduler cannot notice its own absence.
        """
        result = await AutomationState.objects.aaggregate(last=Max("last_dispatched_at"))
        return result["last"]

    @classmethod
    async def _check(cls, user: UserType, operation: Operation, name: str) -> None:
        if not await cls.has_perms(user, operation, raise_on_deny=False):
            raise PermissionDenied(f"Not permitted to {operation} {name!r}")

    @staticmethod
    async def _ensure_state(automation: Automation) -> AutomationState:
        """The state row, created if the first tick has not run yet.

        An unusable schedule has no next occurrence, but `enabled` still needs a row.
        """
        from django.utils import timezone

        from django_ai_sdk.automations.runner import ensure_state

        now = timezone.now()
        try:
            state, _ = await ensure_state(automation, now=now)
        except ImproperlyConfigured:
            logger.warning("Automation %r has no usable schedule", automation.name)
            state, _ = await AutomationState.objects.aget_or_create(name=automation.name)
        return state

    @staticmethod
    async def _latest_runs(names: list[str]) -> dict[str, WorkflowRun]:
        """The newest run per automation, in one query."""
        from django_ai_sdk.workflows.models import WorkflowRun

        newest = Subquery(
            WorkflowRun.objects.filter(automation=OuterRef("automation"))
            .order_by("-created_at")
            .values("id")[:1]
        )
        return {
            run.automation: run
            async for run in WorkflowRun.objects.filter(automation__in=names, id=newest)
        }

    @classmethod
    def _describe(
        cls,
        automation: Automation,
        state: AutomationState | None,
        last_run: WorkflowRun | None,
    ) -> AutomationOut:
        enabled, source = is_enabled(automation, state)
        try:
            schedule, detail = str(automation.get_schedule()), None
        except Exception as exc:
            schedule, detail = "invalid", str(exc)
        return AutomationOut(
            name=automation.name,
            label=automation.label or automation.name,
            description=automation.description,
            schedule=schedule,
            next_run_at=state.next_run_at if state else None,
            last_dispatched_at=state.last_dispatched_at if state else None,
            enabled=enabled,
            enabled_source=source,
            workflow=automation.workflow,
            run_as=automation.run_as,
            last_run=cls._describe_run(last_run) if last_run else None,
            detail=detail,
        )

    @staticmethod
    def _describe_run(run: WorkflowRun) -> AutomationRunOut:
        return AutomationRunOut(
            id=str(run.id),
            status=run.status,
            created_at=run.created_at,
            started_at=run.started_at,
            finished_at=run.completed_at,
            output=run.outputs,
            error=run.error,
        )


__all__ = ["AutomationService"]
