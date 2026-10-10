"""Scheduled workflows: run agent work on a clock.

An automation says when it runs, as whom, and which workflow. The SDK claims and
dispatches due work as WorkflowRuns; the deployment provides the tick
(`manage.py run_automations --loop 60`) and a django-tasks worker. Declare them in
any installed app's `automations.py`.

    from django_ai_sdk.automations import Automation, register

    @register
    class MorningDigest(Automation):
        name = "morning-digest"
        cron = "0 7 * * mon-fri"
        timezone = "Europe/Amsterdam"
        workflow = "digest"
        run_as = "digest-bot@example.com"
        input = "Summarise the open tasks since {last_run_at}."
"""

from __future__ import annotations

from .base import Automation
from .models import AutomationState
from .registry import (
    get_automation,
    get_automations,
    get_invalid_automations,
    register,
    reset_registry,
)
from .schedule import Cron

__all__ = [
    "Automation",
    "AutomationState",
    "Cron",
    "get_automation",
    "get_automations",
    "get_invalid_automations",
    "register",
    "reset_registry",
]
