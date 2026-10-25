"""Output schemas for automations. Plain pydantic: the SDK does not pick the framework."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel


class AutomationRunOut(BaseModel):
    """One dispatched run: the WorkflowRun an automation opened."""

    id: str
    status: str
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    # The workflow's outputs, keyed by each step's output_key.
    output: Any = None
    error: str = ""


class AutomationOut(BaseModel):
    """One automation's current state, for a generic GET /automations list."""

    name: str
    label: str
    description: str = ""

    schedule: str
    next_run_at: datetime | None = None
    last_dispatched_at: datetime | None = None

    enabled: bool
    # Which layer decided `enabled`: "kill-switch" | "db" | "code".
    enabled_source: str

    workflow: str
    # Empty when it runs as the deployment itself.
    run_as: str = ""

    last_run: AutomationRunOut | None = None

    # Why the schedule is unusable, if it is.
    detail: str | None = None


__all__ = ["AutomationOut", "AutomationRunOut"]
