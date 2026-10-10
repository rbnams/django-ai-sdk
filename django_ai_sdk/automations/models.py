"""Persistence for the scheduler: one cursor row per automation.

Runs are WorkflowRun rows tagged with the automation's name, so there is no run
table of its own.
"""

from __future__ import annotations

import uuid

from django.db import models


class AutomationState(models.Model):
    """The scheduler's cursor for one automation.

    Rows are created lazily by the first tick, never at app load.
    """

    id = models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)
    name = models.CharField(max_length=255, unique=True, db_index=True)

    # Advanced by the claim itself, so a winning tick has already scheduled the next.
    next_run_at = models.DateTimeField(null=True, blank=True, db_index=True)

    last_dispatched_at = models.DateTimeField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        app_label = "django_ai_sdk"
        db_table = "django_ai_sdk_automation_state"
        ordering = ["name"]
        verbose_name = "Automation"
        verbose_name_plural = "Automations"

    def __str__(self) -> str:
        return self.name
