"""Read-only admin for the ledger: rows can be looked at here, never changed."""

from __future__ import annotations

from typing import Any

from django.contrib import admin

from django_ai_sdk.oversight.models import Checkpoint, Decision, Observation, ReviewItem, Turn


class ReadOnly(admin.ModelAdmin):
    def has_add_permission(self, request: Any, obj: Any = None) -> bool:
        return False

    def has_change_permission(self, request: Any, obj: Any = None) -> bool:
        return False

    def has_delete_permission(self, request: Any, obj: Any = None) -> bool:
        return False


@admin.register(Turn)
class TurnAdmin(ReadOnly):
    list_display = ("created_at", "channel", "agent_name", "outcome", "shadow_outcome", "severity")
    list_filter = ("outcome", "shadow_outcome", "severity", "channel", "policy")
    # The actor is sealed; content lives in evidence and is read through the service.
    exclude = ("actor_ref",)


@admin.register(Observation)
class ObservationAdmin(ReadOnly):
    list_display = ("created_at", "lens", "stage", "answerer", "value", "confidence", "error")
    list_filter = ("lens", "stage", "answerer")


@admin.register(Decision)
class DecisionAdmin(ReadOnly):
    list_display = ("created_at", "rule", "stage", "mode", "acted")
    list_filter = ("rule", "mode", "acted")


@admin.register(ReviewItem)
class ReviewItemAdmin(ReadOnly):
    list_display = ("created_at", "queue", "lens", "rule", "priority", "status")
    list_filter = ("queue", "status")


@admin.register(Checkpoint)
class CheckpointAdmin(ReadOnly):
    list_display = ("created_at", "tree_size", "root_hash", "signed", "anchored_at")
