"""Inbound events: an integration a platform pushes messages into.

An agent answers in a channel by implementing verify, parse and reply on a
WebhookIntegration, mounting webhooks/urls.py, and running a worker. `telegram`,
`slack` and `github_webhook` ship with the SDK.

Reference: docs/content/manual/webhooks.md.
"""

from __future__ import annotations

from django_ai_sdk.integrations.webhooks.base import (
    InboundEvent,
    WebhookIntegration,
    current_event,
)

__all__ = ["InboundEvent", "WebhookIntegration", "current_event"]
