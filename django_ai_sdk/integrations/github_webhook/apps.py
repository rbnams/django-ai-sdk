from __future__ import annotations

from django_ai_sdk.integrations.apps import IntegrationAppConfig


class GitHubWebhookConfig(IntegrationAppConfig):
    default = True
    name = "django_ai_sdk.integrations.github_webhook"
    # Namespaced so the SDK never claims the global app label, and distinct
    # from the `github` MCP integration's app: the two are the inbound and
    # outbound halves of the platform, installed separately (as mcp/apps.py
    # does for its own label).
    label = "django_ai_sdk_github_webhook"
    integration = "django_ai_sdk.integrations.github_webhook.integration.GitHubWebhookIntegration"
