"""GitHub as a webhook integration: an agent reviews pull requests."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import httpx

from django_ai_sdk.integrations.github_webhook.verify import is_signed_by_github
from django_ai_sdk.integrations.webhooks.base import InboundEvent, WebhookIntegration

if TYPE_CHECKING:
    from django.http import HttpRequest, HttpResponse

logger = logging.getLogger(__name__)

API_BASE = "https://api.github.com"
POST_TIMEOUT_SECONDS = 10

# GitHub rejects a comment body longer than this.
MAX_COMMENT_LENGTH = 65536

# A draft waits for ready_for_review; the other actions are new code to look at.
DEFAULT_ACTIONS = frozenset({"opened", "reopened", "synchronize", "ready_for_review"})

SIGNATURE_HEADER = "X-Hub-Signature-256"
EVENT_HEADER = "X-GitHub-Event"
DELIVERY_HEADER = "X-GitHub-Delivery"


class GitHubWebhookIntegration(WebhookIntegration):
    """Runs the agent on a pull request and posts its answer as a PR comment.

    Needs a repository or organization webhook pointed at this integration's
    webhook path (HTTPS, the `pull_request` event, a sha256 secret), plus a token
    that may comment on the repositories it serves:

        AI_SDK_INTEGRATIONS = {"github_webhook": {
            "WEBHOOK_SECRET": env("GITHUB_WEBHOOK_SECRET"),
            "TOKEN": env("GITHUB_TOKEN"),
            "ALLOW_WORKSPACES": ["owner/repo"],
            "AGENT": "myapp.agents.ReviewAgent",
        }}

    The event text names the pull request; the agent's instructions say what a
    review is. To read the diff it needs the `github` integration in its
    `integrations`.
    """

    name = "github_webhook"
    label = "GitHub"

    # GitHub signs no timestamp, so a captured request replays past verify()
    # once this window has passed: the window is GitHub's replay bound.
    dedup_ttl = 7 * 24 * 60 * 60

    def missing_config(self) -> str | None:
        missing = [key for key in ("WEBHOOK_SECRET", "TOKEN") if not self.secret(key)]
        if missing:
            return (
                f"Missing {', '.join(missing)}. Set them in "
                f"AI_SDK_INTEGRATIONS['github_webhook']: the secret from the webhook "
                f"settings, and a token that may comment on pull requests."
            )
        # The secret belongs to the webhook, and one webhook can cover a whole
        # organization, so the repositories are named here or nothing is answered.
        if not self._config().get("ALLOW_WORKSPACES"):
            return (
                "No repositories. Set AI_SDK_INTEGRATIONS['github_webhook']"
                f"['ALLOW_WORKSPACES'] to owner/repo names, or to {self.ANY!r}."
            )
        return None

    def verify(self, request: HttpRequest) -> bool:
        return is_signed_by_github(
            self.secret("WEBHOOK_SECRET"),
            request.headers.get(SIGNATURE_HEADER, ""),
            request.body,
        )

    def parse(self, request: HttpRequest) -> InboundEvent | HttpResponse | None:
        # The ping GitHub sends when the webhook is created needs only the 200.
        if request.headers.get(EVENT_HEADER, "") != "pull_request":
            return None
        try:
            body = json.loads(request.body)
        except ValueError:
            logger.warning("GitHub sent a body that is not JSON")
            return None

        action = body.get("action", "")
        pr = body.get("pull_request") or {}
        if action not in self._actions() or (pr.get("draft") and action != "ready_for_review"):
            return None

        repository = (body.get("repository") or {}).get("full_name", "")
        number = pr.get("number")
        sender = (body.get("sender") or {}).get("login", "")
        if not repository or number is None:
            return None
        # A push by the bot's own account would otherwise review its own work.
        bot = self.secret("BOT_USERNAME")
        if bot and sender.lower() == bot.lower():
            return None

        head = (pr.get("head") or {}).get("sha", "")
        parts = [
            f"Pull request #{number} in {repository} ({action}): {pr.get('title') or ''}",
            (pr.get("body") or "").strip(),
            f"{pr.get('html_url') or ''}\nHead: {head}",
        ]
        return InboundEvent(
            integration=self.name,
            event_id=request.headers.get(DELIVERY_HEADER, "") or f"{repository}#{number}:{head}",
            text="\n\n".join(part for part in parts if part),
            conversation_id=repository,
            thread_ref=str(number),
            external_user_id=sender,
            workspace_id=repository,
        )

    def thread_scope(self, event: InboundEvent) -> str | None:
        # One SDK thread per pull request, so a new push continues the review.
        if not event.conversation_id or not event.thread_ref:
            return None
        return f"ai-sdk:webhook:github_webhook:{event.conversation_id}:{event.thread_ref}"

    async def reply(self, event: InboundEvent, text: str) -> None:
        # A pull request is addressable as an issue, which is where comments live.
        url = f"{API_BASE}/repos/{event.conversation_id}/issues/{event.thread_ref}/comments"
        async with httpx.AsyncClient(timeout=POST_TIMEOUT_SECONDS) as client:
            response = await client.post(
                url,
                json={"body": text[:MAX_COMMENT_LENGTH]},
                headers={
                    "Authorization": f"Bearer {self.secret('TOKEN')}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
            )
        if not response.is_success:
            logger.error(
                "GitHub refused a comment on %s#%s: %s %s",
                event.conversation_id,
                event.thread_ref,
                response.status_code,
                response.text,
            )

    def _actions(self) -> frozenset[str]:
        return frozenset(self._config().get("ACTIONS") or DEFAULT_ACTIONS)
