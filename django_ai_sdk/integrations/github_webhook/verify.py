"""GitHub webhook signing, HMAC-SHA256 over the body."""

from __future__ import annotations

import hashlib
import hmac

PREFIX = "sha256="


def is_signed_by_github(webhook_secret: str, signature: str, body: bytes) -> bool:
    """Whether body carries the GitHub signature made with webhook_secret.

    GitHub signs an HMAC-SHA256 over the raw body and no timestamp, so unlike
    Slack's scheme this check cannot refuse a replay: the
    de-duplication window is the replay bound, which is why the integration
    ships a wide one. Configure the webhook with a sha256 secret token in the
    repository or organization settings; the sha1 scheme is deprecated and not
    accepted here.
    """
    if not webhook_secret or not signature:
        return False
    if not signature.startswith(PREFIX):
        return False
    expected = hmac.new(webhook_secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(PREFIX + expected, signature)
