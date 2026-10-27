"""Who may see what: pseudonyms, and the two views of a turn.

A reviewer reads the turn itself: the question, the answer, the sources it drew
on (where an indirect injection hides) and every verdict with its reason. An
auditor reads the decisions and the chain, and no content at all. A person shows
up as a stable pseudonym until someone records why they must know.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
from typing import TYPE_CHECKING, Any

from django.conf import settings

from django_ai_sdk.oversight.types import Purpose

if TYPE_CHECKING:
    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser


def _secret() -> bytes:
    from django_ai_sdk.oversight.conf import setting

    configured = setting("PSEUDONYM_SECRET")
    if configured:
        return str(configured).encode()
    # Derived, not reused: SECRET_KEY rotation should not be the only way to unlink.
    return hashlib.sha256(
        b"django-ai-sdk.oversight.pseudonym:" + settings.SECRET_KEY.encode()
    ).digest()


def pseudonym(user: AbstractBaseUser | AnonymousUser | None) -> str:
    """A stable, unlinkable-without-the-secret name for `user`: "u-k3j9x2q7"."""
    pk = getattr(user, "pk", None)
    if pk is None:
        return "anonymous"
    digest = hmac.new(_secret(), f"user:{pk}".encode(), hashlib.sha256).digest()
    return "u-" + base64.b32encode(digest[:5]).decode().lower()


def staff_label(user: AbstractBaseUser | AnonymousUser | None) -> str:
    """How a reader is named on the access log: readers are accountable by name."""
    if user is None or getattr(user, "pk", None) is None:
        return "system"
    username = getattr(user, "get_username", lambda: "")()
    return f"{username} ({user.pk})" if username else str(user.pk)


# What each view shows of a turn. Anything not listed is withheld.
VIEWS: dict[Purpose, frozenset[str]] = {
    Purpose.REVIEW: frozenset(
        {
            "input",
            "output",
            "shown_input",
            "shown_output",
            "sources",
            "tool_calls",
            "observations",
            "decisions",
            "history",
        }
    ),
    # Verdicts without their reasons: a rationale can quote the content.
    Purpose.AUDIT: frozenset({"decisions", "chain", "observations"}),
}


def visible(purpose: Purpose, part: str) -> bool:
    return part in VIEWS.get(purpose, frozenset())


def mask_evidence(content: dict[str, Any] | None, purpose: Purpose) -> dict[str, Any] | None:
    """`content` with the parts `purpose` may not see removed, not blanked."""
    if content is None:
        return None
    return {key: value for key, value in content.items() if visible(purpose, key)}


__all__ = [
    "VIEWS",
    "mask_evidence",
    "pseudonym",
    "staff_label",
    "visible",
]
