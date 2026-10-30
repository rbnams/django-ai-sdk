"""Thin wrapper around TypeSafe's AsyncTypeSafeClient, for the TypeSafe answerer.

TypeSafe is a hosted classification service, not a chat model: it answers typed
questions (Choice / Noul / Score) about a state. This module never pretends otherwise,
and never logs what a request sent or what came back.
"""

from __future__ import annotations

import asyncio
import importlib.util
import weakref
from typing import TYPE_CHECKING, Any

from django_ai_sdk.utils import resolve_setting

if TYPE_CHECKING:
    from collections.abc import Mapping


def typesafe_api_key() -> str | None:
    """Django setting or empty when unset."""
    value = resolve_setting("TYPESAFE_API_KEY")
    if not value or not str(value).strip():
        return None
    return str(value).strip()


def typesafe_model() -> str | None:
    """TYPESAFE_MODEL, a pinned or alias model id; None leaves the choice to typesafe-sdk."""
    value = resolve_setting("TYPESAFE_MODEL")
    if value and str(value).strip():
        return str(value).strip()
    return None


class TypeSafeUnavailable(RuntimeError):
    """No API key, or typesafe-sdk is not installed: a configuration gap, not an outage."""


def typesafe_missing() -> str:
    """Why TypeSafe cannot be asked here, or "" when it can."""
    if not typesafe_api_key():
        return "TYPESAFE_API_KEY is not set"
    if importlib.util.find_spec("typesafe_sdk") is None:
        return "typesafe-sdk is not installed (pip install django-ai-sdk[typesafe])"
    return ""


# One client per event loop: its connection pool belongs to the loop that opened it,
# and a background task under async_to_sync runs on a loop of its own.
_clients: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, tuple[str, Any]] = (
    weakref.WeakKeyDictionary()
)


def _client(api_key: str) -> Any:
    from typesafe_sdk import AsyncTypeSafeClient

    loop = asyncio.get_running_loop()
    cached = _clients.get(loop)
    if cached is None or cached[0] != api_key:
        cached = (api_key, AsyncTypeSafeClient(api_key=api_key, model=typesafe_model()))
        _clients[loop] = cached
    return cached[1]


async def request(
    state: Any,
    questions: Mapping[str, Any],
    *,
    model: str | None = None,
    timeout: float | None = None,
    retry: Any = None,
) -> dict[str, Any]:
    """One TypeSafe request, as a JSON-safe summary.

    Raises TypeSafeUnavailable without a key or the package; the SDK's own errors
    pass through. `timeout` and `retry` (a RetryPolicy) override the client's.
    """
    missing = typesafe_missing()
    if missing:
        raise TypeSafeUnavailable(missing)
    options: dict[str, Any] = {}
    if timeout is not None:
        options["timeout"] = timeout
    if retry is not None:
        options["retry"] = retry
    response = await _client(typesafe_api_key() or "").system_one(
        state=state, questions=questions, model=model or typesafe_model(), **options
    )
    return _summarize(response)


def _summarize(response: Any) -> dict[str, Any]:
    """Flatten TypeSafe's response into a UI-friendly payload."""
    judgments: list[dict[str, Any]] = []

    for key, answer in getattr(response, "nouls", {}).items():
        judgments.append(
            {
                "id": key,
                "type": "noul",
                "label": _label(key),
                "value": float(answer.noul),
            }
        )

    for key, answer in getattr(response, "choices", {}).items():
        judgments.append(
            {
                "id": key,
                "type": "choice",
                "label": _label(key),
                "value": answer.choice,
                "probabilities": dict(answer.probabilities or {}),
                "confidence": float(answer.confidence)
                if getattr(answer, "confidence", None) is not None
                else None,
            }
        )

    for key, answer in getattr(response, "scores", {}).items():
        judgments.append(
            {
                "id": key,
                "type": "score",
                "label": _label(key),
                "value": float(answer.score),
                "legend": dict(answer.legend or {}),
                "probabilities": dict(answer.probabilities or {}),
                "confidence": float(answer.confidence)
                if getattr(answer, "confidence", None) is not None
                else None,
            }
        )

    return {
        "model": getattr(response, "model", None) or typesafe_model() or "",
        "judgments": judgments,
    }


def _label(question_id: str) -> str:
    """Human label from a question id like ``vulgar_language`` → ``Vulgar language``."""
    return question_id.replace("_", " ").strip().capitalize()
