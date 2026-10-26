"""Thin wrapper around TypeSafe's AsyncTypeSafeClient.

Jev is not a chat model: it answers typed questions (Choice / Noul / Score)
about a state. This module never pretends otherwise.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from django_ai_sdk.logger import get_logger
from django_ai_sdk.utils import resolve_setting

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = get_logger(__name__)


def typesafe_api_key() -> str | None:
    """Django setting or empty when unset."""
    value = resolve_setting("TYPESAFE_API_KEY")
    if not value or not str(value).strip():
        return None
    return str(value).strip()


def typesafe_model() -> str:
    """Pinned or alias model id; defaults to jev-latest."""
    value = resolve_setting("TYPESAFE_MODEL")
    if value and str(value).strip():
        return str(value).strip()
    return "jev-latest"


async def evaluate(
    state: Any,
    questions: Mapping[str, Any],
    *,
    model: str | None = None,
) -> dict[str, Any] | None:
    """Run one System One request; return a JSON-safe summary or None on failure.

    Fail-open: missing key, missing package, or API errors log and return None
    so a chat turn is never blocked by screening.
    """
    api_key = typesafe_api_key()
    if not api_key:
        logger.warning("TYPESAFE_API_KEY unset; skipping TypeSafe evaluate")
        return None
    if not questions:
        return None

    try:
        from typesafe_sdk import AsyncTypeSafeClient
    except ImportError:
        logger.warning("typesafe-sdk is not installed; install django-ai-sdk[typesafe] to screen")
        return None

    try:
        async with AsyncTypeSafeClient(api_key=api_key, model=typesafe_model()) as client:
            response = await client.system_one(
                state=state,
                questions=questions,
                model=model or typesafe_model(),
            )
    except Exception:
        logger.exception("TypeSafe system_one failed; skipping screen")
        return None

    return _summarize(response)


def _summarize(response: Any) -> dict[str, Any]:
    """Flatten a SystemOneResponse into a UI-friendly payload."""
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
        "model": getattr(response, "model", None) or typesafe_model(),
        "judgments": judgments,
    }


def _label(question_id: str) -> str:
    """Human label from a question id like ``vulgar_language`` → ``Vulgar language``."""
    return question_id.replace("_", " ").strip().capitalize()
