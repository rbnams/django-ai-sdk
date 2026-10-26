"""What a model call cost, from AI_SDK_MODEL_PRICES.

Prices are per million tokens, in whatever currency the deployment bills in::

    AI_SDK_MODEL_PRICES = {
        "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
        "claude-": {"input": 3.00, "output": 15.00},   # every claude-* model
    }

A key matches a model exactly, or as its longest prefix. No match means no cost,
never a guess.
"""

from __future__ import annotations

from typing import Any

from django_ai_sdk.utils import resolve_setting


def model_price(model: str | None) -> dict[str, float] | None:
    """The price entry for `model`: an exact key, else the longest prefix key."""
    if not model:
        return None
    prices: dict[str, dict[str, float]] = resolve_setting("AI_SDK_MODEL_PRICES", {}) or {}
    if model in prices:
        return prices[model]
    # Provider prefixes ("openai:gpt-4.1") price like the bare name.
    bare = model.split(":", 1)[-1].split("/", 1)[-1]
    if bare in prices:
        return prices[bare]
    matches = [key for key in prices if model.startswith(key) or bare.startswith(key)]
    return prices[max(matches, key=len)] if matches else None


def cost_for(
    model: str | None, prompt_tokens: int | None, completion_tokens: int | None
) -> float | None:
    """The cost of one call, or None when the model has no price or no usage."""
    price = model_price(model)
    if price is None or (prompt_tokens is None and completion_tokens is None):
        return None
    total = (prompt_tokens or 0) * float(price.get("input", 0)) + (completion_tokens or 0) * float(
        price.get("output", 0)
    )
    return round(total / 1_000_000, 8)


def usage_from_meta(meta: dict[str, Any] | None) -> tuple[int | None, int | None]:
    """(prompt, completion) tokens from a Haystack reply's meta, whatever the vendor."""
    usage = (meta or {}).get("usage") or {}
    if not isinstance(usage, dict):
        return None, None
    prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
    completion = usage.get("completion_tokens", usage.get("output_tokens"))
    return (
        int(prompt) if isinstance(prompt, int | float) else None,
        int(completion) if isinstance(completion, int | float) else None,
    )


__all__ = ["cost_for", "model_price", "usage_from_meta"]
