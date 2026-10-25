"""Tiny TypeSafe screener for the demo pirate agent (no host batteries)."""

from __future__ import annotations

from typing import Any

from django_ai_sdk.typesafe import evaluate

_TOOL_THRESHOLD = 0.35

INBOUND_QUESTIONS: dict[str, Any] = {
    "should_use_tool": {
        "type": "noul",
        "instructions": (
            "Would a capable assistant need an external tool for this message?"
        ),
        "criteria": {
            "true": "Needs live data, tools, or files.",
            "false": "Greeting or answerable from the message alone.",
        },
    },
}

OUTBOUND_QUESTIONS: dict[str, Any] = {
    "quality": {
        "type": "score",
        "instructions": "How helpful is the assistant_reply?",
        "criteria": [
            "Unhelpful or off-topic.",
            "Partially useful.",
            "Clear and on-topic.",
        ],
    },
}


def control_from_inbound(summary: dict[str, Any]) -> dict[str, Any] | None:
    """Map inbound Nouls to SDK control hints (unit-tested; no live API)."""
    for judgment in summary.get("judgments") or []:
        if judgment.get("id") != "should_use_tool":
            continue
        try:
            value = float(judgment.get("value") or 0)
        except (TypeError, ValueError):
            return None
        if value < _TOOL_THRESHOLD:
            return {"strip_tools": True}
        return None
    return None


class DemoTypeSafeScreener:
    """Inbound tool gate + outbound quality card for PirateBasicAgent."""

    async def screen_input(self, text: str) -> dict[str, Any] | None:
        stripped = (text or "").strip()
        if not stripped:
            return None
        summary = await evaluate(
            state={"message": stripped},
            questions=INBOUND_QUESTIONS,
        )
        if not summary:
            return None
        payload: dict[str, Any] = {"side": "input", **summary}
        control = control_from_inbound(summary)
        if control:
            payload["control"] = control
            payload["acted"] = ["stripped_tools"]
        return payload

    async def screen_output(self, user_text: str, reply: str) -> dict[str, Any] | None:
        if not (user_text or "").strip() or not (reply or "").strip():
            return None
        summary = await evaluate(
            state={"user_message": user_text, "assistant_reply": reply},
            questions=OUTBOUND_QUESTIONS,
        )
        return {"side": "output", **summary} if summary else None
