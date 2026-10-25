"""Format TypeSafe score payloads for plain-text channels (Slack, Telegram)."""

from __future__ import annotations

from typing import Any


def format_typesafe_card(payload: dict[str, Any]) -> str:
    """Compact mrkdwn-ish block for a chat reply footer."""
    side = payload.get("side")
    title = {
        "input": "TypeSafe (inbound)",
        "output": "TypeSafe (reply)",
        "translate": "TypeSafe (translate)",
    }.get(side, "TypeSafe")
    model = payload.get("model")
    header = f"*{title}*"
    if model:
        header = f"{header} · `{model}`"

    lines = [header]
    acted = payload.get("acted")
    if acted:
        if isinstance(acted, list):
            lines.append(f"_acted: {', '.join(str(a) for a in acted)}_")
        else:
            lines.append(f"_acted: {acted}_")
    for judgment in payload.get("judgments") or []:
        label = judgment.get("label") or judgment.get("id") or "?"
        jtype = judgment.get("type") or "?"
        lines.append(f"• [{jtype}] {label}: {_format_value(judgment)}")
        for extra in _format_details(judgment):
            lines.append(f"  {extra}")
    return "\n".join(lines)


def _fmt_unit(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    text = f"{float(value):.3f}".rstrip("0").rstrip(".")
    return text


def _format_value(judgment: dict[str, Any]) -> str:
    jtype = judgment.get("type")
    value = judgment.get("value")
    conf = judgment.get("confidence")
    conf_bit = ""
    if isinstance(conf, int | float):
        conf_bit = f" conf={_fmt_unit(float(conf))}"

    if jtype == "noul" and isinstance(value, int | float):
        return f"P(yes)={_fmt_unit(float(value))}"
    if jtype == "score" and isinstance(value, int | float):
        return f"{_fmt_unit(float(value))}{conf_bit}"
    return f"{value}{conf_bit}"


def _format_details(judgment: dict[str, Any]) -> list[str]:
    """Secondary lines: probability mass and score legend."""
    details: list[str] = []
    probs = judgment.get("probabilities")
    if isinstance(probs, dict) and probs:
        ordered = sorted(probs.items(), key=lambda kv: (-float(kv[1]), str(kv[0])))
        if judgment.get("type") == "score":
            bits = [f"L{k}={_fmt_unit(float(v))}" for k, v in ordered]
        else:
            bits = [f"{k}={_fmt_unit(float(v))}" for k, v in ordered]
        details.append(" ".join(bits))

    legend = judgment.get("legend")
    if isinstance(legend, dict) and legend:
        try:
            items = sorted(legend.items(), key=lambda kv: (float(kv[0]), str(kv[0])))
        except (TypeError, ValueError):
            items = sorted(legend.items(), key=lambda kv: str(kv[0]))
        details.append(" · ".join(f"{k}: {text}" for k, text in items))
    return details


def append_typesafe_cards(reply: str, *payloads: dict[str, Any] | None) -> str:
    """Append non-empty cards under the reply; leave reply alone when none."""
    cards = [format_typesafe_card(p) for p in payloads if p]
    if not cards:
        return reply
    return f"{reply.rstrip()}\n\n" + "\n\n".join(cards)
