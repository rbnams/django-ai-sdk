"""Plain-text TypeSafe card formatting (Slack / Telegram footers)."""

from __future__ import annotations

from django_ai_sdk.typesafe.format import append_typesafe_cards, format_typesafe_card


def test_format_typesafe_card_inbound() -> None:
    text = format_typesafe_card(
        {
            "side": "input",
            "model": "jev-1.13.0",
            "acted": ["stripped_tools"],
            "judgments": [
                {"id": "angry", "type": "noul", "label": "Angry", "value": 0.02},
                {
                    "id": "tone",
                    "type": "choice",
                    "label": "Tone",
                    "value": "calm",
                    "confidence": 1.0,
                    "probabilities": {"calm": 0.91, "frustrated": 0.07, "angry": 0.02},
                },
                {
                    "id": "quality",
                    "type": "score",
                    "label": "Quality",
                    "value": 1.8,
                    "confidence": 0.74,
                    "probabilities": {"0": 0.05, "1": 0.2, "2": 0.75},
                    "legend": {
                        "0": "Unhelpful",
                        "1": "Partial",
                        "2": "Clear",
                    },
                },
            ],
        }
    )
    assert "*TypeSafe (inbound)*" in text
    assert "`jev-1.13.0`" in text
    assert "acted: stripped_tools" in text
    assert "[noul] Angry: P(yes)=0.02" in text
    assert "[choice] Tone: calm conf=1" in text
    assert "calm=0.91" in text
    assert "[score] Quality: 1.8 conf=0.74" in text
    assert "L2=0.75" in text
    assert "0: Unhelpful" in text


def test_append_typesafe_cards_skips_empty() -> None:
    assert append_typesafe_cards("Hello", None, None) == "Hello"
    out = append_typesafe_cards(
        "Hello",
        {"side": "output", "judgments": [{"id": "q", "type": "noul", "label": "Q", "value": 0.9}]},
    )
    assert out.startswith("Hello\n\n")
    assert "*TypeSafe (reply)*" in out
    assert "P(yes)=0.9" in out
