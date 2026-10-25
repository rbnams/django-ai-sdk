"""TypeSafe inbound/outbound screener helpers (no live API calls)."""

from __future__ import annotations

import pytest

from django_ai_sdk.adapters.base import Stream
from django_ai_sdk.common import ChatMessage
from django_ai_sdk.events import DataEvent


class _FakeScreener:
    def __init__(
        self,
        inbound: dict | None = None,
        outbound: dict | None = None,
        *,
        boom_in: bool = False,
        boom_out: bool = False,
    ) -> None:
        self.inbound = inbound
        self.outbound = outbound
        self.boom_in = boom_in
        self.boom_out = boom_out

    async def screen_input(self, text: str) -> dict | None:
        if self.boom_in:
            raise RuntimeError("typesafe down")
        del text
        return self.inbound

    async def screen_output(self, user_text: str, reply: str) -> dict | None:
        if self.boom_out:
            raise RuntimeError("typesafe down")
        del user_text, reply
        return self.outbound


class _StubStream:
    """Minimal stand-in so we can call Stream screen helpers unbound."""

    def __init__(self, screener: _FakeScreener | None, reply: str = "") -> None:
        self.typesafe_screener = screener
        self.message_result = ChatMessage(role="assistant", content=reply) if reply else None

    get_typesafe_input_screen = Stream.get_typesafe_input_screen
    get_typesafe_output_screen = Stream.get_typesafe_output_screen


@pytest.mark.asyncio
async def test_get_typesafe_input_screen_yields_data_event() -> None:
    stub = _StubStream(
        _FakeScreener(
            inbound={
                "side": "input",
                "model": "jev-test",
                "judgments": [{"id": "angry", "type": "noul", "label": "Angry", "value": 0.9}],
            }
        )
    )
    event = await stub.get_typesafe_input_screen(
        [ChatMessage(role="user", content="I hate this")]
    )
    assert isinstance(event, DataEvent)
    assert event.data_type == "typesafe"
    assert event.data["judgments"][0]["value"] == 0.9


@pytest.mark.asyncio
async def test_get_typesafe_input_screen_fail_open() -> None:
    assert (
        await _StubStream(None).get_typesafe_input_screen(
            [ChatMessage(role="user", content="hi")]
        )
        is None
    )
    assert (
        await _StubStream(_FakeScreener(boom_in=True)).get_typesafe_input_screen(
            [ChatMessage(role="user", content="hi")]
        )
        is None
    )
    assert (
        await _StubStream(_FakeScreener(inbound={})).get_typesafe_input_screen(
            [ChatMessage(role="user", content="")]
        )
        is None
    )


@pytest.mark.asyncio
async def test_get_typesafe_output_screen_yields_data_event() -> None:
    stub = _StubStream(
        _FakeScreener(
            outbound={
                "side": "output",
                "model": "jev-test",
                "judgments": [
                    {"id": "hallucination", "type": "noul", "label": "Hallucination", "value": 0.1},
                    {"id": "quality", "type": "score", "label": "Quality", "value": 1.8},
                ],
            }
        ),
        reply="Paris is the capital of France.",
    )
    event = await stub.get_typesafe_output_screen(
        [ChatMessage(role="user", content="What is the capital of France?")]
    )
    assert isinstance(event, DataEvent)
    assert event.data["side"] == "output"
    assert event.data["judgments"][1]["value"] == 1.8


@pytest.mark.asyncio
async def test_get_typesafe_output_screen_fail_open() -> None:
    assert (
        await _StubStream(None, reply="hi").get_typesafe_output_screen(
            [ChatMessage(role="user", content="x")]
        )
        is None
    )
    assert (
        await _StubStream(_FakeScreener(boom_out=True), reply="hi").get_typesafe_output_screen(
            [ChatMessage(role="user", content="x")]
        )
        is None
    )
    assert (
        await _StubStream(_FakeScreener(outbound={}), reply="").get_typesafe_output_screen(
            [ChatMessage(role="user", content="x")]
        )
        is None
    )
