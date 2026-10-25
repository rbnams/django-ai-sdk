"""TypeSafe unit tests for tool control hints (no live API)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from django_ai_sdk.adapters.base import Stream
from django_ai_sdk.common import ChatMessage
from django_ai_sdk.events import DataEvent
from django_ai_sdk.typesafe.control import (
    apply_control_to_agent_component,
    apply_tool_control,
    enrich_payload_acted,
)


def test_apply_tool_control_strip_tools() -> None:
    tools = [SimpleNamespace(name="weather_get"), SimpleNamespace(name="wiki_search")]
    assert apply_tool_control(tools, {"strip_tools": True}) == []


def test_apply_tool_control_skip_rag_and_names() -> None:
    tools = [
        SimpleNamespace(name="weather_get"),
        SimpleNamespace(name="bm25_rag_tool"),
        SimpleNamespace(name="wiki_search"),
        SimpleNamespace(name="wiki_read"),
    ]
    out = apply_tool_control(
        tools,
        {"skip_rag": True, "remove_tool_names": ["wiki_search", "wiki_read"]},
    )
    assert [t.name for t in out] == ["weather_get"]


def test_apply_control_to_agent_component() -> None:
    agent = SimpleNamespace(tools=[SimpleNamespace(name="a"), SimpleNamespace(name="b")])
    apply_control_to_agent_component(agent, {"strip_tools": True})
    assert agent.tools == []


def test_enrich_payload_acted() -> None:
    payload = enrich_payload_acted({"control": {"strip_tools": True}, "judgments": []})
    assert payload["acted"] == ["stripped_tools"]


class _FakeScreener:
    def __init__(self, inbound: dict | None) -> None:
        self.inbound = inbound

    async def screen_input(self, text: str) -> dict | None:
        del text
        return self.inbound

    async def screen_output(self, user_text: str, reply: str) -> dict | None:
        del user_text, reply
        return None


class _StubStream:
    def __init__(self, screener: _FakeScreener, tools: list) -> None:
        self.typesafe_screener = screener
        self.agent_component = SimpleNamespace(tools=tools)
        self.message_result = None

    get_typesafe_input_screen = Stream.get_typesafe_input_screen


@pytest.mark.asyncio
async def test_inbound_screen_applies_control_to_tools() -> None:
    tools = [SimpleNamespace(name="wiki_search"), SimpleNamespace(name="weather_get")]
    stub = _StubStream(
        _FakeScreener(
            {
                "side": "input",
                "judgments": [],
                "control": {"strip_tools": True},
            }
        ),
        tools,
    )
    event = await stub.get_typesafe_input_screen(
        [ChatMessage(role="user", content="hello")]
    )
    assert isinstance(event, DataEvent)
    assert stub.agent_component.tools == []
    assert event.data["acted"] == ["stripped_tools"]
