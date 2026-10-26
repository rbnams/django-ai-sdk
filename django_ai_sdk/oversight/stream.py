"""What a chat stream says when oversight stepped in.

Every intervention reaches the client as a `data-oversight` part, so a UI can
show that a question was masked, a tool refused, an answer withheld or replaced.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from django_ai_sdk.events import (
    DataEvent,
    MessageEndEvent,
    MessageStartEvent,
    StreamEndEvent,
    TextChunkEvent,
    TextEndEvent,
    ToolInputCompleteEvent,
    ToolOutputEvent,
)
from django_ai_sdk.oversight.types import ToolCall

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from django_ai_sdk.adapters.suggestions import SuggestionGenerator
    from django_ai_sdk.common import ChatMessage
    from django_ai_sdk.events import StreamEvent
    from django_ai_sdk.oversight.session import TurnSession

# The finish reason a client sees on an answer oversight withheld.
CONTENT_FILTER = "content-filter"


def notice_event(notice: dict[str, Any]) -> DataEvent:
    return DataEvent(data_type="oversight", data=notice)


class RefusalStream:
    """The whole reply to a question the gate refused: the policy's message, nothing stored."""

    model: str | None = None
    instructions: str | None = None
    suggestion_generator: SuggestionGenerator | None = None
    typesafe_screener: Any = None

    def __init__(self, message: str, notice: dict[str, Any] | None = None) -> None:
        self.message = message
        self.notice = notice or {}

    async def stream(self, messages: list[ChatMessage]) -> AsyncGenerator[StreamEvent, None]:
        yield MessageStartEvent(message_id=str(uuid.uuid4()))
        yield notice_event({**self.notice, "message": self.message})
        yield TextChunkEvent(content=self.message)
        yield TextEndEvent()
        yield MessageEndEvent(finish_reason=CONTENT_FILTER)
        yield StreamEndEvent()


def observe_event(session: TurnSession, event: StreamEvent) -> None:
    """Tell the session about a tool call or its result as the stream reports it."""
    if isinstance(event, ToolInputCompleteEvent):
        session.add_tool_call(
            ToolCall(
                id=event.tool_call_id,
                name=event.tool_name,
                arguments=event.tool_input,
                agent=event.agent or "",
            )
        )
    elif isinstance(event, ToolOutputEvent):
        output = event.tool_output
        failed = isinstance(output, dict) and bool(output.get("error"))
        session.add_tool_call(ToolCall(id=event.tool_call_id, name="", result=output, error=failed))


def restrict_agent_tools(agent_component: Any, names: list[str]) -> None:
    """Take the gate's restricted tools off a built Haystack agent; [] takes them all."""
    from django_ai_sdk.typesafe.control import apply_control_to_agent_component

    control = {"remove_tool_names": names} if names else {"strip_tools": True}
    apply_control_to_agent_component(agent_component, control)


__all__ = [
    "CONTENT_FILTER",
    "RefusalStream",
    "notice_event",
    "observe_event",
    "restrict_agent_tools",
]
