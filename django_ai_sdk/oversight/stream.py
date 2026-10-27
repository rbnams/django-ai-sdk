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


def stored_notices(notices: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Notices as kept on the stored message: "replace" says only that the text changed.

    The stored content is the replacement already; a second copy in the metadata
    would go stale when reflect masks the message later.
    """
    return [
        {**{k: v for k, v in n.items() if k != "replace"}, "replaced": True}
        if "replace" in n
        else n
        for n in notices
    ]


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


def observe_replies(session: TurnSession, replies: list[Any]) -> None:
    """Tell the session what a headless tool loop did: its calls, results and usage.

    A streamed turn reports these event by event; a run that does not stream has
    only the finished messages.
    """
    from django_ai_sdk.adapters.citations.registry import current_registry
    from django_ai_sdk.pricing import usage_from_meta

    for message in replies:
        for call in getattr(message, "tool_calls", None) or []:
            session.add_tool_call(
                ToolCall(id=call.id or "", name=call.tool_name, arguments=call.arguments)
            )
        for result in getattr(message, "tool_call_results", None) or []:
            session.add_tool_call(
                ToolCall(
                    id=result.origin.id or "",
                    name=result.origin.tool_name if not result.origin.id else "",
                    result=result.result,
                    error=bool(result.error),
                )
            )
        meta = getattr(message, "meta", None) or {}
        prompt, completion = usage_from_meta(meta)
        if prompt is not None or completion is not None:
            session.add_usage(meta.get("model"), prompt, completion)
    registry = current_registry.get()
    if registry is not None:
        session.set_sources(registry.all, seeded=registry.seeded)


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
    "observe_replies",
    "restrict_agent_tools",
    "stored_notices",
]
