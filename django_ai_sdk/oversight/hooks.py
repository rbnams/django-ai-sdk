"""Watching tool calls: a Haystack before_tool hook that asks the turn's session.

Installed by `default_hooks` on every tool agent; without an active session it
returns at once. When any pending call is refused, the whole step is answered
with stubs: Haystack runs a step's calls together or not at all, and the model
is told which call was refused and that the rest may be retried.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from haystack.components.agents.state.state import replace_values
from haystack.dataclasses import ChatMessage

from django_ai_sdk.oversight.types import ToolCall

if TYPE_CHECKING:
    from haystack.components.agents.state.state import State

# message meta on a tool result oversight refused
REFUSED_META_KEY = "django_ai_sdk.oversight.refused"


class OversightToolHook:
    """Refuse tool calls the turn's policy says no to, before they run."""

    allowed_hook_points = ["before_tool"]

    def run(self, state: State) -> None:
        # Agents built for sync runs have no session to ask; nothing to do.
        return

    async def run_async(self, state: State) -> None:
        from django_ai_sdk.oversight.session import current_session

        session = current_session()
        if session is None:
            return
        messages = state.data.get("messages") or []
        if not messages or not messages[-1].tool_calls:
            return
        calls = messages[-1].tool_calls
        refusals: dict[int, str] = {}
        for index, call in enumerate(calls):
            reason = await session.check_tool(
                ToolCall(id=call.id or "", name=call.tool_name, arguments=call.arguments)
            )
            if reason:
                refusals[index] = reason
        if not refusals:
            return
        stubs = [
            ChatMessage.from_tool(
                json.dumps(
                    {"refused": refusals[index]}
                    if index in refusals
                    else {
                        "skipped": "Not run: another call in this step was refused. "
                        "Call it again on its own if you still need it."
                    }
                ),
                origin=call,
                error=True,
                meta={REFUSED_META_KEY: index in refusals},
            )
            for index, call in enumerate(calls)
        ]
        state.set("messages", [*messages, *stubs], handler_override=replace_values)


__all__ = ["REFUSED_META_KEY", "OversightToolHook"]
