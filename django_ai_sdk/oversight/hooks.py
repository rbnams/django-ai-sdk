"""Watching tool calls: Haystack hooks that ask the turn's session.

Installed by `default_hooks` on every tool agent; without an active session they
return at once. Before tools run, when any pending call is refused, the whole step
is answered with stubs: Haystack runs a step's calls together or not at all, and
the model is told which call was refused and that the rest may be retried. After
they run, their results are masked the way the gate masks the question, before
the model reads them.
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
# message meta on a tool result oversight has masked (or found nothing to mask in)
MASKED_META_KEY = "django_ai_sdk.oversight.masked"


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
        if session.on_tool_results is not None:
            await session.on_tool_results(stubs)


class OversightToolResultHook:
    """Mask tool results before the model reads them, as the gate masks the question.

    A tool can hand the model what the gate keeps from it: a record's number, a
    client's name. Each result is judged once, when it arrives.
    """

    allowed_hook_points = ["after_tool"]

    def run(self, state: State) -> None:
        return

    async def run_async(self, state: State) -> None:
        from django_ai_sdk.oversight.session import current_session

        session = current_session()
        if session is None:
            return
        messages = list(state.data.get("messages") or [])
        changed = False
        for index, message in enumerate(messages):
            result = message.tool_call_result
            if (
                result is None
                or not isinstance(result.result, str)
                or MASKED_META_KEY in message.meta
            ):
                continue
            messages[index] = ChatMessage.from_tool(
                await session.mask_text(result.result),
                origin=result.origin,
                error=result.error,
                meta={**message.meta, MASKED_META_KEY: True},
            )
            changed = True
        if changed:
            state.set("messages", messages, handler_override=replace_values)


__all__ = ["MASKED_META_KEY", "REFUSED_META_KEY", "OversightToolHook", "OversightToolResultHook"]
