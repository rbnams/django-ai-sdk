"""Watching tool calls: Haystack hooks that ask the turn's session.

Installed on every tool agent, subagents included; without an active session they
return at once. Every call and its result is recorded in the turn. Before tools
run, when any pending call is refused, the whole step is answered with stubs:
Haystack runs a step's calls together or not at all, and the model is told which
call was refused and that the rest may be retried. After they run, their results
are masked the way the gate masks the question, before the model reads them.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from django.utils.translation import gettext_lazy
from haystack.components.agents.state.state import replace_values
from haystack.dataclasses import ChatMessage, TextContent

from django_ai_sdk.oversight.types import ToolCall

if TYPE_CHECKING:
    from haystack.components.agents.state.state import State

    from django_ai_sdk.oversight.session import TurnSession

# What the model is told about a call that was not run because another was refused.
SKIPPED = gettext_lazy(
    "Not run: another call in this step was refused. Call it again on its own if you still need it."
)

# message meta on a tool result oversight refused
REFUSED_META_KEY = "django_ai_sdk.oversight.refused"
# message meta on a tool result oversight has masked (or found nothing to mask in)
MASKED_META_KEY = "django_ai_sdk.oversight.masked"
# Why every call of a sync run inside a turn is refused: its hooks cannot await the session.
UNWATCHABLE = gettext_lazy("This tool cannot be watched in a sync run, so it is not run.")


class OversightToolHook:
    """Refuse tool calls the turn's policy says no to, before they run."""

    allowed_hook_points = ["before_tool"]

    def run(self, state: State) -> None:
        """A sync run: outside a turn, nothing to do; inside one, every call is refused.

        The session can only be asked from async code, so a call here could not be
        judged: none runs unwatched.
        """
        from django_ai_sdk.oversight.session import current_session

        session = current_session()
        if session is None:
            return
        messages = state.data.get("messages") or []
        if not messages or not messages[-1].tool_calls:
            return
        calls = messages[-1].tool_calls
        for call in calls:
            session.add_tool_call(
                ToolCall(id=call.id or "", name=call.tool_name, arguments=call.arguments)
            )
        session.note_error("tool called in a sync run")
        _answer_with_stubs(state, messages, {i: str(UNWATCHABLE) for i in range(len(calls))})

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
            seen = ToolCall(id=call.id or "", name=call.tool_name, arguments=call.arguments)
            session.add_tool_call(seen)
            reason = await session.check_tool(seen)
            if reason:
                refusals[index] = reason
        if not refusals:
            return
        stubs = _answer_with_stubs(state, messages, refusals)
        if session.on_tool_results is not None:
            await session.on_tool_results(stubs)


def _answer_with_stubs(
    state: State, messages: list[ChatMessage], refusals: dict[int, str]
) -> list[ChatMessage]:
    """Answer the step's calls with stubs: the refused ones say why, the rest were skipped."""
    calls = messages[-1].tool_calls
    stubs = [
        ChatMessage.from_tool(
            json.dumps(
                {"refused": refusals[index]} if index in refusals else {"skipped": str(SKIPPED)}
            ),
            origin=call,
            error=True,
            meta={REFUSED_META_KEY: index in refusals},
        )
        for index, call in enumerate(calls)
    ]
    state.set("messages", [*messages, *stubs], handler_override=replace_values)
    return stubs


class OversightToolResultHook:
    """Mask tool results before the model reads them, as the gate masks the question.

    A tool can hand the model what the gate keeps from it: a record's number, a
    client's name. Each result is judged once, when it arrives.
    """

    allowed_hook_points = ["after_tool"]

    def run(self, state: State) -> None:
        """A sync run inside a turn: a result that is not a refusal is withheld unread.

        The before-tool hook refuses every call there, so this only meets results it
        did not see run.
        """
        from django_ai_sdk.oversight.session import UNCHECKED, current_session

        session = current_session()
        if session is None:
            return
        messages = list(state.data.get("messages") or [])
        changed = False
        for index, message in enumerate(messages):
            result = message.tool_call_result
            if result is None or MASKED_META_KEY in message.meta:
                continue
            # Stubs (refused or skipped) carry REFUSED_META_KEY; anything else ran unwatched.
            if REFUSED_META_KEY not in message.meta:
                messages[index] = ChatMessage.from_tool(
                    str(UNCHECKED),
                    origin=result.origin,
                    error=result.error,
                    meta={**message.meta, MASKED_META_KEY: True},
                )
                changed = True
        if changed:
            state.set("messages", messages, handler_override=replace_values)

    async def run_async(self, state: State) -> None:
        from django_ai_sdk.oversight.session import current_session

        session = current_session()
        if session is None:
            return
        messages = list(state.data.get("messages") or [])
        changed = False
        for index, message in enumerate(messages):
            result = message.tool_call_result
            if result is None or MASKED_META_KEY in message.meta:
                continue
            masked = await masked_result(session, result.origin.id or "", result.result)
            session.add_tool_call(
                ToolCall(
                    id=result.origin.id or "",
                    name="" if result.origin.id else result.origin.tool_name,
                    result=masked if isinstance(masked, str) else _text_of(masked),
                    error=bool(result.error),
                )
            )
            messages[index] = ChatMessage.from_tool(
                masked,
                origin=result.origin,
                error=result.error,
                meta={**message.meta, MASKED_META_KEY: True},
            )
            changed = True
        if changed:
            state.set("messages", messages, handler_override=replace_values)


async def masked_result(session: TurnSession, call_id: str, result: Any) -> Any:
    """A tool's result as the model and the person may see it: text masked, the rest kept.

    A result is a string or a list of content parts; images and files pass as they are.
    """
    if isinstance(result, str):
        return await session.mask_result(call_id, result)
    if not isinstance(result, list):
        return result
    return [
        TextContent(text=await session.mask_result(f"{call_id}#{index}", part.text))
        if isinstance(part, TextContent)
        else part
        for index, part in enumerate(result)
    ]


def _text_of(parts: list[Any]) -> str:
    return "\n".join(part.text for part in parts if isinstance(part, TextContent))


__all__ = [
    "MASKED_META_KEY",
    "REFUSED_META_KEY",
    "OversightToolHook",
    "OversightToolResultHook",
    "masked_result",
]
