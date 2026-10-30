"""Which policy an agent's turn gets, and how a turn knows its agent.

An agent asks for a policy in code (its `oversight` attribute, or an async
`get_oversight(channel=, user=)` hook); the console may assign another by the
agent's key. These are the code side of that choice.
"""

from __future__ import annotations

import hashlib
import inspect
from typing import TYPE_CHECKING, Any

from django_ai_sdk.oversight.conf import setting

if TYPE_CHECKING:
    from django_ai_sdk.agents.base import Agent

# Hex characters of an agent's fingerprint.
FINGERPRINT_CHARS = 16


def agent_key(agent: Agent) -> str:
    """The id turns and console choices know an agent by: its runtime id, else its registry id.

    A runtime agent built on a registered class has both; the console assigns it by
    its own id.
    """
    return str(getattr(agent, "agent_id", "") or "")


async def acode_policy(agent: Agent, *, channel: str = "", user: Any = None) -> tuple[str, str]:
    """The policy the agent's code asks for this turn, and why: "agent", "off" or "default".

    Awaits the agent's `get_oversight(channel=, user=)`; an object without one is
    read by its `oversight` attribute. Raises what the hook raises, and TypeError
    for what names no policy: the caller decides what a failure means.
    """
    getter = getattr(agent, "get_oversight", None)
    if callable(getter):
        chosen = getter(channel=channel, user=user)
        if inspect.isawaitable(chosen):
            chosen = await chosen
    else:
        chosen = getattr(agent, "oversight", None)
    return _code_choice(chosen)


def code_policy(agent: Any) -> tuple[str, str]:
    """What the agent's class says, without asking its `get_oversight()`: for checks.

    The policy's name and why: "agent", "off" or "default".
    """
    return _code_choice(getattr(agent, "oversight", None))


def _code_choice(chosen: Any) -> tuple[str, str]:
    from django_ai_sdk.oversight.registry import policy_name

    if chosen is False:
        return "", "off"
    if chosen is None or chosen == "":
        return str(setting("DEFAULT_POLICY") or ""), "default"
    return policy_name(chosen), "agent"


def fingerprint(agent: Agent) -> str:
    """What the agent was, compactly: a change of model, prompt or tools changes it."""
    parts: list[str] = []
    for getter in ("get_model", "get_system_prompt"):
        fn = getattr(agent, getter, None)
        try:
            parts.append(str(fn() if callable(fn) else ""))
        except Exception:
            parts.append("")
    tools = getattr(type(agent), "tools", []) or []
    parts.append(",".join(sorted(getattr(t, "__name__", str(t)) for t in tools)))
    parts.append(",".join(sorted(getattr(agent, "integrations", []) or [])))
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:FINGERPRINT_CHARS]


__all__ = ["FINGERPRINT_CHARS", "acode_policy", "agent_key", "code_policy", "fingerprint"]
