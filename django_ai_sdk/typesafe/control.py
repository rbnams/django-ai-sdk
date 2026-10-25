"""Generic TypeSafe control hints — product policy stays in the host app.

A screener may attach ``control`` to its inbound payload. Stream / Agent apply
these hints without knowing question ids or thresholds.
"""

from __future__ import annotations

from typing import Any


def tool_name(tool: Any) -> str:
    return str(getattr(tool, "name", "") or "")


def is_rag_tool(tool: Any) -> bool:
    """Heuristic: Haystack RAG ComponentTools use names containing ``rag``."""
    return "rag" in tool_name(tool).lower()


def apply_tool_control(tools: list[Any], control: dict[str, Any] | None) -> list[Any]:
    """Filter a tool list according to optional control hints."""
    if not control:
        return tools
    if control.get("strip_tools"):
        return []
    remaining = list(tools)
    if control.get("skip_rag"):
        remaining = [t for t in remaining if not is_rag_tool(t)]
    remove = {str(n) for n in (control.get("remove_tool_names") or []) if n}
    if remove:
        remaining = [t for t in remaining if tool_name(t) not in remove]
    return remaining


def acted_from_control(control: dict[str, Any] | None) -> list[str]:
    """Human-readable actions taken from control hints (for UI cards)."""
    if not control:
        return []
    acted: list[str] = []
    if control.get("strip_tools"):
        acted.append("stripped_tools")
    else:
        if control.get("skip_rag"):
            acted.append("skipped_rag")
        names = control.get("remove_tool_names") or []
        if names:
            acted.append(f"removed:{','.join(str(n) for n in names)}")
    return acted


def apply_control_to_agent_component(agent_component: Any, control: dict[str, Any] | None) -> None:
    """Mutate a Haystack Agent component's tools in place."""
    if not control or agent_component is None:
        return
    current = list(getattr(agent_component, "tools", None) or [])
    agent_component.tools = apply_tool_control(current, control)


def enrich_payload_acted(payload: dict[str, Any]) -> dict[str, Any]:
    """Ensure ``acted`` is set when ``control`` produced side effects."""
    control = payload.get("control")
    if not isinstance(control, dict):
        return payload
    if payload.get("acted"):
        return payload
    acted = acted_from_control(control)
    if acted:
        payload = {**payload, "acted": acted}
    return payload
