"""TypeSafe screener protocol — batteries live in the host app."""

from __future__ import annotations

from typing import Any, Protocol


class TypeSafeScreener(Protocol):
    """Something Stream can call for inbound and/or outbound score cards.

    Implementations may attach optional ``control`` hints on the inbound
    payload (``strip_tools``, ``skip_rag``, ``remove_tool_names``). Stream and
    Agent apply those generically; see ``django_ai_sdk.typesafe.control``.
    """

    async def screen_input(self, text: str) -> dict[str, Any] | None:
        """Score the latest user message before generation."""
        ...

    async def screen_output(self, user_text: str, reply: str) -> dict[str, Any] | None:
        """Score the finished assistant reply (display only)."""
        ...
