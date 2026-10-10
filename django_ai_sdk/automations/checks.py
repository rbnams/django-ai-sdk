"""Startup diagnostics for automations.

All Warnings: a background job must never fail boot.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from django.core.checks import Warning as CheckWarning

if TYPE_CHECKING:
    from collections.abc import Sequence

ID_PREFIX = "ai_sdk.automations"


def check_automations(app_configs: Any = None, **kwargs: Any) -> Sequence[Any]:
    """Report anything that would stop a declared automation from running."""
    from django_ai_sdk.automations.registry import get_automations, get_invalid_automations

    issues: list[Any] = _check_refused(get_invalid_automations())
    for name, automation in get_automations().items():
        issues.extend(_check_workflow(name, automation))
    return issues


def _check_refused(refused: dict[str, str]) -> list[Any]:
    """Declarations register() kept out of the registry, one Warning each."""
    return [
        CheckWarning(
            f"Automation {name!r} was not registered: {reason}",
            hint="Fix the declaration in the app's automations module.",
            id=f"{ID_PREFIX}.W006",
        )
        for name, reason in sorted(refused.items())
    ]


def _check_workflow(name: str, automation: Any) -> list[Any]:
    """An automation naming a workflow nothing declares.

    Code-declared only: the check is synchronous, so a stored workflow cannot be seen.
    """
    from django_ai_sdk.workflows.registry import get_declared_workflows

    declared = get_declared_workflows()
    if automation.workflow in declared:
        return []
    return [
        CheckWarning(
            f"Automation {name!r} runs workflow {automation.workflow!r}, which no "
            "installed app declares. If it is not a database workflow, its runs will "
            "be skipped.",
            hint="Declared workflows: " + (", ".join(sorted(declared)) or "none"),
            id=f"{ID_PREFIX}.W003",
        )
    ]


__all__ = ["check_automations"]
