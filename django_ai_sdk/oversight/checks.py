"""System checks: a policy that cannot run fails no request, so nothing else would say."""

from __future__ import annotations

from typing import Any

from django.core.checks import Error, Warning


def check_oversight(app_configs: Any = None, **kwargs: Any) -> list[Any]:
    from django_ai_sdk.oversight import vault
    from django_ai_sdk.oversight.answerers.base import alias_defined, resolve
    from django_ai_sdk.oversight.conf import is_enabled, setting
    from django_ai_sdk.oversight.policy import build
    from django_ai_sdk.oversight.registry import get_invalid_policies, get_policies

    messages: list[Any] = []
    for name, reason in get_invalid_policies().items():
        messages.append(
            Error(f"Oversight policy {name!r} was not registered: {reason}", id="oversight.E001")
        )
    for name in get_policies():
        try:
            resolved = build(name)
        except Exception as exc:
            messages.append(
                Error(f"Oversight policy {name!r} does not resolve: {exc}", id="oversight.E002")
            )
            continue
        for rule in resolved.rules.values():
            unknown = rule.lenses() - set(resolved.lenses)
            if unknown:
                messages.append(
                    Error(
                        f"Rule {rule.name!r} in policy {name!r} reads unknown lenses {sorted(unknown)}",
                        id="oversight.E003",
                    )
                )
        for lens in resolved.lenses.values():
            for spec in lens.answerers:
                if isinstance(spec, str) and spec.endswith("?"):
                    continue
                if isinstance(spec, str) and not alias_defined(spec):
                    messages.append(
                        Error(
                            f"Lens {lens.name!r} in policy {name!r} names answerer alias {spec!r}, "
                            "which AI_SDK_OVERSIGHT['ANSWERERS'] does not define",
                            id="oversight.E004",
                        )
                    )
    for alias, spec in (setting("ANSWERERS") or {}).items():
        try:
            resolve(spec)
        except Exception as exc:
            messages.append(
                Error(f"Answerer alias {alias!r} cannot be built: {exc}", id="oversight.E005")
            )
    default = setting("DEFAULT_POLICY")
    if default and default not in get_policies():
        messages.append(
            Error(
                f"AI_SDK_OVERSIGHT['DEFAULT_POLICY'] names unknown policy {default!r}",
                id="oversight.E006",
            )
        )
    if is_enabled() and not vault.available():
        messages.append(
            Warning(
                "Oversight stores evidence in the clear: set AI_SDK_OVERSIGHT['MASTER_KEY'] and "
                "install django-ai-sdk[oversight] to encrypt it and allow crypto-shredding.",
                id="oversight.W004",
            )
        )
    if is_enabled() and not setting("SIGNING_KEY"):
        messages.append(
            Warning(
                "Oversight checkpoints are unsigned: set AI_SDK_OVERSIGHT['SIGNING_KEY'].",
                id="oversight.W005",
            )
        )
    return messages


__all__ = ["check_oversight"]
