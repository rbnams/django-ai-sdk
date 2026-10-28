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
        messages.extend(_lens_checks(name, resolved))
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
    messages.extend(_alias_checks())
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


def _lens_checks(name: str, resolved: Any) -> list[Any]:
    """Lenses that never run on this deployment, or run only in part.

    Only settings aliases are known here; the console checks its own aliases live.
    """
    from django_ai_sdk.oversight.stages import lens_health
    from django_ai_sdk.oversight.types import Mode

    messages: list[Any] = []
    dead: set[str] = set()
    # Dead only because every answerer is an optional alias left undefined: a choice.
    optional: set[str] = set()
    for lens in resolved.lenses.values():
        if not lens.enabled:
            continue
        health = lens_health(lens)
        if not health["runnable"]:
            dead.add(lens.name)
            if lens.answerers and all(
                isinstance(spec, str) and spec.endswith("?") for spec in lens.answerers
            ):
                optional.add(lens.name)
            why = (
                f"{', '.join(health['skipped'])} not defined in AI_SDK_OVERSIGHT['ANSWERERS']"
                if health["skipped"]
                else "none of its answerers can run (see oversight.W008)"
            )
            messages.append(
                Warning(
                    f"Lens {lens.name!r} in policy {name!r} has no answerer here: {why}. "
                    "Console aliases are checked live.",
                    id="oversight.W006",
                )
            )
        if health["unreachable"]:
            messages.append(
                Warning(
                    f"Lens {lens.name!r} in policy {name!r} never asks "
                    f"{', '.join(health['unreachable'])}: a detector before them is already "
                    "sure when it finds nothing.",
                    id="oversight.W007",
                )
            )
        if health["misplaced"]:
            messages.append(
                Warning(
                    f"Lens {lens.name!r} in policy {name!r} reads the question at watch or "
                    "review, which only judge the answer: it never runs there.",
                    id="oversight.W009",
                )
            )
    for rule in resolved.rules.values():
        blind = sorted(rule.lenses() & dead)
        if rule.mode is not Mode.ENFORCE or not blind:
            continue
        if set(blind) <= optional:
            messages.append(
                Warning(
                    f"Enforced rule {rule.name!r} in policy {name!r} reads {blind}, whose "
                    "answerers are all optional and none is defined here: it never acts.",
                    id="oversight.W011",
                )
            )
        else:
            messages.append(
                Error(
                    f"Enforced rule {rule.name!r} in policy {name!r} reads {blind}, which no "
                    "answerer here can answer.",
                    id="oversight.E007",
                )
            )
    return messages


def _alias_checks() -> list[Any]:
    from django_ai_sdk.oversight.answerers.base import resolve
    from django_ai_sdk.oversight.answerers.llm import LLMJudge
    from django_ai_sdk.oversight.answerers.typesafe import TypeSafe
    from django_ai_sdk.oversight.conf import setting
    from django_ai_sdk.pricing import model_price
    from django_ai_sdk.typesafe.client import typesafe_missing

    messages: list[Any] = []
    for alias in setting("ANSWERERS") or {}:
        try:
            answerer = resolve(alias)
        except Exception:
            continue  # E005 reports it.
        if isinstance(answerer, TypeSafe) and typesafe_missing():
            messages.append(
                Warning(
                    f"Answerer alias {alias!r} is Jev, but {typesafe_missing()}: "
                    "its lenses abstain.",
                    id="oversight.W008",
                )
            )
        if (
            setting("BUDGET_PER_DAY") is not None
            and isinstance(answerer, LLMJudge)
            and answerer.metered
            and model_price(answerer.model) is None
        ):
            messages.append(
                Warning(
                    f"Answerer alias {alias!r} uses model {answerer.model!r}, which "
                    "AI_SDK_MODEL_PRICES does not price: BUDGET_PER_DAY never counts it.",
                    id="oversight.W010",
                )
            )
    return messages


__all__ = ["check_oversight"]
