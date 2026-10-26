"""Process-wide registry of oversight policies.

Every installed app's `oversight` module is imported on startup, so decorating a
Policy subclass with `register` there is enough. A policy that cannot resolve is
kept out and reported by the oversight check, so one app's typo does not stop
the site from booting, and an agent naming it is simply unwatched.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from django.core.exceptions import ImproperlyConfigured

from django_ai_sdk.logger import get_logger

if TYPE_CHECKING:
    from django_ai_sdk.oversight.policy import Policy

logger = get_logger(__name__)

_registry: dict[str, type[Policy]] = {}

# Rejected declarations, name -> reason. Read by the system check.
_invalid: dict[str, str] = {}


def register[T: type[Policy]](cls: T) -> T:
    """Class decorator: add a Policy to the registry unless it is incomplete."""
    try:
        validate(cls)
    except ImproperlyConfigured as exc:
        _invalid[cls.name or cls.__name__] = str(exc)
        logger.warning("Oversight policy not registered: {}", exc)
        return cls

    _invalid.pop(cls.name, None)
    existing = _registry.get(cls.name)
    if existing is not None and existing is not cls:
        logger.warning(
            "Oversight policy {!r} is declared by both {} and {}; the later import wins",
            cls.name,
            existing.__name__,
            cls.__name__,
        )
    _registry[cls.name] = cls
    return cls


def validate(cls: type[Policy]) -> None:
    """Raise ImproperlyConfigured unless every lens and rule is complete and consistent.

    Inherited lenses are checked when the policy resolves (oversight.E002), since
    a parent may register after its child.
    """
    from django_ai_sdk.oversight.lenses import Choice

    if not cls.name:
        raise ImproperlyConfigured(f"{cls.__name__} must set a non-empty `name`.")
    policy = cls()
    names: set[str] = set()
    for lens in policy.get_lenses():
        if not lens.name:
            raise ImproperlyConfigured(f"Policy {cls.name!r} has a lens without a `name`.")
        if lens.name in names:
            raise ImproperlyConfigured(f"Policy {cls.name!r} declares lens {lens.name!r} twice.")
        if isinstance(lens, Choice) and not lens.labels:
            raise ImproperlyConfigured(f"Choice lens {lens.name!r} declares no `labels`.")
        if not lens.answerers:
            raise ImproperlyConfigured(f"Lens {lens.name!r} has no `answerers`.")
        names.add(lens.name)
    rules: set[str] = set()
    for rule in policy.get_rules():
        if rule.name in rules:
            raise ImproperlyConfigured(f"Policy {cls.name!r} declares rule {rule.name!r} twice.")
        if not rule.then:
            raise ImproperlyConfigured(f"Rule {rule.name!r} does nothing: give it `then`.")
        rules.add(rule.name)


def get_policy_class(name: str) -> type[Policy] | None:
    return _registry.get(name)


def get_policies() -> dict[str, type[Policy]]:
    return dict(_registry)


def get_invalid_policies() -> dict[str, str]:
    return dict(_invalid)


def reset_registry() -> None:
    """Clear the registry, and every cached resolution — for tests."""
    from django_ai_sdk.oversight.policy import invalidate

    _registry.clear()
    _invalid.clear()
    invalidate()


__all__ = [
    "get_invalid_policies",
    "get_policies",
    "get_policy_class",
    "register",
    "reset_registry",
    "validate",
]
