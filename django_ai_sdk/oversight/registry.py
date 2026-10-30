"""Process-wide registry of oversight policies.

Every installed app's `oversight` module is imported on startup, so decorating a
Policy subclass with `register` there is enough. A policy that cannot resolve is
kept out and reported by the oversight check, so one app's typo does not stop
the site from booting. Two classes declaring one name are an error too: the first
keeps the name, and the check fails until one is renamed.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

from django.core.exceptions import ImproperlyConfigured

from django_ai_sdk.logger import get_logger
from django_ai_sdk.oversight.rules import Mask

if TYPE_CHECKING:
    from collections.abc import Mapping

    from django_ai_sdk.oversight.lenses import Lens
    from django_ai_sdk.oversight.policy import Policy
    from django_ai_sdk.oversight.rules import Rule

logger = get_logger(__name__)

_registry: dict[str, type[Policy]] = {}

# Rejected declarations, name -> reason. Read by the system check.
_invalid: dict[str, str] = {}

# Names declared by more than one class, name -> every class path that declares it.
# The first one registered keeps the name; the system check reports the clash.
_duplicates: dict[str, list[str]] = {}


def _path(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _declaration(cls: type) -> tuple[str, str]:
    """Where a class is declared: one file imported under two module names is one declaration."""
    try:
        where = inspect.getfile(cls)
    except (TypeError, OSError):
        where = cls.__module__
    return where, cls.__qualname__


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
    if existing is not None and _declaration(existing) != _declaration(cls):
        # Two policies under one name: an agent naming it could run either. The first
        # keeps it, and the check (ai_sdk.oversight.E010) fails until one is renamed.
        clash = _duplicates.setdefault(cls.name, [_path(existing)])
        if _path(cls) not in clash:
            clash.append(_path(cls))
        logger.error("Oversight policy {!r} is declared by both {} and {}", cls.name, *clash[:2])
        return cls
    # The same declaration again (a re-import) replaces itself.
    _registry[cls.name] = cls
    return cls


def validate(cls: type[Policy]) -> None:
    """Raise ImproperlyConfigured unless every lens and rule is complete and consistent.

    Inherited lenses are checked when the policy resolves (ai_sdk.oversight.E002), since
    a parent may register after its child.
    """
    if not cls.name:
        raise ImproperlyConfigured(f"{cls.__name__} must set a non-empty `name`.")
    policy = cls()
    lenses: dict[str, Lens] = {}
    for lens in policy.get_lenses():
        lens.check()
        if lens.name in lenses:
            raise ImproperlyConfigured(f"Policy {cls.name!r} declares lens {lens.name!r} twice.")
        if not lens.answerers:
            raise ImproperlyConfigured(f"Lens {lens.name!r} has no `answerers`.")
        lenses[lens.name] = lens
    rules: set[str] = set()
    for rule in policy.get_rules():
        if rule.name in rules:
            raise ImproperlyConfigured(f"Policy {cls.name!r} declares rule {rule.name!r} twice.")
        if not rule.then:
            raise ImproperlyConfigured(f"Rule {rule.name!r} does nothing: give it `then`.")
        check_rule(rule, lenses)
        rules.add(rule.name)


def check_rule(rule: Rule, lenses: Mapping[str, Lens]) -> None:
    """Raise ImproperlyConfigured when `rule` asks what its lenses cannot give.

    A Mask masks where its lenses point, and only an Entities lens says where: a
    Mask over a Flag, Choice, Score or Text lens could never mask anything, nor
    one over a tool-call lens (arguments are judged, never rewritten). Lenses
    not in `lenses` (inherited, not resolved yet) are checked when the policy is.
    """
    for action in rule.then:
        if not isinstance(action, Mask):
            continue
        for name in sorted(set(action.lenses) or rule.lenses()):
            lens = lenses.get(name)
            if lens is not None and lens.subject == "tool_call":
                raise ImproperlyConfigured(
                    f"Rule {rule.name!r} masks what tool-call lens {name!r} found, but a tool "
                    "call's arguments are never masked: block the call (Block) or take the "
                    "tool away (RestrictTools). Tool results are masked by the gate's mask rules."
                )
            if lens is not None and lens.kind != "entities":
                raise ImproperlyConfigured(
                    f"Rule {rule.name!r} masks what {lens.kind} lens {name!r} found, but "
                    "only an entities lens says where: name the lenses to mask, "
                    "Mask(lenses=[...])."
                )


def get_policy_class(name: str) -> type[Policy] | None:
    return _registry.get(name)


def policy_name(policy: type[Policy] | str) -> str:
    """The name `policy` is known by: a name as given, a Policy subclass's `name`.

    A subclass nobody registered yet is registered here, so naming it on an agent
    (`oversight = SupportPolicy`) is enough.
    """
    if isinstance(policy, str):
        return policy
    from django_ai_sdk.oversight.policy import Policy

    if not (isinstance(policy, type) and issubclass(policy, Policy)):
        raise TypeError(f"Expected a Policy subclass or a policy name, got {policy!r}")
    known = _registry.get(policy.name)
    if known is None and policy.name not in _invalid:
        register(policy)
    elif known is not None and known is not policy and _declaration(known) == _declaration(policy):
        register(policy)  # the same declaration, imported again: the copy in use wins
    return policy.name


def get_policies() -> dict[str, type[Policy]]:
    return dict(_registry)


def get_invalid_policies() -> dict[str, str]:
    return dict(_invalid)


def get_duplicate_policies() -> dict[str, list[str]]:
    """Names more than one Policy class declares, with the class paths that do."""
    return {name: list(paths) for name, paths in _duplicates.items()}


def reset_registry() -> None:
    """Clear the registry, and every cached resolution — for tests."""
    from django_ai_sdk.oversight.policy import forget_last_good, invalidate

    _registry.clear()
    _invalid.clear()
    _duplicates.clear()
    invalidate()
    forget_last_good()


__all__ = [
    "check_rule",
    "get_duplicate_policies",
    "get_invalid_policies",
    "get_policies",
    "get_policy_class",
    "policy_name",
    "register",
    "reset_registry",
    "validate",
]
