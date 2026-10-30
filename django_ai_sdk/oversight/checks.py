"""System checks: a policy that cannot run fails no request, so nothing else would say.

Ids are `ai_sdk.oversight.E001`… and `ai_sdk.oversight.W001`…, like the SDK's
other apps. W001 and W002 (no MASTER_KEY, no SIGNING_KEY) are deploy checks:
`manage.py check --deploy` reports them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from django.core.checks import Error, Warning

PREFIX = "ai_sdk.oversight"


def check_oversight(app_configs: Any = None, **kwargs: Any) -> list[Any]:
    from django_ai_sdk.oversight.answerers.base import alias_defined, resolve
    from django_ai_sdk.oversight.conf import is_enabled, setting
    from django_ai_sdk.oversight.policy import build
    from django_ai_sdk.oversight.registry import (
        get_duplicate_policies,
        get_invalid_policies,
        get_policies,
    )

    messages: list[Any] = []
    # Lenses no answerer here can answer, reported together in one warning.
    unanswered: dict[str, Unanswered] = {}
    in_use = _policies_in_use()
    for name, reason in get_invalid_policies().items():
        messages.append(
            Error(f"Oversight policy {name!r} was not registered: {reason}", id=f"{PREFIX}.E001")
        )
    for name, paths in get_duplicate_policies().items():
        messages.append(
            Error(
                f"Oversight policy name {name!r} is declared by {', '.join(paths)}: an agent "
                "naming it could run either. The first one registered keeps the name.",
                hint="Give each policy class a name of its own.",
                id=f"{PREFIX}.E010",
            )
        )
    for name in get_policies():
        try:
            resolved = build(name)
        except Exception as exc:
            messages.append(
                Error(f"Oversight policy {name!r} does not resolve: {exc}", id=f"{PREFIX}.E002")
            )
            continue
        for rule in resolved.rules.values():
            unknown = rule.lenses() - set(resolved.lenses)
            if unknown:
                messages.append(
                    Error(
                        f"Rule {rule.name!r} in policy {name!r} reads unknown lenses {sorted(unknown)}",
                        id=f"{PREFIX}.E003",
                    )
                )
        messages.extend(_rejected_settings(name, resolved))
        # Lenses nothing here answers: only where a policy is in use is that worth a word.
        messages.extend(_lens_checks(name, resolved, unanswered if name in in_use else {}))
        if name in in_use:
            messages.extend(_live_checks(name, resolved))
        for lens in resolved.lenses.values():
            for spec in lens.answerers:
                if isinstance(spec, str) and spec.endswith("?"):
                    continue
                if isinstance(spec, str) and not alias_defined(spec):
                    messages.append(
                        Error(
                            f"Lens {lens.name!r} in policy {name!r} names answerer alias {spec!r}, "
                            "which AI_SDK_OVERSIGHT['ANSWERERS'] does not define",
                            id=f"{PREFIX}.E004",
                        )
                    )
    if unanswered:
        messages.append(_unanswered_warning(unanswered))
    for alias, spec in (setting("ANSWERERS") or {}).items():
        try:
            resolve(spec)
        except Exception as exc:
            messages.append(
                Error(f"Answerer alias {alias!r} cannot be built: {exc}", id=f"{PREFIX}.E005")
            )
    messages.extend(_alias_checks())
    default = setting("DEFAULT_POLICY")
    if default and default not in get_policies():
        messages.append(
            Error(
                f"AI_SDK_OVERSIGHT['DEFAULT_POLICY'] names unknown policy {default!r}",
                id=f"{PREFIX}.E006",
            )
        )
    configured = setting("POLICIES") or {}
    unnamed = sorted(str(key) for key in configured if key not in get_policies())
    if unnamed:
        # The console may define them: settings tune those too.
        unnamed = sorted(set(unnamed) - _console_policies())
    if unnamed:
        messages.append(
            Warning(
                f"AI_SDK_OVERSIGHT['POLICIES'] tunes unknown policies: {', '.join(unnamed)}.",
                hint="Each key names a registered policy; these tune nothing.",
                id=f"{PREFIX}.W014",
            )
        )
    messages.extend(_settings_checks())
    if is_enabled():
        messages.extend(_key_errors())
        messages.extend(_runtime_checks(in_use))
    messages.extend(_agent_checks())
    return messages


def check_oversight_deploy(app_configs: Any = None, **kwargs: Any) -> list[Any]:
    """What a deployment should set: keys that encrypt evidence and sign checkpoints."""
    from django_ai_sdk.oversight.conf import is_enabled

    return _key_warnings() if is_enabled() else []


def _policies_in_use() -> set[str]:
    """DEFAULT_POLICY, the policies agent classes name, and every policy those extend.

    What the console assigns is not known here: it lives in the database.
    """
    from django_ai_sdk.oversight.conf import setting
    from django_ai_sdk.oversight.policy import code_ancestors

    names = {str(setting("DEFAULT_POLICY") or "")}
    for cls in _agent_classes():
        try:
            names.add(_named(cls))
        except Exception:
            continue  # E011 reports it.
    names.discard("")
    for name in list(names):
        names.update(code_ancestors(name))
    return names


def _agent_classes() -> list[type]:
    """Registered agent classes, and the runtime bases AgentSettings rows are built on."""
    from django_ai_sdk.agents.registry import registry

    try:
        classes = [type(agent) for agent in registry.all().values()]
    except RuntimeError:
        # Not set up yet (checks run after AppConfig.ready() in a real project).
        classes = []
    return classes + _runtime_bases()


def _runtime_bases() -> list[type]:
    from django_ai_sdk.agents.config import get_runtime_agent_bases

    try:
        return list(get_runtime_agent_bases())
    except Exception:
        return []


def _named(cls: type) -> str:
    """The policy an agent class names, or ""; raises TypeError for what names none."""
    from django_ai_sdk.oversight.registry import policy_name

    chosen = getattr(cls, "oversight", None)
    if chosen is None or chosen is False:
        return ""
    if chosen is True or chosen == "":
        raise TypeError(f"Expected a Policy subclass or a policy name, got {chosen!r}")
    return policy_name(chosen)


def _agent_checks() -> list[Any]:
    """What agent classes say about policies: a name nothing registers, a value that names none."""
    from django_ai_sdk.agents.registry import registry
    from django_ai_sdk.oversight.registry import get_policies

    messages: list[Any] = []
    known = get_policies()
    try:
        agents = registry.all().values()
    except RuntimeError:
        # Not set up yet (checks run after AppConfig.ready() in a real project).
        return messages
    for agent in agents:
        cls = type(agent)
        try:
            own = _named(cls)
        except Exception as exc:
            messages.append(
                Error(
                    f"Agent {cls.__qualname__}.oversight names no policy: {exc}. Its turns "
                    "cannot be watched.",
                    hint="Set a Policy subclass, a registered policy's name, False or None.",
                    id=f"{PREFIX}.E011",
                )
            )
            continue
        if own and own not in known:
            messages.append(
                Warning(
                    f"Agent {cls.__qualname__} names oversight policy {own!r}, which no app "
                    "registers: its turns are refused or run unwatched (FAIL_MODE['gate']) "
                    "unless the console defines it.",
                    id=f"{PREFIX}.W009",
                )
            )
        messages.extend(_adapter_checks(cls, own))
    return messages


def _adapter_checks(cls: type, policy: str) -> list[Any]:
    """An agent whose chat adapter is not the SDK's Stream: its answers are never finished."""
    import typing

    from django_ai_sdk.adapters.base import Stream
    from django_ai_sdk.oversight.conf import setting
    from django_ai_sdk.oversight.policy import build
    from django_ai_sdk.oversight.types import Stage

    policy = policy or str(setting("DEFAULT_POLICY") or "")
    method = getattr(cls, "get_pipeline_adapter", None)
    if not policy or method is None:
        return []
    try:
        returned = typing.get_type_hints(method).get("return")
        finishes = build(policy).has_stage(Stage.FINISH)
    except Exception:
        return []  # Unknown: the annotation names what is not importable, or no policy.
    if not (finishes and isinstance(returned, type)) or issubclass(returned, Stream):
        return []
    return [
        Warning(
            f"Agent {cls.__qualname__} streams chat through {returned.__qualname__}, which is "
            f"not the SDK's Stream: policy {policy!r} judges the question, but never watches "
            "or finishes the answer.",
            hint="Return a Stream (or a subclass) from get_pipeline_adapter().",
            id=f"{PREFIX}.W013",
        )
    ]


def _settings_checks() -> list[Any]:
    """AI_SDK_OVERSIGHT keys nothing reads: a misspelt one would be ignored."""
    from django_ai_sdk.oversight.conf import DEFAULTS
    from django_ai_sdk.utils import resolve_setting

    configured = resolve_setting("AI_SDK_OVERSIGHT", {}) or {}
    unknown = sorted(str(key) for key in configured if key not in DEFAULTS)
    if not unknown:
        return []
    return [
        Warning(
            f"AI_SDK_OVERSIGHT has keys nothing reads: {', '.join(unknown)}.",
            hint=f"The keys are {', '.join(DEFAULTS)}.",
            id=f"{PREFIX}.W010",
        )
    ]


def _console_policies() -> set[str]:
    """Policies the console defined; empty when the database cannot be read (before migrate)."""
    try:
        from django_ai_sdk.oversight.models import PolicyOverride

        return set(
            PolicyOverride.objects.filter(
                target_kind="policy", field="definition", active=True
            ).values_list("policy", flat=True)
        )
    except Exception:
        return set()


def _rejected_settings(name: str, resolved: Any) -> list[Any]:
    """AI_SDK_OVERSIGHT['POLICIES'] entries for `name` that could not apply: they would be ignored."""
    rejected = [entry for entry in resolved.rejected if entry.get("source") == "settings"]
    return [
        Warning(
            f"AI_SDK_OVERSIGHT['POLICIES'][{name!r}] sets {entry.get('target_kind')} "
            f"{entry.get('target') or name!r} {entry.get('field')!r}, which does not apply: "
            f"{entry.get('reason')}",
            id=f"{PREFIX}.W015",
        )
        for entry in rejected
    ]


def _live_checks(name: str, resolved: Any) -> list[Any]:
    """An enforced finish mask or block on the answer, while the answer streams live."""
    if resolved.stream_mode != "live":
        return []
    acting = resolved.changes_answer_at_finish()
    if not acting:
        return []
    return [
        Warning(
            f"Policy {name!r} masks or withholds the answer at finish ({', '.join(acting)}), "
            "but streams it live: the client has the text before finish judges it.",
            hint="Set stream_mode 'auto' or 'hold' on the policy (or AI_SDK_OVERSIGHT['STREAM_MODE']), "
            "or use a client that applies the data-oversight 'replace'.",
            id=f"{PREFIX}.W011",
        )
    ]


def _runtime_checks(in_use: set[str]) -> list[Any]:
    """Runtime agents (AgentSettings rows) that no policy covers until the console assigns one."""
    from django_ai_sdk.oversight.conf import setting
    from django_ai_sdk.utils import resolve_setting

    if setting("DEFAULT_POLICY"):
        return []
    # Only where oversight is set up: a fresh install says nothing.
    if not in_use and not resolve_setting("AI_SDK_OVERSIGHT", {}):
        return []
    bare = []
    for cls in _runtime_bases():
        try:
            if not _named(cls):
                bare.append(cls.__qualname__)
        except Exception:
            continue  # E011 reports it.
    if not bare:
        return []
    return [
        Warning(
            f"Runtime agents built on {', '.join(bare)} run unwatched until the console "
            "assigns them a policy: any user who may create an agent can make one.",
            hint="Set AI_SDK_OVERSIGHT['DEFAULT_POLICY'], or give the runtime base class "
            "(AI_SDK_RUNTIME_AGENT_BASES) an `oversight` policy.",
            id=f"{PREFIX}.W012",
        )
    ]


def _key_errors() -> list[Any]:
    """MASTER_KEY or SIGNING_KEY set but unusable: that fails every turn, or every seal."""
    from django.core.exceptions import ImproperlyConfigured

    from django_ai_sdk.oversight import seal, vault
    from django_ai_sdk.oversight.conf import setting

    messages: list[Any] = []
    try:
        vault.available()
    except ImproperlyConfigured as exc:
        messages.append(Error(str(exc), hint=_key_hint(), id=f"{PREFIX}.E008"))
    if setting("SIGNING_KEY"):
        try:
            seal.private_key(setting("SIGNING_KEY"))
        except ImproperlyConfigured as exc:
            messages.append(Error(str(exc), hint=_key_hint(), id=f"{PREFIX}.E009"))
    return messages


def _key_warnings() -> list[Any]:
    """No MASTER_KEY, no SIGNING_KEY: fine in development, worth a word on deploy."""
    from django.core.exceptions import ImproperlyConfigured

    from django_ai_sdk.oversight import vault
    from django_ai_sdk.oversight.conf import setting

    messages: list[Any] = []
    try:
        encrypted = vault.available()
    except ImproperlyConfigured:
        encrypted = True  # E008 reports it.
    if not encrypted:
        messages.append(
            Warning(
                "Oversight stores evidence in the clear: set AI_SDK_OVERSIGHT['MASTER_KEY'] "
                "and install django-ai-sdk[oversight] to encrypt it and allow crypto-shredding.",
                hint=_key_hint(),
                id=f"{PREFIX}.W001",
            )
        )
    if not setting("SIGNING_KEY"):
        messages.append(
            Warning(
                "Oversight checkpoints are unsigned: set AI_SDK_OVERSIGHT['SIGNING_KEY'].",
                hint=_key_hint(),
                id=f"{PREFIX}.W002",
            )
        )
    return messages


def _key_checks() -> list[Any]:
    """Every key check, errors and deploy warnings alike."""
    return [*_key_errors(), *_key_warnings()]


def _key_hint() -> str:
    from django_ai_sdk.oversight import vault

    return (
        "python manage.py oversight_keys prints new keys."
        if vault.has_cryptography()
        else "pip install 'django-ai-sdk[oversight]'"
    )


def _lens_checks(
    name: str, resolved: Any, unanswered: dict[str, Unanswered] | None = None
) -> list[Any]:
    """Lenses that never run on this deployment, or run only in part.

    A lens no answerer here can answer is skipped on every turn. It goes in
    `unanswered`, which check_oversight reports as one warning for every policy;
    without `unanswered`, this policy's are reported here, in one warning.
    Only settings aliases are known here; the console checks its own aliases live.
    """
    from django_ai_sdk.oversight.stages import lens_health
    from django_ai_sdk.oversight.types import Mode

    messages: list[Any] = []
    own: dict[str, Unanswered] = {} if unanswered is None else unanswered
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
            entry = own.setdefault(lens.name, Unanswered())
            entry.policies.add(name)
            entry.aliases.update(spec.removesuffix("?") for spec in health["skipped"])
            entry.cannot_run |= not health["skipped"]
        if health["unreachable"]:
            messages.append(
                Warning(
                    f"Lens {lens.name!r} in policy {name!r} never asks "
                    f"{', '.join(health['unreachable'])}: a detector before them is already "
                    "sure when it finds nothing.",
                    id=f"{PREFIX}.W004",
                )
            )
        if health["misplaced"]:
            messages.append(
                Warning(
                    f"Lens {lens.name!r} in policy {name!r} reads the question at watch or "
                    "finish, which only judge the answer: it never runs there.",
                    id=f"{PREFIX}.W006",
                )
            )
    if unanswered is None and own:
        messages.append(_unanswered_warning(own))
    for rule in resolved.rules.values():
        blind = sorted(rule.lenses() & dead)
        if rule.mode is not Mode.ENFORCE or not blind:
            continue
        if set(blind) <= optional:
            messages.append(
                Warning(
                    f"Enforced rule {rule.name!r} in policy {name!r} reads {blind}, whose "
                    "answerers are all optional and none is defined here: it never acts.",
                    id=f"{PREFIX}.W008",
                )
            )
        else:
            messages.append(
                Error(
                    f"Enforced rule {rule.name!r} in policy {name!r} reads {blind}, which no "
                    "answerer here can answer.",
                    id=f"{PREFIX}.E007",
                )
            )
    return messages


@dataclass
class Unanswered:
    """A lens no answerer here can answer: where it is, and what it lacks."""

    policies: set[str] = field(default_factory=set)
    # Optional aliases it names that AI_SDK_OVERSIGHT["ANSWERERS"] does not define.
    aliases: set[str] = field(default_factory=set)
    # Its answerers are not optional aliases, but none can be built or run here.
    cannot_run: bool = False


def _unanswered_warning(unanswered: dict[str, Unanswered]) -> Any:
    """One warning naming every lens no answerer here can answer, by the policies it is in."""
    by_policies: dict[tuple[str, ...], list[str]] = {}
    for lens, entry in sorted(unanswered.items()):
        by_policies.setdefault(tuple(sorted(entry.policies)), []).append(lens)
    where = "; ".join(
        f"in {', '.join(policies)}: {', '.join(lenses)}"
        for policies, lenses in sorted(by_policies.items())
    )
    aliases = sorted(set().union(*(entry.aliases for entry in unanswered.values())))
    hints = []
    if aliases:
        hints.append(
            f"Define {', '.join(repr(a) for a in aliases)} in AI_SDK_OVERSIGHT['ANSWERERS'] "
            "to run them."
        )
    if any(entry.cannot_run for entry in unanswered.values()):
        hints.append(
            f"Some name answerers that cannot be built or run here: see {PREFIX}.E004, "
            "E005 and W005."
        )
    hints.append("Console aliases are checked live.")
    count = len(unanswered)
    return Warning(
        f"{count} oversight lens{'es have' if count != 1 else ' has'} no answerer here and "
        f"{'are' if count != 1 else 'is'} skipped on every turn: {where}.",
        hint=" ".join(hints),
        id=f"{PREFIX}.W003",
    )


def _alias_checks() -> list[Any]:
    from django_ai_sdk.oversight.answerers.base import resolve
    from django_ai_sdk.oversight.answerers.llm import LLMJudge
    from django_ai_sdk.oversight.conf import setting
    from django_ai_sdk.pricing import model_price

    messages: list[Any] = []
    for alias in setting("ANSWERERS") or {}:
        try:
            answerer = resolve(alias)
        except Exception:
            continue  # E005 reports it.
        if missing := answerer.missing():
            messages.append(
                Warning(
                    f"Answerer alias {alias!r} is {type(answerer).__name__}, but {missing}: "
                    "its lenses abstain.",
                    id=f"{PREFIX}.W005",
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
                    id=f"{PREFIX}.W007",
                )
            )
    return messages


__all__ = ["check_oversight", "check_oversight_deploy"]
