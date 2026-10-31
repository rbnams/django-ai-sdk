"""Policies: an app's own lenses and rules, layered over the built-in packs.

    @register
    class Support(Policy):
        name = "support"
        extends = ["security", "quality"]
        lenses = [Topic()]
        rules = [Rule("hr_to_people", when=L("topic") == "hr", then=Queue("hr"))]

Every effective value says where it came from, in this order of precedence:
the extended packs, then the policy class, then AI_SDK_OVERSIGHT["POLICIES"],
then the database overlay an operator edits in the console. The overlay can
tune (modes, thresholds, samples, which answerer) but not rewrite what a code
rule tests; that takes a deploy.

The console can also define whole policies, lenses and rules (a "definition"
entry), name answerers (`ANSWERERS` aliases) and choose an agent's policy. The
last two live on the deployment-wide scope `GLOBAL` ("*"), not on a policy.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from django.core.exceptions import ImproperlyConfigured

from django_ai_sdk.logger import get_logger
from django_ai_sdk.oversight.conditions import from_dict, same_shape
from django_ai_sdk.oversight.lenses import Lens, lens_from_config
from django_ai_sdk.oversight.registry import check_rule, get_policy_class
from django_ai_sdk.oversight.rules import Rule, action_from_dict
from django_ai_sdk.oversight.types import Capture, Mode, Stage, Subject

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

logger = get_logger(__name__)

CAPTURE_ORDER = [Capture.NONE, Capture.METADATA, Capture.MASKED, Capture.FULL]

# What the overlay may change, per target.
LENS_FIELDS = {"enabled", "sample", "answerers", "min_confidence", "stages"}
RULE_FIELDS = {"mode", "when", "then", "message", "notice", "stages"}
POLICY_FIELDS = {"capture", "stream_mode", "retention"}
RETENTION_KEYS = ("content", "metadata")
# "live" streams the answer as it is written; "hold" sends it once finish passed it;
# "auto" holds it only when an enforced rule may mask or withhold it at finish.
STREAM_MODES = ("auto", "live", "hold")

# The overlay scope for what is not one policy's: answerer aliases, agent choices.
GLOBAL = "*"


class Policy:
    """A named bundle of lenses and rules. Subclass it in an app's `oversight.py`."""

    name: str = ""
    label: str = ""
    description: str = ""
    # Packs or other policies whose lenses and rules this one starts from. A lens
    # or rule declared here replaces an inherited one of the same name.
    extends: list[str] = []  # noqa: RUF012
    lenses: list[Lens | type[Lens]] = []  # noqa: RUF012
    rules: list[Rule] = []  # noqa: RUF012
    # None inherits; the deployment's CAPTURE is a ceiling either way.
    capture: Capture | None = None
    # "live" or "hold"; None inherits.
    stream_mode: str | None = None
    # Days to keep this policy's turns: {"content": 90, "metadata": 730}, either
    # key alone. Unset keys come from AI_SDK_OVERSIGHT["RETENTION"].
    retention: dict[str, int] = {}  # noqa: RUF012
    # Where this policy's values say they come from on the policy screen.
    origin: str = "code"

    def get_lenses(self) -> list[Lens]:
        """The lenses this class adds. Override to build them from settings or the database."""
        return [lens() if isinstance(lens, type) else lens for lens in self.lenses]

    def get_rules(self) -> list[Rule]:
        return list(self.rules)


def pick[T: (Lens | type[Lens], Rule)](
    items: Iterable[T],
    *names: str,
    mode: Mode | str | None = None,
    notice: str | None = None,
) -> list[T]:
    """The lenses or rules called `names`, in that order, out of a pack's `lenses` or `rules`.

    Start small: take what you need from the packs instead of extending them whole::

        lenses = pick(SecurityPack.lenses, "prompt_injection", "pii_input")
        rules = pick(SecurityPack.rules, "block_injection", "mask_pii_in_question")

    Rules come back as copies, so `mode` and `notice` change the copy and the pack keeps its own.
    """
    by_name = {item.name: item for item in items}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise ImproperlyConfigured(f"Cannot pick {missing}: the choices are {sorted(by_name)}")
    picked = [copy.copy(by_name[name]) for name in names]
    if mode is not None or notice is not None:
        for item in picked:
            if not isinstance(item, Rule):
                raise TypeError("pick(mode=..., notice=...) applies to rules only")
            if mode is not None:
                item.mode = Mode(mode)
            if notice is not None:
                item.notice = notice
    return picked


@dataclass
class ResolvedPolicy:
    """A policy with its packs, settings and overlay applied: what actually runs."""

    name: str
    label: str = ""
    description: str = ""
    lenses: dict[str, Lens] = field(default_factory=dict)
    rules: dict[str, Rule] = field(default_factory=dict)
    capture: Capture = Capture.FULL
    stream_mode: str = "auto"
    retention: dict[str, int] = field(default_factory=dict)
    # "lens:pii.sample" -> "pack:security" | "code" | "settings" | "db"
    sources: dict[str, str] = field(default_factory=dict)
    # Overlay entries that could not apply, with why: shown, never raised.
    rejected: list[dict[str, Any]] = field(default_factory=list)
    # "code", "pack:<name>", or "db" for a policy defined in the console.
    origin: str = "code"
    extends: list[str] = field(default_factory=list)

    def changes_answer_at_finish(self) -> list[str]:
        """The enforced rules that may mask or withhold the answer at finish, by name."""
        finished = {
            lens.name
            for lens in self.active_lenses(Stage.FINISH)
            if lens.subject in {Subject.OUTPUT, Subject.EXCHANGE}
        }
        return sorted(
            rule.name
            for rule in self.rules.values()
            if rule.mode is Mode.ENFORCE
            and any(action.kind in {"mask", "block"} for action in rule.then)
            and rule.lenses() & finished
            and (not rule.stages or Stage.FINISH in rule.stages)
        )

    def holds(self) -> bool:
        """Whether the answer is held until finish passed it: always, never, or when needed."""
        if self.stream_mode == "auto":
            return bool(self.changes_answer_at_finish())
        return self.stream_mode == "hold"

    def active_lenses(self, stage: Stage, subject: Subject | None = None) -> list[Lens]:
        return [
            lens
            for lens in self.lenses.values()
            if lens.enabled
            and stage in lens.stages
            and (subject is None or lens.subject is subject)
        ]

    def active_rules(self) -> list[Rule]:
        return [rule for rule in self.rules.values() if rule.mode is not Mode.OFF]

    def has_stage(self, stage: Stage) -> bool:
        return any(stage in lens.stages for lens in self.lenses.values() if lens.enabled)

    def describe(self) -> dict[str, Any]:
        from django_ai_sdk.oversight.answerers.base import alias_defined, resolve

        def answerer(spec: Any) -> dict[str, Any]:
            optional = isinstance(spec, str) and spec.endswith("?")
            if optional and not alias_defined(spec[:-1]):
                # Skipped, not broken: the pack asks for it only where it exists.
                return {"name": spec[:-1], "optional": True, "configured": False}
            try:
                return {**resolve(spec).describe(), "optional": optional, "configured": True}
            except (ImproperlyConfigured, ImportError) as exc:
                return {"name": str(spec), "error": str(exc), "configured": False}

        return {
            "name": self.name,
            "label": self.label or self.name,
            "description": self.description,
            "origin": self.origin,
            "extends": list(self.extends),
            "capture": str(self.capture),
            "stream_mode": self.stream_mode,
            "holds": self.holds(),
            "retention": dict(self.retention),
            "lenses": [
                {
                    **lens.describe(),
                    "answerers": [answerer(a) for a in lens.answerers],
                    "aliases": [a for a in lens.answerers if isinstance(a, str)],
                    "sources": {
                        k.split(".", 1)[1]: v
                        for k, v in self.sources.items()
                        if k.startswith(f"lens:{lens.name}.")
                    },
                }
                for lens in self.lenses.values()
            ],
            "rules": [
                {
                    **rule.to_dict(),
                    "sources": {
                        k.split(".", 1)[1]: v
                        for k, v in self.sources.items()
                        if k.startswith(f"rule:{rule.name}.")
                    },
                }
                for rule in self.rules.values()
            ],
            "sources": {k: v for k, v in self.sources.items() if k.startswith("policy.")},
            "rejected": self.rejected,
        }


def build(
    name: str,
    *,
    overlay: Iterable[dict[str, Any]] = (),
    ancestors: Mapping[str, Iterable[dict[str, Any]]] | None = None,
) -> ResolvedPolicy:
    """Resolve `name` from code and settings, then apply `overlay` entries from the database.

    `ancestors` holds the overlay entries of the policies `name` extends, by name:
    tuning a parent in the console reaches every policy built on it.
    Raises LookupError for an unknown policy, ImproperlyConfigured for a cycle.
    """
    from django_ai_sdk.oversight.conf import setting
    from django_ai_sdk.oversight.registry import get_policy_class

    overlay = list(overlay)
    ancestors = ancestors or {}
    definition = _definition(overlay)
    if get_policy_class(name) is None and definition is not None:
        resolved = _from_definition(name, definition, ancestors)
    else:
        resolved = _from_code(name, (), ancestors)
    _tune(resolved, name, overlay)
    ceiling = Capture(setting("CAPTURE"))
    if CAPTURE_ORDER.index(resolved.capture) > CAPTURE_ORDER.index(ceiling):
        resolved.capture = ceiling
        resolved.sources["policy.capture"] = "settings"
    return resolved


def _tune(resolved: ResolvedPolicy, name: str, overlay: Iterable[dict[str, Any]]) -> None:
    """Apply what settings and then the database say about policy `name`."""
    from django_ai_sdk.oversight.conf import setting

    configured = setting("POLICIES") or {}
    if name in configured:
        apply_overlay(resolved, _entries_from_settings(configured[name]), source="settings")
    apply_overlay(resolved, overlay, source="db")


def _definition(overlay: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The last policy definition among `overlay`, if the console defined this policy."""
    found = None
    for entry in overlay:
        if entry.get("target_kind") == "policy" and entry.get("field") == "definition":
            found = entry.get("value")
    return found if isinstance(found, dict) else None


def _from_definition(
    name: str, definition: dict[str, Any], ancestors: Mapping[str, Iterable[dict[str, Any]]]
) -> ResolvedPolicy:
    """A policy the console defined: its parents from code, then its own overlay."""
    from django_ai_sdk.oversight.conf import setting

    resolved = ResolvedPolicy(
        name=name,
        label=str(definition.get("label") or name),
        description=str(definition.get("description") or ""),
        capture=Capture(setting("CAPTURE")),
        stream_mode=_default_stream_mode(),
        retention=_retention(setting("RETENTION")),
        origin="db",
        extends=[str(p) for p in definition.get("extends") or []],
    )
    for parent_name in resolved.extends:
        _inherit(resolved, _parent(parent_name, (name,), ancestors), parent_name)
    resolved.sources["policy.definition"] = "db"
    return resolved


def _parent(
    name: str, chain: tuple[str, ...], ancestors: Mapping[str, Iterable[dict[str, Any]]]
) -> ResolvedPolicy:
    """Parent `name` as its children inherit it: with its settings and console tuning."""
    parent = _from_code(name, chain, ancestors)
    _tune(parent, name, ancestors.get(name, ()))
    return parent


def _inherit(resolved: ResolvedPolicy, parent: ResolvedPolicy, parent_name: str) -> None:
    resolved.lenses.update(parent.lenses)
    resolved.rules.update(parent.rules)
    for key, value in parent.sources.items():
        if value == "db":
            value = f"db:{parent_name}"
        elif not value.startswith(("pack:", "db:")) and value != "settings":
            value = f"pack:{parent_name}"
        resolved.sources[key] = value
    # What a parent set itself carries over; what it took from settings already matches.
    if parent.sources.get("policy.capture"):
        resolved.capture = parent.capture
    if parent.sources.get("policy.stream_mode"):
        resolved.stream_mode = parent.stream_mode
    if parent.sources.get("policy.retention"):
        resolved.retention = {**resolved.retention, **parent.retention}


def _from_code(
    name: str, chain: tuple[str, ...], ancestors: Mapping[str, Iterable[dict[str, Any]]]
) -> ResolvedPolicy:
    from django_ai_sdk.oversight.conf import setting
    from django_ai_sdk.oversight.registry import get_policy_class

    if name in chain:
        raise ImproperlyConfigured(f"Policy {name!r} extends itself: {' -> '.join((*chain, name))}")
    cls = get_policy_class(name)
    if cls is None:
        raise LookupError(f"No oversight policy {name!r}")
    policy = cls()
    resolved = ResolvedPolicy(
        name=policy.name,
        label=policy.label,
        description=policy.description,
        capture=Capture(setting("CAPTURE")),
        stream_mode=_default_stream_mode(),
        retention=_retention(setting("RETENTION")),
        origin=cls.origin,
        extends=list(policy.extends),
    )
    for parent_name in policy.extends:
        _inherit(resolved, _parent(parent_name, (*chain, name), ancestors), parent_name)
    origin = cls.origin
    for lens in policy.get_lenses():
        lens = copy.copy(lens)
        # Answerers built in code have no alias to send back from the console: it
        # names them "code:<index>" instead, so a chain can keep them when reordered.
        lens._code_answerers = list(lens.answerers)
        resolved.lenses[lens.name] = lens
        _mark_definition(resolved, "lens", lens.name, origin)
    for rule in policy.get_rules():
        resolved.rules[rule.name] = copy.copy(rule)
        _mark_definition(resolved, "rule", rule.name, origin)
    for rule in resolved.rules.values():
        # Against inherited lenses too, which registering could not see.
        try:
            check_rule(rule, resolved.lenses)
        except ImproperlyConfigured as exc:
            raise ImproperlyConfigured(f"Policy {name!r}: {exc}") from None
    if policy.capture is not None:
        resolved.capture = Capture(policy.capture)
        resolved.sources["policy.capture"] = origin
    if policy.stream_mode is not None:
        try:
            resolved.stream_mode = _stream_mode(policy.stream_mode)
        except ValueError as exc:
            raise ImproperlyConfigured(f"Policy {name!r}: {exc}") from None
        resolved.sources["policy.stream_mode"] = origin
    if policy.retention:
        resolved.retention = {**resolved.retention, **_retention(policy.retention)}
        resolved.sources["policy.retention"] = origin
    return resolved


def _mark_definition(resolved: ResolvedPolicy, kind: str, target: str, source: str) -> None:
    """Every field of a lens or rule comes from where it was defined, until tuned."""
    fields = LENS_FIELDS if kind == "lens" else RULE_FIELDS
    for name in (*fields, "origin"):
        resolved.sources[f"{kind}:{target}.{name}"] = source


def _retention(value: Any) -> dict[str, int]:
    """Days per kind of data, checked: only "content" and "metadata", at least a day each."""
    if not isinstance(value, dict):
        raise ValueError('retention is {"content": days, "metadata": days}')
    unknown = sorted(set(value) - set(RETENTION_KEYS))
    if unknown:
        raise ValueError(f"unknown retention {unknown}; one of {list(RETENTION_KEYS)}")
    days = {key: int(v) for key, v in value.items()}
    if any(d < 1 for d in days.values()):
        raise ValueError("retention keeps data for at least a day")
    return days


def _stream_mode(value: Any) -> str:
    """`value`, checked: a typo such as "held" must not quietly stream live."""
    if value not in STREAM_MODES:
        raise ValueError(f"stream_mode {value!r} is not one of {list(STREAM_MODES)}")
    return str(value)


def _default_stream_mode() -> str:
    from django_ai_sdk.oversight.conf import setting

    try:
        return _stream_mode(setting("STREAM_MODE"))
    except ValueError as exc:
        raise ImproperlyConfigured(f'AI_SDK_OVERSIGHT["STREAM_MODE"]: {exc}') from None


def _entries_from_settings(config: dict[str, Any]) -> list[dict[str, Any]]:
    """{"lenses": {name: {field: value}}, "rules": {...}, "capture": ...} as overlay entries."""
    entries: list[dict[str, Any]] = []
    for kind, plural in (("lens", "lenses"), ("rule", "rules")):
        for target, fields in (config.get(plural) or {}).items():
            for key, value in fields.items():
                entries.append(
                    {"target_kind": kind, "target": target, "field": key, "value": value}
                )
    for key in POLICY_FIELDS:
        if key in config:
            entries.append(
                {"target_kind": "policy", "target": "", "field": key, "value": config[key]}
            )
    return entries


def apply_overlay(
    resolved: ResolvedPolicy, entries: Iterable[dict[str, Any]], *, source: str
) -> None:
    """Apply overlay entries in order. One that cannot apply is recorded, never raised.

    Entries from the database pass `console_entry` first: a row written before a
    kind left the catalog, or by hand, never imports what the console may not.
    """
    for entry in entries:
        try:
            _apply_one(resolved, console_entry(entry) if source == "db" else entry, source)
        except (ImproperlyConfigured, ImportError, ValueError, KeyError, TypeError) as exc:
            resolved.rejected.append({**_jsonable(entry), "source": source, "reason": str(exc)})


def console_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """`entry` with every answerer in it as the console may write it; ValueError otherwise."""
    from django_ai_sdk.oversight.answerers.base import console_config
    from django_ai_sdk.oversight.catalog import INLINE_KINDS, alias_kinds

    kind, key, value = entry["target_kind"], entry["field"], entry.get("value")
    if value is None:
        return entry
    if kind == "answerer":
        value = console_config(value, kinds=alias_kinds())
    elif kind == "lens" and key == "answerers":
        value = [console_config(spec, kinds=INLINE_KINDS) for spec in value]
    elif kind == "lens" and key == "definition":
        if not isinstance(value, dict):
            raise ValueError("a lens definition is an object")
        if "answerers" in value:
            specs = value["answerers"] or []
            value = {**value, "answerers": [console_config(a, kinds=INLINE_KINDS) for a in specs]}
    return {**entry, "value": value}


def code_answerer(lens: Lens, spec: Any) -> Any:
    """`spec`, or the code-built answerer a "code:<index>" reference names."""
    if isinstance(spec, str) and spec.startswith("code:"):
        built = getattr(lens, "_code_answerers", [])
        try:
            return built[int(spec.removeprefix("code:"))]
        except (ValueError, IndexError):
            raise ValueError(f"{lens.name} has no code answerer {spec!r}") from None
    return spec


def answerer_spec(lens: Lens, spec: Any) -> Any:
    """How the console writes one of a lens's answerers: alias, config, or "code:<index>"."""
    if isinstance(spec, str | dict):
        return spec
    for index, built in enumerate(getattr(lens, "_code_answerers", [])):
        if built is spec:
            return f"code:{index}"
    return repr(spec)


def _jsonable(entry: dict[str, Any]) -> dict[str, Any]:
    return {k: entry.get(k) for k in ("target_kind", "target", "field", "value")}


def _apply_one(resolved: ResolvedPolicy, entry: dict[str, Any], source: str) -> None:
    kind = entry["target_kind"]
    apply = {"policy": _apply_policy, "lens": _apply_lens, "rule": _apply_rule}.get(kind)
    if apply is None:
        raise ValueError(f"unknown overlay target {kind!r}")
    apply(resolved, entry.get("target", ""), entry["field"], entry.get("value"), source)


def _apply_policy(resolved: ResolvedPolicy, target: str, key: str, value: Any, source: str) -> None:
    if key == "definition":
        if resolved.origin != "db":
            raise ValueError("a policy declared in code cannot be redefined in the console")
        # Applied before the rest of the overlay, by build().
        return
    if key == "capture":
        resolved.capture = Capture(value)
    elif key == "stream_mode":
        resolved.stream_mode = _stream_mode(value)
    elif key == "retention":
        resolved.retention = {**resolved.retention, **_retention(value)}
    else:
        raise ValueError(f"{key!r} is not a policy setting the overlay may change")
    resolved.sources[f"policy.{key}"] = source


def _apply_lens(resolved: ResolvedPolicy, target: str, key: str, value: Any, source: str) -> None:
    if key == "definition":
        # A lens declared outside code: settings or the console.
        _check_replaces(resolved, "lens", target, source)
        resolved.lenses[target] = lens_from_config({**value, "name": target})
        _mark_definition(resolved, "lens", target, source)
        return
    lens = resolved.lenses.get(target)
    if lens is None:
        raise KeyError(f"no lens {target!r}")
    if key not in LENS_FIELDS:
        raise ValueError(f"{key!r} is not a lens setting the overlay may change")
    lens = copy.copy(lens)
    if key == "stages":
        value = tuple(Stage(s) for s in value)
    elif key in {"sample", "min_confidence"}:
        value = float(value)
        if not 0 <= value <= 1:
            raise ValueError(f"{key} must be between 0 and 1")
    elif key == "answerers":
        from django_ai_sdk.oversight.answerers.base import alias_defined, resolve

        value = [code_answerer(lens, spec) for spec in value]
        for spec in value:
            # "strong?" is fine where no strong model is set up: the lens skips it.
            if isinstance(spec, str) and spec.endswith("?") and not alias_defined(spec[:-1]):
                continue
            resolve(spec)
    elif key == "enabled":
        value = bool(value)
    setattr(lens, key, value)
    resolved.lenses[target] = lens
    resolved.sources[f"lens:{target}.{key}"] = source


def _apply_rule(resolved: ResolvedPolicy, target: str, key: str, value: Any, source: str) -> None:
    if key == "definition":
        _check_replaces(resolved, "rule", target, source)
        rule = Rule.from_dict({**value, "name": target})
        unknown = rule.lenses() - set(resolved.lenses)
        if unknown:
            raise ValueError(f"rule reads unknown lenses {sorted(unknown)}")
        check_rule(rule, resolved.lenses)
        resolved.rules[target] = rule
        _mark_definition(resolved, "rule", target, source)
        return
    rule = resolved.rules.get(target)
    if rule is None:
        raise KeyError(f"no rule {target!r}")
    rule = copy.copy(rule)
    if key == "mode":
        rule.mode = Mode(value)
    elif key == "when":
        code_defined = resolved.sources.get(f"rule:{target}.origin", "code") != "db"
        if code_defined and not same_shape(rule.when.to_dict(), value):
            raise ValueError("the overlay may change a code rule's thresholds, not its test")
        when = from_dict(value)
        unknown = when.lenses() - set(resolved.lenses)
        if unknown:
            raise ValueError(f"rule reads unknown lenses {sorted(unknown)}")
        rule.when = when
    elif key == "then":
        rule.then = [action_from_dict(a) for a in value]
    elif key == "stages":
        rule.stages = tuple(Stage(s) for s in value)
    elif key == "message":
        rule.message = str(value)
    elif key == "notice":
        rule.notice = str(value)
    else:
        raise ValueError(f"{key!r} is not a rule setting the overlay may change")
    check_rule(rule, resolved.lenses)
    resolved.rules[target] = rule
    resolved.sources[f"rule:{target}.{key}"] = source


def _check_replaces(resolved: ResolvedPolicy, kind: str, target: str, source: str) -> None:
    """A definition may replace one from its own layer only: what code or a pack
    declares changes with a deploy, and the overlay tunes it instead."""
    existing = resolved.sources.get(f"{kind}:{target}.origin")
    if existing is not None and existing.split(":", 1)[0] != source:
        raise ValueError(
            f"{kind} {target!r} is already declared ({existing}): tune it, "
            "or give the new one another name"
        )


# name -> (resolved at, ResolvedPolicy or None when it did not load, its ancestors)
_cache: dict[str, tuple[float, ResolvedPolicy | None, frozenset[str]]] = {}
# name -> why it did not load
_broken: dict[str, str] = {}
# name -> the last resolution built with its overlay. `invalidate` keeps it: it is
# what a turn runs when the overlay cannot be read, so one failed read never drops
# what an operator enforced.
_last_good: dict[str, ResolvedPolicy] = {}

# (loaded at, {"answerers": {alias: config}, "agents": {agent id: policy or ""}})
_console: tuple[float, dict[str, dict[str, Any]]] | None = None
# The last console entries read; `invalidate` keeps them, for the same reason.
_console_last_good: dict[str, dict[str, Any]] | None = None


async def aconsole(*, uncached: bool = False) -> dict[str, dict[str, Any]]:
    """The deployment-wide console entries, cached like policies.

    Loading them also hands the aliases to answerer resolution, so a lens that
    names a console alias resolves in sync code too. `uncached` reads them fresh
    and leaves the cache to the turns.
    """
    from django_ai_sdk.oversight.answerers.base import set_console_aliases
    from django_ai_sdk.oversight.conf import setting

    global _console, _console_last_good
    ttl = float(setting("POLICY_CACHE_SECONDS"))
    if not uncached and _console and time.monotonic() - _console[0] < ttl:
        return _console[1]
    loaded: dict[str, dict[str, Any]] = {"answerers": {}, "agents": {}}
    try:
        entries = (await _aload_rows([GLOBAL]))[GLOBAL]
    except Exception:
        # An unreadable console must never take chat down, nor forget which agent has
        # which policy: keep the last good read, uncached, so the next turn tries again.
        logger.opt(exception=True).warning("Oversight console entries could not be loaded")
        if _console:
            return _console[1]
        return _console_last_good if _console_last_good is not None else loaded
    for entry in entries:
        if entry["target_kind"] == "answerer" and entry["field"] == "definition":
            try:
                entry = console_entry(entry)
            except ValueError as exc:
                logger.warning("Console answerer {!r} is not used: {}", entry["target"], exc)
                continue
            loaded["answerers"][entry["target"]] = entry["value"]
        elif entry["target_kind"] == "agent" and entry["field"] == "policy":
            loaded["agents"][entry["target"]] = str(entry["value"] or "")
    set_console_aliases(loaded["answerers"])
    if not uncached:
        _console = (time.monotonic(), loaded)
        _console_last_good = loaded
    return loaded


async def aassigned_policy(agent_id: str) -> str | None:
    """The policy the console chose for an agent: a name, "" for none, None if it did not."""
    if not agent_id:
        return None
    return (await aconsole())["agents"].get(agent_id)


async def aget_policy(name: str, *, uncached: bool = False) -> ResolvedPolicy | None:
    """The effective policy, overlay included, cached for POLICY_CACHE_SECONDS.

    None for an unknown or broken policy: why it broke is logged once per cache
    period and kept for the console (`broken`); the caller decides what a turn does
    then (`TurnSession.start` follows FAIL_MODE["gate"]). When the overlay cannot
    be read, turns run the last good resolution (kept across `invalidate`), else
    the policy as code declares it, else None; none of it is cached, so the next
    turn reads again. `uncached` builds it from the database without reading or
    filling the cache, for console pages, so looking at a policy never flushes what
    turns use; there an unreadable overlay is reported, not papered over.
    """
    from django_ai_sdk.oversight.conf import setting
    from django_ai_sdk.oversight.registry import get_policy_class

    ttl = float(setting("POLICY_CACHE_SECONDS"))
    await aconsole(uncached=uncached)
    cached = None if uncached else _cache.get(name)
    if cached and time.monotonic() - cached[0] < ttl:
        return cached[1]
    try:
        own, ancestors = await load_overlay(name)
    except Exception:
        logger.opt(exception=True).warning("Oversight overlay for {!r} could not be read", name)
        if uncached:
            # Shown in the console; the database's own words stay in the log.
            _broken[name] = "its console changes could not be read"
            return None
        if cached and cached[1] is not None:
            return cached[1]
        if name in _last_good:
            return _last_good[name]
        if get_policy_class(name) is None:
            logger.error(
                "Oversight policy {!r} cannot be loaded: its definition could not be read", name
            )
            return None
        return _build(name)
    resolved = _build(name, overlay=own, ancestors=ancestors)
    if not uncached:
        # A policy that does not build stays broken until code or the overlay changes,
        # so that is cached too: it logs once per period, not once per turn.
        _cache[name] = (time.monotonic(), resolved, frozenset(ancestors))
        if resolved is not None:
            _last_good[name] = resolved
    return resolved


def _build(name: str, **kwargs: Any) -> ResolvedPolicy | None:
    """build(), or None with why kept for the console."""
    try:
        resolved = build(name, **kwargs)
    except Exception as exc:
        if (
            isinstance(exc, LookupError)
            and not kwargs.get("overlay")
            and get_policy_class(name) is None
        ):
            # No such policy (deleted, or a name nothing registers): the caller says 404.
            # A missing parent is different: that policy exists and is broken.
            logger.info("Oversight policy {!r} is unknown", name)
            _broken[name] = str(exc) or type(exc).__name__
            return None
        logger.exception("Oversight policy {!r} cannot be loaded: {}", name, exc)
        _broken[name] = str(exc) or type(exc).__name__
        return None
    _broken.pop(name, None)
    return resolved


def broken(name: str) -> str:
    """Why `name` did not load the last time it was asked for, or ""."""
    return _broken.get(name, "")


async def load_overlay(name: str) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """`name`'s active overlay entries, and those of every policy it extends, by name.

    One query for a policy in code. A console policy's parents are in its own
    definition, so they take a second.
    """
    from django_ai_sdk.oversight.registry import get_policy_class

    if get_policy_class(name) is not None:
        rows = await _aload_rows([name, *code_ancestors(name)])
        return rows.pop(name), rows
    own = (await _aload_rows([name]))[name]
    definition = _definition(own) or {}
    return own, await load_ancestors([str(p) for p in definition.get("extends") or []])


async def load_ancestors(parents: Iterable[str]) -> dict[str, list[dict[str, Any]]]:
    """The overlay entries of `parents` and every policy they extend, by name."""
    names = {a for parent in parents for a in (parent, *code_ancestors(parent))}
    return await _aload_rows(names) if names else {}


def code_ancestors(name: str) -> list[str]:
    """Every policy `name` extends in code, nearest first; a cycle is build()'s to report."""
    from django_ai_sdk.oversight.registry import get_policy_class

    found: list[str] = []
    todo = list(getattr(get_policy_class(name), "extends", []) or [])
    while todo:
        parent = todo.pop(0)
        if parent in found or parent == name:
            continue
        found.append(parent)
        todo.extend(getattr(get_policy_class(parent), "extends", []) or [])
    return found


async def _aload_rows(names: Iterable[str]) -> dict[str, list[dict[str, Any]]]:
    from django_ai_sdk.oversight.models import PolicyOverride

    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
    async for o in PolicyOverride.objects.filter(policy__in=list(rows), active=True).order_by(
        "changed_at"
    ):
        rows[o.policy].append(
            {"target_kind": o.target_kind, "target": o.target, "field": o.field, "value": o.value}
        )
    return rows


def forget_last_good() -> None:
    """Drop the last good resolutions too, which `invalidate` keeps — for tests."""
    global _console_last_good
    _last_good.clear()
    _console_last_good = None


def invalidate(name: str | None = None) -> None:
    """Forget cached resolutions, so the next turn reads the overlay again.

    Forgetting a policy also forgets every policy that extends it. The last good
    resolutions stay: they stand in only while the overlay cannot be read.
    """
    global _console
    if name is None or name == GLOBAL:
        _cache.clear()
        _console = None
        return
    for cached, (_, _, lineage) in list(_cache.items()):
        if cached == name or name in lineage:
            _cache.pop(cached, None)


__all__ = [
    "CAPTURE_ORDER",
    "GLOBAL",
    "Policy",
    "ResolvedPolicy",
    "aassigned_policy",
    "answerer_spec",
    "aconsole",
    "aget_policy",
    "apply_overlay",
    "broken",
    "build",
    "code_ancestors",
    "code_answerer",
    "console_entry",
    "invalidate",
    "load_ancestors",
    "load_overlay",
    "pick",
]
