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
from django_ai_sdk.oversight.rules import Rule, action_from_dict
from django_ai_sdk.oversight.types import Capture, Mode, Stage, Subject

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = get_logger(__name__)

CAPTURE_ORDER = [Capture.NONE, Capture.METADATA, Capture.REDACTED, Capture.FULL]

# What the overlay may change, per target.
LENS_FIELDS = {"enabled", "sample", "answerers", "confident", "stages"}
RULE_FIELDS = {"mode", "when", "then", "message", "stages"}
POLICY_FIELDS = {"capture", "stream_mode", "retention"}

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
    stream_mode: str | None = None
    # Days per kind of content: "security", "quality", "metadata".
    retention: dict[str, int] = {}  # noqa: RUF012
    # Where this policy's values say they come from on the policy screen.
    origin: str = "code"

    def get_lenses(self) -> list[Lens]:
        """The lenses this class adds. Override to build them from settings or the database."""
        return [lens() if isinstance(lens, type) else lens for lens in self.lenses]

    def get_rules(self) -> list[Rule]:
        return list(self.rules)


@dataclass
class ResolvedPolicy:
    """A policy with its packs, settings and overlay applied: what actually runs."""

    name: str
    label: str = ""
    description: str = ""
    lenses: dict[str, Lens] = field(default_factory=dict)
    rules: dict[str, Rule] = field(default_factory=dict)
    capture: Capture = Capture.FULL
    stream_mode: str = "live"
    retention: dict[str, int] = field(default_factory=dict)
    # "lens:pii.sample" -> "pack:security" | "code" | "settings" | "db"
    sources: dict[str, str] = field(default_factory=dict)
    # Overlay entries that could not apply, with why: shown, never raised.
    rejected: list[dict[str, Any]] = field(default_factory=list)
    # "code", "pack:<name>", or "db" for a policy defined in the console.
    origin: str = "code"
    extends: list[str] = field(default_factory=list)

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
            except ImproperlyConfigured as exc:
                return {"name": str(spec), "error": str(exc), "configured": False}

        return {
            "name": self.name,
            "label": self.label or self.name,
            "description": self.description,
            "origin": self.origin,
            "extends": list(self.extends),
            "capture": str(self.capture),
            "stream_mode": self.stream_mode,
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


def build(name: str, *, overlay: Iterable[dict[str, Any]] = ()) -> ResolvedPolicy:
    """Resolve `name` from code and settings, then apply `overlay` entries from the database.

    Raises LookupError for an unknown policy, ImproperlyConfigured for a cycle.
    """
    from django_ai_sdk.oversight.conf import setting
    from django_ai_sdk.oversight.registry import get_policy_class

    overlay = list(overlay)
    definition = _definition(overlay)
    if get_policy_class(name) is None and definition is not None:
        resolved = _from_definition(name, definition)
    else:
        resolved = _from_code(name, ())
    configured = setting("POLICIES") or {}
    if name in configured:
        apply_overlay(resolved, _entries_from_settings(configured[name]), source="settings")
    apply_overlay(resolved, overlay, source="db")
    ceiling = Capture(setting("CAPTURE"))
    if CAPTURE_ORDER.index(resolved.capture) > CAPTURE_ORDER.index(ceiling):
        resolved.capture = ceiling
        resolved.sources["policy.capture"] = "settings"
    return resolved


def _definition(overlay: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The last policy definition among `overlay`, if the console defined this policy."""
    found = None
    for entry in overlay:
        if entry.get("target_kind") == "policy" and entry.get("field") == "definition":
            found = entry.get("value")
    return found if isinstance(found, dict) else None


def _from_definition(name: str, definition: dict[str, Any]) -> ResolvedPolicy:
    """A policy the console defined: its parents from code, then its own overlay."""
    from django_ai_sdk.oversight.conf import setting

    resolved = ResolvedPolicy(
        name=name,
        label=str(definition.get("label") or name),
        description=str(definition.get("description") or ""),
        capture=Capture(setting("CAPTURE")),
        stream_mode=str(setting("STREAM_MODE")),
        retention=dict(setting("RETENTION")),
        origin="db",
        extends=[str(p) for p in definition.get("extends") or []],
    )
    for parent_name in resolved.extends:
        parent = _from_code(parent_name, (name,))
        _inherit(resolved, parent, parent_name)
    resolved.sources["policy.definition"] = "db"
    return resolved


def _inherit(resolved: ResolvedPolicy, parent: ResolvedPolicy, parent_name: str) -> None:
    resolved.lenses.update(parent.lenses)
    resolved.rules.update(parent.rules)
    for key, value in parent.sources.items():
        resolved.sources[key] = value if value.startswith("pack:") else f"pack:{parent_name}"
    if parent.capture is not None and parent.sources.get("policy.capture"):
        resolved.capture = parent.capture


def _from_code(name: str, chain: tuple[str, ...]) -> ResolvedPolicy:
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
        stream_mode=str(setting("STREAM_MODE")),
        retention=dict(setting("RETENTION")),
        origin=cls.origin,
        extends=list(policy.extends),
    )
    for parent_name in policy.extends:
        parent = _from_code(parent_name, (*chain, name))
        _inherit(resolved, parent, parent_name)
    origin = cls.origin
    for lens in policy.get_lenses():
        lens = copy.copy(lens)
        # Answerers built in code have no alias to send back from the console: it
        # names them "code:<index>" instead, so a chain can keep them when reordered.
        lens._code_answerers = list(lens.answerers)
        resolved.lenses[lens.name] = lens
        for key in (*LENS_FIELDS, "origin"):
            resolved.sources[f"lens:{lens.name}.{key}"] = origin
    for rule in policy.get_rules():
        resolved.rules[rule.name] = copy.copy(rule)
        for key in (*RULE_FIELDS, "origin"):
            resolved.sources[f"rule:{rule.name}.{key}"] = origin
    if policy.capture is not None:
        resolved.capture = Capture(policy.capture)
        resolved.sources["policy.capture"] = origin
    if policy.stream_mode is not None:
        resolved.stream_mode = policy.stream_mode
        resolved.sources["policy.stream_mode"] = origin
    if policy.retention:
        resolved.retention = {**resolved.retention, **policy.retention}
        resolved.sources["policy.retention"] = origin
    return resolved


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
    """Apply overlay entries in order. One that cannot apply is recorded, never raised."""
    for entry in entries:
        try:
            _apply_one(resolved, entry, source)
        except (ImproperlyConfigured, ValueError, KeyError, TypeError) as exc:
            resolved.rejected.append({**_jsonable(entry), "source": source, "reason": str(exc)})


def _code_answerer(lens: Lens, spec: Any) -> Any:
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
    kind, target, key, value = (
        entry["target_kind"],
        entry.get("target", ""),
        entry["field"],
        entry.get("value"),
    )
    if kind == "policy":
        if key == "definition":
            if resolved.origin != "db":
                raise ValueError("a policy declared in code cannot be redefined in the console")
            # Applied before the rest of the overlay, by build().
            return
        if key not in POLICY_FIELDS:
            raise ValueError(f"{key!r} is not a policy setting the overlay may change")
        if key == "capture":
            value = Capture(value)
        setattr(resolved, key, value)
        resolved.sources[f"policy.{key}"] = source
        return

    if kind == "lens":
        if key == "definition":
            # A lens declared outside code: settings or the console.
            lens = lens_from_config({**value, "name": target})
            resolved.lenses[target] = lens
            for name in (*LENS_FIELDS, "origin"):
                resolved.sources[f"lens:{target}.{name}"] = source
            return
        lens = resolved.lenses.get(target)
        if lens is None:
            raise KeyError(f"no lens {target!r}")
        if key not in LENS_FIELDS:
            raise ValueError(f"{key!r} is not a lens setting the overlay may change")
        lens = copy.copy(lens)
        if key == "stages":
            value = tuple(Stage(s) for s in value)
        elif key == "sample" or key == "confident":
            value = float(value)
            if not 0 <= value <= 1:
                raise ValueError(f"{key} must be between 0 and 1")
        elif key == "answerers":
            from django_ai_sdk.oversight.answerers.base import alias_defined, resolve

            value = [_code_answerer(lens, spec) for spec in value]
            for spec in value:
                # "guard?" is fine where no guard is set up: the lens skips it.
                if isinstance(spec, str) and spec.endswith("?") and not alias_defined(spec[:-1]):
                    continue
                resolve(spec)
        elif key == "enabled":
            value = bool(value)
        setattr(lens, key, value)
        resolved.lenses[target] = lens
        resolved.sources[f"lens:{target}.{key}"] = source
        return

    if kind == "rule":
        if key == "definition":
            rule = Rule.from_dict({**value, "name": target})
            unknown = rule.lenses() - set(resolved.lenses)
            if unknown:
                raise ValueError(f"rule reads unknown lenses {sorted(unknown)}")
            resolved.rules[target] = rule
            for name in (*RULE_FIELDS, "origin"):
                resolved.sources[f"rule:{target}.{name}"] = source
            return
        rule = resolved.rules.get(target)
        if rule is None:
            raise KeyError(f"no rule {target!r}")
        if key not in RULE_FIELDS:
            raise ValueError(f"{key!r} is not a rule setting the overlay may change")
        rule = copy.copy(rule)
        if key == "mode":
            rule.mode = Mode(value)
        elif key == "when":
            code_defined = resolved.sources.get(f"rule:{target}.origin", "code") != "db"
            if code_defined and not same_shape(rule.when.to_dict(), value):
                raise ValueError("the overlay may change a code rule's thresholds, not its test")
            rule.when = from_dict(value)
        elif key == "then":
            rule.then = [action_from_dict(a) for a in value]
        elif key == "stages":
            rule.stages = tuple(Stage(s) for s in value)
        else:
            rule.message = str(value)
        resolved.rules[target] = rule
        resolved.sources[f"rule:{target}.{key}"] = source
        return

    raise ValueError(f"unknown overlay target {kind!r}")


# name -> (resolved at, ResolvedPolicy)
_cache: dict[str, tuple[float, ResolvedPolicy]] = {}

# (loaded at, {"answerers": {alias: config}, "agents": {agent id: policy or ""}})
_console: tuple[float, dict[str, dict[str, Any]]] | None = None


async def aconsole() -> dict[str, dict[str, Any]]:
    """The deployment-wide console entries, cached like policies.

    Loading them also hands the aliases to answerer resolution, so a lens that
    names a console alias resolves in sync code too.
    """
    from django_ai_sdk.oversight.answerers.base import set_console_aliases
    from django_ai_sdk.oversight.conf import setting

    global _console
    ttl = float(setting("POLICY_CACHE_SECONDS"))
    if _console and time.monotonic() - _console[0] < ttl:
        return _console[1]
    loaded: dict[str, dict[str, Any]] = {"answerers": {}, "agents": {}}
    try:
        entries = await _aload_overlay(GLOBAL)
    except Exception:
        # Like a broken policy, an unreadable console must never take chat down.
        logger.opt(exception=True).warning("Oversight console entries could not be loaded")
        return loaded
    for entry in entries:
        if entry["target_kind"] == "answerer" and entry["field"] == "definition":
            loaded["answerers"][entry["target"]] = entry["value"]
        elif entry["target_kind"] == "agent" and entry["field"] == "policy":
            loaded["agents"][entry["target"]] = str(entry["value"] or "")
    set_console_aliases(loaded["answerers"])
    _console = (time.monotonic(), loaded)
    return loaded


async def aassigned_policy(agent_id: str) -> str | None:
    """The policy the console chose for an agent: a name, "" for none, None if it did not."""
    if not agent_id:
        return None
    return (await aconsole())["agents"].get(agent_id)


async def aget_policy(name: str) -> ResolvedPolicy | None:
    """The effective policy, overlay included, cached for POLICY_CACHE_SECONDS.

    None for an unknown or broken policy: a typo must never take chat down.
    """
    from django_ai_sdk.oversight.conf import setting

    ttl = float(setting("POLICY_CACHE_SECONDS"))
    await aconsole()
    cached = _cache.get(name)
    if cached and time.monotonic() - cached[0] < ttl:
        return cached[1]
    try:
        overlay = await _aload_overlay(name)
        resolved = build(name, overlay=overlay)
    except LookupError:
        logger.warning("Agent names oversight policy {!r}, which is not registered", name)
        return None
    except ImproperlyConfigured:
        logger.exception("Oversight policy {!r} cannot be resolved", name)
        return None
    _cache[name] = (time.monotonic(), resolved)
    return resolved


async def _aload_overlay(name: str) -> list[dict[str, Any]]:
    from django_ai_sdk.oversight.models import PolicyOverride

    return [
        {"target_kind": o.target_kind, "target": o.target, "field": o.field, "value": o.value}
        async for o in PolicyOverride.objects.filter(policy=name, active=True).order_by(
            "changed_at"
        )
    ]


def invalidate(name: str | None = None) -> None:
    """Forget cached resolutions, so the next turn reads the overlay again."""
    global _console
    if name is None or name == GLOBAL:
        _cache.clear()
        _console = None
    else:
        _cache.pop(name, None)


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
    "build",
    "invalidate",
]
