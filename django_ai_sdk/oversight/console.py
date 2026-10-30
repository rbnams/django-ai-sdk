"""The console's half of OversightService: build and tune policies, agents and answerers.

Every change is an overlay row (PolicyOverride) chained onto the ledger, so the
console's history is as tamper-evident as the turns it governs.
"""

from __future__ import annotations

import re
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from asgiref.sync import sync_to_async
from django.utils import timezone

from django_ai_sdk.errors import UserError
from django_ai_sdk.logger import get_logger
from django_ai_sdk.oversight import budget
from django_ai_sdk.oversight.privacy import staff_label
from django_ai_sdk.oversight.schemas import (
    MAX_LIMIT,
    AgentPolicyIn,
    AgentRow,
    AnswererIn,
    OverrideIn,
    OverrideOut,
    PolicyIn,
    SimulateIn,
    TryAnswererIn,
    TryLensIn,
)
from django_ai_sdk.oversight.types import Answer
from django_ai_sdk.permissions import Operation

logger = get_logger(__name__)

if TYPE_CHECKING:
    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser

    UserType = AbstractBaseUser | AnonymousUser | None


class ConsoleMixin:
    """Policy, agent and answerer editing; mixed into OversightService, whose checks it uses."""

    if TYPE_CHECKING:
        # Provided by OversightService.

        @classmethod
        async def _check(cls, user: UserType, operation: Operation, **kwargs: Any) -> None: ...

        @classmethod
        async def get_policy(cls, name: str, *, user: UserType) -> dict[str, Any] | None: ...

    # Policies.

    @classmethod
    async def set_override(
        cls, name: str, change: OverrideIn, *, user: UserType
    ) -> dict[str, Any] | None:
        """Tune one field of a policy from the console; rejected changes raise ValueError."""
        from django_ai_sdk.oversight.policy import build, invalidate, load_overlay
        from django_ai_sdk.oversight.registry import get_policy_class

        await cls._check(user, Operation.MANAGE_OVERSIGHT)
        if get_policy_class(name) is None and name not in await _console_policies():
            return None
        if change.target_kind not in {"lens", "rule", "policy"}:
            raise UserError("Only lenses, rules and the policy itself can be changed here.")
        entry = {
            "target_kind": change.target_kind,
            "target": change.target,
            "field": change.field,
            "value": change.value,
        }
        if change.field == "definition" and change.target_kind in {"lens", "rule"}:
            _check_slug(change.target, change.target_kind)
        if not change.reset:
            entry = _console_entry(entry)
            change = change.model_copy(update={"value": entry["value"]})
            own, ancestors = await load_overlay(name)
            trial = build(name, overlay=[*own, entry], ancestors=ancestors)
            rejected = [
                r
                for r in trial.rejected
                if r.get("source") == "db" and {k: r.get(k) for k in entry} == entry
            ]
            if rejected:
                raise UserError(rejected[-1]["reason"])
            if change.target_kind == "lens" and change.field in {"definition", "stages", "subject"}:
                _refuse_misplaced(trial.lenses.get(change.target))
        await sync_to_async(_write_override, thread_sensitive=True)(name, change, user)
        invalidate(name)
        return await cls.get_policy(name, user=user)

    @classmethod
    async def create_policy(cls, payload: PolicyIn, *, user: UserType) -> dict[str, Any]:
        """A new policy, defined in the console: blank, or from one of the examples."""
        from django_ai_sdk.oversight.catalog import lens_example, rule_example
        from django_ai_sdk.oversight.examples import POLICY_EXAMPLES
        from django_ai_sdk.oversight.policy import GLOBAL, build, invalidate, load_ancestors
        from django_ai_sdk.oversight.registry import get_policy_class

        await cls._check(user, Operation.MANAGE_OVERSIGHT)
        name = payload.name.strip()
        _check_slug(name, "policy")
        if name == GLOBAL or get_policy_class(name) or name in await _console_policies():
            raise UserError(f"A policy called {name!r} exists already.")
        example: dict[str, Any] = {}
        if payload.example:
            example = next((e for e in POLICY_EXAMPLES if e["name"] == payload.example), {})
            if not example:
                raise UserError(f"No example policy {payload.example!r}.")
        extends = payload.extends or list(example.get("extends") or [])
        for parent in extends:
            if get_policy_class(parent) is None:
                raise UserError(
                    f"{parent!r} is not a policy in code or a pack; it cannot be extended."
                )
        entries = [
            {
                "target_kind": "policy",
                "target": "",
                "field": "definition",
                "value": {
                    "label": payload.label or example.get("label") or name,
                    "description": payload.description or example.get("description", ""),
                    "extends": extends,
                },
            }
        ]
        for key in ("capture", "stream_mode"):
            if example.get(key):
                entries.append(
                    {"target_kind": "policy", "target": "", "field": key, "value": example[key]}
                )
        for lens_name in example.get("lenses") or []:
            entries.append(
                {
                    "target_kind": "lens",
                    "target": lens_name,
                    "field": "definition",
                    "value": lens_example(lens_name),
                }
            )
        for rule_name in example.get("rules") or []:
            entries.append(
                {
                    "target_kind": "rule",
                    "target": rule_name,
                    "field": "definition",
                    "value": rule_example(rule_name),
                }
            )
        entries = [_console_entry(entry) for entry in entries]
        trial = build(name, overlay=entries, ancestors=await load_ancestors(extends))
        if trial.rejected:
            raise UserError(trial.rejected[0]["reason"])
        for lens_name in example.get("lenses") or []:
            _refuse_misplaced(trial.lenses.get(lens_name))
        note = payload.note or (f"From the {example['label']} example" if example else "Created")
        await sync_to_async(_write_entries, thread_sensitive=True)(name, entries, user, note)
        invalidate(name)
        described = await cls.get_policy(name, user=user)
        assert described is not None
        return described

    @classmethod
    async def delete_policy(cls, name: str, *, user: UserType, note: str = "") -> bool:
        """Withdraw a console policy; agents that used it fall back to their code's choice."""
        from django_ai_sdk.oversight.policy import GLOBAL, invalidate
        from django_ai_sdk.oversight.registry import get_policy_class

        await cls._check(user, Operation.MANAGE_OVERSIGHT)
        if get_policy_class(name) is not None:
            raise UserError("A policy declared in code can only be removed in code.")
        if name not in await _console_policies():
            return False
        await sync_to_async(_withdraw_policy, thread_sensitive=True)(name, user, note)
        invalidate(GLOBAL)
        return True

    # Agents.

    @classmethod
    async def _agent_rows(cls) -> list[AgentRow]:
        from django.db.models import Count, Max

        from django_ai_sdk.oversight.models import Turn
        from django_ai_sdk.oversight.policy import aconsole

        console = (await aconsole(uncached=True))["agents"]
        rows = await _agents()
        recent = Turn.objects.filter(created_at__gte=timezone.now() - timedelta(days=30))
        stats = {
            row["agent_id"]: row
            async for row in recent.values("agent_id").annotate(
                n=Count("id"), last=Max("created_at")
            )
        }
        channels: dict[str, set[str]] = {}
        async for row in recent.values("agent_id", "channel").distinct():
            channels.setdefault(row["agent_id"], set()).add(row["channel"])
        out = []
        for row in rows:
            assigned = console.get(row["id"])
            if assigned is not None:
                effective, source = assigned, "console"
            else:
                effective, source = row["code_policy"], row["code_reason"]
            stat = stats.get(row["id"]) or {}
            out.append(
                AgentRow(
                    **row,
                    assigned=assigned,
                    effective=effective,
                    source=source,
                    turns=stat.get("n", 0),
                    last_turn=stat.get("last"),
                    channels=sorted(channels.get(row["id"], ())),
                )
            )
        return out

    @classmethod
    async def list_agents(cls, *, user: UserType) -> list[AgentRow]:
        """Every agent, the policy that watches it, and whether the console chose that."""
        await cls._check(user, Operation.VIEW_OVERSIGHT)
        return await cls._agent_rows()

    @classmethod
    async def set_agent_policy(
        cls, agent_id: str, payload: AgentPolicyIn, *, user: UserType
    ) -> AgentRow | None:
        """Choose an agent's policy in the console, switch it off, or hand it back to code."""
        from django_ai_sdk.oversight.policy import GLOBAL, aget_policy, invalidate

        await cls._check(user, Operation.MANAGE_OVERSIGHT)
        rows = {row.id: row for row in await cls._agent_rows()}
        if agent_id not in rows:
            return None
        if not payload.reset and payload.policy and await aget_policy(payload.policy) is None:
            raise UserError(f"No policy {payload.policy!r}.")
        change = OverrideIn(
            target_kind="agent",
            target=agent_id,
            field="policy",
            value=None if payload.reset else payload.policy,
            reset=payload.reset,
            note=payload.note
            or f"{rows[agent_id].name}: "
            + ("back to its code" if payload.reset else payload.policy or "oversight off"),
        )
        await sync_to_async(_write_override, thread_sensitive=True)(GLOBAL, change, user)
        invalidate(GLOBAL)
        return next(row for row in await cls._agent_rows() if row.id == agent_id)

    # Answerers.

    @classmethod
    async def catalog(cls, *, user: UserType) -> dict[str, Any]:
        """What the editor builds with: kinds, aliases, models and examples."""
        from django_ai_sdk.oversight.catalog import catalog
        from django_ai_sdk.oversight.policy import aconsole

        await cls._check(user, Operation.VIEW_OVERSIGHT)
        await aconsole()
        data = await sync_to_async(catalog, thread_sensitive=False)()
        data["agents"] = [{"id": row.id, "name": row.name} for row in await cls._agent_rows()]
        data["usage"] = await cls._alias_usage()
        return data

    @classmethod
    async def _alias_usage(cls) -> dict[str, list[dict[str, str]]]:
        """alias -> the lenses that ask it, by policy."""
        from django_ai_sdk.oversight.policy import aget_policy
        from django_ai_sdk.oversight.registry import get_policies

        usage: dict[str, list[dict[str, str]]] = {}
        for name in sorted({*get_policies(), *await _console_policies()}):
            resolved = await aget_policy(name)
            if resolved is None or resolved.origin.startswith("pack:"):
                continue
            for lens in resolved.lenses.values():
                for spec in lens.answerers:
                    if isinstance(spec, str):
                        usage.setdefault(spec.removesuffix("?"), []).append(
                            {"policy": name, "lens": lens.name, "enabled": str(lens.enabled)}
                        )
        return usage

    @classmethod
    async def set_answerer(
        cls, alias: str, payload: AnswererIn, *, user: UserType
    ) -> dict[str, Any]:
        """Define a named answerer, or withdraw the console's definition of one."""
        from django_ai_sdk.oversight.answerers.base import alias_table, build_alias
        from django_ai_sdk.oversight.catalog import ALIAS_KINDS
        from django_ai_sdk.oversight.policy import GLOBAL, aconsole, invalidate

        await cls._check(user, Operation.MANAGE_OVERSIGHT)
        _check_slug(alias, "answerer")
        if alias == "human":
            raise UserError("'human' is built in.")
        if not payload.reset:
            if not payload.config:
                raise UserError("Give the answerer a configuration.")
            config = _console_config(payload.config, ALIAS_KINDS)
            await aconsole(uncached=True)
            try:
                build_alias(alias, {**alias_table(), alias: config})
            except Exception as exc:
                raise UserError(f"That answerer cannot be built: {exc}") from exc
        change = OverrideIn(
            target_kind="answerer",
            target=alias,
            field="definition",
            value=None if payload.reset else config,
            reset=payload.reset,
            note=payload.note,
        )
        await sync_to_async(_write_override, thread_sensitive=True)(GLOBAL, change, user)
        invalidate(GLOBAL)
        return await cls.catalog(user=user)

    @classmethod
    async def try_answerer(cls, payload: TryAnswererIn, *, user: UserType) -> dict[str, Any]:
        """Ask a draft answerer one yes/no question: does it work, and how fast."""
        from django_ai_sdk.oversight.answerers.base import ask, resolve
        from django_ai_sdk.oversight.catalog import ALIAS_KINDS
        from django_ai_sdk.oversight.lenses import Flag
        from django_ai_sdk.oversight.simulate import answer_out
        from django_ai_sdk.oversight.types import TurnView

        await cls._check(user, Operation.MANAGE_OVERSIGHT)
        config = _console_config(payload.config, ALIAS_KINDS)
        try:
            answerer = resolve(config)
        except Exception as exc:
            raise UserError(f"That answerer cannot be built: {exc}") from exc
        lens = Flag(name="console_test", question=payload.question, answerers=[answerer])
        async with budget.charged():
            answer = await ask(answerer, lens, TurnView(input=payload.input))
        if answer is None:
            answer = Answer(answerer=answerer.label(), error="abstained", confidence=0.0)
        return answer_out(answer)

    # Dry runs.

    @classmethod
    async def simulate(
        cls, name: str, payload: SimulateIn, *, user: UserType
    ) -> dict[str, Any] | None:
        """What the policy would do with one exchange. Nothing is recorded."""
        from django_ai_sdk.oversight.simulate import simulate

        await cls._check(user, Operation.MANAGE_OVERSIGHT)
        tool = (
            {"name": payload.tool_name, "arguments": payload.tool_arguments}
            if payload.tool_name
            else None
        )
        return await simulate(
            name, input=payload.input, output=payload.output, tool_call=tool, user=user
        )

    @classmethod
    async def try_lens(
        cls, name: str, payload: TryLensIn, *, user: UserType
    ) -> dict[str, Any] | None:
        """Ask one lens, saved or a draft, with nothing recorded."""
        from django_ai_sdk.oversight.catalog import INLINE_KINDS
        from django_ai_sdk.oversight.lenses import lens_from_config
        from django_ai_sdk.oversight.policy import aget_policy, code_answerer
        from django_ai_sdk.oversight.simulate import try_lens

        await cls._check(user, Operation.MANAGE_OVERSIGHT)
        resolved = await aget_policy(name)
        if resolved is None:
            return None
        if payload.definition:
            config = dict(payload.definition)
            if "answerers" in config:
                config["answerers"] = [
                    _console_config(spec, INLINE_KINDS) for spec in config["answerers"] or []
                ]
            config.setdefault("name", payload.lens or "draft")
            # A draft of a code lens may keep its code-built answerers ("code:0").
            saved = resolved.lenses.get(payload.lens)
            if saved is not None:
                config["answerers"] = [
                    code_answerer(saved, spec) for spec in config.get("answerers") or []
                ]
            try:
                lens = lens_from_config(config)
            except Exception as exc:
                raise UserError(f"That lens is not complete: {exc}") from exc
        else:
            lens = resolved.lenses.get(payload.lens)
            if lens is None:
                return None
        async with budget.charged():
            return await try_lens(lens, input=payload.input, output=payload.output)

    @classmethod
    async def changes(
        cls, *, user: UserType, policy: str = "", limit: int = 100
    ) -> list[OverrideOut]:
        """The console's change trail: every edit, newest first, with who and why."""
        from django_ai_sdk.oversight.models import PolicyOverride

        await cls._check(user, Operation.VIEW_OVERSIGHT)
        rows = PolicyOverride.objects.all()
        if policy:
            rows = rows.filter(policy=policy)
        limit = max(1, min(limit, MAX_LIMIT))
        return [_override_out(o) async for o in rows.order_by("-changed_at")[:limit]]


def _console_config(config: Any, kinds: list[dict[str, Any]]) -> Any:
    from django_ai_sdk.oversight.answerers.base import console_config

    try:
        return console_config(config, kinds=kinds)
    except ValueError as exc:
        raise UserError(str(exc)) from exc


def _refuse_misplaced(lens: Any) -> None:
    from django_ai_sdk.oversight.stages import misplaced

    if lens is not None and misplaced(lens):
        raise UserError(
            f"Lens {lens.name!r} reads the question, so it can run before the model or in "
            "the background, not while the answer is written or finished."
        )


def _console_entry(entry: dict[str, Any]) -> dict[str, Any]:
    from django_ai_sdk.oversight.policy import console_entry

    try:
        return console_entry(entry)
    except ValueError as exc:
        raise UserError(str(exc)) from exc


SLUG = re.compile(r"[a-z][a-z0-9_-]{1,62}")


def _check_slug(value: str, what: str) -> None:
    # fullmatch: "$" would let a trailing newline through.
    if not SLUG.fullmatch(value or ""):
        raise UserError(
            f"A {what} name is lowercase letters, digits, - and _, starting with a letter."
        )


async def _console_policies() -> set[str]:
    """Names of the policies the console defined and has not withdrawn."""
    from django_ai_sdk.oversight.models import PolicyOverride

    return {
        name
        async for name in PolicyOverride.objects.filter(
            target_kind="policy", field="definition", active=True
        ).values_list("policy", flat=True)
    }


async def _agents() -> list[dict[str, Any]]:
    """Registered agent classes and runtime agents, with what their code asks for.

    The code's choice is what `get_oversight()` returns outside any turn (no
    channel, no user). One that raises shows as reason "error" instead of failing
    the listing.
    """
    from django_ai_sdk.oversight.session import acode_policy

    found = await sync_to_async(_agent_instances, thread_sensitive=True)()
    rows: list[dict[str, Any]] = []
    for row, agent in found:
        try:
            policy, reason = await acode_policy(agent)
        except Exception:
            logger.exception("Oversight could not ask {} for its policy", row["name"])
            policy, reason = "", "error"
        rows.append({**row, "code_policy": policy, "code_reason": reason})
    return rows


def _agent_instances() -> list[tuple[dict[str, Any], Any]]:
    """Registered agent classes and runtime agents: each row, and the agent to ask."""
    from django_ai_sdk.agents.config import get_runtime_agent_class
    from django_ai_sdk.agents.models import AgentSettings
    from django_ai_sdk.agents.registry import registry
    from django_ai_sdk.oversight.session import agent_key

    found: list[tuple[dict[str, Any], Any]] = []
    for agent in registry.all().values():
        row = {
            "id": agent_key(agent),
            "name": getattr(agent, "name", "") or type(agent).__name__,
            "kind": "code",
            "description": str(getattr(agent, "description", "") or "").strip(),
            "model": str(getattr(agent, "model", "") or ""),
            "integrations": [str(i) for i in getattr(agent, "integrations", []) or []],
        }
        found.append((row, agent))
    for config in AgentSettings.objects.filter(active=True).order_by("name"):
        # The class the runtime builds for this row: only a listed base, else the first.
        # Without a console assignment of its own, it runs under that class's policy.
        cls = get_runtime_agent_class(config.agent)
        row = {
            "id": str(config.id),
            "name": config.name,
            "kind": "runtime",
            "slug": config.slug,
            "model": config.model,
            "integrations": [str(i) for i in config.integrations or []],
        }
        found.append((row, _runtime_probe(cls, config)))
    return found


def _runtime_probe(cls: Any, config: Any) -> Any:
    """The agent a row builds, so a per-instance get_oversight() is asked; else its class."""
    try:
        return cls(config)
    except Exception:
        logger.opt(exception=True).debug(
            "Could not build runtime agent {} to read its policy", config.slug
        )
        return type("Probe", (), {"oversight": getattr(cls, "oversight", None)})()


def _override_out(o: Any) -> OverrideOut:
    return OverrideOut(
        id=str(o.id),
        policy=o.policy,
        target_kind=o.target_kind,
        target=o.target,
        field=o.field,
        value=o.value,
        active=o.active,
        changed_by=o.changed_by_label,
        note=o.note,
        changed_at=o.changed_at,
    )


def _write_entries(name: str, entries: list[dict[str, Any]], user: Any, note: str) -> None:
    _write_changes([(name, OverrideIn(**entry, note=note)) for entry in entries], user)


def _withdraw_policy(name: str, user: Any, note: str) -> None:
    from django.db import transaction

    from django_ai_sdk.oversight.models import PolicyOverride
    from django_ai_sdk.oversight.policy import GLOBAL

    with transaction.atomic():
        changes = [
            (
                name,
                OverrideIn(
                    target_kind=override.target_kind,
                    target=override.target,
                    field=override.field,
                    reset=True,
                    note=note or "Policy removed",
                ),
            )
            for override in PolicyOverride.objects.filter(policy=name, active=True)
        ]
        changes += [
            (
                GLOBAL,
                OverrideIn(
                    target_kind="agent",
                    target=override.target,
                    field="policy",
                    reset=True,
                    note=f"Policy {name} removed",
                ),
            )
            for override in PolicyOverride.objects.filter(
                policy=GLOBAL, target_kind="agent", field="policy", value=name, active=True
            )
        ]
        _write_changes(changes, user)


def _write_override(name: str, change: OverrideIn, user: Any) -> None:
    _write_changes([(name, change)], user)


def _write_changes(changes: list[tuple[str, OverrideIn]], user: Any) -> None:
    """Write overlay rows in order, each withdrawing the rows it replaces; chain them at once."""
    from django.db import transaction

    from django_ai_sdk.oversight import ledger
    from django_ai_sdk.oversight.models import PolicyOverride

    reader = user if getattr(user, "pk", None) is not None else None
    with transaction.atomic():
        rows = []
        for name, change in changes:
            replaced = PolicyOverride.objects.filter(
                policy=name, target_kind=change.target_kind, target=change.target, active=True
            )
            if change.field != "definition" or change.target_kind in {
                "policy",
                "answerer",
                "agent",
            }:
                replaced = replaced.filter(field=change.field)
            # A lens or rule (re)defined in the console starts clean: its earlier tuning
            # was of the old definition.
            replaced.update(active=False)
            rows.append(
                PolicyOverride.objects.create(
                    policy=name,
                    target_kind=change.target_kind,
                    target=change.target,
                    field=change.field,
                    value=None if change.reset else change.value,
                    active=not change.reset,
                    changed_by=reader,
                    changed_by_label=staff_label(user),
                    note=change.note,
                )
            )
        ledger.append_many([("override", row) for row in rows])


__all__ = ["ConsoleMixin"]
