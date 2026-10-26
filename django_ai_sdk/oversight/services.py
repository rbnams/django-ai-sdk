"""Service-layer facade for oversight: the seam an HTTP layer sits on.

Every read through here is itself written to the ledger as an AccessRecord, with
the purpose it was made for. Content is shown through that purpose's view only.
Unknown ids return None, so a router answers 404 without catching; a refused
operation raises PermissionDenied.
"""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Any

from asgiref.sync import sync_to_async
from django.core.exceptions import ValidationError
from django.utils import timezone

from django_ai_sdk.errors import UserError
from django_ai_sdk.logger import get_logger
from django_ai_sdk.oversight.conf import setting
from django_ai_sdk.oversight.privacy import (
    SEES_ACTOR,
    lens_visible,
    mask_evidence,
    pseudonym,
    staff_label,
)
from django_ai_sdk.oversight.schemas import (
    AccessOut,
    AgentPolicyIn,
    AgentRow,
    AnswererIn,
    ChainProof,
    CheckpointOut,
    DecisionOut,
    ObservationOut,
    OverrideIn,
    OverrideOut,
    Overview,
    PolicyIn,
    PolicySummary,
    RevealOut,
    ReviewOut,
    RuleHit,
    SimulateIn,
    TryAnswererIn,
    TryLensIn,
    TurnDetail,
    TurnPage,
    TurnRow,
    VerdictIn,
    VerifyOut,
)
from django_ai_sdk.oversight.types import Answer, Purpose, Stage
from django_ai_sdk.permissions import (
    Operation,
    PermissionDenied,
    PermissionDomain,
    PermissionsMixin,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser

    from django_ai_sdk.oversight.models import Turn
    from django_ai_sdk.oversight.policy import ResolvedPolicy

    UserType = AbstractBaseUser | AnonymousUser | None

logger = get_logger(__name__)

# Evidence keys without text in them: every purpose may see these.
META_KEYS = frozenset({"agent", "input_chars", "output_chars"})
PREVIEW_CHARS = 160


def _purpose(value: Purpose | str) -> Purpose:
    try:
        return Purpose(value)
    except ValueError as exc:
        raise UserError(f"Unknown purpose {value!r}; one of {[str(p) for p in Purpose]}") from exc


class OversightService(PermissionsMixin):
    """Read the ledger, answer its queues, tune its policies."""

    domain = PermissionDomain.OVERSIGHT

    @classmethod
    async def _check(cls, user: UserType, operation: Operation, **kwargs: Any) -> None:
        if not await cls.has_perms(user, operation, raise_on_deny=False, **kwargs):
            raise PermissionDenied(f"Not permitted to {operation}")

    @classmethod
    async def _check_purpose(cls, user: UserType, purpose: Purpose) -> None:
        operation = (
            Operation.AUDIT_OVERSIGHT if purpose is Purpose.AUDIT else Operation.VIEW_OVERSIGHT
        )
        await cls._check(user, operation, purpose=str(purpose))

    # The feed.

    @classmethod
    async def list_turns(
        cls,
        *,
        user: UserType,
        purpose: Purpose | str = Purpose.SECURITY,
        outcome: str = "",
        shadow_outcome: str = "",
        agent_id: str = "",
        channel: str = "",
        policy: str = "",
        severity: str = "",
        label: str = "",
        value: str = "",
        rule: str = "",
        review: str = "",
        thread_id: str = "",
        since: Any = None,
        until: Any = None,
        limit: int = 50,
        offset: int = 0,
    ) -> TurnPage:
        """Turns newest first, filtered; what each row shows depends on `purpose`."""
        from django_ai_sdk.oversight.models import Turn

        purpose = _purpose(purpose)
        await cls._check_purpose(user, purpose)
        qs = Turn.objects.all()
        if outcome:
            qs = qs.filter(outcome=outcome)
        if shadow_outcome:
            # "Would have been X" means it was not X in fact.
            qs = qs.filter(shadow_outcome=shadow_outcome).exclude(outcome=shadow_outcome)
        if agent_id:
            qs = qs.filter(agent_id=agent_id)
        if channel:
            qs = qs.filter(channel__startswith=channel)
        if policy:
            qs = qs.filter(policy=policy)
        if severity:
            qs = qs.filter(severity=severity)
        if thread_id:
            qs = qs.filter(thread_id=thread_id)
        if label:
            qs = qs.filter(**{f"labels__{label}__isnull": False})
            if value != "":
                qs = qs.filter(**{f"labels__{label}": _typed(value)})
        if rule:
            qs = qs.filter(decisions__rule=rule).distinct()
        if review == "open":
            qs = qs.filter(reviews__status="open").distinct()
        elif review == "reviewed":
            qs = qs.exclude(observations__reviewer="").filter(observations__isnull=False).distinct()
        if since:
            qs = qs.filter(created_at__gte=since)
        if until:
            qs = qs.filter(created_at__lt=until)
        total = await qs.acount()
        page = [
            t
            async for t in qs.select_related("evidence").order_by("-created_at")[
                offset : offset + limit
            ]
        ]
        rows = await cls._rows(page, purpose)
        await cls._access(
            user,
            purpose,
            "list",
            query={
                k: v
                for k, v in {
                    "outcome": outcome,
                    "shadow_outcome": shadow_outcome,
                    "agent_id": agent_id,
                    "channel": channel,
                    "policy": policy,
                    "severity": severity,
                    "label": label,
                    "value": value,
                    "rule": rule,
                    "review": review,
                    "thread_id": thread_id,
                    "offset": offset,
                }.items()
                if v
            },
        )
        return TurnPage(items=rows, total=total, limit=limit, offset=offset)

    @classmethod
    async def _rows(cls, turns: list[Turn], purpose: Purpose) -> list[TurnRow]:
        from django_ai_sdk.oversight.models import Decision, ReviewItem

        ids = [t.id for t in turns]
        hits: dict[Any, list[RuleHit]] = {}
        async for d in Decision.objects.filter(turn_id__in=ids).order_by("created_at"):
            hits.setdefault(d.turn_id, []).append(
                RuleHit(
                    rule=d.rule,
                    stage=d.stage,
                    mode=d.mode,
                    acted=d.acted,
                    actions=[a.get("kind", "") for a in d.actions],
                )
            )
        open_reviews: dict[Any, int] = {}
        async for turn_id in ReviewItem.objects.filter(turn_id__in=ids, status="open").values_list(
            "turn_id", flat=True
        ):
            open_reviews[turn_id] = open_reviews.get(turn_id, 0) + 1
        contents = await sync_to_async(_contents)(turns) if purpose is not Purpose.AUDIT else {}
        return [
            cls._row(t, purpose, contents.get(t.id), hits.get(t.id, []), open_reviews.get(t.id, 0))
            for t in turns
        ]

    @staticmethod
    def _row(
        turn: Turn,
        purpose: Purpose,
        content: dict[str, Any] | None,
        hits: list[RuleHit],
        reviews_open: int,
        *,
        revealed: str = "",
    ) -> dict[str, Any] | TurnRow:
        preview = None
        sources = None
        if content is not None:
            text = content.get("shown_input") or content.get("input")
            preview = text[:PREVIEW_CHARS] if isinstance(text, str) else None
            if isinstance(content.get("sources"), list):
                sources = len(content["sources"])
        tokens = None
        if turn.prompt_tokens is not None or turn.completion_tokens is not None:
            tokens = (turn.prompt_tokens or 0) + (turn.completion_tokens or 0)
        return TurnRow(
            id=str(turn.id),
            created_at=turn.created_at,
            channel=turn.channel,
            agent_id=turn.agent_id,
            agent_name=turn.agent_name,
            policy=turn.policy,
            actor=revealed or (turn.actor_pseudonym if purpose in SEES_ACTOR else "hidden"),
            outcome=turn.outcome,
            shadow_outcome=turn.shadow_outcome,
            severity=turn.severity,
            labels=turn.labels or {},
            preview=preview,
            hits=hits,
            model=turn.model,
            tokens=tokens,
            cost=turn.cost,
            currency=str(setting("CURRENCY")),
            latency_ms=turn.latency_ms,
            sources=sources,
            reviews_open=reviews_open,
            reflected=turn.reflected_at is not None,
            chained=bool(turn.entry_hash),
        )

    @classmethod
    async def get_turn(
        cls, turn_id: str, *, user: UserType, purpose: Purpose | str = Purpose.SECURITY
    ) -> TurnDetail | None:
        """One turn through `purpose`'s view, with its proof on the chain."""
        from django_ai_sdk.oversight.models import (
            AccessRecord,
            Decision,
            Observation,
            ReviewItem,
            Turn,
        )
        from django_ai_sdk.oversight.policy import aget_policy

        purpose = _purpose(purpose)
        await cls._check_purpose(user, purpose)
        turn = await _aget(Turn.objects.select_related("evidence"), turn_id)
        if turn is None:
            return None
        resolved = await aget_policy(turn.policy)
        content = None
        state = ""
        if purpose is Purpose.AUDIT:
            state = "withheld"
        else:
            content, state = await sync_to_async(_content_state)(turn)
        observations = [o async for o in Observation.objects.filter(turn=turn)]
        decisions = [d async for d in Decision.objects.filter(turn=turn)]
        reviews = [r async for r in ReviewItem.objects.filter(turn=turn)]
        reader = staff_label(user)
        revealed = await AccessRecord.objects.filter(
            turn_ref=str(turn.id), action="reveal", reader_label=reader
        ).aexists()
        hits = [
            RuleHit(
                rule=d.rule,
                stage=d.stage,
                mode=d.mode,
                acted=d.acted,
                actions=[a.get("kind", "") for a in d.actions],
            )
            for d in decisions
        ]
        actor = ""
        if revealed:
            actor = (await cls._actor(turn)).get("label", "")
        row = cls._row(
            turn, purpose, content, hits, sum(r.status == "open" for r in reviews), revealed=actor
        )
        masked = mask_evidence(content, purpose)
        if content is not None and masked is not None:
            masked.update({k: v for k, v in content.items() if k in META_KEYS})
            masked.pop("salt", None)
        detail = TurnDetail(
            **row.model_dump(),
            purpose=str(purpose),
            evidence=masked,
            evidence_state=state,
            observations=[
                _observation_out(o, resolved, purpose)
                for o in observations
                if _lens_shown(o.lens, resolved, purpose)
            ],
            decisions=[_decision_out(d) for d in decisions],
            reviews=[_review_out(r) for r in reviews],
            chain=await sync_to_async(_proof)(turn),
            thread_id=turn.thread_id if purpose is not Purpose.AUDIT else "",
            message_id=turn.message_id if purpose is not Purpose.AUDIT else "",
            agent_fingerprint=turn.agent_fingerprint,
            revealed=revealed,
        )
        await cls._access(user, purpose, "view", turn_ref=str(turn.id))
        return detail

    @classmethod
    async def reveal(cls, turn_id: str, *, user: UserType, reason: str) -> RevealOut | None:
        """Who asked, with the reason on the ledger next to who wanted to know."""
        from django_ai_sdk.oversight.models import Turn

        await cls._check(user, Operation.REVEAL_ACTOR)
        reason = (reason or "").strip()
        if len(reason) < 8:
            raise UserError("Give a reason of at least a few words to reveal who asked.")
        turn = await _aget(Turn.objects.all(), turn_id)
        if turn is None:
            return None
        actor = await cls._actor(turn)
        record = await cls._access(
            user, Purpose.SECURITY, "reveal", turn_ref=str(turn.id), reason=reason
        )
        return RevealOut(
            turn_id=str(turn.id),
            actor=turn.actor_pseudonym,
            user_id=actor.get("pk", ""),
            username=actor.get("username", ""),
            access_id=str(record.id),
        )

    @staticmethod
    async def _actor(turn: Turn) -> dict[str, str]:
        from django.contrib.auth import get_user_model

        from django_ai_sdk.oversight import vault

        if not turn.actor_ref:
            return {"label": turn.actor_pseudonym}
        pk = await sync_to_async(vault.open_text)(f"actor:{turn.actor_pseudonym}", turn.actor_ref)
        if not pk:
            return {"label": f"{turn.actor_pseudonym} (forgotten)"}
        user_model = get_user_model()
        found = await user_model._default_manager.filter(pk=pk).afirst()
        username = found.get_username() if found else ""
        return {"pk": pk, "username": username, "label": f"{username or pk}"}

    # Review queues.

    @classmethod
    async def list_reviews(
        cls,
        *,
        user: UserType,
        queue: str = "",
        status: str = "open",
        purpose: Purpose | str = Purpose.QUALITY,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ReviewOut]:
        from django_ai_sdk.oversight.models import ReviewItem

        purpose = _purpose(purpose)
        await cls._check(user, Operation.REVIEW_OVERSIGHT)
        qs = ReviewItem.objects.select_related("turn", "turn__evidence", "assignee")
        if queue:
            qs = qs.filter(queue=queue)
        if status:
            qs = qs.filter(status=status)
        items = [r async for r in qs.order_by("-priority", "created_at")[offset : offset + limit]]
        rows = {row.id: row for row in await cls._rows([r.turn for r in items], purpose)}
        return [_review_out(r, rows.get(str(r.turn_id))) for r in items]

    @classmethod
    async def queues(cls, *, user: UserType) -> dict[str, int]:
        from django.db.models import Count

        from django_ai_sdk.oversight.models import ReviewItem

        await cls._check(user, Operation.REVIEW_OVERSIGHT)
        return {
            row["queue"]: row["n"]
            async for row in ReviewItem.objects.filter(status="open")
            .values("queue")
            .annotate(n=Count("id"))
        }

    @classmethod
    async def resolve_review(
        cls, item_id: str, verdict: VerdictIn, *, user: UserType, dismiss: bool = False
    ) -> ReviewOut | None:
        """Close a review item with a person's verdict, which goes on the ledger."""
        from django_ai_sdk.oversight.models import ReviewItem

        await cls._check(user, Operation.REVIEW_OVERSIGHT)
        item = await _aget(ReviewItem.objects.select_related("turn"), item_id)
        if item is None:
            return None
        if not dismiss:
            lens = verdict.lens or item.lens
            if lens:
                await cls._human_answer(item.turn, lens, verdict, user)
            elif item.rule:
                await cls._human_answer(
                    item.turn,
                    f"rule:{item.rule}",
                    VerdictIn(value=verdict.agree, note=verdict.note),
                    user,
                )
        item.status = ReviewItem.Status.DISMISSED if dismiss else ReviewItem.Status.DONE
        item.resolved_at = timezone.now()
        item.resolution = {
            "by": pseudonym(user),
            "value": verdict.value,
            "agree": verdict.agree,
            "note": verdict.note,
        }
        await item.asave(update_fields=["status", "resolved_at", "resolution"])
        await cls._access(user, Purpose.QUALITY, "review", turn_ref=str(item.turn_id))
        return _review_out(item)

    @classmethod
    async def add_verdict(
        cls, turn_id: str, verdict: VerdictIn, *, user: UserType
    ) -> ObservationOut | None:
        """A person's answer to a lens on any turn, queued or not."""
        from django_ai_sdk.oversight.models import Turn

        await cls._check(user, Operation.REVIEW_OVERSIGHT)
        if not verdict.lens:
            raise UserError("Name the lens the verdict answers.")
        turn = await _aget(Turn.objects.all(), turn_id)
        if turn is None:
            return None
        return await cls._human_answer(turn, verdict.lens, verdict, user)

    @classmethod
    async def _human_answer(
        cls, turn: Turn, lens_name: str, verdict: VerdictIn, user: UserType
    ) -> ObservationOut:
        from django_ai_sdk.oversight.lenses import Flag
        from django_ai_sdk.oversight.models import Observation
        from django_ai_sdk.oversight.policy import aget_policy
        from django_ai_sdk.oversight.stages import Observed, evaluate_rules
        from django_ai_sdk.oversight.tasks import answers_from, record

        resolved = await aget_policy(turn.policy)
        lens = resolved.lenses.get(lens_name) if resolved else None
        value = verdict.value if verdict.agree is None or lens is not None else verdict.agree
        if lens is None:
            # A verdict on a rule's match: yes or no, on a lens of its own.
            lens = Flag(
                name=lens_name, question="Was the rule right to match?", answerers=["human"]
            )
            value = verdict.agree if verdict.agree is not None else verdict.value
        try:
            value = lens.parse(value)
        except (TypeError, ValueError) as exc:
            raise UserError(f"Not a valid answer for {lens_name!r}: {exc}") from exc
        reviewer = pseudonym(user)
        answer = Answer(value=value, confidence=1.0, rationale=verdict.note, answerer="human")
        observed = [Observed(lens, Stage.RETRO, answer)]
        matched = []
        if resolved is not None:
            existing = [o async for o in Observation.objects.filter(turn=turn)]
            answers = answers_from(existing)
            answers[lens_name] = answer
            from django_ai_sdk.oversight.models import Decision

            done = {
                d async for d in Decision.objects.filter(turn=turn).values_list("rule", flat=True)
            }
            matched = evaluate_rules(resolved, answers, Stage.RETRO, done=done)
        await sync_to_async(record, thread_sensitive=True)(
            turn, resolved, observed, matched, reviewer=reviewer
        )
        obs = (
            await Observation.objects.filter(turn=turn, lens=lens_name, reviewer=reviewer)
            .order_by("-created_at")
            .afirst()
        )
        assert obs is not None
        return _observation_out(obs, resolved, Purpose.QUALITY)

    # Policies.

    @classmethod
    async def list_policies(cls, *, user: UserType) -> list[PolicySummary]:
        """Every policy: packs, code and the console's own, with the agents each watches."""
        from django_ai_sdk.oversight.policy import aget_policy, invalidate
        from django_ai_sdk.oversight.registry import get_policies
        from django_ai_sdk.oversight.types import Mode

        await cls._check(user, Operation.VIEW_OVERSIGHT)
        invalidate()
        agents: dict[str, list[str]] = {}
        rows = await cls._agent_rows()
        code_names = {row.name for row in rows if row.kind == "code"}
        for row in rows:
            if row.effective:
                # A configured agent may share a bot's name; it answers in the web chat.
                clash = row.kind == "runtime" and row.name in code_names
                label = f"{row.name} (web chat)" if clash else row.name
                agents.setdefault(row.effective, []).append(label)
        names = {*get_policies(), *await _console_policies()}
        summaries = []
        for name in sorted(names):
            resolved = await aget_policy(name)
            cls_ = get_policies().get(name)
            if resolved is None:
                summaries.append(
                    PolicySummary(
                        name=name,
                        label=name,
                        origin="db" if cls_ is None else cls_.origin,
                        error="does not resolve: check what it extends",
                        agents=sorted(agents.get(name, [])),
                    )
                )
                continue
            modes = [rule.mode for rule in resolved.rules.values()]
            summaries.append(
                PolicySummary(
                    name=name,
                    label=resolved.label or name,
                    description=resolved.description,
                    builtin=resolved.origin.startswith("pack:"),
                    origin=resolved.origin,
                    extends=list(resolved.extends),
                    agents=sorted(agents.get(name, [])),
                    lenses=sum(1 for lens in resolved.lenses.values() if lens.enabled),
                    rules=len(resolved.rules),
                    enforced=modes.count(Mode.ENFORCE),
                    shadow=modes.count(Mode.SHADOW),
                )
            )
        return summaries

    @classmethod
    async def get_policy(cls, name: str, *, user: UserType) -> dict[str, Any] | None:
        """The effective policy: every lens and rule, and where each value comes from."""
        from django_ai_sdk.oversight.answerers.base import alias_table
        from django_ai_sdk.oversight.policy import aget_policy, answerer_spec, invalidate

        await cls._check(user, Operation.VIEW_OVERSIGHT)
        invalidate(name)
        resolved = await aget_policy(name)
        if resolved is None:
            return None
        described = resolved.describe()
        for lens in described["lenses"]:
            raw = resolved.lenses[lens["name"]]
            lens["answerer_specs"] = [answerer_spec(raw, spec) for spec in raw.answerers]
            lens["guidance"] = raw.guidance
            for attr in ("criteria", "default", "low", "high", "yes", "no", "max_chars"):
                if hasattr(raw, attr):
                    lens[attr] = getattr(raw, attr)
            lens["origin"] = lens["sources"].pop("origin", "code")
        for rule in described["rules"]:
            rule["origin"] = rule["sources"].pop("origin", "code")
        described["aliases"] = sorted([*alias_table(), "human"])
        described["agents"] = [
            row.model_dump() for row in await cls._agent_rows() if row.effective == name
        ]
        return described

    @classmethod
    async def set_override(
        cls, name: str, change: OverrideIn, *, user: UserType
    ) -> dict[str, Any] | None:
        """Tune one field of a policy from the console; rejected changes raise ValueError."""
        from django_ai_sdk.oversight.policy import _aload_overlay, build, invalidate
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
            trial = build(name, overlay=[*await _aload_overlay(name), entry])
            rejected = [
                r
                for r in trial.rejected
                if r.get("source") == "db" and {k: r.get(k) for k in entry} == entry
            ]
            if rejected:
                raise UserError(rejected[-1]["reason"])
        await sync_to_async(_write_override, thread_sensitive=True)(name, change, user)
        invalidate(name)
        return await cls.get_policy(name, user=user)

    @classmethod
    async def create_policy(cls, payload: PolicyIn, *, user: UserType) -> dict[str, Any]:
        """A new policy, defined in the console: blank, or from one of the examples."""
        from django_ai_sdk.oversight.catalog import POLICY_EXAMPLES, lens_example, rule_example
        from django_ai_sdk.oversight.policy import GLOBAL, build, invalidate
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
        trial = build(name, overlay=entries)
        if trial.rejected:
            raise UserError(trial.rejected[0]["reason"])
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

        console = (await aconsole())["agents"]
        rows = await sync_to_async(_agents, thread_sensitive=True)()
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
        from django_ai_sdk.oversight.answerers.base import _from_alias, set_console_aliases
        from django_ai_sdk.oversight.policy import GLOBAL, aconsole, invalidate

        await cls._check(user, Operation.MANAGE_OVERSIGHT)
        _check_slug(alias, "answerer")
        if alias == "human":
            raise UserError("'human' is built in.")
        if not payload.reset:
            if not payload.config:
                raise UserError("Give the answerer a configuration.")
            current = dict((await aconsole())["answerers"])
            trial = {**current, alias: payload.config}
            set_console_aliases(trial)
            try:
                _from_alias(alias)
            except Exception as exc:
                raise UserError(f"That answerer cannot be built: {exc}") from exc
            finally:
                set_console_aliases(current)
        change = OverrideIn(
            target_kind="answerer",
            target=alias,
            field="definition",
            value=None if payload.reset else payload.config,
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
        from django_ai_sdk.oversight.lenses import Flag
        from django_ai_sdk.oversight.simulate import answer_out
        from django_ai_sdk.oversight.types import TurnView

        await cls._check(user, Operation.MANAGE_OVERSIGHT)
        try:
            answerer = resolve(payload.config)
        except Exception as exc:
            raise UserError(f"That answerer cannot be built: {exc}") from exc
        lens = Flag(name="console_test", question=payload.question, answerers=[answerer])
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
        from django_ai_sdk.oversight.lenses import lens_from_config
        from django_ai_sdk.oversight.policy import _code_answerer, aget_policy
        from django_ai_sdk.oversight.simulate import try_lens

        await cls._check(user, Operation.MANAGE_OVERSIGHT)
        resolved = await aget_policy(name)
        if resolved is None:
            return None
        if payload.definition:
            config = dict(payload.definition)
            config.setdefault("name", payload.lens or "draft")
            # A draft of a code lens may keep its code-built answerers ("code:0").
            saved = resolved.lenses.get(payload.lens)
            if saved is not None:
                config["answerers"] = [
                    _code_answerer(saved, spec) for spec in config.get("answerers") or []
                ]
            try:
                lens = lens_from_config(config)
            except Exception as exc:
                raise UserError(f"That lens is not complete: {exc}") from exc
        else:
            lens = resolved.lenses.get(payload.lens)
            if lens is None:
                return None
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
        return [_override_out(o) async for o in rows.order_by("-changed_at")[: max(1, limit)]]

    @classmethod
    async def list_overrides(cls, name: str, *, user: UserType) -> list[OverrideOut]:
        return await cls.changes(user=user, policy=name)

    # Insight.

    @classmethod
    async def overview(
        cls, *, user: UserType, days: int = 14, agent_id: str = "", policy: str = ""
    ) -> Overview:
        from django_ai_sdk.oversight.metrics import overview

        await cls._check(user, Operation.VIEW_OVERSIGHT)
        since = timezone.now() - timedelta(days=max(1, min(days, 365)))
        return Overview(**await overview(since, agent_id=agent_id, policy=policy))

    # The ledger itself.

    @classmethod
    async def checkpoints(cls, *, user: UserType, limit: int = 50) -> list[CheckpointOut]:
        from django_ai_sdk.oversight.models import Checkpoint

        await cls._check(user, Operation.AUDIT_OVERSIGHT)
        return [
            CheckpointOut(
                id=str(c.id),
                tree_size=c.tree_size,
                root_hash=c.root_hash,
                note=c.note,
                signed=c.signed,
                created_at=c.created_at,
                anchored_at=c.anchored_at,
                anchor_ref=c.anchor_ref,
            )
            async for c in Checkpoint.objects.order_by("-tree_size")[:limit]
        ]

    @classmethod
    async def verify(cls, *, user: UserType) -> VerifyOut:
        """Walk the whole chain and check the latest checkpoint against it."""
        await cls._check(user, Operation.AUDIT_OVERSIGHT)
        result = await sync_to_async(_verify_all, thread_sensitive=True)()
        await cls._access(user, Purpose.AUDIT, "verify", query={"ok": result.ok})
        return result

    @classmethod
    async def seal_now(cls, *, user: UserType) -> CheckpointOut | None:
        from django_ai_sdk.oversight.models import Checkpoint
        from django_ai_sdk.oversight.tasks import seal

        await cls._check(user, Operation.MANAGE_OVERSIGHT)
        made = await sync_to_async(seal, thread_sensitive=True)()
        if made is None:
            return None
        c = await Checkpoint.objects.aget(id=made["id"])
        return CheckpointOut(
            id=str(c.id),
            tree_size=c.tree_size,
            root_hash=c.root_hash,
            note=c.note,
            signed=c.signed,
            created_at=c.created_at,
            anchored_at=c.anchored_at,
            anchor_ref=c.anchor_ref,
        )

    @classmethod
    async def access_log(
        cls, *, user: UserType, turn_id: str = "", limit: int = 100, offset: int = 0
    ) -> list[AccessOut]:
        from django_ai_sdk.oversight.models import AccessRecord

        await cls._check(user, Operation.AUDIT_OVERSIGHT)
        qs = AccessRecord.objects.all()
        if turn_id:
            qs = qs.filter(turn_ref=turn_id)
        return [
            AccessOut(
                id=str(r.id),
                reader=r.reader_label,
                purpose=r.purpose,
                action=r.action,
                turn_ref=r.turn_ref,
                reason=r.reason,
                created_at=r.created_at,
            )
            async for r in qs.order_by("-created_at")[offset : offset + limit]
        ]

    @classmethod
    async def export(
        cls, *, user: UserType, purpose: Purpose | str = Purpose.AUDIT, days: int = 30
    ) -> AsyncIterator[dict[str, Any]]:
        """Every turn of the window through `purpose`'s view, one dict at a time."""
        from django_ai_sdk.oversight.models import Turn

        purpose = _purpose(purpose)
        await cls._check(user, Operation.MANAGE_OVERSIGHT)
        await cls._check_purpose(user, purpose)
        since = timezone.now() - timedelta(days=max(1, min(days, 3650)))
        await cls._access(user, purpose, "export", query={"days": days})

        async def rows() -> AsyncIterator[dict[str, Any]]:
            batch: list[Turn] = []
            async for turn in (
                Turn.objects.filter(created_at__gte=since)
                .select_related("evidence")
                .order_by("created_at")
                .aiterator(chunk_size=200)
            ):
                batch.append(turn)
                if len(batch) >= 200:
                    for row in await cls._rows(batch, purpose):
                        yield row.model_dump(mode="json")
                    batch = []
            for row in await cls._rows(batch, purpose):
                yield row.model_dump(mode="json")

        return rows()

    @classmethod
    async def forget_thread(cls, thread_id: str, *, user: UserType, reason: str) -> bool:
        """Crypto-shred everything oversight holds about one thread."""
        from django_ai_sdk.oversight.tasks import shred_scope

        await cls._check(user, Operation.MANAGE_OVERSIGHT)
        if len((reason or "").strip()) < 8:
            raise UserError("Give a reason of at least a few words.")
        shredded = await sync_to_async(shred_scope, thread_sensitive=True)(f"thread:{thread_id}")
        await cls._access(
            user, Purpose.AUDIT, "shred", query={"thread_id": thread_id}, reason=reason
        )
        return shredded

    # Recording reads.

    @classmethod
    async def _access(
        cls,
        user: UserType,
        purpose: Purpose,
        action: str,
        *,
        turn_ref: str = "",
        query: dict[str, Any] | None = None,
        reason: str = "",
    ) -> Any:
        return await sync_to_async(_write_access, thread_sensitive=True)(
            user, purpose, action, turn_ref, query or {}, reason
        )


def _typed(value: str) -> Any:
    """A filter value from a query string, as the JSON type labels store."""
    lowered = value.lower()
    if lowered in {"true", "yes"}:
        return True
    if lowered in {"false", "no"}:
        return False
    try:
        return float(value) if "." in value else int(value)
    except ValueError:
        return value


async def _aget(qs: Any, pk: str) -> Any:
    try:
        return await qs.filter(pk=pk).afirst()
    except (ValidationError, ValueError):
        return None


def _contents(turns: list[Turn]) -> dict[Any, dict[str, Any] | None]:
    from django_ai_sdk.oversight.tasks import load_evidence

    return {t.id: load_evidence(t) for t in turns}


def _content_state(turn: Turn) -> tuple[dict[str, Any] | None, str]:
    from django_ai_sdk.oversight.tasks import load_evidence

    evidence = turn.evidence
    if not turn.evidence_hash:
        return None, "none"
    if evidence is None or evidence.shredded_at is not None:
        expired = (
            evidence is not None and evidence.expires_at and evidence.expires_at <= timezone.now()
        )
        return None, "expired" if expired else "shredded"
    content = load_evidence(turn)
    if content is None:
        return None, "shredded"
    return content, "metadata" if evidence.capture == "metadata" else evidence.capture


def _proof(turn: Turn) -> ChainProof:
    from django_ai_sdk.oversight import ledger
    from django_ai_sdk.oversight.models import Checkpoint, LedgerEntry

    entry = LedgerEntry.objects.filter(kind="turn", ref=str(turn.id)).first()
    if entry is None:
        return ChainProof(evidence_hash=turn.evidence_hash)
    checkpoint = Checkpoint.objects.filter(tree_size__gte=entry.seq).order_by("tree_size").first()
    intact = ledger.sha256(ledger.canonical(ledger.turn_payload(turn))) == entry.payload_hash
    return ChainProof(
        seq=entry.seq,
        entry_hash=entry.entry_hash,
        prev_hash=entry.prev_hash,
        evidence_hash=turn.evidence_hash,
        checkpoint_size=checkpoint.tree_size if checkpoint else None,
        checkpoint_root=checkpoint.root_hash if checkpoint else "",
        checkpoint_signed=checkpoint.signed if checkpoint else False,
        intact=intact,
    )


def _verify_all() -> VerifyOut:
    from django_ai_sdk.oversight import ledger, seal
    from django_ai_sdk.oversight.models import Checkpoint, LedgerEntry

    result = ledger.verify()
    checkpoint_ok = None
    checkpoint = Checkpoint.objects.order_by("-tree_size").first()
    if checkpoint is not None:
        leaves = [
            bytes.fromhex(h)
            for h in LedgerEntry.objects.order_by("seq").values_list("entry_hash", flat=True)[
                : checkpoint.tree_size
            ]
        ]
        checkpoint_ok = (
            len(leaves) == checkpoint.tree_size
            and seal.merkle_root(leaves).hex() == checkpoint.root_hash
        )
    return VerifyOut(
        **result.as_dict(),
        checkpoint_ok=checkpoint_ok,
        checkpoint_size=checkpoint.tree_size if checkpoint else None,
    )


def _write_access(
    user: Any, purpose: Purpose, action: str, turn_ref: str, query: dict[str, Any], reason: str
) -> Any:
    from django.db import transaction

    from django_ai_sdk.oversight import ledger
    from django_ai_sdk.oversight.models import AccessRecord

    reader = user if getattr(user, "pk", None) is not None else None
    with transaction.atomic():
        record = AccessRecord.objects.create(
            reader=reader,
            reader_label=staff_label(user),
            purpose=str(purpose),
            action=action,
            turn_ref=turn_ref,
            query=query,
            reason=reason,
        )
        ledger.append("access", record)
    return record


SLUG = r"^[a-z][a-z0-9_-]{1,62}$"


def _check_slug(value: str, what: str) -> None:
    import re

    if not re.match(SLUG, value or ""):
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


def _agents() -> list[dict[str, Any]]:
    """Registered agent classes and runtime agents, with what their code asks for."""
    from django.utils.module_loading import import_string

    from django_ai_sdk.agents.models import AgentSettings
    from django_ai_sdk.agents.registry import registry
    from django_ai_sdk.oversight.session import agent_key, code_policy

    rows: list[dict[str, Any]] = []
    for agent in registry.all().values():
        policy, reason = code_policy(agent)
        rows.append(
            {
                "id": agent_key(agent),
                "name": getattr(agent, "name", "") or type(agent).__name__,
                "kind": "code",
                "description": _code_description(agent),
                "code_policy": policy,
                "code_reason": reason,
            }
        )
    for config in AgentSettings.objects.filter(active=True).order_by("name"):
        try:
            cls = import_string(config.agent) if config.agent else None
        except ImportError:
            cls = None
        probe = type("Probe", (), {"oversight": getattr(cls, "oversight", None)})()
        policy, reason = code_policy(probe)
        rows.append(
            {
                "id": str(config.id),
                "name": config.name,
                "kind": "runtime",
                "description": f"Configured agent · {config.slug} · {config.model}",
                "code_policy": policy,
                "code_reason": reason,
            }
        )
    return rows


def _code_description(agent: Any) -> str:
    described = str(getattr(agent, "description", "") or "").strip()
    integrations = ", ".join(getattr(agent, "integrations", []) or [])
    parts = [described[:160]] if described else ["Code agent"]
    if integrations:
        parts.append(f"integrations: {integrations}")
    return " · ".join(parts)


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
    from django.db import transaction

    with transaction.atomic():
        for entry in entries:
            _write_override(name, OverrideIn(**entry, note=note), user)


def _withdraw_policy(name: str, user: Any, note: str) -> None:
    from django.db import transaction

    from django_ai_sdk.oversight.models import PolicyOverride
    from django_ai_sdk.oversight.policy import GLOBAL

    with transaction.atomic():
        for override in PolicyOverride.objects.filter(policy=name, active=True):
            _write_override(
                name,
                OverrideIn(
                    target_kind=override.target_kind,
                    target=override.target,
                    field=override.field,
                    reset=True,
                    note=note or "Policy removed",
                ),
                user,
            )
        for override in PolicyOverride.objects.filter(
            policy=GLOBAL, target_kind="agent", field="policy", value=name, active=True
        ):
            _write_override(
                GLOBAL,
                OverrideIn(
                    target_kind="agent",
                    target=override.target,
                    field="policy",
                    reset=True,
                    note=f"Policy {name} removed",
                ),
                user,
            )


def _write_override(name: str, change: OverrideIn, user: Any) -> None:
    from django.db import transaction

    from django_ai_sdk.oversight import ledger
    from django_ai_sdk.oversight.models import PolicyOverride

    reader = user if getattr(user, "pk", None) is not None else None
    with transaction.atomic():
        replaced = PolicyOverride.objects.filter(
            policy=name, target_kind=change.target_kind, target=change.target, active=True
        )
        if change.field != "definition" or change.target_kind in {"policy", "answerer", "agent"}:
            replaced = replaced.filter(field=change.field)
        # A lens or rule (re)defined in the console starts clean: its earlier tuning
        # was of the old definition.
        replaced.update(active=False)
        override = PolicyOverride.objects.create(
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
        ledger.append("override", override)


def _lens_shown(name: str, resolved: ResolvedPolicy | None, purpose: Purpose) -> bool:
    if name.startswith("rule:"):
        return True
    lens = resolved.lenses.get(name) if resolved else None
    return lens is None or lens_visible(lens.purposes, purpose)


def _observation_out(obs: Any, resolved: ResolvedPolicy | None, purpose: Purpose) -> ObservationOut:
    lens = resolved.lenses.get(obs.lens) if resolved else None
    hide_reasons = purpose is Purpose.AUDIT
    return ObservationOut(
        id=str(obs.id),
        lens=obs.lens,
        label=(lens.label or lens.name.replace("_", " ")) if lens else obs.lens,
        kind=lens.kind if lens else "",
        stage=obs.stage,
        answerer=obs.answerer,
        model=obs.model,
        value=obs.value,
        confidence=obs.confidence,
        rationale="" if hide_reasons else obs.rationale,
        spans=[] if hide_reasons else obs.spans,
        trail=obs.trail,
        error=obs.error,
        queued=obs.queued,
        cost=obs.cost,
        latency_ms=obs.latency_ms,
        reviewer=obs.reviewer,
        created_at=obs.created_at,
    )


def _decision_out(decision: Any) -> DecisionOut:
    return DecisionOut(
        id=str(decision.id),
        rule=decision.rule,
        stage=decision.stage,
        mode=decision.mode,
        acted=decision.acted,
        actions=decision.actions,
        condition=decision.condition,
        created_at=decision.created_at,
    )


def _review_out(item: Any, row: TurnRow | None = None) -> ReviewOut:
    return ReviewOut(
        id=str(item.id),
        turn_id=str(item.turn_id),
        queue=item.queue,
        lens=item.lens,
        rule=item.rule,
        priority=item.priority,
        status=item.status,
        assignee=pseudonym(item.assignee) if item.assignee_id else "",
        created_at=item.created_at,
        resolved_at=item.resolved_at,
        resolution=item.resolution,
        turn=row,
    )


__all__ = ["OversightService"]
