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
from django_ai_sdk.oversight.console import ConsoleMixin, _console_policies
from django_ai_sdk.oversight.privacy import mask_evidence, pseudonym, staff_label
from django_ai_sdk.oversight.schemas import (
    AccessOut,
    ChainProof,
    CheckpointOut,
    DecisionOut,
    ObservationOut,
    Overview,
    PolicySummary,
    RevealOut,
    ReviewOut,
    RuleHit,
    TurnDetail,
    TurnFilter,
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


class OversightService(ConsoleMixin, PermissionsMixin):
    """Read the ledger, answer its queues, tune its policies (the console's part: ConsoleMixin)."""

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
        filters: TurnFilter | None = None,
        *,
        user: UserType,
        purpose: Purpose | str = Purpose.REVIEW,
        limit: int = 50,
        offset: int = 0,
    ) -> TurnPage:
        """Turns newest first, filtered; what each row shows depends on `purpose`."""
        from django_ai_sdk.oversight.models import Turn

        purpose = _purpose(purpose)
        await cls._check_purpose(user, purpose)
        filters = filters or TurnFilter()
        qs = _filtered(Turn.objects.all(), filters)
        total = await qs.acount()
        page = [
            t
            async for t in qs.select_related("evidence").order_by("-created_at")[
                offset : offset + limit
            ]
        ]
        rows = await cls._rows(page, purpose)
        query = filters.model_dump(mode="json", exclude_defaults=True)
        await cls._access(
            user, purpose, "list", query={**query, **({"offset": offset} if offset else {})}
        )
        return TurnPage(items=rows, total=total, limit=limit, offset=offset)

    @classmethod
    async def _rows(cls, turns: list[Turn], purpose: Purpose) -> list[TurnRow]:
        from django_ai_sdk.oversight.models import Decision, ReviewItem

        ids = [t.id for t in turns]
        hits: dict[Any, list[RuleHit]] = {}
        async for d in Decision.objects.filter(turn_id__in=ids).order_by("created_at"):
            hits.setdefault(d.turn_id, []).append(_hit(d))
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
    ) -> TurnRow:
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
            actor=revealed or turn.actor_pseudonym,
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
        cls, turn_id: str, *, user: UserType, purpose: Purpose | str = Purpose.REVIEW
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
        reader_pk = getattr(user, "pk", None)
        revealed = reader_pk is not None and (
            await AccessRecord.objects.filter(
                turn_ref=str(turn.id), action="reveal", reader_id=reader_pk
            ).aexists()
        )
        hits = [_hit(d) for d in decisions]
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
                _observation_out(o, resolved, purpose, texts=(content or {}).get("texts"))
                for o in observations
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
            user, Purpose.REVIEW, "reveal", turn_ref=str(turn.id), reason=reason
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
        purpose: Purpose | str = Purpose.REVIEW,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ReviewOut]:
        from django_ai_sdk.oversight.models import ReviewItem

        purpose = _purpose(purpose)
        await cls._check(user, Operation.REVIEW_OVERSIGHT)
        await cls._check_purpose(user, purpose)
        qs = ReviewItem.objects.select_related("turn", "turn__evidence", "assignee")
        if queue:
            qs = qs.filter(queue=queue)
        if status:
            qs = qs.filter(status=status)
        items = [r async for r in qs.order_by("-priority", "created_at")[offset : offset + limit]]
        rows = {row.id: row for row in await cls._rows([r.turn for r in items], purpose)}
        await cls._access(
            user, purpose, "reviews", query={"queue": queue, "status": status, "offset": offset}
        )
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
        await cls._access(user, Purpose.REVIEW, "review", turn_ref=str(item.turn_id))
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
        texts = {lens_name: answer.value} if lens.kind == "text" else None
        return _observation_out(obs, resolved, Purpose.REVIEW, texts=texts)

    # Policies.

    @classmethod
    async def list_policies(cls, *, user: UserType) -> list[PolicySummary]:
        """Every policy: packs, code and the console's own, with the agents each watches."""
        from django_ai_sdk.oversight.policy import aget_policy, broken
        from django_ai_sdk.oversight.registry import get_policies
        from django_ai_sdk.oversight.types import Mode

        await cls._check(user, Operation.VIEW_OVERSIGHT)
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
            resolved = await aget_policy(name, uncached=True)
            cls_ = get_policies().get(name)
            if resolved is None:
                summaries.append(
                    PolicySummary(
                        name=name,
                        label=name,
                        origin="db" if cls_ is None else cls_.origin,
                        error=broken(name) or "does not resolve",
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
        from django_ai_sdk.oversight.policy import aget_policy, answerer_spec, broken
        from django_ai_sdk.oversight.stages import lens_health

        await cls._check(user, Operation.VIEW_OVERSIGHT)
        resolved = await aget_policy(name, uncached=True)
        if resolved is None:
            from django_ai_sdk.oversight.registry import get_policy_class

            known = get_policy_class(name) is not None or name in await _console_policies()
            if known and broken(name):
                raise UserError(f"Cannot load: {broken(name)}")
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
            # After aget_policy, so console aliases count.
            lens["health"] = lens_health(raw)
        for rule in described["rules"]:
            rule["origin"] = rule["sources"].pop("origin", "code")
        # The console folds inherited checks per parent, under the parent's own label.
        parents = {
            origin.split(":", 1)[1]
            for item in (*described["lenses"], *described["rules"])
            if (origin := item["origin"]).startswith(("pack:", "db:"))
        }
        labels: dict[str, str] = {}
        for parent in sorted(parents):
            found = await aget_policy(parent)
            labels[parent] = (found.label if found else "") or parent
        described["parent_labels"] = labels
        described["aliases"] = sorted([*alias_table(), "human"])
        described["agents"] = [
            row.model_dump() for row in await cls._agent_rows() if row.effective == name
        ]
        return described

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
        return [_checkpoint_out(c) async for c in Checkpoint.objects.order_by("-tree_size")[:limit]]

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
        return _checkpoint_out(await Checkpoint.objects.aget(id=made["id"]))

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


def _filtered(qs: Any, f: TurnFilter) -> Any:
    if f.outcome:
        qs = qs.filter(outcome=f.outcome)
    if f.shadow_outcome:
        # "Would have been X" means it was not X in fact.
        qs = qs.filter(shadow_outcome=f.shadow_outcome).exclude(outcome=f.shadow_outcome)
    if f.agent_id:
        qs = qs.filter(agent_id=f.agent_id)
    if f.channel:
        qs = qs.filter(channel__startswith=f.channel)
    if f.policy:
        qs = qs.filter(policy=f.policy)
    if f.severity:
        qs = qs.filter(severity=f.severity)
    if f.thread_id:
        qs = qs.filter(thread_id=f.thread_id)
    if f.label:
        qs = qs.filter(**{f"labels__{f.label}__isnull": False})
        if f.value != "":
            qs = qs.filter(**{f"labels__{f.label}": _typed(f.value)})
    if f.rule:
        qs = qs.filter(decisions__rule=f.rule).distinct()
    if f.review == "open":
        qs = qs.filter(reviews__status="open").distinct()
    elif f.review == "reviewed":
        qs = qs.exclude(observations__reviewer="").filter(observations__isnull=False).distinct()
    if f.since:
        qs = qs.filter(created_at__gte=f.since)
    if f.until:
        qs = qs.filter(created_at__lt=f.until)
    return qs


def _hit(decision: Any) -> RuleHit:
    return RuleHit(
        rule=decision.rule,
        stage=decision.stage,
        mode=decision.mode,
        acted=decision.acted,
        actions=[a.get("kind", "") for a in decision.actions],
    )


def _checkpoint_out(c: Any) -> CheckpointOut:
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


def _observation_out(
    obs: Any,
    resolved: ResolvedPolicy | None,
    purpose: Purpose,
    *,
    texts: dict[str, Any] | None = None,
) -> ObservationOut:
    lens = resolved.lenses.get(obs.lens) if resolved else None
    hide_reasons = purpose is Purpose.AUDIT
    value = obs.value
    if lens is not None and lens.kind == "text":
        # Text answers are kept with the evidence, and shown where the content is.
        value = None if hide_reasons else (texts or {}).get(obs.lens, value)
    return ObservationOut(
        id=str(obs.id),
        lens=obs.lens,
        label=(lens.label or lens.name.replace("_", " ")) if lens else obs.lens,
        kind=lens.kind if lens else "",
        stage=obs.stage,
        answerer=obs.answerer,
        model=obs.model,
        value=value,
        confidence=obs.confidence,
        rationale="" if hide_reasons else obs.rationale,
        spans=[] if hide_reasons else obs.spans,
        items={} if hide_reasons else obs.items,
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
