"""Service-layer facade for oversight: the seam an HTTP layer sits on.

Reading turns and reviews is itself written to the ledger as an AccessRecord, with
the purpose it was made for; the same reader repeating the same read within
REPEAT_SECONDS (a page polling) is recorded once. Aggregates (overview, queues),
policy and checkpoint reads, the access log, the admin and management commands
are not recorded. Content is shown through that
purpose's view only. Unknown ids return None, so a router answers 404 without
catching; a refused operation raises PermissionDenied.
"""

from __future__ import annotations

import re
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from asgiref.sync import sync_to_async
from django.core.exceptions import ImproperlyConfigured, ValidationError
from django.utils import timezone

from django_ai_sdk.errors import UserError
from django_ai_sdk.logger import get_logger
from django_ai_sdk.oversight.conf import setting
from django_ai_sdk.oversight.console import ConsoleMixin, _console_policies
from django_ai_sdk.oversight.privacy import mask_evidence, pseudonym, staff_label
from django_ai_sdk.oversight.record import stored_value
from django_ai_sdk.oversight.schemas import (
    MAX_LIMIT,
    AccessOut,
    AgentRef,
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
    ConflictError,
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
    from django_ai_sdk.oversight.stages import Matched, Observed

    UserType = AbstractBaseUser | AnonymousUser | None

logger = get_logger(__name__)

# Evidence keys without text in them: every purpose may see these.
META_KEYS = frozenset({"agent", "capture", "input_chars", "output_chars"})
PREVIEW_CHARS = 160
# A label filter names a lens; it becomes part of an ORM lookup, so nothing else.
LABEL = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,99}")
# Reads that a page repeats as it polls, and how long one record covers them.
REPEATABLE = frozenset({"list", "view", "reviews"})
REPEAT_SECONDS = 60


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
        limit, offset = _page(limit, offset)
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
    async def _rows(
        cls,
        turns: list[Turn],
        purpose: Purpose,
        contents: dict[Any, dict[str, Any] | None] | None = None,
    ) -> list[TurnRow]:
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
        if contents is None:
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
            report_outcome=turn.report_outcome,
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
        audit = purpose is Purpose.AUDIT
        content = None
        state = ""
        if audit:
            state = "withheld"
        else:
            content, state = await sync_to_async(_content_state)(turn)
        observations = [o async for o in Observation.objects.filter(turn=turn)]
        decisions = [d async for d in Decision.objects.filter(turn=turn)]
        reviews = [r async for r in ReviewItem.objects.filter(turn=turn)]
        reader_pk = getattr(user, "pk", None)
        # Audit reads the record, never who asked: not even a name this reader revealed.
        revealed = (
            not audit
            and reader_pk is not None
            and (
                await AccessRecord.objects.filter(
                    turn_ref=str(turn.id), action="reveal", reader_id=reader_pk
                ).aexists()
            )
        )
        hits = [_hit(d) for d in decisions]
        actor: dict[str, str] = {}
        if revealed:
            actor = await cls._actor(turn)
        row = cls._row(
            turn,
            purpose,
            content,
            hits,
            sum(r.status == "open" for r in reviews),
            revealed=actor.get("label", ""),
        )
        masked = mask_evidence(content, purpose)
        if content is not None and masked is not None:
            masked.update({k: v for k, v in content.items() if k in META_KEYS})
            masked.pop("salt", None)
        texts = (content or {}).get("texts") or {}
        notes = _notes(turn, content)
        detail = TurnDetail(
            **row.model_dump(),
            purpose=str(purpose),
            evidence=masked,
            evidence_state=state,
            observations=[
                _observation_out(o, resolved, purpose, texts=texts) for o in observations
            ],
            decisions=[_decision_out(d) for d in decisions],
            reviews=[_review_out(r, notes=notes) for r in reviews],
            chain=await sync_to_async(_proof)(turn),
            thread_id=turn.thread_id if not audit else "",
            message_id=turn.message_id if not audit else "",
            error=turn.error,
            parent_turn_id=str(turn.parent_turn_id or ""),
            agent_fingerprint=turn.agent_fingerprint,
            revealed=revealed,
            actor_forgotten=bool(actor.get("forgotten")),
        )
        await cls._access(user, purpose, "view", turn_ref=str(turn.id))
        return detail

    @classmethod
    async def reveal(cls, turn_id: str, *, user: UserType, reason: str) -> RevealOut | None:
        """Who asked, with the reason on the ledger next to who wanted to know."""
        from django_ai_sdk.oversight.models import Turn

        await cls._check(user, Operation.REVEAL_OVERSIGHT_ACTOR)
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
            forgotten=bool(actor.get("forgotten")),
        )

    @staticmethod
    async def _actor(turn: Turn) -> dict[str, str]:
        from django.contrib.auth import get_user_model

        from django_ai_sdk.oversight import vault

        if not turn.actor_ref:
            # Forgetting the thread forgot who asked; otherwise no one was signed in.
            scope = f"thread:{turn.thread_id}" if turn.thread_id else f"turn:{turn.id}"
            if await sync_to_async(vault.forgotten)(scope):
                return {"label": turn.actor_pseudonym, "forgotten": "yes"}
            return {"label": turn.actor_pseudonym}
        try:
            pk = await sync_to_async(vault.open_text)(
                f"actor:{turn.actor_pseudonym}", turn.actor_ref
            )
        except vault.KeyMismatch:
            # Still there for the key that sealed it: not forgotten.
            return {"label": f"{turn.actor_pseudonym} (sealed under another MASTER_KEY)"}
        except ImproperlyConfigured:
            return {"label": f"{turn.actor_pseudonym} (sealed: MASTER_KEY is unusable)"}
        if not pk:
            return {"label": turn.actor_pseudonym, "forgotten": "yes"}
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
        limit, offset = _page(limit, offset)
        qs = ReviewItem.objects.select_related("turn", "turn__evidence")
        if queue:
            qs = qs.filter(queue=queue)
        if status:
            qs = qs.filter(status=status)
        items = [r async for r in qs.order_by("-priority", "created_at")[offset : offset + limit]]
        turns = [r.turn for r in items]
        audit = purpose is Purpose.AUDIT
        contents = await sync_to_async(_contents)(turns) if not audit else {}
        rows = {row.id: row for row in await cls._rows(turns, purpose, contents)}
        await cls._access(
            user, purpose, "reviews", query={"queue": queue, "status": status, "offset": offset}
        )
        return [
            _review_out(r, rows.get(str(r.turn_id)), notes=_notes(r.turn, contents.get(r.turn_id)))
            for r in items
        ]

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
        """Close a review item with a person's verdict, which goes on the ledger.

        Raises ConflictError for an item already closed, by anyone.
        """
        from django_ai_sdk.oversight.models import ReviewItem

        await cls._check(user, Operation.REVIEW_OVERSIGHT)
        item = await _aget(ReviewItem.objects.select_related("turn", "turn__evidence"), item_id)
        if item is None:
            return None
        if item.status != ReviewItem.Status.OPEN:
            raise ConflictError("This review is closed already.")
        resolved, observed, matched = None, [], []
        name = verdict.lens or item.lens or (f"rule:{item.rule}" if item.rule else "")
        if not dismiss and name:
            resolved, observed, matched = await cls._verdict(item.turn, name, verdict)
        given = observed[0] if observed else None
        item.status = ReviewItem.Status.DISMISSED if dismiss else ReviewItem.Status.DONE
        item.resolved_at = timezone.now()
        item.resolution = {
            "by": pseudonym(user),
            # As the record keeps it: no Text answer, no name given in a person's words.
            "value": stored_value(given.lens, given.answer) if given else None,
            "agree": verdict.agree,
        }
        # The note goes with the evidence, like a judge's reason: kept where the text is.
        written = await sync_to_async(_write_verdict, thread_sensitive=True)(
            item.turn, resolved, observed, matched, pseudonym(user), close=item, note=verdict.note
        )
        if written is None:
            raise ConflictError("This review is closed already.")
        await cls._access(user, Purpose.REVIEW, "review", turn_ref=str(item.turn_id))
        return _review_out(item, notes=written[1])

    @classmethod
    async def add_verdict(
        cls, turn_id: str, verdict: VerdictIn, *, user: UserType
    ) -> ObservationOut | None:
        """A person's answer to a lens on any turn, queued or not."""
        from django_ai_sdk.oversight.models import Observation, Turn

        await cls._check(user, Operation.REVIEW_OVERSIGHT)
        if not verdict.lens:
            raise UserError("Name the lens the verdict answers.")
        turn = await _aget(Turn.objects.select_related("evidence"), turn_id)
        if turn is None:
            return None
        resolved, observed, matched = await cls._verdict(turn, verdict.lens, verdict)
        reviewer = pseudonym(user)
        written = await sync_to_async(_write_verdict, thread_sensitive=True)(
            turn, resolved, observed, matched, reviewer
        )
        obs = (
            await Observation.objects.filter(turn=turn, lens=verdict.lens, reviewer=reviewer)
            .order_by("-created_at")
            .afirst()
        )
        assert obs is not None and written is not None
        return _observation_out(obs, resolved, Purpose.REVIEW, texts=written[0])

    @classmethod
    async def _verdict(
        cls, turn: Turn, lens_name: str, verdict: VerdictIn
    ) -> tuple[ResolvedPolicy | None, list[Observed], list[Matched]]:
        """A person's answer, checked against the turn, and the rules it now trips.

        `lens_name` is a lens of the turn's policy, or "rule:<name>" for a call on
        a rule that matched the turn; anything else is a UserError.
        """
        from django_ai_sdk.oversight.lenses import Flag
        from django_ai_sdk.oversight.models import Decision, Observation
        from django_ai_sdk.oversight.policy import aget_policy
        from django_ai_sdk.oversight.stages import Observed, evaluate_rules
        from django_ai_sdk.oversight.tasks import answers_from

        resolved = await aget_policy(turn.policy)
        decisions = {
            d async for d in Decision.objects.filter(turn=turn).values_list("rule", flat=True)
        }
        if lens_name.startswith("rule:"):
            rule = lens_name.removeprefix("rule:")
            if rule not in decisions:
                raise UserError(f"Rule {rule!r} did not match this turn.")
            # Was the rule right to match: yes or no, on a lens of its own.
            lens = Flag(
                name=lens_name, question="Was the rule right to match?", answerers=["human"]
            )
            value = verdict.agree if verdict.agree is not None else verdict.value
        else:
            lens = resolved.lenses.get(lens_name) if resolved else None
            if lens is None:
                raise UserError(f"{lens_name!r} is not a lens of policy {turn.policy!r}.")
            value = verdict.value
        try:
            value = lens.parse(value)
        except (TypeError, ValueError) as exc:
            raise UserError(f"Not a valid answer for {lens_name!r}: {exc}") from exc
        answer = Answer(value=value, confidence=1.0, rationale=verdict.note, answerer="human")
        matched = []
        if resolved is not None:
            answers = answers_from([o async for o in Observation.objects.filter(turn=turn)])
            answers[lens_name] = answer
            matched = evaluate_rules(resolved, answers, Stage.REFLECT, done=decisions)
        return resolved, [Observed(lens, Stage.REFLECT, answer)], matched

    # Policies.

    @classmethod
    async def list_policies(cls, *, user: UserType) -> list[PolicySummary]:
        """Every policy: packs, code and the console's own, with the agents each watches."""
        from django_ai_sdk.oversight.policy import aget_policy, broken
        from django_ai_sdk.oversight.registry import get_policies
        from django_ai_sdk.oversight.types import Mode

        await cls._check(user, Operation.VIEW_OVERSIGHT)
        watched: dict[str, list[AgentRef]] = {}
        for row in await cls._agent_rows():
            if row.effective:
                watched.setdefault(row.effective, []).append(
                    AgentRef(id=row.id, name=row.name, kind=row.kind)
                )

        def agents(name: str) -> dict[str, Any]:
            refs = sorted(watched.get(name, []), key=lambda a: (a.name, a.kind, a.id))
            return {"agents": [a.name for a in refs], "agent_refs": refs}

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
                        **agents(name),
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
                    **agents(name),
                    lenses=sum(1 for lens in resolved.lenses.values() if lens.enabled),
                    rules=len(resolved.rules),
                    enforced=modes.count(Mode.ENFORCE),
                    report=modes.count(Mode.REPORT),
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
        limit, _ = _page(limit)
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
        limit, offset = _page(limit, offset)
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
    if f.report_outcome:
        # "Would have been X" means it was not X in fact.
        qs = qs.filter(report_outcome=f.report_outcome).exclude(outcome=f.report_outcome)
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
        if not LABEL.fullmatch(f.label) or "__" in f.label:
            raise UserError("A label filter names a lens: letters, digits, - and _.")
        qs = qs.filter(labels__has_key=f.label)
        if f.value != "":
            qs = qs.filter(**{f"labels__{f.label}": _typed(f.value)})
    if f.rule:
        qs = qs.filter(decisions__rule=f.rule).distinct()
    if f.review == "open":
        qs = qs.filter(reviews__status="open").distinct()
    elif f.review in {"reviewed", "unreviewed"}:
        from django.db.models import Exists, OuterRef

        from django_ai_sdk.oversight.models import Observation, ReviewItem

        # A person answered a lens, or closed one of its reviews.
        person = Exists(Observation.objects.filter(turn=OuterRef("pk")).exclude(reviewer="")) | (
            Exists(ReviewItem.objects.filter(turn=OuterRef("pk")).exclude(status="open"))
        )
        qs = qs.filter(person if f.review == "reviewed" else ~person)
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


def _page(limit: int, offset: int = 0) -> tuple[int, int]:
    return max(1, min(limit, MAX_LIMIT)), max(0, offset)


def _notes(turn: Turn, content: dict[str, Any] | None) -> dict[str, str] | None:
    """What reviewers wrote on `turn`, by review item id, from its evidence.

    None when the turn keeps no text (none or metadata capture, or its content is
    gone or unreadable here): its reviews then show no note at all.
    """
    evidence = turn.evidence
    if content is None or evidence is None or evidence.capture not in {"masked", "full"}:
        return None
    return content.get("notes") or {}


def _write_verdict(
    turn: Turn,
    resolved: ResolvedPolicy | None,
    observed: list[Observed],
    matched: list[Matched],
    reviewer: str,
    *,
    close: Any = None,
    note: str = "",
) -> tuple[dict[str, Any], dict[str, str]] | None:
    """Record a person's answer; with `close`, close that review item in the same transaction.

    `note` is the closing reviewer's, kept with the evidence under the item's id.
    Returns what the evidence took (see `tasks.record`); None, with nothing
    written, when someone closed the item first.
    """
    from django.db import transaction

    from django_ai_sdk.oversight.models import ReviewItem
    from django_ai_sdk.oversight.tasks import record

    with transaction.atomic():
        if close is not None:
            claimed = ReviewItem.objects.filter(pk=close.pk, status=ReviewItem.Status.OPEN).update(
                status=close.status, resolved_at=close.resolved_at, resolution=close.resolution
            )
            if not claimed:
                return None
        notes = {str(close.pk): note} if close is not None and note else None
        return record(turn, resolved, observed, matched, reviewer=reviewer, notes=notes)


async def _aget(qs: Any, pk: str) -> Any:
    try:
        return await qs.filter(pk=pk).afirst()
    except (ValidationError, ValueError):
        return None


def _contents(turns: list[Turn]) -> dict[Any, dict[str, Any] | None]:
    from django_ai_sdk.oversight.tasks import load_evidence

    return {t.id: load_evidence(t) for t in turns}


def _content_state(turn: Turn) -> tuple[dict[str, Any] | None, str]:
    """The turn's content and what state it is in.

    "unreadable" when MASTER_KEY cannot open it: another key, none, or one that is
    set but unusable (which fails ai_sdk.oversight.E008 too).
    """
    from django_ai_sdk.oversight import vault

    evidence = turn.evidence
    if not turn.evidence_hash:
        return None, "none"
    if evidence is None or evidence.shredded_at is not None:
        expired = (
            evidence is not None and evidence.expires_at and evidence.expires_at <= timezone.now()
        )
        return None, "expired" if expired else "shredded"
    content = evidence.content
    if content is None:
        try:
            content = vault.decrypt_json(evidence.key, evidence.ciphertext, evidence.scope)
        except (vault.VaultError, ImproperlyConfigured):
            # Another MASTER_KEY, none, or one that is unusable: there, not readable here.
            return None, "unreadable"
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
    from django_ai_sdk.oversight.models import Checkpoint

    result = ledger.verify()
    checkpoint = Checkpoint.objects.order_by("-tree_size").first()
    checkpoint_ok = seal.matches(checkpoint) if checkpoint is not None else None
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
        if reader is not None and action in REPEATABLE and not reason:
            # A page polling: one record per reader, read and minute, not one per poll.
            last = (
                AccessRecord.objects.filter(
                    reader=reader,
                    purpose=str(purpose),
                    action=action,
                    turn_ref=turn_ref,
                    created_at__gte=timezone.now() - timedelta(seconds=REPEAT_SECONDS),
                )
                .order_by("-created_at")
                .first()
            )
            if last is not None and last.query == query:
                return last
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
    """One answer through `purpose`'s view.

    Review merges back what the evidence keeps of it (`texts`, by observation id):
    the reason, a `Text` answer, the names an `Entities` answerer gave, each item's
    score. Audit shows the record alone: no reason, spans or items.
    """
    lens = resolved.lenses.get(obs.lens) if resolved else None
    audit = purpose is Purpose.AUDIT
    kept: dict[str, Any] = {} if audit else (texts or {}).get(str(obs.id)) or {}
    trail = obs.trail
    said_trail = kept.get("trail")
    if isinstance(said_trail, list) and len(said_trail) == len(trail):
        trail = [{**step, "value": value} for step, value in zip(trail, said_trail, strict=True)]
    return ObservationOut(
        id=str(obs.id),
        lens=obs.lens,
        label=(lens.label or lens.name.replace("_", " ")) if lens else obs.lens,
        kind=lens.kind if lens else "",
        stage=obs.stage,
        answerer=obs.answerer,
        model=obs.model,
        value=kept.get("value", obs.value),
        confidence=obs.confidence,
        rationale=kept.get("rationale", ""),
        spans=[] if audit else obs.spans,
        items=kept.get("items") or {},
        trail=trail,
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


def _review_out(
    item: Any, row: TurnRow | None = None, *, notes: dict[str, str] | None = None
) -> ReviewOut:
    """A review item; `notes` (from the evidence) adds the reviewer's note to its resolution.

    None leaves the note out: audit, or a turn that keeps no text.
    """
    resolution = item.resolution
    if resolution is not None and notes is not None:
        resolution = {**resolution, "note": notes.get(str(item.id), "")}
    return ReviewOut(
        id=str(item.id),
        turn_id=str(item.turn_id),
        queue=item.queue,
        lens=item.lens,
        rule=item.rule,
        priority=item.priority,
        status=item.status,
        created_at=item.created_at,
        resolved_at=item.resolved_at,
        resolution=resolution,
        turn=row,
    )


__all__ = ["OversightService"]
