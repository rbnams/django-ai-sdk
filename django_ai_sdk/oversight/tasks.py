"""Background stages: reflect on a turn, seal the ledger, forget what has expired."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Any

from asgiref.sync import async_to_sync, sync_to_async
from django.utils import timezone
from django_tasks import task

from django_ai_sdk.logger import get_logger
from django_ai_sdk.oversight.stages import Matched, Observed, evaluate_rules, run_lenses
from django_ai_sdk.oversight.types import (
    Answer,
    HistoryItem,
    Source,
    Stage,
    ToolCall,
    TurnView,
)

if TYPE_CHECKING:
    from django_ai_sdk.oversight.models import Turn
    from django_ai_sdk.oversight.policy import ResolvedPolicy

logger = get_logger(__name__)


@task(queue_name="default")
def reflect_turn(turn_id: str) -> None:
    """Sync task entry point; the worker calls this."""
    async_to_sync(areflect)(turn_id)


@task(queue_name="default")
def seal_checkpoint() -> None:
    seal()


@task(queue_name="default")
def prune_expired() -> None:
    prune()


def load_evidence(turn: Turn) -> dict[str, Any] | None:
    """The turn's evidence, decrypted; None when there is none or it was shredded."""
    from django_ai_sdk.oversight import vault

    evidence = turn.evidence
    if evidence is None or evidence.shredded_at is not None:
        return None
    if evidence.content is not None:
        return evidence.content
    try:
        return vault.decrypt_json(evidence.key, evidence.ciphertext, evidence.scope)
    except Exception:
        logger.exception("Evidence for turn {} could not be decrypted", turn.id)
        return None


def view_from_turn(turn: Turn, content: dict[str, Any] | None) -> TurnView:
    """A TurnView rebuilt from the ledger: what reflect, review queues and replays judge."""
    content = content or {}
    return TurnView(
        turn_id=str(turn.id),
        channel=turn.channel,
        agent_id=turn.agent_id,
        agent_name=turn.agent_name,
        input=content.get("shown_input") or content.get("input") or "",
        output=content.get("shown_output") or content.get("output") or "",
        history=[HistoryItem.model_validate(h) for h in content.get("history") or []],
        sources=[
            Source.model_validate(s) for s in content.get("sources") or [] if isinstance(s, dict)
        ],
        tool_calls=[
            ToolCall.model_validate(c) for c in content.get("tool_calls") or [] if c.get("name")
        ],
    )


def answers_from(observations: list[Any]) -> dict[str, Answer]:
    """The latest usable answer per lens; a human's beats a machine's."""
    answers: dict[str, Answer] = {}
    for obs in sorted(observations, key=lambda o: (bool(o.reviewer), o.created_at)):
        if obs.error or obs.value is None:
            if obs.lens not in answers:
                answers[obs.lens] = Answer(error=obs.error or "no value", answerer=obs.answerer)
            continue
        answers[obs.lens] = Answer(
            value=obs.value,
            confidence=obs.confidence if obs.confidence is not None else 1.0,
            answerer=obs.answerer,
            model=obs.model,
        )
    return answers


async def areflect(turn_id: str) -> int:
    """Run the policy's reflect lenses and every pending escalation; returns answers added."""
    from django_ai_sdk.oversight import budget as budgets
    from django_ai_sdk.oversight.answerers.base import ask, resolve
    from django_ai_sdk.oversight.models import Decision, Observation, Turn
    from django_ai_sdk.oversight.policy import aget_policy
    from django_ai_sdk.oversight.session import unobserved

    turn = await Turn.objects.select_related("evidence").filter(id=turn_id).afirst()
    if turn is None:
        return 0
    policy = await aget_policy(turn.policy)
    if policy is None:
        return 0
    content = await sync_to_async(load_evidence)(turn)
    view = view_from_turn(turn, content)
    existing = [o async for o in Observation.objects.filter(turn=turn)]
    view.answers = answers_from(existing)
    lenses = policy.active_lenses(Stage.REFLECT)
    # A lens that ran out of time inline is answered here instead: late, not lost.
    timed_out = {o.lens for o in existing if o.error == "deadline"}
    lenses += [
        policy.lenses[name]
        for name in sorted(timed_out)
        if name in policy.lenses and policy.lenses[name] not in lenses
    ]

    budget = await budgets.aload()
    token = budgets.activate(budget)
    try:
        with unobserved():
            observed = await run_lenses(lenses, view, Stage.REFLECT)
            async for decision in Decision.objects.filter(turn=turn, acted=True):
                for action in decision.actions:
                    if action.get("kind") != "escalate":
                        continue
                    lens = policy.lenses.get(action.get("lens", ""))
                    if lens is None:
                        continue
                    answer = await ask(resolve(action.get("answerer") or "strong"), lens, view)
                    if answer is not None:
                        observed.append(Observed(lens, Stage.REFLECT, answer))
    finally:
        budgets.deactivate(token)
    for o in observed:
        view.answers[o.lens.name] = o.answer
    done = {
        rule async for rule in Decision.objects.filter(turn=turn).values_list("rule", flat=True)
    }
    matched = evaluate_rules(policy, view.answers, Stage.REFLECT, done=done)
    await sync_to_async(record, thread_sensitive=True)(turn, policy, observed, matched)
    await budget.flush()
    await _after_effects(turn, view, observed, matched)
    return len(observed)


def record(
    turn: Turn,
    policy: ResolvedPolicy,
    observed: list[Observed],
    matched: list[Matched],
    *,
    reviewer: str = "",
) -> None:
    """Chain later observations and decisions onto a written turn, and update its labels."""
    from django.db import transaction

    from django_ai_sdk.oversight import ledger
    from django_ai_sdk.oversight.models import Decision, Observation, ReviewItem, Turn
    from django_ai_sdk.oversight.session import observation_fields

    with transaction.atomic():
        labels = dict(turn.labels or {})
        for o in observed:
            obs = Observation.objects.create(
                turn=turn, reviewer=reviewer, **observation_fields(o.lens.name, o.stage, o.answer)
            )
            ledger.append("observation", obs)
            if o.answer.ok and o.lens.kind in {"choice", "flag", "entities", "score"}:
                labels[o.lens.name] = obs.value
            if o.answer.queued:
                ReviewItem.objects.get_or_create(
                    turn=turn, queue=o.answer.queued, lens=o.lens.name, rule=""
                )
        severity = turn.severity
        for match in matched:
            decision = Decision.objects.create(
                turn=turn,
                rule=match.rule.name,
                stage=str(match.stage),
                mode=str(match.rule.mode),
                actions=[a.to_dict() for a in match.rule.then],
                acted=bool(match.acted),
                condition=match.rule.when.to_dict(),
            )
            ledger.append("decision", decision)
            for action in match.acted:
                if action.kind == "queue":
                    ReviewItem.objects.get_or_create(
                        turn=turn,
                        queue=action.queue,
                        lens="",
                        rule=match.rule.name,
                        defaults={"priority": action.priority},
                    )
                if action.kind in {"alert", "notify"}:
                    severity = _max_severity(severity, action.severity)
        Turn.objects.filter(pk=turn.pk).update(
            labels=labels, severity=severity, reflected_at=timezone.now()
        )


def _max_severity(a: str, b: str) -> str:
    from django_ai_sdk.oversight.session import SEVERITIES

    return max(a, b, key=lambda s: SEVERITIES.index(s) if s in SEVERITIES else 0)


async def _after_effects(
    turn: Turn, view: TurnView, observed: list[Observed], matched: list[Matched]
) -> None:
    """What an enforced reflect rule can still do: mask the stored answer, notify."""
    from django_ai_sdk.oversight.answerers.pii import redact
    from django_ai_sdk.oversight.exporters import emit

    for match in matched:
        for action in match.acted:
            if action.kind == "redact" and turn.message_id:
                names = set(action.lenses) or match.rule.lenses()
                spans = [s for o in observed if o.lens.name in names for s in o.answer.spans]
                if spans:
                    await _redact_stored(
                        turn.message_id, redact(view.output, spans, action.template)
                    )
            elif action.kind == "notify":
                await emit(
                    {
                        "type": "oversight.decision",
                        "turn_id": str(turn.id),
                        "rule": match.rule.name,
                        "stage": "reflect",
                        "severity": action.severity,
                        "agent": turn.agent_name,
                        "channel": turn.channel,
                    },
                    names=action.exporters,
                )


async def _redact_stored(message_id: str, text: str) -> None:
    """Replace a stored assistant message's text after the fact."""
    from django_ai_sdk.conversation.models import Message

    message = await Message.objects.filter(id=message_id).afirst()
    if message is None or not isinstance(message.result, dict):
        return
    message.result = {**message.result, "content": text}
    await message.asave(update_fields=["result"])


def seal(*, origin: str | None = None) -> dict[str, Any] | None:
    """Checkpoint the ledger as it stands: a Merkle root, signed when a key is set.

    Synchronous. Returns the checkpoint, or None when nothing was added since the last.
    """
    from django_ai_sdk.oversight import seal as merkle
    from django_ai_sdk.oversight.conf import log_origin, setting
    from django_ai_sdk.oversight.models import Checkpoint, LedgerEntry

    last = Checkpoint.objects.order_by("-tree_size").first()
    size = LedgerEntry.objects.count()
    if size == 0 or (last is not None and last.tree_size >= size):
        return None
    leaves = [
        bytes.fromhex(h)
        for h in LedgerEntry.objects.order_by("seq").values_list("entry_hash", flat=True)[:size]
    ]
    root = merkle.merkle_root(leaves)
    body = merkle.checkpoint_body(origin or log_origin(), size, root)
    key = setting("SIGNING_KEY")
    signed = bool(key)
    note = (
        merkle.sign_note(body, name=setting("SIGNING_NAME") or log_origin(), private_key_b64=key)
        if signed
        else body
    )
    checkpoint = Checkpoint.objects.create(
        tree_size=size, root_hash=root.hex(), note=note, signed=signed
    )
    payload = {
        "type": "oversight.checkpoint",
        "tree_size": size,
        "root_hash": root.hex(),
        "note": note,
    }
    from django_ai_sdk.oversight.exporters import emit_sync

    anchor = emit_sync(payload, names=None, anchors=True)
    if anchor:
        Checkpoint.objects.filter(pk=checkpoint.pk).update(
            anchored_at=timezone.now(), anchor_ref=anchor[:2000]
        )
    return {**payload, "id": str(checkpoint.id), "signed": signed}


def prune(*, now: Any = None) -> dict[str, int]:
    """Forget what retention says to: shred expired evidence, drop expired metadata.

    Evidence expires per row; turns (and their observations) after the metadata
    period. The chain keeps their hashes, so verify reports them as pruned.
    """
    from django_ai_sdk.oversight.conf import setting
    from django_ai_sdk.oversight.models import Evidence, Turn

    now = now or timezone.now()
    expired = Evidence.objects.filter(expires_at__lte=now, shredded_at__isnull=True)
    shredded = expired.update(content=None, ciphertext=None, shredded_at=now)
    metadata_days = int((setting("RETENTION") or {}).get("metadata", 730))
    dropped, _ = Turn.objects.filter(created_at__lte=now - timedelta(days=metadata_days)).delete()
    return {"evidence_shredded": shredded, "rows_dropped": dropped}


def shred_scope(scope: str) -> bool:
    """Crypto-shred one scope (a thread): its key goes, and with it every copy's meaning.

    Chained as a "shred" entry, so forgetting is on the record too.
    """
    from django.db import transaction

    from django_ai_sdk.oversight import ledger
    from django_ai_sdk.oversight.models import DataKey, Evidence

    with transaction.atomic():
        key, _ = DataKey.objects.get_or_create(scope=scope)
        if key.shredded_at is not None:
            return False
        key.wrapped = None
        key.shredded_at = timezone.now()
        key.save(update_fields=["wrapped", "shredded_at"])
        Evidence.objects.filter(scope=scope).update(content=None, shredded_at=key.shredded_at)
        ledger.append("shred", key)
    return True


__all__ = [
    "areflect",
    "load_evidence",
    "prune",
    "prune_expired",
    "record",
    "reflect_turn",
    "seal",
    "seal_checkpoint",
    "shred_scope",
    "view_from_turn",
]
