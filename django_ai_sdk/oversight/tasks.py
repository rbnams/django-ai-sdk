"""Background stages: reflect on a turn, seal the ledger, forget what has expired."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from asgiref.sync import async_to_sync, sync_to_async
from django.core.exceptions import ImproperlyConfigured
from django.utils import timezone
from django_tasks import task

from django_ai_sdk.logger import get_logger
from django_ai_sdk.oversight.stages import (
    Matched,
    Observed,
    asks_again,
    evaluate_rules,
    run_lenses,
)
from django_ai_sdk.oversight.types import (
    Answer,
    HistoryItem,
    Source,
    Stage,
    Subject,
    ToolCall,
    TurnView,
)

if TYPE_CHECKING:
    from django_ai_sdk.oversight.lenses import Lens
    from django_ai_sdk.oversight.models import Turn
    from django_ai_sdk.oversight.policy import ResolvedPolicy

logger = get_logger(__name__)


@task(queue_name="default")
def reflect_turn(turn_id: str) -> None:
    """Sync task entry point; the worker calls this."""
    async_to_sync(areflect)(turn_id)


@task(queue_name="default")
def export_events(events: list[list[Any]]) -> None:
    """Send a turn's notifications, [event, exporter names] each, off the chat stream."""
    from django_ai_sdk.oversight.exporters import emit

    for event, names in events:
        async_to_sync(emit)(event, names=names)


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
    """Run the policy's reflect lenses and every pending escalation; returns answers added.

    A turn reflected before returns 0: a retried task adds nothing twice.
    """
    from django_ai_sdk.oversight import budget as budgets
    from django_ai_sdk.oversight.models import Decision, Observation, Turn
    from django_ai_sdk.oversight.policy import aget_policy
    from django_ai_sdk.oversight.session import unobserved

    turn = await Turn.objects.select_related("evidence").filter(id=turn_id).afirst()
    if turn is None or turn.reflected_at is not None:
        return 0
    policy = await aget_policy(turn.policy)
    if policy is None:
        return 0
    content = await sync_to_async(load_evidence)(turn)
    view = view_from_turn(turn, content)
    existing = [o async for o in Observation.objects.filter(turn=turn)]
    view.answers = answers_from(existing)
    has_text = content is not None and "input" in content and "output" in content
    if not has_text:
        logger.info("Turn {} kept no text; reflect judges only what it kept", turn.id)
    lenses = [
        lens for lens in policy.active_lenses(Stage.REFLECT) if _judgeable(lens, content, has_text)
    ]
    # A lens that ran out of time inline, or found Jev down, is answered here: late, not lost.
    timed_out = {
        o.lens
        for o in existing
        if asks_again(o.error, (o.trail[-1] or {}).get("error", "") if o.trail else "")
    }
    late = [
        policy.lenses[name]
        for name in sorted(timed_out)
        if name in policy.lenses
        and policy.lenses[name] not in lenses
        and _judgeable(policy.lenses[name], content, has_text)
    ]

    budget = await budgets.aload()
    token = budgets.activate(budget)
    try:
        with unobserved():
            observed = await run_lenses(lenses, view, Stage.REFLECT)
            # Never sampled away: they were due inline, on every turn.
            observed += await run_lenses(late, view, Stage.REFLECT, sample=False)
            if has_text:
                observed += await _escalations(turn, policy, view)
    finally:
        budgets.deactivate(token)
    for o in observed:
        view.answers[o.lens.name] = o.answer
    done = {
        rule async for rule in Decision.objects.filter(turn=turn).values_list("rule", flat=True)
    }
    matched = evaluate_rules(policy, view.answers, Stage.REFLECT, done=done)
    await sync_to_async(record, thread_sensitive=True)(
        turn, policy, observed, matched, reflected=True
    )
    await budget.flush()
    await _after_effects(turn, view, observed, matched)
    return len(observed)


def _judgeable(lens: Lens, content: dict[str, Any] | None, has_text: bool) -> bool:
    """Whether reflect can answer `lens` from what the turn kept.

    Never a tool call lens: reflect has no call to show it. Without the turn's
    text, only a sources lens, and only when the sources were kept.
    """
    if lens.subject is Subject.TOOL_CALL:
        return False
    if has_text:
        return True
    return lens.subject is Subject.SOURCES and bool((content or {}).get("sources"))


async def _escalations(turn: Turn, policy: ResolvedPolicy, view: TurnView) -> list[Observed]:
    """Ask every enforced Escalate's stronger answerer about its lens."""
    from django_ai_sdk.oversight.answerers.base import ask, resolve
    from django_ai_sdk.oversight.models import Decision

    observed: list[Observed] = []
    async for decision in Decision.objects.filter(turn=turn, acted=True):
        for action in decision.actions:
            if action.get("kind") != "escalate":
                continue
            lens = policy.lenses.get(action.get("lens", ""))
            if lens is None:
                continue
            name = action.get("answerer") or "strong"
            try:
                answer = await ask(resolve(name), lens, view)
            except ImproperlyConfigured:
                logger.warning("Escalation to {!r} for turn {}: no such answerer", name, turn.id)
                answer = Answer(error="unknown answerer", answerer=name)
            if answer is not None:
                observed.append(Observed(lens, Stage.REFLECT, answer))
    return observed


def record(
    turn: Turn,
    policy: ResolvedPolicy,
    observed: list[Observed],
    matched: list[Matched],
    *,
    reviewer: str = "",
    reflected: bool = False,
) -> None:
    """Chain later observations and decisions onto a written turn, and update its labels.

    `reflected` marks the turn as reflected on: only reflect itself says so.
    """
    from django.db import transaction

    from django_ai_sdk.oversight import ledger
    from django_ai_sdk.oversight.models import Turn
    from django_ai_sdk.oversight.session import record_judgments, text_values
    from django_ai_sdk.oversight.types import most_severe

    evidence = turn.evidence
    # Like at close: reasons and texts are kept only where the turn's text is.
    reasons = (
        evidence is not None
        and evidence.shredded_at is None
        and evidence.capture not in {"none", "metadata"}
    )
    with transaction.atomic():
        texts = text_values(observed) if reasons else {}
        if texts:
            _add_texts(turn, texts)
        labels, severity, chained = record_judgments(
            turn, observed, matched, reasons=reasons, reviewer=reviewer
        )
        # Read again under a lock: a person's verdict and reflect may record at once.
        current = Turn.objects.select_for_update().only("labels", "severity").get(pk=turn.pk)
        update: dict[str, Any] = {
            "labels": {**(current.labels or {}), **labels},
            "severity": most_severe(current.severity, severity),
        }
        if reflected:
            update["reflected_at"] = timezone.now()
        Turn.objects.filter(pk=turn.pk).update(**update)
        ledger.append_many(chained)


def _add_texts(turn: Turn, texts: dict[str, Any]) -> None:
    """Add Text lens answers to the turn's evidence, sealed under the same key."""
    from django_ai_sdk.oversight import vault
    from django_ai_sdk.oversight.models import Evidence

    content = load_evidence(turn)
    if content is None:
        return
    content = {**content, "texts": {**(content.get("texts") or {}), **texts}}
    evidence = turn.evidence
    sealed = vault.encrypt_json(evidence.scope, content)
    Evidence.objects.filter(pk=evidence.pk).update(
        key=sealed[0] if sealed else None,
        ciphertext=sealed[1] if sealed else None,
        content=None if sealed else content,
    )


async def _after_effects(
    turn: Turn, view: TurnView, observed: list[Observed], matched: list[Matched]
) -> None:
    """What an enforced reflect rule can still do: mask the stored answer, notify."""
    from django_ai_sdk.oversight.exporters import emit
    from django_ai_sdk.oversight.session import spans_for

    redactions: list[tuple[list[Any], str]] = []
    rules: list[str] = []
    for match in matched:
        for action in match.acted:
            if action.kind == "redact":
                names = set(action.lenses) or match.rule.lenses()
                spans = spans_for(observed, names, Subject.OUTPUT, view.input)
                if spans:
                    redactions.append((spans, action.template))
                    rules.append(match.rule.name)
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
    if redactions and turn.message_id:
        await _redact_stored(turn, view.output, redactions, rules)


async def _redact_stored(
    turn: Turn, judged: str, redactions: list[tuple[list[Any], str]], rules: list[str]
) -> None:
    """Mask the stored answer after the fact, with a notice a reloaded thread shows.

    Spans index the text reflect judged, so they apply only when the stored text is
    that same text. They cannot apply to the reasoning, which is dropped.
    """
    from django_ai_sdk.conversation.models import Message
    from django_ai_sdk.oversight.answerers.pii import redact_all

    message = await Message.objects.filter(id=turn.message_id).afirst()
    if message is None or not isinstance(message.result, dict):
        return
    if message.result.get("content") != judged:
        logger.warning("Turn {}: stored text differs from the evidence; not redacted", turn.id)
        return
    metadata = dict(message.result.get("metadata") or {})
    notice = {
        "turn_id": str(turn.id),
        "stage": "reflect",
        "outcome": "redacted",
        "rules": rules,
        "replaced": True,
    }
    metadata["oversight"] = [*(metadata.get("oversight") or []), notice]
    message.result = {
        **message.result,
        "content": redact_all(judged, redactions),
        "reasoning": "",
        "metadata": metadata,
    }
    await message.asave(update_fields=["result"])


def seal(*, origin: str | None = None) -> dict[str, Any] | None:
    """Checkpoint the ledger as it stands: a Merkle root, signed when a key is set.

    Synchronous. Returns the checkpoint, or None when nothing was added since the last.
    """
    from django_ai_sdk.oversight import seal as merkle
    from django_ai_sdk.oversight.conf import log_origin, setting, signing_name
    from django_ai_sdk.oversight.exporters import emit
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
    note = merkle.sign_note(body, name=signing_name(), private_key_b64=key) if signed else body
    checkpoint = Checkpoint.objects.create(
        tree_size=size, root_hash=root.hex(), note=note, signed=signed
    )
    payload = {
        "type": "oversight.checkpoint",
        "tree_size": size,
        "root_hash": root.hex(),
        "note": note,
    }
    anchor = async_to_sync(emit)(payload, anchors=True)
    if anchor:
        Checkpoint.objects.filter(pk=checkpoint.pk).update(
            anchored_at=timezone.now(), anchor_ref=anchor[:2000]
        )
    return {**payload, "id": str(checkpoint.id), "signed": signed}


def prune(*, now: Any = None) -> dict[str, int]:
    """Forget what retention says to: shred expired evidence, drop expired metadata.

    Evidence and turns (with their observations) each expire per row, by their
    policy's retention. The chain keeps their hashes, so verify reports them as pruned.
    """
    from django_ai_sdk.oversight.models import Evidence, Observation, Turn

    now = now or timezone.now()
    expired = Evidence.objects.filter(expires_at__lte=now, shredded_at__isnull=True)
    # A rationale can quote the turn; it is not hashed, so clearing it keeps the chain.
    Observation.objects.filter(turn__evidence__in=expired).update(rationale="")
    shredded = expired.update(content=None, ciphertext=None, shredded_at=now)
    dropped, _ = Turn.objects.filter(expires_at__lte=now).delete()
    return {"evidence_shredded": shredded, "rows_dropped": dropped}


def shred_scope(scope: str) -> bool:
    """Crypto-shred one scope (a thread): its key goes, and with it every copy's meaning.

    Chained as a "shred" entry, so forgetting is on the record too.
    """
    from django.db import transaction

    from django_ai_sdk.oversight import ledger
    from django_ai_sdk.oversight.models import DataKey, Evidence, Observation

    with transaction.atomic():
        key, _ = DataKey.objects.get_or_create(scope=scope)
        if key.shredded_at is not None:
            return False
        key.wrapped = None
        key.shredded_at = timezone.now()
        key.save(update_fields=["wrapped", "shredded_at"])
        Evidence.objects.filter(scope=scope).update(content=None, shredded_at=key.shredded_at)
        # Not hashed, so clearing what judges quoted leaves the chain intact.
        Observation.objects.filter(turn__evidence__scope=scope).update(rationale="")
        ledger.append("shred", key)
    return True


__all__ = [
    "areflect",
    "export_events",
    "load_evidence",
    "prune",
    "record",
    "reflect_turn",
    "seal",
    "shred_scope",
    "view_from_turn",
]
