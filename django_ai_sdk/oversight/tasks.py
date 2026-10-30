"""Background stages: reflect on a turn, seal the ledger, forget what has expired."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from asgiref.sync import async_to_sync, sync_to_async
from django.core.exceptions import ImproperlyConfigured
from django.utils import timezone
from django_tasks import task

from django_ai_sdk.logger import get_logger
from django_ai_sdk.oversight.conf import setting
from django_ai_sdk.oversight.ledger import UNHASHED
from django_ai_sdk.oversight.rules import Mask, Notify
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
    Outcome,
    Source,
    Stage,
    Subject,
    ToolCall,
    TurnView,
)

if TYPE_CHECKING:
    from django_ai_sdk.oversight.budget import Budget
    from django_ai_sdk.oversight.lenses import Lens
    from django_ai_sdk.oversight.models import Evidence, Turn
    from django_ai_sdk.oversight.policy import ResolvedPolicy
    from django_ai_sdk.oversight.record import Masker

logger = get_logger(__name__)

# The most of an anchoring exporter's reference (a URL, a receipt) a checkpoint keeps.
ANCHOR_REF_CHARS = 2000


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
    """The turn's evidence, decrypted; None when there is none, it was shredded, or it
    does not open here."""
    if turn.evidence is None:
        return None
    return open_evidence(turn.evidence, turn_id=turn.id)


def open_evidence(evidence: Evidence, *, turn_id: Any = "") -> dict[str, Any] | None:
    """`evidence`'s content, decrypted; None when shredded or it does not open here."""
    from django_ai_sdk.oversight import vault

    if evidence.shredded_at is not None:
        return None
    if evidence.content is not None:
        return evidence.content
    try:
        return vault.decrypt_json(evidence.key, evidence.ciphertext, evidence.scope)
    except vault.KeyMismatch:
        logger.error("Evidence for turn {} does not open with this MASTER_KEY", turn_id)
        return None
    except Exception:
        logger.exception("Evidence for turn {} could not be decrypted", turn_id)
        return None


def view_from_turn(turn: Turn, content: dict[str, Any] | None) -> TurnView:
    """A TurnView rebuilt from the ledger: what reflect and review queues judge."""
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
    """Run the policy's reflect lenses; returns answers added.

    The turn is claimed first, in one update: a retried or doubled task finds it
    claimed and returns 0, adding nothing twice. A run that records nothing (the
    turn could not be read, its policy could not be loaded, judging failed) gives
    its claim back, so a retry can still reflect. Once recorded, the after-effects
    (mask the stored answer, notify) run before the spend is written: a spend that
    cannot be written is logged and never undoes them, and an after-effect that
    fails is recorded on the turn's `error`.
    """
    from django_ai_sdk.oversight.models import Turn

    claimed = await Turn.objects.filter(id=turn_id, reflected_at__isnull=True).aupdate(
        reflected_at=timezone.now()
    )
    if not claimed:
        return 0
    try:
        turn = await Turn.objects.select_related("evidence").filter(id=turn_id).afirst()
        if turn is None:
            return 0
        judged = await _judge(turn)
        if judged is None:
            # Its policy did not load: give the claim back, a later run may reflect.
            await _unclaim(turn_id)
            return 0
        policy, view, observed, matched, budget = judged
        await sync_to_async(record, thread_sensitive=True)(
            turn, policy, observed, matched, view=view
        )
    except BaseException:
        # Nothing was recorded: give the claim back, so a retry can still reflect.
        await _unclaim(turn_id)
        raise
    try:
        await _after_effects(turn, view, observed, matched)
    except Exception:
        logger.exception("Reflect could not carry out what it decided on turn {}", turn.id)
        await _note_error(turn, "reflect after-effects failed")
    try:
        await budget.flush()
    except Exception:
        logger.exception("Reflect could not record spend for turn {}", turn.id)
    return len(observed)


async def _unclaim(turn_id: str) -> None:
    from django_ai_sdk.oversight.models import Turn

    await Turn.objects.filter(id=turn_id).aupdate(reflected_at=None)


async def _note_error(turn: Turn, error: str) -> None:
    """Add `error` to what went wrong in `turn`, as the session does at close."""
    from django_ai_sdk.oversight.models import Turn
    from django_ai_sdk.oversight.record import _fit

    try:
        current = (
            await Turn.objects.filter(pk=turn.pk).values_list("error", flat=True).afirst()
        ) or ""
        if error not in current.split("; "):
            chained = f"{current}; {error}" if current else error
            await Turn.objects.filter(pk=turn.pk).aupdate(error=_fit(Turn, "error", chained))
    except Exception:
        logger.exception("Reflect could not record an error on turn {}", turn.id)


async def _judge(
    turn: Turn,
) -> tuple[ResolvedPolicy, TurnView, list[Observed], list[Matched], Budget] | None:
    """Reflect's answers on `turn` and the rules they make hold; None without a policy."""
    from django_ai_sdk.oversight import budget as budgets
    from django_ai_sdk.oversight.models import Decision, Observation
    from django_ai_sdk.oversight.policy import aget_policy
    from django_ai_sdk.oversight.session import unobserved

    policy = await aget_policy(turn.policy)
    if policy is None:
        return None
    content = await sync_to_async(load_evidence)(turn)
    view = view_from_turn(turn, content)
    if content is not None and turn.evidence is not None and turn.evidence.capture == "masked":
        # The ledger keeps the turn masked; a chat thread keeps it as it was shown. Reflect
        # judges that, so a mask it asks for points into the stored answer. What it says
        # is masked, with what it found there, before it is kept.
        view.input = await _stored_text(turn.user_message_id) or view.input
        view.output = await _stored_text(turn.message_id) or view.output
    existing = [o async for o in Observation.objects.filter(turn=turn)]
    view.answers = answers_from(existing)
    has_text = content is not None and "input" in content and "output" in content
    if not has_text:
        logger.info("Turn {} kept no text; reflect judges only what it kept", turn.id)
    # The gate stopped the question: there is no answer to judge, empty as it is.
    answered = bool(view.output.strip()) or turn.outcome != Outcome.BLOCKED
    lenses = [
        lens
        for lens in policy.active_lenses(Stage.REFLECT)
        if _judgeable(lens, content, has_text, answered=answered)
    ]
    # A lens that ran out of time inline, or found TypeSafe down, is answered here: late, not lost.
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
        and _judgeable(policy.lenses[name], content, has_text, answered=answered)
    ]

    budget = await budgets.aload()
    token = budgets.activate(budget)
    try:
        with unobserved():
            observed = await run_lenses(lenses, view, Stage.REFLECT)
            # Never sampled away: they were due inline, on every turn.
            observed += await run_lenses(late, view, Stage.REFLECT, sample=False)
    finally:
        budgets.deactivate(token)
    for o in observed:
        view.answers[o.lens.name] = o.answer
    done = {
        rule async for rule in Decision.objects.filter(turn=turn).values_list("rule", flat=True)
    }
    # A lens whose latest answer failed is unknown to the rules, not "no".
    failed = {name for name, answer in view.answers.items() if answer.error and not answer.queued}
    matched = evaluate_rules(policy, view.answers, Stage.REFLECT, done=done, errored=failed)
    _drop_idle_masks(view, observed, matched)
    return policy, view, observed, matched, budget


def _drop_idle_masks(view: TurnView, observed: list[Observed], matched: list[Matched]) -> None:
    """A Mask with nothing to mask in the stored answer did not act: the record says so.

    Reflect cannot block or withhold; a mask whose lenses found something without
    saying where, or pointed outside the answer, masks nothing either.
    """
    from django_ai_sdk.oversight.answerers.pii import mask_all
    from django_ai_sdk.oversight.session import spans_for

    for match in matched:
        acted = []
        for action in match.acted:
            if isinstance(action, Mask):
                names = set(action.lenses) or match.rule.lenses()
                spans = [
                    s
                    for s in spans_for(observed, names, Subject.OUTPUT, view.input)
                    if 0 <= s.start < s.end <= len(view.output)
                ]
                if not spans or mask_all(view.output, [(spans, action.template)]) == view.output:
                    continue
            acted.append(action)
        match.acted = acted


async def _stored_text(message_id: str) -> str | None:
    """A thread message's text as stored; None when the turn has no such message."""
    from django.core.exceptions import ValidationError

    from django_ai_sdk.conversation.models import Message

    if not message_id:
        return None
    try:
        message = await Message.objects.filter(id=message_id).afirst()
    except (ValidationError, ValueError):
        return None
    result = message.result if message is not None else None
    content = result.get("content") if isinstance(result, dict) else None
    return content if isinstance(content, str) else None


def _judgeable(
    lens: Lens, content: dict[str, Any] | None, has_text: bool, *, answered: bool = True
) -> bool:
    """Whether reflect can answer `lens` from what the turn kept.

    Never a tool call lens: reflect has no call to show it, nor a lens on the
    answer when there was none to give (`answered`). Without the turn's text, only
    a sources lens, and only when the sources were kept.
    """
    if lens.subject is Subject.TOOL_CALL:
        return False
    if not answered and lens.subject in {Subject.OUTPUT, Subject.EXCHANGE}:
        return False
    if has_text:
        return True
    return lens.subject is Subject.SOURCES and bool((content or {}).get("sources"))


def record(
    turn: Turn,
    # None for a person's verdict on a turn whose policy is gone.
    policy: ResolvedPolicy | None,
    observed: list[Observed],
    matched: list[Matched],
    *,
    reviewer: str = "",
    view: TurnView | None = None,
    notes: dict[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Chain later observations and decisions onto a written turn, and update its labels.

    What can quote the turn goes into its evidence, as at close: the answers' text
    sections and `notes` (a reviewer's, by review item id). MASKED capture masks
    them, with what the lenses pointed at in `view` (the text reflect judged).
    Returns what the evidence took, (texts by observation id, notes by item id):
    nothing when the turn keeps no text, or its content is gone.
    """
    from django.db import transaction

    from django_ai_sdk.oversight import ledger
    from django_ai_sdk.oversight.models import Turn
    from django_ai_sdk.oversight.record import (
        build_observations,
        found_literals,
        record_judgments,
        without_nul,
    )
    from django_ai_sdk.oversight.types import most_severe

    found = found_literals(observed, {s: view.text(s) for s in Subject}) if view else None
    keep = keeper_for(turn, found)
    observations, texts = build_observations(observed, keep=keep, reviewer=reviewer)
    texts = without_nul(texts)
    kept_notes = (
        {key: keep.text(note) for key, note in (notes or {}).items() if note} if keep else {}
    )
    with transaction.atomic():
        if (texts or kept_notes) and not add_to_evidence(turn, texts=texts, notes=kept_notes):
            texts, kept_notes = {}, {}
        if not observed and not matched:
            return texts, kept_notes
        labels, severity, chained = record_judgments(turn, observations, observed, matched)
        # Read again under a lock: a person's verdict and reflect may record at once.
        current = Turn.objects.select_for_update().only("labels", "severity").get(pk=turn.pk)
        Turn.objects.filter(pk=turn.pk).update(
            labels={**(current.labels or {}), **labels},
            severity=most_severe(current.severity, severity),
        )
        ledger.append_many(chained)
    return texts, kept_notes


def keeper_for(turn: Turn, found: dict[str, str] | None = None) -> Masker | None:
    """How text added later is kept with `turn`'s evidence; None when it keeps none.

    As at close: by the capture the evidence was written at, and not at all once it
    was shredded, expired, or never written.
    """
    from django_ai_sdk.oversight.record import keeper

    evidence = turn.evidence
    if evidence is None or evidence.shredded_at is not None:
        return None
    return keeper(evidence.capture, found)


def add_to_evidence(
    turn: Turn, *, texts: dict[str, Any] | None = None, notes: dict[str, str] | None = None
) -> bool:
    """Add answers' text sections and reviewers' notes to the turn's evidence.

    Sealed again under the same scope's key, the row locked so a verdict and reflect
    adding at once both land. False, with nothing written, when the evidence is gone,
    does not open here, or cannot be sealed (forgotten since, or MASTER_KEY unusable).
    """
    from django.db import transaction

    from django_ai_sdk.oversight import vault
    from django_ai_sdk.oversight.models import Evidence

    if turn.evidence_id is None:
        return False
    with transaction.atomic():
        evidence = Evidence.objects.select_for_update().filter(pk=turn.evidence_id).first()
        content = open_evidence(evidence, turn_id=turn.id) if evidence else None
        if evidence is None or content is None:
            return False
        content = {
            **content,
            "texts": {**(content.get("texts") or {}), **(texts or {})},
            "notes": {**(content.get("notes") or {}), **(notes or {})},
        }
        content = {k: v for k, v in content.items() if v != {} or k not in UNHASHED}
        try:
            sealed = vault.encrypt_json(evidence.scope, content)
        except (vault.VaultError, ImproperlyConfigured):
            return False
        Evidence.objects.filter(pk=evidence.pk).update(
            key=sealed[0] if sealed else None,
            ciphertext=sealed[1] if sealed else None,
            content=None if sealed else content,
        )
    return True


async def _after_effects(
    turn: Turn, view: TurnView, observed: list[Observed], matched: list[Matched]
) -> None:
    """What an enforced reflect rule can still do: mask the stored answer, notify."""
    from django_ai_sdk.oversight.exporters import emit
    from django_ai_sdk.oversight.session import spans_for

    masks: list[tuple[list[Any], str]] = []
    rules: list[str] = []
    for match in matched:
        for action in match.acted:
            if isinstance(action, Mask):
                names = set(action.lenses) or match.rule.lenses()
                spans = [
                    s
                    for s in spans_for(observed, names, Subject.OUTPUT, view.input)
                    if 0 <= s.start < s.end <= len(view.output)
                ]
                if spans:
                    masks.append((spans, action.template))
                    rules.append(match.rule.name)
            elif isinstance(action, Notify):
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
    if masks and turn.message_id:
        await _mask_stored(turn, view.output, masks, rules)


async def _mask_stored(
    turn: Turn, judged: str, masks: list[tuple[list[Any], str]], rules: list[str]
) -> None:
    """Mask the stored answer after the fact, with a notice a reloaded thread shows.

    Spans index the text reflect judged, so they apply only when the stored text is
    that same text. They cannot apply to the reasoning, which is dropped.
    """
    from django_ai_sdk.conversation.models import Message
    from django_ai_sdk.oversight.answerers.pii import mask_all

    message = await Message.objects.filter(id=turn.message_id).afirst()
    if message is None or not isinstance(message.result, dict):
        return
    if message.result.get("content") != judged:
        logger.warning("Turn {}: stored text differs from the evidence; not masked", turn.id)
        return
    metadata = dict(message.result.get("metadata") or {})
    notice: dict[str, Any] = {
        "turn_id": str(turn.id),
        "stage": "reflect",
        "outcome": str(Outcome.MASKED),
        "replaced": True,
    }
    # Whoever reloads the thread reads this: rule names only for a developer view.
    if setting("CLIENT_DETAIL") == "full":
        notice["rules"] = rules
    metadata["oversight"] = [*(metadata.get("oversight") or []), notice]
    message.result = {
        **message.result,
        "content": mask_all(judged, masks),
        "reasoning": "",
        "metadata": metadata,
    }
    await message.asave(update_fields=["result"])


def seal(*, origin: str | None = None) -> dict[str, Any] | None:
    """Checkpoint the ledger as it stands: a Merkle root, signed when a key is set.

    Synchronous. Returns the checkpoint, or None when nothing was added since the last.
    A checkpoint another sealer wrote at the same moment is returned, not sent again.
    """
    from django_ai_sdk.oversight import seal as merkle
    from django_ai_sdk.oversight.conf import log_origin, signing_name
    from django_ai_sdk.oversight.exporters import emit
    from django_ai_sdk.oversight.models import Checkpoint

    found = merkle.write_checkpoint(
        origin=origin or log_origin(), name=signing_name(), key=setting("SIGNING_KEY")
    )
    if found is None:
        return None
    checkpoint, created = found
    payload = {
        "type": "oversight.checkpoint",
        "tree_size": checkpoint.tree_size,
        "root_hash": checkpoint.root_hash,
        "note": checkpoint.note,
    }
    result = {**payload, "id": str(checkpoint.id), "signed": checkpoint.signed}
    if not created:
        return result
    anchor = async_to_sync(emit)(payload, anchors=True)
    if anchor:
        Checkpoint.objects.filter(pk=checkpoint.pk).update(
            anchored_at=timezone.now(), anchor_ref=anchor[:ANCHOR_REF_CHARS]
        )
    return result


def prune(*, now: Any = None) -> dict[str, int]:
    """Forget what retention says to: shred expired evidence, drop expired metadata.

    Evidence and turns (with their observations) each expire per row, by their
    policy's retention. Expired evidence takes what judges and reviewers wrote
    with it. The chain keeps the turns' hashes, so verify reports them as pruned.
    """
    from django.db import transaction

    from django_ai_sdk.oversight import ledger
    from django_ai_sdk.oversight.models import Evidence, Turn

    now = now or timezone.now()
    expired = Evidence.objects.filter(expires_at__lte=now, shredded_at__isnull=True)
    shredded = expired.update(content=None, ciphertext=None, shredded_at=now)
    with transaction.atomic():
        dropped = ledger.prune_turns(Turn.objects.filter(expires_at__lte=now))
    return {"evidence_shredded": shredded, "rows_dropped": dropped}


def shred_scope(scope: str) -> bool:
    """Crypto-shred one scope (a thread): its key goes, and with it every copy's meaning.

    What judges and reviewers wrote lives in the evidence and goes with it; who
    asked goes from the thread's turns too, which keep only a pseudonym. Chained
    as a "shred" entry, so forgetting is on the record too.
    """
    from django.db import transaction
    from django.db.models import Q

    from django_ai_sdk.oversight import ledger
    from django_ai_sdk.oversight.models import DataKey, Evidence, Turn

    with transaction.atomic():
        key, _ = DataKey.objects.get_or_create(scope=scope)
        if key.shredded_at is not None:
            return False
        key.wrapped = None
        key.shredded_at = timezone.now()
        key.save(update_fields=["wrapped", "shredded_at"])
        Evidence.objects.filter(scope=scope).update(
            content=None, ciphertext=None, shredded_at=key.shredded_at
        )
        # Not hashed: the chain still verifies.
        turns = Q(evidence__scope=scope)
        if scope.startswith("thread:"):
            turns |= Q(thread_id=scope.removeprefix("thread:"))
        Turn.objects.filter(turns).exclude(actor_ref="").update(actor_ref="")
        ledger.append("shred", key)
    return True


__all__ = [
    "add_to_evidence",
    "areflect",
    "export_events",
    "keeper_for",
    "load_evidence",
    "open_evidence",
    "prune",
    "record",
    "reflect_turn",
    "seal",
    "shred_scope",
    "view_from_turn",
]
