"""Writing a closed turn: its evidence, the turn row, answers and decisions, chained.

TurnSession.close() hands the session to `_persist`, in one transaction; reflect and
a person's verdict add their answers later through `tasks.record`.

Everything that can quote the turn lives with its evidence: judges' reasons, `Text`
answers, what an `Entities` answerer named in its own words, each item's score by
its key (a source's URL), reviewers' notes. The rows the chain hashes (turn,
observation, decision) keep values, labels, scores, counts and errors. So capture
decides what of that text is kept, MASTER_KEY seals it with the evidence, and
forgetting the thread or expiring the content destroys it.
"""

from __future__ import annotations

import re
import secrets
import time
import uuid
from functools import cache
from typing import TYPE_CHECKING, Any

from django.core.exceptions import ImproperlyConfigured
from django.utils import timezone

from django_ai_sdk.logger import get_logger
from django_ai_sdk.oversight.answerers.pii import mask_all
from django_ai_sdk.oversight.lenses import Entities
from django_ai_sdk.oversight.rules import Alert, Notify, Queue
from django_ai_sdk.oversight.stages import asks_again
from django_ai_sdk.oversight.types import (
    Answer,
    Capture,
    Outcome,
    Severity,
    Source,
    Stage,
    Subject,
    TurnView,
    most_severe,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from django_ai_sdk.oversight.lenses import Lens
    from django_ai_sdk.oversight.models import Observation
    from django_ai_sdk.oversight.session import TurnSession
    from django_ai_sdk.oversight.stages import Matched, Observed

logger = get_logger(__name__)

# Random bytes salting each turn's evidence, so its hash says nothing of what it holds.
SALT_BYTES = 16
# What the record keeps of a name an `Entities` answerer gave in its own words: one per
# name, so `.any` and counts still hold, while the name itself stays with the evidence.
FOUND = "[found]"
# The shortest stretch MASKED capture masks wherever it occurs: shorter ones are words.
MIN_LITERAL = 3


def _source(item: Any) -> Source:
    if isinstance(item, Source):
        return item
    if isinstance(item, dict):
        return Source.model_validate({k: v for k, v in item.items() if k in Source.model_fields})
    return Source(
        key=str(getattr(item, "key", "") or ""),
        title=str(getattr(item, "title", "") or ""),
        url=str(getattr(item, "url", "") or ""),
        content=str(getattr(item, "content", "") or ""),
        kind=str(getattr(item, "kind", "") or ""),
        doc_id=str(getattr(item, "doc_id", "") or ""),
        chunk_id=str(getattr(item, "chunk_id", "") or ""),
        memory_id=str(getattr(item, "memory_id", "") or ""),
    )


def evidence_content(
    session: TurnSession, capture: Capture, masker: Masker | None = None
) -> dict[str, Any] | None:
    """What of the turn is kept, at `capture`. Salted, so its hash says nothing.

    MASKED runs `masker` (by default the turn's own, `turn_masker`) over every string
    it keeps: the question, the answer, the history, every field of the sources and
    of the tool calls, their arguments and results.
    """
    view = session.view
    if capture is Capture.NONE:
        return None
    base: dict[str, Any] = {
        "salt": secrets.token_hex(SALT_BYTES),
        "agent": {"id": view.agent_id, "name": view.agent_name},
        # Lens spans point into the text as said: only FULL keeps that text.
        "capture": str(capture),
    }
    if capture is Capture.METADATA:
        # No text at all: a title or a url can name who or what the turn was about.
        return {
            **base,
            "input_chars": len(session.original_input or view.input),
            "output_chars": len(session.original_output or view.output),
            "sources": [{"kind": s.kind, "seeded": s.seeded} for s in view.sources],
            "tool_calls": [{"name": c.name, "error": c.error} for c in view.tool_calls],
        }
    full = capture is Capture.FULL
    keep = Masker(active=False) if full else masker or turn_masker(session)
    return {
        **base,
        # What was passed on: the question as the gate left it, the answer as finish did.
        "input": session.original_input if full else keep.text(view.input),
        "output": (session.original_output or view.output) if full else keep.text(view.output),
        # What was actually shown or stored, when it differs from what was said.
        "shown_input": view.input if full and view.input != session.original_input else None,
        "shown_output": view.output
        if full and session.original_output and view.output != session.original_output
        else None,
        "history": keep([h.model_dump() for h in view.history]),
        "sources": keep([s.model_dump() for s in view.sources]),
        "tool_calls": keep([c.model_dump(mode="json") for c in view.tool_calls]),
    }


class Masker:
    """What MASKED capture runs over every string it keeps.

    The built-in detectors first (every `PII` kind, the country kinds too, and
    `Secrets`), then each stretch of text a lens of the turn pointed at, wherever
    it occurs, longest first, as its label: `[EMAIL]`. `active=False` keeps text
    as it is: FULL capture.
    """

    def __init__(self, found: Mapping[str, str] | None = None, *, active: bool = True) -> None:
        self.active = active
        self.found = {
            literal: label
            for literal, label in (found or {}).items()
            if len(literal.strip()) >= MIN_LITERAL
        }
        ordered = sorted(self.found, key=len, reverse=True)
        self._pattern = re.compile("|".join(map(re.escape, ordered))) if ordered else None

    def text(self, text: str) -> str:
        if not self.active or not text:
            return text
        text = mask_detected(text)
        if self._pattern is None:
            return text
        return self._pattern.sub(lambda m: f"[{(self.found[m.group()] or 'masked').upper()}]", text)

    def __call__(self, value: Any) -> Any:
        """`value` with every string masked, at any depth of a JSON value."""
        if not self.active:
            return value
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, dict):
            return {k: self(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self(v) for v in value]
        return value

    def keys(self, value: dict[str, float]) -> dict[str, float]:
        """`value` with its keys masked: a source's key can be its URL."""
        masked: dict[str, float] = {}
        for key, score in value.items():
            name = self.text(key)
            masked[name] = max(score, masked.get(name, score))
        return masked


def keeper(capture: Capture | str, found: Mapping[str, str] | None = None) -> Masker | None:
    """How what can quote the turn is kept at `capture`; None keeps none of it.

    FULL keeps it as it is, MASKED masks it (`found`: what the turn's lenses
    pointed at), NONE and METADATA keep no text at all.
    """
    if capture == Capture.FULL:
        return Masker(active=False)
    if capture == Capture.MASKED:
        return Masker(found)
    return None


def turn_masker(session: TurnSession) -> Masker:
    """The masker for this turn's content: detectors, and what its lenses found where."""
    view = session.view
    said_output = session.original_output or view.output
    texts = {
        # The gate's lenses read the question as asked; the gate masks it after.
        Subject.INPUT: session.original_input or view.input,
        Subject.OUTPUT: said_output,
        Subject.EXCHANGE: f"{view.input}\n\n{said_output}",
        Subject.SOURCES: "\n\n".join(s.content for s in view.fresh_sources()),
    }
    return Masker(found_literals(session.observed, texts))


def found_literals(observed: Iterable[Observed], texts: Mapping[Subject, str]) -> dict[str, str]:
    """Each stretch of text an `Entities` lens pointed at, with its label: what MASKED
    masks everywhere.

    Only entities: what they find (a name, an address) is what the record must not
    keep. A topic or keyword lens points at ordinary words, which stay readable.
    `texts` holds the text each subject's lenses read; a span that does not fit it
    (a tool call's, whose text is not kept) is left out.
    """
    found: dict[str, str] = {}
    for o in observed:
        if o.lens.kind != "entities":
            continue
        text = texts.get(o.lens.subject, "")
        for span in o.answer.spans:
            if 0 <= span.start < span.end <= len(text):
                found.setdefault(text[span.start : span.end], span.label or o.lens.name)
    return found


def mask_detected(value: Any) -> Any:
    """`value` with what the built-in detectors find (personal data, secrets) masked.

    In every string, at any depth of a JSON value. Every `PII` kind, the country
    kinds too: what MASKED capture keeps is masked for any country.
    """
    if isinstance(value, str):
        if not value:
            return value
        from django_ai_sdk.oversight.answerers.rules import scan

        view = TurnView()
        spans = [
            span
            for detector in _detectors()
            for span in scan(value, lambda chunk, d=detector: d.detect(chunk, view, _CAPTURED))
        ]
        return mask_all(value, [(spans, "[{label}]")]) if spans else value
    if isinstance(value, dict):
        return {k: mask_detected(v) for k, v in value.items()}
    if isinstance(value, list):
        return [mask_detected(v) for v in value]
    return value


@cache
def _detectors() -> tuple[Any, ...]:
    from django_ai_sdk.oversight.answerers.pii import KINDS, PII, Secrets

    return PII(kinds=list(KINDS)), Secrets()


# The lens the built-in detectors answer for `mask_detected`, outside any policy.
_CAPTURED = Entities(name="captured")


def _persist(session: TurnSession) -> str:
    """Write evidence, turn, observations and decisions, chained, in one transaction.

    A forgotten thread keeps no new content, nor what judges said about it, nor who
    asked. Content that cannot be sealed as configured (an unusable MASTER_KEY) is
    left out too, never stored in the clear instead; the turn and its decisions are
    still written.
    """
    from datetime import timedelta

    from django.db import transaction

    from django_ai_sdk.oversight import ledger, vault
    from django_ai_sdk.oversight.models import Evidence, Turn

    assert session.policy is not None
    policy = session.policy
    now = timezone.now()
    created = session.created_at or now
    masker = turn_masker(session) if policy.capture is Capture.MASKED else None
    # A NUL is no text a database column takes (PostgreSQL refuses it): never kept.
    content = without_nul(evidence_content(session, policy.capture, masker))
    scope = f"thread:{session.thread_id}" if session.thread_id else f"turn:{session.id}"
    keep = (masker or keeper(policy.capture)) if content is not None else None
    observations, texts = build_observations(session.observed, keep=keep)
    texts = without_nul(texts)

    with transaction.atomic():
        forgotten = vault.forgotten(scope)
        evidence = None
        evidence_hash = ""
        if content is not None:
            evidence_hash = ledger.evidence_digest(content)
            stored = {**content, "texts": texts} if texts else content
            sealed = None
            try:
                sealed = None if forgotten else vault.encrypt_json(scope, stored)
            except vault.Forgotten:
                forgotten = True
            except (ImproperlyConfigured, vault.VaultError):
                logger.exception(
                    "Turn {} is recorded without its content: it cannot be sealed", session.id
                )
                content = None
            # What judges said went into `stored` only: with no evidence, it is not kept.
            if content is not None and not forgotten:
                evidence = Evidence.objects.create(
                    digest=evidence_hash,
                    scope=scope,
                    key=sealed[0] if sealed else None,
                    ciphertext=sealed[1] if sealed else None,
                    content=None if sealed else stored,
                    capture=str(policy.capture),
                    expires_at=now + timedelta(days=policy.retention["content"]),
                )
        turn = Turn.objects.create(
            id=session.id,
            created_at=created,
            expires_at=created + timedelta(days=policy.retention["metadata"]),
            channel=_fit(Turn, "channel", session.channel),
            agent_id=_fit(Turn, "agent_id", session.view.agent_id),
            agent_name=_fit(Turn, "agent_name", session.view.agent_name),
            agent_fingerprint=session.agent_fingerprint,
            policy=policy.name,
            thread_id=session.thread_id,
            user_message_id=session.user_message_id,
            message_id=session.message_id,
            parent_turn_id=_uuid(session.parent_turn_id),
            actor_pseudonym=session.actor,
            actor_ref="" if forgotten else _actor_ref(session),
            model=_fit(Turn, "model", session.usage.model),
            prompt_tokens=session.usage.prompt_tokens,
            completion_tokens=session.usage.completion_tokens,
            cost=_cost(session),
            latency_ms=round((time.perf_counter() - session._started) * 1000, 1),
            outcome=str(session.outcome),
            report_outcome=str(session.report_outcome),
            error=_fit(Turn, "error", session.error),
            evidence=evidence,
            evidence_hash=evidence_hash,
        )
        labels, severity, chained = record_judgments(
            turn, observations, session.observed, session.matched
        )
        if session.outcome is Outcome.BLOCKED:
            severity = most_severe(severity, Severity.HIGH)
        Turn.objects.filter(pk=turn.pk).update(labels=labels, severity=severity)
        ledger.append_many([("turn", turn), *chained])
        if _needs_reflect(session):
            transaction.on_commit(lambda: _enqueue_reflect(turn.id))
        if events := session.notifications():
            transaction.on_commit(lambda: _enqueue_export(events))
    return str(turn.id)


def _actor_ref(session: TurnSession) -> str:
    """Who asked, sealed under the actor's own key; "" when forgotten or not sealable."""
    from django_ai_sdk.oversight import vault

    if not session.actor_pk:
        return ""
    try:
        return vault.seal_text(f"actor:{session.actor}", session.actor_pk)
    except vault.Forgotten:
        return ""
    except (ImproperlyConfigured, vault.VaultError):
        logger.exception("Turn {} is recorded without who asked: it cannot be sealed", session.id)
        return ""


def build_observations(
    observed: Iterable[Observed], *, keep: Masker | None, reviewer: str = ""
) -> tuple[list[Observation], dict[str, dict[str, Any]]]:
    """The Observation rows of `observed`, unsaved, and what of them goes with the evidence.

    The second is keyed by each row's id: its `texts` section. `keep` (see
    `keeper`) decides what of that is kept, and how; None keeps none of it.
    """
    from django_ai_sdk.oversight.models import Observation

    observations: list[Observation] = []
    texts: dict[str, dict[str, Any]] = {}
    for o in observed:
        columns, kept = split_answer(o.lens, o.stage, o.answer, keep=keep)
        row = Observation(reviewer=reviewer, **columns)
        observations.append(row)
        if kept:
            texts[str(row.id)] = kept
    return observations, texts


def record_judgments(
    turn: Any,
    observations: list[Observation],
    observed: list[Observed],
    matched: list[Matched],
) -> tuple[dict[str, Any], str, list[tuple[str, Any]]]:
    """Write `observations` (from `build_observations(observed)`), decisions and reviews of `turn`.

    Sync, in a transaction. Returns the labels the answers set, the severity the
    enforced alerts and notifications ask for, and the rows to chain, in order, for
    `ledger.append_many`.
    """
    from django_ai_sdk.oversight.models import Decision, Observation, ReviewItem

    for row in observations:
        row.turn = turn
    observations = Observation.objects.bulk_create(observations)
    decisions = Decision.objects.bulk_create(
        Decision(
            turn=turn,
            rule=match.rule.name,
            stage=str(match.stage),
            mode=str(match.rule.mode),
            actions=[a.to_dict() for a in match.rule.then],
            acted=bool(match.acted),
            condition=match.rule.when.to_dict(),
        )
        for match in matched
    )
    acted = [(match, action) for match in matched for action in match.acted]
    reviews = {(o.answer.queued, o.lens.name, "") for o in observed if o.answer.queued}
    reviews |= {(a.queue, "", m.rule.name) for m, a in acted if isinstance(a, Queue)}
    priority = {a.queue: a.priority for m, a in reversed(acted) if isinstance(a, Queue)}
    # A retried reflect or a second verdict may ask for a review that is already open.
    ReviewItem.objects.bulk_create(
        [
            ReviewItem(
                turn=turn, queue=queue, lens=lens, rule=rule, priority=priority.get(queue, 0)
            )
            for queue, lens, rule in sorted(reviews)
        ],
        ignore_conflicts=True,
    )
    # Lenses only: a person's call on a rule ("rule:<name>") is not a label.
    labels = {
        obs.lens: obs.value
        for o, obs in zip(observed, observations, strict=True)
        if o.answer.ok
        and o.lens.kind in {"choice", "flag", "entities", "score"}
        and not o.lens.name.startswith("rule:")
    }
    severity = most_severe(*(a.severity for m, a in acted if isinstance(a, Alert | Notify)))
    chained = [("observation", o) for o in observations] + [("decision", d) for d in decisions]
    return labels, severity, chained


def split_answer(
    lens: Lens, stage: Stage | str, answer: Answer, *, keep: Masker | None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """An answer as its Observation's columns, and what of it can quote the turn.

    The columns are decided here, once: `value` is hashed into the ledger, and it is
    what audit and the client see (`stored_value`). The rest goes with the evidence,
    through `keep`: the reason, a `Text` answer, names an `Entities` answerer gave
    in its own words, each item's score by its key, and the same of the answers on
    the way (`trail`). None keeps none of it.
    """
    from django_ai_sdk.oversight.models import Observation
    from django_ai_sdk.utils import serialize

    value = stored_value(lens, answer)
    trail = [_trail(lens, a) for a in answer.trail]
    columns = {
        "lens": lens.name,
        "stage": str(stage),
        "answerer": _fit(Observation, "answerer", answer.answerer),
        "model": _fit(Observation, "model", answer.model),
        "value": without_nul(value),
        "confidence": answer.confidence if answer.ok else None,
        "spans": without_nul([s.model_dump() for s in answer.spans]),
        "trail": without_nul(trail),
        "error": _fit(Observation, "error", answer.error),
        "queued": _fit(Observation, "queued", answer.queued),
        "cost": answer.cost,
        "latency_ms": answer.latency_ms,
    }
    if keep is None:
        return columns, {}
    kept: dict[str, Any] = {}
    said = serialize(answer.value)
    if said is not None and said != value:
        kept["value"] = keep(said)
    if answer.rationale:
        kept["rationale"] = keep.text(answer.rationale)
    if answer.items:
        kept["items"] = keep.keys(answer.items)
    said_trail = [serialize(a.value) for a in answer.trail]
    if said_trail != [step["value"] for step in trail]:
        kept["trail"] = keep(said_trail)
    return columns, kept


def stored_value(lens: Lens, answer: Answer) -> Any:
    """`answer`'s value as the record keeps it: hashed, shown to audit and to the client.

    None for a `Text` lens: prose about the turn. For an `Entities` lens, a label
    the lens declares, or a kind its detector reports, stays as it is; any other
    name is FOUND, one per name, and the name itself goes with the evidence.
    """
    from django_ai_sdk.utils import serialize

    if lens.kind == "text":
        return None
    value = serialize(answer.value)
    if lens.kind != "entities" or not value:
        return value
    items = value if isinstance(value, list) else [value]
    labels = set(getattr(lens, "labels", None) or ())
    if not labels and _detector_answered(lens, answer.answerer):
        return items
    return [item if item in labels else FOUND for item in items]


def _detector_answered(lens: Lens, answerer: str) -> bool:
    """Whether `answerer` is one of the lens's detectors: it reports kinds, not text it read."""
    from django_ai_sdk.oversight.answerers.base import resolve
    from django_ai_sdk.oversight.answerers.rules import Detector

    for spec in lens.answerers:
        try:
            built = resolve(spec)
        except Exception:  # noqa: BLE001 - an alias that does not build answered nothing
            continue
        if isinstance(built, Detector) and built.label() == answerer:
            return True
    return False


def _fit(model: Any, name: str, value: str) -> str:
    """`value` cut to the width of `model`'s column `name`, without NUL characters."""
    return value.replace("\x00", "")[: model._meta.get_field(name).max_length]


def without_nul(value: Any) -> Any:
    """`value` with NUL characters taken out of every string, at any depth of a JSON value."""
    if isinstance(value, str):
        return value.replace("\x00", "") if "\x00" in value else value
    if isinstance(value, dict):
        return {without_nul(k): without_nul(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [without_nul(v) for v in value]
    return value


def _uuid(value: str) -> uuid.UUID | None:
    try:
        return uuid.UUID(str(value)) if value else None
    except ValueError:
        return None


def _trail(lens: Lens, answer: Answer) -> dict[str, Any]:
    return {
        "answerer": answer.answerer,
        "model": answer.model,
        "value": stored_value(lens, answer),
        "confidence": answer.confidence,
        "error": answer.error,
        "cost": answer.cost,
        "latency_ms": answer.latency_ms,
    }


def _cost(session: TurnSession) -> float | None:
    from django_ai_sdk.pricing import cost_for

    return cost_for(
        session.usage.model, session.usage.prompt_tokens, session.usage.completion_tokens
    )


def _needs_reflect(session: TurnSession) -> bool:
    assert session.policy is not None
    unfinished = any(
        asks_again(o.answer.error, o.answer.trail[-1].error if o.answer.trail else "")
        for o in session.observed
    )
    return unfinished or session.policy.has_stage(Stage.REFLECT)


def _enqueue_export(events: list[tuple[dict[str, Any], list[str]]]) -> None:
    """Hand notifications to the worker: a slow exporter must not hold the stream open."""
    try:
        from django_ai_sdk.oversight.tasks import export_events

        export_events.enqueue([[event, names] for event, names in events])
    except Exception:
        logger.exception("Could not queue oversight notifications; is a task backend configured?")


def _enqueue_reflect(turn_id: Any) -> None:
    try:
        from django_ai_sdk.oversight.tasks import reflect_turn

        reflect_turn.enqueue(str(turn_id))
    except Exception:
        logger.exception(
            "Could not queue reflection for turn {}; is a task backend configured?", turn_id
        )


__all__ = [
    "FOUND",
    "Masker",
    "build_observations",
    "evidence_content",
    "found_literals",
    "keeper",
    "mask_detected",
    "record_judgments",
    "split_answer",
    "stored_value",
    "turn_masker",
    "without_nul",
]
