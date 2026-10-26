"""The hash chain: append-only, verifiable, and blind to content.

    entry_hash = sha256(seq, kind, ref, payload_hash, prev_hash)

Each domain row's payload is a fixed set of its fields in canonical JSON. Edit a
row and its payload no longer matches; edit or drop an entry and every later
`prev_hash` stops matching. Content is committed to only by `evidence_hash`,
so shredding it changes nothing here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from django.db import transaction

if TYPE_CHECKING:
    from collections.abc import Callable

    from django.db import models

    from django_ai_sdk.oversight.models import LedgerEntry

GENESIS = "0" * 64


def canonical(value: Any) -> bytes:
    """One byte string per value: sorted keys, no whitespace, ISO datetimes, UTF-8."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=_default
    ).encode()


def _default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def sha256(data: bytes | str) -> str:
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def entry_digest(seq: int, kind: str, ref: str, payload_hash: str, prev_hash: str) -> str:
    return sha256(f"{seq}\n{kind}\n{ref}\n{payload_hash}\n{prev_hash}")


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def turn_payload(turn: Any) -> dict[str, Any]:
    return {
        "id": str(turn.id),
        "created_at": _iso(turn.created_at),
        "channel": turn.channel,
        "agent_id": turn.agent_id,
        "agent_fingerprint": turn.agent_fingerprint,
        "policy": turn.policy,
        "thread_id": turn.thread_id,
        "message_id": turn.message_id,
        "actor": turn.actor_pseudonym,
        "outcome": turn.outcome,
        "shadow_outcome": turn.shadow_outcome,
        "evidence": turn.evidence_hash,
    }


def observation_payload(obs: Any) -> dict[str, Any]:
    return {
        "id": str(obs.id),
        "turn": str(obs.turn_id),
        "lens": obs.lens,
        "stage": obs.stage,
        "answerer": obs.answerer,
        "model": obs.model,
        "value": obs.value,
        "confidence": obs.confidence,
        "error": obs.error,
        "reviewer": obs.reviewer,
    }


def decision_payload(decision: Any) -> dict[str, Any]:
    return {
        "id": str(decision.id),
        "turn": str(decision.turn_id),
        "rule": decision.rule,
        "stage": decision.stage,
        "mode": decision.mode,
        "actions": decision.actions,
        "acted": decision.acted,
        "condition": decision.condition,
    }


def access_payload(record: Any) -> dict[str, Any]:
    return {
        "id": str(record.id),
        "reader": record.reader_label,
        "purpose": record.purpose,
        "action": record.action,
        "turn": record.turn_ref,
        "query": record.query,
        "reason": record.reason,
    }


def override_payload(override: Any) -> dict[str, Any]:
    # Not `active`: a later override withdraws this one, and is chained itself.
    return {
        "id": str(override.id),
        "policy": override.policy,
        "target_kind": override.target_kind,
        "target": override.target,
        "field": override.field,
        "value": override.value,
        "by": override.changed_by_label,
        "note": override.note,
    }


def shred_payload(key: Any) -> dict[str, Any]:
    return {"id": str(key.id), "scope": key.scope, "at": _iso(key.shredded_at)}


@dataclass(frozen=True)
class Kind:
    model: str
    payload: Callable[[Any], dict[str, Any]]
    # Domain rows carry their own entry_hash, so a reader can check one without the chain.
    stamps: bool = True


KINDS: dict[str, Kind] = {
    "turn": Kind("Turn", turn_payload),
    "observation": Kind("Observation", observation_payload),
    "decision": Kind("Decision", decision_payload),
    "access": Kind("AccessRecord", access_payload),
    "override": Kind("PolicyOverride", override_payload),
    "shred": Kind("DataKey", shred_payload, stamps=False),
}


def append(kind: str, row: models.Model) -> LedgerEntry:
    """Chain `row` onto the ledger. Call inside transaction.atomic(), synchronously.

    The head row's lock serializes appenders: one gapless sequence, one tip.
    """
    from django_ai_sdk.oversight.models import LedgerEntry, LedgerHead

    if not transaction.get_connection().in_atomic_block:
        raise RuntimeError("ledger.append must run inside transaction.atomic()")
    spec = KINDS[kind]
    LedgerHead.objects.get_or_create(id=1)
    head = LedgerHead.objects.select_for_update().get(id=1)
    payload_hash = sha256(canonical(spec.payload(row)))
    seq = head.seq + 1
    digest = entry_digest(seq, kind, str(row.pk), payload_hash, head.entry_hash)
    entry = LedgerEntry.objects.create(
        seq=seq,
        kind=kind,
        ref=str(row.pk),
        payload_hash=payload_hash,
        prev_hash=head.entry_hash,
        entry_hash=digest,
    )
    head.seq, head.entry_hash = seq, digest
    head.save(update_fields=["seq", "entry_hash"])
    if spec.stamps:
        type(row)._default_manager.filter(pk=row.pk).update(entry_hash=digest)
        row.entry_hash = digest  # type: ignore[attr-defined]
    return entry


@dataclass
class Verification:
    """The outcome of walking the chain."""

    ok: bool = True
    checked: int = 0
    # Rows the retention prune removed: their entries still chain, their payload is gone.
    pruned: int = 0
    head: str = GENESIS
    problems: list[dict[str, Any]] = field(default_factory=list)

    def fail(self, seq: int, problem: str, **detail: Any) -> None:
        self.ok = False
        if len(self.problems) < 50:
            self.problems.append({"seq": seq, "problem": problem, **detail})

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "checked": self.checked,
            "pruned": self.pruned,
            "head": self.head,
            "problems": self.problems,
        }


def verify(*, start: int = 1, stop: int | None = None, batch: int = 2_000) -> Verification:
    """Walk the chain from `start`, recomputing every link and every payload.

    Synchronous and read-only; run it from a command or sync_to_async.
    """
    from django.apps import apps

    from django_ai_sdk.oversight.models import APP_LABEL, LedgerEntry

    result = Verification()
    previous = GENESIS
    expected_seq = start
    if start > 1:
        prior = LedgerEntry.objects.filter(seq=start - 1).first()
        previous = prior.entry_hash if prior else GENESIS
    query = LedgerEntry.objects.filter(seq__gte=start).order_by("seq")
    if stop is not None:
        query = query.filter(seq__lte=stop)
    last = start - 1
    while True:
        entries = list(query.filter(seq__gt=last)[:batch])
        if not entries:
            break
        by_kind: dict[str, list[str]] = {}
        for entry in entries:
            by_kind.setdefault(entry.kind, []).append(entry.ref)
        rows: dict[tuple[str, str], Any] = {}
        for kind, refs in by_kind.items():
            spec = KINDS.get(kind)
            if spec is None:
                continue
            model = apps.get_model(APP_LABEL, spec.model)
            for row in model._default_manager.filter(pk__in=refs):
                rows[(kind, str(row.pk))] = row
        for entry in entries:
            result.checked += 1
            if entry.seq != expected_seq:
                result.fail(entry.seq, "gap", expected=expected_seq)
            expected_seq = entry.seq + 1
            if entry.prev_hash != previous:
                result.fail(entry.seq, "broken link", expected=previous, found=entry.prev_hash)
            digest = entry_digest(
                entry.seq, entry.kind, entry.ref, entry.payload_hash, entry.prev_hash
            )
            if digest != entry.entry_hash:
                result.fail(entry.seq, "entry altered")
            spec = KINDS.get(entry.kind)
            row = rows.get((entry.kind, entry.ref))
            if spec is None:
                result.fail(entry.seq, "unknown kind", kind=entry.kind)
            elif row is None:
                result.pruned += 1
            elif sha256(canonical(spec.payload(row))) != entry.payload_hash:
                result.fail(entry.seq, "record altered", kind=entry.kind, ref=entry.ref)
            previous = entry.entry_hash
        last = entries[-1].seq
    result.head = previous
    return result


__all__ = ["GENESIS", "KINDS", "Verification", "append", "canonical", "sha256", "verify"]
