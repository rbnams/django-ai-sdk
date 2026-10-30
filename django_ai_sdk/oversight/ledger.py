"""The hash chain: append-only, verifiable, and blind to content.

    entry_hash = sha256(seq, kind, ref, payload_hash, prev_hash)

Each domain row's payload is a fixed set of its fields in canonical JSON. Edit a
row and its payload no longer matches; edit or drop an entry and every later
`prev_hash` stops matching. Content is committed to only by `evidence_hash`,
so shredding it changes nothing here; verify recomputes it for evidence it can
still read. Drop the newest entries and the chain no longer reaches `LedgerHead`,
unless the head was rolled back too: only a published checkpoint protects the end.

Retention deletes old turns with their observations and decisions, never their
entries: it chains a "prune" entry naming the run of entries whose rows went,
and verify accepts a missing row only under one. Delete any other way and
verify says "record missing". A prune entry is itself only a claim: anyone who
can write the chain can append one, so check what was pruned against retention.
"""

from __future__ import annotations

import hashlib
import json
from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

from django.db import transaction

if TYPE_CHECKING:
    from collections.abc import Callable

    from django.db import models

    from django_ai_sdk.oversight.models import (
        AccessRecord,
        DataKey,
        Decision,
        LedgerEntry,
        Observation,
        PolicyOverride,
        Turn,
    )

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


# Sections of a turn's evidence kept beside what its hash commits to: what judges and
# reviewers wrote, added as they write it. They go when the evidence goes.
UNHASHED = frozenset({"texts", "notes"})


def evidence_digest(content: dict[str, Any]) -> str:
    """The hash a turn commits to for its evidence: all of it but the UNHASHED sections."""
    return sha256(canonical({k: v for k, v in content.items() if k not in UNHASHED}))


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


# Hashed fields: never change them for an existing kind; add a new kind instead.
# verify() recomputes every stored row's payload with these functions, so a renamed
# or reformatted field breaks the chain for every row written before the change.
def turn_payload(turn: Turn) -> dict[str, Any]:
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
        "report_outcome": turn.report_outcome,
        "evidence": turn.evidence_hash,
    }


def observation_payload(obs: Observation) -> dict[str, Any]:
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


def decision_payload(decision: Decision) -> dict[str, Any]:
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


def access_payload(record: AccessRecord) -> dict[str, Any]:
    return {
        "id": str(record.id),
        "reader": record.reader_label,
        "purpose": record.purpose,
        "action": record.action,
        "turn": record.turn_ref,
        "query": record.query,
        "reason": record.reason,
    }


def override_payload(override: PolicyOverride) -> dict[str, Any]:
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


def shred_payload(key: DataKey) -> dict[str, Any]:
    return {"id": str(key.id), "scope": key.scope, "at": _iso(key.shredded_at)}


@dataclass(frozen=True)
class Kind:
    model: str
    payload: Callable[[Any], dict[str, Any]]
    # The row keeps its entry_hash: the feed shows a turn as chained without a join.
    stamps: bool = False


KINDS: dict[str, Kind] = {
    "turn": Kind("Turn", turn_payload, stamps=True),
    "observation": Kind("Observation", observation_payload),
    "decision": Kind("Decision", decision_payload),
    "access": Kind("AccessRecord", access_payload),
    "override": Kind("PolicyOverride", override_payload),
    "shred": Kind("DataKey", shred_payload),
}

# Chained by `prune_turns`, ref "first-last": the run of entries whose rows retention
# deleted. It commits to the chain itself, so it has no row and is not in KINDS.
PRUNE = "prune"
# What retention deletes. A missing row of any other kind is never pruning.
PRUNABLE = frozenset({"turn", "observation", "decision"})


def prune_payload(ref: str) -> dict[str, Any]:
    first, last = _span(ref)
    return {"from": first, "to": last}


def _span(ref: str) -> tuple[int, int]:
    first, _, last = ref.partition("-")
    return int(first), int(last)


def append(kind: str, row: models.Model) -> LedgerEntry:
    """Chain `row` onto the ledger. Call inside transaction.atomic(), synchronously."""
    return append_many([(kind, row)])[0]


def append_many(items: list[tuple[str, models.Model]]) -> list[LedgerEntry]:
    """Chain `items` onto the ledger in order. Call inside transaction.atomic(), synchronously.

    The head row's lock serializes appenders: one gapless sequence, one tip. Take
    it last in the transaction, so other turns wait for it as briefly as they can.
    """
    return _chain(
        [
            (kind, str(row.pk), sha256(canonical(KINDS[kind].payload(row))), row)
            for kind, row in items
        ]
    )


def _chain(items: list[tuple[str, str, str, models.Model | None]]) -> list[LedgerEntry]:
    """Link (kind, ref, payload hash, row) onto the tip, under the head row's lock."""
    from django_ai_sdk.oversight.models import LedgerEntry, LedgerHead

    if not transaction.get_connection().in_atomic_block:
        raise RuntimeError("ledger.append must run inside transaction.atomic()")
    if not items:
        return []
    head, _ = LedgerHead.objects.select_for_update().get_or_create(id=1)
    seq, previous = head.seq, head.entry_hash
    entries: list[LedgerEntry] = []
    stamped: list[Turn] = []
    for kind, ref, payload_hash, row in items:
        seq += 1
        digest = entry_digest(seq, kind, ref, payload_hash, previous)
        entries.append(
            LedgerEntry(
                seq=seq,
                kind=kind,
                ref=ref,
                payload_hash=payload_hash,
                prev_hash=previous,
                entry_hash=digest,
            )
        )
        if row is not None and KINDS[kind].stamps:
            # Only turns stamp (see KINDS), and a Turn has an entry_hash.
            turn = cast("Turn", row)
            turn.entry_hash = digest
            stamped.append(turn)
        previous = digest
    LedgerEntry.objects.bulk_create(entries)
    head.seq, head.entry_hash = seq, previous
    head.save(update_fields=["seq", "entry_hash"])
    for turn in stamped:
        type(turn)._default_manager.filter(pk=turn.pk).update(entry_hash=turn.entry_hash)
    return entries


def prune_turns(turns: models.QuerySet[Turn], *, batch: int = 100) -> int:
    """Delete `turns` with their observations and decisions, and chain what went.

    Each run of consecutive entries whose rows went gets one "prune" entry: the
    only way verify accepts a missing row. Returns the rows deleted, cascades
    included. Call inside transaction.atomic(), synchronously.
    """
    from django_ai_sdk.oversight.models import Decision, LedgerEntry, Observation

    if not transaction.get_connection().in_atomic_block:
        raise RuntimeError("ledger.prune_turns must run inside transaction.atomic()")
    ids = list(turns.values_list("pk", flat=True))
    seqs: list[int] = []
    deleted = 0
    for i in range(0, len(ids), batch):
        chunk = ids[i : i + batch]
        refs = {
            "turn": chunk,
            "observation": list(
                Observation.objects.filter(turn__in=chunk).values_list("pk", flat=True)
            ),
            "decision": list(Decision.objects.filter(turn__in=chunk).values_list("pk", flat=True)),
        }
        for kind, pks in refs.items():
            found = LedgerEntry.objects.filter(kind=kind, ref__in=[str(pk) for pk in pks])
            seqs.extend(found.values_list("seq", flat=True))
        deleted += turns.model._default_manager.filter(pk__in=chunk).delete()[0]
    runs: list[list[int]] = []
    for seq in sorted(seqs):
        if runs and runs[-1][1] == seq - 1:
            runs[-1][1] = seq
        else:
            runs.append([seq, seq])
    spans = [f"{first}-{last}" for first, last in runs]
    _chain([(PRUNE, ref, sha256(canonical(prune_payload(ref))), None) for ref in spans])
    return deleted


@dataclass
class Verification:
    """The outcome of walking the chain."""

    ok: bool = True
    checked: int = 0
    # Rows retention deleted under a "prune" entry: their entries still chain.
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

    A row may be missing only when a later "prune" entry covers its entry; prune
    entries past `stop` are read but not walked. A turn's evidence that still opens
    here must hash to the turn's `evidence_hash`. Without `stop`, the chain must
    reach the head as it stood when the walk began. Synchronous and read-only; run
    it from a command or sync_to_async.
    """
    from django.apps import apps

    from django_ai_sdk.oversight.models import APP_LABEL, LedgerEntry, LedgerHead

    result = Verification()
    previous = GENESIS
    expected_seq = start
    head = LedgerHead.objects.filter(id=1).first() if stop is None else None
    if start > 1:
        prior = LedgerEntry.objects.filter(seq=start - 1).first()
        previous = prior.entry_hash if prior else GENESIS
    starts, ends = _pruned_spans(start)
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
            found = model._default_manager.filter(pk__in=refs)
            if kind == "turn":
                found = found.select_related("evidence__key")
            for row in found:
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
            if entry.kind == PRUNE:
                if _prune_hash(entry.ref) != entry.payload_hash:
                    result.fail(entry.seq, "record altered", kind=entry.kind, ref=entry.ref)
            elif spec is None:
                result.fail(entry.seq, "unknown kind", kind=entry.kind)
            elif row is None:
                i = bisect_right(starts, entry.seq) - 1
                if entry.kind in PRUNABLE and i >= 0 and entry.seq <= ends[i]:
                    result.pruned += 1
                else:
                    result.fail(entry.seq, "record missing", kind=entry.kind, ref=entry.ref)
            elif sha256(canonical(spec.payload(row))) != entry.payload_hash:
                result.fail(entry.seq, "record altered", kind=entry.kind, ref=entry.ref)
            elif entry.kind == "turn" and (problem := _evidence_problem(row)):
                result.fail(entry.seq, problem, kind=entry.kind, ref=entry.ref)
            previous = entry.entry_hash
        last = entries[-1].seq
    result.head = previous
    if stop is None:
        _reaches_head(result, head, walked=last)
    return result


def _reaches_head(result: Verification, head: Any, *, walked: int) -> None:
    """Whether the chain still holds the head it had when the walk began.

    Read before the walk, so entries appended meanwhile do not count against it.
    Deleting the newest entries (and their rows) leaves the head pointing past the
    end; rolling the head back with them is not caught here, only by a checkpoint
    published where the database's writers cannot change it.
    """
    from django_ai_sdk.oversight.models import LedgerEntry, LedgerHead

    if head is None:
        head = LedgerHead.objects.filter(id=1).first()
        if head is None:
            if walked > 0:
                result.fail(walked, "head missing")
            return
    if head.seq == 0:
        return
    tip = LedgerEntry.objects.filter(seq=head.seq).values_list("entry_hash", flat=True).first()
    if tip is None:
        result.fail(head.seq, "chain ends early", head=head.entry_hash, last=walked)
    elif tip != head.entry_hash:
        result.fail(head.seq, "head mismatch", head=head.entry_hash, found=tip)


def _evidence_problem(turn: Any) -> str:
    """What is wrong with a turn's evidence, if it still opens here: "" when nothing.

    Shredded or expired evidence, and evidence sealed under another MASTER_KEY, have
    nothing to compare. Evidence that is neither shredded nor holds content is missing.
    """
    from django.core.exceptions import ImproperlyConfigured

    from django_ai_sdk.oversight import vault

    evidence = turn.evidence
    if evidence is None or evidence.shredded_at is not None:
        return ""
    content = evidence.content
    if content is None:
        if evidence.ciphertext is None:
            return "evidence missing"
        try:
            content = vault.decrypt_json(evidence.key, evidence.ciphertext, evidence.scope)
        except (vault.VaultError, ImproperlyConfigured, ValueError):
            return ""
        if content is None:
            return ""
    if not isinstance(content, dict) or evidence_digest(content) != turn.evidence_hash:
        return "evidence altered"
    return ""


def _pruned_spans(start: int) -> tuple[list[int], list[int]]:
    """The seq runs that "prune" entries from `start` on cover, merged: (starts, ends).

    A prune entry covers only entries before itself: it cannot excuse a later deletion.
    """
    from django_ai_sdk.oversight.models import LedgerEntry

    spans: list[tuple[int, int]] = []
    for seq, ref in LedgerEntry.objects.filter(kind=PRUNE, seq__gte=start).values_list(
        "seq", "ref"
    ):
        try:
            first, last = _span(ref)
        except ValueError:
            continue  # The walk reports it as altered.
        if first <= last < seq:
            spans.append((first, last))
    starts: list[int] = []
    ends: list[int] = []
    for first, last in sorted(spans):
        if ends and first <= ends[-1] + 1:
            ends[-1] = max(ends[-1], last)
        else:
            starts.append(first)
            ends.append(last)
    return starts, ends


def _prune_hash(ref: str) -> str | None:
    try:
        return sha256(canonical(prune_payload(ref)))
    except ValueError:
        return None


__all__ = [
    "GENESIS",
    "KINDS",
    "PRUNABLE",
    "PRUNE",
    "UNHASHED",
    "Verification",
    "append",
    "append_many",
    "canonical",
    "evidence_digest",
    "prune_turns",
    "sha256",
    "verify",
]
