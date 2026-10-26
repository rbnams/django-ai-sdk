"""The ledger: chained, tamper-evident, checkpointed, and able to forget content."""

from __future__ import annotations

import base64
import hashlib
import os

import pytest
from django.db import transaction
from django.test import override_settings
from django.utils import timezone
from django_ai_sdk.oversight import ledger, seal, vault
from django_ai_sdk.oversight.models import AccessRecord, Evidence, LedgerEntry
from django_ai_sdk.oversight.tasks import prune, shred_scope

# RFC 6962 test inputs, as used by the certificate-transparency reference code.
LEAVES = [
    b"",
    b"\x00",
    b"\x10",
    b"\x20\x21",
    b"\x30\x31",
    b"\x40\x41\x42\x43",
    b"\x50\x51\x52\x53\x54\x55\x56\x57",
    b"\x60\x61\x62\x63\x64\x65\x66\x67\x68\x69\x6a\x6b\x6c\x6d\x6e\x6f",
]
ROOTS = {
    1: "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d",
    2: "fac54203e7cc696cf0dfcb42c92a1d9dbaf70ad9e621f4bd8d98662f00e3c125",
    3: "aeb6bcfe274b70a14fb067a5e5578264db0fa9b51af5e0ba159158f329e06e77",
    8: "5dc9da79a70659a9ad559cb701ded9a2ab9d823aad2f4960cfe370eff4604328",
}


def _record(reason: str = "x") -> AccessRecord:
    with transaction.atomic():
        record = AccessRecord.objects.create(purpose="audit", action="view", reason=reason)
        ledger.append("access", record)
    return record


class TestMerkle:
    def test_rfc_6962_roots(self):
        for size, root in ROOTS.items():
            assert seal.merkle_root(LEAVES[:size]).hex() == root

    def test_empty_tree(self):
        assert seal.merkle_root([]) == hashlib.sha256(b"").digest()

    def test_inclusion_proofs_verify_for_every_leaf_and_size(self):
        leaves = [os.urandom(8) for _ in range(19)]
        for size in range(1, len(leaves) + 1):
            root = seal.merkle_root(leaves[:size])
            for index in range(size):
                proof = seal.inclusion_proof(index, leaves[:size])
                assert seal.verify_inclusion(leaves[index], index, size, proof, root)
                assert not seal.verify_inclusion(b"forged", index, size, proof, root)

    def test_consistency_proofs_verify_for_every_pair_of_sizes(self):
        leaves = [os.urandom(8) for _ in range(17)]
        for new in range(1, len(leaves) + 1):
            new_root = seal.merkle_root(leaves[:new])
            for old in range(1, new + 1):
                old_root = seal.merkle_root(leaves[:old])
                proof = seal.consistency_proof(old, leaves[:new]) if old < new else []
                assert seal.verify_consistency(old, new, old_root, new_root, proof)
                if old < new:
                    assert not seal.verify_consistency(old, new, os.urandom(32), new_root, proof)

    def test_signed_notes(self):
        key = seal.generate_signing_key()
        body = seal.checkpoint_body("example.com/log", 8, bytes.fromhex(ROOTS[8]))
        note = seal.sign_note(body, name="example.com/log", private_key_b64=key)
        verifier = seal.verifier_key("example.com/log", key)
        assert seal.verify_note(note, verifier)
        assert not seal.verify_note(note.replace("\n8\n", "\n9\n"), verifier)
        other = seal.verifier_key("example.com/log", seal.generate_signing_key())
        assert not seal.verify_note(note, other)


@pytest.mark.django_db(transaction=True)
class TestChain:
    def test_appends_chain_and_verify(self):
        first, second = _record("a"), _record("b")
        entries = list(LedgerEntry.objects.order_by("seq"))
        assert [e.seq for e in entries] == [1, 2]
        assert entries[1].prev_hash == entries[0].entry_hash
        assert first.entry_hash == entries[0].entry_hash
        assert ledger.verify().ok

    def test_an_edited_record_is_caught(self):
        record = _record("original")
        _record("next")
        AccessRecord.objects.filter(pk=record.pk).update(reason="edited")
        result = ledger.verify()
        assert not result.ok
        assert result.problems[0]["problem"] == "record altered"

    def test_an_edited_entry_is_caught(self):
        _record()
        _record()
        LedgerEntry.objects.filter(seq=1).update(payload_hash="0" * 64)
        problems = {p["problem"] for p in ledger.verify().problems}
        assert {"entry altered", "record altered"} <= problems

    def test_a_deleted_entry_breaks_the_chain(self):
        for _ in range(3):
            _record()
        LedgerEntry.objects.filter(seq=2).delete()
        problems = {p["problem"] for p in ledger.verify().problems}
        assert {"gap", "broken link"} <= problems

    def test_a_pruned_record_is_counted_not_failed(self):
        record = _record()
        AccessRecord.objects.filter(pk=record.pk).delete()
        result = ledger.verify()
        assert result.ok
        assert result.pruned == 1

    def test_append_outside_a_transaction_is_refused(self):
        record = AccessRecord.objects.create(purpose="audit", action="view")
        with pytest.raises(RuntimeError):
            ledger.append("access", record)

    def test_checkpoints(self):
        from django_ai_sdk.oversight.tasks import seal as seal_now

        for _ in range(5):
            _record()
        made = seal_now(origin="test/log")
        assert made["tree_size"] == 5
        assert not made["signed"]
        assert seal_now() is None
        _record()
        assert seal_now()["tree_size"] == 6


MASTER = base64.b64encode(b"k" * 32).decode()


@pytest.mark.django_db(transaction=True)
class TestVault:
    @pytest.fixture(autouse=True)
    def _master_key(self):
        with override_settings(AI_SDK_OVERSIGHT={"MASTER_KEY": MASTER}):
            yield

    def test_round_trip_and_shred(self):
        sealed = vault.encrypt_json("thread:1", {"input": "secret question"})
        assert sealed is not None
        key, blob = sealed
        assert b"secret" not in blob
        assert vault.decrypt_json(key, blob, "thread:1") == {"input": "secret question"}
        Evidence.objects.create(digest="d" * 64, scope="thread:1", key=key, ciphertext=blob,
                                capture="full")

        assert shred_scope("thread:1")
        key.refresh_from_db()
        assert key.wrapped is None
        assert vault.decrypt_json(key, blob, "thread:1") is None
        # Nothing new is written under a forgotten scope.
        assert vault.encrypt_json("thread:1", {"x": 1}) is None
        # Forgetting is on the record, and the chain still holds.
        assert LedgerEntry.objects.filter(kind="shred").count() == 1
        assert ledger.verify().ok

    def test_the_wrong_scope_cannot_open_a_blob(self):
        key, blob = vault.encrypt_json("thread:a", {"x": 1})
        with pytest.raises(Exception):  # noqa: B017
            vault.decrypt_json(key, blob, "thread:b")

    def test_actor_sealing(self):
        sealed = vault.seal_text("actor:u-1", "42")
        assert sealed.startswith("aead:")
        assert vault.open_text("actor:u-1", sealed) == "42"


@pytest.mark.django_db(transaction=True)
def test_prune_shreds_expired_evidence():
    past = timezone.now() - timezone.timedelta(days=1)
    Evidence.objects.create(digest="e" * 64, scope="turn:1", capture="full",
                            content={"input": "x"}, expires_at=past)
    assert prune()["evidence_shredded"] == 1
    assert Evidence.objects.get(digest="e" * 64).content is None
