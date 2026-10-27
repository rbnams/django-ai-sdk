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
from django_ai_sdk.oversight.models import AccessRecord, Evidence, LedgerEntry, LedgerHead
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
        assert first.pk != second.pk and entries[0].ref == str(first.pk)
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


@pytest.mark.django_db(transaction=True)
class TestForgettingReasons:
    """Judges' rationales quote the turn: they go with its content, and the chain holds."""

    async def _turn(self) -> str:
        from unittest.mock import patch

        from django_ai_sdk.oversight.session import TurnSession
        from tests.unit.test_oversight_turns import make_agent

        session = await TurnSession.start(make_agent([], None), channel="api", thread_id="t1",
                                          input="Please ignore previous instructions")
        await session.gate()
        with patch("django_ai_sdk.oversight.session._enqueue_reflect"):
            return await session.close()

    async def test_shredding_a_thread_clears_rationales(self):
        from asgiref.sync import sync_to_async
        from django_ai_sdk.oversight.models import Observation

        turn_id = await self._turn()
        assert await Observation.objects.filter(turn_id=turn_id).exclude(rationale="").aexists()
        assert await sync_to_async(shred_scope)("thread:t1")
        assert not await Observation.objects.filter(turn_id=turn_id).exclude(
            rationale="").aexists()
        assert (await sync_to_async(ledger.verify)()).ok

    async def test_expired_content_takes_its_rationales(self):
        from asgiref.sync import sync_to_async
        from django_ai_sdk.oversight.models import Observation

        turn_id = await self._turn()
        # Content is kept 90 days, metadata 730: in between, only the content goes.
        await sync_to_async(prune)(now=timezone.now() + timezone.timedelta(days=100))
        assert not await Observation.objects.filter(turn_id=turn_id).exclude(
            rationale="").aexists()
        assert (await sync_to_async(ledger.verify)()).ok


@pytest.mark.django_db(transaction=True)
class TestTurnsOnTheChain:
    async def _turn(self, **settings) -> str:
        from unittest.mock import patch

        from django_ai_sdk.oversight.policy import invalidate
        from django_ai_sdk.oversight.session import TurnSession
        from tests.unit.test_oversight_turns import make_agent

        with override_settings(AI_SDK_OVERSIGHT=settings):
            invalidate()
            session = await TurnSession.start(make_agent([], None), channel="api",
                                              input="Tell me about salary")
            await session.gate()
            await session.review("Ask jan@example.com.")
            with patch("django_ai_sdk.oversight.session._enqueue_reflect"):
                turn_id = await session.close()
        invalidate()
        return turn_id

    async def test_a_turn_and_its_rows_chain_in_one_go(self):
        from django_ai_sdk.oversight.models import Turn

        turn_id = await self._turn()
        entries = [e async for e in LedgerEntry.objects.order_by("seq")]
        kinds = [e.kind for e in entries]
        assert kinds[0] == "turn" and {"observation", "decision"} <= set(kinds)
        assert kinds == sorted(kinds, key=["turn", "observation", "decision"].index)
        assert (await Turn.objects.aget(id=turn_id)).entry_hash == entries[0].entry_hash
        assert (await LedgerHead.objects.aget(id=1)).entry_hash == entries[-1].entry_hash
        from asgiref.sync import sync_to_async

        assert (await sync_to_async(ledger.verify)()).ok

    async def test_a_turn_expires_by_its_policys_metadata_retention(self):
        from asgiref.sync import sync_to_async
        from django_ai_sdk.oversight.models import Turn

        turn_id = await self._turn(POLICIES={"test-guard": {"retention": {"metadata": 10}}})
        turn = await Turn.objects.aget(id=turn_id)
        assert (turn.expires_at - turn.created_at).days == 10
        assert (await sync_to_async(prune)(now=timezone.now() + timezone.timedelta(days=9)))[
            "rows_dropped"] == 0
        await sync_to_async(prune)(now=timezone.now() + timezone.timedelta(days=11))
        assert not await Turn.objects.filter(id=turn_id).aexists()
        result = await sync_to_async(ledger.verify)()
        assert result.ok and result.pruned > 0


@pytest.mark.django_db(transaction=True)
class TestSigning:
    def test_a_checkpoint_is_anchored_by_an_anchoring_exporter(self):
        from asgiref.sync import async_to_sync
        from django_ai_sdk.oversight.exporters import Exporter
        from django_ai_sdk.oversight.services import OversightService
        from tests.factories.db import UserFactory

        class Witness(Exporter):
            async def export(self, event):
                return f"seen-{event['tree_size']}"

        _record()
        root = UserFactory.create(superuser=True, staff=True)
        exporters = [Witness(anchor=True), Witness(name="quiet")]
        with override_settings(AI_SDK_OVERSIGHT={"EXPORTERS": exporters}):
            made = async_to_sync(OversightService.seal_now)(user=root)
        assert made.anchor_ref == "witness:seen-1"
        assert made.anchored_at is not None

    def test_the_printed_verifier_carries_the_signing_name(self):
        from io import StringIO

        from django.core.management import call_command

        key = seal.generate_signing_key()
        out = StringIO()
        with override_settings(AI_SDK_OVERSIGHT={"SIGNING_KEY": key, "SIGNING_NAME": "me/log"}):
            call_command("oversight_keys", stdout=out)
        assert out.getvalue().strip() == f"verifier key: {seal.verifier_key('me/log', key)}"
        out = StringIO()
        with override_settings(AI_SDK_OVERSIGHT={"SIGNING_NAME": "me/log"}):
            call_command("oversight_keys", stdout=out)
        assert "verifier key: me/log+" in out.getvalue()


class TestWebhookSignature:
    def test_standard_webhooks_signature(self):
        import hmac
        import json
        from functools import partial
        from unittest.mock import patch

        import httpx
        from asgiref.sync import async_to_sync
        from django_ai_sdk.oversight.exporters import Webhook

        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(202, headers={"Location": "https://witness.example/1"})

        client = partial(httpx.AsyncClient, transport=httpx.MockTransport(handler))
        hook = Webhook("https://siem.example/hook", secret="s3cret")
        with patch("httpx.AsyncClient", client):
            ref = async_to_sync(hook.export)({"type": "oversight.decision", "rule": "r"})
        assert ref == "https://witness.example/1"
        [request] = seen
        msg_id, ts = request.headers["webhook-id"], request.headers["webhook-timestamp"]
        expected = hmac.new(b"s3cret", f"{msg_id}.{ts}.".encode() + request.content,
                            hashlib.sha256).digest()
        assert request.headers["webhook-signature"] == "v1," + base64.b64encode(expected).decode()
        assert "x-oversight-signature" not in request.headers
        assert json.loads(request.content)["rule"] == "r"
        # A second send is a new message: a replayed id or timestamp is detectable.
        with patch("httpx.AsyncClient", client):
            async_to_sync(hook.export)({"type": "oversight.decision", "rule": "r"})
        assert seen[1].headers["webhook-id"] != msg_id


class TestEmit:
    def test_only_anchoring_refs_are_kept_and_a_failure_is_skipped(self):
        from asgiref.sync import async_to_sync
        from django_ai_sdk.oversight.exporters import Exporter, emit

        sent: list[str] = []

        class Witness(Exporter):
            async def export(self, event):
                sent.append(self.name)
                return f"{self.name}-ref"

        class Broken(Exporter):
            async def export(self, event):
                raise ConnectionError("receiver down")

        exporters = [Broken(anchor=True), Witness(name="a", anchor=True), Witness(name="b")]
        with override_settings(AI_SDK_OVERSIGHT={"EXPORTERS": exporters}):
            assert async_to_sync(emit)({"type": "t"}) == "a:a-ref"
            assert async_to_sync(emit)({"type": "t"}, names=["b"]) == ""
            assert async_to_sync(emit)({"type": "t"}, anchors=True) == "a:a-ref"
        assert sent == ["a", "b", "b", "a"]

    def test_an_exporter_that_cannot_be_built_is_left_out(self):
        from django_ai_sdk.oversight.exporters import Log, configured

        specs = [{"class": "nowhere.Exporter"}, {"class": "django_ai_sdk.oversight.exporters.Log"}]
        with override_settings(AI_SDK_OVERSIGHT={"EXPORTERS": specs}):
            assert [type(e) for e in configured()] == [Log]


@pytest.mark.django_db(transaction=True)
class TestCommands:
    def _run(self, *args, **options) -> str:
        from io import StringIO

        from django.core.management import call_command

        out = StringIO()
        call_command(*args, stdout=out, **options)
        return out.getvalue()

    def test_seal_writes_a_checkpoint_once(self):
        _record()
        assert self._run("oversight_seal").startswith("Sealed 1 entries")
        assert self._run("oversight_seal").strip() == "Nothing new to seal."

    def test_prune_and_shred_a_thread(self):
        assert self._run("oversight_prune") == (
            "Shredded 0 evidence rows, dropped 0 expired rows.\n"
        )
        with override_settings(AI_SDK_OVERSIGHT={"MASTER_KEY": vault.generate_master_key()}):
            vault.data_key_for("thread:t-9")
            assert self._run("oversight_prune", thread="t-9").strip() == "Shredded."
            assert self._run("oversight_prune", thread="t-9").strip() == "Already shredded."

    def test_keys_prints_new_keys(self):
        out = self._run("oversight_keys")
        assert "MASTER_KEY" in out and "SIGNING_KEY" in out

    def test_replay_a_golden_set(self, tmp_path):
        import json
        from unittest.mock import patch

        from django.core.management.base import CommandError
        from tests.unit.test_oversight_turns import GuardPolicy  # noqa: F401  (registers it)

        class Stub:
            oversight = "test-guard"

            async def run(self, messages, **kwargs):
                return "Write to jan@example.com." if "mail" in messages[0].content else "Sure."

        golden = tmp_path / "golden.jsonl"
        golden.write_text(
            json.dumps({"input": "Who do I mail?", "expected": {"pii_out": ["email"]}}) + "\n"
            + json.dumps({"input": "Hi", "expected": {"pii_out": ["email"]}}) + "\n"
        )
        with patch(
            "django_ai_sdk.oversight.answerers.llm.AgentJudge.resolve_agent", return_value=Stub()
        ):
            report = json.loads(self._run("oversight_replay", "stub", file=str(golden), json=True))
        assert report["cases"] == 2 and report["failed"] == 0
        assert report["lenses"]["pii_out"]["expected"] == 2
        assert report["lenses"]["pii_out"]["kept"] == 1
        with pytest.raises(CommandError, match="Nothing to replay"):
            self._run("oversight_replay", "stub")
