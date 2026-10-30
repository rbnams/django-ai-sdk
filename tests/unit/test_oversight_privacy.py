"""What oversight keeps where: everything that can quote a turn lives with its evidence.

Regression tests from the privacy review. Per capture level, the rows the chain hashes
(turn, observation, decision) and the review items hold no text of the turn; the
evidence holds it as capture says; forgetting a thread takes it, and who asked.
"""

from __future__ import annotations

import base64
import json
import uuid
from unittest.mock import patch

import pytest
from asgiref.sync import sync_to_async
from django.db import connection
from django.test import override_settings
from django_ai_sdk.common import ChatMessage
from django_ai_sdk.oversight import (
    PII,
    Entities,
    Flag,
    Function,
    L,
    Mask,
    Mode,
    Policy,
    Queue,
    Rule,
    Text,
    ledger,
    register,
    vault,
)
from django_ai_sdk.oversight.answerers.pii import DEFAULT_KINDS
from django_ai_sdk.oversight.models import (
    AccessRecord,
    Decision,
    Evidence,
    LedgerEntry,
    LedgerHead,
    Observation,
    ReviewItem,
    Turn,
)
from django_ai_sdk.oversight.policy import invalidate
from django_ai_sdk.oversight.record import FOUND, keeper, split_answer, stored_value
from django_ai_sdk.oversight.schemas import VerdictIn
from django_ai_sdk.oversight.services import OversightService
from django_ai_sdk.oversight.session import TurnSession
from django_ai_sdk.oversight.tasks import areflect, load_evidence, shred_scope
from django_ai_sdk.oversight.types import Answer, Stage, Subject, TextSpan, ToolCall
from haystack.dataclasses import ChatMessage as HaystackChatMessage
from tests.factories.db import UserFactory
from tests.unit.test_oversight_ledger import _record
from tests.unit.test_oversight_turns import Storage, chat, make_agent, oversight_parts

EMAIL = "jan@example.com"
BSN = "111222333"  # passes the 11-check: a country kind no lens of the policy names
KEY = "sk-proj-abcdefghijklmnopqrstuvwxyz0123456789ABCD"
NAME = "Jan Jansen"  # found by a judge, not by any detector
IBAN = "NL91ABNA0417164300"
SECRETS = (EMAIL, BSN, KEY, NAME, "Jansen")
MASTER = base64.b64encode(b"k" * 32).decode()
OTHER = base64.b64encode(b"o" * 32).decode()


def _naming_judge(lens, view):
    # A judge that names who it found in its own words (no `labels`), and says where.
    at = view.input.find(NAME)
    if at < 0:
        return Answer(value=[])
    return Answer(value=[NAME], spans=[TextSpan(start=at, end=at + len(NAME), label="person")])


def _judged(lens, view):
    return Answer(value=True, confidence=0.9, rationale=f"It quotes {NAME}, {EMAIL}, {BSN}, {KEY}")


def _gist(lens, view):
    return f"Asks to mail {EMAIL} for {NAME}, BSN {BSN}"


def _taken(lens, view):
    # Like TypeSafe.per_source: the titles of the sources, each one's score by its key.
    return Answer(
        value=[s.title for s in view.sources],
        confidence=0.9,
        items={s.key: 0.9 for s in view.sources},
    )


def _later(lens, view):
    return f"Later: {EMAIL} again"


@register
class PrivacyPolicy(Policy):
    name = "test-privacy"
    lenses = [  # noqa: RUF012
        Entities(
            name="pii_in",
            question="?",
            stages=[Stage.GATE],
            answerers=[PII(kinds=[*DEFAULT_KINDS, "bsn"])],
        ),
        Entities(name="who", question="?", stages=[Stage.GATE], answerers=[Function(_naming_judge)]),
        Flag(
            name="judged",
            question="?",
            subject=Subject.OUTPUT,
            stages=[Stage.FINISH],
            answerers=[Function(_judged)],
        ),
        Text(
            name="gist",
            question="?",
            subject=Subject.OUTPUT,
            stages=[Stage.FINISH],
            answerers=[Function(_gist)],
        ),
        Entities(
            name="taken",
            question="?",
            subject=Subject.OUTPUT,
            stages=[Stage.FINISH],
            answerers=[Function(_taken)],
        ),
        Text(
            name="later",
            question="?",
            subject=Subject.OUTPUT,
            stages=[Stage.REFLECT],
            answerers=[Function(_later)],
        ),
    ]
    rules = [  # noqa: RUF012
        # Report mode: nothing is masked by a rule, so capture alone decides.
        Rule("mask_pii_in", when=L("pii_in").any, then=Mask(), mode=Mode.REPORT),
        Rule("queue_judged", when=L("judged").is_true, then=Queue("q"), mode=Mode.ENFORCE),
    ]


@register
class NamingPolicy(Policy):
    name = "test-privacy-naming"
    lenses = [  # noqa: RUF012
        Entities(
            name="who",
            question="?",
            stages=[Stage.GATE],
            answerers=[
                Function(
                    lambda lens, v: Answer(
                        value=[EMAIL],
                        spans=[TextSpan(start=v.input.find(EMAIL),
                                        end=v.input.find(EMAIL) + len(EMAIL), label="email")],
                    )
                    if EMAIL in v.input
                    else Answer(value=[])
                )
            ],
        )
    ]
    rules = [Rule("mask_who", when=L("who").any, then=Mask(), mode=Mode.ENFORCE)]  # noqa: RUF012


@pytest.fixture(autouse=True)
def _fresh():
    invalidate()
    yield
    invalidate()


async def run_turn(capture: str = "full", *, thread_id: str | None = None, user=None,
                   **settings) -> Turn:
    with override_settings(AI_SDK_OVERSIGHT={"CAPTURE": capture, **settings}):
        invalidate()
        messages = [
            ChatMessage(role="user", content=f"Earlier: I am {NAME}, mail {EMAIL}, key {KEY}"),
            ChatMessage(role="assistant", content=f"Noted, {NAME}, BSN {BSN}"),
            ChatMessage(role="user", content=f"Now: {NAME} {EMAIL} BSN {BSN}"),
        ]
        session = await TurnSession.start(
            make_agent([], None, policy="test-privacy"), channel="api", messages=messages,
            thread_id=thread_id, user=user,
        )
        assert session.active
        await session.gate()
        session.set_sources([{
            "key": f"https://crm/{EMAIL}", "title": f"File of {NAME} {EMAIL}",
            "url": f"https://crm/{EMAIL}", "content": f"{NAME}: mail {EMAIL}, BSN {BSN}, {KEY}",
            "kind": "doc",
        }])
        session.add_tool_call(ToolCall(
            id="c1", name="lookup", arguments={"q": f"{NAME} {EMAIL} {BSN}"},
            result=f"Found {NAME}, {EMAIL}, BSN {BSN}, key {KEY}",
        ))
        await session.finish(f"Answer for {NAME}: {EMAIL}, BSN {BSN}")
        with patch("django_ai_sdk.oversight.record._enqueue_reflect"):
            turn_id = await session.close()
    invalidate()
    return await Turn.objects.select_related("evidence__key").aget(id=turn_id)


def _rows(turn: Turn) -> str:
    """Every row the chain hashes, and the review items, as one string."""
    found = [
        *Turn.objects.filter(pk=turn.pk).values(),
        *Observation.objects.filter(turn=turn).values(),
        *Decision.objects.filter(turn=turn).values(),
        *ReviewItem.objects.filter(turn=turn).values(),
    ]
    return json.dumps(found, default=str)


def _leaks(text: str) -> list[str]:
    return [secret for secret in SECRETS if secret in text]


async def _kept(turn: Turn) -> dict:
    """What the turn's evidence holds now, read again."""
    fresh = await Turn.objects.select_related("evidence__key").aget(pk=turn.pk)
    return await sync_to_async(load_evidence)(fresh) or {}


def _by_lens(turn: Turn) -> dict[str, Observation]:
    return {o.lens: o for o in Observation.objects.filter(turn=turn)}


@pytest.mark.django_db(transaction=True)
class TestWhatCaptureKeeps:
    @pytest.mark.parametrize("capture", ["none", "metadata", "masked", "full"])
    async def test_the_rows_hold_no_text_of_the_turn(self, capture):
        turn = await run_turn(capture)
        assert _leaks(await sync_to_async(_rows)(turn)) == []
        observations = await sync_to_async(_by_lens)(turn)
        # A detector's kinds stay; names a judge gave in its own words are counted.
        assert observations["pii_in"].value == ["bsn", "email"]
        assert observations["who"].value == [FOUND]
        assert observations["taken"].value == [FOUND]
        assert observations["gist"].value is None
        assert turn.labels["who"] == turn.labels["taken"] == [FOUND]
        assert "gist" not in turn.labels

    @pytest.mark.parametrize("capture", ["none", "metadata"])
    async def test_none_and_metadata_keep_no_text_anywhere(self, capture):
        turn = await run_turn(capture)
        kept = await _kept(turn)
        assert _leaks(json.dumps(kept)) == []
        assert "File of" not in json.dumps(kept) and "crm" not in json.dumps(kept)
        assert "texts" not in kept

    async def test_masked_capture_masks_every_string_it_keeps(self):
        turn = await run_turn("masked")
        kept = await _kept(turn)
        assert _leaks(json.dumps(kept)) == []
        assert kept["input"] == "Now: [PERSON] [EMAIL] BSN [BSN]"
        assert kept["history"][0]["content"] == "Earlier: I am [PERSON], mail [EMAIL], key [API_KEY]"
        [source] = kept["sources"]
        assert (source["key"], source["title"]) == ("https://crm/[EMAIL]", "File of [PERSON] [EMAIL]")
        [call] = kept["tool_calls"]
        assert call["result"] == "Found [PERSON], [EMAIL], BSN [BSN], key [API_KEY]"
        assert call["arguments"] == {"q": "[PERSON] [EMAIL] [BSN]"}
        observations = await sync_to_async(_by_lens)(turn)
        texts = kept["texts"]
        assert texts[str(observations["judged"].id)] == {
            "rationale": "It quotes [PERSON], [EMAIL], [BSN], [API_KEY]"
        }
        assert texts[str(observations["gist"].id)] == {
            "value": "Asks to mail [EMAIL] for [PERSON], BSN [BSN]"
        }
        assert texts[str(observations["who"].id)]["value"] == ["[PERSON]"]
        assert texts[str(observations["taken"].id)] == {
            "value": ["File of [PERSON] [EMAIL]"],
            "items": {"https://crm/[EMAIL]": 0.9},
        }

    async def test_full_capture_keeps_the_words_with_the_evidence_only(self):
        turn = await run_turn("full")
        kept = await _kept(turn)
        observations = await sync_to_async(_by_lens)(turn)
        texts = kept["texts"]
        assert texts[str(observations["who"].id)]["value"] == [NAME]
        assert texts[str(observations["judged"].id)]["rationale"].startswith(f"It quotes {NAME}")
        # Outside what the turn's evidence hash commits to: added later, as judges answer.
        assert ledger.evidence_digest(kept) == turn.evidence_hash
        assert (await sync_to_async(ledger.verify)()).ok

    async def test_reflect_keeps_its_words_like_close_did(self):
        turn = await run_turn("masked")
        assert await areflect(str(turn.id)) == 1
        later = await Observation.objects.aget(turn=turn, lens="later")
        assert later.value is None
        assert (await _kept(turn))["texts"][str(later.id)] == {"value": "Later: [EMAIL] again"}

    async def test_reflect_still_masks_the_stored_answer_when_the_ledger_keeps_it_masked(self):
        import uuid

        from django_ai_sdk.conversation.models import Message, Thread

        answer = "Call +31 6 12345678 for it."
        with override_settings(AI_SDK_OVERSIGHT={"CAPTURE": "masked"}):
            invalidate()
            session = await TurnSession.start(
                make_agent([], None, policy="test-masking"), channel="api", input="who?"
            )
            session.message_id = str(uuid.uuid4())
            await session.finish(answer)
            with patch("django_ai_sdk.oversight.record._enqueue_reflect"):
                turn_id = await session.close()
            stored = ChatMessage(id=session.message_id, role="assistant", content=answer)
            await Message.objects.acreate(
                id=session.message_id, thread=await Thread.objects.acreate(),
                result=stored.model_dump(),
            )
            turn = await Turn.objects.select_related("evidence").aget(id=turn_id)
            # The ledger never held the number; the thread did, and reflect fixes that copy.
            assert (await _kept(turn))["output"] == "Call [PHONE] for it."
            await areflect(turn_id)
        message = await Message.objects.aget(id=session.message_id)
        assert message.result["content"] == "Call [PHONE] for it."
        assert _leaks(json.dumps(await _kept(turn))) == []


@pytest.mark.django_db(transaction=True)
class TestWhoSeesWhat:
    async def test_the_detail_says_what_went_wrong_and_which_turn_it_ran_under(self):
        staff = await UserFactory.acreate(staff=True)
        turn = await run_turn("full")
        outer = uuid.uuid4()
        await Turn.objects.filter(id=turn.id).aupdate(error="gate: TimeoutError", parent_turn_id=outer)
        detail = await OversightService.get_turn(str(turn.id), user=staff, purpose="review")
        assert (detail.error, detail.parent_turn_id) == ("gate: TimeoutError", str(outer))
        # The client highlights lens spans only where the text is the text as said.
        assert detail.evidence is not None and detail.evidence["capture"] == "full"
        audit = await OversightService.get_turn(str(turn.id), user=staff, purpose="audit")
        assert audit.error == "gate: TimeoutError"

    async def test_review_merges_the_words_back_and_audit_reads_the_record(self):
        staff = await UserFactory.acreate(staff=True)
        turn = await run_turn("full")
        review = await OversightService.get_turn(str(turn.id), user=staff, purpose="review")
        shown = {o.lens: o for o in review.observations}
        # The Sources card reads these: same fields, now from the evidence.
        assert shown["taken"].value == [f"File of {NAME} {EMAIL}"]
        assert shown["taken"].items == {f"https://crm/{EMAIL}": 0.9}
        assert shown["who"].value == [NAME]
        assert shown["judged"].rationale.startswith("It quotes")
        assert shown["gist"].value.startswith("Asks to mail")
        audit = await OversightService.get_turn(str(turn.id), user=staff, purpose="audit")
        assert _leaks(audit.model_dump_json()) == []
        shown = {o.lens: o for o in audit.observations}
        assert (shown["taken"].value, shown["taken"].items) == ([FOUND], {})
        assert (shown["judged"].rationale, shown["gist"].value) == ("", None)

    async def test_a_note_lives_with_the_evidence_and_audit_never_reads_it(self):
        staff = await UserFactory.acreate(staff=True)
        turn = await run_turn("masked")
        item = await ReviewItem.objects.aget(turn=turn, queue="q")
        closed = await OversightService.resolve_review(
            str(item.id), VerdictIn(agree=True, note=f"Called {EMAIL}"), user=staff
        )
        assert closed.resolution["note"] == "Called [EMAIL]"
        await item.arefresh_from_db()
        assert set(item.resolution) == {"by", "value", "agree"}
        kept = await _kept(turn)
        assert kept["notes"] == {str(item.id): "Called [EMAIL]"}
        # The person's verdict carries the note as its reason, masked the same way.
        verdict = await Observation.objects.aget(turn=turn, lens="rule:queue_judged")
        assert kept["texts"][str(verdict.id)] == {"rationale": "Called [EMAIL]"}
        review = await OversightService.get_turn(str(turn.id), user=staff, purpose="review")
        assert review.reviews[0].resolution["note"] == "Called [EMAIL]"
        audit = await OversightService.get_turn(str(turn.id), user=staff, purpose="audit")
        assert "note" not in audit.reviews[0].resolution
        [listed] = await OversightService.list_reviews(user=staff, status="done", purpose="audit")
        assert "note" not in listed.resolution

    async def test_a_persons_names_are_counted_like_a_judges(self):
        staff = await UserFactory.acreate(staff=True)
        turn = await run_turn("full")
        out = await OversightService.add_verdict(
            str(turn.id), VerdictIn(lens="who", value=[NAME, "Piet"], note="two"), user=staff
        )
        assert (out.value, out.rationale) == ([NAME, "Piet"], "two")
        row = await Observation.objects.aget(id=out.id)
        assert row.value == [FOUND, FOUND]

    async def test_audit_never_shows_a_revealed_name(self):
        root = await UserFactory.acreate(staff=True, superuser=True)
        asker = await UserFactory.acreate()
        turn = await run_turn("full", user=asker)
        revealed = await OversightService.reveal(str(turn.id), user=root, reason="incident 7 check")
        assert revealed.username == asker.get_username()
        review = await OversightService.get_turn(str(turn.id), user=root, purpose="review")
        assert review.actor == asker.get_username() and review.revealed
        audit = await OversightService.get_turn(str(turn.id), user=root, purpose="audit")
        assert audit.actor == turn.actor_pseudonym and not audit.revealed
        assert asker.get_username() not in audit.model_dump_json()

    async def test_client_detail_names_nothing_a_mask_took_out(self):
        with override_settings(AI_SDK_OVERSIGHT={"CLIENT_DETAIL": "full"}):
            invalidate()
            storage = Storage()
            agent = make_agent(
                [HaystackChatMessage.from_assistant("ok")], storage, policy="test-privacy-naming"
            )
            parts = await chat(agent, f"Mail {EMAIL} please")
        [gate, *_] = oversight_parts(parts)
        assert gate["input"] == "Mail [EMAIL] please"
        assert gate["detail"]["observations"][0]["value"] == [FOUND]
        assert EMAIL not in json.dumps(parts)
        assert EMAIL not in json.dumps([m.model_dump() for m in storage.stored], default=str)


@pytest.mark.django_db(transaction=True)
class TestAtRest:
    async def test_with_a_master_key_no_row_holds_the_text(self):
        staff = await UserFactory.acreate(staff=True)
        asker = await UserFactory.acreate()
        turn = await run_turn("full", thread_id="t-rest", user=asker, MASTER_KEY=MASTER)
        assert turn.evidence.content is None and turn.evidence.ciphertext
        item = await ReviewItem.objects.aget(turn=turn)
        with override_settings(AI_SDK_OVERSIGHT={"MASTER_KEY": MASTER}):
            await OversightService.resolve_review(
                str(item.id), VerdictIn(agree=True, note=f"Called {EMAIL}"), user=staff
            )
            kept = await _kept(turn)
        assert kept["notes"] == {str(item.id): f"Called {EMAIL}"}

        def raw() -> str:
            dump = []
            with connection.cursor() as cursor:
                for table in ("turns", "observations", "decisions", "reviews", "evidence"):
                    cursor.execute(f"select * from django_ai_sdk_oversight_{table}")  # noqa: S608
                    dump.extend(cursor.fetchall())
            return json.dumps(dump, default=str)

        assert _leaks(await sync_to_async(raw)()) == []


@pytest.mark.django_db(transaction=True)
class TestForgetting:
    async def test_forgetting_a_thread_forgets_who_asked(self):
        asker = await UserFactory.acreate()
        root = await UserFactory.acreate(staff=True, superuser=True)
        with override_settings(AI_SDK_OVERSIGHT={"MASTER_KEY": MASTER}):
            before = await run_turn("full", thread_id="t-f", user=asker, MASTER_KEY=MASTER)
            assert before.actor_ref
            assert await OversightService.forget_thread("t-f", user=root, reason="erasure request")
            await before.arefresh_from_db()
            assert before.actor_ref == ""
            revealed = await OversightService.reveal(str(before.id), user=root, reason="probe it")
            assert (revealed.forgotten, revealed.user_id, revealed.username) == (True, "", "")
            detail = await OversightService.get_turn(str(before.id), user=root)
            assert detail.actor_forgotten and detail.actor == before.actor_pseudonym
            assert (detail.evidence_state, _leaks(detail.model_dump_json())) == ("shredded", [])
            # A new turn in the thread keeps nothing of it, whatever the capture.
            for capture in ("none", "full"):
                after = await run_turn(capture, thread_id="t-f", user=asker, MASTER_KEY=MASTER)
                assert (after.actor_ref, after.evidence_id) == ("", None)
                revealed = await OversightService.reveal(str(after.id), user=root,
                                                         reason="probe it again")
                assert revealed.forgotten
        assert (await sync_to_async(ledger.verify)()).ok

    async def test_forgetting_without_a_master_key_clears_who_asked_too(self):
        asker = await UserFactory.acreate()
        turn = await run_turn("full", thread_id="t-g", user=asker)
        assert turn.actor_ref == f"plain:{asker.pk}"
        await sync_to_async(shred_scope)("thread:t-g")
        turn = await Turn.objects.select_related("evidence").aget(pk=turn.pk)
        assert (turn.actor_ref, turn.evidence.content) == ("", None)


@pytest.mark.django_db(transaction=True)
class TestVerify:
    def test_deleting_the_newest_entries_is_caught(self):
        for reason in "abcde":
            _record(reason)
        for entry in LedgerEntry.objects.filter(seq__gt=3):
            AccessRecord.objects.filter(pk=entry.ref).delete()
            entry.delete()
        result = ledger.verify()
        assert [(p["seq"], p["problem"]) for p in result.problems] == [(5, "chain ends early")]
        # Rolled back with them, the head says nothing: only a published checkpoint can.
        tip = LedgerEntry.objects.get(seq=3).entry_hash
        LedgerHead.objects.filter(id=1).update(seq=3, entry_hash=tip)
        assert ledger.verify().ok

    def test_a_head_that_disagrees_with_its_entry_is_caught(self):
        _record("a")
        LedgerHead.objects.filter(id=1).update(entry_hash="0" * 64)
        assert [p["problem"] for p in ledger.verify().problems] == ["head mismatch"]

    async def test_readable_evidence_must_hash_to_what_the_turn_committed_to(self):
        turn = await run_turn("full")
        evidence = turn.evidence
        problems = sync_to_async(lambda: [p["problem"] for p in ledger.verify().problems])
        # What judges and reviewers add is beside the hash.
        await Evidence.objects.filter(pk=evidence.pk).aupdate(
            content={**evidence.content, "notes": {"x": "a note"}}
        )
        assert await problems() == []
        await Evidence.objects.filter(pk=evidence.pk).aupdate(
            content={**evidence.content, "input": "rewritten"}
        )
        assert await problems() == ["evidence altered"]
        await Evidence.objects.filter(pk=evidence.pk).aupdate(content=None)
        assert await problems() == ["evidence missing"]

    async def test_sealed_evidence_is_checked_where_it_opens(self):
        turn = await run_turn("full", thread_id="t-v", MASTER_KEY=MASTER)
        problems = sync_to_async(lambda: [p["problem"] for p in ledger.verify().problems])
        with override_settings(AI_SDK_OVERSIGHT={"MASTER_KEY": MASTER}):
            content = await _kept(turn)
            assert await problems() == []
            _, blob = await sync_to_async(vault.encrypt_json)(
                "thread:t-v", {**content, "output": "rewritten"}
            )
            await Evidence.objects.filter(pk=turn.evidence_id).aupdate(ciphertext=blob)
            assert await problems() == ["evidence altered"]
        # Under another key there is nothing to compare: not a problem of the chain.
        with override_settings(AI_SDK_OVERSIGHT={"MASTER_KEY": OTHER}):
            assert await problems() == []


@pytest.mark.django_db(transaction=True)
class TestAnUnusableMasterKey:
    async def test_reading_says_unreadable_instead_of_failing(self):
        root = await UserFactory.acreate(staff=True, superuser=True)
        turn = await run_turn("full", thread_id="t-u", user=await UserFactory.acreate(),
                              MASTER_KEY=MASTER)
        with override_settings(AI_SDK_OVERSIGHT={"MASTER_KEY": "not base64!"}):
            detail = await OversightService.get_turn(str(turn.id), user=root)
            assert (detail.evidence, detail.evidence_state) == (None, "unreadable")
            page = await OversightService.list_turns(user=root)
            assert page.items[0].preview is None
            revealed = await OversightService.reveal(str(turn.id), user=root, reason="who was it")
            assert (revealed.user_id, revealed.forgotten) == ("", False)
            assert (await OversightService.verify(user=root)).ok


class TestTheRecordsValue:
    def test_an_entities_value_keeps_labels_and_counts_names(self):
        from django_ai_sdk.oversight.packs.quality import TakenFrom
        from django_ai_sdk.oversight.packs.security import InputPII

        titles = Answer(value=[f"Dossier {EMAIL} (BSN {BSN})"], answerer="typesafe")
        assert stored_value(TakenFrom(), titles) == [FOUND]
        # A detector reports kinds; a person, like a judge, may type anything.
        assert stored_value(InputPII(), Answer(value=["email"], answerer="pii")) == ["email"]
        assert stored_value(InputPII(), Answer(value=["email"], answerer="human")) == [FOUND]
        labelled = Entities(name="kind", question="?", labels=["hr", "pay"])
        assert stored_value(labelled, Answer(value=["hr", NAME], answerer="judge")) == ["hr", FOUND]
        assert stored_value(Text(name="t", question="?"), Answer(value=NAME)) is None
        assert stored_value(Flag(name="f", question="?"), Answer(value=True)) is True

    def test_a_reason_and_its_scores_are_kept_as_capture_says(self):
        lens = Flag(name="f", question="?")
        answer = Answer(value=True, confidence=0.9, rationale=f"It quotes {IBAN}.",
                        items={f"https://crm/{EMAIL}": 0.95})
        columns, masked = split_answer(lens, "finish", answer, keep=keeper("masked"))
        assert "rationale" not in columns and "items" not in columns
        assert masked == {"rationale": "It quotes [IBAN].", "items": {"https://crm/[EMAIL]": 0.95}}
        assert split_answer(lens, "finish", answer, keep=keeper("full"))[1] == {
            "rationale": f"It quotes {IBAN}.",
            "items": {f"https://crm/{EMAIL}": 0.95},
        }
        for capture in ("metadata", "none"):
            assert split_answer(lens, "finish", answer, keep=keeper(capture))[1] == {}

    def test_an_earlier_answerers_names_stay_out_of_the_trail(self):
        lens = Entities(name="who", question="?")
        earlier = Answer(value=[NAME], confidence=0.4, answerer="judge")
        answer = Answer(value=[NAME, "Piet"], answerer="strong", trail=[earlier])
        columns, kept = split_answer(lens, "reflect", answer, keep=keeper("full"))
        assert [step["value"] for step in columns["trail"]] == [[FOUND]]
        assert kept == {"value": [NAME, "Piet"], "trail": [[NAME]]}
