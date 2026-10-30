"""OversightService: who reads what, through which purpose, and that every read is recorded."""

from __future__ import annotations

from datetime import timedelta

import pytest
from asgiref.sync import sync_to_async
from django.test import override_settings
from django.utils import timezone
from django_ai_sdk.errors import UserError
from django_ai_sdk.oversight import ledger
from django_ai_sdk.oversight.models import AccessRecord, PolicyOverride, ReviewItem, Turn
from django_ai_sdk.oversight.policy import invalidate
from django_ai_sdk.oversight.schemas import OverrideIn, TurnFilter, VerdictIn
from django_ai_sdk.oversight.services import OversightService
from django_ai_sdk.oversight.session import TurnSession
from django_ai_sdk.permissions import PermissionDenied
from tests.factories.db import UserFactory
from tests.unit.test_oversight_turns import make_agent  # registers the test-guard policy


async def a_turn(text: str = "What about my salary?", output: str = "Ask jan@example.com.",
                 user=None) -> str:
    session = await TurnSession.start(make_agent([], None), channel="api", input=text, user=user)
    await session.gate()
    await session.finish(output)
    from unittest.mock import patch

    with patch("django_ai_sdk.oversight.record._enqueue_reflect"):
        return await session.close()


@pytest.fixture(autouse=True)
def _fresh():
    invalidate()


@pytest.mark.django_db(transaction=True)
class TestReading:
    async def test_members_see_nothing(self):
        member = await UserFactory.acreate()
        with pytest.raises(PermissionDenied):
            await OversightService.list_turns(user=member)

    async def test_review_sees_the_turn_and_a_pseudonym(self):
        asker = await UserFactory.acreate()
        staff = await UserFactory.acreate(staff=True)
        with override_settings(AI_SDK_OVERSIGHT={"CAPTURE": "full"}):
            invalidate()
            turn_id = await a_turn(user=asker)
        invalidate()
        page = await OversightService.list_turns(user=staff, purpose="review")
        [row] = page.items
        assert row.preview == "What about my salary?"
        assert row.actor.startswith("u-")
        assert row.labels["topic"] == "hr"
        assert {h.rule for h in row.hits} >= {"hr_to_people", "would_block_hr"}
        detail = await OversightService.get_turn(turn_id, user=staff, purpose="review")
        assert detail.evidence["input"] == "What about my salary?"
        # Sources are where an indirect injection hides: the reviewer sees them.
        assert "sources" in detail.evidence
        assert detail.actor == row.actor
        assert detail.chain.intact is True
        assert detail.chain.seq is not None

    async def test_the_old_purposes_are_gone(self):
        staff = await UserFactory.acreate(staff=True)
        with pytest.raises(UserError):
            await OversightService.list_turns(user=staff, purpose="security")

    async def test_audit_sees_no_content_and_no_reasons(self):
        staff = await UserFactory.acreate(staff=True)
        turn_id = await a_turn()
        detail = await OversightService.get_turn(turn_id, user=staff, purpose="audit")
        assert detail.evidence is None
        assert detail.evidence_state == "withheld"
        assert all(o.rationale == "" for o in detail.observations)
        assert detail.decisions

    async def test_every_read_is_on_the_ledger(self):
        staff = await UserFactory.acreate(staff=True)
        turn_id = await a_turn()
        await OversightService.list_turns(TurnFilter(outcome="flagged"), user=staff)
        await OversightService.get_turn(turn_id, user=staff)
        records = [r async for r in AccessRecord.objects.order_by("created_at")]
        assert [r.action for r in records] == ["list", "view"]
        assert records[0].query == {"outcome": "flagged"}
        assert (await sync_to_async(ledger.verify)()).ok

    async def test_filters(self):
        staff = await UserFactory.acreate(staff=True)
        await a_turn()
        await a_turn("hello there", "Hi!")
        async def total(**fields):
            return (await OversightService.list_turns(TurnFilter(**fields), user=staff)).total

        assert await total() == 2
        assert await total(label="topic", value="hr") == 1
        assert await total(report_outcome="blocked") == 1
        assert await total(review="open") == 1
        assert await total(until=timezone.now() - timedelta(days=1)) == 0
        assert await total(since=timezone.now() - timedelta(days=1)) == 2

    async def test_an_unknown_turn_is_none(self):
        staff = await UserFactory.acreate(staff=True)
        assert await OversightService.get_turn("nope", user=staff) is None


@pytest.mark.django_db(transaction=True)
class TestReveal:
    async def test_only_superusers_with_a_reason(self):
        asker = await UserFactory.acreate()
        staff = await UserFactory.acreate(staff=True)
        root = await UserFactory.acreate(superuser=True, staff=True)
        turn_id = await a_turn(user=asker)
        with pytest.raises(PermissionDenied):
            await OversightService.reveal(turn_id, user=staff, reason="I want to know")
        with pytest.raises(UserError):
            await OversightService.reveal(turn_id, user=root, reason="x")
        revealed = await OversightService.reveal(turn_id, user=root, reason="incident 42 follow-up")
        assert revealed.user_id == str(asker.pk)
        record = await AccessRecord.objects.aget(action="reveal")
        assert record.reason == "incident 42 follow-up"
        detail = await OversightService.get_turn(turn_id, user=root)
        assert detail.revealed

    async def test_a_reveal_follows_the_reader_not_their_name(self):
        turn_id = await a_turn(user=await UserFactory.acreate())
        root = await UserFactory.acreate(superuser=True, staff=True)
        other = await UserFactory.acreate(superuser=True, staff=True)
        await OversightService.reveal(turn_id, user=root, reason="incident 42 follow-up")
        setattr(root, root.USERNAME_FIELD, "renamed@example.com")
        await root.asave()
        assert (await OversightService.get_turn(turn_id, user=root)).revealed
        assert not (await OversightService.get_turn(turn_id, user=other)).revealed


@pytest.mark.django_db(transaction=True)
class TestReviews:
    async def test_a_verdict_closes_the_item_and_lands_as_a_human_answer(self):
        staff = await UserFactory.acreate(staff=True)
        await a_turn()
        [item] = await OversightService.list_reviews(user=staff, queue="hr")
        assert item.turn is not None
        done = await OversightService.resolve_review(
            item.id, VerdictIn(agree=True, note="right call"), user=staff
        )
        assert done.status == "done"
        assert (await ReviewItem.objects.aget()).status == "done"
        detail = await OversightService.get_turn(item.turn_id, user=staff, purpose="review")
        human = [o for o in detail.observations if o.reviewer]
        assert human[0].lens == "rule:hr_to_people"
        assert human[0].value is True

    async def test_reading_the_queue_is_checked_and_recorded(self):
        from unittest.mock import AsyncMock, patch

        staff = await UserFactory.acreate(staff=True)
        await a_turn()
        await OversightService.list_reviews(user=staff, queue="hr")
        record = await AccessRecord.objects.aget(action="reviews")
        assert (record.purpose, record.query) == (
            "review", {"queue": "hr", "status": "open", "offset": 0})
        refused = AsyncMock(side_effect=PermissionDenied("no"))
        with patch.object(OversightService, "_check_purpose", refused):
            with pytest.raises(PermissionDenied):
                await OversightService.list_reviews(user=staff, purpose="audit")
        refused.assert_awaited_once()

    async def test_a_verdict_on_a_lens(self):
        staff = await UserFactory.acreate(staff=True)
        turn_id = await a_turn()
        obs = await OversightService.add_verdict(
            turn_id, VerdictIn(lens="topic", value="other"), user=staff
        )
        assert (obs.value, obs.answerer) == ("other", "human")
        # A person's answer is not a reflection: reflect still runs later.
        assert (await Turn.objects.aget(id=turn_id)).reflected_at is None
        with pytest.raises(UserError):
            await OversightService.add_verdict(turn_id, VerdictIn(lens="topic", value="zz"),
                                               user=staff)

    async def test_queues(self):
        staff = await UserFactory.acreate(staff=True)
        await a_turn()
        assert await OversightService.queues(user=staff) == {"hr": 1}


@pytest.mark.django_db(transaction=True)
class TestPolicies:
    async def test_describe_with_sources(self):
        staff = await UserFactory.acreate(staff=True)
        policy = await OversightService.get_policy("test-guard", user=staff)
        lens = next(l for l in policy["lenses"] if l["name"] == "pii")
        assert lens["sources"]["sample"] == "code"
        summaries = await OversightService.list_policies(user=staff)
        assert {"default", "security", "test-guard"} <= {p.name for p in summaries}

    async def test_overrides_need_a_superuser_and_land_on_the_chain(self):
        staff = await UserFactory.acreate(staff=True)
        root = await UserFactory.acreate(superuser=True, staff=True)
        change = OverrideIn(target_kind="rule", target="would_block_hr", field="mode",
                            value="enforce", note="tested in report mode")
        with pytest.raises(PermissionDenied):
            await OversightService.set_override("test-guard", change, user=staff)
        policy = await OversightService.set_override("test-guard", change, user=root)
        rule = next(r for r in policy["rules"] if r["name"] == "would_block_hr")
        assert (rule["mode"], rule["sources"]["mode"]) == ("enforce", "db")

        session = await TurnSession.start(make_agent([], None), channel="api", input="my salary?")
        assert (await session.gate()).blocked

        reset = OverrideIn(target_kind="rule", target="would_block_hr", field="mode", reset=True)
        policy = await OversightService.set_override("test-guard", reset, user=root)
        rule = next(r for r in policy["rules"] if r["name"] == "would_block_hr")
        assert rule["mode"] == "report"
        assert await PolicyOverride.objects.acount() == 2
        assert (await sync_to_async(ledger.verify)()).ok

    async def test_a_structural_change_is_refused(self):
        root = await UserFactory.acreate(superuser=True, staff=True)
        change = OverrideIn(target_kind="rule", target="hr_to_people", field="when",
                            value={"lens": "pii", "op": "any"})
        with pytest.raises(UserError):
            await OversightService.set_override("test-guard", change, user=root)


@pytest.mark.django_db(transaction=True)
class TestLedgerAndInsight:
    async def test_overview(self):
        staff = await UserFactory.acreate(staff=True)
        await a_turn()
        overview = await OversightService.overview(user=staff)
        assert overview.turns == 1
        assert overview.by_report_outcome == {"blocked": 1}
        assert any(r["rule"] == "would_block_hr" and r.get("report") for r in overview.rules)
        assert overview.labels["topic"] == {"hr": 1}

    async def test_verify_and_seal(self):
        staff = await UserFactory.acreate(staff=True)
        root = await UserFactory.acreate(superuser=True, staff=True)
        await a_turn()
        checkpoint = await OversightService.seal_now(user=root)
        assert checkpoint.tree_size > 0
        result = await OversightService.verify(user=staff)
        assert (result.ok, result.checkpoint_ok) == (True, True)
        assert [c.tree_size for c in await OversightService.checkpoints(user=staff)] == [
            checkpoint.tree_size
        ]

    async def test_export_streams_rows_through_the_purpose(self):
        root = await UserFactory.acreate(superuser=True, staff=True)
        await a_turn()
        rows = [row async for row in await OversightService.export(user=root, purpose="audit")]
        assert len(rows) == 1
        assert rows[0]["preview"] is None

    async def test_forget_a_thread(self):
        root = await UserFactory.acreate(superuser=True, staff=True)
        assert await OversightService.forget_thread("t-1", user=root, reason="erasure request")
        assert not await OversightService.forget_thread("t-1", user=root, reason="erasure again")


async def total(staff, **fields) -> int:
    return (await OversightService.list_turns(TurnFilter(**fields), user=staff)).total


@pytest.mark.django_db(transaction=True)
class TestFilters:
    async def test_a_label_filter_names_a_lens_only(self):
        staff = await UserFactory.acreate(staff=True)
        await a_turn()
        for bad in ["topic__contains", "topic\n", "a b", "rule:hr_to_people"]:
            with pytest.raises(UserError):
                await total(staff, label=bad)
        assert await total(staff, label="topic") == 1

    async def test_reviewed_and_unreviewed(self):
        staff = await UserFactory.acreate(staff=True)
        first = await a_turn()
        second = await a_turn("hello there", "Hi!")
        assert await total(staff, review="reviewed") == 0
        assert await total(staff, review="unreviewed") == 2
        # Both turns have machine answers; only the first a person's.
        await OversightService.add_verdict(first, VerdictIn(lens="topic", value="other"),
                                           user=staff)
        page = await OversightService.list_turns(TurnFilter(review="reviewed"), user=staff)
        assert [row.id for row in page.items] == [first]
        page = await OversightService.list_turns(TurnFilter(review="unreviewed"), user=staff)
        assert [row.id for row in page.items] == [second]

    async def test_limits_are_capped(self):
        from django_ai_sdk.oversight.schemas import MAX_LIMIT

        staff = await UserFactory.acreate(staff=True)
        page = await OversightService.list_turns(user=staff, limit=10_000, offset=-5)
        assert (page.limit, page.offset) == (MAX_LIMIT, 0)


@pytest.mark.django_db(transaction=True)
class TestVerdicts:
    async def test_a_call_on_a_rule_is_not_a_label(self):
        staff = await UserFactory.acreate(staff=True)
        await a_turn()
        [item] = await OversightService.list_reviews(user=staff, queue="hr")
        await OversightService.resolve_review(item.id, VerdictIn(agree=True), user=staff)
        turn = await Turn.objects.aget(id=item.turn_id)
        assert turn.labels["topic"] == "hr"
        assert not [k for k in turn.labels if k.startswith("rule:")]
        overview = await OversightService.overview(user=staff)
        assert not [k for k in overview.labels if k.startswith("rule:")]
        assert not [s for s in overview.lenses if s["lens"].startswith("rule:")]

    async def test_a_review_is_resolved_once(self):
        from django_ai_sdk.oversight.models import Observation
        from django_ai_sdk.permissions import ConflictError

        staff = await UserFactory.acreate(staff=True)
        await a_turn()
        [item] = await OversightService.list_reviews(user=staff, queue="hr")
        await OversightService.resolve_review(item.id, VerdictIn(agree=True), user=staff)
        with pytest.raises(ConflictError):
            await OversightService.resolve_review(item.id, VerdictIn(agree=False), user=staff)
        with pytest.raises(ConflictError):
            await OversightService.resolve_review(item.id, VerdictIn(), user=staff, dismiss=True)
        assert (await ReviewItem.objects.aget()).resolution["agree"] is True
        assert await Observation.objects.exclude(reviewer="").acount() == 1

    async def test_two_people_resolving_at_once_answer_once(self):
        import asyncio

        from django_ai_sdk.oversight.models import Observation
        from django_ai_sdk.permissions import ConflictError

        staff = await UserFactory.acreate(staff=True)
        other = await UserFactory.acreate(staff=True)
        await a_turn()
        [item] = await OversightService.list_reviews(user=staff, queue="hr")
        results = await asyncio.gather(
            OversightService.resolve_review(item.id, VerdictIn(agree=True), user=staff),
            OversightService.resolve_review(item.id, VerdictIn(agree=False), user=other),
            return_exceptions=True,
        )
        assert sum(isinstance(r, ConflictError) for r in results) == 1
        assert await Observation.objects.exclude(reviewer="").acount() == 1

    async def test_a_verdict_names_a_lens_of_the_policy_or_a_rule_that_matched(self):
        from django_ai_sdk.oversight.models import Observation

        staff = await UserFactory.acreate(staff=True)
        turn_id = await a_turn()
        with pytest.raises(UserError):
            await OversightService.add_verdict(turn_id, VerdictIn(lens="made_up", value=True),
                                               user=staff)
        with pytest.raises(UserError):
            await OversightService.add_verdict(
                turn_id, VerdictIn(lens="rule:block_injection", agree=True), user=staff
            )
        assert not await Observation.objects.exclude(reviewer="").aexists()
        obs = await OversightService.add_verdict(
            turn_id, VerdictIn(lens="rule:hr_to_people", agree=False), user=staff
        )
        assert obs.value is False


@pytest.mark.django_db(transaction=True)
class TestNotes:
    async def test_audit_sees_decisions_not_notes(self):
        staff = await UserFactory.acreate(staff=True)
        await a_turn()
        [item] = await OversightService.list_reviews(user=staff, queue="hr")
        await OversightService.resolve_review(
            item.id, VerdictIn(agree=True, note="the asker is Jan"), user=staff
        )
        audit = await OversightService.get_turn(item.turn_id, user=staff, purpose="audit")
        [review] = audit.reviews
        assert review.resolution["agree"] is True
        assert "note" not in review.resolution
        assert all(o.rationale == "" for o in audit.observations)
        [listed] = await OversightService.list_reviews(user=staff, status="done", purpose="audit")
        assert "note" not in listed.resolution
        shown = await OversightService.get_turn(item.turn_id, user=staff, purpose="review")
        assert shown.reviews[0].resolution["note"] == "the asker is Jan"

    async def test_a_text_verdict_stays_with_the_evidence(self):
        from django_ai_sdk.oversight import Text, register
        from django_ai_sdk.oversight.policy import Policy

        @register
        class Notes(Policy):
            name = "test-notes"
            lenses = [Text(name="gist", question="?", answerers=["human"])]  # noqa: RUF012

        staff = await UserFactory.acreate(staff=True)
        session = await TurnSession.start(
            make_agent([], None, policy="test-notes"), channel="api", input="Jan?"
        )
        await session.gate()
        from unittest.mock import patch

        with patch("django_ai_sdk.oversight.record._enqueue_reflect"):
            turn_id = await session.close()
        item = await ReviewItem.objects.acreate(turn_id=turn_id, queue="notes", lens="gist")
        await OversightService.resolve_review(
            str(item.id), VerdictIn(value="asks about Jan's pay"), user=staff
        )
        await item.arefresh_from_db()
        assert item.resolution["value"] is None
        detail = await OversightService.get_turn(turn_id, user=staff, purpose="review")
        assert next(o for o in detail.observations if o.reviewer).value == "asks about Jan's pay"
        audit = await OversightService.get_turn(turn_id, user=staff, purpose="audit")
        assert next(o for o in audit.observations if o.reviewer).value is None

    async def test_a_forgotten_thread_shows_no_notes(self):
        from unittest.mock import patch

        staff = await UserFactory.acreate(staff=True)
        root = await UserFactory.acreate(superuser=True, staff=True)
        session = await TurnSession.start(
            make_agent([], None), channel="api", input="What about my salary?", thread_id="t-9"
        )
        await session.gate()
        await session.finish("Ask HR.")
        with patch("django_ai_sdk.oversight.record._enqueue_reflect"):
            turn_id = await session.close()
        [item] = await OversightService.list_reviews(user=staff, queue="hr")
        await OversightService.resolve_review(
            item.id, VerdictIn(agree=True, note="Jan asked twice"), user=staff
        )
        await OversightService.forget_thread("t-9", user=root, reason="erasure request")
        detail = await OversightService.get_turn(turn_id, user=staff, purpose="review")
        assert detail.evidence_state == "shredded"
        assert "note" not in detail.reviews[0].resolution
        assert all(o.rationale == "" for o in detail.observations)
        [listed] = await OversightService.list_reviews(user=staff, status="done")
        assert "note" not in listed.resolution

    async def test_a_forgotten_actor_is_a_flag_not_a_label(self):
        asker = await UserFactory.acreate()
        root = await UserFactory.acreate(superuser=True, staff=True)
        turn_id = await a_turn(user=asker)
        await OversightService.reveal(turn_id, user=root, reason="incident 42 follow-up")
        # As if the actor's key was shredded.
        await Turn.objects.filter(id=turn_id).aupdate(actor_ref="aead:AAAA")
        detail = await OversightService.get_turn(turn_id, user=root)
        assert detail.actor.startswith("u-") and detail.actor_forgotten


@pytest.mark.django_db(transaction=True)
class TestReadsAtScale:
    async def test_a_polled_read_is_recorded_once(self):
        staff = await UserFactory.acreate(staff=True)
        turn_id = await a_turn()
        for _ in range(3):
            await OversightService.get_turn(turn_id, user=staff)
            await OversightService.list_turns(user=staff)
        await OversightService.list_turns(TurnFilter(outcome="flagged"), user=staff)
        await OversightService.get_turn(turn_id, user=staff, purpose="audit")
        records = [(r.action, r.purpose) async for r in AccessRecord.objects.order_by("created_at")]
        assert records == [("view", "review"), ("list", "review"), ("list", "review"),
                           ("view", "audit")]
        assert (await sync_to_async(ledger.verify)()).ok

    async def test_the_overview_rates_only_answers_and_filters_queues(self):
        from unittest.mock import patch

        from django_ai_sdk.oversight.models import Observation

        staff = await UserFactory.acreate(staff=True)
        turn_id = await a_turn()
        await Observation.objects.acreate(turn_id=turn_id, lens="probe", stage="gate",
                                          answerer="x", value=True)
        for _ in range(3):
            await Observation.objects.acreate(turn_id=turn_id, lens="probe", stage="gate",
                                              answerer="x", error="timeout")
        overview = await OversightService.overview(user=staff)
        probe = next(s for s in overview.lenses if s["lens"] == "probe")
        assert (probe["answers"], probe["errors"], probe["positive_rate"]) == (4, 3, 1.0)
        assert overview.queues == {"hr": 1}
        assert (await OversightService.overview(user=staff, policy="other")).queues == {}
        assert not overview.sampled
        with patch("django_ai_sdk.oversight.metrics.MAX_ROWS", 1):
            assert (await OversightService.overview(user=staff)).sampled


MASTER = "a2tra2tra2tra2tra2tra2tra2tra2tra2tra2tra2s="  # base64 of b"k" * 32
OTHER = "b29vb29vb29vb29vb29vb29vb29vb29vb29vb29vb28="  # base64 of b"o" * 32


@pytest.mark.django_db(transaction=True)
class TestAnotherMasterKey:
    """Content sealed under another MASTER_KEY is there, unreadable here: not forgotten."""

    async def test_the_evidence_reads_as_unreadable_not_shredded(self):
        from django.test import override_settings

        staff = await UserFactory.acreate(staff=True)
        with override_settings(AI_SDK_OVERSIGHT={"MASTER_KEY": MASTER}):
            turn_id = await a_turn()
        with override_settings(AI_SDK_OVERSIGHT={"MASTER_KEY": OTHER}):
            detail = await OversightService.get_turn(turn_id, user=staff, purpose="review")
        assert (detail.evidence, detail.evidence_state) == (None, "unreadable")

    async def test_who_asked_reads_as_sealed_under_another_key_not_forgotten(self):
        from django.test import override_settings

        asker = await UserFactory.acreate()
        root = await UserFactory.acreate(superuser=True, staff=True)
        with override_settings(AI_SDK_OVERSIGHT={"MASTER_KEY": MASTER}):
            turn_id = await a_turn(user=asker)
            await OversightService.reveal(turn_id, user=root, reason="incident 42 follow-up")
        with override_settings(AI_SDK_OVERSIGHT={"MASTER_KEY": OTHER}):
            detail = await OversightService.get_turn(turn_id, user=root)
        assert detail.actor.endswith("(sealed under another MASTER_KEY)")
        assert not detail.actor_forgotten


@pytest.mark.django_db(transaction=True)
class TestForgettingNotes:
    """What a reviewer wrote lives in the turn's evidence, and goes with it."""

    async def _noted(self, thread_id: str = "") -> ReviewItem:
        staff = await UserFactory.acreate(staff=True)
        session = await TurnSession.start(
            make_agent([], None), channel="api", input="What about my salary?",
            thread_id=thread_id or None,
        )
        await session.gate()
        await session.finish("Ask HR.")
        from unittest.mock import patch

        with patch("django_ai_sdk.oversight.record._enqueue_reflect"):
            await session.close()
        [item] = await OversightService.list_reviews(user=staff, queue="hr")
        closed = await OversightService.resolve_review(
            item.id, VerdictIn(agree=True, note="Jan asked twice"), user=staff
        )
        assert closed.resolution["note"] == "Jan asked twice"
        return await ReviewItem.objects.select_related("turn__evidence").aget(id=item.id)

    async def _notes(self, item: ReviewItem) -> dict:
        from django_ai_sdk.oversight.tasks import load_evidence

        turn = await Turn.objects.select_related("evidence__key").aget(id=item.turn_id)
        return ((await sync_to_async(load_evidence)(turn)) or {}).get("notes") or {}

    async def test_a_note_is_kept_with_the_evidence_not_on_the_item(self):
        item = await self._noted("t-notes")
        assert item.resolution == {"by": item.resolution["by"], "value": True, "agree": True}
        assert await self._notes(item) == {str(item.id): "Jan asked twice"}

    async def test_forgetting_a_thread_takes_its_notes(self):
        from django_ai_sdk.oversight.tasks import shred_scope

        item = await self._noted("t-notes")
        await sync_to_async(shred_scope)("thread:t-notes")
        assert await self._notes(item) == {}
        await item.arefresh_from_db()
        assert item.resolution["agree"] is True and "note" not in item.resolution
        assert (await sync_to_async(ledger.verify)()).ok

    async def test_expired_content_takes_its_notes(self):
        from django_ai_sdk.oversight.tasks import prune

        item = await self._noted()
        # Content is kept 90 days, metadata 730: in between, only the content goes.
        await sync_to_async(prune)(now=timezone.now() + timedelta(days=100))
        assert await self._notes(item) == {}
        staff = await UserFactory.acreate(staff=True)
        detail = await OversightService.get_turn(str(item.turn_id), user=staff)
        assert "note" not in detail.reviews[0].resolution


@pytest.mark.django_db(transaction=True)
class TestCheckpointCheck:
    async def test_verify_says_when_the_latest_checkpoint_no_longer_matches(self):
        from django_ai_sdk.oversight.models import Checkpoint

        staff = await UserFactory.acreate(staff=True)
        root = await UserFactory.acreate(superuser=True, staff=True)
        await a_turn()
        await OversightService.seal_now(user=root)
        await Checkpoint.objects.all().aupdate(root_hash="0" * 64)
        result = await OversightService.verify(user=staff)
        assert (result.ok, result.checkpoint_ok) == (True, False)
