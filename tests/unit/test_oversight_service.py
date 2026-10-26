"""OversightService: who reads what, through which purpose, and that every read is recorded."""

from __future__ import annotations

import pytest
from asgiref.sync import sync_to_async
from django_ai_sdk.errors import UserError
from django_ai_sdk.oversight import ledger
from django_ai_sdk.oversight.models import AccessRecord, PolicyOverride, ReviewItem
from django_ai_sdk.oversight.policy import invalidate
from django_ai_sdk.oversight.schemas import OverrideIn, VerdictIn
from django_ai_sdk.oversight.services import OversightService
from django_ai_sdk.oversight.session import TurnSession
from django_ai_sdk.permissions import PermissionDenied
from tests.factories.db import UserFactory
from tests.unit.test_oversight_turns import make_agent  # registers the test-guard policy


async def a_turn(text: str = "What about my salary?", output: str = "Ask jan@example.com.",
                 user=None) -> str:
    session = await TurnSession.start(make_agent([], None), channel="api", input=text, user=user)
    await session.gate()
    await session.review(output)
    from unittest.mock import patch

    with patch("django_ai_sdk.oversight.session._enqueue_reflect"):
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

    async def test_security_sees_the_question_and_a_pseudonym(self):
        asker = await UserFactory.acreate()
        staff = await UserFactory.acreate(staff=True)
        turn_id = await a_turn(user=asker)
        page = await OversightService.list_turns(user=staff, purpose="security")
        [row] = page.items
        assert row.preview == "What about my salary?"
        assert row.actor.startswith("u-")
        assert row.labels["topic"] == "hr"
        assert {h.rule for h in row.hits} >= {"hr_to_people", "would_block_hr"}
        detail = await OversightService.get_turn(turn_id, user=staff, purpose="security")
        assert detail.evidence["input"] == "What about my salary?"
        assert "sources" not in detail.evidence
        assert detail.chain.intact is True
        assert detail.chain.seq is not None

    async def test_quality_hides_who_asked(self):
        staff = await UserFactory.acreate(staff=True)
        turn_id = await a_turn(user=await UserFactory.acreate())
        detail = await OversightService.get_turn(turn_id, user=staff, purpose="quality")
        assert detail.actor == "hidden"
        assert "sources" in detail.evidence

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
        await OversightService.list_turns(user=staff, outcome="flagged")
        await OversightService.get_turn(turn_id, user=staff)
        actions = [r.action async for r in AccessRecord.objects.order_by("created_at")]
        assert actions == ["list", "view"]
        assert (await sync_to_async(ledger.verify)()).ok

    async def test_filters(self):
        staff = await UserFactory.acreate(staff=True)
        await a_turn()
        await a_turn("hello there", "Hi!")
        assert (await OversightService.list_turns(user=staff)).total == 2
        assert (await OversightService.list_turns(user=staff, label="topic", value="hr")).total == 1
        would = await OversightService.list_turns(user=staff, shadow_outcome="blocked")
        assert would.total == 1
        assert (await OversightService.list_turns(user=staff, review="open")).total == 1

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
        detail = await OversightService.get_turn(item.turn_id, user=staff, purpose="quality")
        human = [o for o in detail.observations if o.reviewer]
        assert human[0].lens == "rule:hr_to_people"
        assert human[0].value is True

    async def test_a_verdict_on_a_lens(self):
        staff = await UserFactory.acreate(staff=True)
        turn_id = await a_turn()
        obs = await OversightService.add_verdict(
            turn_id, VerdictIn(lens="topic", value="other"), user=staff
        )
        assert (obs.value, obs.answerer) == ("other", "human")
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
                            value="enforce", note="tested in shadow")
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
        assert rule["mode"] == "shadow"
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
        assert overview.by_shadow_outcome == {"blocked": 1}
        assert any(r["rule"] == "would_block_hr" and r.get("shadow") for r in overview.rules)
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


@pytest.mark.django_db(transaction=True)
def test_the_ninja_router(client):
    from django.test import Client
    from django.urls import path
    from ninja import NinjaAPI

    from django_ai_sdk.contrib.ninja import get_oversight_router, register_error_handlers

    from ninja.security import SessionAuth

    api = NinjaAPI(auth=SessionAuth(), urls_namespace="oversight-test")
    register_error_handlers(api)
    api.add_router("/", get_oversight_router())
    urlpatterns = [path("api/", api.urls)]

    user = UserFactory.create(staff=True)
    browser = Client()
    browser.force_login(user)
    from django.test import override_settings

    with override_settings(ROOT_URLCONF=type("urls", (), {"urlpatterns": urlpatterns})):
        first = browser.get("/api/oversight/turns/")
        assert first.status_code == 200, first.content
        assert first.json()["total"] == 0
        assert browser.get("/api/oversight/turns/x/").status_code == 404
        assert browser.get("/api/oversight/turns/?purpose=bogus").status_code == 400
        assert browser.get("/api/oversight/policies/").status_code == 200
        response = browser.post("/api/oversight/turns/x/reveal/", {"reason": "because of reasons"},
                                content_type="application/json")
        assert response.status_code == 403
