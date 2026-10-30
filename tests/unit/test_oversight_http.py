"""The ninja router: every console write over HTTP, who may make it, and the export."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from django.test import AsyncClient, override_settings
from django.urls import path
from django_ai_sdk.oversight.models import AccessRecord, PolicyOverride, Turn
from django_ai_sdk.oversight.policy import invalidate
from django_ai_sdk.oversight.session import agent_key
from tests.factories.db import UserFactory
from tests.unit.test_oversight_console import BLOCK_WORD
from tests.unit.test_oversight_service import a_turn
from tests.unit.test_oversight_turns import make_agent  # registers the test-guard policy


def _urls() -> type:
    from django_ai_sdk.contrib.ninja import get_oversight_router, register_error_handlers
    from ninja import NinjaAPI
    from ninja.security import SessionAuth

    api = NinjaAPI(auth=SessionAuth(), urls_namespace="oversight-http-test")
    register_error_handlers(api)
    api.add_router("/", get_oversight_router())
    return type("urls", (), {"urlpatterns": [path("api/", api.urls)]})


URLS = _urls()


@pytest.fixture(autouse=True)
def _routes():
    invalidate()
    with override_settings(ROOT_URLCONF=URLS):
        yield
    invalidate()


async def browser(**user_kwargs) -> AsyncClient:
    client = AsyncClient()
    await client.aforce_login(await UserFactory.acreate(**user_kwargs))
    return client


async def post(client: AsyncClient, url: str, body: dict | None = None):
    return await client.post(f"/api/oversight/{url}", body or {}, content_type="application/json")


@pytest.mark.django_db(transaction=True)
class TestReads:
    async def test_the_feed_and_its_errors(self):
        staff = await browser(staff=True)
        first = await staff.get("/api/oversight/turns/")
        assert first.status_code == 200, first.content
        assert first.json()["total"] == 0
        assert (await staff.get("/api/oversight/turns/x/")).status_code == 404
        assert (await staff.get("/api/oversight/turns/?purpose=bogus")).status_code == 400
        since = await staff.get("/api/oversight/turns/?since=2001-01-01T00:00:00Z&outcome=allowed")
        assert since.status_code == 200, since.content
        assert (await staff.get("/api/oversight/policies/x/overrides/")).status_code == 405
        assert (await staff.get("/api/oversight/policies/")).status_code == 200
        # Staff may read, but revealing who asked is a superuser's.
        reveal = await post(staff, "turns/x/reveal/", {"reason": "because of reasons"})
        assert reveal.status_code == 403

    async def test_members_read_nothing(self):
        member = await browser()
        assert (await member.get("/api/oversight/turns/")).status_code == 403
        assert (await member.get("/api/oversight/policies/")).status_code == 403

    async def test_the_export_is_ndjson_and_recorded(self):
        with override_settings(AI_SDK_OVERSIGHT={"CAPTURE": "full"}):
            invalidate()
            await a_turn()
        invalidate()
        root = await browser(staff=True, superuser=True)
        response = await root.get("/api/oversight/export/?purpose=review&days=7")
        assert response.status_code == 200
        assert response["Content-Type"] == "application/x-ndjson"
        assert 'filename="oversight-review.jsonl"' in response["Content-Disposition"]
        body = b"".join([chunk async for chunk in response.streaming_content]).decode()
        [row] = [json.loads(line) for line in body.splitlines()]
        assert row["preview"] == "What about my salary?"
        assert await AccessRecord.objects.filter(action="export").acount() == 1

    async def test_staff_may_not_export(self):
        staff = await browser(staff=True)
        assert (await staff.get("/api/oversight/export/")).status_code == 403


@pytest.mark.django_db(transaction=True)
class TestConsoleWrites:
    async def test_only_managers_build_policies(self):
        staff = await browser(staff=True)
        assert (await post(staff, "policies/", {"name": "fruit"})).status_code == 403
        member = await browser()
        assert (await post(member, "policies/", {"name": "fruit"})).status_code == 403

    async def test_a_policy_from_create_to_delete(self):
        root = await browser(staff=True, superuser=True)
        created = await post(root, "policies/", {"name": "fruit", "label": "Fruit"})
        assert created.status_code == 201, created.content
        assert created.json()["name"] == "fruit"
        assert (await post(root, "policies/", {"name": "Not OK"})).status_code == 400

        lens = {"target_kind": "lens", "target": "pineapple", "field": "definition",
                "value": BLOCK_WORD}
        assert (await post(root, "policies/fruit/overrides/", lens)).status_code == 200
        rule = {"target_kind": "rule", "target": "no_pineapple", "field": "definition",
                "value": {"when": {"lens": "pineapple", "op": "is_true"},
                          "then": [{"kind": "block", "message": "No fruit."}],
                          "mode": "enforce"}}
        policy = await post(root, "policies/fruit/overrides/", rule)
        assert [r["name"] for r in policy.json()["rules"]] == ["no_pineapple"]

        simulated = await post(root, "policies/fruit/simulate/",
                               {"input": "I like pineapple", "output": "Me too"})
        assert simulated.status_code == 200, simulated.content
        assert simulated.json()["outcome"] == "blocked"
        assert (await post(root, "policies/nope/simulate/", {"input": "x"})).status_code == 404

        tried = await post(root, "policies/fruit/try-lens/",
                           {"lens": "pineapple", "input": "pineapple pizza"})
        assert tried.status_code == 200, tried.content
        assert tried.json()["value"] is True
        assert await Turn.objects.acount() == 0  # dry runs store nothing

        assert (await root.delete("/api/oversight/policies/fruit/?note=done")).status_code == 204
        assert (await root.delete("/api/oversight/policies/fruit/")).status_code == 404
        # A policy in code is removed in code.
        assert (await root.delete("/api/oversight/policies/test-guard/")).status_code == 400
        assert await PolicyOverride.objects.filter(note="done").aexists()

    async def test_assigning_an_agent(self):
        agent = make_agent([], None)
        key = agent_key(agent) or "guarded-agent"
        rows = [{"id": key, "name": "guarded", "kind": "code", "description": "",
                 "code_policy": "test-guard", "code_reason": "agent"}]
        root = await browser(staff=True, superuser=True)
        staff = await browser(staff=True)
        with patch("django_ai_sdk.oversight.console._agents", return_value=rows):
            listed = await staff.get("/api/oversight/agents/")
            assert [r["effective"] for r in listed.json()] == ["test-guard"]
            assert (await post(staff, f"agents/{key}/", {"policy": ""})).status_code == 403

            off = await post(root, f"agents/{key}/", {"policy": ""})
            assert off.status_code == 200, off.content
            assert (off.json()["effective"], off.json()["source"]) == ("", "console")
            assert (await post(root, f"agents/{key}/", {"policy": "nope"})).status_code == 400
            assert (await post(root, "agents/missing/", {"policy": ""})).status_code == 404
            back = await post(root, f"agents/{key}/", {"reset": True})
            assert back.json()["effective"] == "test-guard"

    async def test_defining_an_answerer(self):
        root = await browser(staff=True, superuser=True)
        config = {"config": {"class": "LLMJudge", "model": "fruit-model"}}
        catalog = await post(root, "answerers/fruit_finder/", config)
        assert catalog.status_code == 200, catalog.content
        row = next(a for a in catalog.json()["aliases"] if a["alias"] == "fruit_finder")
        assert (row["source"], row["class"]) == ("console", "LLMJudge")
        broken = await post(root, "answerers/broken/", {"config": {"class": "Nope"}})
        assert broken.status_code == 400
        staff = await browser(staff=True)
        assert (await post(staff, "answerers/fruit_finder/", config)).status_code == 403
        assert (await post(staff, "answerers/fruit_finder/", {"reset": True})).status_code == 403


# A body each write accepts, so a refusal is the permission's and not the input's.
BODIES = {
    "reveal_oversight_actor": {"reason": "because of reasons"},
    "add_oversight_verdict": {"lens": "topic", "value": "hr"},
    "set_oversight_override": {"target_kind": "rule", "field": "mode"},
    "create_oversight_policy": {"name": "fruit"},
    "simulate_oversight_policy": {"input": "hi"},
    "try_oversight_answerer": {"config": "judge"},
    "forget_oversight_thread": {"reason": "because of reasons"},
}


@pytest.mark.django_db(transaction=True)
class TestShape:
    async def test_every_route_checks_a_permission(self):
        from django_ai_sdk.contrib.ninja.oversight import routes

        member = await browser()
        for method, route, func, _ in routes._ops:
            url = "/api" + route.replace("{", "").replace("}", "")
            body = json.dumps(BODIES.get(func.__name__, {}))
            response = await member.generic(method, url, body, content_type="application/json")
            assert response.status_code == 403, (func.__name__, response.content)

    async def test_policy_responses_are_typed(self):
        staff = await browser(staff=True)
        policy = (await staff.get("/api/oversight/policies/test-guard/")).json()
        assert {"lenses", "rules", "sources", "rejected", "parent_labels", "aliases",
                "agents", "stream_mode", "retention"} <= set(policy)
        assert policy["lenses"][0]["sources"]
        listed = (await staff.get("/api/oversight/policies/")).json()
        assert "agent_refs" in listed[0]

    async def test_a_closed_review_is_a_conflict(self):
        await a_turn()
        staff = await browser(staff=True)
        [item] = (await staff.get("/api/oversight/reviews/?queue=hr")).json()
        assert (await post(staff, f"reviews/{item['id']}/", {"agree": True})).status_code == 200
        again = await post(staff, f"reviews/{item['id']}/", {"agree": False})
        assert (again.status_code, again.json()["code"]) == (409, "conflict")
