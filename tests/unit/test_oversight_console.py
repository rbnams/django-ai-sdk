"""The console as an editor: policies, lenses and rules, agents, answerers, and dry runs."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from asgiref.sync import sync_to_async
from django_ai_sdk.errors import UserError
from django_ai_sdk.oversight.answerers.base import alias_defined, resolve
from django_ai_sdk.oversight.models import PolicyOverride, Turn
from django_ai_sdk.oversight.policy import GLOBAL, invalidate
from django_ai_sdk.oversight.schemas import (
    AgentPolicyIn,
    AnswererIn,
    OverrideIn,
    PolicyIn,
    SimulateIn,
    TryAnswererIn,
    TryLensIn,
)
from django_ai_sdk.oversight.services import OversightService, _write_override
from django_ai_sdk.oversight.session import TurnSession, agent_key
from django_ai_sdk.permissions import PermissionDenied
from tests.factories.db import UserFactory
from tests.unit.test_oversight_turns import make_agent  # registers the test-guard policy


@pytest.fixture(autouse=True)
def _fresh():
    invalidate()
    yield
    invalidate()


async def root():
    return await UserFactory.acreate(staff=True, superuser=True)


BLOCK_WORD = {
    "kind": "flag",
    "label": "Says the word",
    "question": "Does the message say pineapple?",
    "subject": "input",
    "stages": ["gate"],
    "answerers": [{"class": "Patterns", "patterns": [r"\bpineapple\b"]}],
}


@pytest.mark.django_db(transaction=True)
class TestPolicies:
    async def test_staff_may_look_but_not_build(self):
        staff = await UserFactory.acreate(staff=True)
        await OversightService.list_policies(user=staff)
        with pytest.raises(PermissionDenied):
            await OversightService.create_policy(PolicyIn(name="mine"), user=staff)

    async def test_a_policy_from_an_example(self):
        user = await root()
        policy = await OversightService.create_policy(
            PolicyIn(name="helpdesk", example="support"), user=user
        )
        assert policy["origin"] == "db"
        assert policy["extends"] == ["security", "quality"]
        assert policy["stream_mode"] == "hold"
        names = {lens["name"]: lens for lens in policy["lenses"]}
        assert names["support_intent"]["origin"] == "db"
        assert names["prompt_injection"]["origin"] == "pack:security"
        assert {r["name"] for r in policy["rules"]} >= {"refunds_to_finance", "block_injection"}
        listed = {p.name: p for p in await OversightService.list_policies(user=user)}
        assert listed["helpdesk"].origin == "db"
        assert listed["helpdesk"].shadow >= 3

    async def test_names_are_checked(self):
        user = await root()
        with pytest.raises(UserError):
            await OversightService.create_policy(PolicyIn(name="Not OK"), user=user)
        with pytest.raises(UserError):
            await OversightService.create_policy(PolicyIn(name="security"), user=user)
        with pytest.raises(UserError):
            await OversightService.create_policy(
                PolicyIn(name="orphan", extends=["nowhere"]), user=user
            )

    async def test_lenses_and_rules_defined_in_the_console(self):
        user = await root()
        await OversightService.create_policy(PolicyIn(name="fruit"), user=user)
        await OversightService.set_override(
            "fruit",
            OverrideIn(target_kind="lens", target="pineapple", field="definition",
                       value=BLOCK_WORD),
            user=user,
        )
        policy = await OversightService.set_override(
            "fruit",
            OverrideIn(
                target_kind="rule",
                target="no_pineapple",
                field="definition",
                value={"when": {"lens": "pineapple", "op": "is_true"},
                       "then": [{"kind": "block", "message": "No fruit."}], "mode": "enforce"},
            ),
            user=user,
        )
        assert [r["name"] for r in policy["rules"]] == ["no_pineapple"]
        result = await OversightService.simulate(
            "fruit", SimulateIn(input="I like pineapple", output="Me too"), user=user
        )
        gate = result["stages"][0]
        assert gate["blocked"] and gate["message"] == "No fruit."
        assert gate["observations"][0]["value"] is True
        assert result["ended_at"] == "gate" and result["outcome"] == "blocked"
        assert await Turn.objects.acount() == 0

        # A rule reading a lens that does not exist is refused.
        with pytest.raises(UserError):
            await OversightService.set_override(
                "fruit",
                OverrideIn(target_kind="rule", target="ghost", field="definition",
                           value={"when": {"lens": "nope", "op": "is_true"},
                                  "then": [{"kind": "alert"}]}),
                user=user,
            )

    async def test_redefining_a_lens_drops_its_old_tuning(self):
        user = await root()
        await OversightService.create_policy(PolicyIn(name="fruit"), user=user)
        define = OverrideIn(target_kind="lens", target="pineapple", field="definition",
                            value=BLOCK_WORD)
        await OversightService.set_override("fruit", define, user=user)
        await OversightService.set_override(
            "fruit", OverrideIn(target_kind="lens", target="pineapple", field="enabled",
                                value=False), user=user)
        await OversightService.set_override("fruit", define, user=user)
        policy = await OversightService.get_policy("fruit", user=user)
        assert policy["lenses"][0]["enabled"] is True

    async def test_a_simulation_shows_shadow_matches_and_every_stage(self):
        user = await root()
        await OversightService.create_policy(PolicyIn(name="helpdesk", example="support"),
                                             user=user)
        result = await OversightService.simulate(
            "helpdesk",
            SimulateIn(input="I want my money back for this order", output="Sure, refunded."),
            user=user,
        )
        stages = [s["stage"] for s in result["stages"]]
        assert stages == ["gate", "review", "reflect"]
        intent = next(o for o in result["stages"][0]["observations"]
                      if o["lens"] == "support_intent")
        assert intent["value"] == "refund"
        matched = [m["rule"] for s in result["stages"] for m in s["matched"]]
        assert "refunds_to_finance" in matched
        assert result["outcome"] == "allowed"
        assert result["shadow_outcome"] == "flagged"

    async def test_trying_a_draft_lens(self):
        user = await root()
        result = await OversightService.try_lens(
            "test-guard", TryLensIn(definition=BLOCK_WORD, input="pineapple pizza"), user=user
        )
        assert result["value"] is True and result["confidence"] == 1.0

    async def test_removing_a_policy_hands_its_agents_back(self):
        user = await root()
        await OversightService.create_policy(PolicyIn(name="fruit"), user=user)
        await sync_to_async(_write_override)(
            GLOBAL, OverrideIn(target_kind="agent", target="a-1", field="policy", value="fruit"),
            user,
        )
        assert await OversightService.delete_policy("fruit", user=user)
        assert "fruit" not in {p.name for p in await OversightService.list_policies(user=user)}
        assert not await PolicyOverride.objects.filter(active=True, target="a-1").aexists()
        with pytest.raises(UserError):
            await OversightService.delete_policy("security", user=user)

    async def test_the_trail_keeps_every_change(self):
        user = await root()
        await OversightService.create_policy(PolicyIn(name="helpdesk", example="support",
                                                      note="For the shop"), user=user)
        trail = await OversightService.changes(user=user, policy="helpdesk")
        assert len(trail) == 1 + 1 + 3 + 3  # definition, stream mode, lenses, rules
        assert {c.note for c in trail} == {"For the shop"}


@pytest.mark.django_db(transaction=True)
class TestAgents:
    async def test_the_console_choice_wins_over_code(self):
        user = await root()
        agent = make_agent([], None)
        key = agent_key(agent) or "guarded-agent"
        type(agent)._agent_id = key
        rows = [{"id": key, "name": "guarded", "kind": "code", "description": "",
                 "code_policy": "test-guard", "code_reason": "agent"}]
        with patch("django_ai_sdk.oversight.services._agents", return_value=rows):
            [row] = await OversightService.list_agents(user=user)
            assert (row.effective, row.source) == ("test-guard", "agent")

            row = await OversightService.set_agent_policy(key, AgentPolicyIn(policy=""),
                                                          user=user)
            assert (row.effective, row.source, row.assigned) == ("", "console", "")
            session = await TurnSession.start(agent, channel="api", input="hi")
            assert not session.active

            await OversightService.create_policy(PolicyIn(name="fruit"), user=user)
            row = await OversightService.set_agent_policy(key, AgentPolicyIn(policy="fruit"),
                                                          user=user)
            assert row.effective == "fruit"
            session = await TurnSession.start(agent, channel="api", input="hi")
            assert session.policy.name == "fruit"

            row = await OversightService.set_agent_policy(key, AgentPolicyIn(reset=True),
                                                          user=user)
            assert (row.effective, row.source, row.assigned) == ("test-guard", "agent", None)
            with pytest.raises(UserError):
                await OversightService.set_agent_policy(key, AgentPolicyIn(policy="nope"),
                                                        user=user)
            assert await OversightService.set_agent_policy("missing", AgentPolicyIn(),
                                                           user=user) is None


@pytest.mark.django_db(transaction=True)
class TestAnswerers:
    async def test_an_alias_defined_in_the_console(self):
        user = await root()
        catalog = await OversightService.set_answerer(
            "fruit_finder",
            AnswererIn(config={"class": "Patterns", "patterns": [r"\bpineapple\b"]}),
            user=user,
        )
        row = next(a for a in catalog["aliases"] if a["alias"] == "fruit_finder")
        assert row["source"] == "console" and row["class"] == "Patterns"
        assert alias_defined("fruit_finder")
        assert type(resolve("fruit_finder")).__name__ == "Patterns"

        await OversightService.set_answerer("fruit_finder", AnswererIn(reset=True), user=user)
        assert not alias_defined("fruit_finder")

    async def test_a_broken_alias_is_refused(self):
        user = await root()
        with pytest.raises(UserError):
            await OversightService.set_answerer(
                "broken", AnswererIn(config={"class": "Nope"}), user=user
            )
        with pytest.raises(UserError):
            await OversightService.set_answerer("human", AnswererIn(config="x"), user=user)
        assert not alias_defined("broken")

    async def test_trying_an_answerer(self):
        user = await root()
        result = await OversightService.try_answerer(
            TryAnswererIn(config={"class": "Patterns", "patterns": ["ignore all"]}), user=user
        )
        assert result["value"] is True

    async def test_the_catalog_offers_kinds_models_and_examples(self):
        user = await root()
        catalog = await OversightService.catalog(user=user)
        assert {k["class"] for k in catalog["alias_kinds"]} >= {"LLMJudge", "TypeSafe"}
        assert {k["class"] for k in catalog["inline_kinds"]} >= {"Patterns", "PII", "Keywords"}
        assert len(catalog["examples"]["policies"]) >= 5
        for example in catalog["examples"]["lenses"]:
            for spec in example["lens"]["answerers"]:
                if isinstance(spec, dict):
                    resolve(spec)


class TestJev:
    async def test_a_score_is_as_sure_as_its_likeliest_level(self):
        from django_ai_sdk.oversight.answerers import TypeSafe
        from django_ai_sdk.oversight.lenses import Score
        from django_ai_sdk.oversight.types import TurnView

        summary = {
            "model": "jev-1",
            "judgments": [{"id": "help", "type": "score", "value": 1.22, "confidence": 0,
                           "probabilities": {"0": 0.26, "1": 0.26, "2": 0.48}}],
        }
        lens = Score(name="help", question="How helpful?", low="useless", high="great",
                     answerers=["x"])
        with patch("django_ai_sdk.typesafe.client.evaluate", AsyncMock(return_value=summary)):
            answer = await TypeSafe().answer(lens, TurnView(input="q", output="a"))
        assert answer.confidence == pytest.approx(0.48)
        assert answer.rationale.startswith("great 0.48")


@pytest.mark.django_db(transaction=True)
class TestCodeAnswerers:
    async def test_a_pack_lens_keeps_its_regex_when_jev_is_added(self):
        user = await root()
        policy = await OversightService.get_policy("security", user=user)
        lens = next(item for item in policy["lenses"] if item["name"] == "prompt_injection")
        assert lens["answerer_specs"][0] == "code:0"
        await OversightService.create_policy(PolicyIn(name="guarded", extends=["security"]),
                                             user=user)
        await OversightService.set_answerer(
            "jev", AnswererIn(config={"class": "Patterns", "patterns": ["nothing"]}), user=user
        )
        policy = await OversightService.set_override(
            "guarded",
            OverrideIn(target_kind="lens", target="prompt_injection", field="answerers",
                       value=["code:0", "jev"]),
            user=user,
        )
        lens = next(item for item in policy["lenses"] if item["name"] == "prompt_injection")
        assert lens["answerer_specs"] == ["code:0", "jev"]
        assert [a["class"] for a in lens["answerers"]] == ["Patterns", "Patterns"]
        with pytest.raises(UserError):
            await OversightService.set_override(
                "guarded",
                OverrideIn(target_kind="lens", target="prompt_injection", field="answerers",
                           value=["code:9"]),
                user=user,
            )

    async def test_optional_answerers_that_are_not_set_up_are_kept(self):
        user = await root()
        await OversightService.create_policy(PolicyIn(name="guarded", extends=["security"]),
                                             user=user)
        policy = await OversightService.set_override(
            "guarded",
            OverrideIn(target_kind="lens", target="prompt_injection", field="answerers",
                       value=["code:0", "nowhere?"]),
            user=user,
        )
        lens = next(item for item in policy["lenses"] if item["name"] == "prompt_injection")
        assert lens["answerer_specs"] == ["code:0", "nowhere?"]
        with pytest.raises(UserError):
            await OversightService.set_override(
                "guarded",
                OverrideIn(target_kind="lens", target="prompt_injection", field="answerers",
                           value=["nowhere"]),
                user=user,
            )


@pytest.mark.django_db(transaction=True)
class TestToolCalls:
    async def test_a_refused_tool_call_says_why_in_the_rules_words(self):
        user = await root()
        await OversightService.create_policy(PolicyIn(name="ops", example="operator"), user=user)
        await OversightService.set_override(
            "ops",
            OverrideIn(target_kind="rule", target="risky_tools_need_a_person", field="mode",
                       value="enforce"),
            user=user,
        )
        result = await OversightService.simulate(
            "ops",
            SimulateIn(input="Remove Jan", tool_name="delete_customer",
                       tool_arguments={"name": "Jan"}),
            user=user,
        )
        watch = next(s for s in result["stages"] if s["stage"] == "watch")
        assert watch["blocked"]
        assert watch["message"] == "That action needs approval. I've asked a colleague."
