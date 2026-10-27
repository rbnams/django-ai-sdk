"""Conditions, rules and policies: one engine for code and console, layered with sources."""

from __future__ import annotations

import pytest
from django.core.exceptions import ImproperlyConfigured
from django.test import override_settings
from django_ai_sdk.oversight import (
    PII,
    Alert,
    Block,
    Choice,
    Flag,
    Keywords,
    L,
    Mode,
    Notify,
    Policy,
    Redact,
    Rule,
    pick,
    register,
)
from django_ai_sdk.oversight.conditions import from_dict, same_shape
from django_ai_sdk.oversight.policy import apply_overlay, build, invalidate
from django_ai_sdk.oversight.packs.security import SecurityPack
from django_ai_sdk.oversight.registry import get_invalid_policies, validate
from django_ai_sdk.oversight.stages import evaluate_rules
from django_ai_sdk.oversight.types import Answer, Capture, Stage


def ans(value, confidence=1.0, error=""):
    return Answer(value=value, confidence=confidence, error=error)


class TestConditions:
    def test_operators(self):
        answers = {"inj": ans(True, 0.95), "tox": ans(0.9), "topic": ans("hr"), "pii": ans(["bsn"])}
        assert (L("inj").is_true & L("inj").confident(0.9)).evaluate(answers)
        assert (L("tox") > 0.8).evaluate(answers)
        assert not (L("tox") < 0.8).evaluate(answers)
        assert (L("topic") == "hr").evaluate(answers)
        assert L("topic").in_({"hr", "legal"}).evaluate(answers)
        assert L("pii").contains("bsn").evaluate(answers)
        assert L("pii").any.evaluate(answers)
        assert (~(L("topic") == "billing")).evaluate(answers)

    def test_a_missing_answer_matches_nothing_but_missing(self):
        answers = {"broken": ans(None, error="timeout")}
        assert not L("broken").is_true.evaluate(answers)
        assert not (L("absent") > 0).evaluate(answers)
        assert L("broken").missing.evaluate(answers)
        assert L("absent").missing.evaluate(answers)

    def test_round_trips_through_json(self):
        condition = (L("a").is_true & (L("b") >= 0.5)) | ~L("c").in_(["x"])
        again = from_dict(condition.to_dict())
        assert again.to_dict() == condition.to_dict()
        assert again.lenses() == {"a", "b", "c"}

    def test_same_shape_allows_thresholds_not_new_tests(self):
        a = (L("tox") > 0.8).to_dict()
        assert same_shape(a, (L("tox") > 0.5).to_dict())
        assert not same_shape(a, (L("tox") < 0.5).to_dict())
        assert not same_shape(a, (L("other") > 0.8).to_dict())
        assert not same_shape(a, ((L("tox") > 0.8) & L("x").is_true).to_dict())

    def test_rule_round_trip(self):
        rule = Rule("r", when=L("a").is_true, then=[Block("no"), Alert("high")], mode="enforce")
        again = Rule.from_dict(rule.to_dict())
        assert again.to_dict() == rule.to_dict()
        assert [a.kind for a in again.ordered_actions()] == ["block", "alert"]


@register
class ParentPolicy(Policy):
    name = "test-parent"
    lenses = [Flag(name="mentions_x", question="X?", answerers=[Keywords({"x": ["x"]})])]
    rules = [Rule("block_x", when=L("mentions_x").is_true, then=Block("no x"))]


@register
class ChildPolicy(Policy):
    name = "test-child"
    extends = ["test-parent"]  # noqa: RUF012
    capture = Capture.REDACTED
    lenses = [Choice(name="topic", question="?", labels=["a", "b"], answerers=[Keywords({"a": ["a"]})])]
    rules = [Rule("note_b", when=L("topic") == "b", then=Alert(), mode=Mode.ENFORCE)]



@register
class PickedPolicy(Policy):
    name = "test-picked"
    lenses = pick(SecurityPack.lenses, "prompt_injection", "secrets_input")
    rules = [
        *pick(SecurityPack.rules, "block_injection"),
        *pick(SecurityPack.rules, "redact_secrets_in_question", mode=Mode.ENFORCE),
    ]


class TestPolicies:
    def setup_method(self):
        invalidate()

    def test_extends_merges_with_sources(self):
        resolved = build("test-child")
        assert set(resolved.lenses) == {"mentions_x", "topic"}
        assert set(resolved.rules) == {"block_x", "note_b"}
        assert resolved.sources["rule:block_x.mode"] == "pack:test-parent"
        assert resolved.sources["rule:note_b.mode"] == "code"
        assert resolved.capture is Capture.REDACTED

    def test_inherited_rules_start_in_shadow(self):
        assert build("test-child").rules["block_x"].mode is Mode.SHADOW

    @override_settings(AI_SDK_OVERSIGHT={"POLICIES": {"test-child": {
        "rules": {"block_x": {"mode": "enforce"}}, "lenses": {"topic": {"sample": 0.5}}}}})
    def test_settings_tune_a_policy(self):
        resolved = build("test-child")
        assert resolved.rules["block_x"].mode is Mode.ENFORCE
        assert resolved.lenses["topic"].sample == 0.5
        assert resolved.sources["rule:block_x.mode"] == "settings"

    @override_settings(AI_SDK_OVERSIGHT={"CAPTURE": "metadata"})
    def test_the_deployment_capture_is_a_ceiling(self):
        assert build("test-child").capture is Capture.METADATA

    def test_the_overlay_tunes_but_cannot_rewrite(self):
        resolved = build("test-child")
        apply_overlay(
            resolved,
            [
                {"target_kind": "rule", "target": "block_x", "field": "mode", "value": "enforce"},
                {"target_kind": "rule", "target": "block_x", "field": "when",
                 "value": {"lens": "topic", "op": "eq", "value": "a"}},
                {"target_kind": "lens", "target": "nope", "field": "sample", "value": 1},
                {"target_kind": "lens", "target": "topic", "field": "sample", "value": 3},
            ],
            source="db",
        )
        assert resolved.rules["block_x"].mode is Mode.ENFORCE
        assert resolved.sources["rule:block_x.mode"] == "db"
        assert resolved.rules["block_x"].when.lenses() == {"mentions_x"}
        assert len(resolved.rejected) == 3

    def test_the_overlay_can_add_lenses_and_rules(self):
        resolved = build("test-child")
        apply_overlay(
            resolved,
            [
                {"target_kind": "lens", "target": "is_long", "field": "definition",
                 "value": {"kind": "flag", "question": "Long?", "answerers": ["human"]}},
                {"target_kind": "rule", "target": "queue_long", "field": "definition",
                 "value": {"when": {"lens": "is_long", "op": "is_true"},
                           "then": [{"kind": "queue", "queue": "long"}]}},
            ],
            source="db",
        )
        assert "is_long" in resolved.lenses
        assert resolved.rules["queue_long"].then[0].kind == "queue"
        assert not resolved.rejected

    def test_incomplete_policies_are_not_registered(self):
        class NoName(Policy):
            pass

        with pytest.raises(ImproperlyConfigured):
            validate(NoName)

        @register
        class Bad(Policy):
            name = "test-bad"
            lenses = [Choice(name="c", question="?", answerers=[PII()])]

        assert "test-bad" in get_invalid_policies()

    def test_a_policy_picks_single_lenses_and_rules_from_a_pack(self):
        resolved = build("test-picked")
        assert list(resolved.lenses) == ["prompt_injection", "secrets_input"]
        assert list(resolved.rules) == ["block_injection", "redact_secrets_in_question"]
        assert resolved.sources["lens:prompt_injection.origin"] == "code"
        assert resolved.rules["redact_secrets_in_question"].mode is Mode.ENFORCE
        assert not get_invalid_policies().get("test-picked")

    def test_picking_leaves_the_pack_alone(self):
        pack = {rule.name: rule for rule in SecurityPack.rules}
        assert pack["redact_secrets_in_question"].mode is Mode.SHADOW
        assert pick(SecurityPack.rules, "block_injection")[0] is not pack["block_injection"]

    def test_picking_an_unknown_name_says_what_there_is(self):
        with pytest.raises(ImproperlyConfigured, match="prompt_injection"):
            pick(SecurityPack.lenses, "prompt_injektion")

    def test_mode_only_applies_to_rules(self):
        with pytest.raises(TypeError):
            pick(SecurityPack.lenses, "prompt_injection", mode=Mode.ENFORCE)

    def test_the_packs_are_registered_and_resolve(self):
        resolved = build("default")
        assert {"prompt_injection", "pii_input", "groundedness", "intent", "self_harm"} <= set(
            resolved.lenses
        )
        assert all(rule.mode is Mode.SHADOW for rule in resolved.rules.values())


class TestRuleEvaluation:
    def test_a_rule_waits_for_lenses_still_to_come(self):
        resolved = build("default")
        early = evaluate_rules(resolved, {"pii_input": ans([])}, Stage.GATE, done=set())
        assert "review_ungrounded" not in {m.rule.name for m in early}
        later = evaluate_rules(
            resolved, {"groundedness": ans(0.1, 0.9)}, Stage.REFLECT, done=set()
        )
        assert "review_ungrounded" in {m.rule.name for m in later}

    def test_shadow_matches_record_but_do_not_act(self):
        resolved = build("test-parent")
        [match] = evaluate_rules(resolved, {"mentions_x": ans(True)}, Stage.GATE, done=set())
        assert not match.acted
        assert match.outcome(shadow=True).value == "blocked"

    def test_enforced_actions_only_where_they_still_can(self):
        resolved = build("test-parent")
        resolved.rules["block_x"].mode = Mode.ENFORCE
        resolved.rules["block_x"].then = [Block(), Redact()]
        [gate] = evaluate_rules(resolved, {"mentions_x": ans(True)}, Stage.GATE, done=set())
        assert [a.kind for a in gate.acted] == ["block", "redact"]
        [late] = evaluate_rules(resolved, {"mentions_x": ans(True)}, Stage.REFLECT, done=set())
        assert [a.kind for a in late.acted] == ["redact"]

    def test_fail_closed_treats_an_error_as_a_match(self):
        resolved = build("test-parent")
        broken = {"mentions_x": ans(None, error="timeout")}

        def closed():
            return evaluate_rules(
                resolved, broken, Stage.GATE, done=set(), errored={"mentions_x"}, fail_closed=True
            )

        # In shadow, a slow answerer would only fill the ledger with false matches.
        assert not closed()
        resolved.rules["block_x"].mode = Mode.ENFORCE
        assert not evaluate_rules(resolved, broken, Stage.GATE, done=set())
        assert [m.rule.name for m in closed()] == ["block_x"]

    def test_fail_closed_never_raises_an_alert(self):
        resolved = build("test-parent")
        resolved.rules["block_x"].mode = Mode.ENFORCE
        resolved.rules["block_x"].then = [Alert("high")]
        broken = {"mentions_x": ans(None, error="timeout")}
        assert not evaluate_rules(
            resolved, broken, Stage.GATE, done=set(), errored={"mentions_x"}, fail_closed=True
        )

    def test_retention_takes_content_and_metadata_only(self):
        resolved = build("test-parent")
        assert resolved.retention == {"content": 90, "metadata": 730}
        apply_overlay(
            resolved,
            [
                {"target_kind": "policy", "target": "", "field": "retention",
                 "value": {"content": 30}},
                {"target_kind": "policy", "target": "", "field": "retention",
                 "value": {"security": 90}},
            ],
            source="db",
        )
        assert resolved.retention == {"content": 30, "metadata": 730}
        assert "unknown retention" in resolved.rejected[0]["reason"]


class TestSeverity:
    def test_error_means_high_and_unknowns_are_refused(self):
        assert Alert("error").severity == "high"
        assert Notify(severity="critical").severity == "critical"
        with pytest.raises(ValueError, match="Unknown severity"):
            Alert("loud")

    def test_the_most_severe_wins(self):
        from django_ai_sdk.oversight.types import most_severe

        assert most_severe("info", "high", "warning") == "high"
        assert most_severe() == ""


@pytest.mark.django_db(transaction=True)
class TestInheritedTuning:
    def setup_method(self):
        invalidate()

    async def _enforce_on_parent(self):
        from django_ai_sdk.oversight.models import PolicyOverride

        await PolicyOverride.objects.acreate(
            policy="test-parent", target_kind="rule", target="block_x", field="mode",
            value="enforce",
        )

    async def test_tuning_a_parent_reaches_its_children(self):
        from django_ai_sdk.oversight.policy import aget_policy

        await self._enforce_on_parent()
        child = await aget_policy("test-child")
        assert child.rules["block_x"].mode is Mode.ENFORCE
        assert child.sources["rule:block_x.mode"] == "db:test-parent"
        assert child.sources["rule:block_x.when"] == "pack:test-parent"

    async def test_invalidating_a_parent_forgets_its_children(self):
        from django_ai_sdk.oversight.policy import aget_policy

        assert (await aget_policy("test-child")).rules["block_x"].mode is Mode.SHADOW
        await self._enforce_on_parent()
        invalidate("test-parent")
        assert (await aget_policy("test-child")).rules["block_x"].mode is Mode.ENFORCE

    async def test_an_uncached_read_leaves_the_cache_alone(self):
        from django_ai_sdk.oversight import policy as policies

        cached = await policies.aget_policy("test-child")
        await self._enforce_on_parent()
        fresh = await policies.aget_policy("test-child", uncached=True)
        assert fresh.rules["block_x"].mode is Mode.ENFORCE
        assert await policies.aget_policy("test-child") is cached
