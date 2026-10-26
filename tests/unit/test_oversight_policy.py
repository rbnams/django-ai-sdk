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
    Policy,
    Redact,
    Rule,
    register,
)
from django_ai_sdk.oversight.conditions import from_dict, same_shape
from django_ai_sdk.oversight.policy import apply_overlay, build, invalidate
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
        assert not evaluate_rules(resolved, broken, Stage.GATE, done=set())
        matched = evaluate_rules(
            resolved, broken, Stage.GATE, done=set(), errored={"mentions_x"}, fail_closed=True
        )
        assert [m.rule.name for m in matched] == ["block_x"]
