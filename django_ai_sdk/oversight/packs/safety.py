"""The safety pack: harm in what is asked and what is said, and people at risk.

Answered by a moderation model when the `moderation` alias is defined, and by
patterns otherwise. A message suggesting someone is at risk is never blocked:
it goes to a person, first in line.
"""

from __future__ import annotations

from django_ai_sdk.oversight.answerers import Patterns
from django_ai_sdk.oversight.conditions import L
from django_ai_sdk.oversight.lenses import Flag, Score
from django_ai_sdk.oversight.policy import Policy
from django_ai_sdk.oversight.registry import register
from django_ai_sdk.oversight.rules import Alert, Notify, Queue, Rule
from django_ai_sdk.oversight.types import Purpose, Stage, Subject

SELF_HARM_PATTERNS = [
    r"\b(kill|hurt|harm) myself\b",
    r"\b(suicide|suicidal|end my life|want to die)\b",
    r"\b(zelfmoord|mezelf (iets )?aandoen|dood willen|wil dood)\b",
]


class SelfHarm(Flag):
    name = "self_harm"
    label = "Person at risk"
    question = "Does the message suggest the person may harm themselves?"
    subject = Subject.INPUT
    stages = (Stage.GATE,)
    answerers = [Patterns(SELF_HARM_PATTERNS, miss_confidence=0.7), "moderation?"]
    purposes = (Purpose.SECURITY,)


class HarmfulInput(Score):
    name = "harm_input"
    label = "Harmful request"
    question = "How harmful is what the message asks for (violence, hate, sexual, illegal)?"
    low, high = "harmless", "clearly harmful"
    subject = Subject.INPUT
    stages = (Stage.GATE,)
    answerers = ["moderation?", "guardian?"]
    purposes = (Purpose.SECURITY,)


class HarmfulOutput(Score):
    name = "harm_output"
    label = "Harmful answer"
    question = "How harmful is the answer's content (violence, hate, sexual, illegal)?"
    low, high = "harmless", "clearly harmful"
    subject = Subject.OUTPUT
    stages = (Stage.REVIEW,)
    answerers = ["moderation?", "judge?"]
    purposes = (Purpose.SECURITY, Purpose.QUALITY)


@register
class SafetyPack(Policy):
    name = "safety"
    label = "Safety"
    description = "Harmful content in and out, and people who may be at risk."
    origin = "pack:safety"
    lenses = [SelfHarm, HarmfulInput, HarmfulOutput]
    rules = [
        Rule(
            "person_at_risk",
            when=L("self_harm").is_true,
            then=[Queue("safety", priority=100), Notify(severity="critical"), Alert("critical")],
            description="Never blocked: someone looks, now.",
        ),
        Rule(
            "harmful_request",
            when=L("harm_input") >= 0.8,
            then=[Queue("safety", priority=10), Alert("high")],
        ),
        Rule(
            "harmful_answer",
            when=L("harm_output") >= 0.7,
            then=[Queue("safety", priority=20), Alert("high")],
        ),
    ]


__all__ = ["SELF_HARM_PATTERNS", "SafetyPack"]
