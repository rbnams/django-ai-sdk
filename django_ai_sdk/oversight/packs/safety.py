"""The safety pack: harm in what is asked and what is said, and people at risk.

Jev answers all three lenses. Where it is unsure, OpenAI's moderation endpoint
(the `moderation` alias) is asked about the question, and the `judge` model about
the answer, which review has time for. A message suggesting someone is at risk is
never blocked: it goes to a person, first in line.
"""

from __future__ import annotations

from django_ai_sdk.oversight.conditions import L
from django_ai_sdk.oversight.lenses import Flag, Score
from django_ai_sdk.oversight.policy import Policy
from django_ai_sdk.oversight.registry import register
from django_ai_sdk.oversight.rules import Alert, Notify, Queue, Rule
from django_ai_sdk.oversight.types import Stage, Subject


class SelfHarm(Flag):
    name = "self_harm"
    label = "Person at risk"
    question = "Does the message suggest the person may harm themselves?"
    yes = "Expresses intent, plan or wish to harm or kill themselves, directly or indirectly."
    no = "Mentions death or harm figuratively, about others, or as a topic."
    subject = Subject.INPUT
    stages = (Stage.GATE,)
    answerers = ["jev?", "moderation?"]


class HarmfulInput(Score):
    name = "harm_input"
    label = "Harmful request"
    question = "How harmful is what the message asks for (violence, hate, sexual, illegal)?"
    low, high = "harmless", "clearly harmful"
    subject = Subject.INPUT
    stages = (Stage.GATE,)
    answerers = ["jev?", "moderation?"]


class HarmfulOutput(Score):
    name = "harm_output"
    label = "Harmful answer"
    question = "How harmful is the answer's content (violence, hate, sexual, illegal)?"
    low, high = "harmless", "clearly harmful"
    subject = Subject.OUTPUT
    stages = (Stage.REVIEW,)
    answerers = ["jev?", "judge?"]


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


__all__ = ["HarmfulInput", "HarmfulOutput", "SafetyPack", "SelfHarm"]
