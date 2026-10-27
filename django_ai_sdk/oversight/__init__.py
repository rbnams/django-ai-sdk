"""Oversight: audit every turn's question and answer, judge it any way you like.

Add "django_ai_sdk.oversight" to INSTALLED_APPS, give an agent a policy, and
every turn it runs, on any channel, lands on a hash-chained ledger with the
answers of the policy's lenses and the decisions of its rules::

    # myapp/oversight.py
    from django_ai_sdk.oversight import L, Choice, Policy, Queue, Rule, register

    @register
    class Support(Policy):
        name = "support"
        extends = ["security", "quality"]
        lenses = [Choice(name="topic", question="What is this about?",
                         labels=["billing", "hr", "other"], answerers=["jev?"])]
        rules = [Rule("hr_goes_to_people", when=L("topic") == "hr", then=Queue("hr"))]

    class SupportAgent(Agent):
        oversight = "support"

Lenses observe, rules act; rules start in test mode (shadow). See docs/oversight.md.

Only the declaration API is re-exported here. Specialised detectors and
classifiers live in `django_ai_sdk.oversight.answerers`; models, services and
tasks in their own modules, so importing this package needs no app registry.
"""

from __future__ import annotations

from django_ai_sdk.oversight.answerers import (
    PII,
    AgentJudge,
    Function,
    Human,
    Keywords,
    LLMJudge,
    Patterns,
    Secrets,
    TypeSafe,
)
from django_ai_sdk.oversight.conditions import L
from django_ai_sdk.oversight.lenses import Choice, Entities, Flag, Lens, Score, Text
from django_ai_sdk.oversight.policy import Policy, pick
from django_ai_sdk.oversight.registry import register
from django_ai_sdk.oversight.rules import (
    Alert,
    Allow,
    Block,
    Escalate,
    Notify,
    Queue,
    Redact,
    RestrictTools,
    Rule,
)
from django_ai_sdk.oversight.types import (
    Answer,
    Capture,
    Mode,
    Purpose,
    Stage,
    Subject,
    TurnView,
)

__all__ = [
    "PII",
    "AgentJudge",
    "Alert",
    "Allow",
    "Answer",
    "Block",
    "Capture",
    "Choice",
    "Entities",
    "Escalate",
    "Flag",
    "Function",
    "Human",
    "Keywords",
    "L",
    "LLMJudge",
    "Lens",
    "Mode",
    "Notify",
    "Patterns",
    "Policy",
    "Purpose",
    "Queue",
    "Redact",
    "RestrictTools",
    "Rule",
    "Score",
    "Secrets",
    "Stage",
    "Subject",
    "Text",
    "TurnView",
    "TypeSafe",
    "pick",
    "register",
]
