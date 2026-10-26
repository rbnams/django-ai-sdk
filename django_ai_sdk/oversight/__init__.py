"""Oversight: audit every turn's question and answer, judge it any way you like.

Add "django_ai_sdk.oversight" to INSTALLED_APPS, give an agent a policy, and
every turn it runs, on any channel, lands on a hash-chained ledger with the
answers of the policy's lenses and the decisions of its rules::

    # myapp/oversight.py
    from django_ai_sdk.oversight import L, Choice, Keywords, Policy, Queue, Rule, register

    @register
    class Support(Policy):
        name = "support"
        extends = ["security", "quality"]
        lenses = [Choice(name="topic", question="What is this about?",
                         labels=["billing", "hr", "other"],
                         answerers=[Keywords({"billing": ["invoice"]}), "fast?"])]
        rules = [Rule("hr_goes_to_people", when=L("topic") == "hr", then=Queue("hr"))]

    class SupportAgent(Agent):
        oversight = "support"

Lenses observe, rules act; rules start in shadow mode. See docs/oversight.md.

Only the declaration API is re-exported here; models, services and tasks are
imported from their modules, so importing this package needs no app registry.
"""

from __future__ import annotations

from django_ai_sdk.oversight.answerers import (
    PII,
    AgentJudge,
    Answerer,
    Cascade,
    EmptyRetrieval,
    Function,
    HFClassifier,
    Human,
    Keywords,
    Language,
    Length,
    LexicalSupport,
    LLMJudge,
    Moderation,
    Patterns,
    Presidio,
    Refusal,
    Sample,
    Secrets,
    ToolErrors,
    TypeSafe,
    Vote,
)
from django_ai_sdk.oversight.conditions import L
from django_ai_sdk.oversight.lenses import Choice, Entities, Flag, Lens, Score, Text
from django_ai_sdk.oversight.policy import Policy
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
    Outcome,
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
    "Answerer",
    "Block",
    "Capture",
    "Cascade",
    "Choice",
    "EmptyRetrieval",
    "Entities",
    "Escalate",
    "Flag",
    "Function",
    "HFClassifier",
    "Human",
    "Keywords",
    "L",
    "LLMJudge",
    "Language",
    "Length",
    "Lens",
    "LexicalSupport",
    "Mode",
    "Moderation",
    "Notify",
    "Outcome",
    "Patterns",
    "Policy",
    "Presidio",
    "Purpose",
    "Queue",
    "Redact",
    "Refusal",
    "RestrictTools",
    "Rule",
    "Sample",
    "Score",
    "Secrets",
    "Stage",
    "Subject",
    "Text",
    "ToolErrors",
    "TurnView",
    "TypeSafe",
    "Vote",
    "register",
]
