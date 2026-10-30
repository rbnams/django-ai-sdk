"""Oversight: check and record every turn an agent runs, whichever way it runs.

Add "django_ai_sdk.oversight" to INSTALLED_APPS and give an agent a policy. Every
turn it runs (chat, Agent.run(), the HTTP run endpoints, webhooks, workflow steps)
is judged by the policy's lenses, acted on by its rules, and lands on a
hash-chained ledger::

    from django_ai_sdk import Agent
    from django_ai_sdk.errors import Blocked
    from django_ai_sdk.oversight import (
        PII, Entities, L, Mask, Mode, Policy, Queue, Rule, Stage, Subject, pick,
    )
    from django_ai_sdk.oversight.packs import SecurityPack

    class SupportPolicy(Policy):
        name = "support"
        stream_mode = "hold"  # the answer is shown once finish has masked it
        lenses = [
            *pick(SecurityPack.lenses, "prompt_injection"),
            Entities(name="pii", question="Which kinds of personal data does the answer contain?",
                     subject=Subject.OUTPUT, stages=(Stage.FINISH,), answerers=[PII()]),
        ]
        rules = [
            *pick(SecurityPack.rules, "block_injection", mode=Mode.ENFORCE),
            Rule("mask_pii", when=L("pii").any, then=Mask(), mode=Mode.ENFORCE),
            Rule("iban_to_people", when=L("pii").contains("iban"), then=Queue("privacy")),
        ]

    class SupportAgent(Agent):
        oversight = SupportPolicy  # registered with the agent; or a registered name

    try:
        answer = await SupportAgent().run(messages, user=user)
    except Blocked as refused:
        answer = refused.message

Lenses observe, rules act; a rule starts in report mode, recording what it would
have done. Use `@register` for a policy no agent class names. See
docs/content/oversight.md.

Only the declaration API is re-exported here. Specialised detectors and vendor
answerers (TypeSafe) live in `django_ai_sdk.oversight.answerers`, the built-in
packs in `django_ai_sdk.oversight.packs`; models, services and tasks in their own
modules, so importing this package needs no app registry.
"""

from __future__ import annotations

from django_ai_sdk.oversight.answerers import (
    PII,
    Function,
    Human,
    Keywords,
    LLMJudge,
    Patterns,
    Secrets,
)
from django_ai_sdk.oversight.conditions import L
from django_ai_sdk.oversight.lenses import Choice, Entities, Flag, Lens, Score, Text
from django_ai_sdk.oversight.policy import Policy, pick
from django_ai_sdk.oversight.registry import register
from django_ai_sdk.oversight.rules import (
    Alert,
    Block,
    Mask,
    Notify,
    Queue,
    RestrictTools,
    Rule,
)
from django_ai_sdk.oversight.types import (
    Answer,
    Capture,
    Mode,
    Purpose,
    Severity,
    Stage,
    Subject,
    TurnView,
)

__all__ = [
    "PII",
    "Alert",
    "Answer",
    "Block",
    "Capture",
    "Choice",
    "Entities",
    "Flag",
    "Function",
    "Human",
    "Keywords",
    "L",
    "LLMJudge",
    "Lens",
    "Mask",
    "Mode",
    "Notify",
    "Patterns",
    "Policy",
    "Purpose",
    "Queue",
    "RestrictTools",
    "Rule",
    "Score",
    "Secrets",
    "Severity",
    "Stage",
    "Subject",
    "Text",
    "TurnView",
    "pick",
    "register",
]
