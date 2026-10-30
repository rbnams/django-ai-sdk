"""Example lenses, rules and policies the console starts from.

Complete definitions in the JSON the overlay stores, so "start from an example"
is a copy and nothing else. `why` says what an example does for you; `try` is a
message it is about, so an admin sees it in use before picking it. Messages a person
sees are marked for translation (gettext_noop) and stored in English; the turn shows
them in the active language.
"""

from __future__ import annotations

from typing import Any

from django.utils.translation import gettext_noop

# Answerers name aliases optionally ("classifier?", "judge?"), so an example
# loads on any deployment; ai_sdk.oversight.W003 says where none of them is defined.
# Each example picks its own: the classifier alone, or the classifier with a judge
# where the classifier may be unsure.

_CLASSIFIER = ["classifier?"]
_CLASSIFIER_THEN_JUDGE = ["classifier?", "judge?"]

LENS_EXAMPLES: list[dict[str, Any]] = [
    # Support
    {
        "group": "Customer support",
        "why": "Route by what the customer wants.",
        "try": "Where is my order? I ordered on Monday.",
        "lens": {
            "kind": "choice",
            "name": "support_intent",
            "label": "Support intent",
            "question": "What does the customer want?",
            "labels": ["order_status", "refund", "complaint", "product_question", "other"],
            "criteria": {
                "order_status": "Where an order is, or when it arrives.",
                "refund": "Money back, a return or compensation.",
                "complaint": "Unhappy about the product or service.",
                "product_question": "How a product works or what it can do.",
            },
            "subject": "input",
            "stages": ["gate"],
            "answerers": _CLASSIFIER,
        },
    },
    {
        "group": "Customer support",
        "why": "Catch angry customers before they leave.",
        "try": "This is the third time I ask. Nobody answers!",
        "lens": {
            "kind": "score",
            "name": "frustration",
            "label": "Frustration",
            "question": "How frustrated is the customer?",
            "low": "calm",
            "high": "furious",
            "subject": "input",
            "stages": ["gate"],
            "answerers": _CLASSIFIER,
        },
    },
    {
        "group": "Customer support",
        "why": "The bot may not promise money.",
        "try": "Answer: Sorry about that, I've refunded the full amount.",
        "lens": {
            "kind": "flag",
            "name": "promised_refund",
            "label": "Promised compensation",
            "question": "Does the answer promise a refund, discount or other compensation?",
            "yes": "Promises or confirms money back, a discount or a free item.",
            "no": "Explains the policy or refers to a colleague.",
            "subject": "output",
            "stages": ["finish"],
            "answerers": _CLASSIFIER_THEN_JUDGE,
        },
    },
    # Health
    {
        "group": "Health",
        "why": "Emergencies go to 112 at once.",
        "try": "I have chest pain and my left arm feels numb.",
        "lens": {
            "kind": "flag",
            "name": "urgent_symptoms",
            "label": "Urgent symptoms",
            "question": "Does the message describe symptoms that need emergency care?",
            "yes": "Describes symptoms happening now that may need emergency care.",
            "no": "Past events, general health questions, or someone else's history.",
            "subject": "input",
            "stages": ["gate"],
            "answerers": _CLASSIFIER,
        },
    },
    {
        "group": "Health",
        "why": "Explaining is fine; dosing is not.",
        "try": "Answer: Take two 500 mg tablets every four hours.",
        "lens": {
            "kind": "flag",
            "name": "gives_dosage",
            "label": "Gives a dosage",
            "question": "Does the answer tell the person how much of a medicine to take?",
            "yes": "Names an amount, frequency or schedule for a medicine.",
            "no": "Mentions medicines without amounts, or refers to a doctor.",
            "subject": "output",
            "stages": ["finish"],
            "answerers": _CLASSIFIER_THEN_JUDGE,
        },
    },
    # HR
    {
        "group": "HR",
        "why": "Nobody asks about a colleague's salary or health.",
        "try": "How much does Sanne from finance earn?",
        "lens": {
            "kind": "flag",
            "name": "about_a_colleague",
            "label": "About a colleague",
            "question": "Does the message ask for another employee's salary, performance, "
            "health or personal details?",
            "yes": "Asks about a named or described colleague's private information.",
            "no": "Asks about the person's own situation, or about general policy.",
            "subject": "input",
            "stages": ["gate"],
            "answerers": _CLASSIFIER,
        },
    },
    {
        "group": "HR",
        "why": "See what the handbook is missing.",
        "try": "How many vacation days do I have left?",
        "lens": {
            "kind": "choice",
            "name": "hr_topic",
            "label": "HR topic",
            "question": "What HR topic is this about?",
            "labels": ["leave", "payroll", "onboarding", "handbook", "other"],
            "subject": "input",
            "stages": ["gate"],
            "answerers": _CLASSIFIER,
        },
    },
    # Agents with tools
    {
        "group": "Agents with tools",
        "why": "Deletes and sends need a person.",
        "try": "Tool call: delete_customer(id=4021)",
        "lens": {
            "kind": "flag",
            "name": "risky_tool",
            "label": "Risky tool call",
            "question": "Would this tool call delete data or send something outside the app?",
            "subject": "tool_call",
            "stages": ["watch"],
            "answerers": _CLASSIFIER,
        },
    },
    # School
    {
        "group": "School",
        "why": "Help students learn, not do their work.",
        "try": "Answer: Here is your full essay, ready to hand in.",
        "lens": {
            "kind": "flag",
            "name": "did_the_homework",
            "label": "Did the homework",
            "question": "Did the assistant write the assignment for the student instead of "
            "helping them learn?",
            "subject": "exchange",
            "stages": ["reflect"],
            "sample": 0.2,
            "answerers": _CLASSIFIER_THEN_JUDGE,
        },
    },
    {
        "group": "School",
        "why": "Content must suit young readers.",
        "try": "Answer: a graphic description of a crime scene.",
        "lens": {
            "kind": "score",
            "name": "age_appropriate",
            "label": "Suits a 14-year-old",
            "question": "How suitable is this answer for a 14-year-old?",
            "low": "unsuitable",
            "high": "fully suitable",
            "subject": "output",
            "stages": ["finish"],
            "answerers": _CLASSIFIER_THEN_JUDGE,
        },
    },
    # Sales
    {
        "group": "Sales",
        "why": "Tell sales when someone is ready to buy.",
        "try": "What does the business plan cost for 20 people?",
        "lens": {
            "kind": "choice",
            "name": "buying_signal",
            "label": "Buying signal",
            "question": "How close is this person to buying?",
            "labels": ["none", "curious", "comparing", "ready_to_buy"],
            "default": "none",
            "subject": "input",
            "stages": ["reflect"],
            "answerers": _CLASSIFIER_THEN_JUDGE,
        },
    },
    {
        "group": "Sales",
        "why": "Which competitors come up.",
        "try": "Is this cheaper than Zendesk?",
        "lens": {
            "kind": "entities",
            "name": "competitors",
            "label": "Competitors named",
            "question": "Which competitors does the message name?",
            "labels": ["acme", "globex", "initech"],
            "subject": "input",
            "stages": ["gate"],
            "answerers": [
                {
                    "class": "Keywords",
                    "keywords": {"acme": ["acme"], "globex": ["globex"], "initech": ["initech"]},
                }
            ],
        },
    },
    # Product insight
    {
        "group": "Product insight",
        "why": "A reading list of what people ask for.",
        "try": "It would be great if I could export to Excel.",
        "lens": {
            "kind": "text",
            "name": "feature_request",
            "label": "Feature request",
            "question": "If the person asks for a feature, describe it in one sentence. "
            "Otherwise answer with an empty string.",
            "subject": "input",
            "stages": ["reflect"],
            "sample": 0.3,
            "max_chars": 200,
            "answerers": ["judge?"],
        },
    },
    {
        "group": "Product insight",
        "why": "Reads like a bug list.",
        "try": "The upload button does nothing on my phone.",
        "lens": {
            "kind": "text",
            "name": "what_went_wrong",
            "label": "What went wrong",
            "question": "If the answer is wrong or unhelpful, say why in one sentence. "
            "Otherwise answer with an empty string.",
            "subject": "exchange",
            "stages": ["reflect"],
            "sample": 0.1,
            "answerers": ["judge?", "strong?"],
        },
    },
    # General
    {
        "group": "General",
        "why": "Every message gets a tone.",
        "try": "Thanks, that helped a lot!",
        "lens": {
            "kind": "choice",
            "name": "tone",
            "label": "Tone",
            "question": "What is the tone of the message?",
            "labels": ["neutral", "friendly", "frustrated", "hostile"],
            "default": "neutral",
            "subject": "input",
            "stages": ["reflect"],
            "answerers": _CLASSIFIER_THEN_JUDGE,
        },
    },
    {
        "group": "General",
        "why": "Answers that are far too long.",
        "try": "Answer: twelve paragraphs for a yes/no question.",
        "lens": {
            "kind": "flag",
            "name": "too_long",
            "label": "Too long",
            "question": "Is the answer longer than 3000 characters?",
            "subject": "output",
            "stages": ["finish"],
            "answerers": [{"class": "Length", "max_chars": 3000}],
        },
    },
]

RULE_EXAMPLES: list[dict[str, Any]] = [
    {
        "group": "Customer support",
        "needs": ["support_intent"],
        "why": "Finance sees every refund request.",
        "try": "“I want my money back” → finance gets it in their queue.",
        "rule": {
            "name": "refunds_to_finance",
            "when": {"lens": "support_intent", "op": "eq", "value": "refund"},
            "then": [{"kind": "queue", "queue": "finance", "priority": 10}],
            "description": "Every refund question lands in the finance queue.",
        },
    },
    {
        "group": "Customer support",
        "needs": ["frustration"],
        "why": "A person steps in before a customer gives up.",
        "try": "“Nobody answers!” → a person takes over.",
        "rule": {
            "name": "angry_to_human",
            "when": {"lens": "frustration", "op": "ge", "value": 0.8},
            "then": [
                {"kind": "queue", "queue": "support", "priority": 50},
                {"kind": "alert", "severity": "warning"},
            ],
            "description": "Very frustrated customers get a person, soon.",
        },
    },
    {
        "group": "Customer support",
        "needs": ["promised_refund"],
        "why": "The bot cannot promise money it may not give.",
        "try": "“I've refunded you” → the answer is withheld.",
        "rule": {
            "name": "no_unapproved_promises",
            "when": {"lens": "promised_refund", "op": "is_true"},
            "then": [
                {
                    "kind": "block",
                    "message": gettext_noop(
                        "A colleague will get back to you about this within a day."
                    ),
                }
            ],
            "description": "In live mode the promise is replaced when the answer ends; hold the "
            "stream to never show it.",
        },
    },
    {
        "group": "Health",
        "needs": ["urgent_symptoms"],
        "why": "Urgent symptoms get the emergency number at once.",
        "try": "“Chest pain, numb arm” → told to call 112 straight away.",
        "rule": {
            "name": "emergency",
            "when": {"lens": "urgent_symptoms", "op": "is_true"},
            "then": [
                {
                    "kind": "block",
                    "message": gettext_noop(
                        "This sounds urgent. Call 112 or the GP emergency line now."
                    ),
                },
                {"kind": "notify", "severity": "critical"},
            ],
        },
    },
    {
        "group": "Health",
        "needs": ["gives_dosage"],
        "why": "Dosing advice never reaches a patient.",
        "try": "“Take two tablets every four hours” → answer withheld.",
        "rule": {
            "name": "no_dosage",
            "when": {"lens": "gives_dosage", "op": "is_true"},
            "then": [
                {"kind": "block", "message": gettext_noop("Please discuss this with your GP.")}
            ],
        },
    },
    {
        "group": "HR",
        "needs": ["about_a_colleague"],
        "why": "Questions about colleagues are refused.",
        "try": "“What does Sanne earn?” → politely refused.",
        "rule": {
            "name": "not_about_others",
            "when": {
                "all": [
                    {"lens": "about_a_colleague", "op": "is_true"},
                    {"lens": "about_a_colleague", "op": "confident", "value": 0.8},
                ]
            },
            "then": [
                {
                    "kind": "block",
                    "message": gettext_noop("I can only help with your own HR questions."),
                }
            ],
        },
    },
    {
        "group": "Agents with tools",
        "needs": ["risky_tool"],
        "why": "Deletes and sends wait for approval.",
        "try": "delete_customer(…) → waits for a person to approve.",
        "rule": {
            "name": "risky_tools_need_a_person",
            "when": {"lens": "risky_tool", "op": "is_true"},
            "then": [
                {
                    "kind": "block",
                    "message": gettext_noop("That action needs approval. I've asked a colleague."),
                },
                {"kind": "queue", "queue": "approvals", "priority": 30},
            ],
        },
    },
    {
        "group": "Agents with tools",
        "needs": ["prompt_injection"],
        "why": "Suspicious, not certain: talk, but no tools.",
        "try": "A half-sure injection → the bot talks, but uses no tools.",
        "rule": {
            "name": "suspicious_means_no_tools",
            "when": {
                "all": [
                    {"lens": "prompt_injection", "op": "is_true"},
                    {"lens": "prompt_injection", "op": "unsure", "value": 0.9},
                ]
            },
            "then": [{"kind": "restrict_tools", "names": []}],
        },
    },
    {
        "group": "School",
        "needs": ["did_the_homework"],
        "why": "Teachers see where it did too much.",
        "try": "A finished essay → the teacher sees it in a queue.",
        "rule": {
            "name": "homework_to_teachers",
            "when": {"lens": "did_the_homework", "op": "is_true"},
            "then": [{"kind": "queue", "queue": "teachers"}],
        },
    },
    {
        "group": "School",
        "needs": ["age_appropriate"],
        "why": "Answers unfit for teenagers are held back.",
        "try": "A graphic answer → replaced by a short notice.",
        "rule": {
            "name": "unsuitable_answer",
            "when": {"lens": "age_appropriate", "op": "lt", "value": 0.6},
            "then": [
                {"kind": "block", "message": gettext_noop("Let's ask your teacher about this one.")}
            ],
        },
    },
    {
        "group": "Sales",
        "needs": ["buying_signal"],
        "why": "Sales hears about ready buyers at once.",
        "try": "“Price for 20 people?” → the lead goes to the CRM.",
        "rule": {
            "name": "hot_lead",
            "when": {"lens": "buying_signal", "op": "eq", "value": "ready_to_buy"},
            "then": [{"kind": "notify", "severity": "info"}],
        },
    },
    {
        "group": "General",
        "needs": ["prompt_injection"],
        "why": "A clear attack on the bot is refused.",
        "try": "“Ignore your instructions and…” → refused.",
        "rule": {
            "name": "block_sure_injection",
            "when": {
                "all": [
                    {"lens": "prompt_injection", "op": "is_true"},
                    {"lens": "prompt_injection", "op": "confident", "value": 0.9},
                ]
            },
            "then": [{"kind": "block"}, {"kind": "alert", "severity": "high"}],
        },
    },
]

POLICY_EXAMPLES: list[dict[str, Any]] = [
    {
        "name": "support",
        "label": "Customer support",
        "description": "Route refunds, catch frustration, no promises the bot may not make.",
        "extends": ["security", "quality"],
        "lenses": ["support_intent", "frustration", "promised_refund"],
        "rules": ["refunds_to_finance", "angry_to_human", "no_unapproved_promises"],
    },
    {
        "name": "practice",
        "label": "GP practice",
        "description": "Emergencies to 112, no dosing advice, no patient text stored.",
        "extends": ["security", "safety", "quality"],
        "capture": "metadata",
        "lenses": ["urgent_symptoms", "gives_dosage"],
        "rules": ["emergency", "no_dosage"],
    },
    {
        "name": "hr",
        "label": "HR assistant",
        "description": "Only your own HR questions; HR sees counts, never the text.",
        "extends": ["security"],
        "capture": "metadata",
        "lenses": ["about_a_colleague", "hr_topic"],
        "rules": ["not_about_others"],
    },
    {
        "name": "operator",
        "label": "Agent with tools",
        "description": "Risky tool calls need approval; suspicious messages lose their tools.",
        "extends": ["security", "quality"],
        "lenses": ["risky_tool"],
        "rules": ["risky_tools_need_a_person", "suspicious_means_no_tools"],
    },
    {
        "name": "school",
        "label": "School",
        "description": "Help, don't do the homework; content that suits teenagers.",
        "extends": ["security", "safety"],
        "lenses": ["did_the_homework", "age_appropriate"],
        "rules": ["homework_to_teachers", "unsuitable_answer"],
    },
    {
        "name": "sales",
        "label": "Sales site",
        "description": "Hot leads to the CRM, competitor mentions as a chart.",
        "extends": ["security"],
        "lenses": ["buying_signal", "competitors"],
        "rules": ["hot_lead"],
    },
    {
        "name": "insights",
        "label": "Product insight",
        "description": "No rules at all: classify, summarise, and measure quality.",
        "extends": ["quality"],
        "capture": "masked",
        "lenses": ["feature_request", "what_went_wrong", "tone"],
        "rules": [],
    },
]

__all__ = ["LENS_EXAMPLES", "POLICY_EXAMPLES", "RULE_EXAMPLES"]
