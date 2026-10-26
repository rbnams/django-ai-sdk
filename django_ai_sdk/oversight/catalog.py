"""What the console offers to build with: answerer kinds, models, and examples.

Everything here is data the editor renders. An answerer kind lists its
parameters, so the console can show a form for it and save the result as a
config dict (`{"class": "Patterns", "patterns": [...]}`) that `resolve` builds.
The examples are complete lens, rule and policy definitions in the same JSON
the overlay stores, so "start from an example" is a copy and nothing else.

Hosts add model choices with AI_SDK_OVERSIGHT["MODEL_CHOICES"]: a list of
`{"id", "label", "group", "kinds"}` or the dotted path of a function returning one.
"""

from __future__ import annotations

from typing import Any

from django.utils.module_loading import import_string

from django_ai_sdk.logger import get_logger

logger = get_logger(__name__)

# Parameter types the editor knows how to render.
#   text, number, bool, choice (from `choices`), multi (several of `choices`),
#   lines (one string per line), labelled_lines ({label: [lines]}), model, agent
P = dict[str, Any]


def _param(name: str, type: str, label: str, help: str = "", **extra: Any) -> P:
    return {"name": name, "type": type, "label": label, "help": help, **extra}


# Answerers an alias can name: the ones worth configuring once and reusing.
ALIAS_KINDS: list[dict[str, Any]] = [
    {
        "class": "LLMJudge",
        "label": "Model judge",
        "summary": "A chat model reads the question and answers it with a value, a confidence "
        "and a reason. Any model your gateway serves.",
        "system_one": False,
        "params": [
            _param("model", "model", "Model", "Which chat model judges.", required=True),
            _param(
                "timeout",
                "number",
                "Timeout (seconds)",
                "After this the answer counts as failed.",
                default=45,
            ),
        ],
    },
    {
        "class": "TypeSafe",
        "label": "Jev (TypeSafe)",
        "summary": "A model built for typed questions: yes/no, one label, or a score. Fast "
        "(about 0.3 s), so it can sit at the gate.",
        "system_one": False,
        "params": [
            _param("model", "model", "Jev version", "Empty uses TYPESAFE_MODEL.", kind="TypeSafe"),
        ],
    },
    {
        "class": "HFClassifier",
        "label": "Hugging Face classifier",
        "summary": "A small classifier such as Llama Prompt Guard, on this machine or an "
        "inference endpoint.",
        "system_one": False,
        "params": [
            _param(
                "model",
                "text",
                "Model id",
                "For example meta-llama/Llama-Prompt-Guard-2-86M.",
                required=True,
            ),
            _param(
                "positive", "lines", "Positive labels", "Labels that mean yes.", default=["LABEL_1"]
            ),
            _param("endpoint", "text", "Endpoint URL", "Empty runs the model locally."),
        ],
    },
    {
        "class": "Moderation",
        "label": "OpenAI moderation",
        "summary": "OpenAI's moderation endpoint. Only on OpenAI itself, not on a gateway.",
        "system_one": False,
        "params": [_param("model", "text", "Model", default="omni-moderation-latest")],
    },
    {
        "class": "AgentJudge",
        "label": "An agent",
        "summary": "One of your agents answers, with its own prompt and tools. Oversight is "
        "off inside its run.",
        "system_one": False,
        "params": [_param("agent", "agent", "Agent", required=True)],
    },
    {
        "class": "Human",
        "label": "A person",
        "summary": "The turn goes to a review queue; someone answers later.",
        "system_one": True,
        "params": [
            _param("queue", "text", "Queue", default="default"),
            _param("priority", "number", "Priority", "Higher is sooner.", default=0),
        ],
    },
]

PII_KINDS = ["email", "iban", "credit_card", "bsn", "phone", "ip_address", "postcode_nl"]

# Free, local answerers a lens can use inline, without an alias.
INLINE_KINDS: list[dict[str, Any]] = [
    {
        "class": "Patterns",
        "label": "Patterns (regex)",
        "summary": "Yes when one of the regular expressions matches. Case-insensitive.",
        "fits": ["flag", "entities", "choice", "score"],
        "params": [
            _param(
                "patterns", "lines", "Patterns", "One regular expression per line.", required=True
            ),
            _param(
                "miss_confidence",
                "number",
                "Sure when nothing matches",
                "Below the lens's threshold, the next answerer is asked.",
                default=0.9,
            ),
        ],
    },
    {
        "class": "Keywords",
        "label": "Keywords",
        "summary": "Whole words per label; the label with the most hits wins. The free way to "
        "classify.",
        "fits": ["choice", "entities", "flag"],
        "params": [
            _param(
                "keywords",
                "labelled_lines",
                "Words per label",
                "One word or phrase per line.",
                required=True,
            ),
            _param("miss_confidence", "number", "Sure when nothing matches", default=0.5),
        ],
    },
    {
        "class": "PII",
        "label": "Personal data",
        "summary": "Email, IBAN (mod-97), card numbers (Luhn), BSN (11-check), phone, IP, "
        "postcode.",
        "fits": ["entities", "flag"],
        "params": [
            _param("kinds", "multi", "Kinds", "None picked: all of them.", choices=PII_KINDS)
        ],
    },
    {
        "class": "Secrets",
        "label": "Secrets",
        "summary": "API keys (OpenAI, GitHub, AWS, Slack), JWTs, private keys, passwords.",
        "fits": ["entities", "flag"],
        "params": [],
    },
    {
        "class": "Language",
        "label": "Language",
        "summary": "Which language (a Choice), or whether the answer's language differs from "
        "the question's (a Flag).",
        "fits": ["choice", "flag"],
        "params": [],
    },
    {
        "class": "Length",
        "label": "Length",
        "summary": "Over a number of characters (a Flag), or how far over (a Score).",
        "fits": ["flag", "score"],
        "params": [
            _param("max_chars", "number", "Maximum characters", required=True, default=4000)
        ],
    },
    {
        "class": "Refusal",
        "label": "Refusal",
        "summary": 'Whether the assistant declined ("I can\'t help with that").',
        "fits": ["flag"],
        "params": [],
    },
    {
        "class": "EmptyRetrieval",
        "label": "Nothing found",
        "summary": "The assistant searched and found no sources.",
        "fits": ["flag"],
        "params": [],
    },
    {
        "class": "ToolErrors",
        "label": "Tool failures",
        "summary": "The share of tool calls that failed.",
        "fits": ["score", "flag"],
        "params": [],
    },
    {
        "class": "LexicalSupport",
        "label": "Word overlap",
        "summary": "How much of the answer's wording is in the sources. Rough, so it is never "
        "sure: a floor under a judge.",
        "fits": ["score"],
        "params": [],
    },
    {
        "class": "Function",
        "label": "Python function",
        "summary": "A function in the host's code, by dotted path: fn(lens, view) -> value.",
        "fits": ["flag", "choice", "score", "entities", "text"],
        "params": [
            _param(
                "fn",
                "text",
                "Dotted path",
                "For example agents.oversight.used_the_wiki",
                required=True,
            )
        ],
    },
]

LENS_KINDS = [
    {"kind": "flag", "label": "Yes / no", "summary": '"Does it…?" A rule tests is yes or is no.'},
    {"kind": "choice", "label": "One label", "summary": "Classify: exactly one of your labels."},
    {"kind": "score", "label": "Score 0–1", "summary": '"How much…?" A rule tests above or below.'},
    {
        "kind": "entities",
        "label": "Which things",
        "summary": "Which of your labels occur, and where, so a rule can redact them.",
    },
    {"kind": "text", "label": "Short text", "summary": "A summary or reason, for people to read."},
]

SUBJECTS = [
    {"value": "input", "label": "The question"},
    {"value": "output", "label": "The answer"},
    {"value": "exchange", "label": "Question and answer"},
    {"value": "sources", "label": "Sources the model read"},
    {"value": "tool_call", "label": "A tool call"},
]

STAGE_INFO = [
    {
        "value": "gate",
        "label": "Before the model",
        "summary": "Can block, redact the question or "
        "take tools away. Every millisecond delays the reply.",
    },
    {
        "value": "watch",
        "label": "While it works",
        "summary": "Checks each tool call before it runs, and the streamed text.",
    },
    {
        "value": "review",
        "label": "Answer finished",
        "summary": "Before the answer is stored. "
        "Can block or redact it; in hold mode the user waits for this.",
    },
    {
        "value": "reflect",
        "label": "In the background",
        "summary": "After the reply, sampled and "
        "budgeted. Can queue and alert, not change the answer.",
    },
]

OPS = [
    {"op": "is_true", "label": "is yes", "kinds": ["flag"], "value": False},
    {"op": "is_false", "label": "is no", "kinds": ["flag"], "value": False},
    {"op": "eq", "label": "is", "kinds": ["choice", "text"], "value": "label"},
    {"op": "ne", "label": "is not", "kinds": ["choice", "text"], "value": "label"},
    {"op": "in", "label": "is one of", "kinds": ["choice", "entities"], "value": "labels"},
    {"op": "contains", "label": "contains", "kinds": ["entities", "text"], "value": "label"},
    {"op": "any", "label": "found anything", "kinds": ["entities", "text"], "value": False},
    {"op": "ge", "label": "is at least", "kinds": ["score"], "value": "number"},
    {"op": "gt", "label": "is above", "kinds": ["score"], "value": "number"},
    {"op": "le", "label": "is at most", "kinds": ["score"], "value": "number"},
    {"op": "lt", "label": "is below", "kinds": ["score"], "value": "number"},
    {
        "op": "confident",
        "label": "answered with confidence at least",
        "kinds": ["flag", "choice", "score", "entities", "text"],
        "value": "number",
    },
    {
        "op": "unsure",
        "label": "answered with confidence below",
        "kinds": ["flag", "choice", "score", "entities", "text"],
        "value": "number",
    },
    {
        "op": "missing",
        "label": "has no answer",
        "kinds": ["flag", "choice", "score", "entities", "text"],
        "value": False,
    },
]

ACTION_KINDS = [
    {
        "kind": "block",
        "label": "Block",
        "summary": "Stop the turn and show a message.",
        "stages": ["gate", "watch", "review"],
        "params": [_param("message", "text", "Message shown", default="")],
    },
    {
        "kind": "redact",
        "label": "Redact",
        "summary": "Mask what the lenses found, as [IBAN].",
        "stages": ["gate", "watch", "review", "reflect"],
        "params": [_param("template", "text", "Mask", default="[{label}]")],
    },
    {
        "kind": "restrict_tools",
        "label": "Take tools away",
        "summary": "For this turn: the named tools, or all when empty.",
        "stages": ["gate", "watch"],
        "params": [_param("names", "lines", "Tool names", "Empty: every tool.")],
    },
    {
        "kind": "queue",
        "label": "Send to a person",
        "summary": "Put the turn in a review queue.",
        "stages": ["gate", "watch", "review", "reflect"],
        "params": [
            _param("queue", "text", "Queue", default="default"),
            _param("priority", "number", "Priority", default=0),
        ],
    },
    {
        "kind": "escalate",
        "label": "Ask a stronger answerer",
        "summary": "Re-judge a lens with another answerer in the background.",
        "stages": ["gate", "watch", "review", "reflect"],
        "params": [
            _param("lens", "lens", "Lens", required=True),
            _param("answerer", "alias", "Answerer", default="strong"),
        ],
    },
    {
        "kind": "notify",
        "label": "Notify",
        "summary": "Send to exporters: Slack, webhook, SIEM.",
        "stages": ["gate", "watch", "review", "reflect"],
        "params": [
            _param(
                "severity",
                "choice",
                "Severity",
                default="warning",
                choices=["info", "warning", "error", "critical"],
            )
        ],
    },
    {
        "kind": "alert",
        "label": "Alert",
        "summary": "Mark the turn in the console.",
        "stages": ["gate", "watch", "review", "reflect"],
        "params": [
            _param(
                "severity",
                "choice",
                "Severity",
                default="warning",
                choices=["info", "warning", "error", "critical"],
            )
        ],
    },
    {
        "kind": "allow",
        "label": "Allow",
        "summary": "Record that it is fine.",
        "stages": ["gate", "watch", "review", "reflect"],
        "params": [],
    },
]


def model_choices() -> list[dict[str, Any]]:
    """The models the console offers, from AI_SDK_OVERSIGHT["MODEL_CHOICES"]."""
    from django_ai_sdk.oversight.conf import setting

    configured = setting("MODEL_CHOICES")
    if isinstance(configured, str):
        try:
            configured = import_string(configured)
        except ImportError:
            logger.warning("MODEL_CHOICES names {!r}, which does not import", configured)
            return []
    if callable(configured):
        try:
            configured = configured()
        except Exception:
            logger.exception("MODEL_CHOICES failed")
            return []
    return [
        {
            "id": str(choice["id"]),
            "label": str(choice.get("label") or choice["id"]),
            "group": str(choice.get("group") or ""),
            "kinds": list(choice.get("kinds") or ["LLMJudge"]),
        }
        for choice in configured or []
        if isinstance(choice, dict) and choice.get("id")
    ]


# Examples. Answerers name aliases optionally ("typesafe?", "fast?"), so an example
# works on any deployment and gets better where those are configured.

_JUDGE = ["typesafe?", "fast?"]

LENS_EXAMPLES: list[dict[str, Any]] = [
    # Support
    {
        "group": "Customer support",
        "why": "Route by what the customer wants.",
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
            "answerers": [
                {
                    "class": "Keywords",
                    "keywords": {
                        "refund": ["refund", "money back", "geld terug", "retour", "return"],
                        "order_status": ["where is my order", "track", "bezorging", "delivery"],
                    },
                },
                *_JUDGE,
            ],
        },
    },
    {
        "group": "Customer support",
        "why": "Catch angry customers before they leave.",
        "lens": {
            "kind": "score",
            "name": "frustration",
            "label": "Frustration",
            "question": "How frustrated is the customer?",
            "low": "calm",
            "high": "furious",
            "subject": "input",
            "stages": ["review"],
            "answerers": _JUDGE,
        },
    },
    {
        "group": "Customer support",
        "why": "The bot may not promise money.",
        "lens": {
            "kind": "flag",
            "name": "promised_refund",
            "label": "Promised compensation",
            "question": "Does the answer promise a refund, discount or other compensation?",
            "yes": "Promises or confirms money back, a discount or a free item.",
            "no": "Explains the policy or refers to a colleague.",
            "subject": "output",
            "stages": ["review"],
            "answerers": _JUDGE,
        },
    },
    # Health
    {
        "group": "Health",
        "why": "Emergencies go to 112 at once.",
        "lens": {
            "kind": "flag",
            "name": "urgent_symptoms",
            "label": "Urgent symptoms",
            "question": "Does the message describe symptoms that need emergency care?",
            "subject": "input",
            "stages": ["gate"],
            "answerers": [
                {
                    "class": "Patterns",
                    "patterns": [
                        r"chest pain|pijn op de borst",
                        r"can'?t breathe|benauwd|ademnood",
                        r"unconscious|bewusteloos",
                    ],
                    "miss_confidence": 0.7,
                },
                *_JUDGE,
            ],
        },
    },
    {
        "group": "Health",
        "why": "Explaining is fine; dosing is not.",
        "lens": {
            "kind": "flag",
            "name": "gives_dosage",
            "label": "Gives a dosage",
            "question": "Does the answer tell the person how much of a medicine to take?",
            "yes": "Names an amount, frequency or schedule for a medicine.",
            "no": "Mentions medicines without amounts, or refers to a doctor.",
            "subject": "output",
            "stages": ["review"],
            "answerers": _JUDGE,
        },
    },
    # HR
    {
        "group": "HR",
        "why": "Nobody asks about a colleague's salary or health.",
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
            "answerers": _JUDGE,
        },
    },
    {
        "group": "HR",
        "why": "See what the handbook is missing.",
        "lens": {
            "kind": "choice",
            "name": "hr_topic",
            "label": "HR topic",
            "question": "What HR topic is this about?",
            "labels": ["leave", "payroll", "onboarding", "handbook", "other"],
            "subject": "input",
            "stages": ["gate"],
            "answerers": [
                {
                    "class": "Keywords",
                    "keywords": {
                        "leave": ["leave", "verlof", "vakantie", "holiday"],
                        "payroll": ["salary", "salaris", "payslip", "loonstrook"],
                        "onboarding": ["first day", "onboarding", "laptop"],
                    },
                },
                *_JUDGE,
            ],
        },
    },
    # Agents with tools
    {
        "group": "Agents with tools",
        "why": "Deletes and sends need a person.",
        "lens": {
            "kind": "flag",
            "name": "risky_tool",
            "label": "Risky tool call",
            "question": "Would this tool call delete data or send something outside the app?",
            "subject": "tool_call",
            "stages": ["watch"],
            "answerers": [
                {
                    "class": "Patterns",
                    "patterns": [r"^(delete|drop|remove|send|email|post)_"],
                    "miss_confidence": 0.9,
                }
            ],
        },
    },
    # School
    {
        "group": "School",
        "why": "Help students learn, not do their work.",
        "lens": {
            "kind": "flag",
            "name": "did_the_homework",
            "label": "Did the homework",
            "question": "Did the assistant write the assignment for the student instead of "
            "helping them learn?",
            "subject": "exchange",
            "stages": ["reflect"],
            "sample": 0.2,
            "answerers": _JUDGE,
        },
    },
    {
        "group": "School",
        "why": "Content must suit young readers.",
        "lens": {
            "kind": "score",
            "name": "age_appropriate",
            "label": "Suits a 14-year-old",
            "question": "How suitable is this answer for a 14-year-old?",
            "low": "unsuitable",
            "high": "fully suitable",
            "subject": "output",
            "stages": ["review"],
            "answerers": _JUDGE,
        },
    },
    # Sales
    {
        "group": "Sales",
        "why": "Tell sales when someone is ready to buy.",
        "lens": {
            "kind": "choice",
            "name": "buying_signal",
            "label": "Buying signal",
            "question": "How close is this person to buying?",
            "labels": ["none", "curious", "comparing", "ready_to_buy"],
            "default": "none",
            "subject": "input",
            "stages": ["review"],
            "answerers": _JUDGE,
        },
    },
    {
        "group": "Sales",
        "why": "Which competitors come up.",
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
            "answerers": ["fast?"],
        },
    },
    {
        "group": "Product insight",
        "why": "Reads like a bug list.",
        "lens": {
            "kind": "text",
            "name": "what_went_wrong",
            "label": "What went wrong",
            "question": "If the answer is wrong or unhelpful, say why in one sentence. "
            "Otherwise answer with an empty string.",
            "subject": "exchange",
            "stages": ["reflect"],
            "sample": 0.1,
            "answerers": ["strong?", "fast?"],
        },
    },
    # General
    {
        "group": "General",
        "why": "Every message gets a tone.",
        "lens": {
            "kind": "choice",
            "name": "tone",
            "label": "Tone",
            "question": "What is the tone of the message?",
            "labels": ["neutral", "friendly", "frustrated", "hostile"],
            "default": "neutral",
            "subject": "input",
            "stages": ["reflect"],
            "answerers": _JUDGE,
        },
    },
    {
        "group": "General",
        "why": "Answers that are far too long.",
        "lens": {
            "kind": "flag",
            "name": "too_long",
            "label": "Too long",
            "question": "Is the answer longer than 3000 characters?",
            "subject": "output",
            "stages": ["review"],
            "answerers": [{"class": "Length", "max_chars": 3000}],
        },
    },
]

RULE_EXAMPLES: list[dict[str, Any]] = [
    {
        "group": "Customer support",
        "needs": ["support_intent"],
        "why": "Refunds to finance.",
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
        "why": "Angry customers to a person.",
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
        "why": "No unapproved promises.",
        "rule": {
            "name": "no_unapproved_promises",
            "when": {"lens": "promised_refund", "op": "is_true"},
            "then": [
                {
                    "kind": "block",
                    "message": "A colleague will get back to you about this within a day.",
                }
            ],
            "description": "Works best with hold mode: the customer never sees the promise.",
        },
    },
    {
        "group": "Health",
        "needs": ["urgent_symptoms"],
        "why": "Emergencies to 112.",
        "rule": {
            "name": "emergency",
            "when": {"lens": "urgent_symptoms", "op": "is_true"},
            "then": [
                {
                    "kind": "block",
                    "message": "This sounds urgent. Call 112 or the GP emergency line now.",
                },
                {"kind": "notify", "severity": "critical"},
            ],
        },
    },
    {
        "group": "Health",
        "needs": ["gives_dosage"],
        "why": "No dosing advice.",
        "rule": {
            "name": "no_dosage",
            "when": {"lens": "gives_dosage", "op": "is_true"},
            "then": [{"kind": "block", "message": "Please discuss this with your GP."}],
        },
    },
    {
        "group": "HR",
        "needs": ["about_a_colleague"],
        "why": "Only your own HR questions.",
        "rule": {
            "name": "not_about_others",
            "when": {
                "all": [
                    {"lens": "about_a_colleague", "op": "is_true"},
                    {"lens": "about_a_colleague", "op": "confident", "value": 0.8},
                ]
            },
            "then": [{"kind": "block", "message": "I can only help with your own HR questions."}],
        },
    },
    {
        "group": "Agents with tools",
        "needs": ["risky_tool"],
        "why": "Approval for risky actions.",
        "rule": {
            "name": "risky_tools_need_a_person",
            "when": {"lens": "risky_tool", "op": "is_true"},
            "then": [
                {"kind": "block", "message": "That action needs approval. I've asked a colleague."},
                {"kind": "queue", "queue": "approvals", "priority": 30},
            ],
        },
    },
    {
        "group": "Agents with tools",
        "needs": ["prompt_injection"],
        "why": "Suspicious, not certain: talk, but no tools.",
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
        "rule": {
            "name": "homework_to_teachers",
            "when": {"lens": "did_the_homework", "op": "is_true"},
            "then": [{"kind": "queue", "queue": "teachers"}],
        },
    },
    {
        "group": "School",
        "needs": ["age_appropriate"],
        "why": "Hold back unsuitable answers.",
        "rule": {
            "name": "unsuitable_answer",
            "when": {"lens": "age_appropriate", "op": "lt", "value": 0.6},
            "then": [{"kind": "block", "message": "Let's ask your teacher about this one."}],
        },
    },
    {
        "group": "Sales",
        "needs": ["buying_signal"],
        "why": "Hot leads to the CRM.",
        "rule": {
            "name": "hot_lead",
            "when": {"lens": "buying_signal", "op": "eq", "value": "ready_to_buy"},
            "then": [{"kind": "notify", "severity": "info"}],
        },
    },
    {
        "group": "General",
        "needs": ["prompt_injection"],
        "why": "A sure injection is refused.",
        "rule": {
            "name": "block_sure_injection",
            "when": {
                "all": [
                    {"lens": "prompt_injection", "op": "is_true"},
                    {"lens": "prompt_injection", "op": "confident", "value": 0.9},
                ]
            },
            "then": [{"kind": "block"}, {"kind": "alert", "severity": "error"}],
        },
    },
    {
        "group": "General",
        "needs": ["prompt_injection"],
        "why": "Unsure: ask a stronger judge.",
        "rule": {
            "name": "double_check_injection",
            "when": {
                "all": [
                    {"lens": "prompt_injection", "op": "is_true"},
                    {"lens": "prompt_injection", "op": "unsure", "value": 0.75},
                ]
            },
            "then": [{"kind": "escalate", "lens": "prompt_injection", "answerer": "strong"}],
        },
    },
]

POLICY_EXAMPLES: list[dict[str, Any]] = [
    {
        "name": "support",
        "label": "Customer support",
        "description": "Route refunds, catch frustration, no promises the bot may not make.",
        "extends": ["security", "quality"],
        "stream_mode": "hold",
        "lenses": ["support_intent", "frustration", "promised_refund"],
        "rules": ["refunds_to_finance", "angry_to_human", "no_unapproved_promises"],
    },
    {
        "name": "practice",
        "label": "GP practice",
        "description": "Emergencies to 112, no dosing advice, no patient text stored.",
        "extends": ["security", "safety", "quality"],
        "stream_mode": "hold",
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
        "stream_mode": "hold",
        "lenses": ["did_the_homework", "age_appropriate"],
        "rules": ["homework_to_teachers", "unsuitable_answer"],
    },
    {
        "name": "sales",
        "label": "Sales site",
        "description": "Hot leads to the CRM, competitor mentions as a chart.",
        "extends": ["security", "classify"],
        "lenses": ["buying_signal", "competitors"],
        "rules": ["hot_lead"],
    },
    {
        "name": "insights",
        "label": "Product insight",
        "description": "No rules at all: classify, summarise, and measure quality.",
        "extends": ["classify", "quality"],
        "capture": "redacted",
        "lenses": ["feature_request", "what_went_wrong", "tone"],
        "rules": [],
    },
]


def lens_example(name: str) -> dict[str, Any] | None:
    return next((dict(e["lens"]) for e in LENS_EXAMPLES if e["lens"]["name"] == name), None)


def rule_example(name: str) -> dict[str, Any] | None:
    return next((dict(e["rule"]) for e in RULE_EXAMPLES if e["rule"]["name"] == name), None)


def catalog() -> dict[str, Any]:
    """Everything the editor offers, in one payload."""
    from django_ai_sdk.oversight.answerers.base import alias_source, alias_table, resolve

    aliases = []
    for alias, config in sorted(alias_table().items()):
        try:
            described = resolve(alias).describe()
            error = ""
        except Exception as exc:
            described, error = {}, str(exc)
        aliases.append(
            {
                "alias": alias,
                "source": alias_source(alias),
                "config": config if isinstance(config, dict | str) else repr(config),
                "class": described.get("class", ""),
                "model": described.get("model", ""),
                "system_one": bool(described.get("system_one")),
                "error": error,
            }
        )
    aliases.append(
        {
            "alias": "human",
            "source": "builtin",
            "config": {"class": "Human"},
            "class": "Human",
            "model": "",
            "system_one": True,
            "error": "",
        }
    )
    return {
        "lens_kinds": LENS_KINDS,
        "subjects": SUBJECTS,
        "stages": STAGE_INFO,
        "ops": OPS,
        "actions": ACTION_KINDS,
        "alias_kinds": ALIAS_KINDS,
        "inline_kinds": INLINE_KINDS,
        "aliases": aliases,
        "models": model_choices(),
        "examples": {
            "lenses": LENS_EXAMPLES,
            "rules": RULE_EXAMPLES,
            "policies": POLICY_EXAMPLES,
        },
    }


__all__ = [
    "ACTION_KINDS",
    "ALIAS_KINDS",
    "INLINE_KINDS",
    "LENS_EXAMPLES",
    "POLICY_EXAMPLES",
    "RULE_EXAMPLES",
    "catalog",
    "lens_example",
    "model_choices",
    "rule_example",
]
