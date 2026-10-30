"""What the console offers to build with: answerer kinds, models, and examples.

Everything here is data the editor renders. An answerer kind lists its
parameters, so the console can show a form for it and save the result as a
config dict (`{"class": "Keywords", "keywords": {...}}`) that `resolve` builds.
These lists are also what the console may write: `console_config` refuses any
other class or parameter, so the editor and the validator cannot drift apart.
The examples live in `examples`.

Hosts add model choices with AI_SDK_OVERSIGHT["MODEL_CHOICES"]: a list of
`{"id", "label", "group", "kinds"}` or the dotted path of a function returning one.
"""

from __future__ import annotations

from typing import Any

from django.utils.module_loading import import_string

from django_ai_sdk.logger import get_logger
from django_ai_sdk.oversight.answerers import pii
from django_ai_sdk.oversight.examples import LENS_EXAMPLES, POLICY_EXAMPLES, RULE_EXAMPLES
from django_ai_sdk.oversight.rules import ACTIONS
from django_ai_sdk.oversight.types import Severity

logger = get_logger(__name__)

# Parameter types the editor knows how to render.
#   text, number, bool, choice (from `choices`), multi (several of `choices`),
#   lines (one string per line), labelled_lines ({label: [lines]}), model
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
        "metered": True,
        "external": "model gateway",
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
        "label": "TypeSafe",
        "summary": "A hosted, paid classification service: answers yes/no, label and score "
        "lenses, and rates each source for a lens about which sources an answer is taken "
        "from. Needs django-ai-sdk[typesafe] and TYPESAFE_API_KEY.",
        "metered": False,
        "external": "TypeSafe",
        "params": [
            _param("model", "model", "Model", "Empty uses TYPESAFE_MODEL.", kind="TypeSafe"),
        ],
    },
    {
        "class": "Human",
        "label": "A person",
        "summary": "The turn goes to a review queue; someone answers later.",
        "metered": False,
        "external": "",
        "params": [
            _param("queue", "text", "Queue", default="default"),
            _param("priority", "number", "Priority", "Higher is sooner.", default=0),
        ],
    },
]

PII_KINDS = list(pii.KINDS)

# Free, local answerers a lens can use inline, without an alias.
INLINE_KINDS: list[dict[str, Any]] = [
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
        "summary": "Email, IBAN (mod-97), card numbers (Luhn), international phone numbers, "
        "IP addresses. Dutch BSN (11-check), phone numbers and postcodes when picked.",
        "fits": ["entities", "flag"],
        "params": [
            _param(
                "kinds",
                "multi",
                "Kinds",
                f"None picked: {', '.join(pii.DEFAULT_KINDS)}.",
                choices=PII_KINDS,
            )
        ],
    },
    {
        "class": "Secrets",
        "label": "Secrets",
        "summary": "API keys (sk-…, Stripe, Google, GitHub, AWS, Slack), JWTs, private keys, "
        "passwords.",
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
]
# Every inline kind runs here: nothing is charged and no turn content leaves.
for _kind in INLINE_KINDS:
    _kind.update(metered=False, external="")

LENS_KINDS = [
    {"kind": "flag", "label": "Yes / no", "summary": '"Does it…?" A rule tests is yes or is no.'},
    {"kind": "choice", "label": "One label", "summary": "Classify: exactly one of your labels."},
    {"kind": "score", "label": "Score 0–1", "summary": '"How much…?" A rule tests above or below.'},
    {
        "kind": "entities",
        "label": "Which things",
        "summary": "Which of your labels occur, and where, so a rule can mask them.",
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


def _stage(value: str, label: str, runs: str, can: str) -> dict[str, str]:
    return {"value": value, "label": label, "runs": runs, "can": can}


# The one wording of the stages: what a lens there looks at, and what a rule there can do.
STAGE_INFO = [
    _stage(
        "gate",
        "Before the model",
        "On the question, before the model sees it.",
        "Block it, mask parts of it or take tools away.",
    ),
    _stage(
        "watch",
        "While it works",
        "On each tool call before it runs, and on the answer as it streams.",
        "Refuse the tool call or stop the answer.",
    ),
    _stage(
        "finish",
        "Before the user sees it",
        "On the finished answer; the stream is fixed when it ends.",
        "Mask or replace the answer, or block it.",
    ),
    _stage(
        "reflect",
        "Afterwards",
        "In the background, seconds after the reply.",
        "Send the turn to review, alert, or mask the stored copy.",
    ),
]

OPS = [
    {"op": "is_true", "label": "is yes", "kinds": ["flag"], "value": False},
    {"op": "is_false", "label": "is no", "kinds": ["flag"], "value": False},
    {"op": "eq", "label": "is", "kinds": ["choice", "text"], "value": "label"},
    {"op": "ne", "label": "is not", "kinds": ["choice", "text"], "value": "label"},
    {"op": "in", "label": "is one of", "kinds": ["choice", "entities"], "value": "labels"},
    {"op": "contains", "label": "finds", "kinds": ["entities", "text"], "value": "label"},
    {"op": "any", "label": "finds something", "kinds": ["entities", "text"], "value": False},
    {"op": "ge", "label": "scores at least", "kinds": ["score"], "value": "number"},
    {"op": "gt", "label": "scores above", "kinds": ["score"], "value": "number"},
    {"op": "le", "label": "scores at most", "kinds": ["score"], "value": "number"},
    {"op": "lt", "label": "scores below", "kinds": ["score"], "value": "number"},
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

ACTION_KINDS: list[dict[str, Any]] = [
    {
        "kind": "block",
        "label": "Block",
        "summary": "Stop the turn and show a message.",
        "params": [_param("message", "text", "Message shown", default="")],
    },
    {
        "kind": "mask",
        "label": "Mask",
        "summary": "Mask what the lenses found, as [IBAN].",
        "params": [_param("template", "text", "Mask", default="[{label}]")],
    },
    {
        "kind": "restrict_tools",
        "label": "Take tools away",
        "summary": "For this turn: the named tools, or all when empty.",
        "params": [_param("names", "lines", "Tool names", "Empty: every tool.")],
    },
    {
        "kind": "queue",
        "label": "Send to a person",
        "summary": "Put the turn in a review queue.",
        "params": [
            _param("queue", "text", "Queue", default="default"),
            _param("priority", "number", "Priority", default=0),
        ],
    },
    {
        "kind": "notify",
        "label": "Send an alert",
        "summary": "Message someone outside the console: Telegram, Slack, a webhook.",
        "params": [
            _param(
                "severity",
                "choice",
                "Severity",
                default="warning",
                choices=[str(s) for s in Severity],
            )
        ],
    },
    {
        "kind": "alert",
        "label": "Flag it",
        "summary": "Mark the turn in the console, at a severity.",
        "params": [
            _param(
                "severity",
                "choice",
                "Severity",
                default="warning",
                choices=[str(s) for s in Severity],
            )
        ],
    },
]
# Where each action can still change the turn.
for _action in ACTION_KINDS:
    _action["stages"] = [str(s) for s in ACTIONS[_action["kind"]].effective]


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


def lens_example(name: str) -> dict[str, Any] | None:
    return next((dict(e["lens"]) for e in LENS_EXAMPLES if e["lens"]["name"] == name), None)


def rule_example(name: str) -> dict[str, Any] | None:
    return next((dict(e["rule"]) for e in RULE_EXAMPLES if e["rule"]["name"] == name), None)


def console_shape(config: Any) -> tuple[Any, bool]:
    """An alias's config as the console shows it, and whether the console can save it back.

    Only the class and that kind's catalog parameters are shown: a settings config
    may hold headers or tokens. Anything the console could not write is read-only.
    """
    from django_ai_sdk.oversight.answerers.base import SDK_ANSWERERS

    if isinstance(config, str):
        return config, True
    if not isinstance(config, dict):
        return repr(config), False
    path = config.get("class")
    short = path.__name__ if isinstance(path, type) else str(path or "")
    short = short.removeprefix(SDK_ANSWERERS)
    kind = next((k for k in ALIAS_KINDS if k["class"] == short), None)
    params = {p["name"] for p in kind["params"]} if kind else set()
    shown = {"class": short, **{k: v for k, v in config.items() if k in params or k == "name"}}
    editable = kind is not None and set(config) <= {"class", "name", *params}
    return shown, editable


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
        shown, editable = console_shape(config)
        aliases.append(
            {
                "alias": alias,
                "source": alias_source(alias),
                "config": shown,
                "editable": editable,
                "class": described.get("class", ""),
                "model": described.get("model", ""),
                "metered": bool(described.get("metered", True)),
                "external": str(described.get("external") or ""),
                "error": error,
            }
        )
    aliases.append(
        {
            "alias": "human",
            "source": "builtin",
            "config": {"class": "Human"},
            "editable": False,
            "class": "Human",
            "model": "",
            "metered": False,
            "external": "",
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
    "catalog",
    "console_shape",
    "lens_example",
    "model_choices",
    "rule_example",
]
