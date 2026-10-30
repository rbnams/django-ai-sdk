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
from django_ai_sdk.oversight.answerers.base import SDK_ANSWERERS, param
from django_ai_sdk.oversight.examples import LENS_EXAMPLES, POLICY_EXAMPLES, RULE_EXAMPLES
from django_ai_sdk.oversight.rules import ACTIONS
from django_ai_sdk.oversight.types import Severity

logger = get_logger(__name__)

# Parameter types the editor knows how to render.
#   text, number, bool, choice (from `choices`), multi (several of `choices`),
#   lines (one string per line), labelled_lines ({label: [lines]}), model
P = dict[str, Any]


_param = param


# The SDK's answerers an alias can name, in the order the console lists them. Each
# says how it is offered (`Answerer.console`); one whose package is missing is not.
SDK_KINDS = ("LLMJudge", "LocalClassifier", "TypeSafe", "Human")

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


def alias_kinds(*, installed_only: bool = False) -> list[dict[str, Any]]:
    """The answerers the console may name: ANSWERER_KINDS, or else the SDK's (SDK_KINDS).

    Each kind is its class's `console` form, with the class (short for the SDK's own,
    else its dotted path), whether it is metered and who receives turn content. A
    judge's server is one of ENDPOINTS. `installed_only`: what the editor offers, so
    a class whose package is missing is not offered, though an alias stored with it
    still loads (and abstains, ai_sdk.oversight.W005).
    """
    from django.utils.module_loading import import_string

    from django_ai_sdk.oversight.answerers.base import Answerer
    from django_ai_sdk.oversight.conf import setting

    configured = setting("ANSWERER_KINDS")
    names = SDK_KINDS if configured is None else configured
    # A name without a dot is one of the SDK's answerers.
    paths = [name if "." in name else f"{SDK_ANSWERERS}{name}" for name in names]
    endpoints = ["", *sorted(setting("ENDPOINTS"))]
    kinds = []
    for path in paths:
        try:
            cls = import_string(path)
        except ImportError:
            logger.warning("ANSWERER_KINDS names {!r}, which does not import", path)
            continue
        if not (isinstance(cls, type) and issubclass(cls, Answerer)) or not cls.console:
            continue
        if installed_only and not cls.installed():
            continue
        kinds.append(
            {
                "class": path.removeprefix(SDK_ANSWERERS),
                "metered": cls.metered,
                "external": cls.external,
                **cls.console,
                "params": [
                    {**p, "choices": endpoints} if p["name"] == "endpoint" else p
                    for p in cls.console.get("params", [])
                ],
            }
        )
    return kinds


# Words in a model id that say it is not a chat model, so it cannot judge.
NOT_CHAT = (
    "embed", "rerank", "moderation", "whisper", "tts", "transcribe", "ocr", "image",
    "dall-e", "flux", "minilm", "bge-", "e5-", "retriever",
)  # fmt: skip


# Seconds a model list is reused: long when every server answered, short when one did not.
MODELS_TTL = 600.0
MODELS_RETRY_TTL = 60.0
_models: tuple[float, float, list[dict[str, Any]]] | None = None


def endpoint_models(timeout: float = 8.0) -> list[dict[str, Any]]:
    """The chat models of the default gateway and of every ENDPOINTS server, from /models.

    For MODEL_CHOICES: "django_ai_sdk.oversight.catalog.endpoint_models". Any
    OpenAI-compatible server lists its models there (Ollama, vLLM, LM Studio and most
    gateways). A server that does not answer is left out and logged, and the list is
    asked again after MODELS_RETRY_TTL seconds instead of MODELS_TTL.
    """
    import time

    global _models
    if _models and time.monotonic() - _models[0] < _models[1]:
        return _models[2]
    choices, complete = _list_models(timeout)
    _models = (time.monotonic(), MODELS_TTL if complete else MODELS_RETRY_TTL, choices)
    return choices


def _list_models(timeout: float) -> tuple[list[dict[str, Any]], bool]:
    import httpx

    from django_ai_sdk.oversight.answerers.llm import _key
    from django_ai_sdk.oversight.conf import setting
    from django_ai_sdk.utils import resolve_setting

    servers = []
    if key := _key("OPENAI_API_KEY"):
        url = resolve_setting("OPENAI_API_URL") or "https://api.openai.com/v1"
        servers.append(("", "Default", url, key))
    for name, config in sorted(setting("ENDPOINTS").items()):
        if isinstance(config, dict) and config.get("base_url"):
            key_env = str(config.get("api_key_env") or "")
            label = str(config.get("label") or name)
            servers.append((name, label, str(config["base_url"]), _key(key_env) if key_env else ""))
    choices: list[dict[str, Any]] = []
    complete = True
    for endpoint, group, url, key in servers:
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        try:
            response = httpx.get(f"{url.rstrip('/')}/models", headers=headers, timeout=timeout)
            response.raise_for_status()
            ids = sorted(m["id"] for m in response.json().get("data", []) if m.get("id"))
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            logger.warning("Could not list the models of {}: {}", group, type(exc).__name__)
            complete = False
            continue
        choices += [
            {
                "id": model_id,
                "label": model_id.split("/")[-1],
                "group": group,
                "kinds": ["LLMJudge"],
                "endpoint": endpoint,
            }
            for model_id in ids
            if not any(word in model_id.lower() for word in NOT_CHAT)
        ]
    return choices, complete


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
            "endpoint": str(choice.get("endpoint") or ""),
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
    if isinstance(config, str):
        return config, True
    if not isinstance(config, dict):
        return repr(config), False
    path = config.get("class")
    short = path.__name__ if isinstance(path, type) else str(path or "")
    short = short.removeprefix(SDK_ANSWERERS)
    kind = next((k for k in alias_kinds() if k["class"] == short), None)
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
        "alias_kinds": alias_kinds(installed_only=True),
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
    "INLINE_KINDS",
    "NOT_CHAT",
    "SDK_KINDS",
    "alias_kinds",
    "catalog",
    "console_shape",
    "endpoint_models",
    "lens_example",
    "model_choices",
    "rule_example",
]
