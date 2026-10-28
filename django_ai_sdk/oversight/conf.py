"""AI_SDK_OVERSIGHT, resolved. One dict, read here only, every key defaulted.

AI_SDK_OVERSIGHT = {
    "DEFAULT_POLICY": "default",
    "ANSWERERS": {
        # Yes/no, label and score lenses.
        "jev": {"class": "django_ai_sdk.oversight.answerers.TypeSafe"},
        # A chat model for text lenses and second opinions, and one for escalations.
        "judge": {"class": "django_ai_sdk.oversight.answerers.LLMJudge",
                  "model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"},
        "strong": {"class": "django_ai_sdk.oversight.answerers.LLMJudge",
                   "model": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"},
    },
    "BUDGET_PER_DAY": 5.0,
}
"""

from __future__ import annotations

from typing import Any

from django.apps import apps
from django.conf import settings

from django_ai_sdk.oversight.types import Capture
from django_ai_sdk.utils import resolve_setting

APP_LABEL = "django_ai_sdk_oversight"

DEFAULTS: dict[str, Any] = {
    # The kill switch. Off leaves every agent unwatched, whatever its policy says.
    "ENABLED": True,
    # The policy an agent without `oversight` gets. Empty: such agents are unwatched.
    "DEFAULT_POLICY": "",
    # The most content the ledger keeps; a policy may keep less, never more.
    "CAPTURE": Capture.FULL,
    # "live" streams the answer as it comes; "hold" shows it once review passed.
    "STREAM_MODE": "live",
    # Per policy, the same tuning the console's overlay allows, kept in settings:
    # {"support": {"rules": {"block_injection": {"mode": "enforce"}}, "capture": "redacted"}}
    "POLICIES": {},
    # Named answerers, so a lens says "judge" and settings say which model that is.
    "ANSWERERS": {},
    # Spend cap for model answerers across the deployment, per UTC day. None: no cap.
    "BUDGET_PER_DAY": None,
    "CURRENCY": "EUR",
    # Milliseconds the gate may add to a reply. Answerers still running are abandoned.
    # Room for Jev (~0.3 s plus network) and code, not for a model judge.
    "GATE_DEADLINE_MS": 1000,
    "REVIEW_DEADLINE_MS": 2_000,
    # A question longer than this is refused at the gate, unread: past it, what the
    # lenses read is only part of it. None: no limit.
    "MAX_INPUT_CHARS": 100_000,
    # What the chat client is told: "actions" only when a rule changed the turn;
    # "full" every stage's answers and matches too, for a developer view; "staff"
    # that view only for users who may read oversight (it shows what a rule tests).
    "CLIENT_DETAIL": "actions",
    # Characters between watch checks while an answer streams.
    "WATCH_EVERY_CHARS": 400,
    # Milliseconds a watch check may pause the stream. Lenses still running answer
    # "deadline" and are asked again in reflect.
    "WATCH_DEADLINE_MS": 300,
    # What an answerer that fails or times out means at each stage. "open" lets the
    # turn through; "closed" treats an enforced rule that blocks, redacts or takes
    # tools away as matched when a lens it reads failed. Rules that only alert,
    # queue or notify never fire on a failure, nor do shadow rules.
    "FAIL_MODE": {"gate": "open", "watch": "open", "review": "open"},
    # Days a turn's content (evidence) and its metadata (the turn row, its answers
    # and decisions) are kept. A policy may set either for its own turns.
    "RETENTION": {"content": 90, "metadata": 730},
    # HMAC secret for actor pseudonyms. Defaults to one derived from SECRET_KEY.
    "PSEUDONYM_SECRET": "",
    # base64 32-byte key that wraps per-thread content keys. Empty stores content
    # in the clear (the oversight.W004 check says so).
    "MASTER_KEY": "",
    # base64 Ed25519 private key (32 bytes) for signed checkpoints, and the key's
    # name in the signature. Usually left unset: the name is then the log origin.
    "SIGNING_KEY": "",
    "SIGNING_NAME": "",
    # The checkpoint's origin line; defaults to the first ALLOWED_HOSTS entry.
    "LOG_ORIGIN": "",
    # Dotted paths or instances of Exporter, sent every decision that notifies.
    "EXPORTERS": [],
    # Models the console offers for judges: a list of {"id", "label", "group", "kinds"},
    # or the dotted path of a function that returns one (say, a gateway's /models).
    "MODEL_CHOICES": [],
    # Seconds a resolved policy (code + database overlay) is reused per process.
    "POLICY_CACHE_SECONDS": 10,
}


def oversight_settings() -> dict[str, Any]:
    """DEFAULTS overlaid with the project's AI_SDK_OVERSIGHT, one level deep for dicts."""
    configured = resolve_setting("AI_SDK_OVERSIGHT", {}) or {}
    merged = dict(DEFAULTS)
    for key, value in configured.items():
        if isinstance(value, dict) and isinstance(DEFAULTS.get(key), dict):
            merged[key] = {**DEFAULTS[key], **value}
        else:
            merged[key] = value
    return merged


def setting(name: str) -> Any:
    return oversight_settings()[name]


def is_installed() -> bool:
    return apps.is_installed("django_ai_sdk.oversight")


def is_enabled() -> bool:
    """The app is installed and the kill switch is on."""
    return is_installed() and bool(setting("ENABLED"))


def log_origin() -> str:
    origin = setting("LOG_ORIGIN")
    if origin:
        return origin
    hosts = [h for h in getattr(settings, "ALLOWED_HOSTS", []) if h not in {"*", ""}]
    return f"{hosts[0] if hosts else 'localhost'}/oversight"


def signing_name() -> str:
    """The name checkpoints are signed under, which a verifier key must carry too."""
    return setting("SIGNING_NAME") or log_origin()


__all__ = [
    "APP_LABEL",
    "DEFAULTS",
    "is_enabled",
    "is_installed",
    "log_origin",
    "oversight_settings",
    "setting",
    "signing_name",
]
