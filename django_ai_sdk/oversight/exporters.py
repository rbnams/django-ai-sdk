"""Exporters: where decisions and checkpoints go besides the ledger.

    AI_SDK_OVERSIGHT = {"EXPORTERS": [
        {"class": "django_ai_sdk.oversight.exporters.Webhook", "name": "siem",
         "url": "https://siem.example/hook", "secret": env("SIEM_SECRET")},
        {"class": "django_ai_sdk.oversight.exporters.Webhook", "name": "anchor",
         "url": "https://witness.example/checkpoints", "anchor": True},
    ]}

An exporter with `anchor=True` also receives every checkpoint: post it somewhere
the deployment does not control, and a rewrite of the history before it shows to
anyone who compares the published root with the ledger.
Exporters never fail a turn; errors are logged. A turn's notifications are sent
from the task worker (`tasks.export_events`), so a slow receiver never holds a chat
stream open.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
import uuid
from typing import Any

from django.utils.module_loading import import_string

from django_ai_sdk.logger import get_logger

logger = get_logger(__name__)


class Exporter:
    """Subclass and implement `export`.

    `export` returns a reference to what the receiver stored (or None). It may
    raise: `emit` logs the error and goes on, so a receiver never fails a turn.
    """

    name: str = ""
    # Also receives checkpoints, and its reply's reference is stored as the anchor.
    anchor: bool = False

    def __init__(self, *, name: str = "", anchor: bool = False) -> None:
        self.name = name or type(self).__name__.lower()
        self.anchor = anchor

    async def export(self, event: dict[str, Any]) -> str | None:
        """Send one event; a reference (url, id) when the receiver returns one."""
        raise NotImplementedError


class Log(Exporter):
    """Write events to the SDK log, at WARNING: for log shipping into a SIEM."""

    async def export(self, event: dict[str, Any]) -> str | None:
        logger.warning("oversight {}", json.dumps(event, default=str, sort_keys=True))
        return None


class Webhook(Exporter):
    """POST events as JSON, signed the Standard Webhooks way when `secret` is set.

    webhook-signature is "v1," + base64 HMAC-SHA256 of "{webhook-id}.{webhook-timestamp}.{body}",
    so a receiver can refuse replays as well as forgeries. A "whsec_" secret is base64.
    """

    def __init__(self, url: str, *, secret: str = "", timeout: float = 5.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.url = url
        self.secret = secret
        self.timeout = timeout

    def _request(self, event: dict[str, Any]) -> tuple[bytes, dict[str, str]]:
        body = json.dumps(event, default=str, sort_keys=True).encode()
        headers = {"Content-Type": "application/json"}
        if self.secret:
            msg_id, timestamp = uuid.uuid4().hex, str(int(time.time()))
            signed = f"{msg_id}.{timestamp}.".encode() + body
            digest = hmac.new(_key(self.secret), signed, hashlib.sha256).digest()
            headers["webhook-id"] = msg_id
            headers["webhook-timestamp"] = timestamp
            headers["webhook-signature"] = "v1," + base64.b64encode(digest).decode()
        return body, headers

    async def export(self, event: dict[str, Any]) -> str | None:
        import httpx

        body, headers = self._request(event)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self.url, content=body, headers=headers)
            response.raise_for_status()
        return response.headers.get("Location") or f"{self.url}#{response.status_code}"


def _key(secret: str) -> bytes:
    if secret.startswith("whsec_"):
        return base64.b64decode(secret.removeprefix("whsec_"))
    return secret.encode()


def configured(names: list[str] | None = None, *, anchors: bool = False) -> list[Exporter]:
    from django_ai_sdk.oversight.conf import setting

    exporters: list[Exporter] = []
    for spec in setting("EXPORTERS") or []:
        try:
            exporter = spec if isinstance(spec, Exporter) else _build(spec)
        except Exception:
            logger.exception("Oversight exporter {!r} could not be built", spec)
            continue
        if anchors and not exporter.anchor:
            continue
        if names and exporter.name not in names:
            continue
        exporters.append(exporter)
    return exporters


def _build(spec: str | dict[str, Any]) -> Exporter:
    if isinstance(spec, str):
        return import_string(spec)()
    config = dict(spec)
    cls = import_string(config.pop("class"))
    return cls(**config)


async def emit(
    event: dict[str, Any], *, names: list[str] | None = None, anchors: bool = False
) -> str:
    """Send `event` to the exporters; the references of anchoring ones, joined."""
    refs: list[str] = []
    for exporter in configured(names, anchors=anchors):
        try:
            ref = await exporter.export(event)
        except Exception:
            logger.exception("Oversight exporter {} failed", exporter.name)
            continue
        if ref and exporter.anchor:
            refs.append(f"{exporter.name}:{ref}")
    return " ".join(refs)


__all__ = ["Exporter", "Log", "Webhook", "configured", "emit"]
