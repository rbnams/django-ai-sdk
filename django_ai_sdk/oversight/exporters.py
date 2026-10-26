"""Exporters: where decisions and checkpoints go besides the ledger.

    AI_SDK_OVERSIGHT = {"EXPORTERS": [
        {"class": "django_ai_sdk.oversight.exporters.Webhook", "name": "siem",
         "url": "https://siem.example/hook", "secret": env("SIEM_SECRET")},
        {"class": "django_ai_sdk.oversight.exporters.OpenTelemetry"},
        {"class": "django_ai_sdk.oversight.exporters.Webhook", "name": "anchor",
         "url": "https://witness.example/checkpoints", "anchor": True},
    ]}

An exporter with `anchor=True` also receives every checkpoint: post it somewhere
the deployment does not control and history before it can no longer be rewritten.
Exporters never fail a turn; errors are logged.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any

from django.utils.module_loading import import_string

from django_ai_sdk.logger import get_logger

logger = get_logger(__name__)


class Exporter:
    name: str = ""
    # Also receives checkpoints, and its reply's reference is stored as the anchor.
    anchor: bool = False

    def __init__(self, *, name: str = "", anchor: bool = False) -> None:
        self.name = name or type(self).__name__.lower()
        self.anchor = anchor

    async def export(self, event: dict[str, Any]) -> str | None:
        """Send one event; a reference (url, id) when the receiver returns one."""
        raise NotImplementedError

    def export_sync(self, event: dict[str, Any]) -> str | None:
        from asgiref.sync import async_to_sync

        return async_to_sync(self.export)(event)


class Log(Exporter):
    """Write events to the SDK log, at WARNING: for log shipping into a SIEM."""

    async def export(self, event: dict[str, Any]) -> str | None:
        logger.warning("oversight {}", json.dumps(event, default=str, sort_keys=True))
        return None


class Webhook(Exporter):
    """POST events as JSON, signed with HMAC-SHA256 in X-Oversight-Signature."""

    def __init__(self, url: str, *, secret: str = "", timeout: float = 5.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.url = url
        self.secret = secret
        self.timeout = timeout

    def _request(self, event: dict[str, Any]) -> tuple[bytes, dict[str, str]]:
        body = json.dumps(event, default=str, sort_keys=True).encode()
        headers = {"Content-Type": "application/json"}
        if self.secret:
            digest = hmac.new(self.secret.encode(), body, hashlib.sha256).hexdigest()
            headers["X-Oversight-Signature"] = f"sha256={digest}"
        return body, headers

    async def export(self, event: dict[str, Any]) -> str | None:
        import httpx

        body, headers = self._request(event)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self.url, content=body, headers=headers)
            response.raise_for_status()
        return response.headers.get("Location") or f"{self.url}#{response.status_code}"

    def export_sync(self, event: dict[str, Any]) -> str | None:
        import httpx

        body, headers = self._request(event)
        response = httpx.post(self.url, content=body, headers=headers, timeout=self.timeout)
        response.raise_for_status()
        return response.headers.get("Location") or f"{self.url}#{response.status_code}"


class OpenTelemetry(Exporter):
    """Emit decisions as `gen_ai.evaluation.result` span events (OTel GenAI semconv).

    Needs opentelemetry-api; with no SDK configured the API is a no-op.
    """

    async def export(self, event: dict[str, Any]) -> str | None:
        try:
            from opentelemetry import trace
        except ImportError:
            logger.warning("OpenTelemetry exporter configured without opentelemetry-api")
            return None
        tracer = trace.get_tracer("django_ai_sdk.oversight")
        with tracer.start_as_current_span("oversight.decision") as span:
            span.set_attribute("openinference.span.kind", "GUARDRAIL")
            span.add_event(
                "gen_ai.evaluation.result",
                {
                    "gen_ai.evaluation.name": str(event.get("rule", "")),
                    "gen_ai.evaluation.explanation": str(event.get("outcome", "")),
                    "gen_ai.evaluation.score.label": str(event.get("severity", "")),
                    "oversight.turn_id": str(event.get("turn_id", "")),
                    "oversight.stage": str(event.get("stage", "")),
                },
            )
        return None


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


async def emit(event: dict[str, Any], *, names: list[str] | None = None) -> None:
    for exporter in configured(names):
        try:
            await exporter.export(event)
        except Exception:
            logger.exception("Oversight exporter {} failed", exporter.name)


def emit_sync(
    event: dict[str, Any], *, names: list[str] | None = None, anchors: bool = False
) -> str:
    """Send `event` synchronously; the references of anchoring exporters, joined."""
    refs: list[str] = []
    for exporter in configured(names, anchors=anchors):
        try:
            ref = exporter.export_sync(event)
        except Exception:
            logger.exception("Oversight exporter {} failed", exporter.name)
            continue
        if ref and exporter.anchor:
            refs.append(f"{exporter.name}:{ref}")
    return " ".join(refs)


__all__ = ["Exporter", "Log", "OpenTelemetry", "Webhook", "configured", "emit", "emit_sync"]
