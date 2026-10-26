"""TurnSession: the one seam every turn passes through, whatever its channel.

Chat, webhooks, workflow steps and your own code all open a session, report what
happens (sources found, tools called, text produced) and ask it at three points
whether to go on: the gate, while watching, and at review. Closing it writes the
turn to the ledger and hands it to the background stages.

    session = await TurnSession.start(agent, channel="api", user=user, input=text)
    gate = await session.gate()
    if gate.blocked:
        return gate.message
    answer = await run_my_agent(gate.input)
    review = await session.review(answer)
    await session.close()
    return review.output

A session for an agent without a policy, or with oversight off, is inert: every
call returns at once and allows everything.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import secrets
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from asgiref.sync import sync_to_async
from django.utils import timezone

from django_ai_sdk.logger import get_logger
from django_ai_sdk.oversight import budget as budgets
from django_ai_sdk.oversight.answerers.pii import redact
from django_ai_sdk.oversight.conf import is_enabled, setting
from django_ai_sdk.oversight.stages import Matched, Observed, evaluate_rules, run_lenses
from django_ai_sdk.oversight.types import (
    Answer,
    Capture,
    HistoryItem,
    Outcome,
    Source,
    Stage,
    Subject,
    TextSpan,
    ToolCall,
    TurnView,
    worst,
)

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable

    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser

    from django_ai_sdk.common import ChatMessage
    from django_ai_sdk.oversight.policy import ResolvedPolicy

logger = get_logger(__name__)

_unobserved: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "django_ai_sdk_oversight_unobserved", default=False
)
_current: contextvars.ContextVar[TurnSession | None] = contextvars.ContextVar(
    "django_ai_sdk_oversight_session", default=None
)

SEVERITIES = ["", "info", "warning", "high", "critical"]


@contextlib.contextmanager
def unobserved() -> Generator[None]:
    """Run code that must not open turns of its own: judges, titles, replays."""
    token = _unobserved.set(True)
    try:
        yield
    finally:
        _unobserved.reset(token)


@contextlib.contextmanager
def active(session: TurnSession | None) -> Generator[None]:
    """Make `session` the one tool hooks and get_tools() see; tasks copy it."""
    token = _current.set(session if session is not None and session.active else None)
    try:
        yield
    finally:
        _current.reset(token)


def current_session() -> TurnSession | None:
    return _current.get()


@dataclass
class GateResult:
    """What the gate decided about the question."""

    blocked: bool = False
    message: str = ""
    # The question as the model should see it: redacted when a rule said so.
    input: str = ""
    redacted: bool = False
    # Tool names taken away; None leaves every tool, [] takes all of them.
    restrict: list[str] | None = None
    # For the client: a `data-oversight` payload, when there is anything to say.
    notice: dict[str, Any] | None = None


@dataclass
class ReviewResult:
    """What review decided about the answer."""

    blocked: bool = False
    message: str = ""
    # The answer as it may be stored and shown.
    output: str = ""
    replaced: bool = False
    notice: dict[str, Any] | None = None


@dataclass
class _Usage:
    model: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    def add(self, model: str | None, prompt: int | None, completion: int | None) -> None:
        self.model = self.model or (model or "")
        if prompt is not None:
            self.prompt_tokens = (self.prompt_tokens or 0) + prompt
        if completion is not None:
            self.completion_tokens = (self.completion_tokens or 0) + completion


@dataclass
class TurnSession:
    """One turn under a policy. Build with `start()`; inert when `active` is False."""

    active: bool = False
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    policy: ResolvedPolicy | None = None
    channel: str = ""
    view: TurnView = field(default_factory=TurnView)
    agent_fingerprint: str = ""
    thread_id: str = ""
    user_message_id: str = ""
    message_id: str = ""
    actor: str = ""
    actor_pk: str = ""
    observed: list[Observed] = field(default_factory=list)
    matched: list[Matched] = field(default_factory=list)
    restrict: list[str] | None = None
    usage: _Usage = field(default_factory=_Usage)
    created_at: Any = None
    error: str = ""
    original_input: str = ""
    original_output: str = ""
    # data-oversight payloads waiting for the stream to send them.
    notices: list[dict[str, Any]] = field(default_factory=list)
    _started: float = field(default_factory=time.perf_counter)
    _watched_chars: int = 0
    _budget: budgets.Budget | None = None
    _closed: bool = False

    # Opening.

    @classmethod
    async def start(
        cls,
        agent: Any,
        *,
        channel: str,
        user: AbstractBaseUser | AnonymousUser | None = None,
        thread_id: str | None = None,
        messages: list[ChatMessage] | None = None,
        input: str | None = None,
        policy: str | None = None,
    ) -> TurnSession:
        """A session for one turn of `agent`; inert unless a policy applies."""
        if _unobserved.get() or not is_enabled():
            return cls()
        from django_ai_sdk.oversight.policy import aassigned_policy, aget_policy

        name = policy
        if not name:
            # The console's choice for this agent wins over what its code says.
            assigned = await aassigned_policy(agent_key(agent))
            name = _policy_name(agent) if assigned is None else assigned
        if not name:
            return cls()

        resolved = await aget_policy(name)
        if resolved is None:
            return cls()
        from django_ai_sdk.oversight.privacy import pseudonym

        messages = messages or []
        if input is None:
            input = next((m.content for m in reversed(messages) if m.role == "user"), "") or ""
        history = [
            HistoryItem(role=m.role, content=m.content or "")
            for m in messages[:-1]
            if m.role in {"user", "assistant"} and m.content
        ][-12:]
        session = cls(
            active=True,
            policy=resolved,
            channel=channel,
            thread_id=str(thread_id or ""),
            actor=pseudonym(user),
            actor_pk=str(getattr(user, "pk", "") or ""),
            created_at=timezone.now(),
            original_input=input,
        )
        session.view = TurnView(
            turn_id=session.id,
            channel=channel,
            agent_id=agent_key(agent),
            agent_name=str(getattr(agent, "name", "") or type(agent).__name__),
            input=input,
            history=history,
        )
        session.agent_fingerprint = fingerprint(agent)
        session._budget = await budgets.aload()
        return session

    # Reporting.

    def set_sources(self, sources: Iterable[Any], *, seeded: Iterable[Any] = ()) -> None:
        """What retrieval found, as NumberedSource, Source or dicts."""
        if not self.active:
            return
        seeded_keys = {_source(s).key for s in seeded}
        fresh = [_source(s) for s in sources]
        for source in fresh:
            source.seeded = source.seeded or (bool(source.key) and source.key in seeded_keys)
        self.view.sources = fresh

    def add_tool_call(self, call: ToolCall | dict[str, Any]) -> None:
        if not self.active:
            return
        call = call if isinstance(call, ToolCall) else ToolCall.model_validate(call)
        for index, existing in enumerate(self.view.tool_calls):
            if call.id and existing.id == call.id:
                self.view.tool_calls[index] = existing.model_copy(
                    update={
                        k: v for k, v in call.model_dump().items() if v not in (None, "", False)
                    }
                )
                return
        self.view.tool_calls.append(call)

    def add_usage(self, model: str | None, prompt: int | None, completion: int | None) -> None:
        if self.active:
            self.usage.add(model, prompt, completion)

    def note_text(self, chunk: str) -> bool:
        """Add streamed text; True when enough has arrived for the next watch check."""
        if not self.active or not chunk:
            return False
        self.view.output += chunk
        due = len(self.view.output) - self._watched_chars >= int(setting("WATCH_EVERY_CHARS"))
        return due and self.policy is not None and bool(self.policy.active_lenses(Stage.WATCH))

    # Deciding.

    async def gate(self) -> GateResult:
        """Judge the question before anyone sees it."""
        result = GateResult(input=self.view.input)
        if not self.active:
            return result
        matched = await self._stage(
            Stage.GATE, subjects={Subject.INPUT}, deadline=setting("GATE_DEADLINE_MS")
        )
        for match in matched:
            for action in match.acted:
                if action.kind == "block" and not result.blocked:
                    result.blocked = True
                    result.message = match.rule.message or action.message
                elif action.kind == "restrict_tools":
                    names = action.names
                    if result.restrict is None:
                        result.restrict = list(names)
                    elif names and result.restrict:
                        result.restrict = sorted({*result.restrict, *names})
                    else:
                        result.restrict = []
                elif action.kind == "redact":
                    spans = self._spans_for(action, match, Subject.INPUT)
                    if spans:
                        self.view.input = redact(self.view.input, spans, action.template)
                        result.redacted = True
        result.input = self.view.input
        self.restrict = result.restrict
        result.notice = self._notice(Stage.GATE, matched, result.blocked, result.message)
        return result

    async def watch(self) -> ReviewResult:
        """Judge the answer so far, while it streams. Only enforced blocks stop it."""
        result = ReviewResult(output=self.view.output)
        if not self.active:
            return result
        self._watched_chars = len(self.view.output)
        matched = await self._stage(
            Stage.WATCH, subjects={Subject.OUTPUT, Subject.EXCHANGE}, deadline=None
        )
        return self._apply_output_actions(Stage.WATCH, matched, result)

    async def check_tool(self, call: ToolCall) -> str | None:
        """Judge one tool call before it runs; the refusal's reason when it may not."""
        if not self.active or self.policy is None:
            return None
        if self.restrict is not None and (not self.restrict or call.name in self.restrict):
            return "This tool is not available for this request."
        lenses = self.policy.active_lenses(Stage.WATCH, Subject.TOOL_CALL)
        if not lenses:
            return None
        view = self.view.model_copy(update={"tool_call": call})
        observed = await run_lenses(
            lenses, view, Stage.WATCH, deadline_ms=setting("GATE_DEADLINE_MS")
        )
        self.observed.extend(observed)
        answers = {o.lens.name: o.answer for o in observed}
        done = {m.rule.name for m in self.matched}
        # Tool rules are judged per call: only those that read a tool lens.
        tool_lenses = {lens.name for lens in lenses}
        matched = [
            m
            for m in evaluate_rules(self.policy, answers, Stage.WATCH, done=done)
            if m.rule.lenses() & tool_lenses
        ]
        self.matched.extend(matched)
        for match in matched:
            refusing = [a for a in match.acted if a.kind in {"block", "restrict_tools"}]
            if refusing:
                block = next((a for a in refusing if a.kind == "block"), None)
                return (
                    match.rule.message
                    or (block.message if block is not None else "")
                    or f"Blocked by policy ({match.rule.name})."
                )
        return None

    async def review(self, output: str | None = None) -> ReviewResult:
        """Judge the finished answer before it is stored."""
        if output is not None:
            self.view.output = output
        self.original_output = self.original_output or self.view.output
        result = ReviewResult(output=self.view.output)
        if not self.active:
            return result
        matched = await self._stage(
            Stage.REVIEW,
            subjects={Subject.OUTPUT, Subject.EXCHANGE, Subject.SOURCES},
            deadline=setting("REVIEW_DEADLINE_MS"),
        )
        return self._apply_output_actions(Stage.REVIEW, matched, result)

    def _apply_output_actions(
        self, stage: Stage, matched: list[Matched], result: ReviewResult
    ) -> ReviewResult:
        for match in matched:
            for action in match.acted:
                if action.kind == "block" and not result.blocked:
                    result.blocked = True
                    result.message = match.rule.message or action.message
                    result.output = result.message
                    result.replaced = True
                elif action.kind == "redact" and not result.blocked:
                    spans = self._spans_for(action, match, Subject.OUTPUT)
                    if spans:
                        result.output = redact(result.output, spans, action.template)
                        result.replaced = True
        if result.replaced:
            self.original_output = self.original_output or self.view.output
            self.view.output = result.output
        result.notice = self._notice(stage, matched, result.blocked, result.message, result)
        return result

    async def _stage(
        self, stage: Stage, *, subjects: set[Subject], deadline: float | None
    ) -> list[Matched]:
        assert self.policy is not None
        lenses = [lens for lens in self.policy.active_lenses(stage) if lens.subject in subjects]
        token = budgets.activate(self._budget)
        try:
            observed = await run_lenses(lenses, self.view, stage, deadline_ms=deadline)
        finally:
            budgets.deactivate(token)
        self.observed.extend(observed)
        for o in observed:
            self.view.answers[o.lens.name] = o.answer
        errored = {o.lens.name for o in observed if o.answer.error and not o.answer.queued}
        fail_closed = (setting("FAIL_MODE") or {}).get(str(stage)) == "closed"
        matched = evaluate_rules(
            self.policy,
            self.view.answers,
            stage,
            done={m.rule.name for m in self.matched},
            errored=errored,
            fail_closed=fail_closed,
        )
        self.matched.extend(matched)
        return matched

    def _spans_for(self, action: Any, match: Matched, subject: Subject) -> list[TextSpan]:
        """The spans a Redact action masks: from the lenses it names, on this subject."""
        names = set(action.lenses) or match.rule.lenses()
        spans: list[TextSpan] = []
        for o in self.observed:
            if o.lens.name in names and o.lens.subject in {subject, Subject.EXCHANGE}:
                spans.extend(o.answer.spans)
        return spans

    def _notice(
        self,
        stage: Stage,
        matched: list[Matched],
        blocked: bool,
        message: str,
        review: ReviewResult | None = None,
    ) -> dict[str, Any] | None:
        acted = [m for m in matched if m.acted]
        detail = setting("CLIENT_DETAIL") == "full"
        if not acted and not detail:
            return None
        notice: dict[str, Any] = {
            "turn_id": self.id,
            "stage": str(stage),
            "outcome": str(worst(*(m.outcome() for m in acted))),
            "rules": [m.rule.name for m in acted],
        }
        if blocked:
            notice["message"] = message
        if review is not None and review.replaced:
            notice["replace"] = review.output
        if stage is Stage.GATE and any(a.kind == "redact" for m in acted for a in m.acted):
            notice["input"] = self.view.input
        if detail:
            notice["detail"] = self._detail(stage, matched)
        return notice

    def _detail(self, stage: Stage, matched: list[Matched]) -> dict[str, Any]:
        """Every answer and match of `stage`, for a developer view in the client."""
        from django_ai_sdk.utils import serialize

        assert self.policy is not None
        return {
            "policy": self.policy.name,
            "shadow_outcome": str(worst(*(m.outcome(shadow=True) for m in matched))),
            "pending": sorted(lens.name for lens in self.policy.active_lenses(Stage.REFLECT))
            if stage is Stage.REVIEW
            else [],
            "observations": [
                {
                    "lens": o.lens.name,
                    "label": o.lens.label or o.lens.name.replace("_", " "),
                    "kind": o.lens.kind,
                    "value": serialize(o.answer.value),
                    "confidence": o.answer.confidence if o.answer.ok else None,
                    "answerer": o.answer.answerer,
                    "model": o.answer.model,
                    "error": o.answer.error,
                    "latency_ms": o.answer.latency_ms,
                }
                for o in self.observed
                if o.stage is stage
            ],
            "matches": [
                {
                    "rule": m.rule.name,
                    "mode": str(m.rule.mode),
                    "acted": [a.kind for a in m.acted],
                    "would": [a.kind for a in m.rule.then],
                }
                for m in matched
            ],
        }

    # Closing.

    @property
    def outcome(self) -> Outcome:
        return worst(*(m.outcome() for m in self.matched if m.acted))

    @property
    def shadow_outcome(self) -> Outcome:
        return worst(*(m.outcome(shadow=True) for m in self.matched))

    async def close(self, *, error: str = "") -> str | None:
        """Write the turn to the ledger and queue the background stages. Never raises.

        Returns the turn's id once written, None for an inert session or a failure.
        """
        if not self.active or self._closed:
            return None
        self._closed = True
        self.error = error
        try:
            turn_id = await sync_to_async(_persist, thread_sensitive=True)(self)
        except Exception:
            logger.exception("Oversight could not record turn {}", self.id)
            return None
        if self._budget is not None:
            try:
                await self._budget.flush()
            except Exception:
                logger.exception("Oversight could not record spend for turn {}", self.id)
        await self._notify()
        return turn_id

    async def _notify(self) -> None:
        notifications = [
            (match, action)
            for match in self.matched
            for action in match.acted
            if action.kind == "notify"
        ]
        if not notifications:
            return
        from django_ai_sdk.oversight.exporters import emit

        for match, action in notifications:
            await emit(
                {
                    "type": "oversight.decision",
                    "turn_id": self.id,
                    "rule": match.rule.name,
                    "stage": str(match.stage),
                    "severity": action.severity,
                    "outcome": str(self.outcome),
                    "agent": self.view.agent_name,
                    "channel": self.channel,
                },
                names=action.exporters,
            )


def agent_key(agent: Any) -> str:
    """The id turns and console choices know an agent by: registry id or runtime id."""
    return str(getattr(type(agent), "_agent_id", "") or getattr(agent, "agent_id", "") or "")


def code_policy(agent: Any) -> tuple[str, str]:
    """The policy the agent's code asks for, and why: "agent", "off" or "default"."""
    getter = getattr(agent, "get_oversight_policy", None)
    name = getter() if callable(getter) else getattr(agent, "oversight", None)
    if name is False:
        return "", "off"
    if isinstance(name, str) and name:
        return name, "agent"
    return str(setting("DEFAULT_POLICY") or ""), "default"


def _policy_name(agent: Any) -> str:
    getter = getattr(agent, "get_oversight_policy", None)
    name = getter() if callable(getter) else getattr(agent, "oversight", None)
    if name is False:
        return ""
    if isinstance(name, str) and name:
        return name
    return str(setting("DEFAULT_POLICY") or "")


def fingerprint(agent: Any) -> str:
    """What the agent was, compactly: a change of model, prompt or tools changes it."""
    parts: list[str] = []
    for getter in ("get_model", "get_system_prompt"):
        fn = getattr(agent, getter, None)
        try:
            parts.append(str(fn() if callable(fn) else ""))
        except Exception:
            parts.append("")
    tools = getattr(type(agent), "tools", []) or []
    parts.append(",".join(sorted(getattr(t, "__name__", str(t)) for t in tools)))
    parts.append(",".join(sorted(getattr(agent, "integrations", []) or [])))
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:16]


def _source(item: Any) -> Source:
    if isinstance(item, Source):
        return item
    if isinstance(item, dict):
        return Source.model_validate({k: v for k, v in item.items() if k in Source.model_fields})
    return Source(
        key=str(getattr(item, "key", "") or ""),
        title=str(getattr(item, "title", "") or ""),
        url=str(getattr(item, "url", "") or ""),
        content=str(getattr(item, "content", "") or ""),
        kind=str(getattr(item, "kind", "") or ""),
        doc_id=str(getattr(item, "doc_id", "") or ""),
        chunk_id=str(getattr(item, "chunk_id", "") or ""),
        memory_id=str(getattr(item, "memory_id", "") or ""),
    )


def evidence_content(session: TurnSession, capture: Capture) -> dict[str, Any] | None:
    """What of the turn is kept, at `capture`. Salted, so its hash says nothing."""
    view = session.view
    if capture is Capture.NONE:
        return None
    base: dict[str, Any] = {
        "salt": secrets.token_hex(16),
        "agent": {"id": view.agent_id, "name": view.agent_name},
    }
    if capture is Capture.METADATA:
        return {
            **base,
            "input_chars": len(session.original_input or view.input),
            "output_chars": len(session.original_output or view.output),
            "sources": [
                {"title": s.title, "url": s.url, "kind": s.kind, "seeded": s.seeded}
                for s in view.sources
            ],
            "tool_calls": [{"name": c.name, "error": c.error} for c in view.tool_calls],
        }
    full = capture is Capture.FULL
    input_text = session.original_input if full else view.input
    output_text = session.original_output if full and session.original_output else view.output
    if not full:
        input_text = _redact_subject(session, input_text, Subject.INPUT)
        output_text = _redact_subject(session, output_text, Subject.OUTPUT)
    return {
        **base,
        "input": input_text,
        "output": output_text,
        # What was actually shown or stored, when it differs from what was said.
        "shown_input": view.input if full and view.input != session.original_input else None,
        "shown_output": view.output
        if full and session.original_output and view.output != session.original_output
        else None,
        "history": [h.model_dump() for h in view.history],
        "sources": [s.model_dump() for s in view.sources],
        "tool_calls": [c.model_dump(mode="json") for c in view.tool_calls],
    }


def _redact_subject(session: TurnSession, text: str, subject: Subject) -> str:
    """Mask what any Entities lens found in `subject`, for REDACTED capture."""
    spans = [
        span
        for o in session.observed
        if o.lens.kind == "entities" and o.lens.subject is subject
        for span in o.answer.spans
    ]
    return redact(text, spans) if spans else text


def _persist(session: TurnSession) -> str:
    """Write evidence, turn, observations and decisions, chained, in one transaction."""
    from datetime import timedelta

    from django.db import transaction

    from django_ai_sdk.oversight import ledger, vault
    from django_ai_sdk.oversight.models import Decision, Evidence, Observation, ReviewItem, Turn

    assert session.policy is not None
    policy = session.policy
    now = timezone.now()
    content = evidence_content(session, policy.capture)
    scope = f"thread:{session.thread_id}" if session.thread_id else f"turn:{session.id}"
    retention = policy.retention or {}
    content_days = max(int(retention.get("security", 90)), int(retention.get("quality", 30)))

    with transaction.atomic():
        evidence = None
        evidence_hash = ""
        if content is not None:
            evidence_hash = ledger.sha256(ledger.canonical(content))
            sealed = vault.encrypt_json(scope, content)
            evidence = Evidence.objects.create(
                digest=evidence_hash,
                scope=scope,
                key=sealed[0] if sealed else None,
                ciphertext=sealed[1] if sealed else None,
                content=None if sealed else content,
                capture=str(policy.capture),
                expires_at=now + timedelta(days=content_days),
            )
        labels = {
            o.lens.name: o.answer.value
            for o in session.observed
            if o.answer.ok and o.lens.kind in {"choice", "flag", "entities", "score"}
        }
        turn = Turn.objects.create(
            id=session.id,
            created_at=session.created_at or now,
            completed_at=now,
            channel=session.channel[:100],
            agent_id=session.view.agent_id[:64],
            agent_name=session.view.agent_name[:255],
            agent_fingerprint=session.agent_fingerprint,
            policy=policy.name,
            thread_id=session.thread_id,
            user_message_id=session.user_message_id,
            message_id=session.message_id,
            actor_pseudonym=session.actor,
            actor_ref=vault.seal_text(f"actor:{session.actor}", session.actor_pk)
            if session.actor_pk
            else "",
            model=session.usage.model[:255],
            prompt_tokens=session.usage.prompt_tokens,
            completion_tokens=session.usage.completion_tokens,
            cost=_cost(session),
            latency_ms=round((time.perf_counter() - session._started) * 1000, 1),
            outcome=str(session.outcome),
            shadow_outcome=str(session.shadow_outcome),
            evidence=evidence,
            evidence_hash=evidence_hash,
            labels=labels,
            severity=_severity(session.matched),
        )
        ledger.append("turn", turn)
        for o in session.observed:
            observation = Observation.objects.create(
                turn=turn, **observation_fields(o.lens.name, o.stage, o.answer)
            )
            ledger.append("observation", observation)
        reviews: set[tuple[str, str, str]] = set()
        for o in session.observed:
            if o.answer.queued:
                reviews.add((o.answer.queued, o.lens.name, ""))
        for match in session.matched:
            decision = Decision.objects.create(
                turn=turn,
                rule=match.rule.name,
                stage=str(match.stage),
                mode=str(match.rule.mode),
                actions=[a.to_dict() for a in match.rule.then],
                acted=bool(match.acted),
                condition=match.rule.when.to_dict(),
            )
            ledger.append("decision", decision)
            for action in match.acted:
                if action.kind == "queue":
                    reviews.add((action.queue, "", match.rule.name))
        for queue, lens, rule in sorted(reviews):
            ReviewItem.objects.get_or_create(
                turn=turn,
                queue=queue,
                lens=lens,
                rule=rule,
                defaults={"priority": _priority(session, queue)},
            )
        if _needs_reflect(session):
            transaction.on_commit(lambda: _enqueue_reflect(turn.id))
    return str(turn.id)


def observation_fields(lens: str, stage: Stage | str, answer: Answer) -> dict[str, Any]:
    from django_ai_sdk.utils import serialize

    return {
        "lens": lens,
        "stage": str(stage),
        "answerer": answer.answerer[:200],
        "model": answer.model[:255],
        "value": serialize(answer.value),
        "confidence": answer.confidence if answer.ok else None,
        "rationale": answer.rationale,
        "spans": [s.model_dump() for s in answer.spans],
        "trail": [_trail(a) for a in answer.trail],
        "error": answer.error[:200],
        "queued": answer.queued[:100],
        "cost": answer.cost,
        "tokens": answer.tokens,
        "latency_ms": answer.latency_ms,
    }


def _trail(answer: Answer) -> dict[str, Any]:
    from django_ai_sdk.utils import serialize

    return {
        "answerer": answer.answerer,
        "model": answer.model,
        "value": serialize(answer.value),
        "confidence": answer.confidence,
        "error": answer.error,
        "cost": answer.cost,
        "latency_ms": answer.latency_ms,
    }


def _cost(session: TurnSession) -> float | None:
    from django_ai_sdk.pricing import cost_for

    return cost_for(
        session.usage.model, session.usage.prompt_tokens, session.usage.completion_tokens
    )


def _severity(matched: list[Matched]) -> str:
    levels = [
        getattr(action, "severity", "")
        for match in matched
        for action in match.acted
        if action.kind in {"alert", "notify"}
    ]
    if any(m.acted and m.outcome() is Outcome.BLOCKED for m in matched):
        levels.append("high")
    return max(levels, key=lambda s: SEVERITIES.index(s) if s in SEVERITIES else 0, default="")


def _priority(session: TurnSession, queue: str) -> int:
    for match in session.matched:
        for action in match.acted:
            if action.kind == "queue" and action.queue == queue:
                return action.priority
    return 0


def _needs_reflect(session: TurnSession) -> bool:
    assert session.policy is not None
    escalations = any(a.kind == "escalate" for m in session.matched for a in m.acted)
    timed_out = any(o.answer.error == "deadline" for o in session.observed)
    return escalations or timed_out or session.policy.has_stage(Stage.REFLECT)


def _enqueue_reflect(turn_id: Any) -> None:
    try:
        from django_ai_sdk.oversight.tasks import reflect_turn

        reflect_turn.enqueue(str(turn_id))
    except Exception:
        logger.exception(
            "Could not queue reflection for turn {}; is a task backend configured?", turn_id
        )


__all__ = [
    "GateResult",
    "ReviewResult",
    "TurnSession",
    "active",
    "agent_key",
    "code_policy",
    "current_session",
    "fingerprint",
    "unobserved",
]
