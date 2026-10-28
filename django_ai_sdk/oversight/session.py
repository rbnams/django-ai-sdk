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

import asyncio
import contextlib
import contextvars
import hashlib
import re
import secrets
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from asgiref.sync import sync_to_async
from django.utils import timezone

from django_ai_sdk.logger import get_logger
from django_ai_sdk.oversight import budget as budgets
from django_ai_sdk.oversight.answerers.pii import redact_all
from django_ai_sdk.oversight.conf import is_enabled, setting
from django_ai_sdk.oversight.rules import Block, Redact
from django_ai_sdk.oversight.stages import (
    STAGE_ORDER,
    Matched,
    Observed,
    asks_again,
    evaluate_rules,
    run_lenses,
)
from django_ai_sdk.oversight.types import (
    Answer,
    Capture,
    HistoryItem,
    Mode,
    Outcome,
    Severity,
    Source,
    Stage,
    Subject,
    TextSpan,
    ToolCall,
    TurnView,
    most_severe,
    worst,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Generator, Iterable

    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser

    from django_ai_sdk.agents.base import Agent
    from django_ai_sdk.common import ChatMessage
    from django_ai_sdk.oversight.lenses import Lens
    from django_ai_sdk.oversight.policy import ResolvedPolicy

logger = get_logger(__name__)

TOO_LONG = "This message is too long to check. Shorten it and try again."

_unobserved: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "django_ai_sdk_oversight_unobserved", default=False
)
_current: contextvars.ContextVar[TurnSession | None] = contextvars.ContextVar(
    "django_ai_sdk_oversight_session", default=None
)


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
    # Refused at the gate unread, for being longer than MAX_INPUT_CHARS.
    too_long: bool = False
    original_input: str = ""
    original_output: str = ""
    # data-oversight payloads waiting for the stream to send them.
    notices: list[dict[str, Any]] = field(default_factory=list)
    # Set by a streaming adapter: refused tool calls' stub results, so the chat
    # shows them refused at once instead of still running.
    on_tool_results: Callable[[list[Any]], Awaitable[None]] | None = None
    _started: float = field(default_factory=time.perf_counter)
    _watched_chars: int = 0
    _budget: budgets.Budget | None = None
    # Whether the person chatting may see oversight detail (CLIENT_DETAIL "staff").
    _staff_view: bool = False
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

        try:
            name = policy
            if not name:
                # The console's choice for this agent wins over what its code says.
                assigned = await aassigned_policy(agent_key(agent))
                name = code_policy(agent)[0] if assigned is None else assigned
            resolved = await aget_policy(name) if name else None
            budget = await budgets.aload() if resolved is not None else None
            staff_view = resolved is not None and await _staff_view(user)
        except Exception:
            # Oversight must never take chat down: a broken policy leaves the turn unwatched.
            logger.exception("Oversight could not open a turn for {!r}", agent_key(agent))
            return cls()
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
        session._budget = budget
        session._staff_view = staff_view
        return session

    # Reporting.

    def set_sources(self, sources: Iterable[Any], *, seeded: Iterable[Any] = ()) -> None:
        """What retrieval found, as NumberedSource, Source or dicts.

        `seeded` names those carried over from earlier turns: sources, or their keys
        (a SourceRegistry keeps keys).
        """
        if not self.active:
            return
        seeded_keys = {s if isinstance(s, str) else _source(s).key for s in seeded}
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

    def unwatched(self) -> bool:
        """Whether answer text arrived since the last watch check, with lenses to judge it."""
        if not self.active or self.policy is None or len(self.view.output) <= self._watched_chars:
            return False
        return any(
            lens.subject in {Subject.OUTPUT, Subject.EXCHANGE}
            for lens in self.policy.active_lenses(Stage.WATCH)
        )

    # Deciding.

    async def gate(self) -> GateResult:
        """Judge the question before anyone sees it."""
        result = GateResult(input=self.view.input)
        if not self.active:
            return result
        limit = setting("MAX_INPUT_CHARS")
        if limit and len(self.view.input) > limit and self.policy.active_lenses(Stage.GATE):
            self.too_long = True
            result.blocked = True
            result.message = TOO_LONG
            result.notice = {
                "turn_id": self.id,
                "stage": str(Stage.GATE),
                "outcome": str(Outcome.BLOCKED),
                "rules": [],
                "message": TOO_LONG,
            }
            return result
        try:
            return await self._gate(result)
        except Exception:
            # Like watch and review: a broken lens or rule follows the gate's fail mode.
            logger.exception("Oversight gate failed for turn {}", self.id)
            self.error = "gate failed"
            if _fail_closed(Stage.GATE):
                result.blocked = True
                result.message = Block.DEFAULT_MESSAGE
            return result

    async def _gate(self, result: GateResult) -> GateResult:
        matched = await self._stage(
            Stage.GATE, subjects={Subject.INPUT}, deadline=setting("GATE_DEADLINE_MS")
        )
        self.matched.extend(matched)
        redactions: list[tuple[list[TextSpan], str]] = []
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
                        redactions.append((spans, action.template))
        if redactions:
            self.view.input = redact_all(self.view.input, redactions)
            result.redacted = True
        result.input = self.view.input
        self.restrict = result.restrict
        result.notice = self._notice(Stage.GATE, matched, result.blocked, result.message)
        return result

    async def mask_history(self, messages: list[ChatMessage]) -> None:
        """Mask earlier messages the way the gate masks the question, in place.

        For history the server did not store itself: what a client sends back can
        carry what the gate masked on an earlier turn. Each message is judged on its
        own by the lenses enforced redact rules read; nothing is recorded.
        """
        lenses = self._redacting_lenses()
        if not lenses or len(messages) < 2:
            return
        earlier = messages[:-1]
        token = budgets.activate(self._budget)
        try:
            masked = await asyncio.gather(*(self._masked(lenses, m.content or "") for m in earlier))
        finally:
            budgets.deactivate(token)
        for message, text in zip(earlier, masked, strict=True):
            message.content = text
        self.view.history = [
            HistoryItem(role=m.role, content=m.content or "")
            for m in earlier
            if m.role in {"user", "assistant"} and m.content
        ][-12:]

    async def mask_text(self, text: str) -> str:
        """`text` masked the way the gate masks the question: for what tools hand the model.

        What it masks is recorded at watch, where it happens: each lens's answer, without
        its spans (they point into the tool's result, not the question), and the rule.
        """
        lenses = self._redacting_lenses()
        if not lenses:
            return text
        token = budgets.activate(self._budget)
        try:
            masked, observed, matched = await self._masking(lenses, text)
        finally:
            budgets.deactivate(token)
        if masked != text:
            self._record_masked_result(observed, matched)
        return masked

    def _record_masked_result(self, observed: list[Observed], matched: list[Matched]) -> None:
        """A tool result's masking, once per lens and rule however many results it masked."""
        seen = {(o.lens.name, o.stage) for o in self.observed}
        self.observed.extend(
            Observed(o.lens, Stage.WATCH, o.answer.model_copy(update={"spans": []}))
            for o in observed
            if o.answer.ok and o.answer.value and (o.lens.name, Stage.WATCH) not in seen
        )
        recorded = {(m.rule.name, m.stage) for m in self.matched}
        self.matched.extend(
            Matched(m.rule, Stage.WATCH, m.acted)
            for m in matched
            if (m.rule.name, Stage.WATCH) not in recorded
        )

    def _redacting_lenses(self) -> list[Lens]:
        """The gate's lenses that an enforced redact rule reads."""
        if not self.active or self.policy is None:
            return []
        redacting = [
            rule
            for rule in self.policy.rules.values()
            if rule.mode is Mode.ENFORCE and any(a.kind == "redact" for a in rule.then)
        ]
        names = {name for rule in redacting for name in rule.lenses()}
        return [
            lens
            for lens in self.policy.active_lenses(Stage.GATE, Subject.INPUT)
            if lens.kind == "entities" and lens.name in names
        ]

    async def _masked(self, lenses: list[Lens], text: str) -> str:
        """`text` with what the enforced redact rules mask at the gate masked."""
        return (await self._masking(lenses, text))[0]

    async def _masking(
        self, lenses: list[Lens], text: str
    ) -> tuple[str, list[Observed], list[Matched]]:
        """`text` masked, with the answers and the rules' masking that did it (none if nothing)."""
        assert self.policy is not None
        if not text:
            return text, [], []
        view = self.view.model_copy(update={"input": text, "answers": {}})
        observed = await run_lenses(
            lenses, view, Stage.GATE, deadline_ms=setting("GATE_DEADLINE_MS")
        )
        answers = {o.lens.name: o.answer for o in observed}
        redactions = []
        masking: list[Matched] = []
        for match in evaluate_rules(self.policy, answers, Stage.GATE, done=set()):
            masked = []
            for action in match.acted:
                if action.kind != "redact":
                    continue
                names = set(action.lenses) or match.rule.lenses()
                if spans := spans_for(observed, names, Subject.INPUT, text):
                    redactions.append((spans, action.template))
                    masked.append(action)
            if masked:
                masking.append(Matched(match.rule, match.stage, masked))
        if not redactions:
            return text, [], []
        return redact_all(text, redactions), observed, masking

    async def watch(self) -> ReviewResult:
        """Judge the answer so far, while it streams. Watch can only stop it.

        A match that blocks, or of a rule pinned to watch, is kept. Every other one
        is left for review, which judges the whole answer with fresh spans.
        """
        result = ReviewResult(output=self.view.output)
        if not self.active:
            return result
        self._watched_chars = len(self.view.output)
        try:
            matched = await self._stage(
                Stage.WATCH,
                subjects={Subject.OUTPUT, Subject.EXCHANGE},
                deadline=setting("WATCH_DEADLINE_MS"),
            )
            kept = [
                m
                for m in matched
                if any(a.kind == "block" for a in m.acted) or Stage.WATCH in m.rule.stages
            ]
            self.matched.extend(kept)
            return self._apply_output_actions(Stage.WATCH, kept, result)
        except Exception:
            return self._failed(Stage.WATCH, result)

    async def check_tool(self, call: ToolCall) -> str | None:
        """Judge one tool call before it runs; the refusal's reason when it may not."""
        if not self.active or self.policy is None:
            return None
        try:
            return await self._check_tool(call)
        except Exception:
            logger.exception("Oversight tool check failed for turn {}", self.id)
            self.error = "tool check failed"
            return Block.DEFAULT_MESSAGE if _fail_closed(Stage.WATCH) else None

    async def _check_tool(self, call: ToolCall) -> str | None:
        assert self.policy is not None
        if self.restrict is not None and (not self.restrict or call.name in self.restrict):
            return "This tool is not available for this request."
        lenses = self.policy.active_lenses(Stage.WATCH, Subject.TOOL_CALL)
        if not lenses:
            return None
        view = self.view.model_copy(update={"tool_call": call})
        token = budgets.activate(self._budget)
        try:
            observed = await run_lenses(
                lenses, view, Stage.WATCH, deadline_ms=setting("GATE_DEADLINE_MS")
            )
        finally:
            budgets.deactivate(token)
        self.observed.extend(observed)
        # This call's answers stay out of view.answers: the next call is judged on its own.
        answers = {o.lens.name: o.answer for o in observed}
        tool_lenses = {lens.name for lens in lenses}
        # Every call is judged, not just the first: a refused call stays refused on retry.
        matched = [
            m
            for m in evaluate_rules(
                self.policy,
                {**self.view.answers, **answers},
                Stage.WATCH,
                done=set(),
                errored=_errored(observed),
                fail_closed=_fail_closed(Stage.WATCH),
            )
            if m.rule.lenses() & tool_lenses
        ]
        # One ledger decision per rule, however many calls it refused.
        recorded = {m.rule.name for m in self.matched}
        self.matched.extend(m for m in matched if m.rule.name not in recorded)
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
        if self.unwatched():
            # The end of the answer is watched too, not only every WATCH_EVERY_CHARS.
            watched = await self.watch()
            if watched.blocked:
                return watched
        self.original_output = self.original_output or self.view.output
        result = ReviewResult(output=self.view.output)
        if not self.active:
            return result
        try:
            matched = await self._stage(
                Stage.REVIEW,
                subjects={Subject.OUTPUT, Subject.EXCHANGE, Subject.SOURCES},
                deadline=setting("REVIEW_DEADLINE_MS"),
            )
            self.matched.extend(matched)
            return self._apply_output_actions(Stage.REVIEW, matched, result)
        except Exception:
            return self._failed(Stage.REVIEW, result)

    def _failed(self, stage: Stage, result: ReviewResult) -> ReviewResult:
        """A stage that broke leaves the answer as it is, unless the stage fails closed."""
        logger.exception("Oversight {} failed for turn {}", stage, self.id)
        self.error = f"{stage} failed"
        if _fail_closed(stage):
            result.blocked = result.replaced = True
            result.message = result.output = self.view.output = Block.DEFAULT_MESSAGE
        return result

    def _apply_output_actions(
        self, stage: Stage, matched: list[Matched], result: ReviewResult
    ) -> ReviewResult:
        redactions: list[tuple[list[TextSpan], str]] = []
        for match in matched:
            for action in match.acted:
                if action.kind == "block" and not result.blocked:
                    result.blocked = True
                    result.message = match.rule.message or action.message
                elif action.kind == "redact":
                    spans = self._spans_for(action, match, Subject.OUTPUT)
                    if spans:
                        redactions.append((spans, action.template))
        if result.blocked:
            result.output = result.message
            result.replaced = True
        elif redactions:
            result.output = redact_all(result.output, redactions)
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
        return evaluate_rules(
            self.policy,
            self.view.answers,
            stage,
            done={m.rule.name for m in self.matched},
            errored=_errored(observed),
            fail_closed=_fail_closed(stage),
        )

    def _spans_for(self, action: Redact, match: Matched, subject: Subject) -> list[TextSpan]:
        """The spans a Redact action masks: from the lenses it names, on this subject."""
        names = set(action.lenses) or match.rule.lenses()
        return spans_for(self.observed, names, subject, self.view.input)

    def _notice(
        self,
        stage: Stage,
        matched: list[Matched],
        blocked: bool,
        message: str,
        review: ReviewResult | None = None,
    ) -> dict[str, Any] | None:
        acted = [m for m in matched if m.acted]
        level = setting("CLIENT_DETAIL")
        detail = level == "full" or (level == "staff" and self._staff_view)
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
        matches = [*matched, *self._tool_matches(stage, matched)]
        return {
            "policy": self.policy.name,
            "shadow_outcome": str(worst(*(m.outcome(shadow=True) for m in matches))),
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
                for m in matches
            ],
        }

    def _tool_matches(self, stage: Stage, matched: list[Matched]) -> list[Matched]:
        """The answer's notice also carries what tool calls matched: no notice of their own says so."""
        if stage is not Stage.REVIEW:
            return []
        return [m for m in self.matched if m.stage is Stage.WATCH and m not in matched]

    # Closing.

    @property
    def outcome(self) -> Outcome:
        refused = [Outcome.BLOCKED] if self.too_long else []
        return worst(*refused, *(m.outcome() for m in self.matched if m.acted))

    @property
    def shadow_outcome(self) -> Outcome:
        refused = [Outcome.BLOCKED] if self.too_long else []
        return worst(*refused, *(m.outcome(shadow=True) for m in self.matched))

    async def close(self, *, error: str = "") -> str | None:
        """Write the turn to the ledger and queue the background stages. Never raises.

        Returns the turn's id once written, None for an inert session or a failure.
        """
        if not self.active or self._closed:
            return None
        self._closed = True
        self.error = error or self.error
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
        return turn_id

    def notifications(self) -> list[tuple[dict[str, Any], list[str]]]:
        """What the turn's enforced Notify actions send: (event, exporter names)."""
        return [
            (
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
                list(action.exporters),
            )
            for match in self.matched
            for action in match.acted
            if action.kind == "notify"
        ]


def _errored(observed: list[Observed]) -> set[str]:
    """Lenses whose answer failed; one waiting for a person has not failed."""
    return {o.lens.name for o in observed if o.answer.error and not o.answer.queued}


def _fail_closed(stage: Stage) -> bool:
    return (setting("FAIL_MODE") or {}).get(str(stage)) == "closed"


def spans_for(
    observed: Iterable[Observed], names: set[str], subject: Subject, input_text: str
) -> list[TextSpan]:
    """The spans lenses `names` found in `subject`, from each lens's latest answer.

    An EXCHANGE lens reads the input, a blank line and the output, so its spans are
    moved onto the part asked for and those in the other part dropped.
    """
    latest: dict[str, Observed] = {}
    for o in observed:
        if o.lens.name in names:
            latest[o.lens.name] = o
    offset = len(input_text) + 2
    spans: list[TextSpan] = []
    for o in latest.values():
        if o.lens.subject is subject:
            spans.extend(o.answer.spans)
        elif o.lens.subject is Subject.EXCHANGE and subject is Subject.OUTPUT:
            spans.extend(
                s.model_copy(update={"start": s.start - offset, "end": s.end - offset})
                for s in o.answer.spans
                if s.start >= offset
            )
        elif o.lens.subject is Subject.EXCHANGE and subject is Subject.INPUT:
            spans.extend(s for s in o.answer.spans if s.end <= len(input_text))
    return spans


async def _staff_view(user: Any) -> bool:
    """Whether `user` may read oversight, asked only when CLIENT_DETAIL is "staff"."""
    if setting("CLIENT_DETAIL") != "staff" or getattr(user, "pk", None) is None:
        return False
    from django_ai_sdk.permissions import (
        Operation,
        PermissionDomain,
        get_domain_permissions,
        has_perms,
    )

    return await has_perms(
        user,
        Operation.VIEW_OVERSIGHT,
        permissions=get_domain_permissions(PermissionDomain.OVERSIGHT),
        raise_on_deny=False,
    )


def agent_key(agent: Agent) -> str:
    """The id turns and console choices know an agent by: its runtime id, else its registry id.

    A runtime agent built on a registered class has both; the console assigns it by
    its own id.
    """
    return str(getattr(agent, "agent_id", "") or "")


def code_policy(agent: Agent) -> tuple[str, str]:
    """The policy the agent's code asks for, and why: "agent", "off" or "default"."""
    getter = getattr(agent, "get_oversight_policy", None)
    name = getter() if callable(getter) else getattr(agent, "oversight", None)
    if name is False:
        return "", "off"
    if isinstance(name, str) and name:
        return name, "agent"
    return str(setting("DEFAULT_POLICY") or ""), "default"


def fingerprint(agent: Agent) -> str:
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
    if full:
        input_text = session.original_input
        output_text = session.original_output or view.output
    else:
        input_text = _redact_subject(session, Subject.INPUT)
        output_text = _redact_subject(session, Subject.OUTPUT)
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


def _redact_subject(session: TurnSession, subject: Subject) -> str:
    """`subject` as passed on, with what any Entities lens found there masked.

    For REDACTED capture. A lens that ran before a rule masked or replaced the text
    read it as said, so its offsets don't fit what was passed on: what it found is
    masked wherever it still shows there.
    """
    view = session.view
    if subject is Subject.INPUT:
        said, shown, masked_at = session.original_input or view.input, view.input, Stage.GATE
    else:
        said, shown, masked_at = session.original_output or view.output, view.output, Stage.REVIEW
    spans: list[TextSpan] = []
    for o in session.observed:
        if o.lens.kind != "entities" or o.lens.subject is not subject:
            continue
        if said == shown or STAGE_ORDER.index(o.stage) > STAGE_ORDER.index(masked_at):
            spans.extend(o.answer.spans)
            continue
        for span in o.answer.spans:
            found = said[span.start : span.end]
            if found:
                spans.extend(
                    TextSpan(start=m.start(), end=m.end(), label=span.label)
                    for m in re.finditer(re.escape(found), shown)
                )
    return redact_all(shown, [(spans, "[{label}]")]) if spans else shown


def _persist(session: TurnSession) -> str:
    """Write evidence, turn, observations and decisions, chained, in one transaction."""
    from datetime import timedelta

    from django.db import transaction

    from django_ai_sdk.oversight import ledger, vault
    from django_ai_sdk.oversight.models import Evidence, Turn

    assert session.policy is not None
    policy = session.policy
    now = timezone.now()
    created = session.created_at or now
    content = evidence_content(session, policy.capture)
    scope = f"thread:{session.thread_id}" if session.thread_id else f"turn:{session.id}"

    # What judges said can quote the turn: kept only where the turn's text is.
    reasons = policy.capture not in {Capture.NONE, Capture.METADATA}
    texts = text_values(session.observed) if reasons else {}

    with transaction.atomic():
        evidence = None
        evidence_hash = ""
        if content is not None:
            evidence_hash = ledger.sha256(ledger.canonical(content))
            stored = {**content, "texts": texts} if texts else content
            sealed = vault.encrypt_json(scope, stored)
            evidence = Evidence.objects.create(
                digest=evidence_hash,
                scope=scope,
                key=sealed[0] if sealed else None,
                ciphertext=sealed[1] if sealed else None,
                content=None if sealed else stored,
                capture=str(policy.capture),
                expires_at=now + timedelta(days=policy.retention["content"]),
            )
        turn = Turn.objects.create(
            id=session.id,
            created_at=created,
            expires_at=created + timedelta(days=policy.retention["metadata"]),
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
        )
        labels, severity, chained = record_judgments(
            turn, session.observed, session.matched, reasons=reasons
        )
        if session.outcome is Outcome.BLOCKED:
            severity = most_severe(severity, Severity.HIGH)
        Turn.objects.filter(pk=turn.pk).update(labels=labels, severity=severity)
        ledger.append_many([("turn", turn), *chained])
        if _needs_reflect(session):
            transaction.on_commit(lambda: _enqueue_reflect(turn.id))
        if events := session.notifications():
            transaction.on_commit(lambda: _enqueue_export(events))
    return str(turn.id)


def record_judgments(
    turn: Any,
    observed: list[Observed],
    matched: list[Matched],
    *,
    reasons: bool,
    reviewer: str = "",
) -> tuple[dict[str, Any], str, list[tuple[str, Any]]]:
    """Write the observations, decisions and review items of `turn`. Sync, in a transaction.

    Returns the labels the answers set, the severity the enforced alerts and
    notifications ask for, and the rows to chain, in order, for `ledger.append_many`.
    """
    from django_ai_sdk.oversight.models import Decision, Observation, ReviewItem

    observations = Observation.objects.bulk_create(
        Observation(
            turn=turn,
            reviewer=reviewer,
            **observation_fields(
                o.lens.name, o.stage, o.answer, text=o.lens.kind == "text", reasons=reasons
            ),
        )
        for o in observed
    )
    decisions = Decision.objects.bulk_create(
        Decision(
            turn=turn,
            rule=match.rule.name,
            stage=str(match.stage),
            mode=str(match.rule.mode),
            actions=[a.to_dict() for a in match.rule.then],
            acted=bool(match.acted),
            condition=match.rule.when.to_dict(),
        )
        for match in matched
    )
    acted = [(match, action) for match in matched for action in match.acted]
    reviews = {(o.answer.queued, o.lens.name, "") for o in observed if o.answer.queued}
    reviews |= {(a.queue, "", m.rule.name) for m, a in acted if a.kind == "queue"}
    priority = {a.queue: a.priority for m, a in reversed(acted) if a.kind == "queue"}
    # A retried reflect or a second verdict may ask for a review that is already open.
    ReviewItem.objects.bulk_create(
        [
            ReviewItem(
                turn=turn, queue=queue, lens=lens, rule=rule, priority=priority.get(queue, 0)
            )
            for queue, lens, rule in sorted(reviews)
        ],
        ignore_conflicts=True,
    )
    labels = {
        obs.lens: obs.value
        for o, obs in zip(observed, observations, strict=True)
        if o.answer.ok and o.lens.kind in {"choice", "flag", "entities", "score"}
    }
    severity = most_severe(*(a.severity for m, a in acted if a.kind in {"alert", "notify"}))
    chained = [("observation", o) for o in observations] + [("decision", d) for d in decisions]
    return labels, severity, chained


def observation_fields(
    lens: str, stage: Stage | str, answer: Answer, *, text: bool = False, reasons: bool = True
) -> dict[str, Any]:
    """An Observation's columns. Decided here, once: `value` is hashed into the ledger.

    A Text lens's value is prose about the turn, so it lives with the evidence
    (`text_values`) and is shredded with it; `reasons=False` drops the rationale.
    """
    from django_ai_sdk.utils import serialize

    return {
        "lens": lens,
        "stage": str(stage),
        "answerer": answer.answerer[:200],
        "model": answer.model[:255],
        "value": None if text else serialize(answer.value),
        "confidence": answer.confidence if answer.ok else None,
        "rationale": answer.rationale if reasons else "",
        "spans": [s.model_dump() for s in answer.spans],
        "items": answer.items if reasons else {},
        "trail": [_trail(a, text=text) for a in answer.trail],
        "error": answer.error[:200],
        "queued": answer.queued[:100],
        "cost": answer.cost,
        "tokens": answer.tokens,
        "latency_ms": answer.latency_ms,
    }


def text_values(observed: Iterable[Observed]) -> dict[str, Any]:
    """Text lens answers by lens, for the evidence's `texts`: outside the hashed content."""
    from django_ai_sdk.utils import serialize

    return {
        o.lens.name: serialize(o.answer.value)
        for o in observed
        if o.lens.kind == "text" and o.answer.ok and o.answer.value is not None
    }


def _trail(answer: Answer, *, text: bool = False) -> dict[str, Any]:
    from django_ai_sdk.utils import serialize

    return {
        "answerer": answer.answerer,
        "model": answer.model,
        "value": None if text else serialize(answer.value),
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


def _needs_reflect(session: TurnSession) -> bool:
    assert session.policy is not None
    escalations = any(a.kind == "escalate" for m in session.matched for a in m.acted)
    unfinished = any(
        asks_again(o.answer.error, o.answer.trail[-1].error if o.answer.trail else "")
        for o in session.observed
    )
    return escalations or unfinished or session.policy.has_stage(Stage.REFLECT)


def _enqueue_export(events: list[tuple[dict[str, Any], list[str]]]) -> None:
    """Hand notifications to the worker: a slow exporter must not hold the stream open."""
    try:
        from django_ai_sdk.oversight.tasks import export_events

        export_events.enqueue([[event, names] for event, names in events])
    except Exception:
        logger.exception("Could not queue oversight notifications; is a task backend configured?")


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
