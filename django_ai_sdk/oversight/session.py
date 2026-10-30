"""TurnSession: the one seam every turn passes through, whatever its channel.

Chat, webhooks, workflow steps and your own code all open a session, report what
happens (sources found, tools called, text produced) and ask it at three points
whether to go on: the gate, while watching, and at finish. Closing it writes the
turn to the ledger (see `record`) and hands it to the background stages.

    session = await TurnSession.start(agent, channel="api", user=user, input=text)
    gate = await session.gate()
    if gate.blocked:
        return gate.message
    answer = await run_my_agent(gate.input)
    finished = await session.finish(answer)
    await session.close()
    return finished.output

A session for an agent without a policy, or with oversight off, is inert: every
call returns at once and allows everything.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import hashlib
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from asgiref.sync import sync_to_async
from django.utils import timezone
from django.utils.translation import gettext_lazy as _

from django_ai_sdk.logger import get_logger
from django_ai_sdk.oversight import budget as budgets
from django_ai_sdk.oversight.answerers.pii import mask_all
from django_ai_sdk.oversight.attach import FINGERPRINT_CHARS, acode_policy, agent_key, fingerprint
from django_ai_sdk.oversight.conf import is_enabled, setting
from django_ai_sdk.oversight.record import as_source, persist_turn
from django_ai_sdk.oversight.rules import Block, Mask, Notify, RestrictTools
from django_ai_sdk.oversight.stages import (
    Matched,
    Observed,
    evaluate_rules,
    run_lenses,
)
from django_ai_sdk.oversight.types import (
    Capture,
    HistoryItem,
    Mode,
    Outcome,
    Stage,
    Subject,
    TextSpan,
    ToolCall,
    TurnView,
    worst,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Generator, Iterable, Mapping

    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser

    from django_ai_sdk.common import ChatMessage
    from django_ai_sdk.oversight.lenses import Lens
    from django_ai_sdk.oversight.policy import Policy, ResolvedPolicy
    from django_ai_sdk.oversight.rules import Action

logger = get_logger(__name__)

TOO_LONG = _("This message is too long to check. Shorten it and try again.")
# What the model reads instead of a tool result (or an earlier message) that could
# not be checked for what the policy masks.
UNCHECKED = _("[withheld: could not be checked]")
# A refused call to a tool the policy took away this turn.
TOOL_UNAVAILABLE = _("This tool is not available for this request.")
# Earlier messages the lenses see with a question.
HISTORY_TURNS = 12
# Characters of answer text a watch check reads again before the new text, so what
# straddles two checks is still read whole.
WATCH_OVERLAP_CHARS = 200
# The most of a turn's errors the ledger keeps.
ERROR_CHARS = 500

_unobserved: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "django_ai_sdk_oversight_unobserved", default=False
)
_current: contextvars.ContextVar[TurnSession | None] = contextvars.ContextVar(
    "django_ai_sdk_oversight_session", default=None
)


@contextlib.contextmanager
def unobserved() -> Generator[None]:
    """Run code that must not open turns of its own: judges, titles."""
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
    """The turn this code runs in: None outside one, and once that turn is closed.

    A task started inside a turn copies it, and may outlive it: a closed turn is
    never joined.
    """
    session = _current.get()
    return session if session is not None and not session._closed else None


@dataclass
class GateResult:
    """What the gate decided about the question."""

    blocked: bool = False
    message: str = ""
    # The question as the model should see it: masked when a rule said so.
    input: str = ""
    masked: bool = False
    # Tool names taken away; None leaves every tool, [] takes all of them.
    restrict: list[str] | None = None
    # For the client: a `data-oversight` payload, when there is anything to say.
    notice: dict[str, Any] | None = None


@dataclass
class FinishResult:
    """What finish (or a watch check) decided about the answer."""

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
    # What went wrong in the turn, "; " between: recorded on the turn.
    error: str = ""
    # Refused at the gate unread, for being longer than MAX_INPUT_CHARS.
    too_long: bool = False
    # A stage refused the turn because it failed closed (FAIL_MODE), its policy could
    # not be loaded, or the caller refused it (`refuse`): the turn is blocked.
    refused: bool = False
    original_input: str = ""
    original_output: str = ""
    # data-oversight payloads waiting for the stream to send them.
    notices: list[dict[str, Any]] = field(default_factory=list)
    # The turn this one runs inside, when it runs under a policy of its own: a
    # subagent's delegation, a nested run().
    parent_turn_id: str = ""
    # Set by a streaming adapter: refused tool calls' stub results, so the chat
    # shows them refused at once instead of still running.
    on_tool_results: Callable[[list[Any]], Awaitable[None]] | None = None
    _started: float = field(default_factory=time.perf_counter)
    _watched_chars: int = 0
    _budget: budgets.Budget | None = None
    # Whether the person chatting may see oversight detail (CLIENT_DETAIL "staff").
    _staff_view: bool = False
    _closed: bool = False
    # Tool results being masked, by call id and text: the stream and the tool hook share one.
    _masking_results: dict[str, asyncio.Future[str]] = field(default_factory=dict)
    # Whether each lens's latest answer this turn failed: a lens that failed at one
    # stage stays failed at the next, until it answers again.
    _lens_failed: dict[str, bool] = field(default_factory=dict)
    # No policy could be chosen or loaded and the gate fails closed: every stage refuses.
    _unloaded: bool = False

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
        oversight: type[Policy] | str | Literal[False] | None = None,
    ) -> TurnSession:
        """A session for one turn of `agent`; inert unless a policy applies.

        `oversight` replaces the agent's own for this turn: a Policy subclass or a
        name. False opens no turn. None takes the agent's own: the console's choice,
        else what its code asks for (`acode_policy`).

        A policy that cannot be chosen (the agent's `get_oversight` raised) or loaded
        (unknown, broken, its definition unreadable) follows FAIL_MODE["gate"]: open,
        the turn runs unwatched and the failure is logged; closed, the session is
        active only to refuse: every stage refuses with Block.DEFAULT_MESSAGE, and
        the turn is recorded blocked with the error.
        """
        if oversight is False or _unobserved.get() or not is_enabled():
            return cls()
        from django_ai_sdk.oversight.policy import aassigned_policy, aget_policy
        from django_ai_sdk.oversight.registry import policy_name

        opening = {
            "agent": agent,
            "channel": channel,
            "user": user,
            "thread_id": thread_id,
            "messages": messages,
            "input": input,
        }
        name = ""
        try:
            name = policy_name(oversight) if oversight else ""
            if not name:
                # The console's choice for this agent wins over what its code says.
                assigned = await aassigned_policy(agent_key(agent))
                if assigned is None:
                    name = (await acode_policy(agent, channel=channel, user=user))[0]
                else:
                    name = assigned
        except Exception:
            logger.exception("Oversight could not choose a policy for {!r}", agent_key(agent))
            return cls._unloadable(name="", error="policy could not be chosen", **opening)
        if not name:
            return cls()
        try:
            resolved = await aget_policy(name)
        except Exception:
            logger.exception("Oversight policy {!r} could not be loaded", name)
            resolved = None
        if resolved is None:
            return cls._unloadable(name=name, error="policy could not be loaded", **opening)
        session = cls._opened(resolved, **opening)
        session._budget = await _budget()
        session._staff_view = await _staff_view(user)
        return session

    @classmethod
    def _opened(
        cls,
        policy: ResolvedPolicy,
        *,
        agent: Any,
        channel: str,
        user: Any,
        thread_id: str | None,
        messages: list[ChatMessage] | None,
        input: str | None,
    ) -> TurnSession:
        from django_ai_sdk.oversight.privacy import pseudonym

        messages = messages or []
        if input is None:
            input = next((m.content for m in reversed(messages) if m.role == "user"), "") or ""
        session = cls(
            active=True,
            policy=policy,
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
        )
        session.set_history(messages)
        session.agent_fingerprint = fingerprint(agent)
        return session

    @classmethod
    def _unloadable(cls, *, name: str, error: str, **opening: Any) -> TurnSession:
        """The session for a turn whose policy could not be chosen or loaded: FAIL_MODE["gate"].

        Open: inert, the turn runs unwatched. Closed: active only to refuse, under an
        empty stand-in for the policy, so the refusal and its error are recorded.
        """
        if not _fail_closed(Stage.GATE):
            logger.error(
                "Oversight runs a turn of {!r} unwatched: its {} (FAIL_MODE gate is open)",
                agent_key(opening["agent"]),
                error,
            )
            return cls()
        try:
            session = cls._opened(_stand_in(name), **opening)
        except Exception:
            # Not even a refusal can be recorded: refuse without a record.
            logger.exception("Oversight could not open a refusing turn")
            session = cls(active=True, policy=_stand_in(name), _closed=True)
        session._unloaded = True
        session.refused = True
        session.note_error(error)
        logger.error(
            "Oversight refuses a turn of {!r}: its {} (FAIL_MODE gate is closed)",
            agent_key(opening["agent"]),
            error,
        )
        return session

    # Reporting.

    def note_error(self, error: str) -> None:
        """Add `error` to what went wrong in this turn: recorded on the turn, no content."""
        if error and error not in self.error.split("; "):
            self.error = f"{self.error}; {error}" if self.error else error

    def refuse(self, error: str = "") -> None:
        """Record the turn as refused by its caller: a result oversight let through
        that the caller still cannot pass on (a masked result that no longer fits its
        schema). The turn is recorded blocked, with `error`."""
        if not self.active:
            return
        self.refused = True
        self.note_error(error)

    def set_history(self, messages: list[ChatMessage]) -> None:
        """The conversation before the question, as the lenses see it: the latest turns."""
        self.view.history = [
            HistoryItem(role=m.role, content=m.content or "")
            for m in messages[:-1]
            if m.role in {"user", "assistant"} and m.content
        ][-HISTORY_TURNS:]

    def set_sources(self, sources: Iterable[Any], *, seeded: Iterable[Any] = ()) -> None:
        """What retrieval found, as NumberedSource, Source or dicts.

        `seeded` names those carried over from earlier turns: sources, or their keys
        (a SourceRegistry keeps keys).
        """
        if not self.active:
            return
        seeded_keys = {s if isinstance(s, str) else as_source(s).key for s in seeded}
        fresh = [as_source(s) for s in sources]
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
        return due and self._watches_text()

    def unwatched(self) -> bool:
        """Whether answer text arrived since the last watch check, with lenses to judge it."""
        if not self.active or len(self.view.output) <= self._watched_chars:
            return False
        return self._watches_text()

    def _watches_text(self) -> bool:
        """Whether a watch lens reads the answer; one on tool calls only never does."""
        return self.policy is not None and any(
            lens.subject in {Subject.OUTPUT, Subject.EXCHANGE}
            for lens in self.policy.active_lenses(Stage.WATCH)
        )

    # Deciding.

    async def gate(self) -> GateResult:
        """Judge the question before anyone sees it."""
        result = GateResult(input=self.view.input)
        if not self.active:
            return result
        # An active session always has its policy.
        assert self.policy is not None
        if self._unloaded:
            return self._refuse_gate(result, str(Block.DEFAULT_MESSAGE))
        limit = setting("MAX_INPUT_CHARS")
        if limit and len(self.view.input) > limit:
            # Whatever the policy reads later (the question beside the answer, in
            # reflect), past the limit it would read only part of it.
            self.too_long = True
            return self._refuse_gate(result, str(TOO_LONG))
        try:
            return await self._gate(result)
        except Exception:
            # Like watch and finish: a broken lens or rule follows the gate's fail mode.
            logger.exception("Oversight gate failed for turn {}", self.id)
            self.note_error("gate failed")
            if _fail_closed(Stage.GATE):
                self.refused = True
                return self._refuse_gate(result, str(Block.DEFAULT_MESSAGE))
            return result

    def _refuse_gate(self, result: GateResult, message: str) -> GateResult:
        result.blocked = True
        result.message = message
        result.notice = {
            "turn_id": self.id,
            "stage": str(Stage.GATE),
            "outcome": str(Outcome.BLOCKED),
            "message": message,
        }
        return result

    async def _gate(self, result: GateResult) -> GateResult:
        matched = await self._stage(
            Stage.GATE, subjects={Subject.INPUT}, deadline=setting("GATE_DEADLINE_MS")
        )
        self.matched.extend(matched)
        masks = self._masks(Stage.GATE, matched, Subject.INPUT)
        # What earlier messages took away (check_history) stays away.
        result.restrict = self.restrict
        for match in matched:
            for action in match.acted:
                if isinstance(action, Block) and not result.blocked:
                    result.blocked = True
                    result.message = str(match.rule.message or action.message)
                elif isinstance(action, RestrictTools):
                    result.restrict = _taken(result.restrict, action.names)
        if masks:
            masked = mask_all(self.view.input, masks)
            # "Masked" only when the question changed.
            result.masked = masked != self.view.input
            self.view.input = masked
        result.input = self.view.input
        self.restrict = result.restrict
        result.notice = self._notice(Stage.GATE, matched, result.blocked, result.message)
        return result

    def _masks(
        self, stage: Stage, matched: list[Matched], subject: Subject
    ) -> list[tuple[list[TextSpan], str]]:
        """What the matches' Mask actions mask in `subject`, as (spans, template) groups.

        A mask that cannot point at what it should mask (a lens it reads failed while
        the stage fails closed, found something without saying where, or pointed
        outside the text) blocks instead: its action becomes a Block. One that would
        change nothing is dropped from what the match did, so the turn never reads
        "masked" when nothing was.
        """
        closed = _fail_closed(stage)
        texts = self._texts()
        text = texts[subject]
        masks: list[tuple[list[TextSpan], str]] = []
        for match in matched:
            acted: list[Action] = []
            for action in match.acted:
                if not isinstance(action, Mask):
                    acted.append(action)
                    continue
                names = set(action.lenses) or match.rule.lenses()
                if _unpointed(self.observed, names, closed=closed, subject=subject, texts=texts):
                    acted.append(Block(match.rule.message))
                    continue
                spans = spans_for(self.observed, names, subject, self.view.input)
                group = (spans, action.template)
                if spans and mask_all(text, [group]) != text:
                    masks.append(group)
                    acted.append(action)
            match.acted = acted
        return masks

    def _texts(self) -> dict[Subject, str]:
        """The text each subject's lenses read, as they read it now."""
        return {
            Subject.INPUT: self.view.input,
            Subject.OUTPUT: self.view.output,
            Subject.EXCHANGE: f"{self.view.input}\n\n{self.view.output}",
            Subject.SOURCES: "",
            Subject.TOOL_CALL: "",
        }

    async def check_history(
        self, messages: list[ChatMessage], *, history: bool = True
    ) -> GateResult:
        """Judge what comes with the question: earlier messages and attachments' context.

        Call it before `gate()` on what the server did not store itself (a stateless
        run, a chat without a thread, a workflow's `history` input). Each earlier
        user and assistant message, and the context of every message's attachments
        (file names, summaries, captions: what the model reads of them, the
        question's own too), is judged on its own by the gate's lenses on the
        question, and masked in place as the gate masks the question; text a mask
        cannot point into is withheld whole. With `history=False` only the
        attachments of the last message are judged: a stored thread was judged turn
        by turn.

        The result is blocked when an enforced rule that blocks, or takes every tool
        away, holds on any of them (a failed lens counts as holding where the gate
        fails closed); tools a rule takes away stay away for the turn. What was
        found is recorded beside the turn, once per lens and rule: never what a mask
        of the question points with, nor what the gate's rules judge the question by.
        Rules that only alert, queue or notify are not raised by history: a client
        sends it back on every turn.
        """
        result = GateResult(input=self.view.input)
        if not self.active or self.policy is None:
            return result
        if self._unloaded:
            return self._refuse_gate(result, str(Block.DEFAULT_MESSAGE))
        lenses = self.policy.active_lenses(Stage.GATE, Subject.INPUT)
        items = _history_items(messages, history=history) if lenses else []
        if not items:
            self.set_history(messages)
            return result
        token = budgets.activate(self._budget)
        try:
            judged = await asyncio.gather(
                *(self._judged(lenses, get(), Stage.GATE, blocks=True) for get, _ in items)
            )
        except Exception:
            logger.exception("Oversight could not check the history of turn {}", self.id)
            self.note_error("history check failed")
            if _fail_closed(Stage.GATE):
                self.refused = True
                return self._refuse_gate(result, str(Block.DEFAULT_MESSAGE))
            return result
        finally:
            budgets.deactivate(token)
        found: list[Matched] = []
        for (_read, put), one in zip(items, judged, strict=True):
            put(one.text)
            found.extend(one.matched)
            if one.block is not None and not result.blocked:
                result.blocked = True
                result.message = one.block
            if one.restrict is not None:
                self.restrict = _taken(self.restrict, one.restrict)
        recorded = self._record_aside(
            Stage.GATE, [o for one in judged for o in one.observed], found
        )
        if result.blocked:
            result.restrict = self.restrict
            result.notice = self._notice(Stage.GATE, recorded, True, result.message)
        self.set_history(messages)
        return result

    async def mask_text(self, text: str) -> str:
        """`text` masked the way the gate masks the question: for what tools hand the model.

        What it masks is recorded at watch, where it happens: each lens's answer, without
        its spans (they point into the tool's result, not the question), and the rule.
        A result that cannot be checked follows the watch fail mode: closed withholds
        all of it, open passes it as it is and records the failure. Never raises.
        """
        try:
            lenses = self._masking_lenses()
            if not lenses or not text:
                return text
            token = budgets.activate(self._budget)
            try:
                judged = await self._judged(lenses, text, Stage.WATCH)
            finally:
                budgets.deactivate(token)
            if judged.text != text or any(o.errored for o in judged.observed):
                self._record_aside(Stage.WATCH, judged.observed, judged.matched)
            return judged.text
        except Exception:
            logger.exception("Oversight could not mask a tool result for turn {}", self.id)
            self.note_error("tool result masking failed")
            return str(UNCHECKED) if _fail_closed(Stage.WATCH) else text

    async def mask_result(self, call_id: str, text: str) -> str:
        """A tool result masked like `mask_text`, judged once per call and text however often asked.

        The stream shows a result before the tool hook hands it to the model: both
        ask here, and both get the same masked text. Another text under the same call
        (a result cut to fit the model's limit) is masked on its own.
        """
        if not call_id:
            return await self.mask_text(text)
        key = f"{call_id}:{hashlib.sha256(text.encode()).hexdigest()[:FINGERPRINT_CHARS]}"
        pending = self._masking_results.get(key)
        if pending is None:
            pending = self._masking_results[key] = asyncio.ensure_future(self.mask_text(text))
        return await asyncio.shield(pending)

    def _record_aside(
        self, stage: Stage, observed: list[Observed], matched: list[Matched]
    ) -> list[Matched]:
        """What lenses found beside the turn's own text, once per lens and rule however often.

        Answers without their spans (they point into another text), marked aside: a
        mask of the question or the answer never points with them, and the rules
        still judge the turn itself. A lens that failed is recorded too. Returns the
        matches recorded now.
        """
        seen = {(o.lens.name, o.stage) for o in self.observed if o.aside}
        for o in observed:
            if ((o.answer.ok and o.answer.value) or o.errored) and (o.lens.name, stage) not in seen:
                seen.add((o.lens.name, stage))
                answer = o.answer.model_copy(update={"spans": []})
                self.observed.append(Observed(o.lens, stage, answer, o.failed, aside=True))
        recorded = {(m.rule.name, m.stage) for m in self.matched if m.aside}
        added: list[Matched] = []
        for m in matched:
            if (m.rule.name, stage) not in recorded:
                recorded.add((m.rule.name, stage))
                added.append(Matched(m.rule, stage, m.acted, aside=True))
        self.matched.extend(added)
        return added

    def _masking_lenses(self) -> list[Lens]:
        """The gate's lenses that an enforced mask rule reads."""
        if not self.active or self.policy is None:
            return []
        masking = [
            rule
            for rule in self.policy.rules.values()
            if rule.mode is Mode.ENFORCE and any(a.kind == "mask" for a in rule.then)
        ]
        names = {name for rule in masking for name in rule.lenses()}
        return [
            lens
            for lens in self.policy.active_lenses(Stage.GATE, Subject.INPUT)
            if lens.kind == "entities" and lens.name in names
        ]

    async def _judged(
        self, lenses: list[Lens], text: str, stage: Stage, *, blocks: bool = False
    ) -> _Judged:
        """`text` judged by `lenses` as the gate judges the question, and masked.

        `stage` names the fail mode that applies. Text a mask cannot point into (a lens
        failed while that stage fails closed, found something without saying where,
        or pointed outside it) is withheld whole: UNCHECKED. With `blocks`, what
        enforced rules that block or take tools away say is kept too; only the rules'
        protective actions are, so nothing beside the turn alerts or notifies.
        """
        assert self.policy is not None
        judged = _Judged(text=text)
        if not text:
            return judged
        view = self.view.model_copy(update={"input": text, "answers": {}})
        judged.observed = await run_lenses(
            lenses, view, Stage.GATE, deadline_ms=setting("GATE_DEADLINE_MS")
        )
        answers = {o.lens.name: o.answer for o in judged.observed}
        closed = _fail_closed(stage)
        texts = {Subject.INPUT: text}
        masks = []
        withheld = False
        for match in evaluate_rules(
            self.policy,
            answers,
            Stage.GATE,
            done=set(),
            errored=_errored(judged.observed),
            fail_closed=closed,
        ):
            acting: list[Action] = []
            for action in match.acted:
                if isinstance(action, Mask):
                    names = set(action.lenses) or match.rule.lenses()
                    if _unpointed(
                        judged.observed, names, closed=closed, subject=Subject.INPUT, texts=texts
                    ):
                        withheld = True
                        acting.append(action)
                    elif spans := spans_for(judged.observed, names, Subject.INPUT, text):
                        if mask_all(text, [(spans, action.template)]) != text:
                            masks.append((spans, action.template))
                            acting.append(action)
                elif blocks and isinstance(action, Block):
                    judged.block = judged.block or str(match.rule.message or action.message)
                    acting.append(action)
                elif blocks and isinstance(action, RestrictTools):
                    if not action.names:
                        judged.block = judged.block or str(match.rule.message or TOOL_UNAVAILABLE)
                    judged.restrict = _taken(judged.restrict, action.names)
                    acting.append(action)
            if acting:
                judged.matched.append(Matched(match.rule, match.stage, acting))
        if withheld:
            judged.text = str(UNCHECKED)
        elif masks:
            judged.text = mask_all(text, masks)
        return judged

    async def watch(self) -> FinishResult:
        """Judge the answer so far, while it streams. Watch can only stop it.

        Each check reads only the text that arrived since the last one, and a little
        of what came before (WATCH_OVERLAP_CHARS), so a long answer costs no more
        per check than a short one. A match that blocks, or of a rule pinned to
        watch, is kept. Every other one is left for finish, which judges the whole
        answer with fresh spans.
        """
        result = FinishResult(output=self.view.output)
        if not self.active:
            return result
        if self._unloaded:
            return self._refuse_output(result)
        start = max(0, self._watched_chars - WATCH_OVERLAP_CHARS)
        self._watched_chars = len(self.view.output)
        try:
            matched = await self._stage(
                Stage.WATCH,
                subjects={Subject.OUTPUT, Subject.EXCHANGE},
                deadline=setting("WATCH_DEADLINE_MS"),
                offset=start,
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
        if self._unloaded:
            return str(Block.DEFAULT_MESSAGE)
        try:
            return await self._check_tool(call)
        except Exception:
            logger.exception("Oversight tool check failed for turn {}", self.id)
            self.note_error("tool check failed")
            return str(Block.DEFAULT_MESSAGE) if _fail_closed(Stage.WATCH) else None

    def _taken_away(self, name: str) -> bool:
        return self.restrict is not None and (not self.restrict or name in self.restrict)

    async def _check_tool(self, call: ToolCall) -> str | None:
        assert self.policy is not None
        if self._taken_away(call.name):
            return str(TOOL_UNAVAILABLE)
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
        self._observe(Stage.WATCH, observed)
        # This call's answers stay out of view.answers: the next call is judged on its own.
        answers = {o.lens.name: o.answer for o in observed}
        tool_lenses = {lens.name for lens in lenses}
        # Other lenses that failed this turn still count; a tool lens only for this call.
        errored = {name for name in self._failed_lenses() if name not in tool_lenses}
        # Every call is judged, not just the first: a refused call stays refused on retry.
        matched = [
            m
            for m in evaluate_rules(
                self.policy,
                {**self.view.answers, **answers},
                Stage.WATCH,
                done=set(),
                errored=errored | _errored(observed),
                fail_closed=_fail_closed(Stage.WATCH),
            )
            if m.rule.lenses() & tool_lenses
        ]
        # One ledger decision per rule, however many calls it refused.
        recorded = {m.rule.name for m in self.matched if not m.aside}
        self.matched.extend(m for m in matched if m.rule.name not in recorded)
        # Tools taken away stay away: later calls are refused, get_tools() leaves them out.
        for match in matched:
            for action in match.acted:
                if isinstance(action, RestrictTools):
                    self.restrict = _taken(self.restrict, action.names)
        for match in matched:
            block = next((a for a in match.acted if isinstance(a, Block)), None)
            if block is not None:
                return str(match.rule.message or block.message)
        if self._taken_away(call.name):
            restricting = next(
                (m for m in matched if any(a.kind == "restrict_tools" for a in m.acted)), None
            )
            return str((restricting and restricting.rule.message) or TOOL_UNAVAILABLE)
        return None

    async def finish(self, output: str | None = None) -> FinishResult:
        """Judge the finished answer before it is stored."""
        if output is not None:
            self.view.output = output
        if self.unwatched():
            # The end of the answer is watched too, not only every WATCH_EVERY_CHARS.
            watched = await self.watch()
            if watched.blocked:
                return watched
        self.original_output = self.original_output or self.view.output
        result = FinishResult(output=self.view.output)
        if not self.active:
            return result
        if self._unloaded:
            return self._refuse_output(result)
        try:
            matched = await self._stage(
                Stage.FINISH,
                subjects={Subject.OUTPUT, Subject.EXCHANGE, Subject.SOURCES},
                deadline=setting("FINISH_DEADLINE_MS"),
            )
            self.matched.extend(matched)
            return self._apply_output_actions(Stage.FINISH, matched, result)
        except Exception:
            return self._failed(Stage.FINISH, result)

    def _failed(self, stage: Stage, result: FinishResult) -> FinishResult:
        """A stage that broke leaves the answer as it is, unless the stage fails closed."""
        logger.exception("Oversight {} failed for turn {}", stage, self.id)
        self.note_error(f"{stage} failed")
        if _fail_closed(stage):
            self.refused = True
            return self._refuse_output(result)
        return result

    def _refuse_output(self, result: FinishResult) -> FinishResult:
        result.blocked = result.replaced = True
        result.message = result.output = self.view.output = str(Block.DEFAULT_MESSAGE)
        return result

    def _apply_output_actions(
        self, stage: Stage, matched: list[Matched], result: FinishResult
    ) -> FinishResult:
        masks = self._masks(stage, matched, Subject.OUTPUT)
        for match in matched:
            for action in match.acted:
                if isinstance(action, Block) and not result.blocked:
                    result.blocked = True
                    result.message = str(match.rule.message or action.message)
        if result.blocked:
            result.output = result.message
            result.replaced = True
        elif masks:
            masked = mask_all(result.output, masks)
            result.replaced = masked != result.output
            result.output = masked
        if result.replaced:
            self.original_output = self.original_output or self.view.output
            self.view.output = result.output
        result.notice = self._notice(stage, matched, result.blocked, result.message, result)
        return result

    async def _stage(
        self,
        stage: Stage,
        *,
        subjects: set[Subject],
        deadline: float | None,
        offset: int = 0,
    ) -> list[Matched]:
        """Ask the stage's lenses on `subjects` and return the rules that now hold.

        `offset`: the lenses read the answer from there on (a watch check); their
        spans are moved to index the whole answer.
        """
        assert self.policy is not None
        lenses = [lens for lens in self.policy.active_lenses(stage) if lens.subject in subjects]
        view = self.view
        if offset:
            view = self.view.model_copy(update={"output": self.view.output[offset:]})
        token = budgets.activate(self._budget)
        try:
            observed = await run_lenses(lenses, view, stage, deadline_ms=deadline)
        finally:
            budgets.deactivate(token)
        if offset:
            observed = [_moved(o, offset, len(self.view.input)) for o in observed]
        self._observe(stage, observed)
        return evaluate_rules(
            self.policy,
            self.view.answers,
            stage,
            done={m.rule.name for m in self.matched if not m.aside},
            errored=self._failed_lenses(),
            fail_closed=_fail_closed(stage),
        )

    def _observe(self, stage: Stage, observed: list[Observed]) -> None:
        """Keep the stage's answers: in the record, and in what rules judge.

        Watch asks the same lenses again and again: per lens it keeps one answer,
        the latest, unless an earlier one found something and the latest did not.
        Every stage keeps each lens's latest failure state.
        """
        for o in observed:
            if o.lens.subject is not Subject.TOOL_CALL:
                # A tool call's lens judges that call alone: its failure is the call's.
                self._lens_failed[o.lens.name] = o.errored
            kept = o
            if stage is Stage.WATCH:
                index = next(
                    (
                        i
                        for i, e in enumerate(self.observed)
                        if e.stage is stage and not e.aside and e.lens.name == o.lens.name
                    ),
                    None,
                )
                if index is not None:
                    earlier = self.observed[index]
                    if _found(earlier) and not _found(o) and not o.errored:
                        kept = earlier
                    self.observed[index] = kept
                else:
                    self.observed.append(o)
            else:
                self.observed.append(o)
            if o.lens.subject is not Subject.TOOL_CALL:
                self.view.answers[o.lens.name] = kept.answer

    def _failed_lenses(self) -> set[str]:
        """Lenses whose latest answer this turn failed."""
        return {name for name, failed in self._lens_failed.items() if failed}

    def _notice(
        self,
        stage: Stage,
        matched: list[Matched],
        blocked: bool,
        message: str,
        finished: FinishResult | None = None,
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
        }
        if blocked:
            notice["message"] = message
        if finished is not None and finished.replaced:
            notice["replace"] = finished.output
        # A refused question is not echoed, masked or not: part of it may not be.
        masked = any(a.kind == "mask" for m in acted for a in m.acted)
        if stage is Stage.GATE and masked and not blocked:
            notice["input"] = self.view.input
        if detail:
            # Rule names say what the policy tests: only for a developer view.
            notice["rules"] = [m.rule.name for m in acted]
            notice["detail"] = self._detail(stage, matched)
        return notice

    def _detail(self, stage: Stage, matched: list[Matched]) -> dict[str, Any]:
        """Every answer and match of `stage`, for a developer view in the client.

        Each value as the record keeps it (`record.stored_value`): no `Text` answer,
        and a name an `Entities` answerer gave in its own words only as "[found]",
        so the client never reads what a mask took out of the question or answer.
        """
        from django_ai_sdk.oversight.record import stored_value

        assert self.policy is not None
        matches = [*matched, *self._tool_matches(stage, matched)]
        return {
            "policy": self.policy.name,
            "report_outcome": str(worst(*(m.outcome(report=True) for m in matches))),
            "pending": sorted(lens.name for lens in self.policy.active_lenses(Stage.REFLECT))
            if stage is Stage.FINISH
            else [],
            "observations": [
                {
                    "lens": o.lens.name,
                    "label": o.lens.label or o.lens.name.replace("_", " "),
                    "kind": o.lens.kind,
                    "value": stored_value(o.lens, o.answer),
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
        if stage is not Stage.FINISH:
            return []
        return [m for m in self.matched if m.stage is Stage.WATCH and m not in matched]

    # Closing.

    @property
    def outcome(self) -> Outcome:
        refused = [Outcome.BLOCKED] if self.too_long or self.refused else []
        return worst(*refused, *(m.outcome() for m in self.matched if m.acted))

    @property
    def report_outcome(self) -> Outcome:
        refused = [Outcome.BLOCKED] if self.too_long or self.refused else []
        return worst(*refused, *(m.outcome(report=True) for m in self.matched))

    async def close(self, *, error: str = "") -> str | None:
        """Write the turn to the ledger and queue the background stages. Never raises.

        Tool results still being masked are cancelled first: nothing is added to a
        closed turn. `error` is what went wrong in the run (the caller's exception,
        a disconnect), recorded with the errors oversight noted itself. Returns the
        turn's id once written, None for an inert session or a failure.
        """
        if not self.active or self._closed:
            return None
        self._closed = True
        await self._cancel_masking()
        self.note_error(error)
        try:
            turn_id = await sync_to_async(persist_turn, thread_sensitive=True)(self)
        except Exception:
            # No content: what the ledger lost, so an operator can tell.
            logger.exception(
                "Oversight could not record turn {}: outcome {}, acted {}, errors {}",
                self.id,
                self.outcome,
                sorted({m.rule.name for m in self.matched if m.acted}),
                self.error or "none",
            )
            return None
        if self._budget is not None:
            try:
                await self._budget.flush()
            except Exception:
                logger.exception("Oversight could not record spend for turn {}", self.id)
        return turn_id

    async def _cancel_masking(self) -> None:
        pending = [f for f in self._masking_results.values() if not f.done()]
        for future in pending:
            future.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

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
            if isinstance(action, Notify)
        ]


def _errored(observed: list[Observed]) -> set[str]:
    """Lenses whose answer failed; one waiting for a person has not failed."""
    return {o.lens.name for o in observed if o.errored}


def _taken(restrict: list[str] | None, names: list[str]) -> list[str]:
    """The tools taken away once `names` go too: None none, [] all of them."""
    if not names or restrict == []:
        return []
    return sorted({*(restrict or []), *names})


def _unpointed(
    observed: Iterable[Observed],
    names: set[str],
    *,
    closed: bool,
    subject: Subject,
    texts: Mapping[Subject, str],
) -> bool:
    """Whether a mask over lenses `names` cannot point at all it should mask in `subject`.

    The latest answer of one of them on that text failed while the stage fails
    closed, found something without saying where, or pointed outside the text it
    read (`texts`): masking around it would pass what it found.
    """
    latest: dict[str, Observed] = {
        o.lens.name: o
        for o in observed
        if o.lens.name in names and not o.aside and o.lens.subject in {subject, Subject.EXCHANGE}
    }
    for o in latest.values():
        if closed and o.errored:
            return True
        if o.answer.ok and o.answer.value:
            size = len(texts.get(o.lens.subject, ""))
            if not o.answer.spans or not all(0 <= s.start < s.end <= size for s in o.answer.spans):
                return True
    return False


def _found(o: Observed) -> bool:
    return o.answer.ok and bool(o.answer.value)


def _moved(o: Observed, offset: int, input_chars: int) -> Observed:
    """`o` from lenses that read the answer from `offset` on, its spans indexing all of it."""
    if not o.answer.spans or o.lens.subject not in {Subject.OUTPUT, Subject.EXCHANGE}:
        return o
    # An EXCHANGE lens read the question, a blank line, then the answer.
    after = input_chars + 2 if o.lens.subject is Subject.EXCHANGE else 0
    spans = [
        s.model_copy(update={"start": s.start + offset, "end": s.end + offset})
        if s.start >= after
        else s
        for s in o.answer.spans
    ]
    return Observed(o.lens, o.stage, o.answer.model_copy(update={"spans": spans}), o.failed)


@dataclass
class _Judged:
    """Text beside the turn's own, judged: as it may be passed on, and what was found."""

    text: str
    observed: list[Observed] = field(default_factory=list)
    matched: list[Matched] = field(default_factory=list)
    # Why an enforced rule refuses the turn for it, if one does.
    block: str | None = None
    # Tools an enforced rule takes away for it: None none, [] all.
    restrict: list[str] | None = None


def _history_items(
    messages: list[ChatMessage], *, history: bool
) -> list[tuple[Callable[[], str], Callable[[str], None]]]:
    """What check_history judges, each as (read it, write it back).

    Each earlier user and assistant message's text (with `history`), and the
    context of every message's attachments, the question's own included.
    """
    items: list[tuple[Callable[[], str], Callable[[str], None]]] = []
    for index, message in enumerate(messages):
        earlier = index < len(messages) - 1
        if history and earlier and message.role in {"user", "assistant"} and message.content:
            items.append(
                (
                    lambda m=message: m.content,
                    lambda text, m=message: setattr(m, "content", text),
                )
            )
        if not history and earlier:
            continue
        for attachment in message.attachments:
            if attachment.context:
                items.append(
                    (
                        lambda a=attachment: a.context,
                        lambda text, a=attachment: setattr(a, "context", text),
                    )
                )
    return items


def _stand_in(name: str) -> ResolvedPolicy:
    """An empty policy under `name`, for a turn whose own could not be loaded: it only refuses."""
    from django_ai_sdk.oversight.policy import ResolvedPolicy

    retention = dict(setting("RETENTION") or {})
    return ResolvedPolicy(
        name=name,
        capture=Capture(setting("CAPTURE")),
        retention={
            "content": int(retention.get("content", 90)),
            "metadata": int(retention.get("metadata", 730)),
        },
        origin="unloaded",
    )


async def _budget() -> budgets.Budget:
    """Today's model budget; spent when it cannot be read, so metered answerers refuse."""
    try:
        return await budgets.aload()
    except Exception:
        logger.exception("Oversight could not read today's spend: model answerers refuse")
        return budgets.Budget(0.0)


def _fail_closed(stage: Stage) -> bool:
    return (setting("FAIL_MODE") or {}).get(str(stage)) == "closed"


def spans_for(
    observed: Iterable[Observed], names: set[str], subject: Subject, input_text: str
) -> list[TextSpan]:
    """The spans lenses `names` found in `subject`, from each lens's latest answer.

    An EXCHANGE lens reads the input, a blank line and the output, so its spans are
    moved onto the part asked for and those in the other part dropped. Answers about
    text beside the turn's own (`aside`) never point into it.
    """
    latest: dict[str, Observed] = {}
    for o in observed:
        if o.lens.name in names and not o.aside:
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
    """Whether `user` may read oversight, asked only when CLIENT_DETAIL is "staff".

    Not staff when the permission cannot be read: the view shows less, never more.
    """
    if setting("CLIENT_DETAIL") != "staff" or getattr(user, "pk", None) is None:
        return False
    from django_ai_sdk import permissions

    try:
        return await permissions.has_perms(
            user,
            permissions.Operation.VIEW_OVERSIGHT,
            permissions=permissions.get_domain_permissions(permissions.PermissionDomain.OVERSIGHT),
            raise_on_deny=False,
        )
    except Exception:
        logger.exception("Oversight could not read whether a user is staff: not staff")
        return False


__all__ = [
    "FinishResult",
    "GateResult",
    "TurnSession",
    "active",
    "current_session",
    "unobserved",
]
