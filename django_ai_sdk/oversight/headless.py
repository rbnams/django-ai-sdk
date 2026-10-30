"""One headless run as one oversight turn: what `Agent.run()` does around the work.

Chat streams its turn through the Stream adapter; everything else that runs an
agent (the HTTP run endpoints, webhooks, workflow steps, your own code) goes
through `run_turn`, so a policy cannot be walked around by picking another door.

Each agent runs under its own policy. A run inside another turn joins that turn
when it resolves the same policy, or none; under a policy of its own it opens a
child turn (its own gate, watch, finish and record), linked to the outer one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ValidationError

from django_ai_sdk.errors import Blocked
from django_ai_sdk.logger import get_logger
from django_ai_sdk.oversight.rules import Block
from django_ai_sdk.oversight.session import TurnSession, active, current_session, unobserved
from django_ai_sdk.oversight.types import Mode, Stage, Subject

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from django.contrib.auth.base_user import AbstractBaseUser
    from django.contrib.auth.models import AnonymousUser

    from django_ai_sdk.common import ChatMessage
    from django_ai_sdk.oversight.policy import Policy
    from django_ai_sdk.storage.base import BaseStorageAdapter

logger = get_logger(__name__)


async def run_turn(
    agent: Any,
    work: Callable[[list[ChatMessage]], Awaitable[Any]],
    messages: list[ChatMessage],
    *,
    channel: str,
    user: AbstractBaseUser | AnonymousUser | None = None,
    thread_id: str | None = None,
    oversight: type[Policy] | str | Literal[False] | None = None,
) -> Any:
    """`work(messages)` as one turn of `agent`, under its policy or `oversight`.

    `oversight`: None takes the agent's own (the console's choice, else its code);
    a Policy subclass or a name replaces it; "" means none of its own; False runs
    the work unwatched, even inside another turn.

    History is judged before the question, unless a stored thread vouches for
    it (`thread_history`): with a `thread_id` whose thread is stored, under a
    policy that masks at the gate, the stored messages are the history, as in
    chat; a history that is the stored thread as it was stored is not judged
    again. Anything else the caller sent is judged by the gate's lenses
    (`check_history`). The caller's list is not rewritten: history is
    judged on copies, and only the last user message is masked in place, so what
    the caller stores afterwards is the masked copy. Tool calls are watched while
    the work runs, and the answer is finished before it is returned. Raises
    `Blocked` when the gate or the finish stops it.
    """
    if oversight is False:
        with active(None), unobserved():
            return await work(messages)
    session = await open_turn(
        agent,
        channel=channel,
        user=user,
        thread_id=thread_id,
        messages=messages,
        oversight=oversight,
    )
    if session is None:
        return await work(messages)
    error = ""
    try:
        messages = _working_copy(messages)
        storage = await _thread_storage(agent, thread_id)
        vouched = await thread_history(session, storage, messages) if storage else None
        if vouched is not None:
            messages = vouched
            session.set_history(messages)
        else:
            checked = await session.check_history(messages)
            if checked.blocked:
                raise Blocked(checked.message, stage=str(Stage.GATE), turn_id=session.id)
        gate = await session.gate()
        if gate.blocked:
            raise Blocked(gate.message, stage=str(Stage.GATE), turn_id=session.id)
        if gate.masked:
            _mask_question(messages, gate.input)
        with active(session):
            result = await work(messages)
        if result is None:
            error = "no result"
            return None
        return await _finish(session, result)
    except Blocked:
        raise
    except BaseException as exc:
        error = type(exc).__name__
        raise
    finally:
        await session.close(error=error)


async def open_turn(
    agent: Any,
    *,
    channel: str,
    user: AbstractBaseUser | AnonymousUser | None = None,
    thread_id: str | None = None,
    messages: list[ChatMessage] | None = None,
    input: str | None = None,
    oversight: type[Policy] | str | None = None,
) -> TurnSession | None:
    """The turn a run of `agent` opens, or None when it joins the current one or none applies.

    Inside another turn, a run that resolves the same policy, or none, joins it.
    One under a policy of its own opens a child turn, linked to the outer one by
    `parent_turn_id`. `oversight` as for `run_turn` (not False).
    """
    outer = current_session()
    if oversight == "":
        return None
    session = await TurnSession.start(
        agent,
        channel=channel,
        user=user,
        thread_id=thread_id,
        messages=messages,
        input=input,
        oversight=oversight,
    )
    if not session.active:
        return None
    if outer is not None:
        if _same_policy(outer, session):
            return None
        session.parent_turn_id = outer.id
    return session


def _same_policy(outer: TurnSession, inner: TurnSession) -> bool:
    return (
        outer.policy is not None
        and inner.policy is not None
        and outer.policy.name == inner.policy.name
    )


def takes_stored_history(session: TurnSession) -> bool:
    """Whether a turn with a stored thread takes the thread's messages as its history.

    Only under a policy that masks the question at the gate: the stored copy is
    the masked one, and each message in it was judged on its own turn. Under any
    other policy what the client sent stays the history, judged by `check_history`,
    so a policy in report mode changes nothing.
    """
    policy = session.policy
    if not session.active or policy is None:
        return False
    masking = {
        name
        for rule in policy.rules.values()
        if rule.mode is Mode.ENFORCE and any(action.kind == "mask" for action in rule.then)
        for name in rule.lenses()
    }
    return any(
        lens.kind == "entities" and lens.name in masking
        for lens in policy.active_lenses(Stage.GATE, Subject.INPUT)
    )


async def thread_history(
    session: TurnSession, storage_adapter: BaseStorageAdapter, messages: list[ChatMessage]
) -> list[ChatMessage] | None:
    """The history a stored thread vouches for, or None when what was sent must be judged.

    Under a policy that masks at the gate, the stored messages replace the sent
    copy (`stored_history`). Otherwise a sent history that is the stored thread as
    it was stored (role and text) was judged turn by turn and is kept; one that
    differs (edited, made up) is not vouched for. Chat and the run endpoints share
    this.
    """
    if takes_stored_history(session):
        return await stored_history(storage_adapter, messages)
    if await _as_stored(storage_adapter, messages):
        return messages
    return None


async def stored_history(
    storage_adapter: BaseStorageAdapter, messages: list[ChatMessage]
) -> list[ChatMessage]:
    """The stored thread as the history of `messages`, then their new question.

    What a client sends back is its own copy: unmasked where the gate masked, or
    made up. Earlier messages that carry the ids of stored ones are taken from the
    store; otherwise the stored messages are taken as far as the client's copy goes
    (so a regenerated or edited turn still drops what came after). Only user and
    assistant messages count. The new question is kept as it is.
    """
    newest = messages[-1] if messages and messages[-1].role == "user" else None
    earlier = messages[:-1] if newest is not None else list(messages)
    stored = [m for m in await storage_adapter.get_messages() if m.role in ("user", "assistant")]
    by_id = {m.id: m for m in stored if m.id}
    if earlier and all(m.id and m.id in by_id for m in earlier):
        kept = [by_id[m.id] for m in earlier]
    else:
        kept = stored[: len(earlier)]
    return [*kept, newest] if newest is not None else kept


async def _as_stored(storage_adapter: BaseStorageAdapter, messages: list[ChatMessage]) -> bool:
    """Whether every message before the question is one the thread stored, as stored."""
    newest = bool(messages) and messages[-1].role == "user"
    earlier = messages[:-1] if newest else list(messages)
    if not earlier:
        return True
    stored = [m for m in await storage_adapter.get_messages() if m.role in ("user", "assistant")]
    by_id = {m.id: m for m in stored if m.id}
    if all(m.id and m.id in by_id for m in earlier):
        kept = [by_id[m.id] for m in earlier]
    elif len(earlier) <= len(stored):
        kept = stored[: len(earlier)]
    else:
        return False
    return all(
        k.role == m.role and (k.content or "") == (m.content or "")
        for k, m in zip(kept, earlier, strict=True)
    )


async def _thread_storage(agent: Any, thread_id: str | None) -> BaseStorageAdapter | None:
    """The storage holding `thread_id`'s stored thread; None when no thread is stored."""
    get_storage = getattr(agent, "get_storage_adapter", None)
    if not thread_id or get_storage is None:
        return None
    try:
        storage = await get_storage(thread_id)
        if storage is not None and await type(storage).get_thread(thread_id):
            return storage
    except Exception:
        # No stored thread to vouch for the history: it is judged as sent.
        logger.opt(exception=True).debug("No stored thread {}", thread_id)
    return None


def _working_copy(messages: list[ChatMessage]) -> list[ChatMessage]:
    """A list to judge and mask: copies of every message but the last user message.

    That one is masked in place, so the caller's copy of the question is the
    masked one; the rest of the caller's list stays as it was.
    """
    last = next((i for i in range(len(messages) - 1, -1, -1) if messages[i].role == "user"), -1)
    return [m if i == last else m.model_copy(deep=True) for i, m in enumerate(messages)]


def _mask_question(messages: list[ChatMessage], masked: str) -> None:
    for message in reversed(messages):
        if message.role == "user":
            message.content = masked
            return


async def _finish(session: TurnSession, result: Any) -> Any:
    """The result as the policy lets it out: masked, or Blocked."""
    if isinstance(result, BaseModel):
        text = result.model_dump_json()
    else:
        text = result if isinstance(result, str) else str(result)
    finished = await session.finish(text)
    if finished.blocked:
        raise Blocked(finished.output, stage=str(Stage.FINISH), turn_id=session.id)
    if not finished.replaced:
        return result
    if isinstance(result, str):
        return finished.output
    rebuilt = _rebuilt(result, finished.output)
    if rebuilt is None:
        # A masked copy that no longer fits the schema must not pass as the original.
        session.refuse("masked result does not fit its schema")
        raise Blocked(str(Block.DEFAULT_MESSAGE), stage=str(Stage.FINISH), turn_id=session.id)
    return rebuilt


def _rebuilt(result: Any, output: str) -> BaseModel | None:
    """A structured result rebuilt from its masked JSON; None when it cannot be."""
    if not isinstance(result, BaseModel):
        return None
    try:
        return type(result).model_validate_json(output)
    except ValidationError:
        return None


__all__ = ["open_turn", "run_turn", "stored_history", "takes_stored_history", "thread_history"]
