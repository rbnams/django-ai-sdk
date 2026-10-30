"""Running a stage: ask its lenses within a deadline, then see which rules now hold."""

from __future__ import annotations

import asyncio
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from django_ai_sdk.logger import get_logger
from django_ai_sdk.oversight.answerers.base import ask, is_unavailable, resolve, sampled
from django_ai_sdk.oversight.types import (
    INLINE_STAGES,
    Answer,
    Mode,
    Outcome,
    Stage,
    Subject,
    worst,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Set as AbstractSet

    from django_ai_sdk.oversight.answerers.base import Answerer
    from django_ai_sdk.oversight.lenses import Lens
    from django_ai_sdk.oversight.policy import ResolvedPolicy
    from django_ai_sdk.oversight.rules import Action, Rule
    from django_ai_sdk.oversight.types import TurnView

logger = get_logger(__name__)

STAGE_ORDER = list(Stage)


@dataclass
class Observed:
    lens: Lens
    stage: Stage
    answer: Answer
    # The chain ended on an error or the deadline after only unsure answers: the
    # lens failed for FAIL_MODE, and the unsure answer is kept for the record.
    failed: bool = False
    # About text beside the turn's own (an earlier message, a tool's result): recorded,
    # but never what a mask of the question or the answer points with.
    aside: bool = False

    @property
    def errored(self) -> bool:
        """Whether the lens failed here; one waiting for a person has not."""
        return self.failed or bool(self.answer.error and not self.answer.queued)


# What a failed lens may trigger under FAIL_MODE "closed": actions that protect.
FAIL_CLOSED_ACTIONS = frozenset({"block", "mask", "restrict_tools"})


@dataclass
class Matched:
    """A rule that held at a stage: what it asks for, and what of that was done."""

    rule: Rule
    stage: Stage
    # The actions this stage can still carry out, when the rule is enforced.
    acted: list[Action] = field(default_factory=list)
    # Held on text beside the turn's own (an earlier message, a tool's result): it
    # counts for the turn's outcome, but leaves the rule to judge the turn itself.
    aside: bool = False

    def outcome(self, *, report: bool = False) -> Outcome:
        """What the turn becomes from this match; with `report`, as if enforced."""
        actions = self.rule.then if report else self.acted
        return worst(*(a.outcome for a in actions))


async def answer_lens(
    lens: Lens,
    view: TurnView,
    stage: Stage,
    *,
    sample: bool = True,
    progress: list[Answer] | None = None,
) -> Answer | None:
    """Ask a lens's answerers in order until one is sure. Reflect samples; inline stages never skip.

    `sample=False` asks regardless of the lens's sample rate: a dry run in the console.
    `progress` collects every answer as it comes, for a caller that may cancel.
    """
    if not lens.answerers or not lens.applies(view):
        return None
    if (
        sample
        and stage not in INLINE_STAGES
        and not sampled(f"{view.turn_id}:{lens.name}", lens.sample)
    ):
        return None
    specs = usable_answerers(lens)
    if not specs:
        return None
    return await _first_sure(specs, lens, view, progress if progress is not None else [])


async def _first_sure(
    specs: list[Any], lens: Lens, view: TurnView, asked: list[Answer]
) -> Answer | None:
    """The first answer sure enough for the lens (`min_confidence`), else the last usable one.

    Each answerer's confidence is on a scale of its own, so they are never compared:
    when none is sure enough, the last one that answered decides. A lens lists its
    answerers from quick to thorough, so that is the one it trusts most.

    Every answer given on the way is in the result's `trail`, in the order asked. A
    person's answer is still to come, so the queue it was asked from carries over.
    """
    for spec in specs:
        result = await ask(spec, lens, view)
        if result is None:
            continue
        if result.ok and result.confidence >= lens.min_confidence:
            result.trail = list(asked)
            asked.append(result)
            return result
        asked.append(result)
    final = next((a for a in reversed(asked) if a.ok), None)
    if final is None:
        return asked[-1] if asked else None
    update: dict[str, Any] = {"trail": [a for a in asked if a is not final]}
    queued = next((a.queued for a in asked if a.queued), "")
    if queued:
        update["queued"] = queued
    return final.model_copy(update=update)


def _ended_unsure(lens: Lens, answer: Answer | None, asked: list[Answer]) -> bool:
    """Whether the chain ended on an error after only unsure answers, `answer` the last of them."""
    if answer is None or not answer.ok or answer.confidence >= lens.min_confidence:
        return False
    last = asked[-1] if asked else None
    return last is not None and bool(last.error) and not last.queued


def asks_again(error: str, last_error: str = "") -> bool:
    """Whether reflect should ask a lens again: the deadline cut it short or a service was down.

    `last_error` is the error at the end of the answer's trail: a usable answer kept
    at the deadline ends its trail with "deadline", one kept while a later answerer's
    service was down with that answerer's error.
    """
    return any(e == "deadline" or is_unavailable(e) for e in (error, last_error))


def usable_answerers(lens: Lens) -> list[Any]:
    """The lens's answerers, without optional aliases ("judge?") the settings don't define.

    Packs name model answerers optionally, so they work on a deployment with no
    model configured and get better on one with.
    """
    from django_ai_sdk.oversight.answerers.base import alias_defined

    usable = []
    for spec in lens.answerers:
        if isinstance(spec, str) and spec.endswith("?"):
            if not alias_defined(spec[:-1]):
                continue
            spec = spec[:-1]
        usable.append(spec)
    return usable


async def run_lenses(
    lenses: list[Lens],
    view: TurnView,
    stage: Stage,
    *,
    deadline_ms: float | None = None,
    sample: bool = True,
) -> list[Observed]:
    """Answer `lenses` concurrently; any still running at the deadline answers "deadline".

    A lens cut off after a usable but unsure answer keeps that answer, with "deadline"
    last in its trail, so reflect still asks the slower answerers; it counts as failed.
    Whatever is still running when the stage ends, or is cancelled, is cancelled and
    awaited: no answerer outlives its stage.
    """
    if not lenses:
        return []
    tasks: dict[asyncio.Future, tuple[Lens, list[Answer]]] = {}
    with ExitStack() as stack:
        for batched in _batching(lenses):
            stack.enter_context(batched(deadline_ms))
        for lens in lenses:
            progress: list[Answer] = []
            task = asyncio.ensure_future(
                answer_lens(lens, view, stage, sample=sample, progress=progress)
            )
            tasks[task] = (lens, progress)
    try:
        done, pending = await asyncio.wait(
            tasks, timeout=deadline_ms / 1000 if deadline_ms else None
        )
    except asyncio.CancelledError:
        await _cancel(tasks)
        raise
    await _cancel(pending)
    observed: list[Observed] = []
    for task in pending:
        lens, progress = tasks[task]
        answer = _at_deadline(progress)
        # The deadline ended its chain: "deadline" is last in the trail.
        failed = _ended_unsure(lens, answer, answer.trail)
        observed.append(Observed(lens, stage, answer, failed=failed))
    for task in done:
        lens, progress = tasks[task]
        try:
            answer = task.result()
        except Exception as exc:
            logger.opt(exception=exc).warning("Lens {} failed", lens.name)
            answer = Answer(error=type(exc).__name__)
        if answer is not None:
            failed = _ended_unsure(lens, answer, progress)
            observed.append(Observed(lens, stage, answer, failed=failed))
    order = {lens.name: index for index, lens in enumerate(lenses)}
    return sorted(observed, key=lambda o: order[o.lens.name])


async def _cancel(tasks: Iterable[asyncio.Future]) -> None:
    """Cancel `tasks` and wait until they have stopped."""
    running = [task for task in tasks if not task.done()]
    for task in running:
        task.cancel()
    if running:
        await asyncio.gather(*running, return_exceptions=True)


def _batching(lenses: list[Lens]) -> list[Any]:
    """The `batched(deadline_ms)` of each answerer these lenses may ask, where it has one.

    Inside it, such an answerer collects what the stage's lenses ask of it and may
    send that together (a Classifier: one call per state).
    """
    found: list[Any] = []
    for lens in lenses:
        for spec in usable_answerers(lens):
            try:
                batched = getattr(resolve(spec), "batched", None)
            except Exception:
                continue
            if batched is not None and batched not in found:
                found.append(batched)
    return found


def _at_deadline(asked: list[Answer]) -> Answer:
    """The last usable answer before the deadline, "deadline" last in its trail."""
    cut = Answer(error="deadline")
    kept = next((a for a in reversed(asked) if a.ok), None)
    if kept is None:
        return cut
    return kept.model_copy(update={"trail": [*(a for a in asked if a is not kept), cut]})


def lens_health(lens: Lens) -> dict[str, Any]:
    """Whether `lens` can run on this deployment, and what in it never will.

    runnable: some answerer is usable here (one with nothing `missing()`: a service
    only with its key and package).
    skipped: optional aliases that are not defined. unreachable: answerers after a
    detector whose "found nothing" is already sure enough to stop at, and
    unreachable_from the index in `lens.answerers` they start at. misplaced: a lens
    on the question at watch or finish, which only ever see the answer.
    """
    from django_ai_sdk.oversight.answerers.base import alias_defined
    from django_ai_sdk.oversight.answerers.rules import Detector

    skipped: list[str] = []
    built: list[tuple[int, Any, Answerer]] = []
    for index, spec in enumerate(lens.answerers):
        if isinstance(spec, str) and spec.endswith("?") and not alias_defined(spec[:-1]):
            skipped.append(spec)
            continue
        try:
            built.append((index, spec, resolve(spec)))
        except Exception:
            continue
    runnable = any(not a.missing() for _, _, a in built)
    stop = next(
        (
            n
            for n, (_, _, a) in enumerate(built[:-1])
            if isinstance(a, Detector) and a.miss_confidence >= lens.min_confidence
        ),
        None,
    )
    after = built[stop + 1 :] if stop is not None else []
    return {
        "runnable": runnable,
        "skipped": skipped,
        "unreachable": [_spec_name(spec) for _, spec, _ in after],
        "unreachable_from": after[0][0] if after else None,
        "misplaced": misplaced(lens),
    }


def misplaced(lens: Lens) -> bool:
    """A lens on the question at watch or finish: those stages only run answer lenses."""
    return lens.subject is Subject.INPUT and bool({Stage.WATCH, Stage.FINISH} & set(lens.stages))


def _spec_name(spec: Any) -> str:
    if isinstance(spec, str):
        return spec
    if isinstance(spec, dict):
        return str(spec.get("name") or spec.get("class") or "")
    return spec.label()


def still_to_answer(policy: ResolvedPolicy, name: str, stage: Stage, answered: set[str]) -> bool:
    """Whether lens `name` may yet answer at a later stage of this turn."""
    lens = policy.lenses.get(name)
    if lens is None or not lens.enabled or name in answered:
        return False
    here = STAGE_ORDER.index(stage)
    return any(STAGE_ORDER.index(s) > here for s in lens.stages)


def evaluate_rules(
    policy: ResolvedPolicy,
    answers: dict[str, Answer],
    stage: Stage,
    *,
    done: set[str],
    errored: AbstractSet[str] = frozenset(),
    fail_closed: bool = False,
) -> list[Matched]:
    """Rules that hold now and have not held before in this turn.

    A rule without `stages` waits until every lens it reads has had its chance,
    so a gate-only lens and a reflect-only lens can share a rule. `errored` names
    the lenses that failed this turn, at this stage or an earlier one: a test on
    one of them is unknown, never true (`Condition.evaluate`), unless it asks
    whether the lens answered. With `fail_closed`, a lens that errored counts as
    matching an enforced rule that reads it, if that rule would block, mask or
    take tools away here. A rule that only alerts, queues or notifies is never
    raised by a slow answerer.
    """
    answered = set(answers) | set(errored)
    matched: list[Matched] = []
    for rule in policy.active_rules():
        if rule.name in done:
            continue
        reads = rule.lenses()
        if rule.stages:
            if stage not in rule.stages:
                continue
        elif any(still_to_answer(policy, name, stage, answered) for name in reads):
            continue
        acted = (
            [a for a in rule.ordered_actions() if stage in a.effective]
            if rule.mode is Mode.ENFORCE
            else []
        )
        if rule.mode is Mode.ENFORCE and stage is Stage.WATCH and _reads_tool_call(policy, reads):
            # A tool call cannot be masked, only run or not: a mask over one refuses it.
            if any(a.kind == "mask" for a in rule.then) and not any(
                a.kind == "block" for a in acted
            ):
                acted.append(_block(rule.message))
        holds = rule.when.evaluate(answers, errored) or (
            fail_closed
            and bool(reads & set(errored))
            and any(a.kind in FAIL_CLOSED_ACTIONS for a in acted)
        )
        if holds:
            matched.append(Matched(rule, stage, acted))
    return matched


def _reads_tool_call(policy: ResolvedPolicy, names: AbstractSet[str]) -> bool:
    return any(
        (lens := policy.lenses.get(name)) is not None and lens.subject is Subject.TOOL_CALL
        for name in names
    )


def _block(message: str) -> Action:
    from django_ai_sdk.oversight.rules import Block

    return Block(message)


__all__ = [
    "Matched",
    "Observed",
    "answer_lens",
    "asks_again",
    "evaluate_rules",
    "lens_health",
    "misplaced",
    "run_lenses",
    "still_to_answer",
]
