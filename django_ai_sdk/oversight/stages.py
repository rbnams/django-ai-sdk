"""Running a stage: ask its lenses within a deadline, then see which rules now hold."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from django_ai_sdk.logger import get_logger
from django_ai_sdk.oversight.answerers.base import ask, resolve, sampled
from django_ai_sdk.oversight.answerers.typesafe import batched
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


# What a failed lens may trigger under FAIL_MODE "closed": actions that protect.
FAIL_CLOSED_ACTIONS = frozenset({"block", "redact", "restrict_tools"})


@dataclass
class Matched:
    """A rule that held at a stage: what it asks for, and what of that was done."""

    rule: Rule
    stage: Stage
    # The actions this stage can still carry out, when the rule is enforced.
    acted: list[Action] = field(default_factory=list)

    def outcome(self, *, shadow: bool = False) -> Outcome:
        """What the turn becomes from this match; with `shadow`, as if enforced."""
        actions = self.rule.then if shadow else self.acted
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
    `progress` collects each usable answer as it comes, for a caller that may cancel.
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
    specs: list[Any], lens: Lens, view: TurnView, progress: list[Answer]
) -> Answer | None:
    """The first answer sure enough for the lens (`confident`), else the surest usable one.

    When no answerer is sure enough, the most confident answer counts (the later on a
    tie): a fallback that is less sure than the first should not overrule it.

    Every answer given on the way is in the result's `trail`. A person's answer is
    still to come, so the queue it was asked from carries over to the result.
    """
    trail: list[Answer] = []
    for spec in specs:
        result = await ask(resolve(spec), lens, view)
        if result is None:
            continue
        if result.ok:
            progress.append(result)
            if result.confidence >= lens.confident:
                result.trail = trail
                return result
        trail.append(result)
    usable = [a for a in trail if a.ok]
    if not usable:
        return trail[-1] if trail else None
    final = max(reversed(usable), key=lambda a: a.confidence)
    update: dict[str, Any] = {"trail": [a for a in trail if a is not final]}
    queued = next((a.queued for a in trail if a.queued), "")
    if queued:
        update["queued"] = queued
    return final.model_copy(update=update)


def asks_again(error: str, last_error: str = "") -> bool:
    """Whether reflect should ask a lens again: the deadline cut it short or Jev was down.

    `last_error` is the error at the end of the answer's trail: a usable answer kept
    at the deadline ends its trail with "deadline".
    """
    return error == "deadline" or error.startswith("jev: ") or last_error == "deadline"


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
    last in its trail, so reflect still asks the slower answerers.
    """
    if not lenses:
        return []
    tasks: dict[asyncio.Future, tuple[Lens, list[Answer]]] = {}
    with batched(deadline_ms):
        for lens in lenses:
            progress: list[Answer] = []
            task = asyncio.ensure_future(
                answer_lens(lens, view, stage, sample=sample, progress=progress)
            )
            tasks[task] = (lens, progress)
    done, pending = await asyncio.wait(tasks, timeout=deadline_ms / 1000 if deadline_ms else None)
    observed: list[Observed] = []
    for task in pending:
        task.cancel()
        lens, progress = tasks[task]
        observed.append(Observed(lens, stage, _at_deadline(progress)))
    for task in done:
        lens, _ = tasks[task]
        try:
            answer = task.result()
        except Exception as exc:
            logger.opt(exception=exc).warning("Lens {} failed", lens.name)
            answer = Answer(error=type(exc).__name__)
        if answer is not None:
            observed.append(Observed(lens, stage, answer))
    order = {lens.name: index for index, lens in enumerate(lenses)}
    return sorted(observed, key=lambda o: order[o.lens.name])


def _at_deadline(progress: list[Answer]) -> Answer:
    cut = Answer(error="deadline")
    if not progress:
        return cut
    return progress[-1].model_copy(update={"trail": [*progress[:-1], cut]})


def lens_health(lens: Lens) -> dict[str, Any]:
    """Whether `lens` can run on this deployment, and what in it never will.

    runnable: some answerer is usable here (Jev counts only with its key and package).
    skipped: optional aliases that are not defined. unreachable: answerers after a
    detector whose "found nothing" is already sure enough to stop at, and
    unreachable_from the index in `lens.answerers` they start at. misplaced: a lens
    on the question at watch or review, which only ever see the answer.
    """
    from django_ai_sdk.oversight.answerers.base import alias_defined
    from django_ai_sdk.oversight.answerers.rules import Detector
    from django_ai_sdk.oversight.answerers.typesafe import TypeSafe
    from django_ai_sdk.typesafe.client import typesafe_missing

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
    runnable = any(not (isinstance(a, TypeSafe) and typesafe_missing()) for _, _, a in built)
    stop = next(
        (
            n
            for n, (_, _, a) in enumerate(built[:-1])
            if isinstance(a, Detector) and a.miss_confidence >= lens.confident
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
    """A lens on the question at watch or review: those stages only run answer lenses."""
    return lens.subject is Subject.INPUT and bool({Stage.WATCH, Stage.REVIEW} & set(lens.stages))


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
    errored: set[str] = frozenset(),
    fail_closed: bool = False,
) -> list[Matched]:
    """Rules that hold now and have not held before in this turn.

    A rule without `stages` waits until every lens it reads has had its chance,
    so a gate-only lens and a reflect-only lens can share a rule. With
    `fail_closed`, a lens that errored counts as matching an enforced rule that
    reads it, if that rule would block, redact or take tools away here. A rule
    that only alerts, queues or notifies is never raised by a slow answerer.
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
        holds = rule.when.evaluate(answers) or (
            fail_closed
            and bool(reads & set(errored))
            and any(a.kind in FAIL_CLOSED_ACTIONS for a in acted)
        )
        if holds:
            matched.append(Matched(rule, stage, acted))
    return matched


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
