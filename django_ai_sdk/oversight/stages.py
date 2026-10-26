"""Running a stage: ask its lenses within a deadline, then see which rules now hold."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from django_ai_sdk.logger import get_logger
from django_ai_sdk.oversight.answerers.base import Cascade, ask, resolve, sampled
from django_ai_sdk.oversight.types import INLINE_STAGES, Answer, Mode, Outcome, Stage, worst

if TYPE_CHECKING:
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


@dataclass
class Matched:
    """A rule that held at a stage: what it asks for, and what of that was done."""

    rule: Rule
    stage: Stage
    # The actions this stage can still carry out, when the rule is enforced.
    acted: list[Action] = field(default_factory=list)

    @property
    def enforced(self) -> bool:
        return self.rule.mode is Mode.ENFORCE

    def outcome(self, *, shadow: bool = False) -> Outcome:
        """What the turn becomes from this match; with `shadow`, as if enforced."""
        actions = self.rule.then if shadow else self.acted
        return worst(*(a.outcome for a in actions))


async def answer_lens(
    lens: Lens, view: TurnView, stage: Stage, *, sample: bool = True
) -> Answer | None:
    """Ask a lens's answerers as a cascade. Reflect samples; inline stages never skip.

    `sample=False` asks regardless of the lens's sample rate: a dry run in the console.
    """
    if not lens.answerers:
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
    answerer = resolve(specs[0]) if len(specs) == 1 else Cascade(*specs)
    return await ask(answerer, lens, view)


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
    """Answer `lenses` concurrently; any still running at the deadline answers "deadline"."""
    if not lenses:
        return []
    tasks = {
        asyncio.ensure_future(answer_lens(lens, view, stage, sample=sample)): lens
        for lens in lenses
    }
    done, pending = await asyncio.wait(tasks, timeout=deadline_ms / 1000 if deadline_ms else None)
    observed: list[Observed] = []
    for task in pending:
        task.cancel()
        observed.append(Observed(tasks[task], stage, Answer(error="deadline")))
    for task in done:
        lens = tasks[task]
        try:
            answer = task.result()
        except Exception as exc:
            logger.opt(exception=exc).warning("Lens {} failed", lens.name)
            answer = Answer(error=type(exc).__name__)
        if answer is not None:
            observed.append(Observed(lens, stage, answer))
    order = {lens.name: index for index, lens in enumerate(lenses)}
    return sorted(observed, key=lambda o: order[o.lens.name])


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
    `fail_closed`, a lens that errored counts as matching every rule that reads it.
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
        holds = rule.when.evaluate(answers)
        if not holds and fail_closed and reads & set(errored):
            holds = True
        if not holds:
            continue
        acted = (
            [a for a in rule.ordered_actions() if stage in a.effective]
            if rule.mode is Mode.ENFORCE
            else []
        )
        matched.append(Matched(rule, stage, acted))
    return matched


__all__ = ["Matched", "Observed", "answer_lens", "evaluate_rules", "run_lenses", "still_to_answer"]
