"""Answerers: whoever or whatever answers a lens, and the ways to combine them."""

from __future__ import annotations

import asyncio
import hashlib
import time
from abc import ABC, abstractmethod
from collections import Counter
from typing import TYPE_CHECKING, Any, ClassVar

from django.core.exceptions import ImproperlyConfigured
from django.utils.module_loading import import_string

from django_ai_sdk.logger import get_logger
from django_ai_sdk.oversight.types import Answer

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from django_ai_sdk.oversight.lenses import Lens
    from django_ai_sdk.oversight.types import TurnView

logger = get_logger(__name__)


class Answerer(ABC):
    """Answers lenses. Return None to abstain: the next answerer, if any, is asked."""

    # Deterministic, local and fast: may run at the gate without a deadline worry,
    # costs nothing, needs no budget.
    system_one: ClassVar[bool] = False
    # Seconds before the answer is abandoned.
    timeout: float = 30.0
    name: str = ""

    @abstractmethod
    async def answer(self, lens: Lens, view: TurnView) -> Answer | None: ...

    def label(self) -> str:
        return self.name or type(self).__name__.lower()

    def describe(self) -> dict[str, Any]:
        return {"name": self.label(), "system_one": self.system_one, "class": type(self).__name__}

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.label()!r}>"


async def ask(answerer: Answerer, lens: Lens, view: TurnView) -> Answer | None:
    """Ask one answerer, timed, typed and contained: a failure is an Answer with `error`.

    A model answerer is charged to the active Budget, and refused once it is spent.
    """
    from django_ai_sdk.oversight import budget

    started = time.perf_counter()
    spend = None if answerer.system_one or _combinator(answerer) else budget.current()
    if spend is not None and not spend.allows():
        return Answer(answerer=answerer.label(), error="budget")
    try:
        answer = await asyncio.wait_for(answerer.answer(lens, view), timeout=answerer.timeout)
    except TimeoutError:
        return Answer(answerer=answerer.label(), error="timeout", latency_ms=_ms(started))
    except Exception as exc:
        logger.opt(exception=exc).warning(
            "Answerer {} failed on lens {}", answerer.label(), lens.name
        )
        return Answer(answerer=answerer.label(), error=type(exc).__name__, latency_ms=_ms(started))
    if answer is None:
        return None
    if spend is not None:
        spend.charge(answerer.label(), answer.cost)
    if answer.value is not None and not answer.error:
        try:
            answer.value = lens.parse(answer.value)
        except (TypeError, ValueError) as exc:
            answer = answer.model_copy(update={"value": None, "error": f"unparsable: {exc}"})
    answer.answerer = answer.answerer or answerer.label()
    if answer.latency_ms is None:
        answer.latency_ms = _ms(started)
    return answer


def _combinator(answerer: Answerer) -> bool:
    # Combinators ask through `ask` themselves; charging them too would count twice.
    return isinstance(answerer, Cascade | Vote | Sample)


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)


class Cascade(Answerer):
    """Ask in order and stop at the first sure answer. Cheap first, judges only when needed.

    An answer is sure when its confidence reaches `confident` (else the lens's own).
    The last answer stands either way, with every earlier one in its `trail`.
    """

    def __init__(
        self, *answerers: Answerer | str, confident: float | None = None, name: str = ""
    ) -> None:
        self.answerers = list(answerers)
        self.confident = confident
        self.name = name
        self.timeout = 600.0

    @property
    def system_one(self) -> bool:  # ty: ignore[invalid-method-override]
        return all(resolve(a).system_one for a in self.answerers)

    def label(self) -> str:
        return self.name or "cascade(" + ",".join(resolve(a).label() for a in self.answerers) + ")"

    async def answer(self, lens: Lens, view: TurnView) -> Answer | None:
        threshold = self.confident if self.confident is not None else lens.confident
        trail: list[Answer] = []
        for spec in self.answerers:
            result = await ask(resolve(spec), lens, view)
            if result is None:
                continue
            if result.ok and result.confidence >= threshold:
                result.trail = trail
                return result
            trail.append(result)
        usable = [a for a in trail if a.ok]
        if not usable:
            return trail[-1] if trail else None
        final = usable[-1]
        return final.model_copy(update={"trail": [a for a in trail if a is not final]})


class Vote(Answerer):
    """Ask every answerer at once and take the most common value.

    Confidence is the share that agreed, so disagreement between models is visible
    on the ledger instead of averaged away. `min_agree` votes or the vote abstains.
    """

    def __init__(self, *answerers: Answerer | str, min_agree: int = 1, name: str = "") -> None:
        self.answerers = list(answerers)
        self.min_agree = min_agree
        self.name = name
        self.timeout = 600.0

    def label(self) -> str:
        return self.name or "vote(" + ",".join(resolve(a).label() for a in self.answerers) + ")"

    async def answer(self, lens: Lens, view: TurnView) -> Answer | None:
        results = await asyncio.gather(*(ask(resolve(a), lens, view) for a in self.answerers))
        answers = [r for r in results if r is not None]
        usable = [a for a in answers if a.ok]
        if not usable:
            return answers[0] if answers else None
        counts = Counter(_vote_key(a.value) for a in usable)
        key, votes = counts.most_common(1)[0]
        if votes < self.min_agree:
            return Answer(error="no agreement", trail=answers, answerer=self.label())
        winner = next(a for a in usable if _vote_key(a.value) == key)
        return Answer(
            value=winner.value,
            confidence=round(votes / len(usable), 3),
            rationale=winner.rationale,
            spans=winner.spans,
            answerer=self.label(),
            cost=_sum(a.cost for a in answers),
            tokens=_sum(a.tokens for a in answers),
            trail=answers,
        )


def _vote_key(value: Any) -> Any:
    if isinstance(value, float):
        # Scores vote by band, or no two judges would ever agree.
        return round(value * 4) / 4
    if isinstance(value, list):
        return tuple(sorted(value))
    return value


def _sum(values: Any) -> Any:
    present = [v for v in values if v is not None]
    return sum(present) if present else None


class Sample(Answerer):
    """Ask `answerer` on a stable `rate` share of turns; abstain on the rest.

    Stable per turn and lens: a replay samples the same turns.
    """

    def __init__(self, answerer: Answerer | str, rate: float, name: str = "") -> None:
        self.inner = answerer
        self.rate = rate
        self.name = name

    def label(self) -> str:
        return self.name or f"sample({resolve(self.inner).label()},{self.rate})"

    async def answer(self, lens: Lens, view: TurnView) -> Answer | None:
        if not sampled(f"{view.turn_id}:{lens.name}", self.rate):
            return None
        return await ask(resolve(self.inner), lens, view)


def sampled(key: str, rate: float) -> bool:
    """Whether `key` falls in the first `rate` share of a uniform hash space."""
    if rate >= 1:
        return True
    if rate <= 0:
        return False
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < rate


class Function(Answerer):
    """Any function as an answerer: `fn(lens, view)` returns a value, an Answer, or None.

    Sync or async. Counted as system one unless `system_one=False` says otherwise.
    """

    def __init__(
        self,
        fn: Callable[[Lens, TurnView], Any | Awaitable[Any]] | str,
        *,
        name: str = "",
        system_one: bool = True,
    ) -> None:
        if isinstance(fn, str):
            # A dotted path, so a lens defined in the console can name host code.
            fn = import_string(fn)
        self.fn = fn
        self.name = name or getattr(fn, "__name__", "function")
        self._system_one = system_one

    @property
    def system_one(self) -> bool:  # ty: ignore[invalid-method-override]
        return self._system_one

    async def answer(self, lens: Lens, view: TurnView) -> Answer | None:
        result = self.fn(lens, view)
        if asyncio.iscoroutine(result):
            result = await result
        if result is None or isinstance(result, Answer):
            return result
        return Answer(value=result)


def resolve(spec: Answerer | str | dict[str, Any]) -> Answerer:
    """An Answerer from an instance, an alias in ANSWERERS, or a config dict.

    A config dict is {"class": dotted path, **kwargs}; aliases may name aliases.
    """
    if isinstance(spec, Answerer):
        return spec
    if isinstance(spec, str):
        return _from_alias(spec.removesuffix("?"))
    if isinstance(spec, dict):
        return _from_config(spec)
    raise ImproperlyConfigured(f"{spec!r} is not an answerer, an alias or a config")


# Aliases defined in the console (PolicyOverride rows on the "*" policy). Loaded
# with the policies, so a sync resolve sees what the last async load found.
_console_aliases: dict[str, Any] = {}


def set_console_aliases(aliases: dict[str, Any]) -> None:
    global _console_aliases
    _console_aliases = dict(aliases)


def alias_table() -> dict[str, Any]:
    """Every alias: AI_SDK_OVERSIGHT["ANSWERERS"], with console definitions on top."""
    from django_ai_sdk.oversight.conf import setting

    return {**setting("ANSWERERS"), **_console_aliases}


def alias_source(alias: str) -> str:
    """Where an alias is defined: "console", "settings", "builtin" or ""."""
    from django_ai_sdk.oversight.conf import setting

    if alias in _console_aliases:
        return "console"
    if alias in setting("ANSWERERS"):
        return "settings"
    return "builtin" if alias in BUILTIN_ALIASES else ""


def alias_defined(alias: str) -> bool:
    return alias in alias_table() or alias in BUILTIN_ALIASES


def _from_alias(alias: str, seen: frozenset[str] = frozenset()) -> Answerer:
    if alias in seen:
        raise ImproperlyConfigured(f"Answerer alias {alias!r} names itself")
    aliases = alias_table()
    if alias not in aliases:
        builtin = BUILTIN_ALIASES.get(alias)
        if builtin is None:
            raise ImproperlyConfigured(
                f"No answerer alias {alias!r}; add it to AI_SDK_OVERSIGHT['ANSWERERS']"
            )
        return builtin()
    target = aliases[alias]
    if isinstance(target, str):
        return _from_alias(target, seen | {alias})
    answerer = resolve(target)
    if not (isinstance(target, dict) and target.get("name")):
        # Known by the alias lenses name it with; the model is on each answer anyway.
        answerer.name = alias
    return answerer


def _from_config(config: dict[str, Any]) -> Answerer:
    config = dict(config)
    path = config.pop("class", "")
    if not path:
        raise ImproperlyConfigured(f"Answerer config {config!r} has no 'class'")
    if isinstance(path, str) and "." not in path:
        # A short name ("Patterns", "LLMJudge") is one of the SDK's own answerers.
        path = f"django_ai_sdk.oversight.answerers.{path}"
    cls = import_string(path) if isinstance(path, str) else path
    if not (isinstance(cls, type) and issubclass(cls, Answerer)):
        raise ImproperlyConfigured(f"{path!r} is not an Answerer subclass")
    return cls(**config)


def _human() -> Answerer:
    from django_ai_sdk.oversight.answerers.human import Human

    return Human()


# Aliases that work without settings.
BUILTIN_ALIASES: dict[str, Callable[[], Answerer]] = {"human": _human}


__all__ = [
    "BUILTIN_ALIASES",
    "Answerer",
    "Cascade",
    "Function",
    "Sample",
    "Vote",
    "alias_defined",
    "alias_source",
    "alias_table",
    "ask",
    "resolve",
    "sampled",
    "set_console_aliases",
]
