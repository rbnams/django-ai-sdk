"""Answerers: whoever or whatever answers a lens."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import re
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from django.core.exceptions import ImproperlyConfigured
from django.utils.module_loading import import_string

from django_ai_sdk.logger import get_logger
from django_ai_sdk.oversight.types import Answer

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from django_ai_sdk.oversight.lenses import Lens
    from django_ai_sdk.oversight.types import TurnView

logger = get_logger(__name__)

# Seconds an answerer may take before its answer is abandoned, unless it sets its own.
DEFAULT_TIMEOUT = 30.0

# The error of an answer whose service failed: down, refused, or broke on its side
# ("unavailable: ConnectionError"). Reflect asks such a lens again: late, not lost.
UNAVAILABLE = "unavailable"


# The error of an answer whose answerer would have had to cut the text it judges:
# it judged none of it, so FAIL_MODE decides, as for any failure.
TOO_LONG = "too long"


def param(name: str, type: str, label: str, help: str = "", **extra: Any) -> dict[str, Any]:
    """One field of a console form. Types the editor renders: text, number, bool, choice
    (from `choices`), multi (several of `choices`), lines, labelled_lines, model."""
    return {"name": name, "type": type, "label": label, "help": help, **extra}


def modules_installed(*modules: str) -> bool:
    """Whether every one of `modules` can be imported, without importing it."""
    from importlib.util import find_spec

    return all(find_spec(module) is not None for module in modules)


def unavailable(exc: BaseException) -> str:
    """The error an answer carries when an external answerer failed with `exc`."""
    return f"{UNAVAILABLE}: {type(exc).__name__}"


def is_unavailable(error: str) -> bool:
    """Whether `error` says the answerer's service failed, so asking again later may work."""
    return error.startswith(f"{UNAVAILABLE}: ")


class Answerer(ABC):
    """Answers lenses. Return None to abstain: the next answerer, if any, is asked."""

    # Charged to and capped by the model budget. Set on the class; Function sets it per instance.
    metered: bool = True
    # Who receives turn content when this answerer runs ("model gateway", a service's name).
    # Empty: it runs in this process and nothing leaves.
    external: ClassVar[str] = ""
    # Seconds before the answer is abandoned.
    timeout: float = DEFAULT_TIMEOUT
    name: str = ""
    # How the console offers this class as a named answerer, or None when it does not:
    # {"label", "summary", "params": [param(...)]}. The SDK's own answerers and the
    # classes in AI_SDK_OVERSIGHT["ANSWERER_KINDS"] are offered alike.
    console: ClassVar[dict[str, Any] | None] = None

    @classmethod
    def installed(cls) -> bool:
        """Whether what this class needs is installed, so the console may offer it."""
        return True

    @abstractmethod
    async def answer(self, lens: Lens, view: TurnView) -> Answer | None: ...

    def label(self) -> str:
        return self.name or type(self).__name__.lower()

    def missing(self) -> str:
        """Why this answerer can't run on this deployment, so it abstains; "" when it can."""
        return ""

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.label(),
            "metered": self.metered,
            "external": self.external,
            "class": type(self).__name__,
        }

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.label()!r}>"


async def ask(
    answerer: Answerer | str | dict[str, Any], lens: Lens, view: TurnView
) -> Answer | None:
    """Ask one answerer, timed, typed and contained: a failure is an Answer with `error`.

    `answerer` may be a spec (`resolve`): one that does not build is an error too, so
    the answers the lens already has are kept. A metered answerer is charged to the
    active Budget, and refused once it is spent. An external answerer that raises is
    `unavailable`: its service failed, not the lens. An answerer that cancels itself
    (we did not cancel it) has failed; a cancellation from outside passes through.
    """
    from django_ai_sdk.oversight import budget

    started = time.perf_counter()
    label = _spec_label(answerer)
    try:
        answerer = resolve(answerer)
    except Exception as exc:
        logger.warning("Answerer {} for lens {} does not build: {}", label, lens.name, exc)
        return Answer(answerer=label, error=type(exc).__name__)
    spend = budget.current() if answerer.metered else None
    if spend is not None and not spend.allows():
        return Answer(answerer=answerer.label(), error="budget")
    try:
        answer = await asyncio.wait_for(answerer.answer(lens, view), timeout=answerer.timeout)
    except TimeoutError:
        return Answer(answerer=answerer.label(), error="timeout", latency_ms=_ms(started))
    except asyncio.CancelledError:
        task = asyncio.current_task()
        if task is not None and task.cancelling():
            raise
        logger.warning("Answerer {} cancelled itself on lens {}", answerer.label(), lens.name)
        return Answer(answerer=answerer.label(), error="CancelledError", latency_ms=_ms(started))
    except Exception as exc:
        if answerer.external:
            # Only the type: a service's error message may quote the request or its reply.
            logger.warning(
                "Answerer {} failed on lens {}: {}", answerer.label(), lens.name, type(exc).__name__
            )
            error = unavailable(exc)
        else:
            logger.opt(exception=exc).warning(
                "Answerer {} failed on lens {}", answerer.label(), lens.name
            )
            error = type(exc).__name__
        return Answer(answerer=answerer.label(), error=error, latency_ms=_ms(started))
    if answer is None:
        return None
    if not isinstance(answer, Answer):
        logger.warning(
            "Answerer {} answered lens {} with a {}, not an Answer",
            answerer.label(),
            lens.name,
            type(answer).__name__,
        )
        return Answer(
            answerer=answerer.label(),
            error=f"not an answer: {type(answer).__name__}",
            latency_ms=_ms(started),
        )
    if spend is not None:
        spend.charge(answerer.label(), answer.cost)
    if answer.value is not None and not answer.error:
        try:
            answer.value = lens.parse(answer.value)
        except Exception as exc:
            # Only the kind of failure: the error is hashed into the ledger, the value may quote.
            answer = answer.model_copy(
                update={"value": None, "error": f"unparsable: {type(exc).__name__}"}
            )
    answer.answerer = answer.answerer or answerer.label()
    if answer.latency_ms is None:
        answer.latency_ms = _ms(started)
    return answer


def _spec_label(spec: Any) -> str:
    if isinstance(spec, Answerer):
        return spec.label()
    if isinstance(spec, dict):
        return str(spec.get("name") or spec.get("class") or "answerer")
    return str(spec)


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)


def clip(text: str, limit: int) -> str:
    """`text` in at most `limit` characters: its start and its end, the middle left out.

    What a model answerer reads of a long text. Both ends, because what matters can
    sit at the end as easily as at the start (an instruction after a page of padding).
    """
    if len(text) <= limit:
        return text
    marker = "\n[... {} characters left out ...]\n"
    keep = max(limit - len(marker.format(len(text))), 0)
    head, tail = keep - keep // 2, keep // 2
    return text[:head] + marker.format(len(text) - keep) + text[len(text) - tail :]


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

    Sync or async. A sync function runs in a thread, so a slow one cannot hold the
    event loop past a stage's deadline. Not charged to the budget unless
    `metered=True` says it calls a model.
    """

    def __init__(
        self,
        fn: Callable[[Lens, TurnView], Any | Awaitable[Any]] | str,
        *,
        name: str = "",
        metered: bool = False,
    ) -> None:
        if isinstance(fn, str):
            # A dotted path, so a lens defined in the console can name host code.
            fn = import_string(fn)
        self.fn = fn
        self.name = name or getattr(fn, "__name__", "function")
        self.metered = metered

    async def answer(self, lens: Lens, view: TurnView) -> Answer | None:
        if inspect.iscoroutinefunction(self.fn):
            result = await self.fn(lens, view)
        else:
            result = await asyncio.to_thread(self.fn, lens, view)
        if inspect.isawaitable(result):
            result = await result
        if result is None or isinstance(result, Answer):
            return result
        return Answer(value=result)


def resolve(
    spec: Answerer | str | dict[str, Any], aliases: Mapping[str, Any] | None = None
) -> Answerer:
    """An Answerer from an instance, an alias, or a config dict.

    A config dict is {"class": dotted path, **kwargs}; aliases may name aliases.
    Aliases come from `aliases`, or else from `alias_table()`.
    """
    if isinstance(spec, Answerer):
        return spec
    if isinstance(spec, str):
        return build_alias(spec.removesuffix("?"), aliases)
    if isinstance(spec, dict):
        return _from_config(spec)
    raise ImproperlyConfigured(f"{spec!r} is not an answerer, an alias or a config")


# Aliases defined in the console (PolicyOverride rows on the "*" policy). Refreshed
# by policy.aconsole, for the paths that resolve without a policy at hand (the
# answerer chain, composite answerers, system checks): a sync resolve sees the last load.
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


def build_alias(
    alias: str, table: Mapping[str, Any] | None = None, seen: frozenset[str] = frozenset()
) -> Answerer:
    """The answerer `alias` names in `table` (default: `alias_table()`)."""
    if alias in seen:
        raise ImproperlyConfigured(f"Answerer alias {alias!r} names itself")
    aliases = alias_table() if table is None else table
    if alias not in aliases:
        builtin = BUILTIN_ALIASES.get(alias)
        if builtin is None:
            raise ImproperlyConfigured(
                f"No answerer alias {alias!r}; add it to AI_SDK_OVERSIGHT['ANSWERERS']"
            )
        return builtin()
    target = aliases[alias]
    if isinstance(target, str):
        return build_alias(target, aliases, seen | {alias})
    answerer = resolve(target, aliases)
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


SDK_ANSWERERS = "django_ai_sdk.oversight.answerers."
# An alias, optionally skipped where missing ("classifier?"), or a code-built answerer ("code:0").
CONSOLE_SPEC = re.compile(r"^[a-z0-9_-]+\??$|^code:\d+$")


def console_config(config: Any, *, kinds: list[dict[str, Any]]) -> str | dict[str, Any]:
    """`config` as the console may store it; ValueError otherwise.

    The console names an alias, or one of `kinds` (a catalog list) with that kind's
    parameters and nothing else, so an editor never imports or calls host code.
    Code and settings are not limited: they build through `resolve` directly.
    """
    if isinstance(config, str):
        if not CONSOLE_SPEC.match(config):
            raise ValueError(f"{config!r} is not an answerer name")
        return config
    if not isinstance(config, dict):
        raise ValueError(f"{config!r} is not an answerer name or configuration")
    path = config.get("class")
    short = path.removeprefix(SDK_ANSWERERS) if isinstance(path, str) else ""
    kind = next((k for k in kinds if k["class"] == short), None)
    if kind is None:
        raise ValueError(f"{path!r} is not an answerer the console offers here")
    allowed = {"class", "name", *(p["name"] for p in kind["params"])}
    extra = sorted(set(config) - allowed)
    if extra:
        raise ValueError(f"{short} does not take {', '.join(extra)} from the console")
    for param in kind["params"]:
        # A choice is one the kind offers: say, a judge's server is one settings define.
        value, choices = config.get(param["name"]), param.get("choices")
        if value is None or choices is None or param["type"] not in {"choice", "multi"}:
            continue
        picked = value if isinstance(value, list) else [value]
        if any(v not in choices for v in picked):
            raise ValueError(f"{value!r} is not a {param['label'].lower()} the console offers")
    return {**config, "class": short}


def _human() -> Answerer:
    from django_ai_sdk.oversight.answerers.human import Human

    return Human()


# Aliases that work without settings.
BUILTIN_ALIASES: dict[str, Callable[[], Answerer]] = {"human": _human}


__all__ = [
    "BUILTIN_ALIASES",
    "DEFAULT_TIMEOUT",
    "TOO_LONG",
    "UNAVAILABLE",
    "Answerer",
    "Function",
    "alias_defined",
    "alias_source",
    "alias_table",
    "ask",
    "build_alias",
    "console_config",
    "is_unavailable",
    "modules_installed",
    "param",
    "resolve",
    "sampled",
    "set_console_aliases",
    "unavailable",
]
