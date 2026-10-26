"""Conditions over lens answers: written in Python, stored and edited as JSON.

    L("prompt_injection").is_true & (L("toxicity") > 0.8)
    L("pii").contains("bsn") | L("topic").in_({"hr", "legal"})
    ~L("grounded").is_true & L("sources").confident(0.7)

Both forms run through one evaluator, so a rule declared in code and a rule an
operator tuned in the console behave identically. A lens with no usable answer
makes every comparison on it false; `L(name).missing` tests for that.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from django_ai_sdk.oversight.types import Answer

# op -> (needs a value, test(answer value, operand))
OPS: dict[str, Any] = {
    "is_true": lambda v, _: v is True,
    "is_false": lambda v, _: v is False,
    "eq": lambda v, x: v == x,
    "ne": lambda v, x: v != x,
    "gt": lambda v, x: _num(v) is not None and _num(v) > x,
    "ge": lambda v, x: _num(v) is not None and _num(v) >= x,
    "lt": lambda v, x: _num(v) is not None and _num(v) < x,
    "le": lambda v, x: _num(v) is not None and _num(v) <= x,
    "in": lambda v, x: v in x if not isinstance(v, list) else bool(set(v) & set(x)),
    "contains": lambda v, x: isinstance(v, list | str) and x in v,
    "any": lambda v, _: bool(v),
}
# Ops that read the answer itself, not its value.
META_OPS = {"missing", "present", "confident", "unsure"}


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    return float(value) if isinstance(value, int | float) else None


class Condition:
    """A test over a turn's answers, keyed by lens name."""

    def evaluate(self, answers: Mapping[str, Answer]) -> bool:
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    def lenses(self) -> set[str]:
        raise NotImplementedError

    def __and__(self, other: Condition) -> Condition:
        return All([*_flatten(self, All), *_flatten(other, All)])

    def __or__(self, other: Condition) -> Condition:
        return AnyOf([*_flatten(self, AnyOf), *_flatten(other, AnyOf)])

    def __invert__(self) -> Condition:
        return Not(self)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.to_dict()}>"


def _flatten(condition: Condition, kind: type) -> list[Condition]:
    return list(condition.items) if isinstance(condition, kind) else [condition]


class Compare(Condition):
    """One test on one lens's answer."""

    def __init__(self, lens: str, op: str, value: Any = None) -> None:
        if op not in OPS and op not in META_OPS:
            raise ValueError(f"Unknown op {op!r}")
        self.lens = lens
        self.op = op
        self.value = sorted(value) if isinstance(value, set | frozenset) else value

    def evaluate(self, answers: Mapping[str, Answer]) -> bool:
        answer = answers.get(self.lens)
        usable = answer is not None and answer.ok
        if self.op == "missing":
            return not usable
        if self.op == "present":
            return usable
        if not usable:
            return False
        if self.op == "confident":
            return answer.confidence >= float(self.value)
        if self.op == "unsure":
            return answer.confidence < float(self.value)
        try:
            return bool(OPS[self.op](answer.value, self.value))
        except TypeError:
            return False

    def to_dict(self) -> dict[str, Any]:
        node: dict[str, Any] = {"lens": self.lens, "op": self.op}
        if self.value is not None:
            node["value"] = self.value
        return node

    def lenses(self) -> set[str]:
        return {self.lens}


class All(Condition):
    def __init__(self, items: list[Condition]) -> None:
        self.items = items

    def evaluate(self, answers: Mapping[str, Answer]) -> bool:
        return all(item.evaluate(answers) for item in self.items)

    def to_dict(self) -> dict[str, Any]:
        return {"all": [item.to_dict() for item in self.items]}

    def lenses(self) -> set[str]:
        return set().union(*(item.lenses() for item in self.items)) if self.items else set()


class AnyOf(Condition):
    def __init__(self, items: list[Condition]) -> None:
        self.items = items

    def evaluate(self, answers: Mapping[str, Answer]) -> bool:
        return any(item.evaluate(answers) for item in self.items)

    def to_dict(self) -> dict[str, Any]:
        return {"any": [item.to_dict() for item in self.items]}

    def lenses(self) -> set[str]:
        return set().union(*(item.lenses() for item in self.items)) if self.items else set()


class Not(Condition):
    def __init__(self, item: Condition) -> None:
        self.item = item

    def evaluate(self, answers: Mapping[str, Answer]) -> bool:
        return not self.item.evaluate(answers)

    def to_dict(self) -> dict[str, Any]:
        return {"not": self.item.to_dict()}

    def lenses(self) -> set[str]:
        return self.item.lenses()


class Always(Condition):
    """Holds for every turn: a rule that always alerts, a lens's plain record."""

    def evaluate(self, answers: Mapping[str, Answer]) -> bool:
        return True

    def to_dict(self) -> dict[str, Any]:
        return {"always": True}

    def lenses(self) -> set[str]:
        return set()


class L:
    """A lens, by name, to compare against. Builds Compare conditions."""

    # Comparison operators build conditions, so an L is not a dict key.
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, name: str) -> None:
        self.name = name

    @property
    def is_true(self) -> Compare:
        return Compare(self.name, "is_true")

    @property
    def is_false(self) -> Compare:
        return Compare(self.name, "is_false")

    @property
    def missing(self) -> Compare:
        return Compare(self.name, "missing")

    @property
    def present(self) -> Compare:
        return Compare(self.name, "present")

    @property
    def any(self) -> Compare:
        """An Entities lens found something."""
        return Compare(self.name, "any")

    def __eq__(self, value: object) -> Compare:  # type: ignore[override]
        return Compare(self.name, "eq", value)

    def __ne__(self, value: object) -> Compare:  # type: ignore[override]
        return Compare(self.name, "ne", value)

    def __gt__(self, value: float) -> Compare:
        return Compare(self.name, "gt", value)

    def __ge__(self, value: float) -> Compare:
        return Compare(self.name, "ge", value)

    def __lt__(self, value: float) -> Compare:
        return Compare(self.name, "lt", value)

    def __le__(self, value: float) -> Compare:
        return Compare(self.name, "le", value)

    def in_(self, values: Any) -> Compare:
        return Compare(self.name, "in", list(values))

    def contains(self, value: str) -> Compare:
        return Compare(self.name, "contains", value)

    def confident(self, at_least: float) -> Compare:
        return Compare(self.name, "confident", at_least)

    def unsure(self, below: float) -> Compare:
        return Compare(self.name, "unsure", below)


def from_dict(node: Mapping[str, Any] | bool) -> Condition:
    """A condition from its JSON form; raises ValueError on anything else."""
    if node is True or (isinstance(node, Mapping) and node.get("always")):
        return Always()
    if not isinstance(node, Mapping):
        raise ValueError(f"{node!r} is not a condition")
    if "all" in node:
        return All([from_dict(item) for item in node["all"]])
    if "any" in node:
        return AnyOf([from_dict(item) for item in node["any"]])
    if "not" in node:
        return Not(from_dict(node["not"]))
    if "lens" in node and "op" in node:
        return Compare(str(node["lens"]), str(node["op"]), node.get("value"))
    raise ValueError(f"{dict(node)!r} is not a condition")


def same_shape(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    """Whether `b` differs from `a` only in compared values: a threshold, not a new rule.

    The console may tune a code rule; changing what it tests takes a deploy.
    """
    if set(a) != set(b):
        return False
    for key in ("all", "any"):
        if key in a:
            return len(a[key]) == len(b[key]) and all(
                same_shape(x, y) for x, y in zip(a[key], b[key], strict=True)
            )
    if "not" in a:
        return same_shape(a["not"], b["not"])
    return a.get("lens") == b.get("lens") and a.get("op") == b.get("op")


__all__ = [
    "All",
    "Always",
    "AnyOf",
    "Compare",
    "Condition",
    "L",
    "Not",
    "from_dict",
    "same_shape",
]
