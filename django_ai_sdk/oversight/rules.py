"""Rules: when answers say so, do something. Lenses observe; only rules act.

    Rule(
        "block_injection",
        when=L("prompt_injection").is_true & L("prompt_injection").confident(0.9),
        then=Block("I can't help with that request."),
        mode=Mode.ENFORCE,
    )

A rule starts in shadow mode: it records what it would have done, on every turn,
so its false-positive rate is known before it can block anyone.
"""

from __future__ import annotations

from typing import Any, ClassVar

from django.core.exceptions import ImproperlyConfigured

from django_ai_sdk.oversight.conditions import Condition, from_dict
from django_ai_sdk.oversight.types import Mode, Outcome, Stage


class Action:
    """What a matched rule does. `outcome` is what the turn becomes when it is enforced."""

    kind: ClassVar[str] = ""
    outcome: ClassVar[Outcome] = Outcome.FLAGGED
    # Stages where the action can still change the turn. Elsewhere it is recorded.
    effective: ClassVar[tuple[Stage, ...]] = tuple(Stage)

    def params(self) -> dict[str, Any]:
        return {}

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, **self.params()}

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.params()}>"


class Allow(Action):
    """Record the match and nothing else: an explicit "this is fine"."""

    kind = "allow"
    outcome = Outcome.ALLOWED


class Alert(Action):
    """Mark the turn for attention at `severity`."""

    kind = "alert"
    outcome = Outcome.FLAGGED

    def __init__(self, severity: str = "warning") -> None:
        self.severity = severity

    def params(self) -> dict[str, Any]:
        return {"severity": self.severity}


class Redact(Action):
    """Mask what `lenses` (default: every lens the rule reads) pointed at.

    At the gate the question is masked before the model or the thread sees it; in
    review the stored answer is masked, and the client told to replace its copy;
    in reflect the stored message is masked after the fact.
    """

    kind = "redact"
    outcome = Outcome.REDACTED
    effective = (Stage.GATE, Stage.WATCH, Stage.REVIEW, Stage.REFLECT)

    def __init__(self, lenses: list[str] | None = None, template: str = "[{label}]") -> None:
        self.lenses = list(lenses or [])
        self.template = template

    def params(self) -> dict[str, Any]:
        return {"lenses": self.lenses, "template": self.template}


class RestrictTools(Action):
    """Take tools away for this turn: `names`, or all of them when empty."""

    kind = "restrict_tools"
    outcome = Outcome.RESTRICTED
    effective = (Stage.GATE, Stage.WATCH)

    def __init__(self, names: list[str] | None = None) -> None:
        self.names = list(names or [])

    def params(self) -> dict[str, Any]:
        return {"names": self.names}


class Block(Action):
    """Refuse: the question never reaches the model, or the answer never reaches the user."""

    kind = "block"
    outcome = Outcome.BLOCKED
    effective = (Stage.GATE, Stage.WATCH, Stage.REVIEW)

    DEFAULT_MESSAGE = "This request can't be answered here."

    def __init__(self, message: str = "") -> None:
        self.message = message or self.DEFAULT_MESSAGE

    def params(self) -> dict[str, Any]:
        return {"message": self.message}


class Queue(Action):
    """Put the turn in front of people: review queue `queue`, at `priority`."""

    kind = "queue"
    outcome = Outcome.FLAGGED

    def __init__(self, queue: str = "default", priority: int = 0) -> None:
        self.queue = queue
        self.priority = priority

    def params(self) -> dict[str, Any]:
        return {"queue": self.queue, "priority": self.priority}


class Escalate(Action):
    """Ask a stronger answerer: re-judge `lens` with `answerer` in the background."""

    kind = "escalate"
    outcome = Outcome.FLAGGED

    def __init__(self, lens: str, answerer: str = "strong") -> None:
        self.lens = lens
        self.answerer = answerer

    def params(self) -> dict[str, Any]:
        return {"lens": self.lens, "answerer": self.answerer}


class Notify(Action):
    """Send the match to every configured exporter, or to `exporters` by name."""

    kind = "notify"
    outcome = Outcome.FLAGGED

    def __init__(self, exporters: list[str] | None = None, severity: str = "warning") -> None:
        self.exporters = list(exporters or [])
        self.severity = severity

    def params(self) -> dict[str, Any]:
        return {"exporters": self.exporters, "severity": self.severity}


ACTIONS: dict[str, type[Action]] = {
    cls.kind: cls for cls in (Allow, Alert, Redact, RestrictTools, Block, Queue, Escalate, Notify)
}

# Enforced in this order; a block makes everything after it moot.
PRECEDENCE = ["block", "restrict_tools", "redact", "escalate", "queue", "notify", "alert", "allow"]


def action_from_dict(node: dict[str, Any]) -> Action:
    node = dict(node)
    kind = node.pop("kind", "")
    if kind not in ACTIONS:
        raise ImproperlyConfigured(f"Unknown action {kind!r}; one of {sorted(ACTIONS)}")
    return ACTIONS[kind](**node)


class Rule:
    """When `when` holds, do `then`. Shadow until someone decides otherwise."""

    def __init__(
        self,
        name: str,
        *,
        when: Condition | dict[str, Any],
        then: Action | list[Action],
        mode: Mode | str = Mode.SHADOW,
        stages: list[Stage | str] | None = None,
        message: str = "",
        description: str = "",
    ) -> None:
        self.name = name
        self.when = when if isinstance(when, Condition) else from_dict(when)
        self.then = [then] if isinstance(then, Action) else list(then)
        self.mode = Mode(mode)
        # Empty: every stage where all the lenses it reads have answered.
        self.stages = tuple(Stage(s) for s in stages or ())
        # Shown to the person when the rule blocks, overriding the action's own.
        self.message = message
        self.description = description

    def __repr__(self) -> str:
        return f"<Rule {self.name!r} {self.mode}>"

    def lenses(self) -> set[str]:
        return self.when.lenses()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "when": self.when.to_dict(),
            "then": [action.to_dict() for action in self.then],
            "mode": str(self.mode),
            "stages": [str(s) for s in self.stages],
            "message": self.message,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, node: dict[str, Any]) -> Rule:
        return cls(
            node["name"],
            when=from_dict(node["when"]),
            then=[action_from_dict(a) for a in node.get("then") or []],
            mode=node.get("mode", Mode.SHADOW),
            stages=node.get("stages") or None,
            message=node.get("message", ""),
            description=node.get("description", ""),
        )

    def ordered_actions(self) -> list[Action]:
        return sorted(self.then, key=lambda action: PRECEDENCE.index(action.kind))


__all__ = [
    "ACTIONS",
    "PRECEDENCE",
    "Action",
    "Alert",
    "Allow",
    "Block",
    "Escalate",
    "Notify",
    "Queue",
    "Redact",
    "RestrictTools",
    "Rule",
    "action_from_dict",
]
