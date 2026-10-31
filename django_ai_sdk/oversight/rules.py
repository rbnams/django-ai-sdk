"""Rules: when answers say so, do something. Lenses observe; only rules act.

    Rule(
        "block_injection",
        when=L("prompt_injection").is_true & L("prompt_injection").confident(0.9),
        then=Block("I can't help with that request."),
        mode=Mode.ENFORCE,
    )

A rule starts in report mode: it records what it would have done, on every turn,
so its false-positive rate is known before it can block anyone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from django.core.exceptions import ImproperlyConfigured
from django.utils.translation import gettext_lazy as _

from django_ai_sdk.oversight.conditions import Condition, from_dict
from django_ai_sdk.oversight.types import Mode, Outcome, Severity, Stage

if TYPE_CHECKING:
    from django.utils.functional import Promise


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


def _severity(value: str) -> str:
    try:
        return str(Severity(value))
    except ValueError:
        raise ValueError(
            f"Unknown severity {value!r}; one of {[str(s) for s in Severity]}"
        ) from None


class Alert(Action):
    """Mark the turn for attention at `severity`."""

    kind = "alert"
    outcome = Outcome.FLAGGED

    def __init__(self, severity: str = "warning") -> None:
        self.severity = _severity(severity)

    def params(self) -> dict[str, Any]:
        return {"severity": self.severity}


class Mask(Action):
    """Mask what `lenses` (default: every lens the rule reads) pointed at.

    At the gate the question is masked before the model or the thread sees it; at
    finish the stored answer is masked, and the client told to replace its copy
    when the stream ends; in reflect the stored message is masked after the fact.
    While the answer streams nothing is masked: a rule that holds there is judged
    again at finish, on the whole answer.
    """

    kind = "mask"
    outcome = Outcome.MASKED
    effective = (Stage.GATE, Stage.FINISH, Stage.REFLECT)

    def __init__(self, lenses: list[str] | None = None, template: str = "[{label}]") -> None:
        try:
            template.format(label="X")
        except (KeyError, IndexError, ValueError) as exc:
            raise ValueError(f"Mask template {template!r} must only use {{label}}: {exc}") from exc
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
    effective = (Stage.GATE, Stage.WATCH, Stage.FINISH)

    # Translated when shown, like a lazy `message`: str() it where it is stored or sent.
    DEFAULT_MESSAGE = _("This request can't be answered here.")

    def __init__(self, message: str | Promise = "") -> None:
        self.message = message or self.DEFAULT_MESSAGE

    def params(self) -> dict[str, Any]:
        return {"message": str(self.message)}


class Queue(Action):
    """Put the turn in front of people: review queue `queue`, at `priority`."""

    kind = "queue"
    outcome = Outcome.FLAGGED

    def __init__(self, queue: str = "default", priority: int = 0) -> None:
        self.queue = queue
        self.priority = priority

    def params(self) -> dict[str, Any]:
        return {"queue": self.queue, "priority": self.priority}


class Notify(Action):
    """Send the match to every configured exporter, or to `exporters` by name."""

    kind = "notify"
    outcome = Outcome.FLAGGED

    def __init__(self, exporters: list[str] | None = None, severity: str = "warning") -> None:
        self.exporters = list(exporters or [])
        self.severity = _severity(severity)

    def params(self) -> dict[str, Any]:
        return {"exporters": self.exporters, "severity": self.severity}


ACTIONS: dict[str, type[Action]] = {
    cls.kind: cls for cls in (Alert, Mask, RestrictTools, Block, Queue, Notify)
}

# Enforced in this order; a block makes everything after it moot.
PRECEDENCE = ["block", "restrict_tools", "mask", "queue", "notify", "alert"]


def action_from_dict(node: dict[str, Any]) -> Action:
    node = dict(node)
    kind = node.pop("kind", "")
    if kind not in ACTIONS:
        raise ImproperlyConfigured(f"Unknown action {kind!r}; one of {sorted(ACTIONS)}")
    return ACTIONS[kind](**node)


class Rule:
    """When `when` holds, do `then`. Report only until someone decides otherwise."""

    def __init__(
        self,
        name: str,
        *,
        when: Condition | dict[str, Any],
        then: Action | list[Action],
        mode: Mode | str = Mode.REPORT,
        stages: list[Stage | str] | None = None,
        message: str = "",
        notice: str = "",
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
        # Told to the person whenever the rule acts, block or not: "The fraud team is alerted."
        # Empty: a queue or an alert happens out of their sight.
        self.notice = notice
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
            "notice": self.notice,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, node: dict[str, Any]) -> Rule:
        return cls(
            node["name"],
            when=from_dict(node["when"]),
            then=[action_from_dict(a) for a in node.get("then") or []],
            mode=node.get("mode", Mode.REPORT),
            stages=node.get("stages") or None,
            message=node.get("message", ""),
            notice=node.get("notice", ""),
            description=node.get("description", ""),
        )

    def ordered_actions(self) -> list[Action]:
        return sorted(self.then, key=lambda action: PRECEDENCE.index(action.kind))


__all__ = [
    "ACTIONS",
    "PRECEDENCE",
    "Action",
    "Alert",
    "Block",
    "Mask",
    "Notify",
    "Queue",
    "RestrictTools",
    "Rule",
    "action_from_dict",
]
