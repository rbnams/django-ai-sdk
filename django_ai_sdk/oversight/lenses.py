"""Lenses: one typed question about a turn, answerable by anything.

A lens fixes the question and the shape of its answer, never who answers it. Pick
the answerer per lens: Jev for most yes/no, label and score lenses, code for exact
spans such as personal data and secrets, a model judge for written text or where
Jev is unsure. Change it later and the lens's history stays one comparable series.

Declare a lens as a subclass, or inline with keyword arguments::

    class Topic(Choice):
        name = "topic"
        question = "What is this message about?"
        labels = ["billing", "hr", "other"]
        answerers = ["jev?", "judge?"]

    Flag(name="mentions_competitor", question="Does it name a competitor?")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from django.core.exceptions import ImproperlyConfigured
from pydantic import BaseModel, Field, create_model

from django_ai_sdk.oversight.types import Stage, Subject, TurnView

if TYPE_CHECKING:
    from typing import Literal  # noqa: F401

    from django_ai_sdk.oversight.answerers.base import Answerer


class Lens:
    """Base lens. Subclass one of Flag, Choice, Score, Entities or Text."""

    kind: ClassVar[str] = ""

    # Registry key within a policy. Observations key on it, so renaming starts a new series.
    name: str = ""
    label: str = ""
    # Asked verbatim of a model answerer; keep it one question.
    question: str = ""
    # What a model judge is told beyond the question: scale anchors, edge cases.
    guidance: str = ""
    subject: Subject = Subject.INPUT
    stages: tuple[Stage, ...] = (Stage.REVIEW,)

    # Tried in order; a later one only when the earlier is unsure (below `confident`)
    # or abstains. Instances, or names of aliases in AI_SDK_OVERSIGHT["ANSWERERS"].
    answerers: list[Answerer | str] = []  # noqa: RUF012
    confident: float = 0.75

    # Share of turns the reflect stage judges, 0..1. Inline stages always run.
    sample: float = 1.0
    enabled: bool = True
    # Only asked when the turn has sources or tool results to judge against, its own
    # or an earlier turn's (for subject SOURCES, its own): groundedness means nothing
    # without them.
    needs_evidence: bool = False

    def __init__(self, **overrides: Any) -> None:
        for key, value in overrides.items():
            if not hasattr(type(self), key):
                raise ImproperlyConfigured(f"{type(self).__name__} has no attribute {key!r}")
            setattr(self, key, value)
        self.stages = tuple(Stage(s) for s in self.stages)
        self.subject = Subject(self.subject)
        self.answerers = list(self.answerers)

    def applies(self, view: TurnView) -> bool:
        """Whether this turn has what the lens judges."""
        if self.subject is Subject.SOURCES:
            return bool(view.evidence())
        if self.needs_evidence:
            return bool(view.evidence(earlier=True))
        return True

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name!r}>"

    # The value contract.

    def parse(self, value: Any) -> Any:
        """Coerce a raw answer to this lens's type; raise ValueError if it isn't one."""
        raise NotImplementedError

    def from_detections(self, labels: list[str]) -> Any:
        """The value a detector's findings mean for this lens (a hit list -> a value)."""
        raise NotImplementedError

    def value_type(self) -> Any:
        """The Python type a structured-output judge returns for `value`."""
        raise NotImplementedError

    def verdict_model(self) -> type[BaseModel]:
        """The structured output a model judge fills in."""
        return create_model(
            f"{type(self).__name__}Verdict",
            rationale=(str, Field(description="One or two sentences: why.")),
            value=(self.value_type(), Field(description=self.value_description())),
            confidence=(float, Field(ge=0, le=1, description="How sure you are, 0 to 1.")),
        )

    def value_description(self) -> str:
        return self.question

    def typesafe_question(self) -> dict[str, Any] | None:
        """This lens as a TypeSafe question, or None when TypeSafe has no such type."""
        return None

    def typesafe_instructions(self) -> str:
        return f"{self.question}\n{self.guidance}" if self.guidance else self.question

    def describe(self) -> dict[str, Any]:
        """JSON-safe summary for the policy screen and the ledger's config events."""
        return {
            "name": self.name,
            "kind": self.kind,
            "label": self.label or self.name.replace("_", " "),
            "question": self.question,
            "subject": str(self.subject),
            "stages": [str(s) for s in self.stages],
            "confident": self.confident,
            "sample": self.sample,
            "enabled": self.enabled,
            **self.extra_description(),
        }

    def extra_description(self) -> dict[str, Any]:
        return {}


class Flag(Lens):
    """A yes/no question. A detector's hit means yes."""

    kind = "flag"

    def parse(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, int | float):
            return value >= 0.5
        if isinstance(value, str) and value.strip().lower() in {"yes", "true", "1", "ja"}:
            return True
        if isinstance(value, str) and value.strip().lower() in {"no", "false", "0", "nee"}:
            return False
        raise ValueError(f"{value!r} is not a yes or a no")

    def from_detections(self, labels: list[str]) -> bool:
        return bool(labels)

    def value_type(self) -> Any:
        return bool

    # What yes and no mean, for TypeSafe's criteria and a judge's guidance.
    yes: str = "Yes."
    no: str = "No."

    def typesafe_question(self) -> dict[str, Any]:
        return {
            "type": "noul",
            "instructions": self.typesafe_instructions(),
            "criteria": {"true": self.yes, "false": self.no},
        }


class Choice(Lens):
    """Exactly one label out of `labels`; `default` when a detector finds none."""

    kind = "choice"
    labels: list[str] = []  # noqa: RUF012
    default: str = "other"

    def parse(self, value: Any) -> str:
        value = str(value).strip()
        for label in self.labels:
            if label.lower() == value.lower():
                return label
        raise ValueError(f"{value!r} is not one of {self.labels}")

    def from_detections(self, labels: list[str]) -> str:
        return next((label for label in labels if label in self.labels), self.default)

    def value_type(self) -> Any:
        from typing import Literal

        return Literal[tuple(self.labels)]  # ty: ignore[invalid-type-form]

    def value_description(self) -> str:
        return f"{self.question} One of: {', '.join(self.labels)}."

    # label -> what it means; a label without one means itself.
    criteria: dict[str, str] = {}  # noqa: RUF012

    def typesafe_question(self) -> dict[str, Any]:
        return {
            "type": "choice",
            "instructions": self.typesafe_instructions(),
            "criteria": {label: self.criteria.get(label, label) for label in self.labels},
        }

    def extra_description(self) -> dict[str, Any]:
        return {"labels": list(self.labels), "default": self.default}


class Score(Lens):
    """A number from 0 to 1, where 1 is the most of what the question asks about."""

    kind = "score"
    # What 0 and 1 mean, for a model judge and for the screen; `middle` is Jev's middle level.
    low: str = "not at all"
    middle: str = "partly"
    high: str = "completely"

    def parse(self, value: Any) -> float:
        number = float(value)
        if number > 1 and number <= 10:
            # A judge that answered on a 1-10 scale.
            number /= 10
        return min(1.0, max(0.0, number))

    def from_detections(self, labels: list[str]) -> float:
        return 1.0 if labels else 0.0

    def value_type(self) -> Any:
        return float

    def value_description(self) -> str:
        return f"{self.question} 0 means {self.low}, 1 means {self.high}."

    def typesafe_question(self) -> dict[str, Any]:
        return {
            "type": "score",
            "instructions": self.typesafe_instructions(),
            "criteria": [self.low, self.middle, self.high],
        }

    def extra_description(self) -> dict[str, Any]:
        return {"low": self.low, "high": self.high}


class Entities(Lens):
    """Which of `labels` occur in the subject, with where. Empty means none."""

    kind = "entities"
    labels: list[str] = []  # noqa: RUF012

    def parse(self, value: Any) -> list[str]:
        if value is None:
            return []
        items = [value] if isinstance(value, str) else list(value)
        found = [str(item) for item in items]
        if self.labels:
            found = [label for label in found if label in self.labels]
        return sorted(set(found))

    def from_detections(self, labels: list[str]) -> list[str]:
        return self.parse(labels)

    def value_type(self) -> Any:
        return list[str]

    def value_description(self) -> str:
        if self.labels:
            return f"{self.question} Any of: {', '.join(self.labels)}; empty for none."
        return self.question

    def extra_description(self) -> dict[str, Any]:
        return {"labels": list(self.labels)}


class Text(Lens):
    """A short free-text answer: a summary, a reason, a suggested fix."""

    kind = "text"
    max_chars: int = 500

    def parse(self, value: Any) -> str:
        return str(value)[: self.max_chars]

    def from_detections(self, labels: list[str]) -> str:
        return ", ".join(labels)

    def value_type(self) -> Any:
        return str


LENS_KINDS: dict[str, type[Lens]] = {cls.kind: cls for cls in (Flag, Choice, Score, Entities, Text)}


def lens_from_config(config: dict[str, Any]) -> Lens:
    """A lens from a JSON config: settings, a database overlay, an import."""
    config = dict(config)
    kind = config.pop("kind", "")
    if kind not in LENS_KINDS:
        raise ImproperlyConfigured(f"Unknown lens kind {kind!r}; one of {sorted(LENS_KINDS)}")
    return LENS_KINDS[kind](**config)


__all__ = [
    "LENS_KINDS",
    "Choice",
    "Entities",
    "Flag",
    "Lens",
    "Score",
    "Text",
    "lens_from_config",
]
