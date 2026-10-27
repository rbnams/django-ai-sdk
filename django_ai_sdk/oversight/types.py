"""The vocabulary every part of oversight shares: what is judged, when, and how it went."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class Subject(StrEnum):
    """What a lens looks at."""

    INPUT = "input"
    OUTPUT = "output"
    # The question and the answer together.
    EXCHANGE = "exchange"
    # What retrieval handed the model, against the question.
    SOURCES = "sources"
    # One tool call's name and arguments, before it runs.
    TOOL_CALL = "tool_call"


class Stage(StrEnum):
    """When a lens runs. Only gate, watch and review can still change the turn."""

    # Before the question is stored or reaches the model.
    GATE = "gate"
    # While the answer streams, and before each tool call.
    WATCH = "watch"
    # Once the answer is complete, before it is stored.
    REVIEW = "review"
    # In the background, after the turn is on the ledger.
    REFLECT = "reflect"
    # Replaying stored turns against a changed agent.
    RETRO = "retro"


# Stages that run while the person waits, in order.
INLINE_STAGES = (Stage.GATE, Stage.WATCH, Stage.REVIEW)


class Mode(StrEnum):
    """What a rule does when its condition holds."""

    OFF = "off"
    # Recorded as what it would have done; the turn is untouched.
    SHADOW = "shadow"
    ENFORCE = "enforce"


class Capture(StrEnum):
    """How much of a turn's content the ledger keeps."""

    # Hashes and decisions only.
    NONE = "none"
    # Lengths, labels, sources' titles and urls, tool names; no text.
    METADATA = "metadata"
    # Text, with every redaction a rule found applied.
    REDACTED = "redacted"
    FULL = "full"


class Purpose(StrEnum):
    """Why someone reads the ledger; it decides what they are shown."""

    # The turn itself: question, answer, sources, tool calls, verdicts and reasons.
    REVIEW = "review"
    # Decisions, bare verdicts and the chain; no content.
    AUDIT = "audit"


class Severity(StrEnum):
    """How urgent an alert or a notification is, least first."""

    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"


SEVERITY_ORDER = ["", *Severity]


def most_severe(*levels: str) -> str:
    """The most urgent of `levels`; "" when there are none."""
    return max(
        levels, key=lambda s: SEVERITY_ORDER.index(s) if s in SEVERITY_ORDER else 0, default=""
    )


class Outcome(StrEnum):
    """What happened to a turn, worst first."""

    BLOCKED = "blocked"
    RESTRICTED = "restricted"
    REDACTED = "redacted"
    FLAGGED = "flagged"
    ALLOWED = "allowed"


OUTCOME_ORDER = list(Outcome)


def worst(*outcomes: Outcome) -> Outcome:
    """The most severe of `outcomes`; ALLOWED when there are none."""
    present = [o for o in outcomes if o is not None]
    return min(present, key=OUTCOME_ORDER.index) if present else Outcome.ALLOWED


class TextSpan(BaseModel):
    """A stretch of a subject's text an answerer pointed at."""

    start: int
    end: int
    label: str = ""


class Answer(BaseModel):
    """One answerer's reply to one lens.

    `value` is typed by the lens: bool for a Flag, a label for a Choice, 0..1 for a
    Score, a list of labels for Entities, a string for Text.
    """

    value: Any = None
    # How sure the answerer is, 0..1. A detector's hit is 1; its miss, its miss_confidence.
    confidence: float = 1.0
    rationale: str = ""
    spans: list[TextSpan] = Field(default_factory=list)

    answerer: str = ""
    model: str = ""
    latency_ms: float | None = None
    tokens: int | None = None
    cost: float | None = None
    # Answers given on the way to this one, by earlier answerers in the lens's list.
    trail: list[Answer] = Field(default_factory=list)
    # Why there is no value: the answerer failed, timed out or ran out of budget.
    error: str = ""
    # The review queue a human answer was asked from; it arrives later.
    queued: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and self.value is not None


class Source(BaseModel):
    """One retrieved source, as the model saw it."""

    key: str = ""
    title: str = ""
    url: str = ""
    content: str = ""
    kind: str = ""
    doc_id: str = ""
    chunk_id: str = ""
    memory_id: str = ""
    # Carried over from an earlier turn rather than found for this one.
    seeded: bool = False


class ToolCall(BaseModel):
    """One tool call of the turn."""

    id: str = ""
    name: str
    arguments: Any = None
    result: Any = None
    error: bool = False
    agent: str = ""


class HistoryItem(BaseModel):
    role: str
    content: str


class TurnView(BaseModel):
    """Everything a lens may look at, at the moment it runs."""

    turn_id: str = ""
    channel: str = ""
    agent_id: str = ""
    agent_name: str = ""
    input: str = ""
    output: str = ""
    history: list[HistoryItem] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    # Set only while a TOOL_CALL lens judges one call.
    tool_call: ToolCall | None = None
    # Answers so far, by lens name, so one lens can read another.
    answers: dict[str, Answer] = Field(default_factory=dict)

    def text(self, subject: Subject) -> str:
        """The plain text a text-only answerer reads for `subject`."""
        if subject is Subject.INPUT:
            return self.input
        if subject is Subject.OUTPUT:
            return self.output
        if subject is Subject.TOOL_CALL:
            call = self.tool_call
            return f"{call.name} {_compact(call.arguments)}" if call else ""
        if subject is Subject.SOURCES:
            return "\n\n".join(s.content for s in self.fresh_sources())
        return f"{self.input}\n\n{self.output}"

    def fresh_sources(self) -> list[Source]:
        """Sources found for this turn, not those carried over from earlier ones."""
        return [s for s in self.sources if not s.seeded]

    def evidence(self) -> list[Source]:
        """What the answer could rest on: this turn's sources, then each tool's result.

        A weather tool's reply grounds "12°C and rain" the way a retrieved chunk does,
        so judges of groundedness read both.
        """
        results = [
            Source(title=f"{call.name} result", content=_compact(call.result), kind="tool")
            for call in self.tool_calls
            if not call.error and call.result not in (None, "", [], {})
        ]
        return [*self.fresh_sources(), *results]


def _compact(value: Any) -> str:
    import json

    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        return str(value)


__all__ = [
    "INLINE_STAGES",
    "Answer",
    "Capture",
    "HistoryItem",
    "Mode",
    "Outcome",
    "Purpose",
    "SEVERITY_ORDER",
    "Severity",
    "Source",
    "Stage",
    "Subject",
    "TextSpan",
    "ToolCall",
    "TurnView",
    "most_severe",
    "worst",
]
