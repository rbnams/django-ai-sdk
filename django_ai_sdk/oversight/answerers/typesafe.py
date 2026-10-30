"""TypeSafe as an answerer: typed questions are what it was built for.

A `Classifier` (answerers.classifier): this module only says how a lens becomes a
TypeSafe question and how its judgments read; what is sent, batching and the
per-source questions are the base's.

TypeSafe is a hosted, commercial classification service; this answerer is optional.
It needs django-ai-sdk[typesafe] and TYPESAFE_API_KEY; without either it abstains
(the ai_sdk.oversight.W005 check says so). Flag, Choice and Score lenses map onto its
yes/no ("noul"), choice and score questions; an Entities lens with `per_source` is
answered source by source. A failed request is an `unavailable` error, so FAIL_MODE
applies and reflect asks again. Within one stage, questions about the same state go
to TypeSafe as one request.

What leaves for TypeSafe, per lens (typesafe_state, source_question; characters; a
longer text keeps its start and its end):

    lens subject      history         user_message  assistant_reply  sources [1]          tool_call
    input             last 6, 500 ea  30,000        -                -                    -
    exchange          last 6, 500 ea  30,000        30,000           -                    -
    output            -               30,000        30,000           yes, earlier too     -
    sources           -               30,000        -                yes, this turn only  -
    tool_call         -               30,000        -                -                    name, args [2]
    per_source [3]    -               30,000        30,000           one per question     -

    [1] each source and tool result: its title (200) and text, 8,000 together; all
        of them 32,000 together, past which the rest are left out (earlier turns'
        sources first, as they come last).
    [2] the tool's name (200) and its arguments, as sent to the tool; past 4,000
        characters of JSON, the start and end of that JSON.
    [3] an Entities lens with `per_source` (the quality pack's "taken from"): each
        source goes in a yes/no question of its own, within the same limits.

With them go each lens's question (its question and guidance, and what yes, no,
each label or each score level means) and the model id. One request carries at
most REQUEST_CHARS characters of the turn. Nothing else: no user id, thread id,
sources' urls or metadata. Secrets (the Secrets patterns) are masked before
anything is cut or sent; personal data is not.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from django_ai_sdk.logger import get_logger
from django_ai_sdk.oversight.answerers.base import modules_installed, param, unavailable
from django_ai_sdk.oversight.answerers.classifier import (
    STATES_FACTS,
    Batch,
    Classifier,
    batched,
    certainty,
    says_yes,
)
from django_ai_sdk.oversight.answerers.llm import DEFAULT_LIMITS, HISTORY_ITEM_CHARS, TITLE_CHARS
from django_ai_sdk.oversight.lenses import Choice, Flag, Score
from django_ai_sdk.oversight.types import Answer

if TYPE_CHECKING:
    from django_ai_sdk.oversight.lenses import Lens
    from django_ai_sdk.oversight.types import Source, TurnView

logger = get_logger(__name__)

# Seconds a request may take, retries included; inline stages give it what they have left.
TYPESAFE_TIMEOUT = 20.0
# The most characters of a turn one request carries. TypeSafe reads at most 32k tokens
# per question (the state and that question) and 64k per request; at three or four
# characters a token, this keeps a question with all its state inside that.
REQUEST_CHARS = 100_000
# The question, and the answer: the start and end of a longer text.
TEXT_CHARS = 30_000
# Sources: each (title and text) in full up to SOURCE_CHARS, all of them up to
# SOURCES_CHARS together.
SOURCE_CHARS = 8_000
SOURCES_CHARS = 32_000
# A tool call's arguments past this many characters (as JSON) are sent as cut JSON text.
TOOL_CALL_CHARS = 4_000
# Earlier messages: as many, and as long, as a model judge reads.
HISTORY_CHARS = DEFAULT_LIMITS["history"] * HISTORY_ITEM_CHARS
assert (
    2 * TEXT_CHARS + SOURCES_CHARS + TITLE_CHARS + TOOL_CALL_CHARS + HISTORY_CHARS <= REQUEST_CHARS
)
# The confidence of a judgment that states none.
UNSTATED_CONFIDENCE = 0.8
# How many of a judgment's likeliest answers its rationale names.
RATIONALE_ANSWERS = 3


def typesafe_question(lens: Lens) -> dict[str, Any] | None:
    """`lens` as a TypeSafe question, or None when TypeSafe has no such type."""
    if isinstance(lens, Flag):
        return {
            "type": "noul",
            "instructions": typesafe_instructions(lens),
            "criteria": {"true": lens.yes, "false": lens.no},
        }
    if isinstance(lens, Choice):
        return {
            "type": "choice",
            "instructions": typesafe_instructions(lens),
            "criteria": {label: lens.criteria.get(label, label) for label in lens.labels},
        }
    if isinstance(lens, Score):
        return {
            "type": "score",
            "instructions": typesafe_instructions(lens),
            "criteria": [lens.low, lens.middle, lens.high],
        }
    return None


def typesafe_instructions(lens: Lens) -> str:
    return f"{lens.question}\n{lens.guidance}" if lens.guidance else lens.question


class TypeSafe(Classifier):
    """Answer with TypeSafe, a hosted commercial classification service.

    Needs the `typesafe` extra (pip install 'django-ai-sdk[typesafe]') and a
    TYPESAFE_API_KEY; without either it abstains. `model` pins a model; the default
    is TYPESAFE_MODEL, else the SDK's.

    Flag, Choice and Score lenses become TypeSafe questions (a yes/no is a "noul": a
    probability of yes). An Entities lens with `per_source = True` (the quality
    pack's "taken from") is answered by rating each source, as `Classifier.per_source`
    does. One request carries at most REQUEST_CHARS characters of the turn; the
    module docstring lists what is sent for each lens.
    """

    external: ClassVar[str] = "TypeSafe"
    timeout = TYPESAFE_TIMEOUT
    console: ClassVar[dict[str, Any] | None] = {
        "label": "TypeSafe",
        "summary": "A hosted, paid classification service: answers yes/no, label and score "
        "lenses, and rates each source for a lens about which sources an answer is taken "
        "from. Needs TYPESAFE_API_KEY.",
        "params": [
            param("model", "model", "Model", "Empty uses TYPESAFE_MODEL.", kind="TypeSafe"),
        ],
    }
    TEXT_CHARS = TEXT_CHARS
    SOURCE_CHARS = SOURCE_CHARS
    SOURCES_CHARS = SOURCES_CHARS
    TOOL_CALL_CHARS = TOOL_CALL_CHARS

    def __init__(self, model: str | None = None, *, name: str = "") -> None:
        self.model = model
        self.name = name or "typesafe"

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "model": self.model or ""}

    @classmethod
    def installed(cls) -> bool:
        return modules_installed("typesafe_sdk")

    def missing(self) -> str:
        from django_ai_sdk.oversight.answerers._typesafe_client import typesafe_missing

        return typesafe_missing()

    def question(self, lens: Lens) -> dict[str, Any] | None:
        return typesafe_question(lens)

    def source_question(self, source: Source) -> dict[str, Any]:
        return source_question(source)

    async def classify(
        self, state: dict[str, Any], questions: dict[str, Any], *, timeout: float | None = None
    ) -> dict[str, Answer]:
        from django_ai_sdk.oversight.answerers._typesafe_client import request

        if timeout is None:
            summary = await request(state, questions, model=self.model)
        else:
            from typesafe_sdk import RetryPolicy

            # Inline, a retry would land after the deadline: one attempt, given the time left.
            summary = await request(
                state,
                questions,
                model=self.model,
                timeout=timeout,
                retry=RetryPolicy(max_retries=0),
            )
        answers = {}
        for name, question in questions.items():
            judged = judgment_answer(summary, name, question, self.model)
            if judged is not None:
                answers[name] = judged
        return answers

    def _failed(self, lens: Lens, exc: Exception) -> Answer:
        # The type only: an SDK error's text can carry the response body.
        logger.warning("TypeSafe failed on lens {}: {}", lens.name, type(exc).__name__)
        return Answer(error=unavailable(exc))


def typesafe_state(lens: Lens, view: TurnView) -> dict[str, Any]:
    """The state TypeSafe reads for `lens`: `Classifier.state` within TypeSafe's limits."""
    return TypeSafe().state(lens, view)


def labelled_sources(view: TurnView, *, earlier: bool = False) -> list[tuple[str, Source]]:
    """The turn's evidence as TypeSafe reads it, each labelled S1, S2, ... in order."""
    return TypeSafe().labelled_sources(view, earlier=earlier)


def source_question(source: Source) -> dict[str, Any]:
    """TypeSafe's yes/no question about one source, the source inside it.

    `source` as labelled_sources gives it, already masked and cut; masked and cut
    again here, so a source from elsewhere is held to the same limits.
    """
    reader = TypeSafe()
    return {
        "type": "noul",
        "instructions": {
            "source": {
                "title": reader.text(source.title, TITLE_CHARS),
                "content": reader.text(source.content, SOURCE_CHARS),
            },
            "question": "Is `assistant_reply` taken from `source`?",
        },
        "criteria": {
            "true": "The reply states at least one fact that is in the source and relies on it.",
            "false": (
                "The reply's facts are not in the source: it only shares the topic, or it "
                "says something else."
            ),
        },
    }


def judgment_answer(
    summary: dict[str, Any], name: str, question: dict[str, Any], model: str | None = None
) -> Answer | None:
    """TypeSafe's judgment on question `name` as an Answer; None when it gave none."""
    judgment = next((j for j in summary.get("judgments") or [] if j.get("id") == name), None)
    if judgment is None:
        return None
    value = judgment.get("value")
    confidence = judgment.get("confidence")
    if judgment.get("type") == "noul" and isinstance(value, float):
        # A noul is a probability of yes; its distance from 0.5 is its certainty.
        confidence = certainty(value)
    elif judgment.get("type") == "score" and isinstance(value, int | float):
        # A score is a level index: 0 to levels - 1, so 1 of 0..2 is the middle.
        levels = len(judgment.get("legend") or {}) or len(question.get("criteria") or [])
        if levels > 1:
            value = min(1.0, max(0.0, value / (levels - 1)))
    probabilities = judgment.get("probabilities") or {}
    if not confidence and probabilities:
        # A score's spread over its levels: the likeliest level's share is how sure.
        confidence = max(float(p or 0) for p in probabilities.values())
    return Answer(
        value=value,
        confidence=float(confidence) if confidence is not None else UNSTATED_CONFIDENCE,
        model=str(summary.get("model") or model or ""),
        rationale=_rationale(judgment, question),
    )


def _rationale(judgment: dict[str, Any], question: dict[str, Any] | None = None) -> str:
    """The likeliest answers with their probabilities, a score's levels by their names."""
    probabilities = judgment.get("probabilities") or {}
    if not probabilities:
        value = judgment.get("value")
        if judgment.get("type") == "noul" and isinstance(value, int | float):
            return f"yes {float(value):.2f}"
        return ""
    criteria = (question or {}).get("criteria")
    names = list(criteria) if isinstance(criteria, list) else []

    def label_name(label: str) -> str:
        if names and str(label).isdigit() and int(label) < len(names):
            return names[int(label)]
        return str(label)

    top = sorted(probabilities.items(), key=lambda item: -float(item[1] or 0))[:RATIONALE_ANSWERS]
    return ", ".join(f"{label_name(label)} {float(p):.2f}" for label, p in top)


__all__ = [
    "REQUEST_CHARS",
    "STATES_FACTS",
    "Batch",
    "TypeSafe",
    "batched",
    "judgment_answer",
    "labelled_sources",
    "says_yes",
    "source_question",
    "typesafe_instructions",
    "typesafe_question",
    "typesafe_state",
]
