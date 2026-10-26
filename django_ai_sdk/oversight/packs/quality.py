"""The quality pack: was the answer any good, and was retrieval?

The RAG triad (context relevance, groundedness, answer relevance) runs in the
background, so it costs the person nothing in latency. Lexical overlap answers
each first; with a `judge` alias defined, a model is asked whenever overlap is
unsure, which, by design, is most of the time.
"""

from __future__ import annotations

from django_ai_sdk.oversight.answerers import (
    EmptyRetrieval,
    Language,
    LexicalSupport,
    Refusal,
    ToolErrors,
)
from django_ai_sdk.oversight.conditions import L
from django_ai_sdk.oversight.lenses import Flag, Score
from django_ai_sdk.oversight.policy import Policy
from django_ai_sdk.oversight.registry import register
from django_ai_sdk.oversight.rules import Alert, Queue, Rule
from django_ai_sdk.oversight.types import Purpose, Stage, Subject


class ContextRelevance(Score):
    name = "context_relevance"
    label = "Context relevance"
    question = "How relevant are the retrieved sources to the user's question?"
    guidance = (
        "1 when the sources contain what is needed to answer; 0.5 when they are on topic "
        "but miss the specific point; 0 when they are about something else."
    )
    low, high = "unrelated", "exactly what is needed"
    subject = Subject.SOURCES
    stages = (Stage.REFLECT,)
    answerers = [LexicalSupport(), "judge?"]
    purposes = (Purpose.QUALITY,)


class Groundedness(Score):
    name = "groundedness"
    label = "Groundedness"
    question = "How much of the answer is supported by the retrieved sources?"
    guidance = (
        "Judge claims, not wording. 1 when every factual claim is in the sources; 0 when "
        "the answer's claims are absent from or contradict them. General knowledge and "
        "pleasantries neither help nor hurt."
    )
    low, high = "unsupported", "fully supported"
    subject = Subject.OUTPUT
    stages = (Stage.REFLECT,)
    answerers = [LexicalSupport(), "judge?"]
    purposes = (Purpose.QUALITY,)


class AnswerRelevance(Score):
    name = "answer_relevance"
    label = "Answer relevance"
    question = "How directly does the answer address what the user asked?"
    low, high = "beside the point", "fully addresses it"
    subject = Subject.EXCHANGE
    stages = (Stage.REFLECT,)
    answerers = ["judge?"]
    purposes = (Purpose.QUALITY,)


class Helpfulness(Score):
    name = "helpfulness"
    label = "Helpfulness"
    question = "How helpful would this answer be to the person who asked?"
    guidance = "Consider correctness, completeness and clarity together."
    low, high = "useless", "excellent"
    subject = Subject.EXCHANGE
    stages = (Stage.REFLECT,)
    answerers = ["judge?"]
    sample = 0.25
    purposes = (Purpose.QUALITY,)


class Refused(Flag):
    name = "refusal"
    label = "Refusal"
    question = "Does the assistant decline to answer?"
    subject = Subject.OUTPUT
    stages = (Stage.REVIEW,)
    answerers = [Refusal()]
    purposes = (Purpose.QUALITY,)


class LanguageMismatch(Flag):
    name = "language_mismatch"
    label = "Language mismatch"
    question = "Is the answer in a different language than the question?"
    subject = Subject.EXCHANGE
    stages = (Stage.REVIEW,)
    answerers = [Language()]
    purposes = (Purpose.QUALITY,)


class NothingFound(Flag):
    name = "empty_retrieval"
    label = "Nothing retrieved"
    question = "Did the assistant look things up and find nothing?"
    subject = Subject.SOURCES
    stages = (Stage.REVIEW,)
    answerers = [EmptyRetrieval()]
    purposes = (Purpose.QUALITY,)


class FailedTools(Score):
    name = "tool_errors"
    label = "Tool failures"
    question = "What share of the turn's tool calls failed?"
    subject = Subject.EXCHANGE
    stages = (Stage.REVIEW,)
    answerers = [ToolErrors()]
    purposes = (Purpose.QUALITY,)


@register
class QualityPack(Policy):
    name = "quality"
    label = "Quality"
    description = "The RAG triad, refusals, language and tool health."
    origin = "pack:quality"
    lenses = [
        ContextRelevance,
        Groundedness,
        AnswerRelevance,
        Helpfulness,
        Refused,
        LanguageMismatch,
        NothingFound,
        FailedTools,
    ]
    rules = [
        Rule(
            "review_ungrounded",
            when=(L("groundedness") < 0.5) & L("groundedness").confident(0.7),
            then=Queue("quality"),
            description="Answers the sources do not support go to a person.",
        ),
        Rule(
            "review_irrelevant_context",
            when=(L("context_relevance") < 0.3) & L("context_relevance").confident(0.7),
            then=Queue("retrieval"),
            description="Retrieval that found the wrong things, to tune the knowledge base.",
        ),
        Rule(
            "note_empty_retrieval",
            when=L("empty_retrieval").is_true,
            then=Alert("info"),
        ),
        Rule(
            "note_tool_failures",
            when=L("tool_errors") >= 0.5,
            then=Alert("warning"),
        ),
    ]


__all__ = ["QualityPack"]
