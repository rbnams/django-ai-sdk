"""Local answerers: deterministic, free, fast enough for the gate.

They never call a model, so they add no cost and no data processor. Use them for
structural facts (lengths, tool failures, empty retrieval) and exact spans; a
judgment call is Jev's. A hit answers with confidence 1; nothing found answers
with the answerer's `miss_confidence`, so a later answerer can still overrule a
quiet "no".
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, ClassVar

from django_ai_sdk.oversight.answerers.base import Answerer
from django_ai_sdk.oversight.types import Answer, Subject, TextSpan

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from django_ai_sdk.oversight.lenses import Lens
    from django_ai_sdk.oversight.types import TurnView

# Detectors read at most this much of a subject: it bounds every pattern's cost,
# and the spans still index the text, since they fall in its prefix.
MAX_SCAN_CHARS = 50_000


class Detector(Answerer):
    """Base for answerers that find labelled spans in the subject's text."""

    metered: ClassVar[bool] = False
    timeout = 2.0
    # How sure "found nothing" is. Below the lens's `confident`, the next answerer is asked.
    miss_confidence: float = 0.9

    def detect(self, text: str, view: TurnView, lens: Lens) -> list[TextSpan]:
        raise NotImplementedError

    def looks_for(self) -> list[str]:
        """The kinds this detector knows, so a miss can say what it looked for."""
        return []

    async def answer(self, lens: Lens, view: TurnView) -> Answer | None:
        text = view.text(lens.subject)
        if not text:
            return None
        spans = self.detect(text[:MAX_SCAN_CHARS], view, lens)
        labels = list(dict.fromkeys(span.label for span in spans))
        notes = [f"found {_listed(labels)}"] if labels else []
        if not labels and (kinds := self.looks_for()):
            notes.append(f"found none of: {_listed(kinds)}")
        if len(text) > MAX_SCAN_CHARS:
            notes.append(f"scanned the first {MAX_SCAN_CHARS:,} characters")
        return Answer(
            value=lens.from_detections(labels),
            confidence=1.0 if spans else self.miss_confidence,
            spans=spans,
            rationale="; ".join(notes),
        )


def _listed(kinds: list[str]) -> str:
    return ", ".join(kind.replace("_", " ") for kind in kinds)


class Patterns(Detector):
    """Regular expressions by label. A list of patterns all share the lens's name."""

    def __init__(
        self,
        patterns: Mapping[str, str | Iterable[str]] | Iterable[str],
        *,
        flags: int = re.IGNORECASE,
        miss_confidence: float = 0.9,
        name: str = "",
    ) -> None:
        if isinstance(patterns, dict):
            items = patterns.items()
        else:
            items = [("", patterns)]
        self.patterns: list[tuple[str, re.Pattern[str]]] = []
        for label, group in items:
            for pattern in [group] if isinstance(group, str) else group:
                self.patterns.append((label, re.compile(pattern, flags)))
        self.miss_confidence = miss_confidence
        self.name = name or "patterns"

    def detect(self, text: str, view: TurnView, lens: Lens) -> list[TextSpan]:
        return [
            TextSpan(start=m.start(), end=m.end(), label=label or lens.name)
            for label, pattern in self.patterns
            for m in pattern.finditer(text)
        ]


class Keywords(Detector):
    """Whole-word keywords by label; the label with the most hits comes first.

    For fixed vocabularies such as product or competitor names: `Keywords({"acme":
    ["acme"], "globex": ["globex"]})` answers an Entities or Choice lens.
    """

    def __init__(
        self, keywords: Mapping[str, Iterable[str]], *, miss_confidence: float = 0.5, name: str = ""
    ) -> None:
        self.keywords = {
            label: re.compile(r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\b", re.I)
            for label, words in keywords.items()
            if words
        }
        self.miss_confidence = miss_confidence
        self.name = name or "keywords"

    def detect(self, text: str, view: TurnView, lens: Lens) -> list[TextSpan]:
        hits: list[tuple[int, list[TextSpan]]] = []
        for label, pattern in self.keywords.items():
            spans = [
                TextSpan(start=m.start(), end=m.end(), label=label) for m in pattern.finditer(text)
            ]
            if spans:
                hits.append((len(spans), spans))
        hits.sort(key=lambda item: -item[0])
        return [span for _, spans in hits for span in spans]


class Length(Answerer):
    """The subject's length against `max_chars`: a Flag when over, a Score of how far."""

    metered: ClassVar[bool] = False

    def __init__(self, max_chars: int, name: str = "") -> None:
        self.max_chars = max_chars
        self.name = name or "length"

    async def answer(self, lens: Lens, view: TurnView) -> Answer | None:
        size = len(view.text(lens.subject))
        if lens.kind == "score":
            return Answer(value=min(1.0, size / self.max_chars), rationale=f"{size} characters")
        return Answer(value=size > self.max_chars, rationale=f"{size} characters")


# A handful of the most frequent words per language: enough to tell them apart.
_STOPWORDS: dict[str, frozenset[str]] = {
    "en": frozenset("the and is are was to of in that it for you with on this have be not".split()),
    "nl": frozenset("de het een en is van dat niet je ik op te zijn met voor maar wat".split()),
    "de": frozenset("der die das und ist nicht ich sie es zu mit den von ein eine auf".split()),
    "fr": frozenset("le la les et est un une des que pas je vous il de pour dans sur".split()),
    "es": frozenset("el la los las y es un una que no de en por para con se lo".split()),
}
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def guess_language(text: str) -> tuple[str, float]:
    """(language code, confidence) from stopword counts; ("", 0) for too little text."""
    words = [w.lower() for w in _WORD.findall(text)]
    if len(words) < 3:
        return "", 0.0
    counts = {lang: sum(w in stop for w in words) for lang, stop in _STOPWORDS.items()}
    lang, best = max(counts.items(), key=lambda item: item[1])
    total = sum(counts.values())
    if not best:
        return "", 0.0
    return lang, round(best / total, 3)


class Language(Answerer):
    """The subject's language (a Choice), or whether output matches input's (a Flag)."""

    metered: ClassVar[bool] = False

    def __init__(self, name: str = "") -> None:
        self.name = name or "language"

    async def answer(self, lens: Lens, view: TurnView) -> Answer | None:
        if lens.kind == "flag":
            asked, _ = guess_language(view.input)
            replied, confidence = guess_language(view.output)
            if not asked or not replied:
                return None
            return Answer(
                value=asked != replied,
                confidence=confidence,
                rationale=f"asked in {asked}, answered in {replied}",
            )
        lang, confidence = guess_language(view.text(lens.subject))
        if not lang:
            return None
        return Answer(value=lang, confidence=confidence)


_REFUSAL = re.compile(
    r"\b(I can(?:no|')t help|I(?:'m| am) (?:not able|unable) to|I (?:won't|will not) "
    r"(?:help|provide)|sorry, (?:but )?I can(?:no|')t|ik kan (?:je )?(?:daar )?niet (?:mee )?helpen"
    r"|dat kan ik niet)",
    re.IGNORECASE,
)


class Refusal(Detector):
    """Whether the answer declines. Too many refusals is a quality signal, not a win."""

    miss_confidence = 0.8

    def __init__(self, name: str = "") -> None:
        self.name = name or "refusal"

    def detect(self, text: str, view: TurnView, lens: Lens) -> list[TextSpan]:
        return [
            TextSpan(start=m.start(), end=m.end(), label=lens.name) for m in _REFUSAL.finditer(text)
        ]


class EmptyRetrieval(Answerer):
    """Whether the turn found no sources while it did call a tool to look."""

    metered: ClassVar[bool] = False

    def __init__(self, name: str = "") -> None:
        self.name = name or "empty_retrieval"

    async def answer(self, lens: Lens, view: TurnView) -> Answer | None:
        looked = bool(view.tool_calls)
        found = len(view.fresh_sources())
        return Answer(value=looked and not found, rationale=f"{found} sources")


class ToolErrors(Answerer):
    """The share of the turn's tool calls that failed: a Score, or a Flag for any."""

    metered: ClassVar[bool] = False

    def __init__(self, name: str = "") -> None:
        self.name = name or "tool_errors"

    async def answer(self, lens: Lens, view: TurnView) -> Answer | None:
        if not view.tool_calls:
            return None
        failed = sum(call.error for call in view.tool_calls)
        if lens.kind == "flag":
            return Answer(value=bool(failed), rationale=f"{failed} failed")
        return Answer(value=failed / len(view.tool_calls), rationale=f"{failed} failed")


# Words too common to say anything about support.
_FILLER = frozenset(
    "a an the and or but is are was were be been to of in on at for with by from as it its this "
    "that these those you your we our they their he she his her i me my not no yes do does did "
    "can could will would should may might de het een en of maar is zijn was van in op voor met "
    "door als dat dit die je jij u wij we zij hij ik niet geen wel".split()
)
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")


def _terms(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text) if len(w) > 2 and w.lower() not in _FILLER}


class LexicalSupport(Answerer):
    """Local RAG checks by word overlap. A floor under the model judges, not a peer.

    By the lens's subject:
      OUTPUT/EXCHANGE: groundedness, the share of answer sentences most of whose
        content words appear in some source.
      SOURCES: context relevance, the share of the question's content words that
        appear in the sources.
    A Score gets the share; a Flag means "below `threshold`".
    """

    metered: ClassVar[bool] = False

    def __init__(
        self,
        *,
        overlap: float = 0.5,
        threshold: float = 0.5,
        confidence: float = 0.6,
        name: str = "",
    ) -> None:
        self.overlap = overlap
        self.threshold = threshold
        # Overlap is a weak signal; leave room for a judge to overrule it.
        self.confidence = confidence
        self.name = name or "lexical_support"

    async def answer(self, lens: Lens, view: TurnView) -> Answer | None:
        sources = [_terms(s.content) for s in view.fresh_sources() if s.content]
        if not sources:
            return None
        if lens.subject is Subject.SOURCES:
            asked = _terms(view.input)
            if not asked:
                return None
            pooled = set().union(*sources)
            share = len(asked & pooled) / len(asked)
            detail = f"{len(asked & pooled)} of {len(asked)} question terms in sources"
        else:
            sentences = [s for s in _SENTENCE.split(view.output) if len(_terms(s)) >= 3]
            if not sentences:
                return None
            supported = sum(
                any(len(_terms(s) & src) / len(_terms(s)) >= self.overlap for src in sources)
                for s in sentences
            )
            share = supported / len(sentences)
            detail = f"{supported} of {len(sentences)} sentences supported"
        value: Any = round(share, 3) if lens.kind == "score" else share < self.threshold
        return Answer(value=value, confidence=self.confidence, rationale=detail)


__all__ = [
    "Detector",
    "EmptyRetrieval",
    "Keywords",
    "Language",
    "Length",
    "LexicalSupport",
    "Patterns",
    "Refusal",
    "ToolErrors",
    "guess_language",
]
