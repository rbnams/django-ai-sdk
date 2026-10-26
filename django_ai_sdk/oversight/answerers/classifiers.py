"""Purpose-built classifiers: Presidio, Hugging Face models, moderation endpoints.

Faster and cheaper than a judge, sharper than a regex. Each one maps its native
labels onto the lens's type, so the lens never learns which classifier it got.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, ClassVar

from django.core.exceptions import ImproperlyConfigured

from django_ai_sdk.oversight.answerers.base import Answerer
from django_ai_sdk.oversight.types import Answer, TextSpan
from django_ai_sdk.utils import resolve_setting

if TYPE_CHECKING:
    from collections.abc import Iterable

    from django_ai_sdk.oversight.lenses import Lens
    from django_ai_sdk.oversight.types import TurnView


class Presidio(Answerer):
    """Microsoft Presidio's analyzer, local. Needs django-ai-sdk[presidio] and a spaCy model.

    `entities` narrows the recognizers (EMAIL_ADDRESS, IBAN_CODE, PERSON, ...);
    labels come back lower-cased, so an Entities lens can list "person".
    """

    system_one: ClassVar[bool] = True
    timeout = 5.0
    _engine: ClassVar[Any] = None

    def __init__(
        self,
        entities: Iterable[str] | None = None,
        *,
        language: str = "en",
        threshold: float = 0.5,
        name: str = "",
    ) -> None:
        self.entities = list(entities) if entities else None
        self.language = language
        self.threshold = threshold
        self.name = name or "presidio"

    @classmethod
    def engine(cls) -> Any:
        if cls._engine is None:
            try:
                from presidio_analyzer import AnalyzerEngine
            except ImportError as exc:  # pragma: no cover - needs the extra
                raise ImproperlyConfigured(
                    "Presidio needs presidio-analyzer: pip install django-ai-sdk[presidio]"
                ) from exc
            cls._engine = AnalyzerEngine()
        return cls._engine

    async def answer(self, lens: Lens, view: TurnView) -> Answer | None:
        text = view.text(lens.subject)
        if not text:
            return None
        results = await asyncio.to_thread(
            self.engine().analyze,
            text=text,
            entities=self.entities,
            language=self.language,
            score_threshold=self.threshold,
        )
        spans = [TextSpan(start=r.start, end=r.end, label=r.entity_type.lower()) for r in results]
        labels = list(dict.fromkeys(s.label for s in spans))
        top = max((r.score for r in results), default=0.0)
        return Answer(
            value=lens.from_detections(labels),
            confidence=float(top) if spans else 0.85,
            spans=spans,
        )


class HFClassifier(Answerer):
    """A Hugging Face text classifier, local (transformers) or behind an endpoint.

        HFClassifier("meta-llama/Llama-Prompt-Guard-2-86M", positive={"MALICIOUS", "LABEL_1"})
        HFClassifier("ibm-granite/granite-guardian-hap-38m", positive={"LABEL_1"},
                     endpoint="https://my-endpoint.endpoints.huggingface.cloud")

    For a Flag or Score lens, `positive` names the labels that mean yes, and the
    value is their probability. For a Choice lens, `labels` maps model labels to
    the lens's labels.
    """

    timeout = 15.0
    _pipelines: ClassVar[dict[str, Any]] = {}

    def __init__(
        self,
        model: str,
        *,
        positive: Iterable[str] = ("LABEL_1",),
        labels: dict[str, str] | None = None,
        endpoint: str | None = None,
        token_setting: str = "HF_TOKEN",
        max_chars: int = 2_000,
        name: str = "",
    ) -> None:
        self.model = model
        self.positive = set(positive)
        self.labels = labels or {}
        self.endpoint = endpoint
        self.token_setting = token_setting
        self.max_chars = max_chars
        self.name = name or f"hf:{model.rsplit('/', 1)[-1]}"

    @property
    def system_one(self) -> bool:  # ty: ignore[invalid-method-override]
        # Local inference costs nothing per call; an endpoint is billed.
        return self.endpoint is None

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "model": self.model}

    async def classify(self, text: str) -> list[dict[str, Any]]:
        if self.endpoint:
            import httpx

            token = resolve_setting(self.token_setting, "")
            headers = {"Authorization": f"Bearer {token}"} if token else {}
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.endpoint,
                    json={"inputs": text, "parameters": {"top_k": None}},
                    headers=headers,
                )
                response.raise_for_status()
                data = response.json()
            # A batch of one comes back nested.
            return data[0] if data and isinstance(data[0], list) else data
        return await asyncio.to_thread(self._local, text)

    def _local(self, text: str) -> list[dict[str, Any]]:
        if self.model not in self._pipelines:
            try:
                from transformers import pipeline
            except ImportError as exc:  # pragma: no cover - needs the extra
                raise ImproperlyConfigured(
                    "A local HFClassifier needs transformers: pip install django-ai-sdk[transformers]"
                ) from exc
            self._pipelines[self.model] = pipeline(
                "text-classification", model=self.model, top_k=None
            )
        result = self._pipelines[self.model](text)
        return result[0] if result and isinstance(result[0], list) else result

    async def answer(self, lens: Lens, view: TurnView) -> Answer | None:
        text = view.text(lens.subject)[: self.max_chars]
        if not text:
            return None
        scores = {str(item["label"]): float(item["score"]) for item in await self.classify(text)}
        if not scores:
            return None
        if lens.kind == "choice":
            label, score = max(scores.items(), key=lambda item: item[1])
            return Answer(value=self.labels.get(label, label), confidence=score, model=self.model)
        probability = sum(score for label, score in scores.items() if label in self.positive)
        value: Any = probability if lens.kind == "score" else probability >= 0.5
        return Answer(
            value=value,
            confidence=max(probability, 1 - probability),
            model=self.model,
            rationale=", ".join(f"{k} {v:.2f}" for k, v in sorted(scores.items())),
        )


class Moderation(Answerer):
    """OpenAI's moderation endpoint: free, fast, multilingual, for safety lenses.

    A Flag means "flagged", a Score the highest category score, an Entities lens
    the flagged categories.
    """

    timeout = 10.0

    def __init__(self, model: str = "omni-moderation-latest", *, name: str = "") -> None:
        self.model = model
        self.name = name or "moderation"

    async def answer(self, lens: Lens, view: TurnView) -> Answer | None:
        from openai import AsyncOpenAI

        text = view.text(lens.subject)
        if not text:
            return None
        response = await AsyncOpenAI().moderations.create(model=self.model, input=text[:20_000])
        result = response.results[0]
        categories = {k: bool(v) for k, v in result.categories.model_dump().items()}
        scores = {k: float(v or 0) for k, v in result.category_scores.model_dump().items()}
        flagged = sorted(k for k, v in categories.items() if v)
        top = max(scores.values(), default=0.0)
        value: Any
        if lens.kind == "entities":
            value = flagged
        elif lens.kind == "score":
            value = top
        else:
            value = bool(result.flagged)
        return Answer(
            value=value,
            confidence=max(top, 1 - top),
            model=self.model,
            rationale=", ".join(flagged),
            cost=0.0,
        )


__all__ = ["HFClassifier", "Moderation", "Presidio"]
