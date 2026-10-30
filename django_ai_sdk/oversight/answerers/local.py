"""A classifier that runs in this process: an ONNX model from the Hugging Face Hub.

No torch and nothing leaves the machine: onnxruntime and tokenizers run the model,
huggingface-hub fetches it once into its cache. Install `django-ai-sdk[local]`.

Two kinds of model:

- **Zero-shot (NLI)**, the default: a model trained to say whether a text entails a
  statement answers any yes/no, label or score lens. A yes/no is whether the text
  entails the lens's `yes` (or, left at its default, its question); a label, which
  of its labels' criteria it entails most, each read as "This text is about
  <criterion>." (`template`); a score, how much it entails `high`. Write `yes` as a
  statement about the text: "This text asks to ignore previous instructions."
  reads far better than a question or "The message tells the assistant ...".
- **A fixed classifier** (`positive=` the label that means yes): a model trained for
  one question, such as a prompt-injection detector, answers yes/no lenses with the
  probability of that label, whatever the lens asks.

    LocalClassifier()                                        # English, 87 MB
    LocalClassifier(model=MULTILINGUAL)                      # 100 languages, 340 MB
    LocalClassifier(model="protectai/deberta-v3-base-prompt-injection-v2",
                    file="onnx/model.onnx", positive="INJECTION")

It reads one text, the one the lens judges: the question, or the answer for a lens
on the answer, secrets masked, at most TEXT_CHARS characters; a longer text is too
long, and the lens asks its next answerer, as do lenses on sources or a tool call,
lenses that need evidence, and "taken from". Until its model is loaded it steps aside
and loads it in the background (`load()` does it at startup).
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
from typing import TYPE_CHECKING, Any, ClassVar

from django_ai_sdk.oversight.answerers.base import modules_installed, param
from django_ai_sdk.oversight.answerers.classifier import Classifier, certainty
from django_ai_sdk.oversight.lenses import Choice, Flag, Score
from django_ai_sdk.oversight.types import Answer, Subject

if TYPE_CHECKING:
    from django_ai_sdk.oversight.lenses import Lens

# A secret's mask ("[SLACK_TOKEN]"): a small model reads bracketed capitals as markup
# smuggled into the text, so it reads them as words instead ("(slack token)").
_MASK = re.compile(r"\[([A-Z][A-Z0-9_]*)\]")

# Small, English, MIT: MoritzLaurer's zero-shot DeBERTa, quantized to int8.
ENGLISH = "MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33"
# Multilingual (XNLI, 100 languages), MIT.
MULTILINGUAL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
QUANTIZED = "onnx/model_quantized.onnx"
# Tokens a DeBERTa reads. TEXT_CHARS keeps a Latin-script text well inside it; a text
# that takes more tokens than that (Chinese, say) is cut at its end by the tokenizer.
MAX_TOKENS = 512


class LocalClassifier(Classifier):
    """Answer yes/no, label and score lenses with an ONNX model in this process."""

    external: ClassVar[str] = ""
    metered = False
    timeout = 10.0
    # About 400 tokens: room for the statement beside it within MAX_TOKENS.
    TEXT_CHARS: ClassVar[int] = 1_600
    console: ClassVar[dict[str, Any] | None] = {
        "label": "Local model",
        "summary": "A small model in this process answers yes/no, label and score lenses in "
        "milliseconds. Nothing leaves the server.",
        "params": [
            param(
                "model",
                "choice",
                "Model",
                "English is small and fast; multilingual reads 100 languages.",
                choices=[ENGLISH, MULTILINGUAL],
                default=ENGLISH,
            ),
        ],
    }

    @classmethod
    def installed(cls) -> bool:
        return modules_installed("onnxruntime", "tokenizers", "huggingface_hub")

    def __init__(
        self,
        model: str = ENGLISH,
        *,
        file: str = QUANTIZED,
        positive: str = "",
        template: str = "This text is about {}.",
        name: str = "",
    ) -> None:
        self.model = model
        self.file = file
        self.positive = positive
        self.template = template
        self.name = name or "local"

    def describe(self) -> dict[str, Any]:
        return {
            **super().describe(),
            "model": self.model,
            "file": self.file,
            "positive": self.positive,
        }

    def missing(self) -> str:
        if not self.installed():
            return "needs django-ai-sdk[local] (onnxruntime, tokenizers, huggingface-hub)"
        if error := _load_errors.get((self.model, self.file)):
            return error
        runner = _loaded(self.model, self.file)
        wanted = self.positive.lower() or "entailment"
        if runner is not None and wanted not in runner.labels:
            hint = "" if self.positive else ": a fixed classifier needs positive="
            return f"{self.model} has no label {wanted!r} ({', '.join(runner.labels)}){hint}"
        return ""

    async def answer(self, lens: Lens, view: Any) -> Answer | None:
        # Until the model is loaded, step aside and load it in the background: the
        # first turn after a deploy goes to the next answerer instead of the deadline.
        if _loaded(self.model, self.file) is None:
            _load_in_background(self.model, self.file)
            return None
        return await super().answer(lens, view)

    async def per_source(self, lens: Lens, view: Any) -> Answer | None:
        # A zero-shot model reads one text and one statement: not an answer beside a source.
        return None

    def question(self, lens: Lens) -> dict[str, Any] | None:
        # One text at a time: the question or the answer, never the sources or a tool
        # call beside it, so lenses that need those are for another answerer.
        if lens.subject not in {Subject.INPUT, Subject.OUTPUT} or lens.needs_evidence:
            return None
        reads = "output" if lens.subject is Subject.OUTPUT else "input"
        if self.positive:
            # A fixed classifier knows one question: it answers yes/no lenses only.
            return {"kind": "fixed", "reads": reads} if isinstance(lens, Flag) else None
        if isinstance(lens, Flag):
            statement = lens.yes if lens.yes not in {"", "Yes."} else lens.question
            return {"kind": "flag", "reads": reads, "statements": [statement]}
        if isinstance(lens, Choice):
            statements = [self.about(lens.criteria.get(label) or label) for label in lens.labels]
            return {
                "kind": "choice",
                "reads": reads,
                "labels": list(lens.labels),
                "statements": statements,
            }
        if isinstance(lens, Score):
            return {"kind": "score", "reads": reads, "statements": [lens.high]}
        return None

    def text(self, text: str, limit: int) -> str:
        return _MASK.sub(
            lambda m: f"({m.group(1).lower().replace('_', ' ')})", super().text(text, limit)
        )

    def about(self, criterion: str) -> str:
        """A label's criterion as a statement about the text: "This text is about leave."."""
        text = criterion.strip().rstrip(".")
        if text.lower().startswith("this text"):
            return f"{text}."
        return self.template.format(text[:1].lower() + text[1:])

    async def classify(
        self, state: dict[str, Any], questions: dict[str, Any], *, timeout: float | None = None
    ) -> dict[str, Answer]:
        texts = {
            "input": state.get("user_message", ""),
            "output": state.get("assistant_reply", ""),
        }
        return await asyncio.to_thread(self._classify, texts, questions)

    def _classify(self, texts: dict[str, str], questions: dict[str, Any]) -> dict[str, Answer]:
        runner = _loaded(self.model, self.file)
        if runner is None:
            return {}
        answers: dict[str, Answer] = {}
        for name, question in questions.items():
            text = texts[question["reads"]]
            if not text.strip():
                continue
            if question["kind"] == "fixed":
                [probabilities] = runner.run([(text, None)])
                yes = probabilities[runner.label_index(self.positive)]
                answers[name] = self._yes(yes, f"{self.positive} {yes:.2f}")
                continue
            entailed = runner.entailment([(text, s) for s in question["statements"]])
            if question["kind"] == "flag":
                answers[name] = self._yes(entailed[0], f"entails {entailed[0]:.2f}")
            elif question["kind"] == "score":
                answers[name] = Answer(value=entailed[0], confidence=0.6, model=self.model)
            else:
                total = sum(entailed) or 1.0
                shares = [e / total for e in entailed]
                best = max(range(len(shares)), key=shares.__getitem__)
                ranked = sorted(zip(question["labels"], shares, strict=True), key=lambda x: -x[1])
                answers[name] = Answer(
                    value=question["labels"][best],
                    confidence=shares[best],
                    model=self.model,
                    rationale=", ".join(f"{label} {share:.2f}" for label, share in ranked[:3]),
                )
        return answers

    def _yes(self, probability: float, rationale: str) -> Answer:
        return Answer(
            value=probability,
            confidence=certainty(probability),
            model=self.model,
            rationale=rationale,
        )


class _Runner:
    """One model's session, tokenizer and labels; thread-safe for `run`."""

    def __init__(self, model: str, file: str) -> None:
        import onnxruntime
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        folder = file.rsplit("/", 1)[0] + "/" if "/" in file else ""
        self.session = onnxruntime.InferenceSession(
            hf_hub_download(model, file), providers=["CPUExecutionProvider"]
        )
        try:
            tokenizer_file = hf_hub_download(model, "tokenizer.json")
        except Exception:
            tokenizer_file = hf_hub_download(model, f"{folder}tokenizer.json")
        self.tokenizer = Tokenizer.from_file(tokenizer_file)
        self.tokenizer.enable_truncation(max_length=MAX_TOKENS)
        self.tokenizer.enable_padding()
        with open(hf_hub_download(model, "config.json"), encoding="utf-8") as f:
            id2label = json.load(f).get("id2label") or {}
        self.labels = [str(id2label[k]).lower() for k in sorted(id2label, key=int)]
        self.inputs = {i.name for i in self.session.get_inputs()}
        self.lock = threading.Lock()

    def label_index(self, label: str) -> int:
        return self.labels.index(label.lower())

    def run(self, pairs: list[tuple[str, str | None]]) -> list[list[float]]:
        """Each (text, statement) pair's label probabilities."""
        import numpy as np

        if all(statement is None for _, statement in pairs):
            encoded = self.tokenizer.encode_batch([text for text, _ in pairs])
        else:
            encoded = self.tokenizer.encode_batch([(text, s or "") for text, s in pairs])
        feed = {
            "input_ids": np.array([e.ids for e in encoded], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encoded], dtype=np.int64),
            "token_type_ids": np.array([e.type_ids for e in encoded], dtype=np.int64),
        }
        with self.lock:
            [logits] = self.session.run(None, {k: v for k, v in feed.items() if k in self.inputs})
        logits = np.asarray(logits, dtype=np.float64)
        exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
        return (exp / exp.sum(axis=-1, keepdims=True)).tolist()

    def entailment(self, pairs: list[tuple[str, str]]) -> list[float]:
        """How much each text entails its statement: entailment against contradiction.

        As Hugging Face's zero-shot pipeline reads it; a three-way model's "neutral"
        is set aside.
        """
        index = self.label_index("entailment")
        against = next((i for i, label in enumerate(self.labels) if "contradiction" in label), None)
        if against is None:
            against = next(i for i, label in enumerate(self.labels) if i != index)
        scores = []
        for row in self.run(list(pairs)):
            yes, no = row[index], row[against]
            scores.append(yes / (yes + no) if yes + no else 0.5)
        return scores


# Loaded models by (model, file), and why one could not load.
_runners: dict[tuple[str, str], _Runner] = {}
_load_errors: dict[tuple[str, str], str] = {}
_loading: set[tuple[str, str]] = set()
_load_lock = threading.Lock()


def _loaded(model: str, file: str) -> _Runner | None:
    return _runners.get((model, file))


def load(model: str = ENGLISH, file: str = QUANTIZED) -> None:
    """Load a model now (downloading it once), say at startup, so no turn waits for it."""
    key = (model, file)
    with _load_lock:
        if key in _runners:
            return
        try:
            runner = _Runner(model, file)
        except Exception as exc:
            _load_errors[key] = f"{model} does not load: {type(exc).__name__}: {exc}"[:300]
            raise
        _runners[key] = runner
        _load_errors.pop(key, None)


def _load_in_background(model: str, file: str) -> None:
    key = (model, file)
    with _load_lock:
        if key in _loading or key in _load_errors:
            return
        _loading.add(key)

    def run() -> None:
        try:
            load(model, file)
        except Exception:
            pass  # kept in _load_errors, which missing() reports
        finally:
            _loading.discard(key)

    threading.Thread(target=run, name=f"oversight-load-{model}", daemon=True).start()


__all__ = ["ENGLISH", "MULTILINGUAL", "LocalClassifier", "load"]
