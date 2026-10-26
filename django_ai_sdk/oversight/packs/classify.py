"""The classify pack: label every message, so the ledger can be sliced by what people ask.

The generic intent labels are a starting point: declare a Choice lens named
`intent` in your own policy, with your own labels and keywords, and it replaces
this one. With a `fast` alias, a model labels what keywords cannot.
"""

from __future__ import annotations

from django_ai_sdk.oversight.answerers import Keywords, Language
from django_ai_sdk.oversight.lenses import Choice
from django_ai_sdk.oversight.policy import Policy
from django_ai_sdk.oversight.registry import register
from django_ai_sdk.oversight.types import Purpose, Stage, Subject


class Intent(Choice):
    name = "intent"
    label = "Intent"
    question = "What is the person trying to do with this message?"
    labels = ["question", "task", "complaint", "feedback", "smalltalk", "other"]
    criteria = {  # noqa: RUF012
        "question": "Wants to know something.",
        "task": "Wants something done or made.",
        "complaint": "Is unhappy about something.",
        "feedback": "Comments on the assistant or a previous answer.",
        "smalltalk": "Greets, thanks or chats.",
    }
    subject = Subject.INPUT
    stages = (Stage.GATE,)
    answerers = [
        Keywords(
            {
                "smalltalk": ["hi", "hello", "hey", "thanks", "thank you", "hoi", "hallo", "dank"],
                "complaint": ["not working", "broken", "terrible", "werkt niet", "klacht"],
                "feedback": ["wrong answer", "that's wrong", "klopt niet", "good answer"],
                "task": ["write", "create", "make", "draft", "schrijf", "maak"],
                "question": [
                    "what",
                    "how",
                    "why",
                    "when",
                    "where",
                    "who",
                    "wat",
                    "hoe",
                    "waarom",
                    "wanneer",
                    "waar",
                    "wie",
                ],
            }
        ),
        "fast?",
    ]
    purposes = (Purpose.QUALITY,)


class InputLanguage(Choice):
    name = "language"
    label = "Language"
    question = "Which language is the message written in?"
    labels = ["en", "nl", "de", "fr", "es", "other"]
    subject = Subject.INPUT
    stages = (Stage.GATE,)
    answerers = [Language()]
    purposes = (Purpose.QUALITY,)


@register
class ClassifyPack(Policy):
    name = "classify"
    label = "Classify"
    description = "Every message gets an intent and a language."
    origin = "pack:classify"
    lenses = [Intent, InputLanguage]


__all__ = ["ClassifyPack"]
