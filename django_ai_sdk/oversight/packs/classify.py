"""The classify pack: label every message, so the ledger can be sliced by what people ask.

Jev labels both at the gate. The generic intent labels are a starting point:
declare a Choice lens named `intent` in your own policy, with your own labels and
criteria, and it replaces this one.
"""

from __future__ import annotations

from django_ai_sdk.oversight.lenses import Choice
from django_ai_sdk.oversight.policy import Policy
from django_ai_sdk.oversight.registry import register
from django_ai_sdk.oversight.types import Stage, Subject


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
    answerers = ["jev?"]


class InputLanguage(Choice):
    name = "language"
    label = "Language"
    question = "Which language is the message written in?"
    labels = ["en", "nl", "de", "fr", "es", "other"]
    criteria = {  # noqa: RUF012
        "en": "English",
        "nl": "Dutch",
        "de": "German",
        "fr": "French",
        "es": "Spanish",
        "other": "Another language, or too little text to tell",
    }
    subject = Subject.INPUT
    stages = (Stage.GATE,)
    answerers = ["jev?"]


@register
class ClassifyPack(Policy):
    name = "classify"
    label = "Classify"
    description = "Every message gets an intent and a language."
    origin = "pack:classify"
    lenses = [Intent, InputLanguage]


__all__ = ["ClassifyPack", "InputLanguage", "Intent"]
