"""A person as an answerer: the lens goes to a review queue and is answered later.

Last in a lens's answerers it means "when the machines are unsure, ask someone". The
verdict lands as an Observation like any other, which is what lets the ledger
measure how often each model agrees with people.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from django_ai_sdk.oversight.answerers.base import Answerer, param
from django_ai_sdk.oversight.types import Answer

if TYPE_CHECKING:
    from django_ai_sdk.oversight.lenses import Lens
    from django_ai_sdk.oversight.types import TurnView


class Human(Answerer):
    """Queue the lens for a person. `priority`: higher is sooner."""

    metered = False
    console: ClassVar[dict[str, Any] | None] = {
        "label": "A person",
        "summary": "The turn goes to a review queue; someone answers later.",
        "params": [
            param("queue", "text", "Queue", default="default"),
            param("priority", "number", "Priority", "Higher is sooner.", default=0),
        ],
    }

    def __init__(self, queue: str = "default", *, priority: int = 0, name: str = "") -> None:
        self.queue = queue
        self.priority = priority
        self.name = name or "human"

    async def answer(self, lens: Lens, view: TurnView) -> Answer | None:
        return Answer(answerer=self.label(), queued=self.queue, confidence=0.0)


__all__ = ["Human"]
