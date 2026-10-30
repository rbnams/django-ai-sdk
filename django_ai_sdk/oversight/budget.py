"""A daily spend cap for model answerers, so oversight can never outspend the product.

Unmetered answerers (local detectors, TypeSafe, people) are never counted. When the cap
is reached, metered answerers abstain with error "budget" and the lens keeps the
answers it has.
"""

from __future__ import annotations

import contextlib
import contextvars
from collections import defaultdict
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any

from django.db.models import F, Sum

from django_ai_sdk.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = get_logger(__name__)


class Budget:
    """One turn's view of today's spend: read once, added to as answers come in."""

    def __init__(self, cap: float | None, spent: float = 0.0) -> None:
        self.cap = cap
        self.spent = spent
        self.pending: dict[str, list[float]] = defaultdict(list)
        self.day = datetime.now(UTC).date()

    def allows(self) -> bool:
        return (
            self.cap is None or self.spent + sum(sum(v) for v in self.pending.values()) < self.cap
        )

    def charge(self, answerer: str, cost: float | None) -> None:
        self.pending[answerer].append(float(cost or 0))

    async def flush(self) -> None:
        """Add this turn's spend to the day's rows: one update per answerer, a row if new."""
        from django.db import IntegrityError

        from django_ai_sdk.oversight.models import DailySpend

        for answerer, costs in self.pending.items():
            rows = DailySpend.objects.filter(date=self.day, answerer=answerer[:200])
            added = {"cost": F("cost") + sum(costs), "calls": F("calls") + len(costs)}
            if await rows.aupdate(**added):
                continue
            try:
                await DailySpend.objects.acreate(
                    date=self.day, answerer=answerer[:200], cost=sum(costs), calls=len(costs)
                )
            except IntegrityError:
                # Another turn created the row first; add to it instead.
                await rows.aupdate(**added)
        self.pending.clear()


_current: contextvars.ContextVar[Budget | None] = contextvars.ContextVar(
    "django_ai_sdk_oversight_budget", default=None
)


def current() -> Budget | None:
    return _current.get()


def activate(budget: Budget | None) -> contextvars.Token[Budget | None]:
    return _current.set(budget)


def deactivate(token: contextvars.Token[Budget | None]) -> None:
    _current.reset(token)


async def aload(cap: float | None = None) -> Budget:
    from django_ai_sdk.oversight.conf import setting
    from django_ai_sdk.oversight.models import DailySpend

    cap = cap if cap is not None else setting("BUDGET_PER_DAY")
    if cap is None:
        return Budget(None)
    total = await DailySpend.objects.filter(date=datetime.now(UTC).date()).aaggregate(s=Sum("cost"))
    return Budget(float(cap), float(total["s"] or 0))


@contextlib.asynccontextmanager
async def charged() -> AsyncGenerator[Budget]:
    """Today's budget, active for the block and recorded after: for console dry runs."""
    budget = await aload()
    token = activate(budget)
    try:
        yield budget
    finally:
        deactivate(token)
        await budget.flush()


async def spend_by_day(since: date) -> list[dict[str, Any]]:
    from django_ai_sdk.oversight.models import DailySpend

    rows = DailySpend.objects.filter(date__gte=since).order_by("date", "answerer")
    return [
        {
            "date": row.date.isoformat(),
            "answerer": row.answerer,
            "cost": row.cost,
            "calls": row.calls,
        }
        async for row in rows
    ]


__all__ = [
    "Budget",
    "activate",
    "aload",
    "charged",
    "current",
    "deactivate",
    "spend_by_day",
]
