"""What the ledger says in aggregate: rates, the RAG triad, shadow impact, who agrees.

Counts over plain columns are SQL aggregates. What depends on a JSON answer
(labels, positive rates, the triad, agreement) is classified in Python, over at
most MAX_ROWS rows newest first, so it runs the same on SQLite and Postgres.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import UTC
from typing import TYPE_CHECKING, Any

from django.db.models import Count, Q, Sum
from django.db.models.functions import TruncDate

from django_ai_sdk.oversight.packs.quality import AnswerRelevance, ContextRelevance, Groundedness

if TYPE_CHECKING:
    from datetime import datetime

    from django.db.models import QuerySet

MAX_ROWS = 50_000
TRIAD = (ContextRelevance.name, Groundedness.name, AnswerRelevance.name)

# One answer as the Python passes read it: turn, lens, value, reviewer, answerer, error.
AnswerRow = tuple[Any, str, Any, str, str, str]


def cohen_kappa(pairs: list[tuple[Any, Any]]) -> float | None:
    """Agreement between two raters beyond chance; None for fewer than 5 pairs."""
    if len(pairs) < 5:
        return None
    labels = sorted({str(x) for pair in pairs for x in pair})
    n = len(pairs)
    observed = sum(str(a) == str(b) for a, b in pairs) / n
    first = Counter(str(a) for a, _ in pairs)
    second = Counter(str(b) for _, b in pairs)
    expected = sum(first[label] * second[label] for label in labels) / (n * n)
    if expected >= 1:
        return 1.0
    return round((observed - expected) / (1 - expected), 3)


def band(value: Any) -> Any:
    """Scores compare by quarter, booleans and labels as they are."""
    if isinstance(value, float):
        return round(value * 4) / 4
    if isinstance(value, list):
        return tuple(sorted(value))
    return value


async def overview(since: datetime, *, agent_id: str = "", policy: str = "") -> dict[str, Any]:
    from django_ai_sdk.oversight import budget as budgets
    from django_ai_sdk.oversight.conf import setting
    from django_ai_sdk.oversight.models import Observation, ReviewItem, Turn

    turns = Turn.objects.filter(created_at__gte=since)
    # The same window, for rows that hang off a turn.
    related: dict[str, Any] = {"turn__created_at__gte": since}
    if agent_id:
        turns = turns.filter(agent_id=agent_id)
        related["turn__agent_id"] = agent_id
    if policy:
        turns = turns.filter(policy=policy)
        related["turn__policy"] = policy
    counts = await _counts(turns)
    window = [
        t
        async for t in turns.only(
            "id", "agent_id", "agent_name", "agent_fingerprint", "labels"
        ).order_by("-created_at")[:MAX_ROWS]
    ]
    answers: list[AnswerRow] = [
        row
        async for row in Observation.objects.filter(**related)
        .order_by("-created_at")
        .values_list("turn_id", "lens", "value", "reviewer", "answerer", "error")[:MAX_ROWS]
    ]
    queues = Counter(
        [q async for q in ReviewItem.objects.filter(status="open").values_list("queue", flat=True)]
    )
    return {
        "since": since,
        **counts,
        "lenses": await _lens_stats(Observation.objects.filter(**related), answers),
        "rules": await _rule_stats(related, counts["turns"]),
        "labels": _labels(window),
        "triad": _triad(window, answers),
        "agreement": _agreement(answers),
        "spend": await budgets.spend_by_day(since.date()),
        "budget_per_day": setting("BUDGET_PER_DAY"),
        "currency": setting("CURRENCY"),
        "queues": dict(queues),
    }


async def _counts(turns: QuerySet[Any]) -> dict[str, Any]:
    """Turns by outcome, shadow outcome, channel, agent and day, from one grouped query."""
    by_outcome: Counter[str] = Counter()
    by_shadow: Counter[str] = Counter()
    by_channel: Counter[str] = Counter()
    by_agent: Counter[str] = Counter()
    daily: dict[str, dict[str, Any]] = defaultdict(lambda: {"turns": 0, "cost": 0.0})
    groups = (
        turns.order_by()
        .values(
            "outcome",
            "shadow_outcome",
            "channel",
            "agent_id",
            "agent_name",
            day=TruncDate("created_at", tzinfo=UTC),
        )
        .annotate(n=Count("id"), spent=Sum("cost"))
    )
    async for row in groups:
        n, outcome, shadow = row["n"], row["outcome"], row["shadow_outcome"]
        by_outcome[outcome] += n
        by_shadow[shadow] += n
        by_channel[row["channel"].split(":", 1)[0]] += n
        by_agent[row["agent_name"] or row["agent_id"]] += n
        day = daily[row["day"].isoformat()]
        day["turns"] += n
        day[outcome] = day.get(outcome, 0) + n
        if shadow != outcome:
            day["shadow_" + shadow] = day.get("shadow_" + shadow, 0) + n
        day["cost"] += row["spent"] or 0
    return {
        "turns": sum(by_outcome.values()),
        "by_outcome": dict(by_outcome),
        "by_shadow_outcome": dict(by_shadow),
        "by_channel": dict(by_channel),
        "by_agent": dict(by_agent),
        "daily": [{"date": day, **values} for day, values in sorted(daily.items())],
    }


def _labels(window: list[Any]) -> dict[str, dict[str, int]]:
    labels: dict[str, Counter[str]] = defaultdict(Counter)
    for t in window:
        for lens, value in (t.labels or {}).items():
            if isinstance(value, str):
                labels[lens][value] += 1
            elif isinstance(value, bool):
                labels[lens]["yes" if value else "no"] += 1
            elif isinstance(value, list):
                for item in value:
                    labels[lens][str(item)] += 1
    return {lens: dict(counts) for lens, counts in labels.items()}


async def _lens_stats(observations: QuerySet[Any], answers: list[AnswerRow]) -> list[dict]:
    """Per lens: answers, errors, cost and latency in SQL; positives and scores from values."""
    stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "answers": 0,
            "errors": 0,
            "cost": 0.0,
            "latency_sum": 0.0,
            "answerers": Counter(),
            "classified": 0,
            "positive": 0,
            "score_sum": 0.0,
            "scores": 0,
        }
    )
    grouped = (
        observations.order_by()
        .values("lens", "answerer")
        .annotate(
            n=Count("id"),
            errors=Count("id", filter=~Q(error="")),
            spent=Sum("cost"),
            latency=Sum("latency_ms"),
        )
    )
    async for row in grouped:
        stat = stats[row["lens"]]
        stat["answers"] += row["n"]
        stat["errors"] += row["errors"]
        stat["answerers"][row["answerer"]] += row["n"]
        stat["cost"] += row["spent"] or 0
        stat["latency_sum"] += row["latency"] or 0
    for _, lens, value, _, _, error in answers:
        stat = stats[lens]
        stat["classified"] += 1
        if error:
            continue
        if value is True or (isinstance(value, list) and value):
            stat["positive"] += 1
        elif isinstance(value, int | float) and not isinstance(value, bool):
            stat["score_sum"] += float(value)
            stat["scores"] += 1
    return [
        {
            "lens": name,
            "answers": s["answers"],
            "errors": s["errors"],
            "positive_rate": round(s["positive"] / s["classified"], 4) if s["classified"] else 0,
            "mean_score": round(s["score_sum"] / s["scores"], 4) if s["scores"] else None,
            "cost": round(s["cost"], 6),
            "mean_latency_ms": round(s["latency_sum"] / s["answers"], 1) if s["answers"] else None,
            "answerers": dict(s["answerers"]),
        }
        for name, s in sorted(stats.items())
    ]


def _triad(window: list[Any], answers: list[AnswerRow]) -> list[dict[str, Any]]:
    """The RAG triad's mean scores per agent version."""
    versions = {t.id: (t.agent_name or t.agent_id, t.agent_fingerprint) for t in window}
    triad: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for turn_id, lens, value, _, _, error in answers:
        version = versions.get(turn_id)
        if version and lens in TRIAD and isinstance(value, int | float) and not error:
            triad[version][lens].append(float(value))
    return [
        {
            "agent": agent,
            "fingerprint": fp,
            **{lens: round(sum(v) / len(v), 3) for lens, v in values.items() if v},
            "turns": max((len(v) for v in values.values()), default=0),
        }
        for (agent, fp), values in sorted(triad.items())
    ]


def _agreement(answers: list[AnswerRow]) -> list[dict[str, Any]]:
    """Each machine answerer against people, per lens."""
    by_turn_lens: dict[tuple[Any, str], dict[str, Any]] = defaultdict(dict)
    # Oldest first, so a lens's latest answer per answerer is the one compared.
    for turn_id, lens, value, reviewer, answerer, error in reversed(answers):
        if error or value is None:
            continue
        by_turn_lens[(turn_id, lens)]["human" if reviewer else answerer] = band(value)
    pairs: dict[tuple[str, str], list[tuple[Any, Any]]] = defaultdict(list)
    for (_, lens), values in by_turn_lens.items():
        if "human" not in values:
            continue
        for answerer, value in values.items():
            if answerer != "human":
                pairs[(answerer, lens)].append((value, values["human"]))
    return [
        {
            "answerer": answerer,
            "lens": lens,
            "pairs": len(p),
            "agreement": round(sum(a == b for a, b in p) / len(p), 3) if p else None,
            "kappa": cohen_kappa(p),
        }
        for (answerer, lens), p in sorted(pairs.items())
    ]


async def _rule_stats(related: dict[str, Any], turns: int) -> list[dict[str, Any]]:
    from django_ai_sdk.oversight.models import Decision

    stats: dict[str, Counter[str]] = defaultdict(Counter)
    grouped = (
        Decision.objects.filter(**related)
        .order_by()
        .values("rule", "mode", "acted")
        .annotate(n=Count("id"))
    )
    async for row in grouped:
        counts = stats[row["rule"]]
        counts["matches"] += row["n"]
        state = "acted" if row["acted"] else "shadow" if row["mode"] == "shadow" else "late"
        counts[state] += row["n"]
    return [
        {"rule": name, **dict(counts), "rate": round(counts["matches"] / turns, 4) if turns else 0}
        for name, counts in sorted(stats.items())
    ]


__all__ = ["TRIAD", "band", "cohen_kappa", "overview"]
