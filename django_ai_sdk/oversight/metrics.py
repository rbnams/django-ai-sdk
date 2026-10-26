"""What the ledger says in aggregate: rates, the RAG triad, shadow impact, who agrees.

Computed in Python over a bounded window rather than in SQL over JSON, so it runs
the same on SQLite and Postgres; MAX_ROWS caps what one request reads.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datetime import datetime

MAX_ROWS = 50_000
TRIAD = ("context_relevance", "groundedness", "answer_relevance")


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


def _band(value: Any) -> Any:
    """Scores compare by quarter, booleans and labels as they are."""
    if isinstance(value, float):
        return round(value * 4) / 4
    if isinstance(value, list):
        return tuple(sorted(value))
    return value


async def overview(since: datetime, *, agent_id: str = "", policy: str = "") -> dict[str, Any]:
    from django_ai_sdk.oversight import budget as budgets
    from django_ai_sdk.oversight.conf import setting
    from django_ai_sdk.oversight.models import Decision, Observation, ReviewItem, Turn

    turns = Turn.objects.filter(created_at__gte=since)
    if agent_id:
        turns = turns.filter(agent_id=agent_id)
    if policy:
        turns = turns.filter(policy=policy)
    rows = [
        t
        async for t in turns.only(
            "id",
            "created_at",
            "channel",
            "agent_id",
            "agent_name",
            "agent_fingerprint",
            "outcome",
            "shadow_outcome",
            "labels",
            "cost",
        ).order_by("-created_at")[:MAX_ROWS]
    ]
    ids = [t.id for t in rows]
    by_outcome = Counter(t.outcome for t in rows)
    by_shadow = Counter(t.shadow_outcome for t in rows)
    by_channel = Counter(t.channel.split(":", 1)[0] for t in rows)
    by_agent = Counter(t.agent_name or t.agent_id for t in rows)

    daily: dict[str, dict[str, Any]] = defaultdict(lambda: {"turns": 0, "cost": 0.0})
    for t in rows:
        day = daily[t.created_at.date().isoformat()]
        day["turns"] += 1
        day[t.outcome] = day.get(t.outcome, 0) + 1
        if t.shadow_outcome != t.outcome:
            day["shadow_" + t.shadow_outcome] = day.get("shadow_" + t.shadow_outcome, 0) + 1
        day["cost"] += t.cost or 0

    labels: dict[str, Counter[str]] = defaultdict(Counter)
    for t in rows:
        for lens, value in (t.labels or {}).items():
            if isinstance(value, str):
                labels[lens][value] += 1
            elif isinstance(value, bool):
                labels[lens]["yes" if value else "no"] += 1
            elif isinstance(value, list):
                for item in value:
                    labels[lens][str(item)] += 1

    observations = [
        o
        async for o in Observation.objects.filter(turn_id__in=ids).only(
            "turn_id",
            "lens",
            "answerer",
            "value",
            "confidence",
            "error",
            "cost",
            "latency_ms",
            "reviewer",
        )[:MAX_ROWS]
    ]
    lens_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "answers": 0,
            "errors": 0,
            "positive": 0,
            "score_sum": 0.0,
            "scores": 0,
            "cost": 0.0,
            "latency_sum": 0.0,
            "answerers": Counter(),
        }
    )
    for o in observations:
        stat = lens_stats[o.lens]
        stat["answers"] += 1
        stat["answerers"][o.answerer] += 1
        stat["cost"] += o.cost or 0
        stat["latency_sum"] += o.latency_ms or 0
        if o.error:
            stat["errors"] += 1
        elif o.value is True or (isinstance(o.value, list) and o.value):
            stat["positive"] += 1
        elif isinstance(o.value, int | float) and not isinstance(o.value, bool):
            stat["score_sum"] += float(o.value)
            stat["scores"] += 1
    lenses = [
        {
            "lens": name,
            "answers": s["answers"],
            "errors": s["errors"],
            "positive_rate": round(s["positive"] / s["answers"], 4) if s["answers"] else 0,
            "mean_score": round(s["score_sum"] / s["scores"], 4) if s["scores"] else None,
            "cost": round(s["cost"], 6),
            "mean_latency_ms": round(s["latency_sum"] / s["answers"], 1) if s["answers"] else None,
            "answerers": dict(s["answerers"]),
        }
        for name, s in sorted(lens_stats.items())
    ]

    triad: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    fingerprints = {t.id: (t.agent_name or t.agent_id, t.agent_fingerprint) for t in rows}
    for o in observations:
        if o.lens in TRIAD and isinstance(o.value, int | float) and not o.error:
            triad[fingerprints[o.turn_id]][o.lens].append(float(o.value))
    triad_rows = [
        {
            "agent": agent,
            "fingerprint": fp,
            **{lens: round(sum(v) / len(v), 3) for lens, v in values.items() if v},
            "turns": max((len(v) for v in values.values()), default=0),
        }
        for (agent, fp), values in sorted(triad.items())
    ]

    # Each machine answerer against people, per lens.
    by_turn_lens: dict[tuple[Any, str], dict[str, Any]] = defaultdict(dict)
    for o in observations:
        if o.error or o.value is None:
            continue
        key = "human" if o.reviewer else o.answerer
        by_turn_lens[(o.turn_id, o.lens)][key] = _band(o.value)
    pairs: dict[tuple[str, str], list[tuple[Any, Any]]] = defaultdict(list)
    for (_, lens), answers in by_turn_lens.items():
        if "human" not in answers:
            continue
        for answerer, value in answers.items():
            if answerer != "human":
                pairs[(answerer, lens)].append((value, answers["human"]))
    agreement = [
        {
            "answerer": answerer,
            "lens": lens,
            "pairs": len(p),
            "agreement": round(sum(a == b for a, b in p) / len(p), 3) if p else None,
            "kappa": cohen_kappa(p),
        }
        for (answerer, lens), p in sorted(pairs.items())
    ]

    decisions = [
        d
        async for d in Decision.objects.filter(turn_id__in=ids).only("rule", "mode", "acted")[
            :MAX_ROWS
        ]
    ]
    rule_stats: dict[str, Counter[str]] = defaultdict(Counter)
    for d in decisions:
        rule_stats[d.rule]["matches"] += 1
        rule_stats[d.rule]["acted" if d.acted else "shadow" if d.mode == "shadow" else "late"] += 1
    rules = [
        {
            "rule": name,
            **dict(counts),
            "rate": round(counts["matches"] / len(rows), 4) if rows else 0,
        }
        for name, counts in sorted(rule_stats.items())
    ]

    queues = Counter(
        [q async for q in ReviewItem.objects.filter(status="open").values_list("queue", flat=True)]
    )
    spend = await budgets.spend_by_day(since.date())
    return {
        "since": since,
        "turns": len(rows),
        "by_outcome": dict(by_outcome),
        "by_shadow_outcome": dict(by_shadow),
        "by_channel": dict(by_channel),
        "by_agent": dict(by_agent),
        "daily": [{"date": day, **values} for day, values in sorted(daily.items())],
        "lenses": lenses,
        "rules": rules,
        "labels": {lens: dict(counts) for lens, counts in labels.items()},
        "triad": triad_rows,
        "agreement": agreement,
        "spend": spend,
        "budget_per_day": setting("BUDGET_PER_DAY"),
        "currency": setting("CURRENCY"),
        "queues": dict(queues),
    }


__all__ = ["TRIAD", "cohen_kappa", "overview"]
