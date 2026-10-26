"""Retro: replay questions through an agent as it is now, and judge the new answers.

From a JSONL file (one {"input": ..., "expected": {lens: value}} per line), or from
turns people have reviewed (--reviewed), whose human verdicts become the expected
values. Prints, per lens, how often the new answers still get the expected verdict:
run it after a prompt, model or knowledge change, before it ships.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from asgiref.sync import async_to_sync
from django.core.management.base import BaseCommand, CommandError

if TYPE_CHECKING:
    from argparse import ArgumentParser


class Command(BaseCommand):
    help = "Replay a golden set or reviewed turns through an agent and judge the answers."

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("agent", help="Agent id, name or class name.")
        parser.add_argument("--file", default="", help="JSONL golden set.")
        parser.add_argument(
            "--reviewed",
            type=int,
            default=0,
            help="Replay the N most recent turns with a human verdict.",
        )
        parser.add_argument("--policy", default="", help="Judge with this policy instead.")
        parser.add_argument("--json", action="store_true")

    def handle(self, *args: Any, **options: Any) -> None:
        cases = self._cases(options)
        if not cases:
            raise CommandError("Nothing to replay: give --file or --reviewed N.")
        report = async_to_sync(replay)(options["agent"], cases, policy=options["policy"])
        if options["json"]:
            self.stdout.write(json.dumps(report, default=str))
            return
        for lens, stats in sorted(report["lenses"].items()):
            share = f"{stats['kept'] / stats['expected']:.0%}" if stats["expected"] else "-"
            self.stdout.write(f"{lens:28} {stats['kept']}/{stats['expected']} kept  {share}")
        self.stdout.write(f"{report['cases']} cases, {report['failed']} failed to run")

    def _cases(self, options: dict[str, Any]) -> list[dict[str, Any]]:
        if options["file"]:
            with open(options["file"], encoding="utf-8") as handle:
                return [json.loads(line) for line in handle if line.strip()]
        if options["reviewed"]:
            from django_ai_sdk.oversight.models import Observation
            from django_ai_sdk.oversight.tasks import load_evidence

            cases: dict[str, dict[str, Any]] = {}
            human = (
                Observation.objects.exclude(reviewer="")
                .select_related("turn", "turn__evidence")
                .order_by("-created_at")
            )
            for obs in human[: options["reviewed"] * 4]:
                case = cases.get(str(obs.turn_id))
                if case is None:
                    content = load_evidence(obs.turn) or {}
                    if not content.get("input"):
                        continue
                    case = cases[str(obs.turn_id)] = {"input": content["input"], "expected": {}}
                case["expected"].setdefault(obs.lens, obs.value)
                if len(cases) >= options["reviewed"]:
                    break
            return list(cases.values())
        return []


async def replay(
    agent_ref: str, cases: list[dict[str, Any]], *, policy: str = ""
) -> dict[str, Any]:
    from django_ai_sdk.common import ChatMessage
    from django_ai_sdk.oversight.answerers.llm import AgentJudge
    from django_ai_sdk.oversight.metrics import _band
    from django_ai_sdk.oversight.policy import aget_policy
    from django_ai_sdk.oversight.session import _policy_name, unobserved
    from django_ai_sdk.oversight.stages import run_lenses
    from django_ai_sdk.oversight.types import Stage, TurnView

    agent = AgentJudge(agent_ref).resolve_agent()
    resolved = await aget_policy(policy or _policy_name(agent))
    if resolved is None:
        raise CommandError("The agent has no oversight policy; pass --policy.")
    lenses: dict[str, Any] = {}
    failed = 0
    for case in cases:
        try:
            with unobserved():
                output = await agent.run(
                    [ChatMessage(role="user", content=case["input"])], tools=True
                )
        except Exception:
            failed += 1
            continue
        view = TurnView(
            turn_id=f"replay:{hash(case['input'])}", input=case["input"], output=str(output or "")
        )
        expected = case.get("expected") or {}
        judged = [
            lens
            for name, lens in resolved.lenses.items()
            if name in expected and (Stage.REVIEW in lens.stages or Stage.REFLECT in lens.stages)
        ]
        with unobserved():
            observed = await run_lenses(judged, view, Stage.RETRO)
        for o in observed:
            stats = lenses.setdefault(o.lens.name, {"expected": 0, "kept": 0, "answers": []})
            stats["expected"] += 1
            if o.answer.ok and _band(o.answer.value) == _band(expected[o.lens.name]):
                stats["kept"] += 1
            stats["answers"].append(o.answer.value)
    return {"cases": len(cases), "failed": failed, "lenses": lenses}
