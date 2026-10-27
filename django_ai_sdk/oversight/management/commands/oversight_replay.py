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
        from django_ai_sdk.oversight.simulate import replay

        try:
            report = async_to_sync(replay)(options["agent"], cases, policy=options["policy"])
        except ValueError as exc:
            raise CommandError(str(exc)) from exc
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
