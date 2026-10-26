"""Apply oversight retention: shred expired evidence, drop expired metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from django.core.management.base import BaseCommand

if TYPE_CHECKING:
    from argparse import ArgumentParser


class Command(BaseCommand):
    help = "Forget what AI_SDK_OVERSIGHT['RETENTION'] says has expired; --thread forgets one."

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--thread", default="", help="Crypto-shred one thread's evidence now.")

    def handle(self, *args: Any, **options: Any) -> None:
        from django_ai_sdk.oversight.tasks import prune, shred_scope

        if options["thread"]:
            done = shred_scope(f"thread:{options['thread']}")
            self.stdout.write("Shredded." if done else "Already shredded.")
            return
        counts = prune()
        self.stdout.write(
            f"Shredded {counts['evidence_shredded']} evidence rows, "
            f"dropped {counts['rows_dropped']} expired rows."
        )
