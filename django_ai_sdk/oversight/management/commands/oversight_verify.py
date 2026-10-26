"""Walk the oversight ledger and check every link, every record and the last checkpoint."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from django.core.management.base import BaseCommand, CommandError

if TYPE_CHECKING:
    from argparse import ArgumentParser


class Command(BaseCommand):
    help = "Verify the oversight hash chain; exits non-zero on the first tampered record."

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--json", action="store_true", help="Print the result as JSON.")
        parser.add_argument(
            "--verifier",
            default="",
            help="Check checkpoint signatures with this "
            "verifier key (name+hash+key), as printed by oversight_keys.",
        )

    def handle(self, *args: Any, **options: Any) -> None:
        from django_ai_sdk.oversight import seal
        from django_ai_sdk.oversight.models import Checkpoint
        from django_ai_sdk.oversight.services import _verify_all

        result = _verify_all()
        bad_signatures = 0
        if options["verifier"]:
            for checkpoint in Checkpoint.objects.filter(signed=True):
                if not seal.verify_note(checkpoint.note, options["verifier"]):
                    bad_signatures += 1
        if options["json"]:
            self.stdout.write(json.dumps({**result.model_dump(), "bad_signatures": bad_signatures}))
        else:
            self.stdout.write(
                f"{result.checked} entries checked, {result.pruned} pruned, head {result.head[:16]}"
            )
            if result.checkpoint_size is not None:
                state = "matches" if result.checkpoint_ok else "DOES NOT MATCH"
                self.stdout.write(f"checkpoint at {result.checkpoint_size} {state}")
            for problem in result.problems:
                self.stdout.write(self.style.ERROR(json.dumps(problem)))
        if not result.ok or result.checkpoint_ok is False or bad_signatures:
            raise CommandError("The oversight ledger does not verify.")
        if not options["json"]:
            self.stdout.write(self.style.SUCCESS("The ledger verifies."))
