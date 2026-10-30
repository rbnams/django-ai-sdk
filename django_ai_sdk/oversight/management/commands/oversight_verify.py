"""Walk the oversight ledger and check every link, every record and the newest checkpoint."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from django.core.management.base import BaseCommand, CommandError

if TYPE_CHECKING:
    from argparse import ArgumentParser

    from django_ai_sdk.oversight.models import Checkpoint


class Command(BaseCommand):
    help = "Verify the oversight hash chain and the newest checkpoint; exits non-zero on a problem."

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--json", action="store_true", help="Print the result as JSON.")
        parser.add_argument(
            "--verifier",
            default="",
            help="Check checkpoint signatures with this "
            "verifier key (name+hash+key), as printed by oversight_keys.",
        )
        parser.add_argument(
            "--since-checkpoint",
            action="store_true",
            help="Walk only the entries after the newest checkpoint signed by --verifier (or "
            "the configured SIGNING_KEY). Entries before it are checked against its root; "
            "their records are not read again.",
        )

    def handle(self, *args: Any, **options: Any) -> None:
        from django_ai_sdk.oversight import ledger, seal
        from django_ai_sdk.oversight.models import Checkpoint

        bad_signatures = 0
        if options["verifier"]:
            try:
                bad_signatures = sum(
                    not seal.signed_by(c, options["verifier"])
                    for c in Checkpoint.objects.filter(signed=True)
                )
            except ValueError as exc:
                raise CommandError("--verifier is not a verifier key (name+hash+key).") from exc
        newest = Checkpoint.objects.order_by("-tree_size").first()
        checkpoint_ok = None if newest is None else seal.matches(newest)
        anchor = self._anchor(options["verifier"]) if options["since_checkpoint"] else None
        start = 1
        if options["since_checkpoint"] and anchor is None:
            self.stderr.write("No checkpoint carries that signature: walking the whole chain.")
        elif anchor is not None and (checkpoint_ok if anchor == newest else seal.matches(anchor)):
            start = anchor.tree_size + 1
        result = ledger.verify(start=start)
        if anchor is not None and start == 1:
            # A signed root the ledger no longer gives: history was rewritten before it.
            result.fail(anchor.tree_size, "checkpoint does not match", tree_size=anchor.tree_size)
        if options["json"]:
            out = {
                **result.as_dict(),
                "start": start,
                "checkpoint_ok": checkpoint_ok,
                "checkpoint_size": newest.tree_size if newest else None,
                "bad_signatures": bad_signatures,
            }
            self.stdout.write(json.dumps(out))
        else:
            since = f" after the signed checkpoint at {start - 1}" if start > 1 else ""
            self.stdout.write(
                f"{result.checked} entries checked{since}, {result.pruned} pruned, "
                f"head {result.head[:16]}"
            )
            if newest is not None:
                state = "matches" if checkpoint_ok else "DOES NOT MATCH"
                self.stdout.write(f"checkpoint at {newest.tree_size} {state}")
            if bad_signatures:
                self.stdout.write(self.style.ERROR(f"{bad_signatures} checkpoint(s) badly signed"))
            for problem in result.problems:
                self.stdout.write(self.style.ERROR(json.dumps(problem)))
        if not result.ok or checkpoint_ok is False or bad_signatures:
            raise CommandError("The oversight ledger does not verify.")
        if not options["json"]:
            self.stdout.write(self.style.SUCCESS("The ledger verifies."))

    def _anchor(self, verifier: str) -> Checkpoint | None:
        """The newest checkpoint signed by `verifier`, or by the configured key."""
        from django_ai_sdk.oversight import seal
        from django_ai_sdk.oversight.conf import setting, signing_name
        from django_ai_sdk.oversight.models import Checkpoint

        key = setting("SIGNING_KEY")
        trusted = verifier or (seal.verifier_key(signing_name(), key) if key else "")
        if not trusted:
            raise CommandError(
                "--since-checkpoint starts from a signed checkpoint: pass --verifier, "
                "or set AI_SDK_OVERSIGHT['SIGNING_KEY']."
            )
        checkpoints = Checkpoint.objects.filter(signed=True).order_by("-tree_size")
        return next((c for c in checkpoints.iterator() if seal.signed_by(c, trusted)), None)
