"""Checkpoint the oversight ledger: a signed Merkle root, sent to anchoring exporters."""

from __future__ import annotations

from typing import Any

from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Write a checkpoint over every ledger entry added since the last one."

    def handle(self, *args: Any, **options: Any) -> None:
        from django_ai_sdk.oversight.tasks import seal

        made = seal()
        if made is None:
            self.stdout.write("Nothing new to seal.")
            return
        signed = "signed" if made["signed"] else "unsigned"
        self.stdout.write(
            self.style.SUCCESS(
                f"Sealed {made['tree_size']} entries, root {made['root_hash'][:16]} ({signed})."
            )
        )
