"""Generate oversight keys, or print the verifier key for the configured signing key."""

from __future__ import annotations

from typing import Any

from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Print new MASTER_KEY and SIGNING_KEY values, and the public verifier key."

    def handle(self, *args: Any, **options: Any) -> None:
        from django_ai_sdk.oversight import seal, vault
        from django_ai_sdk.oversight.conf import setting, signing_name

        configured = setting("SIGNING_KEY")
        if configured:
            self.stdout.write(f"verifier key: {seal.verifier_key(signing_name(), configured)}")
            return
        self.stdout.write(f"MASTER_KEY  = {vault.generate_master_key()!r}")
        key = seal.generate_signing_key()
        self.stdout.write(f"SIGNING_KEY = {key!r}")
        self.stdout.write(f"verifier key: {seal.verifier_key(signing_name(), key)}")
