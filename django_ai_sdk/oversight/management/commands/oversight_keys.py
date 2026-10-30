"""Generate oversight keys, or print the verifier key for the configured signing key."""

from __future__ import annotations

from typing import Any

from django.core.exceptions import ImproperlyConfigured
from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Print new MASTER_KEY and SIGNING_KEY values, and the public verifier key."

    def handle(self, *args: Any, **options: Any) -> None:
        from django_ai_sdk.oversight import seal, vault
        from django_ai_sdk.oversight.conf import setting, signing_name

        configured = setting("SIGNING_KEY")
        try:
            if configured:
                self.stdout.write(f"verifier key: {seal.verifier_key(signing_name(), configured)}")
                return
            key = seal.generate_signing_key()
            verifier = seal.verifier_key(signing_name(), key)
        except ImproperlyConfigured as exc:
            raise CommandError(str(exc)) from exc
        self.stdout.write(f"MASTER_KEY  = {vault.generate_master_key()!r}")
        self.stdout.write(f"SIGNING_KEY = {key!r}")
        self.stdout.write(f"verifier key: {verifier}")
