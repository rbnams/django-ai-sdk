from __future__ import annotations

from django.apps import AppConfig
from django.core.checks import register as register_check
from django.utils.module_loading import autodiscover_modules


class OversightConfig(AppConfig):
    """opt-in oversight app: the turn ledger, its lenses and rules"""

    default_auto_field = "django.db.models.BigAutoField"
    name = "django_ai_sdk.oversight"
    label = "django_ai_sdk_oversight"
    verbose_name = "Oversight"

    def ready(self) -> None:
        # The packs first, so an app's policy can extend them by name.
        import django_ai_sdk.oversight.packs  # noqa: F401

        autodiscover_modules("oversight")

        from django_ai_sdk.oversight import signals
        from django_ai_sdk.oversight.checks import check_oversight, check_oversight_deploy

        register_check(check_oversight)
        register_check(check_oversight_deploy, deploy=True)
        # Deleting a thread forgets what oversight holds about it.
        signals.connect()
