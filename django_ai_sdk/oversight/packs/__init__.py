"""Built-in policies to extend: security, quality, safety, classify, and all of them as default.

Importing this module registers them; OversightConfig.ready() does that.
"""

from __future__ import annotations

from django_ai_sdk.oversight.packs.classify import ClassifyPack
from django_ai_sdk.oversight.packs.quality import QualityPack
from django_ai_sdk.oversight.packs.safety import SafetyPack
from django_ai_sdk.oversight.packs.security import SecurityPack
from django_ai_sdk.oversight.policy import Policy
from django_ai_sdk.oversight.registry import register


@register
class DefaultPolicy(Policy):
    """Every pack, every rule in shadow: watch first, decide later."""

    name = "default"
    label = "Default"
    description = "All built-in packs, every rule in shadow mode."
    origin = "pack:default"
    extends = ["security", "safety", "quality", "classify"]  # noqa: RUF012


__all__ = ["ClassifyPack", "DefaultPolicy", "QualityPack", "SafetyPack", "SecurityPack"]
