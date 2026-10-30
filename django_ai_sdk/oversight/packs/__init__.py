"""Built-in policies: security, quality, safety, and all of them as default.

Extend a pack whole, or take single lenses and rules from it with `policy.pick`.
Importing this module registers them; OversightConfig.ready() does that.
"""

from __future__ import annotations

from django_ai_sdk.oversight.packs.quality import QualityPack
from django_ai_sdk.oversight.packs.safety import SafetyPack
from django_ai_sdk.oversight.packs.security import SecurityPack
from django_ai_sdk.oversight.policy import Policy
from django_ai_sdk.oversight.registry import register


@register
class DefaultPolicy(Policy):
    """Every pack, every rule in report mode: watch first, decide later."""

    name = "default"
    label = "Default"
    description = "All built-in packs, every rule in report mode."
    origin = "pack:default"
    extends = ["security", "safety", "quality"]  # noqa: RUF012


__all__ = ["DefaultPolicy", "QualityPack", "SafetyPack", "SecurityPack"]
