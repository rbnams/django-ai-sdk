"""Automation declarations for the studio, autodiscovered on startup.

Disabled, so the demo makes no model calls on a timer:
`manage.py run_automations harbour-report` runs it once.
"""

from __future__ import annotations

from django_ai_sdk.automations import Automation, register


@register
class HarbourReport(Automation):
    name = "harbour-report"
    label = "Harbour report"
    description = "The Rotterdam forecast as a ship's log entry."
    enabled = False

    cron = "0 7 * * mon-fri"
    timezone = "Europe/Amsterdam"
    workflow = "harbour-report"
    input = "Report the harbour conditions since {last_run_at}."
