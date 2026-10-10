"""The Automation contract: which workflow runs, when, and as whom.

A declaration only; nothing here starts a clock. It names a workflow rather than
containing one, so one workflow serves chat, the API and a schedule.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from django.core.exceptions import ImproperlyConfigured

from django_ai_sdk.automations.schedule import Cron

if TYPE_CHECKING:
    from datetime import datetime


class Automation:
    """One scheduled workflow. Subclass it in an app's `automations.py` and register."""

    # Registry key. State and run history key on it, so renaming orphans them.
    name: str = ""
    label: str = ""
    description: str = ""

    # 5-field cron expression, days of the week by name (mon-fri). Required.
    cron: str = ""
    # IANA zone the `cron` expression is read in, so a wall-clock time survives DST.
    timezone: str = "UTC"

    # Name of a registered workflow. Required.
    workflow: str = ""

    # The account a run acts as, by the user model's USERNAME_FIELD. Empty runs with
    # no user, as the deployment itself.
    run_as: str = ""

    # The turn the default `get_inputs` supplies as the `prompt` input. {user},
    # {last_run_at} and {name} are substituted.
    input: str = "Run the {name} automation."

    # False keeps it out of the schedule; `run_automations <name>` still runs it.
    enabled: bool = True

    # Bare `{word}` only: str.format would reach `{name.__class__}` and raise on prose.
    _PLACEHOLDER = re.compile(r"\{(\w+)\}")

    def get_inputs(self, *, user: Any = None, last_run_at: datetime | None = None) -> dict:
        """The workflow inputs a run starts from.

        A schedule has no human asking, so the automation writes the first turn. The
        workflow has to declare every key returned here, or the run is skipped.
        """
        return {"prompt": self.render_input(user=user, last_run_at=last_run_at)}

    def get_schedule(self) -> Cron:
        """The effective schedule, straight from the class."""
        if not self.cron:
            raise ImproperlyConfigured(f"Automation {self.name!r} must set `cron`.")
        return Cron(self.cron, tz=self.timezone)

    def render_input(self, *, user: Any = None, last_run_at: datetime | None = None) -> str:
        """Fill the input's {user}, {last_run_at} and {name} placeholders.

        Any other brace is left as written.
        """
        values = {
            "user": user.get_username() if user else "",
            "last_run_at": last_run_at.isoformat() if last_run_at else "the beginning",
            "name": self.name,
        }
        return self._PLACEHOLDER.sub(lambda m: values.get(m.group(1), m.group(0)), self.input)

    def __str__(self) -> str:
        return self.label or self.name or type(self).__name__


__all__ = ["Automation"]
