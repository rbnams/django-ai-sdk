"""Studio: the full-featured demo. Shared settings come from ``demo_base``."""

from __future__ import annotations

from pathlib import Path

import environ

BASE_DIR = Path(__file__).resolve().parent.parent

# Before the shared settings: they read DEBUG, API keys, ... from the environment.
environ.Env.read_env(BASE_DIR / ".env")

from demo_base.settings import *  # noqa: E402

ROOT_URLCONF = "studio.urls"
ASGI_APPLICATION = "studio.asgi.application"
WSGI_APPLICATION = "studio.wsgi.application"

INSTALLED_APPS = [
    *INSTALLED_APPS,
    "studio.integrations.linear",
    "django_ai_sdk.integrations.slack",
    "django_ai_sdk.integrations.telegram",
    "django_ai_sdk.integrations.github_webhook",
    "apps.users",
    "apps.agents",
    "apps.memories",
    "apps.integrations",
]

AUTH_USER_MODEL = "users.User"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

STORAGES = {
    "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
    "staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"},
    # Uploaded documents and thread files, inside this demo's folder.
    "django_ai_sdk": {
        "BACKEND": "django.core.files.storage.FileSystemStorage",
        "OPTIONS": {"location": BASE_DIR},
    },
}


# Agents

AI_SDK_AGENTS = [
    "apps.agents.pirate_basic.PirateBasicAgent",
    "apps.agents.agent_swarm.PirateSwarmAgent",
    "apps.agents.deep_research.DeepResearchAgent",
]

# Base classes available for runtime configured agents
AI_SDK_RUNTIME_AGENT_BASES = [
    "apps.agents.runtime.DefaultRuntimeAgent",
]

# Tools selectable in runtime agent configuration
AI_SDK_RUNTIME_AGENT_TOOLS = {
    "get_today": "apps.agents.tools.get_today",
    "get_memory_files": "apps.agents.tools.get_memory_files",
    "get_memory_file": "apps.agents.tools.get_memory_file",
}

# Workflow composables. The package ships neither step types nor actions; both
# registries are the gate on what a JSON definition may name.
AI_SDK_WORKFLOW_STEPS = {
    "gather_thread": "apps.agents.steps.GatherStep",
    "thread_digest": "apps.agents.steps.DigestStep",
}
AI_SDK_WORKFLOW_ACTIONS = {
    "console_log": "apps.agents.actions.ConsoleLogAction",
    "thread_message": "apps.agents.actions.ThreadMessageAction",
}

# Permission overrides by domain
AI_SDK_PERMISSIONS = {
    "memory": ["apps.memories.permissions.AllowAnonymousMemoryPermission"],
    "thread": ["apps.agents.permissions.DemoThreadPermission"],
}

# Files and images

# A vision-capable model for VisionAgent
AI_SDK_VISION_MODEL = env("AI_SDK_VISION_MODEL", default=None)
AI_SDK_VISION_AGENT = "apps.agents.vision.VisionAgent" if AI_SDK_VISION_MODEL else None

# The thread-file download view
AI_SDK_THREAD_FILE_URL = "api-1.0.0:download_thread_file"

# The shared text types, plus what the studio's PDF and image pipelines read.
AI_SDK_ALLOWED_FILES = {
    **AI_SDK_ALLOWED_FILES,
    ".pdf": "application/pdf",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}

# Integrations are Django apps (see INSTALLED_APPS) that register themselves on ready().
# This dict configures them by name. A missing credential doesn't crash boot: the
# integration reports that it needs setup instead. `weather` needs none at all.
AI_SDK_INTEGRATIONS = {
    "linear": {"TOKEN": env("LINEAR_API_KEY", default="")},
    # Unset by default, so the demo boots without a Slack app or a Telegram bot:
    # the integration reports disconnected and
    # its webhook 404s. An empty ALLOW_FROM answers every sender the platform
    # delivers and an empty ALLOW_WORKSPACES every installation; list platform ids
    # to narrow either. RUN_AS names the account a run acts as: unset answers
    # anonymously, which reaches no integration.
    "slack": {
        "BOT_TOKEN": env("SLACK_BOT_TOKEN", default=""),
        "SIGNING_SECRET": env("SLACK_SIGNING_SECRET", default=""),
        "ALLOW_FROM": env.list("SLACK_ALLOW_FROM", default=[]),
        "ALLOW_WORKSPACES": env.list("SLACK_ALLOW_WORKSPACES", default=[]),
        "RUN_AS": env("SLACK_RUN_AS", default=""),
        "AGENT": "apps.agents.pirate_basic.PirateBasicAgent",
    },
    "telegram": {
        "BOT_TOKEN": env("TELEGRAM_BOT_TOKEN", default=""),
        "WEBHOOK_SECRET": env("TELEGRAM_WEBHOOK_SECRET", default=""),
        # Telegram has no admission control of its own, so the integration refuses
        # to serve until this says who may ask. "*" answers anyone who messages it.
        "ALLOW_FROM": env.list("TELEGRAM_ALLOW_FROM", default=["*"]),
        "RUN_AS": env("TELEGRAM_RUN_AS", default=""),
        "AGENT": "apps.agents.pirate_basic.PirateBasicAgent",
    },
    # Reviews pull requests in the listed repositories. The agent needs the
    # "github" MCP integration in its `integrations` to read the diff.
    "github_webhook": {
        "WEBHOOK_SECRET": env("GITHUB_WEBHOOK_SECRET", default=""),
        "TOKEN": env("GITHUB_TOKEN", default=""),
        "ALLOW_WORKSPACES": env.list("GITHUB_WEBHOOK_REPOS", default=[]),
        "RUN_AS": env("GITHUB_WEBHOOK_RUN_AS", default=""),
        "AGENT": "apps.agents.pirate_basic.PirateBasicAgent",
    },
}

# Seconds a webhook-triggered agent run may take before the channel gets a timeout
# reply instead.
AI_SDK_WEBHOOK_TIMEOUT = 120
