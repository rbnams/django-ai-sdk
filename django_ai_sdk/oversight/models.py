"""The oversight ledger and what hangs off it.

Rows that record what happened (turns, observations, decisions, access, overlay
changes) are each chained into `LedgerEntry` when written and never updated
after, except for the denormalized fields marked as such. Content lives apart,
in `Evidence`, so it can expire or be shredded while the chain still verifies:
the turn's text, and everything that can quote it (judges' reasons, `Text`
answers, names an `Entities` answerer gave in its own words, per-item scores by
key, reviewers' notes). The rows here keep values, labels, scores, counts, errors.
"""

from __future__ import annotations

import uuid

from django.conf import settings
from django.db import models

from .types import Purpose

APP_LABEL = "django_ai_sdk_oversight"


class LedgerEntry(models.Model):
    """One link of the hash chain. `seq` is gapless; `entry_hash` covers the one before."""

    seq = models.BigIntegerField(primary_key=True)
    kind = models.CharField(max_length=32)
    # The id of the row this entry commits to.
    ref = models.CharField(max_length=64, db_index=True)
    payload_hash = models.CharField(max_length=64)
    prev_hash = models.CharField(max_length=64)
    entry_hash = models.CharField(max_length=64, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        app_label = APP_LABEL
        db_table = "django_ai_sdk_oversight_ledger"
        ordering = ["seq"]

    def __str__(self) -> str:
        return f"#{self.seq} {self.kind} {self.entry_hash[:12]}"


class LedgerHead(models.Model):
    """A single row whose lock serializes appends to the chain."""

    id = models.PositiveSmallIntegerField(primary_key=True, default=1)
    seq = models.BigIntegerField(default=0)
    entry_hash = models.CharField(max_length=64, default="0" * 64)

    class Meta:
        app_label = APP_LABEL
        db_table = "django_ai_sdk_oversight_ledger_head"


class DataKey(models.Model):
    """The key that encrypts one scope's evidence (a thread, or a lone turn).

    Deleting `wrapped` is crypto-shredding: every evidence row of the scope becomes
    unreadable at once, and the ledger, which only holds hashes, still verifies.
    """

    id = models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True)
    scope = models.CharField(max_length=100, unique=True)
    wrapped = models.BinaryField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    shredded_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        app_label = APP_LABEL
        db_table = "django_ai_sdk_oversight_data_keys"

    def __str__(self) -> str:
        return self.scope


class Evidence(models.Model):
    """What a turn was about, and what can quote it. Content-addressed.

    The content: question, answer, history, sources, tool calls, at the policy's
    capture level. Beside it, outside what `digest` commits to (ledger.UNHASHED):
    `texts`, by observation id, what an answer said in words (its reason, a `Text`
    answer, names an `Entities` answerer gave, per-item scores by key, the same of
    the answers on its trail); and `notes`, by review item id, what reviewers wrote.
    """

    id = models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True)
    # sha256 of the canonical plaintext, which includes a random salt so a short
    # question cannot be found by hashing guesses.
    digest = models.CharField(max_length=64, unique=True)
    scope = models.CharField(max_length=100, db_index=True)
    key = models.ForeignKey(DataKey, null=True, blank=True, on_delete=models.SET_NULL)
    capture = models.CharField(max_length=16)
    # Exactly one of these is set while the evidence is readable.
    content = models.JSONField(null=True, blank=True)
    ciphertext = models.BinaryField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField(null=True, blank=True, db_index=True)
    shredded_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        app_label = APP_LABEL
        db_table = "django_ai_sdk_oversight_evidence"


class Turn(models.Model):
    """One question and its answer, on any channel: chat, webhook, workflow, API."""

    class Outcome(models.TextChoices):
        BLOCKED = "blocked", "Blocked"
        RESTRICTED = "restricted", "Restricted"
        MASKED = "masked", "Masked"
        FLAGGED = "flagged", "Flagged"
        ALLOWED = "allowed", "Allowed"

    id = models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True)
    created_at = models.DateTimeField(db_index=True)

    channel = models.CharField(max_length=100, db_index=True)
    # Indexed with created_at, like policy and outcome: the console lists by them.
    agent_id = models.CharField(max_length=64)
    agent_name = models.CharField(max_length=255, blank=True)
    # Hash of model, instructions and tool names: a step in quality lines up with one.
    agent_fingerprint = models.CharField(max_length=16)
    policy = models.CharField(max_length=100)

    # Ids only, no foreign keys: deleting a thread must not delete its audit trail.
    thread_id = models.CharField(max_length=64, blank=True, db_index=True)
    user_message_id = models.CharField(max_length=64, blank=True)
    message_id = models.CharField(max_length=64, blank=True)
    # The turn this one ran inside, under a policy of its own: a subagent's
    # delegation, a nested run. Null for a turn of its own. Not on the ledger.
    parent_turn_id = models.UUIDField(null=True, blank=True, db_index=True)

    actor_pseudonym = models.CharField(max_length=32, blank=True)
    # Who asked, sealed: only a recorded reveal reads it. Cleared when the thread is
    # forgotten; not hashed.
    actor_ref = models.TextField(blank=True)

    model = models.CharField(max_length=255, blank=True)
    prompt_tokens = models.IntegerField(null=True, blank=True)
    completion_tokens = models.IntegerField(null=True, blank=True)
    cost = models.FloatField(null=True, blank=True)
    # Of the agent; oversight's own spend is on observations.
    latency_ms = models.FloatField(null=True, blank=True)

    outcome = models.CharField(max_length=16, choices=Outcome.choices)
    # What the turn would have become with every rule in report mode enforced.
    report_outcome = models.CharField(max_length=16, choices=Outcome.choices, db_index=True)
    # What went wrong in the turn, "; " between: a stage that broke, a policy that
    # did not load, the run's own exception, a disconnect, reflect's after-effects.
    # No content. Not on the ledger: reflect may add to it later.
    error = models.CharField(max_length=500, blank=True)

    evidence = models.ForeignKey(Evidence, null=True, blank=True, on_delete=models.SET_NULL)
    # Type hint for the FK id attr
    evidence_id: uuid.UUID | None
    evidence_hash = models.CharField(max_length=64, blank=True)
    entry_hash = models.CharField(max_length=64, blank=True)
    # When prune drops the turn: its policy's metadata retention after created_at.
    expires_at = models.DateTimeField(null=True, blank=True, db_index=True)

    # Denormalized, updated as later stages answer; not on the ledger.
    labels = models.JSONField(default=dict, blank=True)
    severity = models.CharField(max_length=16, blank=True, db_index=True)
    reflected_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        app_label = APP_LABEL
        db_table = "django_ai_sdk_oversight_turns"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["agent_id", "created_at"]),
            models.Index(fields=["policy", "created_at"]),
            models.Index(fields=["outcome", "created_at"]),
        ]

    def __str__(self) -> str:
        return f"{self.channel} {self.outcome} {self.created_at:%Y-%m-%d %H:%M}"


class Observation(models.Model):
    """One answer to one lens about one turn, from one answerer, at one stage.

    No text that can quote the turn: a `Text` answer's value is None, and a name an
    `Entities` answerer gave in its own words is FOUND ("[found]"). The words, the
    reason and per-item scores are in the evidence's `texts`, under this row's id.
    """

    id = models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True)
    turn = models.ForeignKey(Turn, on_delete=models.CASCADE, related_name="observations")
    # Type hint for the FK id attr
    turn_id: uuid.UUID
    lens = models.CharField(max_length=100, db_index=True)
    stage = models.CharField(max_length=16)
    answerer = models.CharField(max_length=200)
    model = models.CharField(max_length=255, blank=True)
    value = models.JSONField(null=True, blank=True)
    confidence = models.FloatField(null=True, blank=True)
    spans = models.JSONField(default=list, blank=True)
    # The answers of earlier answerers in the lens's list, on the way to this one.
    trail = models.JSONField(default=list, blank=True)
    error = models.CharField(max_length=200, blank=True)
    queued = models.CharField(max_length=100, blank=True)
    cost = models.FloatField(null=True, blank=True)
    latency_ms = models.FloatField(null=True, blank=True)
    # The person behind a human answer, as a pseudonym.
    reviewer = models.CharField(max_length=32, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        app_label = APP_LABEL
        db_table = "django_ai_sdk_oversight_observations"
        ordering = ["created_at"]
        indexes = [models.Index(fields=["turn", "lens"])]


class Decision(models.Model):
    """A rule whose condition held for a turn, and whether it acted."""

    id = models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True)
    turn = models.ForeignKey(Turn, on_delete=models.CASCADE, related_name="decisions")
    # Type hint for the FK id attr
    turn_id: uuid.UUID
    rule = models.CharField(max_length=100, db_index=True)
    stage = models.CharField(max_length=16)
    mode = models.CharField(max_length=16)
    actions = models.JSONField(default=list)
    # False in report mode, or when the stage was too late for the action.
    acted = models.BooleanField(default=False)
    condition = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        app_label = APP_LABEL
        db_table = "django_ai_sdk_oversight_decisions"
        ordering = ["created_at"]


class ReviewItem(models.Model):
    """A turn waiting for a person: a lens to answer, or a rule's match to confirm."""

    class Status(models.TextChoices):
        OPEN = "open", "Open"
        DONE = "done", "Done"
        DISMISSED = "dismissed", "Dismissed"

    id = models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True)
    turn = models.ForeignKey(Turn, on_delete=models.CASCADE, related_name="reviews")
    # Type hint for the FK id attr
    turn_id: uuid.UUID
    queue = models.CharField(max_length=100, db_index=True, default="default")
    lens = models.CharField(max_length=100, blank=True)
    rule = models.CharField(max_length=100, blank=True)
    priority = models.IntegerField(default=0)
    status = models.CharField(
        max_length=16, choices=Status.choices, default=Status.OPEN, db_index=True
    )
    created_at = models.DateTimeField(auto_now_add=True)
    resolved_at = models.DateTimeField(null=True, blank=True)
    # Who closed it (a pseudonym), the value and whether they agreed; their note is
    # in the evidence's `notes`, under this row's id.
    resolution = models.JSONField(null=True, blank=True)

    class Meta:
        app_label = APP_LABEL
        db_table = "django_ai_sdk_oversight_reviews"
        ordering = ["-priority", "created_at"]
        indexes = [models.Index(fields=["status", "queue", "-priority", "created_at"])]
        constraints = [
            models.UniqueConstraint(
                fields=["turn", "queue", "lens", "rule"],
                name="oversight_one_review_per_question",
            )
        ]


class AccessRecord(models.Model):
    """Someone read the ledger: who, why, what. Chained, so reading leaves a trace too."""

    id = models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True)
    reader = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="+",
    )
    reader_label = models.CharField(max_length=255, blank=True)
    purpose = models.CharField(max_length=16, choices=[(p.value, p.name.title()) for p in Purpose])
    # list, view, reveal, export, verify, review, override
    action = models.CharField(max_length=16)
    turn_ref = models.CharField(max_length=64, blank=True, db_index=True)
    query = models.JSONField(default=dict, blank=True)
    reason = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        app_label = APP_LABEL
        db_table = "django_ai_sdk_oversight_access"
        ordering = ["-created_at"]
        indexes = [models.Index(fields=["-created_at"])]


class Checkpoint(models.Model):
    """A signed Merkle root over the first `tree_size` ledger entries."""

    id = models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True)
    tree_size = models.BigIntegerField(unique=True)
    root_hash = models.CharField(max_length=64)
    # The checkpoint as a C2SP signed note: origin, size, root, signature lines.
    note = models.TextField()
    signed = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    anchored_at = models.DateTimeField(null=True, blank=True)
    anchor_ref = models.TextField(blank=True)

    class Meta:
        app_label = APP_LABEL
        db_table = "django_ai_sdk_oversight_checkpoints"
        ordering = ["-tree_size"]


class PolicyOverride(models.Model):
    """An operator's tuning of a policy: one field of one lens or rule."""

    id = models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True)
    policy = models.CharField(max_length=100, db_index=True)
    # lens | rule | policy | agent | answerer
    target_kind = models.CharField(max_length=16)
    target = models.CharField(max_length=100, blank=True)
    field = models.CharField(max_length=32)
    value = models.JSONField(null=True)
    # Whether this is the latest row for its field: a later row (or a reset) clears
    # it, so the history stays on the ledger. Materialized for the overlay's query,
    # left out of override_payload, and the one field updated after insert.
    active = models.BooleanField(default=True, db_index=True)
    changed_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="+",
    )
    changed_by_label = models.CharField(max_length=255, blank=True)
    note = models.TextField(blank=True)
    changed_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        app_label = APP_LABEL
        db_table = "django_ai_sdk_oversight_overrides"
        ordering = ["-changed_at"]


class DailySpend(models.Model):
    """What model answerers spent on one UTC day, per answerer."""

    date = models.DateField()
    answerer = models.CharField(max_length=200)
    cost = models.FloatField(default=0)
    calls = models.IntegerField(default=0)

    class Meta:
        app_label = APP_LABEL
        db_table = "django_ai_sdk_oversight_spend"
        constraints = [
            models.UniqueConstraint(fields=["date", "answerer"], name="oversight_spend_per_day")
        ]


__all__ = [
    "AccessRecord",
    "Checkpoint",
    "DailySpend",
    "DataKey",
    "Decision",
    "Evidence",
    "LedgerEntry",
    "LedgerHead",
    "Observation",
    "PolicyOverride",
    "ReviewItem",
    "Turn",
]
