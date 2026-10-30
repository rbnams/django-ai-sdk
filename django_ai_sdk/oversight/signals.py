"""Oversight follows the threads it watched: a deleted thread's content is forgotten.

Deleting a thread (through ThreadService, the admin or a cascade) crypto-shreds
what oversight holds about it, through the same path as `forget_thread`: the
evidence and who asked go, the decisions stay on the ledger, and the shred is
chained there too.
"""

from __future__ import annotations

from typing import Any

from django.db import transaction
from django.db.models.signals import post_delete

from django_ai_sdk.logger import get_logger

logger = get_logger(__name__)


def forget_deleted_thread(sender: Any, instance: Any, **kwargs: Any) -> None:
    """Shred a deleted thread's oversight content; a thread oversight never saw costs a read."""
    from django_ai_sdk.oversight.models import DataKey, Turn
    from django_ai_sdk.oversight.tasks import shred_scope

    thread_id = str(instance.pk)
    scope = f"thread:{thread_id}"
    try:
        # A savepoint: a failure here must not break the transaction deleting the thread.
        with transaction.atomic():
            if (
                Turn.objects.filter(thread_id=thread_id).exists()
                or DataKey.objects.filter(scope=scope).exists()
            ):
                shred_scope(scope)
    except Exception:
        logger.exception("Oversight could not forget deleted thread {}", thread_id)


def connect() -> None:
    from django_ai_sdk.conversation.models import Thread

    post_delete.connect(
        forget_deleted_thread,
        sender=Thread,
        dispatch_uid="django_ai_sdk.oversight.forget_deleted_thread",
    )


__all__ = ["connect", "forget_deleted_thread"]
