"""Evidence at rest: AES-GCM per scope, keys wrapped by one master key.

One data key per thread means "forget this conversation" is one row deleted,
whatever number of turns, observations and backups mention it: crypto-shredding.
Without MASTER_KEY (or without django-ai-sdk[oversight]) content is stored in
the clear, and the oversight.W004 check says so on every deploy.
"""

from __future__ import annotations

import base64
import os
from typing import TYPE_CHECKING, Any

from django.core.exceptions import ImproperlyConfigured

from django_ai_sdk.oversight.ledger import canonical

if TYPE_CHECKING:
    from django_ai_sdk.oversight.models import DataKey

NONCE = 12


def master_key() -> bytes | None:
    from django_ai_sdk.oversight.conf import setting

    raw = setting("MASTER_KEY")
    if not raw:
        return None
    key = base64.b64decode(raw)
    if len(key) != 32:
        raise ImproperlyConfigured("AI_SDK_OVERSIGHT['MASTER_KEY'] must be 32 bytes, base64")
    return key


def available() -> bool:
    """Encryption is configured and the library is there."""
    try:
        import cryptography  # noqa: F401
    except ImportError:
        return False
    return master_key() is not None


def _aead(key: bytes) -> Any:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    return AESGCM(key)


def seal_bytes(key: bytes, data: bytes, aad: bytes) -> bytes:
    nonce = os.urandom(NONCE)
    return nonce + _aead(key).encrypt(nonce, data, aad)


def open_bytes(key: bytes, blob: bytes, aad: bytes) -> bytes:
    return _aead(key).decrypt(blob[:NONCE], blob[NONCE:], aad)


def data_key_for(scope: str) -> tuple[DataKey, bytes] | None:
    """The scope's key row and its plaintext key, created on first use. Synchronous.

    None when encryption is off, or when the scope was shredded: nothing new is
    ever written under a forgotten scope.
    """
    from django_ai_sdk.oversight.models import DataKey

    master = master_key()
    if master is None:
        return None
    row, created = DataKey.objects.get_or_create(scope=scope)
    if row.shredded_at is not None:
        return None
    if created or not row.wrapped:
        key = os.urandom(32)
        row.wrapped = seal_bytes(master, key, scope.encode())
        row.save(update_fields=["wrapped"])
        return row, key
    return row, open_bytes(master, bytes(row.wrapped), scope.encode())


def unwrap(row: DataKey) -> bytes | None:
    master = master_key()
    if master is None or not row.wrapped or row.shredded_at is not None:
        return None
    return open_bytes(master, bytes(row.wrapped), row.scope.encode())


def encrypt_json(scope: str, value: Any) -> tuple[DataKey, bytes] | None:
    found = data_key_for(scope)
    if found is None:
        return None
    row, key = found
    return row, seal_bytes(key, canonical(value), scope.encode())


def decrypt_json(row: DataKey | None, blob: bytes | None, scope: str) -> Any:
    import json

    if row is None or blob is None:
        return None
    key = unwrap(row)
    if key is None:
        return None
    return json.loads(open_bytes(key, bytes(blob), scope.encode()))


def seal_text(scope: str, text: str) -> str:
    """A short secret (who asked) sealed under the scope's key, as text; clear if off."""
    found = data_key_for(scope)
    if found is None:
        return f"plain:{text}"
    row, key = found
    return "aead:" + base64.b64encode(seal_bytes(key, text.encode(), b"actor")).decode()


def open_text(scope: str, sealed: str) -> str | None:
    from django_ai_sdk.oversight.models import DataKey

    if sealed.startswith("plain:"):
        return sealed[6:]
    if not sealed.startswith("aead:"):
        return None
    row = DataKey.objects.filter(scope=scope).first()
    key = unwrap(row) if row else None
    if key is None:
        return None
    return open_bytes(key, base64.b64decode(sealed[5:]), b"actor").decode()


def generate_master_key() -> str:
    return base64.b64encode(os.urandom(32)).decode()


__all__ = [
    "available",
    "data_key_for",
    "decrypt_json",
    "encrypt_json",
    "generate_master_key",
    "open_text",
    "seal_text",
]
