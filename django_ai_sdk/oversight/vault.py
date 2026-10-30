"""Evidence at rest: AES-GCM per scope, keys wrapped by one master key.

One data key per thread means "forget this conversation" is one row cleared,
whatever number of turns it holds: crypto-shredding. Everything that can quote a
turn is sealed in its evidence (judges' reasons, `Text` answers, names an answerer
gave, reviewers' notes), so it goes with the key. Who asked is sealed under the
asker's own key, and forgetting a thread clears it from the thread's turns. A
database backup taken before still holds the wrapped key, so it stays readable
with MASTER_KEY until the backup itself is gone.

Without MASTER_KEY content is stored in the clear, and the ai_sdk.oversight.W001 deploy check
says so on every deploy. A MASTER_KEY that is set but unusable (not base64 of
32 bytes, or django-ai-sdk[oversight] not installed) raises ImproperlyConfigured
and fails ai_sdk.oversight.E008: it never reads as "no key".
"""

from __future__ import annotations

import base64
import json
import os
from typing import TYPE_CHECKING, Any

from django.core.exceptions import ImproperlyConfigured

from django_ai_sdk.oversight.ledger import canonical

if TYPE_CHECKING:
    from django_ai_sdk.oversight.models import DataKey

NONCE = 12


class VaultError(Exception):
    """Sealed content that cannot be written or read as asked."""


class Forgotten(VaultError):
    """The scope was crypto-shredded: nothing is sealed under it again, nor stored in the clear."""


class KeyMismatch(VaultError):
    """MASTER_KEY cannot open what was sealed: another key, none, or an altered blob.

    Not forgotten: the content is still there for the key that sealed it.
    """


def has_cryptography() -> bool:
    try:
        import cryptography  # noqa: F401
    except ImportError:
        return False
    return True


def master_key() -> bytes | None:
    """MASTER_KEY, decoded; None when it is not set.

    Raises ImproperlyConfigured when it is set but unusable, rather than read as
    unset: that would store content in the clear.
    """
    from django_ai_sdk.oversight.conf import setting

    raw = (setting("MASTER_KEY") or "").strip()
    if not raw:
        return None
    try:
        key = base64.b64decode(raw, validate=True)
    except ValueError:
        key = b""
    if len(key) != 32:
        raise ImproperlyConfigured("AI_SDK_OVERSIGHT['MASTER_KEY'] must be base64 of 32 bytes.")
    if not has_cryptography():
        raise ImproperlyConfigured(
            "AI_SDK_OVERSIGHT['MASTER_KEY'] is set, but the cryptography package is not "
            "installed: no turn can be recorded."
        )
    return key


def available() -> bool:
    """Whether content is encrypted. Raises ImproperlyConfigured for an unusable MASTER_KEY."""
    return master_key() is not None


def _aead(key: bytes) -> Any:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    return AESGCM(key)


def seal_bytes(key: bytes, data: bytes, aad: bytes) -> bytes:
    """`data` encrypted under `key`, a fresh nonce in front; `aad` must match on open."""
    nonce = os.urandom(NONCE)
    return nonce + _aead(key).encrypt(nonce, data, aad)


def open_bytes(key: bytes, blob: bytes, aad: bytes) -> bytes:
    """The plaintext of a `seal_bytes` blob. KeyMismatch if the key, blob or `aad` differ."""
    from cryptography.exceptions import InvalidTag

    try:
        return _aead(key).decrypt(blob[:NONCE], blob[NONCE:], aad)
    except InvalidTag as exc:
        raise KeyMismatch("The key does not open this, or it was altered.") from exc


def data_key_for(scope: str) -> tuple[DataKey, bytes] | None:
    """The scope's key row and its plaintext key, created on first use. Synchronous.

    None when encryption is off. Raises Forgotten when the scope was shredded,
    encryption on or off: nothing new is ever written under a forgotten scope.
    """
    from django_ai_sdk.oversight.models import DataKey

    master = master_key()
    if master is None:
        if DataKey.objects.filter(scope=scope, shredded_at__isnull=False).exists():
            raise Forgotten(scope)
        return None
    row = DataKey.objects.filter(scope=scope).first()
    if row is None:
        # Created with its key in one insert: two first turns of a thread at once
        # agree on one key, instead of the later one overwriting the first.
        key = os.urandom(32)
        row, created = DataKey.objects.get_or_create(
            scope=scope, defaults={"wrapped": seal_bytes(master, key, scope.encode())}
        )
        if created:
            return row, key
    if row.shredded_at is not None:
        raise Forgotten(scope)
    if not row.wrapped:
        key = os.urandom(32)
        row.wrapped = seal_bytes(master, key, scope.encode())
        row.save(update_fields=["wrapped"])
        return row, key
    return row, open_bytes(master, bytes(row.wrapped), scope.encode())


def forgotten(scope: str) -> bool:
    """Whether `scope` was crypto-shredded: nothing new is kept under it, not even who asked."""
    from django_ai_sdk.oversight.models import DataKey

    return DataKey.objects.filter(scope=scope, shredded_at__isnull=False).exists()


def unwrap(row: DataKey) -> bytes | None:
    """The scope's plaintext data key, opened with the master key.

    None when the scope was shredded. Raises KeyMismatch when MASTER_KEY is
    unset or is not the key that wrapped it.
    """
    if row.shredded_at is not None or not row.wrapped:
        return None
    master = master_key()
    if master is None:
        raise KeyMismatch(f"{row.scope} is encrypted, but AI_SDK_OVERSIGHT['MASTER_KEY'] is unset.")
    return open_bytes(master, bytes(row.wrapped), row.scope.encode())


def encrypt_json(scope: str, value: Any) -> tuple[DataKey, bytes] | None:
    """`value` sealed under the scope's data key, with the key row to store beside it.

    None only when encryption is off: store `value` in the clear. Raises Forgotten
    when the scope was shredded: store nothing, never the clear text.
    """
    found = data_key_for(scope)
    if found is None:
        return None
    row, key = found
    return row, seal_bytes(key, canonical(value), scope.encode())


def decrypt_json(row: DataKey | None, blob: bytes | None, scope: str) -> Any:
    """The value `encrypt_json` sealed under the scope's data key.

    None when there is nothing to open or the scope was shredded. Raises
    KeyMismatch when MASTER_KEY cannot open it: there, but unreadable here.
    """
    if row is None or blob is None:
        return None
    key = unwrap(row)
    if key is None:
        return None
    return json.loads(open_bytes(key, bytes(blob), scope.encode()))


def seal_text(scope: str, text: str) -> str:
    """A short secret (who asked) sealed under the scope's key, as text; clear if off.

    Raises Forgotten when the scope was shredded, as `encrypt_json` does.
    """
    found = data_key_for(scope)
    if found is None:
        return f"plain:{text}"
    row, key = found
    return "aead:" + base64.b64encode(seal_bytes(key, text.encode(), b"actor")).decode()


def open_text(scope: str, sealed: str) -> str | None:
    """What `seal_text` sealed. None when the scope was shredded; KeyMismatch as `decrypt_json`."""
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
    "Forgotten",
    "KeyMismatch",
    "VaultError",
    "available",
    "data_key_for",
    "decrypt_json",
    "encrypt_json",
    "forgotten",
    "generate_master_key",
    "has_cryptography",
    "master_key",
    "open_text",
    "seal_text",
]
