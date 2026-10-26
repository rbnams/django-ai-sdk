"""Checkpoints: a signed Merkle root over the ledger, in the transparency-log formats.

The chain proves order; a checkpoint makes it cheap to prove to an outsider.
Publish a checkpoint (an exporter can post it anywhere you don't control) and
the deployment can no longer rewrite history before it without the published
root disagreeing. Trees are RFC 6962; the note is a C2SP signed note, so
standard tlog tooling reads it.
"""

from __future__ import annotations

import base64
import hashlib
from typing import TYPE_CHECKING, Any

from django.core.exceptions import ImproperlyConfigured

if TYPE_CHECKING:
    from collections.abc import Sequence


def leaf_hash(data: bytes) -> bytes:
    return hashlib.sha256(b"\x00" + data).digest()


def node_hash(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(b"\x01" + left + right).digest()


def _split(n: int) -> int:
    """The largest power of two smaller than n."""
    k = 1
    while k << 1 < n:
        k <<= 1
    return k


def merkle_root(leaves: Sequence[bytes]) -> bytes:
    """RFC 6962 Merkle Tree Hash of `leaves` (raw leaf data, hashed here)."""
    if not leaves:
        return hashlib.sha256(b"").digest()
    level = [leaf_hash(leaf) for leaf in leaves]
    return _root(level)


def _root(hashes: Sequence[bytes]) -> bytes:
    if len(hashes) == 1:
        return hashes[0]
    k = _split(len(hashes))
    return node_hash(_root(hashes[:k]), _root(hashes[k:]))


def inclusion_proof(index: int, leaves: Sequence[bytes]) -> list[bytes]:
    """The audit path for leaf `index` in a tree of `leaves` (RFC 6962 2.1.1)."""
    hashes = [leaf_hash(leaf) for leaf in leaves]
    if not 0 <= index < len(hashes):
        raise IndexError(index)
    return _path(index, hashes)


def _path(index: int, hashes: Sequence[bytes]) -> list[bytes]:
    if len(hashes) <= 1:
        return []
    k = _split(len(hashes))
    if index < k:
        return [*_path(index, hashes[:k]), _root(hashes[k:])]
    return [*_path(index - k, hashes[k:]), _root(hashes[:k])]


def verify_inclusion(
    leaf: bytes, index: int, size: int, proof: Sequence[bytes], root: bytes
) -> bool:
    """Check an inclusion proof (RFC 9162 2.1.3.2)."""
    if not 0 <= index < size:
        return False
    fn, sn = index, size - 1
    current = leaf_hash(leaf)
    for sibling in proof:
        if sn == 0:
            return False
        if fn & 1 or fn == sn:
            current = node_hash(sibling, current)
            if not fn & 1:
                while fn and not fn & 1:
                    fn >>= 1
                    sn >>= 1
        else:
            current = node_hash(current, sibling)
        fn >>= 1
        sn >>= 1
    return sn == 0 and current == root


def consistency_proof(old_size: int, leaves: Sequence[bytes]) -> list[bytes]:
    """Proof that the first `old_size` leaves are a prefix of `leaves` (RFC 6962 2.1.2)."""
    hashes = [leaf_hash(leaf) for leaf in leaves]
    if not 0 < old_size <= len(hashes):
        raise ValueError("old_size must be between 1 and the tree size")
    return _subproof(old_size, hashes, True)


def _subproof(m: int, hashes: Sequence[bytes], complete: bool) -> list[bytes]:
    n = len(hashes)
    if m == n:
        return [] if complete else [_root(hashes)]
    k = _split(n)
    if m <= k:
        return [*_subproof(m, hashes[:k], complete), _root(hashes[k:])]
    return [*_subproof(m - k, hashes[k:], False), _root(hashes[:k])]


def verify_consistency(
    old_size: int, new_size: int, old_root: bytes, new_root: bytes, proof: Sequence[bytes]
) -> bool:
    """Check a consistency proof (RFC 9162 2.1.4.2)."""
    if old_size == new_size:
        return old_root == new_root and not proof
    if not 0 < old_size < new_size or not proof:
        return False
    path = list(proof)
    if old_size & (old_size - 1) == 0:
        # A power of two: the old root is itself the first node of the path.
        path = [old_root, *path]
    fn, sn = old_size - 1, new_size - 1
    while fn & 1:
        fn >>= 1
        sn >>= 1
    fr = sr = path[0]
    for c in path[1:]:
        if sn == 0:
            return False
        if fn & 1 or fn == sn:
            fr = node_hash(c, fr)
            sr = node_hash(c, sr)
            while fn and not fn & 1:
                fn >>= 1
                sn >>= 1
        else:
            sr = node_hash(sr, c)
        fn >>= 1
        sn >>= 1
    return fr == old_root and sr == new_root and sn == 0


# Signed notes (https://c2sp.org/signed-note) and checkpoints (https://c2sp.org/tlog-checkpoint).

ED25519_ALG = b"\x01"


def _private_key(raw_b64: str) -> Any:
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    except ImportError as exc:  # pragma: no cover - needs the extra
        raise ImproperlyConfigured(
            "Signing checkpoints needs cryptography: pip install django-ai-sdk[oversight]"
        ) from exc
    raw = base64.b64decode(raw_b64)
    if len(raw) != 32:
        raise ImproperlyConfigured("AI_SDK_OVERSIGHT['SIGNING_KEY'] must be 32 bytes, base64")
    return Ed25519PrivateKey.from_private_bytes(raw)


def key_id(name: str, public_key: bytes) -> bytes:
    return hashlib.sha256(name.encode() + b"\n" + ED25519_ALG + public_key).digest()[:4]


def checkpoint_body(origin: str, size: int, root: bytes) -> str:
    return f"{origin}\n{size}\n{base64.b64encode(root).decode()}\n"


def sign_note(body: str, *, name: str, private_key_b64: str) -> str:
    """`body` followed by one Ed25519 signature line."""
    from cryptography.hazmat.primitives import serialization

    key = _private_key(private_key_b64)
    public = key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
    )
    signature = key.sign(body.encode())
    blob = base64.b64encode(key_id(name, public) + signature).decode()
    return f"{body}\n— {name} {blob}\n"


def verifier_key(name: str, private_key_b64: str) -> str:
    """The public verifier key string, `name+keyhash+base64(alg||pub)`, to publish."""
    from cryptography.hazmat.primitives import serialization

    public = (
        _private_key(private_key_b64)
        .public_key()
        .public_bytes(encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)
    )
    return f"{name}+{key_id(name, public).hex()}+{base64.b64encode(ED25519_ALG + public).decode()}"


def verify_note(note: str, verifier: str) -> bool:
    """Whether `note` carries a valid signature by `verifier` (a verifier key string)."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    name, _hash, key_b64 = verifier.split("+", 2)
    key_bytes = base64.b64decode(key_b64)
    if key_bytes[:1] != ED25519_ALG:
        return False
    public = key_bytes[1:]
    body, _, signatures = note.partition("\n\n")
    body += "\n"
    for line in signatures.splitlines():
        if not line.startswith(f"— {name} "):
            continue
        blob = base64.b64decode(line.rsplit(" ", 1)[1])
        if blob[:4] != key_id(name, public):
            continue
        try:
            Ed25519PublicKey.from_public_bytes(public).verify(blob[4:], body.encode())
        except InvalidSignature:
            return False
        return True
    return False


def generate_signing_key() -> str:
    """A new base64 Ed25519 private key for AI_SDK_OVERSIGHT['SIGNING_KEY']."""
    import secrets

    return base64.b64encode(secrets.token_bytes(32)).decode()


__all__ = [
    "checkpoint_body",
    "consistency_proof",
    "generate_signing_key",
    "inclusion_proof",
    "merkle_root",
    "sign_note",
    "verifier_key",
    "verify_consistency",
    "verify_inclusion",
    "verify_note",
]
