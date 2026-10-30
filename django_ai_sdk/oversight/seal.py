"""Checkpoints: a signed Merkle root over the ledger, in the transparency-log formats.

The chain proves order; a checkpoint makes it cheap to prove to an outsider.
Publish a checkpoint (an exporter can post it anywhere you don't control) and
the deployment can no longer rewrite history before it without the published
root disagreeing, for anyone who compares the two. Trees are RFC 6962; the note is a C2SP signed note, so
standard tlog tooling reads it.
"""

from __future__ import annotations

import base64
import hashlib
from typing import TYPE_CHECKING, Any

from django.core.exceptions import ImproperlyConfigured

if TYPE_CHECKING:
    from collections.abc import Sequence

    from django_ai_sdk.oversight.models import Checkpoint


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


def private_key(raw_b64: str) -> Any:
    """The Ed25519 key in `raw_b64`. ImproperlyConfigured when it is unusable."""
    try:
        raw = base64.b64decode((raw_b64 or "").strip(), validate=True)
    except ValueError:
        raw = b""
    if len(raw) != 32:
        raise ImproperlyConfigured("AI_SDK_OVERSIGHT['SIGNING_KEY'] must be base64 of 32 bytes.")
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    except ImportError as exc:
        raise ImproperlyConfigured(
            "AI_SDK_OVERSIGHT['SIGNING_KEY'] is set, but the cryptography package is not "
            "installed: no checkpoint can be signed."
        ) from exc
    return Ed25519PrivateKey.from_private_bytes(raw)


def key_id(name: str, public_key: bytes) -> bytes:
    return hashlib.sha256(name.encode() + b"\n" + ED25519_ALG + public_key).digest()[:4]


def checkpoint_body(origin: str, size: int, root: bytes) -> str:
    return f"{origin}\n{size}\n{base64.b64encode(root).decode()}\n"


def sign_note(body: str, *, name: str, private_key_b64: str) -> str:
    """`body` followed by one Ed25519 signature line."""
    key = private_key(private_key_b64)
    from cryptography.hazmat.primitives import serialization

    public = key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
    )
    signature = key.sign(body.encode())
    blob = base64.b64encode(key_id(name, public) + signature).decode()
    return f"{body}\n— {name} {blob}\n"


def verifier_key(name: str, private_key_b64: str) -> str:
    """The public verifier key string, `name+keyhash+base64(alg||pub)`, to publish."""
    key = private_key(private_key_b64)
    from cryptography.hazmat.primitives import serialization

    public = key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
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
        try:
            blob = base64.b64decode(line.rsplit(" ", 1)[1])
        except ValueError:
            continue
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


# Checkpoints of this deployment's ledger.


def ledger_root(size: int) -> bytes | None:
    """The Merkle root over the first `size` ledger entries; None when there are fewer."""
    from django_ai_sdk.oversight.models import LedgerEntry

    hashes = LedgerEntry.objects.order_by("seq").values_list("entry_hash", flat=True)[:size]
    leaves = [bytes.fromhex(h) for h in hashes]
    return merkle_root(leaves) if len(leaves) == size else None


def matches(checkpoint: Checkpoint) -> bool:
    """Whether the ledger's first `tree_size` entries still give the checkpoint's root."""
    root = ledger_root(checkpoint.tree_size)
    return root is not None and root.hex() == checkpoint.root_hash


def signed_by(checkpoint: Checkpoint, verifier: str) -> bool:
    """Whether `verifier` signed the checkpoint's note, and the note states its size and root.

    The row's tree_size and root_hash are plain columns: only the note is signed.
    """
    try:
        _origin, size, root = checkpoint.note.split("\n")[:3]
        stated = int(size) == checkpoint.tree_size and (
            base64.b64decode(root).hex() == checkpoint.root_hash
        )
    except ValueError:
        return False
    return stated and verify_note(checkpoint.note, verifier)


def write_checkpoint(
    *, origin: str, name: str = "", key: str = ""
) -> tuple[Checkpoint, bool] | None:
    """Checkpoint the ledger as it stands, signed with `key` when given. Synchronous.

    Returns (checkpoint, created), or None when nothing was added since the last.
    One checkpoint per tree size: a second sealer at the same moment gets the
    first one's back with created False, never an error.
    """
    from django_ai_sdk.oversight.models import Checkpoint, LedgerEntry

    last = Checkpoint.objects.order_by("-tree_size").first()
    size = LedgerEntry.objects.count()
    if size == 0 or (last is not None and last.tree_size >= size):
        return None
    root = ledger_root(size)
    if root is None:
        return None
    body = checkpoint_body(origin, size, root)
    note = sign_note(body, name=name or origin, private_key_b64=key) if key else body
    return Checkpoint.objects.get_or_create(
        tree_size=size, defaults={"root_hash": root.hex(), "note": note, "signed": bool(key)}
    )


__all__ = [
    "checkpoint_body",
    "consistency_proof",
    "generate_signing_key",
    "inclusion_proof",
    "ledger_root",
    "matches",
    "merkle_root",
    "private_key",
    "sign_note",
    "signed_by",
    "verifier_key",
    "verify_consistency",
    "verify_inclusion",
    "verify_note",
    "write_checkpoint",
]
