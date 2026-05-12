"""Cryptographic receipts for purge — verifiable forgetting.

A receipt is an Ed25519-signed statement: *at time T, this system
acknowledged the deletion of records with these text hashes, when the
event log's Merkle root was R.*

The Merkle root commits to the entire append-only event log up to and
including the purge events themselves.  Tampering with any past event
afterwards changes the recomputed root and invalidates the receipt —
that's the auditable property.

Without database access, a verifier can confirm:
  - the signature is from the stated public key (non-forgery)
  - the purge happened (the receipt's hashes match the records they claim)
With database access, a verifier can additionally confirm:
  - the event log has not been tampered with (root recomputes identically)

cryptography is an optional dep: `pip install 'lethe[crypto]'`.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

try:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey, Ed25519PublicKey,
    )
    HAVE_CRYPTO = True
except ImportError:
    HAVE_CRYPTO = False


# ─── data ──────────────────────────────────────────────────────────

@dataclass
class PurgeReceipt:
    version: str
    purged_ids: list[int]
    purged_text_hashes: list[str]    # sha256 hex of each deleted text
    purge_event_ids: list[int]       # event log ids the purge produced
    last_event_id: int               # highest event id at receipt time
    event_log_merkle_root: str       # hex
    timestamp: float
    public_key: str                  # hex
    signature: str = ""              # hex; filled by sign()

    def to_dict(self) -> dict:
        return asdict(self)

    def canonical_payload(self) -> bytes:
        """Bytes signed (and verified against).  Stable serialization."""
        d = self.to_dict()
        d.pop("signature", None)
        return json.dumps(d, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False).encode("utf-8")

    @classmethod
    def from_dict(cls, d: dict) -> "PurgeReceipt":
        return cls(**d)


# ─── Merkle ────────────────────────────────────────────────────────

def event_hash(event_row) -> bytes:
    """SHA-256 over a canonical string representation of one event row.
    Accepts a tuple (id, memory_id, kind, depth_before, depth_after,
    timestamp, meta_json)."""
    eid, mid, kind, db, da, ts, meta = event_row
    canonical = (
        f"{eid}|{mid}|{kind}|"
        f"{'' if db is None else repr(db)}|"
        f"{'' if da is None else repr(da)}|"
        f"{ts:.6f}|{meta or ''}"
    )
    return hashlib.sha256(canonical.encode("utf-8")).digest()


def merkle_root(leaves: list[bytes]) -> bytes:
    """Binary Merkle tree; odd nodes promoted (duplicated) at each layer.
    Empty input returns 32 zero bytes — a distinct "no events" root."""
    if not leaves:
        return b"\x00" * 32
    layer = list(leaves)
    while len(layer) > 1:
        nxt: list[bytes] = []
        for i in range(0, len(layer), 2):
            a = layer[i]
            b = layer[i + 1] if i + 1 < len(layer) else a
            nxt.append(hashlib.sha256(a + b).digest())
        layer = nxt
    return layer[0]


def text_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


# ─── Ed25519 ───────────────────────────────────────────────────────

def _require_crypto() -> None:
    if not HAVE_CRYPTO:
        raise ImportError(
            "purge receipts require the cryptography package — "
            "install via `pip install 'lethe[crypto]'`."
        )


def generate_key() -> tuple[bytes, bytes]:
    """Returns (private_key_bytes, public_key_bytes), 32 bytes each."""
    _require_crypto()
    priv = Ed25519PrivateKey.generate()
    priv_b = priv.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub_b = priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return priv_b, pub_b


def public_key_from_private(private_key_bytes: bytes) -> bytes:
    _require_crypto()
    priv = Ed25519PrivateKey.from_private_bytes(private_key_bytes)
    return priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def sign(message: bytes, private_key_bytes: bytes) -> bytes:
    _require_crypto()
    priv = Ed25519PrivateKey.from_private_bytes(private_key_bytes)
    return priv.sign(message)


def verify_signature(message: bytes, signature: bytes,
                     public_key_bytes: bytes) -> bool:
    _require_crypto()
    pub = Ed25519PublicKey.from_public_bytes(public_key_bytes)
    try:
        pub.verify(signature, message)
        return True
    except InvalidSignature:
        return False


# ─── key storage ───────────────────────────────────────────────────

def default_key_path() -> Path:
    return Path.home() / ".lethe" / "keys" / "purge_signing.key"


def save_private_key(path: Path, private_key_bytes: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(private_key_bytes)
    try:
        os.chmod(path, 0o600)
    except Exception:
        pass


def load_private_key(path: Path) -> bytes:
    return path.read_bytes()


# ─── high-level verification ───────────────────────────────────────

def verify_receipt(receipt: PurgeReceipt, lethe=None) -> dict:
    """Verify a receipt.

    Without `lethe`: signature only — proves the issuer claimed this purge.
    With `lethe`:    additionally recompute the Merkle root over events
                     up to `last_event_id` and check it still matches.
                     A mismatch means the event log has been tampered with.
    """
    out: dict = {
        "signature_valid": False,
        "log_root_valid": None,
        "issues": [],
    }

    # 1. signature
    pub_bytes = bytes.fromhex(receipt.public_key)
    sig_bytes = bytes.fromhex(receipt.signature)
    msg = receipt.canonical_payload()
    out["signature_valid"] = verify_signature(msg, sig_bytes, pub_bytes)
    if not out["signature_valid"]:
        out["issues"].append("invalid signature")

    # 2. optional log check
    if lethe is not None:
        events = lethe.conn.execute(
            "SELECT id, memory_id, kind, depth_before, depth_after, "
            "timestamp, meta FROM event WHERE id <= ? ORDER BY id",
            (receipt.last_event_id,),
        ).fetchall()
        if not events:
            out["log_root_valid"] = False
            out["issues"].append("no events found in database")
        elif events[-1][0] != receipt.last_event_id:
            out["log_root_valid"] = False
            out["issues"].append(
                f"missing events: db max id={events[-1][0]}, "
                f"receipt last_event_id={receipt.last_event_id}"
            )
        else:
            recomputed = merkle_root([event_hash(e) for e in events]).hex()
            out["log_root_valid"] = (recomputed == receipt.event_log_merkle_root)
            if not out["log_root_valid"]:
                out["issues"].append(
                    "merkle root mismatch — event log was tampered with"
                )

    out["valid"] = (
        out["signature_valid"]
        and out["log_root_valid"] is not False
    )
    return out
