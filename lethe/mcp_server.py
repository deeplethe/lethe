"""Lethe MCP server.

Exposes the Lethe API as MCP tools over stdio.  Connect from Claude
Desktop, Claude Code, Cursor, or any MCP client by adding a config like:

    {
      "mcpServers": {
        "lethe": {
          "command": "python",
          "args": ["-m", "lethe.mcp_server"],
          "env": {"LETHE_DB": "/path/to/agent.db"}
        }
      }
    }

DB path comes from LETHE_DB or defaults to ~/.lethe/agent.db.
Embedder defaults to MiniLM-L6-v2; override with LETHE_EMBEDDER.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

from lethe import Lethe


# ─── singleton (one Lethe per server process) ──────────────────────

_LETHE: Optional[Lethe] = None


def _db_path() -> str:
    if env := os.environ.get("LETHE_DB"):
        return env
    home = Path.home() / ".lethe"
    home.mkdir(parents=True, exist_ok=True)
    return str(home / "agent.db")


def _embedder():
    from fastembed import TextEmbedding
    name = os.environ.get("LETHE_EMBEDDER",
                          "sentence-transformers/all-MiniLM-L6-v2")
    model = TextEmbedding(name)
    def embed(text: str) -> list[float]:
        return list(next(iter(model.embed([text]))))
    return embed


def _signing_key() -> Optional[bytes]:
    """Load Ed25519 signing key for purge receipts.  Env: LETHE_SIGNING_KEY
    is a path to the key file; falls back to ~/.lethe/keys/purge_signing.key
    if it exists.  Returns None if no key is configured (purge_signed will
    raise)."""
    from lethe.receipt import default_key_path, load_private_key
    explicit = os.environ.get("LETHE_SIGNING_KEY")
    path = Path(explicit) if explicit else default_key_path()
    if path.exists():
        return load_private_key(path)
    return None


def _get() -> Lethe:
    global _LETHE
    if _LETHE is None:
        dim = int(os.environ.get("LETHE_DIM", "384"))
        _LETHE = Lethe(_db_path(), vector_dim=dim, embedder=_embedder(),
                       signing_key=_signing_key())
    return _LETHE


# ─── server ────────────────────────────────────────────────────────

mcp = FastMCP("lethe")


@mcp.tool()
def inscribe(text: str) -> dict:
    """Store a fact on the surface of memory (depth = 1.0).
    Returns the new memory's integer id."""
    mid = _get().inscribe(text)
    return {"id": mid}


@mcp.tool()
def recall(query: str, k: int = 5, lexical: bool = False,
           hybrid: bool = True, at: Optional[float] = None) -> list[dict]:
    """Retrieve memories matching the query.

    lexical=True: pure BM25 (identifier lookup, e.g. emails, API keys).
    hybrid=True (default): RRF blend of vec + BM25.
    hybrid=False: vec-only.
    at: unix timestamp for time-travel — what did the agent believe at T?
    """
    hits = _get().recall(query, k=k, lexical=lexical,
                         hybrid=hybrid, at=at)
    return [
        {"id": h.memory.id, "text": h.memory.text,
         "depth": h.memory.depth, "score": h.score,
         "similarity": h.similarity}
        for h in hits
    ]


@mcp.tool()
def release(ids: list[int]) -> dict:
    """Soft-evict: depth → 0.  Memory is preserved on disk and visible to
    time-travel, but does not surface in normal recall."""
    n = _get().surrender(ids, mode="release")
    return {"released": n}


@mcp.tool()
def purge(ids: list[int]) -> dict:
    """Hard-delete from disk.  Only the event log remembers.  Use for
    GDPR / compliance — irreversible at the data level."""
    n = _get().surrender(ids, mode="purge")
    return {"purged": n}


@mcp.tool()
def purge_signed(ids: list[int]) -> dict:
    """Hard-delete + return an Ed25519-signed PurgeReceipt anchored to the
    event-log Merkle root.  Lets third parties verify, without database
    access, that the deletion was acknowledged at a specific moment.

    Requires a signing key (env LETHE_SIGNING_KEY or default key file)."""
    receipt = _get().purge_with_receipt(ids)
    return receipt.to_dict()


@mcp.tool()
def supersede(old_id: int, new_text: str, reason: str = "") -> dict:
    """Replace an old fact with a new one.  The old memory sinks
    (depth → 0); the new one rises (depth = 1.0).  A supersession edge
    is recorded for blame() to walk later."""
    target: dict = {"old": old_id, "new": new_text}
    if reason:
        target["reason"] = reason
    new_id = _get().surrender(target, mode="supersede")
    return {"old_id": old_id, "new_id": new_id}


@mcp.tool()
def pin(id: int) -> dict:
    """Pin a memory above the surface (depth = +∞), immune to gravity.
    Only purge() can still remove it."""
    n = _get().pin(id)
    return {"pinned": n}


@mcp.tool()
def unpin(id: int) -> dict:
    """Inverse of pin: drop back to surface (depth = 1.0)."""
    n = _get().unpin(id)
    return {"unpinned": n}


@mcp.tool()
def consolidate(tau_seconds: float = 86400.0 * 7) -> dict:
    """Apply Hypnos gravity to every unpinned memory.  Frequently-accessed
    facts sink slower; long-survivors get promoted above the surface.
    tau_seconds is the time constant (default: 7 days)."""
    r = _get().consolidate(tau_seconds=tau_seconds)
    return {"decayed": r.decayed, "promoted": r.promoted,
            "collapsed": r.collapsed, "duration_ms": r.duration_ms}


@mcp.tool()
def blame(query: str, k: int = 5) -> list[dict]:
    """For each memory matching the query, return its supersession chain:
    what it superseded, and what (if anything) has since superseded it."""
    entries = _get().blame(query, k=k)
    return [
        {"id": e.memory.id, "text": e.memory.text,
         "depth": e.memory.depth,
         "superseded_by": e.superseded_by,
         "supersedes": e.supersedes}
        for e in entries
    ]


@mcp.tool()
def log(kind: Optional[str] = None, since: Optional[float] = None,
        until: Optional[float] = None, limit: int = 50) -> list[dict]:
    """Query the append-only event log.  Filter by kind (inscribe / recall /
    release / supersede / purge / pin / flow / promote) and/or time range."""
    events = _get().log(kind=kind, since=since, until=until)
    if limit and len(events) > limit:
        events = events[-limit:]
    return [
        {"id": ev.id, "memory_id": ev.memory_id, "kind": ev.kind,
         "depth_before": ev.depth_before, "depth_after": ev.depth_after,
         "timestamp": ev.timestamp}
        for ev in events
    ]


def main():
    mcp.run()


if __name__ == "__main__":
    main()
