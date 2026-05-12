"""Lethe CLI.

Thin wrapper over the Python API.  Every subcommand maps 1:1 to a core
method — the CLI adds path resolution and lazy embedder loading, nothing
more.

  lethe inscribe TEXT
  lethe recall  QUERY [-k N] [--lexical | --no-hybrid] [--at ISO|UNIX]
  lethe release ID [ID...]
  lethe purge   ID [ID...]
  lethe supersede OLD_ID --new TEXT
  lethe pin     ID  /  lethe unpin ID
  lethe consolidate [--half-life SECONDS]
  lethe blame   QUERY [-k N]
  lethe log     [--kind KIND] [--limit N] [--since ISO|UNIX]

DB path:  --db PATH, env LETHE_DB, or default ~/.lethe/agent.db
Embedder: --embedder MODEL (only loaded when needed; first call is slow)
Output:   --json for machine-readable on any subcommand
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from lethe import Lethe


# ─── helpers ───────────────────────────────────────────────────────

def _default_db() -> str:
    if env := os.environ.get("LETHE_DB"):
        return env
    home = Path.home() / ".lethe"
    home.mkdir(parents=True, exist_ok=True)
    return str(home / "agent.db")


def _make_embedder(model_name: str):
    try:
        from fastembed import TextEmbedding
    except ImportError as e:
        raise SystemExit(
            "embedder requires fastembed: pip install 'lethe[embed]'"
        ) from e
    model = TextEmbedding(model_name)
    def embed(text: str) -> list[float]:
        return list(next(iter(model.embed([text]))))
    return embed


def _open(args, *, need_embedder: bool) -> Lethe:
    db_path = args.db or _default_db()
    embedder = _make_embedder(args.embedder) if need_embedder else None
    return Lethe(db_path, vector_dim=args.dim, embedder=embedder)


def _parse_time(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    # Accept ISO 8601 datetime or raw unix timestamp.
    try:
        return float(s)
    except ValueError:
        pass
    return datetime.fromisoformat(s).timestamp()


def _emit(args, payload, fallback_lines: list[str]) -> None:
    if args.json:
        print(json.dumps(payload, default=str))
    else:
        for line in fallback_lines:
            print(line)


# ─── subcommands ───────────────────────────────────────────────────

def cmd_inscribe(args):
    lethe = _open(args, need_embedder=True)
    try:
        mid = lethe.inscribe(args.text)
        _emit(args, {"id": mid}, [f"inscribed id={mid}"])
    finally:
        lethe.close()


def cmd_recall(args):
    need_emb = not args.lexical
    lethe = _open(args, need_embedder=need_emb)
    try:
        hits = lethe.recall(
            args.query,
            k=args.k,
            hybrid=not args.no_hybrid,
            lexical=args.lexical,
            at=_parse_time(args.at),
        )
        payload = [
            {
                "id": h.memory.id,
                "text": h.memory.text,
                "depth": h.memory.depth,
                "score": h.score,
                "similarity": h.similarity,
            }
            for h in hits
        ]
        lines = [
            f"  [{h.score:+.4f}] depth={h.memory.depth:.2f}  "
            f"id={h.memory.id}  {h.memory.text}"
            for h in hits
        ] or ["(no matches)"]
        _emit(args, payload, lines)
    finally:
        lethe.close()


def cmd_release(args):
    lethe = _open(args, need_embedder=False)
    try:
        n = lethe.surrender(args.ids, mode="release")
        _emit(args, {"released": n}, [f"released {n}"])
    finally:
        lethe.close()


def cmd_purge(args):
    lethe = _open(args, need_embedder=False)
    try:
        n = lethe.surrender(args.ids, mode="purge")
        _emit(args, {"purged": n}, [f"purged {n}"])
    finally:
        lethe.close()


def cmd_supersede(args):
    lethe = _open(args, need_embedder=True)
    try:
        target = {"old": args.old_id, "new": args.new}
        if args.reason:
            target["reason"] = args.reason
        new_id = lethe.surrender(target, mode="supersede")
        _emit(args, {"old": args.old_id, "new_id": new_id},
              [f"superseded {args.old_id} → new id {new_id}"])
    finally:
        lethe.close()


def cmd_pin(args):
    lethe = _open(args, need_embedder=False)
    try:
        n = lethe.pin(args.id)
        _emit(args, {"pinned": n}, [f"pinned {n}"])
    finally:
        lethe.close()


def cmd_unpin(args):
    lethe = _open(args, need_embedder=False)
    try:
        n = lethe.unpin(args.id)
        _emit(args, {"unpinned": n}, [f"unpinned {n}"])
    finally:
        lethe.close()


def cmd_consolidate(args):
    lethe = _open(args, need_embedder=False)
    try:
        report = lethe.consolidate(tau_seconds=args.tau)
        payload = {
            "decayed":     report.decayed,
            "promoted":    report.promoted,
            "collapsed":   report.collapsed,
            "duration_ms": report.duration_ms,
        }
        lines = [
            "consolidated:",
            *(f"  {k:<12} {v}" for k, v in payload.items()),
        ]
        _emit(args, payload, lines)
    finally:
        lethe.close()


def cmd_blame(args):
    lethe = _open(args, need_embedder=True)
    try:
        entries = lethe.blame(args.query, k=args.k)
        payload = [
            {
                "id": e.memory.id,
                "text": e.memory.text,
                "depth": e.memory.depth,
                "superseded_by": e.superseded_by,
                "supersedes": e.supersedes,
            }
            for e in entries
        ]
        lines = []
        for e in entries:
            tail = ""
            if e.superseded_by:
                tail = f"  → superseded by #{e.superseded_by}"
            elif e.supersedes:
                tail = f"  ← supersedes {e.supersedes}"
            lines.append(
                f"  #{e.memory.id} depth={e.memory.depth:.2f}  "
                f"{e.memory.text}{tail}"
            )
        _emit(args, payload, lines or ["(no matches)"])
    finally:
        lethe.close()


def cmd_log(args):
    lethe = _open(args, need_embedder=False)
    try:
        events = lethe.log(
            kind=args.kind,
            since=_parse_time(args.since),
        )
        if args.limit and len(events) > args.limit:
            events = events[-args.limit:]
        payload = [
            {
                "id": ev.id,
                "memory_id": ev.memory_id,
                "kind": ev.kind,
                "depth_before": ev.depth_before,
                "depth_after": ev.depth_after,
                "timestamp": ev.timestamp,
            }
            for ev in events
        ]
        lines = [
            f"  {datetime.fromtimestamp(ev.timestamp).isoformat(timespec='seconds')}"
            f"  {ev.kind:<10} mem={ev.memory_id}  "
            f"depth {ev.depth_before} → {ev.depth_after}"
            for ev in events
        ] or ["(no events)"]
        _emit(args, payload, lines)
    finally:
        lethe.close()


# ─── parser ────────────────────────────────────────────────────────

def _add_global(p):
    p.add_argument("--db", help="SQLite path (default: $LETHE_DB or ~/.lethe/agent.db)")
    p.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2",
                   help="fastembed model name (default: MiniLM-L6-v2)")
    p.add_argument("--dim", type=int, default=384,
                   help="embedding dimension (default: 384)")
    p.add_argument("--json", action="store_true", help="machine-readable output")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lethe",
                                description="A river for AI memory.")
    _add_global(p)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("inscribe", help="store a fact")
    s.add_argument("text")
    s.set_defaults(fn=cmd_inscribe)

    s = sub.add_parser("recall", help="retrieve memories")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=5)
    s.add_argument("--no-hybrid", action="store_true",
                   help="vec-only (skip BM25 leg)")
    s.add_argument("--lexical", action="store_true",
                   help="pure BM25 ranking (for identifier lookup)")
    s.add_argument("--at", help="time-travel: ISO datetime or unix timestamp")
    s.set_defaults(fn=cmd_recall)

    s = sub.add_parser("release", help="depth → 0 (soft-evict, reversible)")
    s.add_argument("ids", nargs="+", type=int)
    s.set_defaults(fn=cmd_release)

    s = sub.add_parser("purge", help="delete from disk (compliance)")
    s.add_argument("ids", nargs="+", type=int)
    s.set_defaults(fn=cmd_purge)

    s = sub.add_parser("supersede", help="replace an old fact with a new one")
    s.add_argument("old_id", type=int)
    s.add_argument("--new", required=True, help="replacement text")
    s.add_argument("--reason", help="why (free-form note)")
    s.set_defaults(fn=cmd_supersede)

    s = sub.add_parser("pin", help="depth → +∞ (immune to gravity)")
    s.add_argument("id", type=int)
    s.set_defaults(fn=cmd_pin)

    s = sub.add_parser("unpin", help="unpin (depth → 1.0)")
    s.add_argument("id", type=int)
    s.set_defaults(fn=cmd_unpin)

    s = sub.add_parser("consolidate", help="apply Hypnos gravity to everything")
    s.add_argument("--tau", type=float, default=86400.0 * 7,
                   help="time constant in seconds (default: 7 days)")
    s.set_defaults(fn=cmd_consolidate)

    s = sub.add_parser("blame", help="supersession history of a belief")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=5)
    s.set_defaults(fn=cmd_blame)

    s = sub.add_parser("log", help="event log (audit trail)")
    s.add_argument("--kind", help="filter by event kind")
    s.add_argument("--since", help="ISO datetime or unix timestamp")
    s.add_argument("--limit", type=int, default=50)
    s.set_defaults(fn=cmd_log)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        rc = args.fn(args)
    except KeyboardInterrupt:
        return 130
    return rc or 0


if __name__ == "__main__":
    sys.exit(main())
