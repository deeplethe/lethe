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
  lethe consolidate [--tau SECONDS]
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


def _open(args, *, need_embedder: bool, need_signing: bool = False) -> Lethe:
    db_path = args.db or _default_db()
    embedder = _make_embedder(args.embedder) if need_embedder else None
    signing_key = None
    if need_signing:
        from lethe.receipt import default_key_path, load_private_key
        key_path = Path(getattr(args, "key", None) or default_key_path())
        if not key_path.exists():
            raise SystemExit(
                f"no signing key at {key_path}. run `lethe keygen` first."
            )
        signing_key = load_private_key(key_path)
    return Lethe(db_path, vector_dim=args.dim, embedder=embedder,
                 signing_key=signing_key)


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


def cmd_ingest(args):
    from lethe.ingest import DEFAULT_PATTERNS, ingest, iter_files
    root = Path(args.path)
    if not root.exists():
        raise SystemExit(f"path does not exist: {root}")
    patterns = args.glob or list(DEFAULT_PATTERNS)
    lethe = _open(args, need_embedder=True)
    try:
        if root.is_file():
            files = [root]
        else:
            files = list(iter_files(root, patterns))
        if not files:
            raise SystemExit(
                f"no files matched {patterns} under {root}"
            )
        def progress(s):
            if not args.json:
                print(f"  scanned={s['files_scanned']:>5}  "
                      f"chunks={s['chunks']:>6}  "
                      f"({s['wall_seconds']}s)", flush=True)
        if not args.json:
            print(f"ingesting {len(files)} file(s) under {root}:")
        stats = ingest(lethe, files,
                       encoding=args.encoding,
                       min_len=args.min_len,
                       batch=args.batch,
                       on_progress=progress)
        _emit(args, stats, [
            f"done: scanned {stats['files_scanned']}, "
            f"skipped {stats['files_skipped']}, "
            f"inscribed {stats['chunks']} chunks "
            f"in {stats['wall_seconds']}s",
        ])
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
    if args.signed:
        lethe = _open(args, need_embedder=False, need_signing=True)
        try:
            receipt = lethe.purge_with_receipt(args.ids)
            d = receipt.to_dict()
            if args.json:
                print(json.dumps(d))
            else:
                # Default: emit the receipt as JSON regardless — receipts are
                # designed to be persisted/shared, not eyeballed.
                print(json.dumps(d, indent=2))
        finally:
            lethe.close()
    else:
        lethe = _open(args, need_embedder=False)
        try:
            n = lethe.surrender(args.ids, mode="purge")
            _emit(args, {"purged": n}, [f"purged {n}"])
        finally:
            lethe.close()


def cmd_keygen(args):
    from lethe.receipt import (
        default_key_path, generate_key, save_private_key,
    )
    out = Path(args.out) if args.out else default_key_path()
    if out.exists() and not args.force:
        raise SystemExit(
            f"refusing to overwrite existing key at {out}. "
            f"pass --force or choose a different --out."
        )
    priv, pub = generate_key()
    save_private_key(out, priv)
    payload = {"private_key_path": str(out), "public_key": pub.hex()}
    if args.json:
        print(json.dumps(payload))
    else:
        print(f"wrote private key: {out}")
        print(f"public key:        {pub.hex()}")


def cmd_verify_receipt(args):
    from lethe.receipt import PurgeReceipt, verify_receipt
    data = json.loads(Path(args.receipt).read_text(encoding="utf-8"))
    receipt = PurgeReceipt.from_dict(data)
    lethe = None
    if args.db_check:
        lethe = _open(args, need_embedder=False)
    try:
        result = verify_receipt(receipt, lethe)
    finally:
        if lethe is not None:
            lethe.close()
    if args.json:
        print(json.dumps(result))
    else:
        verdict = "VALID" if result["valid"] else "INVALID"
        print(f"  verdict:         {verdict}")
        print(f"  signature_valid: {result['signature_valid']}")
        print(f"  log_root_valid:  {result['log_root_valid']}")
        if result["issues"]:
            print("  issues:")
            for i in result["issues"]:
                print(f"    - {i}")
    return 0 if result["valid"] else 1


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

    s = sub.add_parser("ingest",
                       help="batch-inscribe every paragraph of every text file under a directory")
    s.add_argument("path", help="file or directory")
    s.add_argument("--glob", action="append",
                   help="filename glob (repeat for multiple; default: *.md *.txt *.rst)")
    s.add_argument("--encoding", default="utf-8")
    s.add_argument("--min-len", type=int, default=16,
                   help="discard chunks shorter than N chars (default 16)")
    s.add_argument("--batch", type=int, default=256,
                   help="inscribe batch size (default 256)")
    s.set_defaults(fn=cmd_ingest)

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
    s.add_argument("--signed", action="store_true",
                   help="issue an Ed25519 signed PurgeReceipt (requires keygen)")
    s.add_argument("--key", help="path to signing key (default: ~/.lethe/keys/purge_signing.key)")
    s.set_defaults(fn=cmd_purge)

    s = sub.add_parser("keygen",
                       help="generate Ed25519 keypair for purge receipts")
    s.add_argument("--out", help="output path (default: ~/.lethe/keys/purge_signing.key)")
    s.add_argument("--force", action="store_true",
                   help="overwrite existing key file")
    s.set_defaults(fn=cmd_keygen)

    s = sub.add_parser("verify-receipt",
                       help="verify a PurgeReceipt JSON file")
    s.add_argument("receipt", help="path to receipt JSON")
    s.add_argument("--db-check", action="store_true",
                   help="also recompute Merkle root from --db and compare "
                        "(detects post-hoc event log tampering)")
    s.set_defaults(fn=cmd_verify_receipt)

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
