"""Lethe — a river for AI memory.

Everything is a depth.  inscribe puts a fact at depth=1 (the surface).
Hypnos's gravity drags it down.  mnemosyne pins it.  surrender lets it sink.
The river decides what survives.
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Callable, Literal, Optional, Union

import sqlite_vec

from lethe.types import (
    BlameEntry, Event, Memory, PINNED, Recalled, SUBMERGED, SURFACE,
)


@dataclass
class ConsolidateReport:
    decayed: int = 0
    promoted: int = 0
    collapsed: int = 0
    duration_ms: int = 0


SurrenderMode = Literal["decay", "release", "supersede", "purge", "edit"]


# ─── FTS5 sanitation ─────────────────────────────────────────────────

_FTS_SAFE = re.compile(r"[^\w\s]+")


def _sanitize_fts(q: str) -> str:
    """Strip FTS5 operator chars from a free-text query.
    The FTS5 tokenizer (porter unicode61) handles stop words itself."""
    return _FTS_SAFE.sub(" ", q).strip()


# ─── Lethe ───────────────────────────────────────────────────────────

class Lethe:
    """A single-file embedded memory store organized as a depth axis.

    Operations:
        inscribe(text)              add at depth=1.0 (the surface)
        recall(query, k, at=None)   retrieve memories still above water
                                    (at=T → return the view that existed at T)
        surrender(target, mode)     let memory sink:
            decay      depth *= 0.5
            release    depth = 0 (default)
            supersede  old.depth=0 + new at 1.0, linked by supersession edge
            purge      DELETE FROM memory (compliance only)
        mnemosyne(target)           pin: depth = +inf  (immune to gravity)
        blame(query)                show supersession history
        log(since, until)           audit event log
    """

    def __init__(
        self,
        db_path: Union[str, Path] = ":memory:",
        *,
        vector_dim: int = 768,
        embedder: Optional[Callable[[str], list[float]]] = None,
        embedder_batch: Optional[Callable[[list[str]], list[list[float]]]] = None,
        llm: Optional[Callable[[str], str]] = None,
        signing_key: Optional[bytes] = None,
    ):
        self.db_path = str(db_path)
        self.vector_dim = vector_dim
        self.embedder = embedder
        # If only a single-text embedder is provided, derive a batch wrapper
        # that simply maps over the list. Callers with a real batchable model
        # should pass embedder_batch for the 5-10x speedup.
        if embedder_batch is not None:
            self.embedder_batch = embedder_batch
        elif embedder is not None:
            self.embedder_batch = lambda texts: [embedder(t) for t in texts]
        else:
            self.embedder_batch = None
        self.llm = llm
        # Optional Ed25519 private key (32 bytes) for purge receipts.
        # When set, callers can request a signed PurgeReceipt via
        # purge_with_receipt().  Receipts let third parties verify a
        # deletion happened and the event log has not been tampered with.
        self.signing_key = signing_key

        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()

    def _init_schema(self) -> None:
        schema = (Path(__file__).parent / "schema.sql").read_text(encoding="utf-8")
        schema = schema.replace("{vector_dim}", str(self.vector_dim))
        self.conn.executescript(schema)
        self.conn.commit()

    # ─── append-only event log ───────────────────────────────────────

    def _log_event(
        self,
        memory_id: int,
        kind: str,
        depth_before: Optional[float],
        depth_after: Optional[float],
        meta: Optional[dict] = None,
    ) -> None:
        self.conn.execute(
            "INSERT INTO event(memory_id, kind, depth_before, depth_after, "
            "timestamp, meta) VALUES (?, ?, ?, ?, ?, ?)",
            (memory_id, kind, depth_before, depth_after, time(),
             json.dumps(meta) if meta else None),
        )

    # ─── inscribe ────────────────────────────────────────────────────

    def inscribe(self, text: str, *, meta: Optional[dict] = None) -> int:
        """Add `text` at depth=1.0. Returns the rowid (shared across the
        three surfaces: memory, memory_vec, memory_fts)."""
        emb = self.embedder(text) if self.embedder else [0.0] * self.vector_dim
        if len(emb) != self.vector_dim:
            raise ValueError(
                f"embedder returned dim {len(emb)} != configured {self.vector_dim}"
            )
        now = time()
        cur = self.conn.execute(
            "INSERT INTO memory(text, depth, created_at, last_access, "
            "access_count, meta) VALUES (?, ?, ?, ?, 0, ?)",
            (text, SURFACE, now, now,
             json.dumps(meta) if meta else None),
        )
        mid = cur.lastrowid
        self.conn.execute(
            "INSERT INTO memory_vec(rowid, embedding) VALUES (?, ?)",
            (mid, sqlite_vec.serialize_float32(emb)),
        )
        self.conn.execute(
            "INSERT INTO memory_fts(rowid, text) VALUES (?, ?)", (mid, text)
        )
        self._log_event(mid, "inscribe", None, SURFACE)
        self.conn.commit()
        return mid

    def inscribe_many(
        self,
        texts: list[str],
        *,
        metas: Optional[list[Optional[dict]]] = None,
    ) -> list[int]:
        """Batch inscribe. Embeds all texts in one call (if embedder_batch is
        available), writes everything in a single transaction. 5-10× faster
        than calling inscribe() N times for ONNX/transformer embedders."""
        if not texts:
            return []
        if metas is None:
            metas = [None] * len(texts)
        elif len(metas) != len(texts):
            raise ValueError("metas length must match texts length")

        if self.embedder_batch is not None:
            embeddings = self.embedder_batch(texts)
        else:
            embeddings = [[0.0] * self.vector_dim] * len(texts)

        now = time()
        rowids: list[int] = []
        for text, emb, meta in zip(texts, embeddings, metas):
            if len(emb) != self.vector_dim:
                raise ValueError(
                    f"embedder returned dim {len(emb)} != configured {self.vector_dim}"
                )
            cur = self.conn.execute(
                "INSERT INTO memory(text, depth, created_at, last_access, "
                "access_count, meta) VALUES (?, ?, ?, ?, 0, ?)",
                (text, SURFACE, now, now,
                 json.dumps(meta) if meta else None),
            )
            mid = cur.lastrowid
            self.conn.execute(
                "INSERT INTO memory_vec(rowid, embedding) VALUES (?, ?)",
                (mid, sqlite_vec.serialize_float32(emb)),
            )
            self.conn.execute(
                "INSERT INTO memory_fts(rowid, text) VALUES (?, ?)", (mid, text)
            )
            self._log_event(mid, "inscribe", None, SURFACE)
            rowids.append(mid)
        self.conn.commit()
        return rowids

    # ─── recall ──────────────────────────────────────────────────────

    def recall(
        self,
        query: str,
        *,
        k: int = 5,
        at: Optional[float] = None,
        hybrid: bool = True,
        lexical: bool = False,
    ) -> list[Recalled]:
        """Retrieve surfaced memories.

        at=T enables time-travel: reconstruct depth at T from event log,
             return only what was above water then.
        hybrid=False skips the BM25 leg (vector only).
        lexical=True forces pure BM25 (no vec, no RRF) — the right primitive
             for identifier lookup (emails, API keys, names), where vector
             similarity blurs near-paraphrases that are semantically distinct.
        """
        if lexical:
            return self._recall_lexical(query, k=k, at=at)

        q_emb = self.embedder(query) if self.embedder else None
        if q_emb is None:
            return self._recall_no_vec(query, k=k, at=at, hybrid=hybrid)

        over_k = max(k * 4, 20)
        q_blob = sqlite_vec.serialize_float32(q_emb)

        # ── vector leg ──
        if at is None:
            rows = self.conn.execute(
                """
                SELECT m.rowid, m.text, m.depth, m.created_at, m.last_access,
                       m.access_count, m.meta, v.distance
                FROM memory_vec v
                JOIN memory m ON m.rowid = v.rowid
                WHERE v.embedding MATCH ? AND k = ? AND m.depth > 0
                ORDER BY v.distance
                """,
                (q_blob, over_k),
            ).fetchall()
        else:
            # time-travel: depth at T from event log
            rows = self.conn.execute(
                """
                WITH state_at_t AS (
                    SELECT e.memory_id, e.depth_after AS depth
                    FROM event e
                    WHERE e.timestamp <= ?
                      AND NOT EXISTS (
                          SELECT 1 FROM event e2
                          WHERE e2.memory_id = e.memory_id
                            AND e2.timestamp > e.timestamp
                            AND e2.timestamp <= ?
                      )
                )
                SELECT m.rowid, m.text, s.depth, m.created_at, m.last_access,
                       m.access_count, m.meta, v.distance
                FROM memory_vec v
                JOIN memory m ON m.rowid = v.rowid
                JOIN state_at_t s ON s.memory_id = m.rowid
                WHERE v.embedding MATCH ? AND k = ? AND s.depth > 0
                ORDER BY v.distance
                """,
                (at, at, q_blob, over_k),
            ).fetchall()

        vec_ranks = {row[0]: (i + 1, row) for i, row in enumerate(rows)}

        # ── BM25 leg ──
        bm25_ranks: dict[int, int] = {}
        if hybrid and query:
            fts_q = _sanitize_fts(query)
            if fts_q:
                fts_rows = self.conn.execute(
                    """
                    SELECT f.rowid
                    FROM memory_fts f
                    JOIN memory m ON m.rowid = f.rowid
                    WHERE memory_fts MATCH ? AND m.depth > 0
                    ORDER BY bm25(memory_fts)
                    LIMIT ?
                    """,
                    (fts_q, over_k),
                ).fetchall()
                for i, (rid,) in enumerate(fts_rows):
                    bm25_ranks[rid] = i + 1

        # ── RRF fusion (equal weight; missing leg contributes 0) ──
        RRF_K = 60.0
        all_ids = set(vec_ranks) | set(bm25_ranks)
        scored: list[Recalled] = []
        for rid in all_ids:
            rrf = 0.0
            if rid in vec_ranks:
                rrf += 1.0 / (RRF_K + vec_ranks[rid][0])
            if rid in bm25_ranks:
                rrf += 1.0 / (RRF_K + bm25_ranks[rid])

            if rid in vec_ranks:
                row = vec_ranks[rid][1]
                mem = self._row_to_memory(row[:-1])      # drop distance
                # vec0 returns squared-L2 distance on normalized vectors,
                # i.e. d² = 2(1 - cos).  So cos = 1 - d²/2.
                d2 = row[-1]
                sim = 1.0 - d2 / 2.0
            else:
                row = self.conn.execute(
                    "SELECT rowid, text, depth, created_at, last_access, "
                    "access_count, meta FROM memory WHERE rowid = ?", (rid,)
                ).fetchone()
                if not row:
                    continue
                mem = self._row_to_memory(row)
                sim = 0.0

            scored.append(Recalled(memory=mem, score=rrf, similarity=sim))

        scored.sort(key=lambda r: r.score, reverse=True)
        results = scored[:k]

        # ── side effect: record access (only for present-time recalls) ──
        if at is None:
            for r in results:
                self._record_access(r.memory.id)

        return results

    def _recall_lexical(self, query, *, k, at):
        # Pure BM25 ranking. SQLite's bm25() is "lower = better"; we negate
        # so score follows the "higher = better" convention of recall().
        # FTS5's implicit operator is AND; with extra qualifier words in the
        # query ("medical record", "customer history") any single missing
        # word kills the match.  For lexical *ranking* (vs. boolean filter)
        # we want OR — BM25 will still rank rows by how many query tokens
        # they hit and how distinctive each token is.
        sanitized = _sanitize_fts(query)
        if not sanitized:
            return []
        terms = [t for t in sanitized.split() if t]
        if not terms:
            return []
        fts_q = " OR ".join(terms)
        if at is None:
            rows = self.conn.execute(
                """
                SELECT m.rowid, m.text, m.depth, m.created_at, m.last_access,
                       m.access_count, m.meta, bm25(memory_fts) AS bm
                FROM memory_fts f JOIN memory m ON m.rowid = f.rowid
                WHERE memory_fts MATCH ? AND m.depth > 0
                ORDER BY bm LIMIT ?
                """,
                (fts_q, k),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                WITH state_at_t AS (
                    SELECT e.memory_id, e.depth_after AS depth
                    FROM event e
                    WHERE e.timestamp <= ?
                      AND NOT EXISTS (
                          SELECT 1 FROM event e2
                          WHERE e2.memory_id = e.memory_id
                            AND e2.timestamp > e.timestamp
                            AND e2.timestamp <= ?
                      )
                )
                SELECT m.rowid, m.text, s.depth, m.created_at, m.last_access,
                       m.access_count, m.meta, bm25(memory_fts) AS bm
                FROM memory_fts f
                JOIN memory m ON m.rowid = f.rowid
                JOIN state_at_t s ON s.memory_id = m.rowid
                WHERE memory_fts MATCH ? AND s.depth > 0
                ORDER BY bm LIMIT ?
                """,
                (at, at, fts_q, k),
            ).fetchall()
        out: list[Recalled] = []
        for r in rows:
            mem = self._row_to_memory(r[:-1])
            out.append(Recalled(memory=mem, score=-r[-1], similarity=0.0))
        if at is None:
            for r in out:
                self._record_access(r.memory.id)
        return out

    def _recall_no_vec(self, query, *, k, at, hybrid):
        # No embedder → can't do vec search. Fall back to BM25 only.
        if not hybrid or not query:
            rows = self.conn.execute(
                "SELECT rowid, text, depth, created_at, last_access, "
                "access_count, meta FROM memory WHERE depth > 0 "
                "ORDER BY depth DESC LIMIT ?", (k,)
            ).fetchall()
            return [Recalled(memory=self._row_to_memory(r), score=r[2])
                    for r in rows]

        fts_q = _sanitize_fts(query)
        if not fts_q:
            return []
        rows = self.conn.execute(
            """
            SELECT m.rowid, m.text, m.depth, m.created_at, m.last_access,
                   m.access_count, m.meta
            FROM memory_fts f JOIN memory m ON m.rowid = f.rowid
            WHERE memory_fts MATCH ? AND m.depth > 0
            ORDER BY bm25(memory_fts) LIMIT ?
            """,
            (fts_q, k),
        ).fetchall()
        return [Recalled(memory=self._row_to_memory(r), score=r[2])
                for r in rows]

    def _row_to_memory(self, row) -> Memory:
        rid, text, depth, created_at, last_access, ac, meta_json = row
        return Memory(
            id=rid, text=text or "", depth=depth or 0.0,
            created_at=created_at or 0.0, last_access=last_access or 0.0,
            access_count=ac or 0,
            meta=json.loads(meta_json) if meta_json else {},
        )

    def _record_access(self, memory_id: int) -> None:
        now = time()
        self.conn.execute(
            "UPDATE memory SET access_count = access_count + 1, "
            "last_access = ? WHERE rowid = ?", (now, memory_id),
        )
        self._log_event(memory_id, "recall", None, None)
        self.conn.commit()

    # ─── surrender ───────────────────────────────────────────────────

    def surrender(
        self,
        target: Union[int, list[int], dict],
        *,
        mode: SurrenderMode = "release",
        new_text: Optional[str] = None,
    ) -> int:
        if mode == "decay":
            return self._apply_force(target, factor=0.5, kind="decay")
        if mode == "release":
            return self._set_depth(target, SUBMERGED, kind="release")
        if mode == "supersede":
            if not isinstance(target, dict) or "old" not in target or "new" not in target:
                raise ValueError("supersede requires target={'old': id, 'new': text}")
            return self._supersede(target["old"], target["new"],
                                   reason=target.get("reason", ""))
        if mode == "purge":
            return self._purge(target)
        if mode == "edit":
            if new_text is None:
                raise ValueError("edit requires new_text=...")
            return self._edit(target, new_text)
        raise ValueError(f"unknown mode: {mode}")

    def _apply_force(self, target, *, factor: float, kind: str) -> int:
        ids = self._coerce_ids(target)
        for mid in ids:
            row = self.conn.execute(
                "SELECT depth FROM memory WHERE rowid = ?", (mid,)
            ).fetchone()
            if not row:
                continue
            before = row[0]
            if math.isinf(before):
                continue                       # pinned — no force applies
            after = before * factor
            self.conn.execute(
                "UPDATE memory SET depth = ? WHERE rowid = ?", (after, mid)
            )
            self._log_event(mid, kind, before, after)
        self.conn.commit()
        return len(ids)

    def _set_depth(self, target, depth: float, *, kind: str) -> int:
        ids = self._coerce_ids(target)
        for mid in ids:
            row = self.conn.execute(
                "SELECT depth FROM memory WHERE rowid = ?", (mid,)
            ).fetchone()
            if not row:
                continue
            before = row[0]
            self.conn.execute(
                "UPDATE memory SET depth = ? WHERE rowid = ?", (depth, mid)
            )
            self._log_event(mid, kind, before, depth)
        self.conn.commit()
        return len(ids)

    def _supersede(self, old_id: int, new_text: str, *, reason: str = "") -> int:
        new_id = self.inscribe(new_text)
        old_row = self.conn.execute(
            "SELECT depth FROM memory WHERE rowid = ?", (old_id,)
        ).fetchone()
        if old_row:
            before = old_row[0]
            self.conn.execute(
                "UPDATE memory SET depth = ? WHERE rowid = ?",
                (SUBMERGED, old_id),
            )
            self._log_event(old_id, "supersede", before, SUBMERGED,
                            meta={"by": new_id, "reason": reason})
        self.conn.execute(
            "INSERT INTO supersession(new_id, old_id, timestamp, reason) "
            "VALUES (?, ?, ?, ?)", (new_id, old_id, time(), reason)
        )
        self.conn.commit()
        return new_id

    def _edit(self, target, new_text: str) -> int:
        """Replace the text (and re-index) of `target` row(s) without
        changing depth.  Logs an `edit` event so time-travel still works.

        Use case: a memory carries multiple facts as one inscribed row
        (e.g. ``"User lives in Berlin and works at Stripe"``) and the
        caller wants to update only one fact while preserving the
        others.  Edit is the *partial-supersede* primitive that the
        atomic ``supersede`` cannot express."""
        ids = self._coerce_ids(target)
        for mid in ids:
            row = self.conn.execute(
                "SELECT depth FROM memory WHERE rowid = ?", (mid,)
            ).fetchone()
            if not row:
                continue
            depth = row[0]
            # Re-embed.  We need the embedder to be configured; if
            # not, edit still rewrites the text but the vector stays
            # stale (callers without an embedder shouldn't be using
            # vector recall anyway).
            if self.embedder is not None:
                new_emb = self.embedder(new_text)
                if len(new_emb) != self.vector_dim:
                    raise ValueError(
                        f"embedder returned dim {len(new_emb)} != "
                        f"configured {self.vector_dim}"
                    )
                self.conn.execute(
                    "DELETE FROM memory_vec WHERE rowid = ?", (mid,)
                )
                self.conn.execute(
                    "INSERT INTO memory_vec(rowid, embedding) VALUES (?, ?)",
                    (mid, sqlite_vec.serialize_float32(new_emb)),
                )
            # Always update text + FTS5 (no embedder still re-indexes
            # the lexical side).
            self.conn.execute(
                "UPDATE memory SET text = ? WHERE rowid = ?",
                (new_text, mid),
            )
            self.conn.execute(
                "DELETE FROM memory_fts WHERE rowid = ?", (mid,)
            )
            self.conn.execute(
                "INSERT INTO memory_fts(rowid, text) VALUES (?, ?)",
                (mid, new_text),
            )
            # Depth doesn't change; we log an edit event with the same
            # before/after depth so the event-log invariant holds and
            # time-travel can reconstruct the text at any past instant.
            self._log_event(mid, "edit", depth, depth,
                            meta={"new_text": new_text})
        self.conn.commit()
        return len(ids)

    def _purge(self, target) -> int:
        ids = self._coerce_ids(target)
        for mid in ids:
            self._log_event(mid, "purge", None, None)
            self.conn.execute("DELETE FROM memory WHERE rowid = ?", (mid,))
            self.conn.execute("DELETE FROM memory_vec WHERE rowid = ?", (mid,))
            self.conn.execute("DELETE FROM memory_fts WHERE rowid = ?", (mid,))
        self.conn.commit()
        return len(ids)

    def purge_with_receipt(self, target):
        """Purge + issue an Ed25519-signed receipt anchored to the event-log
        Merkle root.  Requires `signing_key` at construction time.

        Returns a PurgeReceipt; verify with `lethe.receipt.verify_receipt`.
        """
        if self.signing_key is None:
            raise RuntimeError(
                "Lethe(...) needs signing_key=<32 bytes> to issue receipts. "
                "Generate one via `lethe.receipt.generate_key()`."
            )
        from lethe.receipt import (
            PurgeReceipt, event_hash, merkle_root,
            public_key_from_private, sign, text_hash,
        )
        requested = self._coerce_ids(target)
        # Snapshot the texts BEFORE delete, AND filter to rows that actually
        # exist.  The receipt must reflect what was truly purged, not what
        # the caller asked for — passing a non-existent id should not show
        # up as a purged record in the audit trail.
        if requested:
            rows = self.conn.execute(
                f"SELECT rowid, text FROM memory "
                f"WHERE rowid IN ({','.join('?' * len(requested))})",
                requested,
            ).fetchall()
        else:
            rows = []
        existing_ids = [rid for rid, _ in rows]
        text_hashes_by_id = {rid: text_hash(text) for rid, text in rows}
        # Only purge what actually exists, so the event log doesn't get
        # spurious "purge" events for nonexistent ids.
        self._purge(existing_ids)
        # Collect the purge events just produced (one per id we just purged).
        if existing_ids:
            purge_evs = self.conn.execute(
                f"""SELECT id FROM event
                    WHERE kind='purge'
                      AND memory_id IN ({','.join('?' * len(existing_ids))})
                    ORDER BY id DESC LIMIT ?""",
                existing_ids + [len(existing_ids)],
            ).fetchall()
            purge_event_ids = sorted(r[0] for r in purge_evs)
        else:
            purge_event_ids = []
        # Snapshot the entire event log and commit to its Merkle root.
        all_events = self.conn.execute(
            "SELECT id, memory_id, kind, depth_before, depth_after, "
            "timestamp, meta FROM event ORDER BY id"
        ).fetchall()
        last_id = all_events[-1][0] if all_events else 0
        root = merkle_root([event_hash(r) for r in all_events])
        pub = public_key_from_private(self.signing_key)
        receipt = PurgeReceipt(
            version="1",
            purged_ids=existing_ids,
            purged_text_hashes=[text_hashes_by_id[rid] for rid in existing_ids],
            purge_event_ids=purge_event_ids,
            last_event_id=last_id,
            event_log_merkle_root=root.hex(),
            timestamp=time(),
            public_key=pub.hex(),
        )
        receipt.signature = sign(receipt.canonical_payload(),
                                 self.signing_key).hex()
        return receipt

    # ─── pin (was: mnemosyne) ────────────────────────────────────────

    def pin(self, target: Union[int, list[int]]) -> int:
        """Pin a memory above the surface: depth = +inf.
        Immune to gravity, immune to surrender(decay/release).
        Only `purge` can still remove it."""
        return self._set_depth(target, PINNED, kind="pin")

    def unpin(self, target: Union[int, list[int]]) -> int:
        """Inverse of pin: drop back to surface (depth = 1.0)."""
        return self._set_depth(target, SURFACE, kind="unpin")

    # ─── consolidate (was: Hypnos.flow) ─────────────────────────────

    def consolidate(
        self,
        *,
        tau_seconds: float = 86400 * 7,
        promote_min_access: int = 5,
        promote_min_depth: float = 0.6,
        promote_to: float = 1.5,
        now: Optional[float] = None,
    ) -> "ConsolidateReport":
        """Apply gravity to every unpinned memory above the waterline.

            depth *= exp(-age / (tau * (1 + log(1 + access_count))))

        Then promote any survivor whose access_count >= promote_min_access
        AND depth >= promote_min_depth up to `promote_to` (above the surface).

        Pinned memories (depth=+inf) and submerged ones (depth<=0) are skipped.
        """
        started = time()
        clock = now if now is not None else started
        report = ConsolidateReport()

        rows = self.conn.execute(
            "SELECT rowid, depth, last_access, access_count FROM memory "
            "WHERE depth > 0 AND depth < ?", (PINNED,)
        ).fetchall()

        for rid, depth, last_access, ac in rows:
            age = clock - (last_access or clock)
            if age <= 0:
                continue
            S = tau_seconds * (1.0 + math.log(1.0 + (ac or 0)))
            factor = math.exp(-age / S)
            new_depth = depth * factor
            self.conn.execute(
                "UPDATE memory SET depth = ? WHERE rowid = ?", (new_depth, rid)
            )
            self._log_event(rid, "flow", depth, new_depth)
            report.decayed += 1

        # Above-surface promotion (was: aletheia)
        promo = self.conn.execute(
            "SELECT rowid, depth FROM memory "
            "WHERE depth >= ? AND depth < ? AND access_count >= ?",
            (promote_min_depth, PINNED, promote_min_access),
        ).fetchall()
        for rid, before in promo:
            self.conn.execute(
                "UPDATE memory SET depth = ? WHERE rowid = ?",
                (promote_to, rid),
            )
            self._log_event(rid, "promote", before, promote_to)
            report.promoted += 1

        self.conn.commit()
        report.duration_ms = int((time() - started) * 1000)
        return report

    # ─── time-travel / audit ─────────────────────────────────────────

    def blame(self, query: str, *, k: int = 5) -> list[BlameEntry]:
        """For memories matching `query`, return their supersession context."""
        hits = self.recall(query, k=k)
        entries: list[BlameEntry] = []
        for hit in hits:
            mid = hit.memory.id
            by = self.conn.execute(
                "SELECT new_id, timestamp FROM supersession WHERE old_id = ? "
                "ORDER BY timestamp DESC LIMIT 1", (mid,)
            ).fetchone()
            of = [r[0] for r in self.conn.execute(
                "SELECT old_id FROM supersession WHERE new_id = ?", (mid,)
            ).fetchall()]
            entries.append(BlameEntry(
                memory=hit.memory,
                superseded_by=by[0] if by else None,
                superseded_at=by[1] if by else None,
                supersedes=of,
            ))
        return entries

    def log(
        self,
        *,
        since: Optional[float] = None,
        until: Optional[float] = None,
        kind: Optional[str] = None,
    ) -> list[Event]:
        sql = "SELECT id, memory_id, kind, depth_before, depth_after, " \
              "timestamp, meta FROM event WHERE 1=1"
        args: list = []
        if since is not None:
            sql += " AND timestamp >= ?"
            args.append(since)
        if until is not None:
            sql += " AND timestamp <= ?"
            args.append(until)
        if kind is not None:
            sql += " AND kind = ?"
            args.append(kind)
        sql += " ORDER BY timestamp ASC"
        rows = self.conn.execute(sql, args).fetchall()
        out: list[Event] = []
        for r in rows:
            eid, mid, ek, db, da, ts, meta_json = r
            out.append(Event(
                id=eid, memory_id=mid, kind=ek,
                depth_before=db, depth_after=da, timestamp=ts,
                meta=json.loads(meta_json) if meta_json else {},
            ))
        return out

    # ─── internal helpers ────────────────────────────────────────────

    @staticmethod
    def _coerce_ids(target) -> list[int]:
        if target is None:
            return []
        if isinstance(target, int):
            return [target]
        if isinstance(target, list):
            return list(target)
        raise TypeError(f"expected int or list[int], got {type(target).__name__}")

    def close(self) -> None:
        self.conn.close()
