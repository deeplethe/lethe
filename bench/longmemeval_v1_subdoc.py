"""LongMemEval-S — Lethe v1, sub-document (per-turn) indexing.

Each USER turn becomes its own memory. The session is recovered at recall
time by aggregating turn-level matches via first-occurrence rank.

The hypothesis: abstract queries (e.g. "Any tips on what to bake?") match
specific turns ("I like lemon lavender pound cake") better than the
diluted concatenated session. No regex, no query detection — fix at the
index, not the query.
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Optional
from uuid import uuid4

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "bench"))

from mempalace_bench import evaluate_retrieval  # type: ignore

from lethe import Lethe


MAX_CHARS = 8000        # per single turn (smaller than per-session)
RECALL_OVER_K = 200     # pull a lot of turns, aggregate to ~30 unique sessions


# ─── ProcessPool worker state ────────────────────────────────────────
# Each subprocess loads its own fastembed model once (in _worker_init).
# Per-question calls reuse it.
_PROC_MODEL = None
_PROC_DIM = None


def _worker_init(model_name: str, dim: int) -> None:
    import os as _os
    _os.environ["OMP_NUM_THREADS"] = "1"
    _os.environ["ONNX_NUM_THREADS"] = "1"
    _os.environ["MKL_NUM_THREADS"] = "1"
    from fastembed import TextEmbedding
    global _PROC_MODEL, _PROC_DIM
    try:
        _PROC_MODEL = TextEmbedding(model_name, threads=1)
    except TypeError:
        _PROC_MODEL = TextEmbedding(model_name)
    _PROC_DIM = dim


def _worker_run(args_tuple):
    idx, entry, db_root_str = args_tuple
    db_root = Path(db_root_str)
    def emb(text: str) -> list[float]:
        return list(next(iter(_PROC_MODEL.embed([text]))))
    try:
        result = run_one(idx, entry, emb, None, _PROC_DIM, db_root)
    except Exception as ex:
        result = {
            "question_id": entry["question_id"],
            "question_type": entry["question_type"],
            "ranked": [], "corpus_ids": [], "skipped": True,
            "error": f"{type(ex).__name__}: {ex}",
        }
    return idx, entry, result


def run_one(idx: int, entry: dict, embedder, embedder_batch, embed_dim: int,
            db_root: Path) -> dict:
    # Unique session ids in encounter order — acts as our "corpus"
    session_ids: list[str] = []
    seen_sid = set()
    for sid in entry["haystack_session_ids"]:
        if sid not in seen_sid:
            session_ids.append(sid)
            seen_sid.add(sid)
    sid_to_idx = {s: i for i, s in enumerate(session_ids)}

    if not session_ids:
        return {
            "question_id": entry["question_id"],
            "question_type": entry["question_type"],
            "ranked": [], "corpus_ids": [], "skipped": True,
        }

    db = db_root / f"q{idx}.db"
    inscribe_start = time.perf_counter()
    lethe = Lethe(str(db), vector_dim=embed_dim,
                  embedder=embedder, embedder_batch=embedder_batch)

    # Collect all turns + their session indices, then inscribe in one batch
    texts: list[str] = []
    metas: list[Optional[dict]] = []
    pending_sidx: list[int] = []
    for session, sid in zip(entry["haystack_sessions"],
                            entry["haystack_session_ids"]):
        # Pair each user turn with the immediately-following assistant turn
        # (if any) — the natural dialog unit. ~half the docs of per-turn,
        # but still preserves assistant content for "what did you say" queries.
        i = 0
        while i < len(session):
            t = session[i]
            role = t.get("role")
            content = (t.get("content") or "").strip()
            if role == "user" and content:
                parts = ["[user] " + content]
                if i + 1 < len(session) and session[i + 1].get("role") == "assistant":
                    asst = (session[i + 1].get("content") or "").strip()
                    if asst:
                        parts.append("[assistant] " + asst)
                    i += 2
                else:
                    i += 1
            elif role == "assistant" and content:
                parts = ["[assistant] " + content]
                i += 1
            else:
                i += 1
                continue

            text = "\n".join(parts)
            if len(text) > MAX_CHARS:
                text = text[: MAX_CHARS // 2] + "\n...\n" + text[-MAX_CHARS // 2:]
            texts.append(text)
            metas.append({"session_id": sid})
            pending_sidx.append(sid_to_idx[sid])

    rowids = lethe.inscribe_many(texts, metas=metas)
    rowid_to_sid_idx = {rid: sidx for rid, sidx in zip(rowids, pending_sidx)}
    n_turns = len(rowids)
    inscribe_ms = (time.perf_counter() - inscribe_start) * 1000

    recall_start = time.perf_counter()
    # Sub-document indexing: per-turn vectors are already specific enough
    # that BM25 adds more noise than signal. Pure vector recall.
    results = lethe.recall(entry["question"], k=RECALL_OVER_K, hybrid=False)
    recall_ms = (time.perf_counter() - recall_start) * 1000

    # ── Aggregate turn ranks → session ranks via first-occurrence ──
    # The best-matching turn of a session determines that session's rank.
    # (K-RRF aggregation was tried and rewarded "many-weak-matches" sessions
    # over "one-strong-match" ones, which is wrong for this benchmark.)
    ranked: list[int] = []
    seen_sessions: set[int] = set()
    for r in results:
        sidx = rowid_to_sid_idx.get(r.memory.id)
        if sidx is None or sidx in seen_sessions:
            continue
        ranked.append(sidx)
        seen_sessions.add(sidx)
    for i in range(len(session_ids)):
        if i not in seen_sessions:
            ranked.append(i)

    lethe.close()
    try:
        for path in db_root.glob(f"q{idx}.db*"):
            path.unlink(missing_ok=True)
    except (PermissionError, OSError):
        pass

    return {
        "question_id": entry["question_id"],
        "question_type": entry["question_type"],
        "ranked": ranked,
        "corpus_ids": session_ids,
        "skipped": False,
        "inscribe_ms": inscribe_ms,
        "recall_ms": recall_ms,
        "n_turns": n_turns,
        "n_sessions": len(session_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--out", type=Path,
                        default=REPO / "bench" / "results_v1_subdoc.jsonl")
    args = parser.parse_args()

    if sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    from fastembed import TextEmbedding
    print(f"  loading embedder: {args.embedder}")
    model = TextEmbedding(args.embedder)
    # Bound fastembed's internal batch to cap memory.  Default is 256;
    # at 32 the peak intermediate tensors stay small.  Throughput hit is
    # tiny because the batch is already passed in big chunks at the call
    # site — fastembed splits internally.
    EMBED_BATCH_SIZE = 32
    def embedder(text: str) -> list[float]:
        return list(next(iter(model.embed([text], batch_size=EMBED_BATCH_SIZE))))
    def embedder_batch(texts: list[str]) -> list[list[float]]:
        return [list(v) for v in model.embed(texts, batch_size=EMBED_BATCH_SIZE)]

    data = json.loads(args.data_file.read_text(encoding="utf-8"))
    if args.limit:
        data = data[: args.limit]

    db_root = REPO / "bench" / f"_subdoc_{uuid4().hex[:8]}"
    db_root.mkdir(exist_ok=True)

    print(f"\n  data:        {args.data_file.name}")
    print(f"  questions:   {len(data)}")
    print(f"  workers:     {args.workers}")
    print(f"  embedder:    {args.embedder} (dim={args.dim})")
    print(f"  granularity: per-USER-turn (sub-document)\n")

    ks = [1, 3, 5, 10, 30, 50]
    metrics = defaultdict(list)
    per_type = defaultdict(lambda: defaultdict(list))

    # Peak RSS tracker — includes worker subprocesses
    try:
        import psutil
        _proc = psutil.Process()
        def _rss_mb() -> float:
            total = _proc.memory_info().rss
            for child in _proc.children(recursive=True):
                try:
                    total += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            return total / (1024 * 1024)
    except ImportError:
        _proc = None
        def _rss_mb() -> float:
            return 0.0

    started = time.perf_counter()
    out_fh = args.out.open("w", encoding="utf-8")
    lock = Lock()
    completed = 0
    results: list[dict] = []
    peak_rss = _rss_mb()

    def task(i, e):
        try:
            return i, e, run_one(i, e, embedder, embedder_batch, args.dim, db_root)
        except Exception as ex:
            return i, e, {
                "question_id": e["question_id"],
                "question_type": e["question_type"],
                "ranked": [], "corpus_ids": [], "skipped": True,
                "error": f"{type(ex).__name__}: {ex}",
            }

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(task, i, e) for i, e in enumerate(data)]
        for fut in as_completed(futures):
            i, entry, r = fut.result()
            with lock:
                completed += 1
                results.append(r)
                out_fh.write(json.dumps(r) + "\n")
                out_fh.flush()
                if not r["skipped"]:
                    answer = set(entry["answer_session_ids"])
                    sids = r["corpus_ids"]
                    for k in ks:
                        ra, _, nd = evaluate_retrieval(r["ranked"], answer, sids, k)
                        metrics[f"recall_any@{k}"].append(ra)
                        metrics[f"ndcg@{k}"].append(nd)
                    per_type[entry["question_type"]]["r@5"].append(
                        metrics["recall_any@5"][-1]
                    )
                    per_type[entry["question_type"]]["r@10"].append(
                        metrics["recall_any@10"][-1]
                    )
                rss = _rss_mb()
                if rss > peak_rss:
                    peak_rss = rss
                if completed % 25 == 0 or completed == len(data):
                    r5 = (sum(metrics["recall_any@5"]) /
                          max(1, len(metrics["recall_any@5"])))
                    elapsed = time.perf_counter() - started
                    eta = elapsed / completed * (len(data) - completed)
                    print(f"  [{completed:>3}/{len(data)}]  r@5={r5:.3f}  "
                          f"elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m  "
                          f"rss={peak_rss:.0f}MB")

    out_fh.close()
    wall = time.perf_counter() - started
    try:
        shutil.rmtree(db_root, ignore_errors=True)
    except Exception:
        pass

    print(f"\n{'=' * 60}")
    print(f"  RESULTS — Lethe v1 sub-document")
    print(f"{'=' * 60}")
    print(f"  Wall: {wall:.1f}s ({wall/len(data):.2f}s/q)")
    print(f"  Peak RSS: {peak_rss:.0f} MB\n")
    print("  SESSION-LEVEL:")
    for k in ks:
        ra = sum(metrics[f"recall_any@{k}"]) / len(metrics[f"recall_any@{k}"])
        nd = sum(metrics[f"ndcg@{k}"]) / len(metrics[f"ndcg@{k}"])
        print(f"    Recall@{k:2}: {ra:.3f}    NDCG@{k:2}: {nd:.3f}")
    print("\n  PER-TYPE R@5 / R@10:")
    for qt in sorted(per_type):
        v = per_type[qt]
        r5 = sum(v["r@5"]) / len(v["r@5"])
        r10 = sum(v["r@10"]) / len(v["r@10"])
        print(f"    {qt:30}  R@5={r5:.3f}  R@10={r10:.3f}  (n={len(v['r@5'])})")

    ok = [r for r in results if not r.get("skipped")]
    if ok:
        ins = [r["inscribe_ms"] for r in ok]
        rec = [r["recall_ms"] for r in ok]
        turns = [r["n_turns"] for r in ok]
        print(f"\n  TIMING / GRANULARITY:")
        print(f"    user turns/Q:   avg={statistics.mean(turns):.0f}  "
              f"min={min(turns)}  max={max(turns)}")
        print(f"    inscribe/Q:     avg={statistics.mean(ins):.0f}ms  "
              f"p95={sorted(ins)[int(len(ins)*0.95)]:.0f}ms")
        print(f"    recall/Q:       avg={statistics.mean(rec):.0f}ms  "
              f"p95={sorted(rec)[int(len(rec)*0.95)]:.0f}ms")


if __name__ == "__main__":
    main()
