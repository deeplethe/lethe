"""LongMemEval-S runner for Lethe v1. Regression check vs v0 numbers.

v0 baseline on this benchmark (MiniLM + adaptive hybrid + 0 API):
  R@5 = 97.2%, R@1 = 87.4%

If v1's depth-physics scoring matches or improves on these, the
refactor is correct.

Run:
    py bench\\longmemeval_v1.py longmemeval_s.json
    py bench\\longmemeval_v1.py longmemeval_s.json --limit 30 --workers 1
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from uuid import uuid4

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "bench"))

from mempalace_bench import evaluate_retrieval  # type: ignore

from lethe import Lethe


# bge-m3 has 8192-token limit; truncate long sessions
MAX_CHARS = 24000


def build_corpus(entry: dict) -> tuple[list[str], list[str]]:
    """Mirror MemPalace's raw mode: one doc per session, user turns joined."""
    corpus: list[str] = []
    ids: list[str] = []
    for session, sid in zip(entry["haystack_sessions"], entry["haystack_session_ids"]):
        user_turns = [t["content"] for t in session if t.get("role") == "user"]
        if user_turns:
            corpus.append("\n".join(user_turns))
            ids.append(sid)
    return corpus, ids


def run_one(idx: int, entry: dict, embedder, embed_dim: int, db_root: Path) -> dict:
    corpus_texts, corpus_ids = build_corpus(entry)
    if not corpus_texts:
        return {
            "question_id": entry["question_id"],
            "question_type": entry["question_type"],
            "ranked": [],
            "corpus_ids": [],
            "skipped": True,
        }

    db = db_root / f"q{idx}.db"
    # New unique DB per question — avoids Windows file lock collisions
    inscribe_start = time.perf_counter()
    lethe = Lethe(str(db), vector_dim=embed_dim, embedder=embedder)

    rowid_to_corpus_idx: dict[int, int] = {}
    for i, text in enumerate(corpus_texts):
        if len(text) > MAX_CHARS:
            text = text[: MAX_CHARS // 2] + "\n...\n" + text[-MAX_CHARS // 2:]
        rid = lethe.inscribe(text)
        rowid_to_corpus_idx[rid] = i
    inscribe_ms = (time.perf_counter() - inscribe_start) * 1000

    recall_start = time.perf_counter()
    results = lethe.recall(entry["question"], k=50)
    recall_ms = (time.perf_counter() - recall_start) * 1000

    ranked = [rowid_to_corpus_idx[r.memory.id] for r in results
              if r.memory.id in rowid_to_corpus_idx]
    seen = set(ranked)
    for i in range(len(corpus_texts)):
        if i not in seen:
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
        "corpus_ids": corpus_ids,
        "skipped": False,
        "inscribe_ms": inscribe_ms,
        "recall_ms": recall_ms,
        "n_corpus": len(corpus_texts),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--out", type=Path,
                        default=REPO / "bench" / "results_v1.jsonl")
    args = parser.parse_args()

    if sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    from fastembed import TextEmbedding
    print(f"  loading embedder: {args.embedder}")
    model = TextEmbedding(args.embedder)
    def embedder(text: str) -> list[float]:
        return list(next(iter(model.embed([text]))))

    data = json.loads(args.data_file.read_text(encoding="utf-8"))
    if args.limit:
        data = data[: args.limit]

    db_root = REPO / "bench" / f"_run_{uuid4().hex[:8]}"
    db_root.mkdir(exist_ok=True)

    print(f"\n  data:        {args.data_file.name}")
    print(f"  questions:   {len(data)}")
    print(f"  workers:     {args.workers}")
    print(f"  embedder:    {args.embedder} (dim={args.dim})\n")

    ks = [1, 3, 5, 10, 30, 50]
    metrics_session = defaultdict(list)
    per_type = defaultdict(lambda: defaultdict(list))

    started = time.perf_counter()
    out_fh = args.out.open("w", encoding="utf-8")
    lock = Lock()
    completed = 0
    results: list[dict] = []

    def task(i, e):
        try:
            return i, e, run_one(i, e, embedder, args.dim, db_root)
        except Exception as ex:
            return i, e, {
                "question_id": e["question_id"],
                "question_type": e["question_type"],
                "ranked": [], "corpus_ids": [],
                "skipped": True,
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
                        metrics_session[f"recall_any@{k}"].append(ra)
                        metrics_session[f"ndcg@{k}"].append(nd)
                    per_type[entry["question_type"]]["r@5"].append(
                        metrics_session["recall_any@5"][-1]
                    )
                    per_type[entry["question_type"]]["r@10"].append(
                        metrics_session["recall_any@10"][-1]
                    )
                if completed % 25 == 0 or completed == len(data):
                    r5 = (sum(metrics_session["recall_any@5"]) /
                          max(1, len(metrics_session["recall_any@5"])))
                    elapsed = time.perf_counter() - started
                    eta = elapsed / completed * (len(data) - completed)
                    print(f"  [{completed:>3}/{len(data)}]  r@5={r5:.3f}  "
                          f"elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m")

    out_fh.close()
    wall = time.perf_counter() - started

    try:
        shutil.rmtree(db_root, ignore_errors=True)
    except Exception:
        pass

    print(f"\n{'=' * 60}")
    print(f"  RESULTS — Lethe v1 (session granularity)")
    print(f"{'=' * 60}")
    print(f"  Wall: {wall:.1f}s ({wall/len(data):.2f}s/q)\n")
    print("  SESSION-LEVEL:")
    for k in ks:
        ra = sum(metrics_session[f"recall_any@{k}"]) / len(metrics_session[f"recall_any@{k}"])
        nd = sum(metrics_session[f"ndcg@{k}"]) / len(metrics_session[f"ndcg@{k}"])
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
        print(f"\n  TIMING:")
        print(f"    inscribe per Q: avg={statistics.mean(ins):.0f}ms  "
              f"p95={sorted(ins)[int(len(ins)*0.95)]:.0f}ms")
        print(f"    recall   per Q: avg={statistics.mean(rec):.0f}ms  "
              f"p95={sorted(rec)[int(len(rec)*0.95)]:.0f}ms")

    skipped = [r for r in results if r.get("skipped")]
    if skipped:
        print(f"\n  skipped: {len(skipped)}")


if __name__ == "__main__":
    main()
