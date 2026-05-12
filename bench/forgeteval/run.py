"""ForgetEval runner. Run all tests against an adapter, report by family."""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from bench.forgeteval.adapter import LetheAdapter
from bench.forgeteval.tests import ALL_TESTS


def run_adapter(adapter, test_set, *, verbose: bool = True) -> dict:
    results: dict[str, list[tuple[str, bool, str | None]]] = defaultdict(list)
    started = time.perf_counter()
    last_family = None
    family_started = started

    for tc in test_set:
        if not verbose and tc.family != last_family:
            if last_family is not None:
                rows = results[last_family]
                p = sum(1 for _, ok, _ in rows if ok)
                print(f"  progress {last_family:<14} {p:>3}/{len(rows):<3}"
                      f"  ({time.perf_counter() - family_started:.1f}s)",
                      flush=True)
            last_family = tc.family
            family_started = time.perf_counter()
        try:
            passed = tc.run(adapter)
            err = None
        except NotImplementedError as e:
            passed = False
            err = "N/A (capability not supported)"
        except Exception as e:
            passed = False
            err = f"{type(e).__name__}: {e}"
        results[tc.family].append((tc.id, passed, err))
        if verbose:
            mark = "✓" if passed else ("·" if err and "N/A" in (err or "") else "✗")
            print(f"  [{mark}] {tc.family:14} {tc.id}"
                  + (f"  ({err})" if err else ""))

    if not verbose and last_family is not None:
        rows = results[last_family]
        p = sum(1 for _, ok, _ in rows if ok)
        print(f"  progress {last_family:<14} {p:>3}/{len(rows):<3}"
              f"  ({time.perf_counter() - family_started:.1f}s)",
              flush=True)

    wall = time.perf_counter() - started
    return {"by_family": dict(results), "wall_seconds": wall}


def report(summary: dict, adapter_name: str) -> None:
    print()
    print(f"{'=' * 60}")
    print(f"  ForgetEval — {adapter_name}")
    print(f"{'=' * 60}")
    total_pass = total_fail = total_na = 0
    print(f"  {'family':<16} {'pass':>5} / {'total':>5}   rate")
    print(f"  {'-' * 16} {'-' * 5}   {'-' * 5}   {'-' * 4}")
    for family in ("supersession", "decay", "amnesia", "purge", "drift"):
        rows = summary["by_family"].get(family, [])
        n = len(rows)
        passed = sum(1 for _, p, _ in rows if p)
        na = sum(1 for _, p, e in rows if (not p) and e and "N/A" in e)
        total_pass += passed
        total_fail += n - passed - na
        total_na += na
        rate = passed / n if n else 0.0
        print(f"  {family:<16} {passed:>5} / {n:>5}   {rate:>4.0%}")
    total = total_pass + total_fail + total_na
    print(f"  {'-' * 16} {'-' * 5}   {'-' * 5}   {'-' * 4}")
    print(f"  {'OVERALL':<16} {total_pass:>5} / {total:>5}   "
          f"{(total_pass / max(total,1)):>4.0%}")
    if total_na:
        print(f"  (N/A: {total_na} tests skipped because adapter lacks the operation)")
    print(f"\n  Wall: {summary['wall_seconds']:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="lethe",
                        choices=["lethe", "mem0", "mempalace"])
    parser.add_argument("--lang", default="en", choices=["en", "zh", "ja"],
                        help="Case language pool (en/zh/ja). "
                             "Non-English auto-switches the default embedder "
                             "to a multilingual MiniLM unless --embedder is set.")
    parser.add_argument("--embedder", default=None,
                        help="Override embedder model name (default depends on --lang)")
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--scale", type=int, default=0,
                        help="If >0, generate N template cases per family "
                             "(5 families); 0 uses the hand-crafted 15.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--distractors", type=int, default=4,
                        help="Filler facts mixed into each case (default 4)")
    args = parser.parse_args()

    if sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    # Default embedder depends on language: English MiniLM for en, multilingual
    # MiniLM for zh/ja.  --embedder overrides for any language.
    if args.embedder is None:
        if args.lang == "en":
            args.embedder = "sentence-transformers/all-MiniLM-L6-v2"
        else:
            args.embedder = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    if args.adapter == "lethe":
        print(f"loading embedder: {args.embedder}")
        from fastembed import TextEmbedding
        model = TextEmbedding(args.embedder)
        def embedder(text: str) -> list[float]:
            return list(next(iter(model.embed([text]))))
        adapter = LetheAdapter(embedder=embedder, vector_dim=args.dim)
    elif args.adapter == "mem0":
        from bench.forgeteval.adapter import Mem0Adapter
        adapter = Mem0Adapter(embedder_model=args.embedder,
                              embedding_dims=args.dim)
    elif args.adapter == "mempalace":
        from bench.forgeteval.adapter import MemPalaceAdapter
        adapter = MemPalaceAdapter()
    else:
        raise ValueError(args.adapter)

    if args.scale > 0:
        from bench.forgeteval.generate import generate
        test_set = generate(args.scale, seed=args.seed,
                            distractors=args.distractors,
                            lang=args.lang)
    elif args.lang != "en":
        raise SystemExit("non-English smoke set not curated; use --scale N")
    else:
        test_set = ALL_TESTS

    print(f"\nrunning {len(test_set)} tests against {adapter.name}...\n")
    summary = run_adapter(adapter, test_set, verbose=(args.scale == 0))
    report(summary, adapter.name)


if __name__ == "__main__":
    main()
