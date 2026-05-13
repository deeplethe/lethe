# Contributing to Lethe

Lethe is small enough that one person can read the whole codebase in
an afternoon.  Please do that before sending a large PR — you'll
discover whether the change you have in mind fits the architecture
faster than reading this document.

## Quick start

```bash
git clone https://github.com/deeplethe/lethe
cd lethe
pip install -e ".[dev,embed,crypto,mcp]"

pytest tests           # 14 tests, ~0.7s
ruff check lethe tests # lint
```

If both pass, you're set up.

## What we want

- **Bug reports** with a minimal reproducer.
- **ForgetEval adapters** for other memory frameworks (Letta, Zep,
  HippoRAG, A-MEM, Cognee, ...).  See *Adding a ForgetEval adapter*.
- **New ForgetEval sub-templates** that probe real edge cases —
  prefix collisions, multilingual paraphrase chains, long-form facts
  with one central claim change.
- **CJK-aware FTS5 tokenizer** for the lexical-purge path
  (`docs/forgeteval.md §10`).
- **Documentation fixes** of any kind.

## What we'd push back on

- **New magic constants.**  Every numeric tunable in the codebase has
  a docstring explaining why it has its value.  If your PR introduces
  a literal like `0.5` or `60.0` without a name and a comment, expect
  a request to extract it into a named, justified parameter.
- **Heuristic-laden query understanding.**  No regex query routers,
  no query-type classifiers, no "if `:` in query then ...".  Recall
  is pure cosine + pure BM25 + RRF; routing is the agent's job.
- **LLM-judge benchmarks.**  ForgetEval is deliberately LLM-free for
  reproducibility.  PRs that introduce LLM-graded scoring will need
  to argue why before they go in.
- **Pluggable backends to non-SQLite stores.**  We will get here
  eventually, but only after we have real adoption pressure for it.
  PRs that abstract the storage layer prematurely tend to ossify
  decisions we're not ready to make.

## Architecture principles

Lethe's API is small because the model is small.  Three things we keep
under discipline:

1. **One axis: `depth`.**  No status flags, no `alive` column, no
   `superseded_at` field.  If you find yourself wanting a new mutable
   piece of state, ask first whether it collapses into `depth`.
2. **No magic constants.**  Score thresholds, decay rates, RRF
   constants — every literal has a name and a docstring.
3. **No regex query understanding.**  The memory store does not
   inspect query semantics; it ranks against an embedding and a BM25
   index.  Semantic dispatch is the agent's responsibility.

## Adding a ForgetEval adapter

If you maintain (or use) another memory framework, write an adapter
that implements the Protocol in `bench/forgeteval/adapter.py`:

```python
class MyAdapter:
    name = "myframework"

    def reset(self) -> None: ...
    def inscribe(self, text: str) -> int | str: ...
    def recall_texts(self, query: str, k: int = 5) -> list[str]: ...

    # Optional — raise NotImplementedError to be scored as N/A.
    def supersede(self, old_query: str, new_text: str) -> None: ...
    def release(self, query: str) -> int: ...
    def purge(self, query: str) -> int: ...
```

Three mandatory methods, three optional.  We will not penalize a
framework for missing primitives — the absence is reported honestly
as a capability gap.

Submit a PR adding your adapter under `bench/forgeteval/adapter.py`
plus the runner wiring; we'll cross-publish results in the README
benchmark table.

## Adding ForgetEval sub-templates

See `docs/forgeteval.md §5` for the existing taxonomy.  New sub-templates
should:

- Probe a specific structural failure mode (substring trap, prefix
  collision, multi-step paraphrase, long-form fact, etc.).
- Be deterministic from `(rng, distractors)` — no LLM, no `time.time()`.
- Have unambiguous pass / fail via substring matching on top-k recall.

Code lives in `bench/forgeteval/generate.py`
(or `lang_zh.py` / `lang_ja.py` for non-English).  Update the
sub-template table in the docs in the same PR.

## Pull request guidance

- **One logical change per PR.**  Two unrelated improvements should
  be two PRs.
- **`pytest tests` must pass.**  Add tests for new behavior.
- **`ruff check lethe tests` must pass.**  CI enforces this.
- **Commit messages: explain WHY in the body, not just WHAT.**  The
  first line is a short subject; the body answers "what was wrong
  before this commit, and why does this fix it?"
- **AI-assisted contributions are welcome and should be disclosed.**
  Append `Co-Authored-By: <model name>` to commits where an AI
  drafted substantive code or prose, the way we do in this repo.
  Don't try to hide it — we'd rather know.

## Reporting bugs

Open an issue with:

- Lethe version (commit SHA, or PyPI version)
- Python version and OS
- Minimal reproducer (ideally 3-5 lines)
- What you expected vs what happened

For ForgetEval regressions specifically, include the failing case ID
(e.g. `purge_gdpr__008`) and the seed — the bench is deterministic,
so a case ID + seed is enough for us to reproduce locally.

## Code of conduct

Be excellent to each other.  We don't have a formal CoC document yet
because the project is small enough that common sense suffices; if it
stops sufficing, we'll adopt one.

## License

By contributing, you agree your contributions are licensed under the
MIT License — see [LICENSE](LICENSE).
