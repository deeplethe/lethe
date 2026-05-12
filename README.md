# Lethe

> *Memory is what survives Lethe. Truth is what survives memory.*

A river for AI memory. **Forgetting is the architecture** — not the absence of it.

```python
from lethe import Lethe

agent = Lethe("./agent.db")

# Inscribe puts a fact on the surface (depth = 1.0)
mid = agent.inscribe("Alice works at Anthropic.")

# Recall retrieves what's still above water
agent.recall("Where does Alice work?")

# Surrender lets a memory sink — four forces:
agent.surrender(mid, mode="decay")           # depth *= 0.5
agent.surrender(mid, mode="release")         # depth = 0  (default)
agent.surrender({"old": mid, "new": "Alice now at OpenAI."},
                mode="supersede")            # old sinks, new floats
agent.surrender(mid, mode="purge")           # delete from disk (compliance)

# Pin keeps a fact above the surface (depth = +∞), immune to gravity
agent.pin(mid)

# Consolidate applies gravity — sleep consolidates memory
agent.consolidate()

# Time-travel: what did the agent believe at moment T?
agent.recall("Where does Alice work?", at=t_last_week)

# Blame: the supersession history of any belief
agent.blame("Alice's job")
```

**One class, one mental model.** Every operation is a force on `depth`.

## Benchmarks

LongMemEval-S (500 questions, MemPalace's own evaluation methodology, same
all-MiniLM-L6-v2 embedder, 0 API calls):

| System                | R@1   | R@3   | R@5   | R@10  | Wall    |
|-----------------------|-------|-------|-------|-------|---------|
| MemPalace (raw)       | 80.6% | 92.6% | 96.6% | 98.2% | 12 min  |
| **Lethe v1**          | **85.4%** | **95.2%** | **97.4%** | **99.0%** | **14 min**  |

Per-question-type R@5:

| Type                       | MemPalace | Lethe v1 | Δ      |
|----------------------------|-----------|----------|--------|
| knowledge-update           | 100.0%    | 100.0%   | tie    |
| multi-session              |  99.2%    |  98.5%   | -0.7   |
| single-session-assistant   |  96.4%    | **100.0%** | **+3.6** |
| single-session-preference  |  96.7%    |  96.7%   | tie    |
| single-session-user        |  91.4%    | **94.3%**  | **+2.9** |
| temporal-reasoning         |  94.7%    | **95.5%**  | **+0.8** |

Reproduce: `py bench/longmemeval_v1_subdoc.py bench/longmemeval_s.json`

## ForgetEval

LongMemEval measures one half of memory: **does it stay when you need it?**
The other half is the one nobody benchmarks: **does it leave when you ask?**

ForgetEval is our attempt at the second axis. 250 generated tests across
five families, each one structurally easy for a system that forgets and
structurally impossible for one that doesn't:

| Family         | Operation needed | What it tests |
|----------------|------------------|---------------|
| supersession   | `supersede`      | New fact wins; old fact does not surface |
| decay          | `release`        | A released fact stays out of top-k       |
| amnesia        | `release`        | Forget one subject; siblings survive     |
| purge          | `purge`          | Hard-delete by identifier (GDPR / PHI / keys) — exact, not fuzzy |
| drift          | `supersede` × N  | Chain of updates; the latest belief wins |

Each case mixes the target facts with 3–5 unrelated distractor facts.
Deterministic via seed, no LLM, no API.

| System        | super  | decay  | amnesia | purge  | drift  | Overall      | Wall   |
|---------------|-------:|-------:|--------:|-------:|-------:|-------------:|-------:|
| **Lethe v1**  | 50/50  | 50/50  | 50/50   | 50/50  | 50/50  | **250/250 · 100%** | **42 s** |
| Mem0 (2.0.2)  | 50/50  | 50/50  | 41/50   | 36/50  | 50/50  | 227/250 · 91% | 151 s |
| MemPalace     | N/A    | N/A    | N/A     | N/A    | N/A    | 0/250 (no forgetting primitives) | 121 s |

MemPalace's zeros aren't a benchmark failure — they are an honest report
that the library was built without supersede, release, or purge. ForgetEval
makes the capability gap visible.

Reproduce: `py bench/forgeteval/run.py --adapter {lethe|mem0|mempalace} --scale 50`

## What we did differently

1. **Sub-document indexing at dialog-pair granularity** — each user+assistant
   exchange becomes one memory, instead of MemPalace's "concatenate all user
   turns of a session." Closer to how an agent actually experiences a turn.
2. **No query-type detection, no BM25 reweighting heuristics, no regex.**
   The recall path is pure cosine over per-pair embeddings, aggregated to
   session level by first-occurrence rank.
3. **Indexes both user and assistant content** — because a real agent has to
   answer "what did you recommend last time?" and MemPalace's user-only
   default cannot. (Strict apples-to-apples with their user-only baseline:
   Lethe still wins R@5 by +1.0 pp.)
4. **Two retrieval primitives, one knob.** `recall(...)` is RRF-blended
   vec + BM25; `recall(..., lexical=True)` is pure BM25. The library uses
   the second for `purge` — deleting `alice@acme.io` is a *lexical* lookup
   by identifier, not a semantic search for "similar customers." ForgetEval
   surfaced this distinction; we made it a first-class API.

## The myth

Lethe is one of the five rivers of Hades. Plato describes the rite in the
*Republic* — souls drink from Lethe before reincarnation, leaving the
former life behind. Its opposite is **Mnemosyne**, memory.

The Greek word for truth is **ἀλήθεια / aletheia** — literally *a-* +
*lethe* = **un-forgetting**. Heidegger built a metaphysics on the etymology.

In this library, those three words are landmarks on a single number — `depth`:

```
depth     state                          who put it there
─────────────────────────────────────────────────────────────────────
+∞        pinned above the surface       you (.pin) / consolidate promote
> 1.0     promoted, deep-survival        consolidate (after long use)
= 1.0     just inscribed, on surface     .inscribe
∈ (0, 1)  sinking under gravity          .consolidate
= 0       submerged, present but mute    .surrender(mode="release")
< 0       erased; only the event log remembers   .surrender(mode="purge")
─────────────────────────────────────────────────────────────────────
```

Every operation is a force on `depth`. There is no `weight`, no `alive`,
no `layer`, no `superseded_at` flag — **one number, one axis**.

## Status

`v1.0.0-alpha`. Depth-physics core implemented and tested.

```bash
$ pytest tests
14 passed in 0.65s
```

Next:
- **Cryptographic receipts** — `.surrender(mode="purge")` will return a signed Merkle proof of erasure for GDPR-grade compliance.
- **CLI** — `lethe blame "user's job"` for time-travel introspection.
- **ForgetEval expansion** — 250 templated cases is a start; we want 1000+ adversarial cases and human-curated edge cases.

## License

TBD.
