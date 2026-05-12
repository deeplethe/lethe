# Lethe

Every memory framework right now is racing the same direction.

Mem0 promises perfect recall. MemPalace promises verbatim retention.
Letta hands an LLM the whole context and asks it to manage itself. The
benchmarks they all compete on — recall@K, hit-rate, MRR — measure one
thing: *how rarely does your agent lose a fact?*

But agents in production don't die of losing facts. They die of keeping
them. The password rotated three months ago and still suggested. The
customer who exercised right-to-deletion, still in the recommender. The
job title wrong since 2023 because the supersede never landed. The OTP
from last Tuesday, permanently embedded next to a real preference.
Memory systems fail by **overgrowth**, not by attrition. And no one is
benchmarking that side.

The Greeks had a name for the missing operation.

> **Lethe** (Λήθη) — one of the five rivers of Hades. Souls drank from
> it before reincarnation, leaving the former life behind.

Its opposite is **Mnemosyne**, memory. The Greek word for *truth* —
`ἀλήθεια` / **aletheia** — is `a-` (un-) + `lethe`. **Truth is
un-forgetting.** Heidegger built a metaphysics on that etymology.

> *Memory is what survives Lethe. Truth is what survives memory.*

## The model

Every fact has one number: `depth`.

```
depth     state                          how it got there
─────────────────────────────────────────────────────────────
+∞        pinned, immune to gravity      .pin()
= 1.0     just inscribed, on surface     .inscribe()
∈ (0, 1)  sinking under gravity          .consolidate()
= 0       submerged, present but mute    .surrender(mode="release")
< 0       erased from disk               .surrender(mode="purge")
─────────────────────────────────────────────────────────────
```

No `weight`, no `alive` flag, no `superseded_at` column. One number, one
axis, one mental model. Every operation is a force on `depth`.

```python
from lethe import Lethe

agent = Lethe("./agent.db")

mid = agent.inscribe("Alice works at Anthropic.")
agent.recall("Where does Alice work?")

# Four forces sink a memory
agent.surrender(mid, mode="decay")                       # depth *= 0.5
agent.surrender(mid, mode="release")                     # depth = 0
agent.surrender({"old": mid, "new": "Alice now at OpenAI."},
                mode="supersede")                        # old sinks, new floats
agent.surrender(mid, mode="purge")                       # delete from disk

agent.pin(mid)              # depth = +∞
agent.consolidate()         # apply gravity to everything else

# Time-travel: what did the agent believe at time T?
agent.recall("Where does Alice work?", at=t_last_week)

# Blame: the supersession chain for any belief
agent.blame("Alice's job")
```

## Benchmarks

### LongMemEval-S — retrieval quality (the conventional axis)

500 questions, MemPalace's own evaluation methodology, same
`all-MiniLM-L6-v2` embedder, zero API calls.

| System          | R@1       | R@3       | R@5       | R@10      | Wall    |
|-----------------|-----------|-----------|-----------|-----------|---------|
| MemPalace (raw) | 80.6%     | 92.6%     | 96.6%     | 98.2%     | 12 min  |
| **Lethe v1**    | **85.4%** | **95.2%** | **97.4%** | **99.0%** | 14 min  |

Per-question-type R@5:

| Type                       | MemPalace | Lethe v1   | Δ        |
|----------------------------|-----------|------------|----------|
| knowledge-update           | 100.0%    | 100.0%     | tie      |
| multi-session              |  99.2%    |  98.5%     | -0.7     |
| single-session-assistant   |  96.4%    | **100.0%** | **+3.6** |
| single-session-preference  |  96.7%    |  96.7%     | tie      |
| single-session-user        |  91.4%    |  **94.3%** | **+2.9** |
| temporal-reasoning         |  94.7%    |  **95.5%** | **+0.8** |

Reproduce: `py bench/longmemeval_v1_subdoc.py bench/longmemeval_s.json`

### ForgetEval — the axis nobody benchmarks

250 generated tests across five families. Each one is structurally easy
for a system that forgets and structurally impossible for one that doesn't.

| Family       | Operation       | What it tests                                                  |
|--------------|-----------------|----------------------------------------------------------------|
| supersession | `supersede`     | New fact wins; old fact does not surface                       |
| decay        | `release`       | A released fact stays out of top-k                             |
| amnesia      | `release`       | Forget one subject; siblings survive                           |
| purge        | `purge`         | Hard-delete by identifier (GDPR / PHI / keys) — exact, not fuzzy |
| drift        | `supersede` × N | Chain of updates; the latest belief wins                       |

Each case mixes the target facts with 3–5 unrelated distractor facts.
Deterministic via seed, no LLM, no API.

| System        | super | decay | amnesia | purge | drift | Overall                          | Wall   |
|---------------|------:|------:|--------:|------:|------:|---------------------------------:|-------:|
| **Lethe v1**  | 50/50 | 50/50 | 50/50   | 50/50 | 50/50 | **250 / 250 · 100%**             | 42 s   |
| Mem0 (2.0.2)  | 50/50 | 50/50 | 41/50   | 36/50 | 50/50 | 227 / 250 · 91%                  | 151 s  |
| MemPalace     | N/A   | N/A   | N/A     | N/A   | N/A   | 0 / 250 (no forgetting primitives) | 121 s  |

MemPalace's zeros are not a benchmark failure. They are an honest report
that the library was built without `supersede`, `release`, or `purge`.
ForgetEval makes the capability gap visible.

Reproduce: `py bench/forgeteval/run.py --adapter {lethe|mem0|mempalace} --scale 50`

## Architecture

- **Sub-document indexing at dialog-pair granularity.** Each
  user+assistant exchange is one memory, not "concatenate all user turns
  of a session." Closer to how an agent experiences a turn.
- **No query-type detection, no BM25 reweighting heuristics, no regex.**
  Recall is pure cosine over per-pair embeddings, aggregated to session
  level by first-occurrence rank.
- **Indexes both user and assistant content.** A real agent has to
  answer *"what did you recommend last time?"* — user-only indexes can't.
- **Two retrieval primitives, one knob.** `recall(...)` is RRF-blended
  vec + BM25; `recall(..., lexical=True)` is pure BM25. The library
  uses the second for `purge` — deleting `alice@acme.io` is a *lexical*
  lookup by identifier, not a semantic search for "similar customers."
  ForgetEval surfaced this distinction; we made it a first-class API.

## Status

`v1.0.0-alpha`. Depth-physics core implemented and tested.

```
$ pytest tests
14 passed in 0.65s
```

Next:
- **Cryptographic receipts.** `surrender(mode="purge")` will return a
  signed Merkle proof of erasure for GDPR-grade compliance.
- **CLI.** `lethe blame "user's job"` for time-travel introspection.
- **ForgetEval expansion.** 250 templated cases is a start; 1000+
  adversarial cases and human-curated edge cases come next.

## License

TBD.
