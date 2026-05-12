<p align="center">
  <img src="assets/logo.png" alt="Lethe" width="320" />
</p>

<p align="center">
  Local-first AI memory. Forgetting as a first-class operation —
  <b>97.4% R@5</b> on LongMemEval, <b>99.3%</b> on ForgetEval (1000 cases), zero API calls.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-1.0.0--alpha-blue" alt="version" />
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="python" />
  <img src="https://img.shields.io/badge/license-MIT-green" alt="license" />
</p>

---

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

We propose **ForgetEval**, a benchmark methodology for the one operation
every other AI memory framework treats as failure: *forgetting on
command*.  Five families of test cases — supersession, decay, amnesia,
purge, drift — each probing one structural property a memory system
must exhibit to be safe in production.

| Family       | Operation       | What it tests                                                    |
|--------------|-----------------|------------------------------------------------------------------|
| supersession | `supersede`     | New fact wins; old fact does not surface                         |
| decay        | `release`       | A released fact stays out of top-k                               |
| amnesia      | `release`       | Forget one subject; siblings survive                             |
| purge        | `purge`         | Hard-delete by identifier (GDPR / PHI / keys) — exact, not fuzzy |
| drift        | `supersede` × N | Chain of updates; the latest belief wins                         |

Every case is short, deterministic, and reproducible without an LLM:
inscribe a small set of facts (mixed with unrelated distractors), apply
a mutation, then check whether the right thing surfaces and the wrong
thing doesn't.  Pass / fail is exact substring matching over top-k
recall — no judge model, no ambiguity.

See **[docs/forgeteval.md](docs/forgeteval.md)** for the full
methodology: case anatomy, generation protocol, adapter contract,
scoring rules, and how to evaluate a new memory system.

**1000 cases** (200 per family, four sub-templates rotating, 4 distractors each):

| System        | super     | decay     | amnesia   | purge     | drift     | Overall                              | Wall    |
|---------------|----------:|----------:|----------:|----------:|----------:|-------------------------------------:|--------:|
| **Lethe v1**  | 200 / 200 | 200 / 200 | 195 / 200 | 200 / 200 | 198 / 200 | **993 / 1000 · 99.3%**               | 13 min  |
| Mem0 (2.0.2)  | 200 / 200 | 200 / 200 | 139 / 200 |  150 / 200 | 199 / 200 | 888 / 1000 · 88.8%                  | 27 min  |
| MemPalace     | N/A       | N/A       | N/A       | N/A       | N/A       | 0 / 1000 (no forgetting primitives)  | 18 min  |

MemPalace's zeros are not a benchmark failure. They are an honest report
that the library was built without `supersede`, `release`, or `purge`.
ForgetEval makes the capability gap visible.

Where the systems diverge most: **amnesia** (forget one entity, peers
survive) and **purge** (delete by identifier, near-paraphrase siblings
must NOT be touched). Mem0 drops to 70% and 75% there; Lethe holds 98%
and 100% because release uses adaptive-gap clustering and purge uses
the dedicated `recall(lexical=True)` primitive.

Reproduce: `py bench/forgeteval/run.py --adapter {lethe|mem0|mempalace} --scale 200`

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

## Verifiable forgetting

`purge` is the only delete that hits disk, so it's also the only one
compliance cares about. Lethe issues an **Ed25519-signed receipt** for
every signed purge — the receipt commits to the *entire event log* via
a Merkle root, so any subsequent tampering with the audit trail
invalidates verification.

```bash
lethe keygen                              # one-time: generate signing key
lethe --db agent.db purge --signed 42     # delete id 42, emit receipt JSON
lethe verify-receipt receipt.json         # signature only (no DB needed)
lethe verify-receipt receipt.json \
      --db agent.db --db-check            # also recompute Merkle root
```

Without database access, a verifier can confirm *the signer made this
claim at time T*. With the database, the verifier additionally confirms
*the event log has not been edited since*. No other open-source memory
framework can produce this guarantee — it falls out of having an
append-only event log in the first place.

## CLI

```bash
pip install -e .[embed]

lethe inscribe "Alice works at Anthropic."
lethe recall "Where does Alice work?"
lethe supersede 1 --new "Alice now at OpenAI."
lethe blame "Alice's job"
lethe consolidate                    # apply Hypnos gravity
lethe log --kind supersede           # event log filtered by kind

# Batch-inscribe a directory.  Paragraph-level chunks, verbatim.
lethe ingest ~/notes                       # default: *.md *.txt *.rst
lethe ingest ./docs --glob '*.md' --batch 512
```

DB defaults to `~/.lethe/agent.db`. Override with `--db PATH` or
`$LETHE_DB`. Any subcommand accepts `--json` for machine-readable output.

## MCP

Lethe ships an MCP server. Add it to Claude Desktop / Claude Code / Cursor
with:

```json
{
  "mcpServers": {
    "lethe": {
      "command": "python",
      "args": ["-m", "lethe.mcp_server"],
      "env": {"LETHE_DB": "/absolute/path/to/agent.db"}
    }
  }
}
```

Ten tools are exposed (`inscribe`, `recall`, `release`, `purge`,
`supersede`, `pin`, `unpin`, `consolidate`, `blame`, `log`) — every core
operation, no glue code required.

## Status

`v1.0.0-alpha`. Depth-physics core implemented and tested.

```
$ pytest tests
14 passed in 0.65s
```

Next:
- **ForgetEval expansion.** 250 templated cases is a start; 1000+
  adversarial cases (long-form facts, multi-language paraphrase, heavy
  distractor pollution) come next.

## License

MIT — see [LICENSE](LICENSE).
