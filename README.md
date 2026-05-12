<p align="center">
  <img src="assets/logo.png" alt="Lethe" width="320" />
</p>

<p align="center">
  <b>Lethe is more than agent memory.</b><br/>
  <b>It's the operation everyone else avoids — forgetting.</b>
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
them. The password rotated three months ago, still suggested. The
customer who exercised right-to-deletion, still in the recommender. The
job title wrong since 2023 because the supersede never landed. The OTP
from last Tuesday, permanently embedded next to a real preference.
**Memory systems fail by overgrowth, not by attrition.** And no one is
benchmarking that side.

The Greeks had a name for the missing operation.

> **Lethe** (Λήθη) — one of the five rivers of Hades. Souls drank from
> it before reincarnation, leaving the former life behind.

Its opposite is **Mnemosyne**, memory. The Greek word for *truth* —
`ἀλήθεια` / **aletheia** — is `a-` (un-) + `lethe`. **Truth is
un-forgetting.**

> *Memory is what survives Lethe. Truth is what survives memory.*

## The model

Every fact has one number: `depth`. Every operation is a force on it.

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

No `weight`. No `alive` flag. No `superseded_at` column. One number,
one axis, one mental model.

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

agent.recall("Where does Alice work?", at=t_last_week)   # time-travel
agent.blame("Alice's job")                               # supersession chain
```

## Benchmarks

**LongMemEval-S** (500 questions, MemPalace's own methodology, same
`all-MiniLM-L6-v2`, zero API):

| System          | R@1       | R@5       | R@10      | Wall   |
|-----------------|-----------|-----------|-----------|--------|
| MemPalace (raw) | 80.6%     | 96.6%     | 98.2%     | 12 min |
| **Lethe v1**    | **85.4%** | **97.4%** | **99.0%** | 14 min |

**ForgetEval** (we propose; 1000 cases, 5 families, same embedder, no LLM):

| System        | super | decay | amnesia | purge | drift | Overall                       |
|---------------|------:|------:|--------:|------:|------:|------------------------------:|
| **Lethe v1**  | 100%  | 100%  | 98%     | 100%  | 99%   | **99.3%** (993 / 1000)        |
| Mem0 (2.0.2)  | 100%  | 100%  | 70%     | 75%   | 100%  | 88.8%                         |
| MemPalace     | N/A   | N/A   | N/A     | N/A   | N/A   | 0% (no forgetting primitives) |

ForgetEval is the axis no other framework benchmarks: *can you forget
on command?*  Five families probe supersession / decay / amnesia /
purge / drift.  Pass / fail is exact substring matching on top-k recall,
no LLM judge.  Full methodology, adapter contract, and reproduction
commands: **[docs/forgeteval.md](docs/forgeteval.md)**.

**Multilingual** (Lethe at scale=50, paraphrase-multilingual-MiniLM-L12-v2):

| Lang | super | decay | amnesia | purge | drift | Overall |
|------|------:|------:|--------:|------:|------:|--------:|
| en   | 100%  | 100%  | 98%     | 100%  | 99%   | 99.3%   |
| zh   |  90%  | 100%  |  36%    |  74%  |  90%  | 78%     |
| ja   |  94%  | 100%  |  44%    |  74%  |  72%  | 77%     |

The CJK drop is real architectural signal — multilingual MiniLM
clusters CJK tighter than English, and FTS5's default tokenizer splits
CJK per-character.  Both addressable, both documented in
[docs/forgeteval.md §10](docs/forgeteval.md).

Reproduce: `py bench/forgeteval/run.py --adapter {lethe|mem0|mempalace} --lang {en|zh|ja} --scale 200`

## Architecture

- **One physical axis: `depth`.**  Every state — pinned, surfaced,
  sinking, submerged, erased — is a numeric region.  No status flags.
- **Single SQLite file.**  Three sub-tables (`memory`, `memory_vec`,
  `memory_fts`) keyed by shared `rowid`; plus an append-only `event`
  log and a `supersession` edge table.  No external services.
- **Two retrieval primitives.**  `recall(query)` is RRF-blended vec +
  BM25; `recall(query, lexical=True)` is pure BM25.  Purge uses the
  second — deleting `alice@acme.io` is a *lexical* lookup by
  identifier, not a semantic search for "similar customers."
- **Verifiable forgetting.**  Every signed purge returns an
  **Ed25519-signed receipt** anchored to a Merkle root over the event
  log.  Tamper with any past event afterwards → receipt fails
  verification.  No other open-source memory framework can produce
  this proof because none of them keep the log to anchor to.
- **Time-travel built in.**  `recall(query, at=T)` reconstructs depth
  state at any past timestamp from the event log.

## Quickstart

```bash
pip install -e .[embed]
```

**Library** — see the code block above.

**CLI** — one subcommand per primitive:

```bash
lethe inscribe "Alice works at Anthropic."
lethe recall "Where does Alice work?"
lethe supersede 1 --new "Alice now at OpenAI."
lethe blame "Alice's job"
lethe consolidate
lethe ingest ~/notes                       # batch: *.md *.txt *.rst

# Verifiable purge
lethe keygen
lethe --db agent.db purge --signed 42      # emits receipt JSON
lethe verify-receipt receipt.json --db agent.db --db-check
```

DB defaults to `~/.lethe/agent.db`.  Pass `--json` on any subcommand for
machine-readable output.

**MCP** — ten tools exposed over stdio.  Add to Claude Desktop / Claude
Code / Cursor:

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

## Status

`v1.0.0-alpha`.  Core implemented and tested:

```
$ pytest tests
14 passed in 0.65s
```

Roadmap: CJK-aware FTS5 tokenizer · 1000+ adversarial ForgetEval
cases · cross-lingual paraphrase family · receipt-verification
benchmark family.

## License

MIT — see [LICENSE](LICENSE).
