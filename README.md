<p align="center">
  <img src="assets/logo.png" alt="Lethe" width="320" />
</p>

<p align="center">
  <b>Lethe is more than agent memory.</b><br/>
  <b>It's the first AI memory built to forget.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/tests-14%2F14-success?style=flat-square" alt="tests" />
  <img src="https://img.shields.io/badge/python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="python" />
  <img src="https://img.shields.io/badge/MCP-ready-7C3AED?style=flat-square" alt="MCP" />
  <img src="https://img.shields.io/badge/license-MIT-007EC6?style=flat-square" alt="license" />
</p>

---

**Every memory framework right now is racing the same direction.**

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

In the myth, Lethe is a river — surface, current, bed.  Everything in
the water has a depth.  A leaf floats; a stone sinks; some things are
weighed down enough to disappear.

Graph stores answer *what is connected to what.*  Vector stores answer
*what is semantically similar.*  Neither answers the question agent
memory actually faces: *how deep is this fact, right now?*

We built the simplest mental model that fits: every fact has one
number — `depth`.  Every operation is a force on it.

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

## Benchmarks

ForgetEval is downstream of the depth model — and the depth model is
downstream of ForgetEval.  We built one to test the other; the bench
then reshaped the architecture.  A failing `purge_gdpr` case in
particular forced `recall(lexical=True)` into the core as a first-class
primitive.  Both numbers below reflect that loop.

**LongMemEval-S** (500 questions, MemPalace's own methodology, same
`all-MiniLM-L6-v2`, zero API):

| System          | R@1       | R@5       | R@10      | Wall   |
|-----------------|-----------|-----------|-----------|--------|
| MemPalace (raw) | 80.6%     | 96.6%     | 98.2%     | 12 min |
| **Lethe v1**    | **85.4%** | **97.4%** | **99.0%** | 14 min |

Recall@K is half the picture — forgetting is the other half, which
is why we wrote ForgetEval.  But on the bench MemPalace built and
headlines with, a single `depth` axis lands higher at every K than a
palace of wings, rooms, and drawers — and the gap is **6× wider at
R@1** (+4.8 pp vs +0.8 pp at R@5).

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

Reproduce: `py bench/forgeteval/run.py --adapter {lethe|mem0|mempalace} --scale 200`

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
pip install -e ".[embed,crypto,mcp]"
```

**Library**:

```python
from lethe import Lethe

agent = Lethe("./agent.db")
mid = agent.inscribe("Alice works at Anthropic.")

agent.surrender(mid, mode="release")            # depth → 0
agent.surrender({"old": mid, "new": "Alice now at OpenAI."},
                mode="supersede")               # old sinks, new surfaces
agent.surrender(mid, mode="purge")              # erased from disk
agent.pin(mid)                                  # depth → +∞
```

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

**MCP** — eleven tools exposed over stdio (every core operation plus
signed-purge receipts).  Add to Claude Desktop / Claude Code / Cursor:

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

Roadmap (next):

- **Human-curated adversarial ForgetEval** — substring traps, prefix
  collisions, paraphrase chains.  Template-generated 1000-case is the
  floor, not the ceiling.
- **Receipt-verification benchmark family** — does the system produce
  auditable proof of deletion?  A new ForgetEval axis no other framework
  even attempts.
- **Adaptive consolidation policies** — `consolidate()` currently uses
  one fixed decay law; we want per-domain policies (financial records
  decay slower than chat memory).
- **Production-density distractor corpora** — replace synthetic
  office-trivia fillers with real long-form text (Wikipedia, code,
  emails) for a tougher recall environment.

## License

MIT — see [LICENSE](LICENSE).
