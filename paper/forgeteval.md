# ForgetEval: A Benchmark for Memory Deletion in AI Agent Systems

> *Draft.  Targeted at arXiv preprint; later peer-reviewed venue
> (NeurIPS D&B / ACL benchmarks workshop).*

## Abstract

Memory frameworks for AI agents are routinely benchmarked on
retrieval recall — *how well can the system find a fact when asked?*
Existing benchmarks (LongMemEval, MTEB-Memory, RAG-bench) all measure
this axis exclusively.  But agents in production fail in the inverse
direction: rotated credentials still suggested, GDPR-deleted users
still in the recommender, stale beliefs persisting through dozens of
contradicting updates.  No benchmark exists for the operation that
fixes this — *forgetting on command*.

We propose **ForgetEval**, a benchmark methodology that measures
five structural properties of forgetting: **supersession**, **decay**,
**amnesia**, **purge**, and **drift**.  Each is operationalized as a
template-generated test case with deterministic pass/fail criteria
that require no LLM judge.  We release a 1000-case English benchmark
suite, 250-case multilingual extensions (Chinese, Japanese), and a
six-method `Adapter` protocol that lets any memory system enter the
evaluation in roughly 130 lines of glue code.

We further introduce **Lethe**, a reference implementation built
around a single physical axis (`depth ∈ ℝ`) with four force-on-depth
operations (decay / release / supersede / purge).  On ForgetEval,
Lethe achieves **99.3% (993/1000)**; Mem0 (v2.0.2) achieves 88.8%,
with characteristic weakness on amnesia (70%) and purge (75%);
MemPalace achieves 0% because the required primitives are not
implemented.  These gaps map directly to production failure modes
documented in the literature on stateful agent systems.

---

## 1. Introduction

Every memory framework for AI agents is racing the same direction.
Mem0 [@mem0] promises perfect recall through automatic
update/delete/no-op routing.  MemPalace [@mempalace] promises
verbatim retention as a feature, achieving 96.6% R@5 on LongMemEval-S
without any compression.  Letta [@letta] delegates memory management
to the LLM itself through hierarchical context-engineering primitives.
Zep [@zep], HippoRAG [@hipporag], A-MEM [@amem] and Cognee [@cognee]
each propose graph-based or episodic-memory variants.  All compete on
the same battery of evaluations: recall@K, hit-rate, MRR.  The unstated
shared assumption is that the failure mode worth engineering against
is *losing* a fact.

In production deployments, the failure mode is the opposite.

- A password that the user rotated three months ago is still
  suggested to the LLM as a candidate credential.
- A customer who exercised their right-to-be-forgotten under GDPR
  Article 17 remains in the recommendation candidate pool because the
  embedding was never deleted.
- A user's job title, updated through six conversational turns over
  two years, surfaces in three contradictory versions depending on
  which session the agent is restored from.
- A one-time verification code from a session two weeks ago lives
  permanently next to the user's stable preferences, drifting through
  recall results.

These failures are not the absence of recall — recall worked, the
fact was retrieved.  They are the failure of *forgetting*: the
inability to direct the system to no longer surface specific content.
The agent retrieved the right thing too well, when the application
needed it to retrieve nothing.

This paper argues:

1. The forgetting axis is **measurable** with the same rigor as
   recall.  We propose ForgetEval, a methodology that operationalizes
   five forgetting operations with deterministic substring-match
   scoring over top-k recall.  No LLM judge is required, making
   the protocol fully reproducible across runs and across systems.
2. Forgetting is **structural**, not residual.  Whether a memory
   framework can pass a test in the supersession or purge families
   depends on whether the system *has* a primitive for that
   operation, not how well it implements one.  ForgetEval explicitly
   distinguishes between "implemented and failed" and "primitive not
   provided" through the adapter contract.
3. The current open-source landscape has a **measurable capability
   gap** on this axis.  Of three production-grade systems we
   evaluate, none score above 90% across all five families; one
   scores zero because the primitives are not provided at all.

We accompany the methodology with **Lethe**, a reference
implementation organized around a single numeric axis (`depth`) with
forgetting operations as first-class primitives.  Lethe is not the
central contribution of this paper — the benchmark is.  But Lethe
serves to demonstrate that the metric is satiable: a 99.3% score
across 1000 cases is achievable with a careful architecture, which
in turn validates that ForgetEval failure modes for other systems
reflect architectural choices rather than benchmark over-difficulty.

The rest of the paper is organized as follows.  §2 surveys related
work in memory-system benchmarks and forgetting in episodic memory
research.  §3 specifies the ForgetEval methodology in full: the five
families, case anatomy, generation protocol, and scoring rules.  §4
documents the `Adapter` Protocol that lets a new memory system enter
the evaluation.  §5 presents Lethe as a reference implementation:
the depth-axis model, surrender primitives, and the cryptographic
purge receipts that come for free from an append-only event log.  §6
reports experimental results across three systems (Lethe, Mem0,
MemPalace) on the 1000-case English suite and 250-case multilingual
extensions.  §7 translates each ForgetEval score into a concrete
production failure mode.  §8 discusses limitations honestly: English
emphasis, template-generated cases, synthetic distractors.  §9
concludes.

The complete benchmark, adapter contract, and Lethe reference
implementation are released under MIT at
[github.com/deeplethe/lethe](https://github.com/deeplethe/lethe).
We invite teams maintaining other memory frameworks to submit
adapters; we will cross-publish results in the public leaderboard.

---

## 2. Related Work

*(to be written)*

Memory-system benchmarks today fall into three clusters:
**conversational recall** (LongMemEval, RealMemBench, ChatBench),
**document-grounded retrieval** (MTEB, BEIR, MS-MARCO), and
**graph-structured episodic recall** (KILT, HippoRAG's own
evaluations).  None measure deletion.

Forgetting in human episodic memory has a long psychology literature
(Ebbinghaus, Bjork's directed-forgetting paradigm, Anderson &
Spellman 1995 on retrieval-induced forgetting), and a more recent
machine-learning literature (machine unlearning, GDPR-driven model
deletion, differential privacy as forgetting).  ForgetEval is
positioned in neither of these — we are not measuring whether an
ML model forgets, we are measuring whether an *agent memory system*
exposes the right primitives.

*[expand: ~3-4 paragraphs naming systems and positioning ForgetEval
relative to each cluster]*

---

## 3. The ForgetEval Methodology

*(largely portable from `docs/forgeteval.md` — port and adapt for
paper voice)*

### 3.1 Five families

| Family       | Operation       | What it tests                                                  |
|--------------|-----------------|----------------------------------------------------------------|
| supersession | `supersede`     | New fact wins; old fact does not surface                       |
| decay        | `release`       | A released fact stays out of top-k                             |
| amnesia      | `release`       | Forget one subject; siblings survive                           |
| purge        | `purge`         | Hard-delete by identifier (GDPR / PHI / keys) — exact match    |
| drift        | `supersede` × N | Chain of updates; the latest belief wins                       |

### 3.2 Case anatomy

*[port from docs §3]*

### 3.3 Distractors

*[port from docs §4]*

### 3.4 Sub-templates

*[port from docs §5]*

### 3.5 Generation protocol

*[port from docs §6]*

### 3.6 Scoring

*[port from docs §7]*

---

## 4. Adapter Contract

*[port from docs §9]*

```python
class Adapter(Protocol):
    name: str

    def reset(self) -> None: ...
    def inscribe(self, text: str) -> int | str: ...
    def recall_texts(self, query: str, k: int = 5) -> list[str]: ...

    # Optional — raise NotImplementedError to be scored as N/A.
    def supersede(self, old_query: str, new_text: str) -> None: ...
    def release(self, query: str) -> int: ...
    def purge(self, query: str) -> int: ...
```

*[expand: design rationale, why six methods, why
NotImplementedError is honest scoring]*

---

## 5. Reference Implementation: Lethe

*(condense the README's Architecture + The model sections)*

### 5.1 One physical axis: depth

### 5.2 The four forces

### 5.3 Two retrieval primitives (RRF hybrid + pure BM25)

### 5.4 Verifiable forgetting: Ed25519-signed purge receipts

### 5.5 Time-travel via append-only event log

---

## 6. Experimental Results

### 6.1 LongMemEval-S (conventional retrieval axis)

| System          | R@1       | R@5       | R@10      |
|-----------------|-----------|-----------|-----------|
| MemPalace (raw) | 80.6%     | 96.6%     | 98.2%     |
| **Lethe v1**    | **85.4%** | **97.4%** | **99.0%** |

### 6.2 ForgetEval (the proposed axis)

| System        | super | decay | amnesia | purge | drift | Overall                       |
|---------------|------:|------:|--------:|------:|------:|------------------------------:|
| **Lethe v1**  | 100%  | 100%  | 98%     | 100%  | 99%   | **99.3%** (993 / 1000)        |
| Mem0 (2.0.2)  | 100%  | 100%  | 70%     | 75%   | 100%  | 88.8%                         |
| MemPalace     | 0%    | 0%    | 0%      | 0%    | 0%    | 0% (no forgetting primitives) |

### 6.3 Multilingual ForgetEval (English, Chinese, Japanese)

*[port multilingual table from docs §10; discuss CJK weaknesses in
both adapter and tokenizer]*

---

## 7. Production Failure Analysis

A 70% amnesia rate means three in ten requests to forget a user
leave fragments reachable to other queries — a GDPR liability and
a stale-context bug.  A 75% purge rate means one in four
identifier-precise deletions either miss the target or take a
neighbor with them — the silent delete-by-similarity failure that
bricks compliance audits.  MemPalace's zeros are the opposite
failure: a system that cannot service a deletion request at all.

*[expand: case studies of each failure mode mapped to specific
production incidents reported in the literature or industry
post-mortems]*

---

## 8. Limitations

*(port limitations from docs §11, expanded for paper voice)*

### 8.1 Language coverage

### 8.2 Template-generated cases

### 8.3 Synthetic distractors

### 8.4 Top-k substring matching

### 8.5 Pre-LLM evaluation only

*[discuss: we deliberately avoid LLM judges for reproducibility, but
a complementary axis (e.g. "did the agent's final answer change
appropriately after the forgetting operation?") could be added in
v0.2]*

---

## 9. Conclusion

*(to be written; ~one page)*

The memory framework community has built sophisticated retrieval
infrastructure on the unstated assumption that the failure mode
worth engineering against is losing a fact.  Production deployments
indicate the opposite.  ForgetEval makes the inverse axis
benchmarkable; Lethe demonstrates the axis is satiable.  We invite
the field to cross-publish on this metric.

---

## Acknowledgments

*(to be written: sqlite-vec, fastembed, FTS5 maintainers; arXiv +
PyPI infrastructure; etc.)*

## Bibliography

See `refs.bib`.
