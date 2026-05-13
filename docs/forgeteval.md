# ForgetEval

A proposed benchmark methodology for memory-system **forgetting**.
Conventional benchmarks (LongMemEval, MTEB-Memory) measure retrieval
recall: *does it stay when you need it?*  ForgetEval measures the
inverse: *does it leave when you ask?*

This document specifies the methodology completely enough that any team
can reproduce it, extend it, or apply it to their own memory system.

---

## 1. Motivation

Production memory systems do not fail by losing facts.  They fail by
keeping them: rotated passwords, GDPR-deleted users still in
recommenders, job titles wrong for two years because the supersede
never landed, OTP codes from last Tuesday permanently embedded next to
real preferences.

Every popular memory framework (Mem0, MemPalace, Letta, Zep, HippoRAG)
optimizes one direction: *remember more, never lose.*  None of them
benchmark the inverse.  ForgetEval exists to make that axis legible.

---

## 2. The five families

Each family probes one structural property the memory system must
exhibit if it is to be safe in production.

### 2.1 Supersession

When a fact changes (job, address, preference), the **new fact must win
recall** and the **old fact must not appear in top-k**.

> User works at Stripe.  *(later)*  → User now works at Anthropic.
> Q: "Where does the user work?"
> Expected: "Anthropic" in top-k; "Stripe" not in top-k.

The default failure mode is to leave both facts live with similar
recall scores — the agent sees both and confabulates.

### 2.2 Decay

A **released** fact (TTL expired, OTP consumed, intent cancelled) must
**stay out of top-k**.  Decay is the simplest forgetting operation but
exposes whether the system has *any* primitive for "this is no longer
relevant."

> Session OTP: 887766.  *(consumed)* → release.
> Q: "What's the OTP?"
> Expected: top-k does not contain 887766.

### 2.3 Amnesia

Forget every fact about **one entity**, while sibling entities survive.
The hardest part is *width control*: too narrow forgets only one fact
out of five; too wide forgets unrelated siblings.

> Dana works at Stripe.  Dana lives in Berlin.  Dana speaks Portuguese.
> Eve works at Notion.  → release everything about Dana.
> Q: "Tell me about people."
> Expected: top-k contains Eve, not Dana.

### 2.4 Purge

Hard-delete by **identifier** (email, API key, patient name).  Unlike
release, purge is non-reversible and *precise*: deleting
`alice@acme.io` must not also delete `bob@acme.io` even though both
rows look semantically near-identical to an embedder.

> Customer alice@acme.io bought 3 items.
> Customer bob@acme.io ordered yesterday.
> → purge alice@acme.io.
> Q: "Show me customer alice@acme.io data."
> Expected: no alice@acme.io in top-k; bob@acme.io still present.

Purge is the operation compliance regulations (GDPR Article 17, HIPAA,
SOC2) actually care about.  Semantic neighborhood is the wrong
primitive here — **purge demands lexical match by identifier**.

### 2.5 Drift

A **chain** of supersedes — the topic gets updated several times.  Only
the most recent belief wins; every intermediate must be unreachable.

> User started at Google in 2020.
> → moved to Meta in 2022.
> → at Anthropic in 2025.
> Q: "Where does the user work?"
> Expected: Anthropic in top-k; Google and Meta absent.

Drift is the long-time-horizon failure mode: a system might handle one
supersede correctly but accumulate stale intermediates over N.

---

## 3. Case anatomy

Every test case is a single dataclass:

```python
@dataclass
class GeneratedCase:
    id: str                          # "supersession__job__042"
    family: str                      # one of the five above
    setup_facts: list[str]           # inscribed in order, mixed with distractors
    mutations: list[tuple]           # ("supersede", old_query, new_text)
                                     # ("release",   query)
                                     # ("purge",     query)
    final_query: str                 # the question to recall under
    must_contain: list[str]          # substrings that MUST appear in top-k
    must_not_contain: list[str]      # substrings that MUST NOT appear in top-k
```

The run loop, in 6 lines:

```python
adapter.reset()
for f in case.setup_facts: adapter.inscribe(f)
for m in case.mutations:   adapter.<op>(*m[1:])
top   = adapter.recall_texts(case.final_query, k=10)
blob  = " ".join(top).lower()
passed = (all(s.lower()     in blob for s in case.must_contain) and
          all(s.lower() not in blob for s in case.must_not_contain))
```

No LLM judge.  No fuzzy matching.  Exact substring check on the joined
text of top-k results — a system either kept the right thing and lost
the wrong thing, or it didn't.

---

## 4. Distractors

Each case mixes the target facts with **N filler facts** unrelated to
the target's tokens (office-life trivia: meeting rooms, coffee
machines, fire drills).  Two reasons:

1. **Forces real retrieval.**  Without distractors, top-k is trivially
   correct: only the target was inscribed.
2. **Stress-tests width control.**  A naive `release(query)` that
   over-evicts will catch the wrong neighbors first; distractors widen
   the candidate pool so the bug shows up.

Default `--distractors 4` matches typical conversational density.
`--distractors 50` simulates large background corpora and surfaces
adaptive-threshold weaknesses.

---

## 5. Sub-templates

Each family has 4 sub-templates the generator cycles through, giving
case-level variety.  All sub-templates are pure functions of an RNG +
the entity pools (`NAMES`, `COMPANIES`, `CITIES`, `COLORS`, `DIETS`,
`THEMES`, `LANGUAGES`, `BOOKS`, `HOBBIES`, `STREETS`, `EMAIL_DOMAINS`,
`CONDITIONS`).

| Family       | Sub-templates                                                                                             |
|--------------|-----------------------------------------------------------------------------------------------------------|
| supersession | `job`, `theme`, `diet`, `long_form` (multi-clause fact where only the central claim changes)              |
| decay        | `otp`, `verification_code`, `flight_cancelled`, `one_of_many` (5 sibling OTPs, release exactly one)       |
| amnesia      | `person`, `multi`, `topic`, `many_facts` (5 facts about target vs 1 about peer)                           |
| purge        | `api_key`, `phi`, `gdpr`, `many_similar` (5 sibling API keys, purge one)                                  |
| drift        | `jobs`, `address`, `color`, `long_chain` (5-step supersession chain)                                      |

The hard sub-templates (`long_form`, `one_of_many`, `many_facts`,
`many_similar`, `long_chain`) are deliberately designed to expose
specific architectural weaknesses we observed during development —
they are the most interesting signal.

Full source: [`bench/forgeteval/generate.py`](../bench/forgeteval/generate.py).

---

## 6. Generation protocol

- **Deterministic.**  `generate(n_per_family, seed=42, distractors=4)`
  returns the same 5 × n cases byte-for-byte every call with the same
  seed.  No floating-point sources, no time.
- **No LLM.**  All cases are template + entity-pool substitution.
- **No API.**  Generation runs offline; embedding (during evaluation)
  uses whichever embedder the adapter is configured with.
- **No training-set contamination.**  The entity pools are short, real
  proper nouns (Alice, Stripe, Berlin); no public benchmark text is
  reused.

Cases per family scale linearly with `n_per_family`.  Typical runs:

| Profile  | n_per_family | distractors | total cases | wall (Lethe, MiniLM) |
|----------|-------------:|------------:|------------:|---------------------:|
| smoke    |  10          |  4          |   50        |  ~4 s                |
| standard |  50          |  4          |  250        |  ~20 s               |
| full     | 200          |  4          | 1000        |  ~13 min             |
| stress   |  50          | 50          |  250        |  varies              |

---

## 7. Scoring

Per case: pass (1) or fail (0).  No partial credit.

Per family: `pass / total` and pass rate.

Overall: `pass / total` across all families, plus the number of cases
marked **N/A** (capability not implemented; counted as a fail in
overall rate, surfaced separately in the report).

The runner ([`bench/forgeteval/run.py`](../bench/forgeteval/run.py))
distinguishes three outcomes:

- `✓ pass` — `must_contain` ⊆ top-k blob ∧ `must_not_contain` ∩ top-k blob = ∅
- `✗ fail` — adapter responded, but the conditions weren't met
- `· N/A`  — adapter raised `NotImplementedError`; the operation isn't supported

N/A is honest, not generous: a memory system that cannot supersede is
worse than one that supersedes badly, because the worse system at least
gives you the primitive to compose with.

---

## 8. Reproducibility

```bash
# Install
pip install -e .[embed]

# Hand-crafted 15-case smoke set (3 per family)
py bench/forgeteval/run.py --adapter lethe

# Templated 250 cases (50 per family, default distractors=4)
py bench/forgeteval/run.py --adapter lethe --scale 50

# Full 1000-case bench
py bench/forgeteval/run.py --adapter lethe --scale 200

# Same protocol against another adapter
py bench/forgeteval/run.py --adapter mem0      --scale 200
py bench/forgeteval/run.py --adapter mempalace --scale 200

# Stress: heavy distractors
py bench/forgeteval/run.py --adapter lethe --scale 50 --distractors 50
```

Seed defaults to 42 across all runs; pass `--seed N` to vary.

---

## 9. Adapter contract

To evaluate a new memory system, write an adapter implementing this
protocol (it's literally a [`typing.Protocol`](../bench/forgeteval/adapter.py)):

```python
class Adapter(Protocol):
    name: str

    def reset(self) -> None: ...
    def inscribe(self, text: str) -> int | str: ...
    def recall_texts(self, query: str, k: int = 5) -> list[str]: ...

    # Optional — raise NotImplementedError for honest N/A
    def supersede(self, old_query: str, new_text: str) -> None: ...
    def release(self, query: str) -> int: ...
    def purge(self, query: str) -> int: ...
```

That's six methods.  Three are mandatory (`reset`, `inscribe`,
`recall_texts`); three are optional and each `NotImplementedError`
correctly maps to N/A on the relevant families.

Existing adapters in the repo (Lethe, Mem0, MemPalace) are <130 lines
each — write a new one and submit a PR; we'll cross-publish results.

---

## 10. Multilingual

`--lang en|zh|ja` selects native-language entity pools and phrasing.
Each non-English module
([`lang_zh.py`](../bench/forgeteval/lang_zh.py),
[`lang_ja.py`](../bench/forgeteval/lang_ja.py))
mirrors the English structure: four sub-templates per family, native
filler facts, the same case anatomy.

`run.py` auto-switches the default embedder to
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` for any
non-English language; pass `--embedder` to override.

Lethe scale=50 (250 cases per language):

| Lang | super | decay | amnesia | purge | drift | Overall |
|------|------:|------:|--------:|------:|------:|--------:|
| en   | 100%  | 100%  | 98%     | 100%  | 99%   | 99.3% (at 1000 cases) |
| zh   |  90%  | 100%  |  36%    |  74%  |  90%  | 78%     |
| ja   |  94%  | 100%  |  44%    |  74%  |  72%  | 77%     |

The drop on CJK is **real signal, not a bug** — and surfaces two
addressable architectural choices:

- The default multilingual embedder (paraphrase-multilingual-MiniLM-L12-v2)
  clusters CJK text more tightly than English, so the adaptive-gap
  release heuristic cannot find a clean split point.  Amnesia is the
  family most exposed to this (98% → 36–44%).
- The default FTS5 tokenizer (`porter unicode61`) splits CJK
  per-character, which destroys word-level BM25 ranking.  Purge,
  which depends on `recall(lexical=True)`, drops to 74% even though
  identifier strings (emails, API keys) are still ASCII.

Roadmap on the CJK side:

- Stronger multilingual embedder (`BAAI/bge-m3` or
  `intfloat/multilingual-e5-large`) — option, not default, because
  they are 5–10× the size.
- CJK-aware FTS5 tokenizer (e.g. `unicode61 tokenchars`, jieba for
  zh, MeCab for ja) — plug-in at schema-init time.

## 11. Limitations

ForgetEval v0 is intentionally narrow.  Honest weaknesses:

- **Three languages.**  English / Chinese / Japanese only.  Korean,
  Arabic, Hindi etc. are not yet covered.
- **Same-language only.**  Cross-lingual paraphrase (English facts,
  Chinese queries) is not tested.
- **Short facts.**  Most setup facts are one sentence.  Long-form
  document forgetting (paragraph-level supersede) is touched only by
  `supersession_long_form` and `amnesia_many_facts`.
- **Synthetic distractors.**  Office-life trivia, not real-world memory
  density.  A `--distractors 50` stress run helps but doesn't replace
  in-the-wild corpora.
- **No human curation.**  Every case is template-generated.  A
  human-curated set of adversarial paraphrases would expose more.
- **Top-k substring matching.**  Cheap, deterministic, but can produce
  false positives if a test entity's substring coincidentally appears
  in a filler fact.  The pools are screened for this but it is not
  formally guaranteed.

Roadmap:

- **v0.2**: human-curated adversarial layer (substring traps, prefix
  collisions, multi-step paraphrase); production-density distractor
  corpora (Wikipedia / email / code, not synthetic office trivia).
- **v0.3**: multilingual scale-up to 1000+ per non-English language;
  CJK-aware tokenizer for the lexical-purge path.
- **v0.4**: cross-lingual paraphrase (English facts, queries in another
  language and vice versa).
- **v0.5**: temporal-reasoning probes (recall at past timestamps —
  `lethe.recall(at=T)` and analogs).
- **v0.6**: receipt-verification family (does the system produce
  auditable proof of deletion, like `lethe.purge_with_receipt()`?).

Currently shipped: v0.1 — 1000 English template cases at scale=200,
five families × four sub-templates, with a Protocol-based adapter
contract; multilingual smoke at 250 cases for en/zh/ja documented in
§10.

---

## 12. Comparison to LongMemEval

| Dimension          | LongMemEval                          | ForgetEval                                |
|--------------------|--------------------------------------|-------------------------------------------|
| What it measures   | Retrieval recall (R@K)               | Forgetting correctness (pass / fail)      |
| Failure mode       | Losing a stored fact                 | Keeping a fact that should be gone        |
| Cases              | 500, human-curated conversations     | 250–1000, generated from templates        |
| Scoring            | R@1 / R@3 / R@5 / R@10 / MRR         | Pass-rate per family + overall            |
| Requires LLM judge | Optional (for generation variant)    | No                                        |
| Capability gaps    | Reported as low R@K                  | Reported as N/A — honest about primitives |

They complement each other.  Lethe's published numbers — 97.4% R@5 on
LongMemEval-S and 99.3% on ForgetEval at 1000 cases — intentionally
cover both.  Mem0 scores 88.8% on the same 1000-case ForgetEval,
weakest on amnesia (70%) and purge (75%).  MemPalace returns 0/1000
because the operations don't exist in its API — that is reported as
N/A across every family.

---

## License

This methodology is published under the same MIT license as the Lethe
codebase.  Forks, re-implementations, and competing leaderboards are
encouraged.
