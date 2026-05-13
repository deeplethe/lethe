# ForgetEval-Adv

A hand-crafted adversarial layer for ForgetEval (v0.2).  Complements the
1000-case template-generated suite (v0.1) with **64 carefully designed
cases** that probe failure modes templates cannot reach.

---

## 1. Why adversarial cases

Template generation is deterministic and large-scale, but its weakness
is the **flatness of its difficulty distribution**: every case is drawn
from the same N sub-templates with i.i.d. entity substitution.  A
system that handles one template case well will handle all of them
well.

Real production failures, by contrast, are **structured attacks** on
the system's heuristics: substrings that accidentally trigger a
distractor, identifiers that share long prefixes, supersessions where
the new fact is a paraphrase rather than a near-copy, negation flips,
temporal qualifiers, etc.

ForgetEval-Adv targets exactly these attacks.

---

## 2. Ten attack categories (112 cases total — v0.4)

| #  | Attack category            | n  | Primary family       | What it probes                                                                       |
|----|----------------------------|---:|----------------------|--------------------------------------------------------------------------------------|
| 1  | `substring_trap`           |  8 | all 5                | `must_not_contain` substring accidentally appears in a distractor or related fact    |
| 2  | `prefix_collision`         | 16 | purge                | identifiers share a long common prefix; deleting one must not take the other         |
| 3  | `paraphrase_supersession`  |  8 | supersession, drift  | old fact and new fact have low surface overlap; only semantic alignment can match    |
| 4  | `negation_trap`            |  8 | supersession, decay  | negated fact ("does NOT work at X") must not be confused with the affirmative        |
| 5  | `temporal_qualifier`       |  8 | supersession, drift  | facts with embedded dates; supersession must respect the temporal window             |
| 6  | `shared_attribute`         | 16 | amnesia              | multiple entities share one attribute; forgetting one must not collapse the other    |
| 7  | `compound_fact`            |  8 | supersession         | one sentence carries two facts; superseding one must preserve the other              |
| 8  | `identifier_obfuscation`   | 16 | purge                | same identifier in different surface forms (case, whitespace, encoding)              |
| 9  | `cross_lingual_identifier` | 16 | purge                | same entity stored under different scripts or romanizations (GDPR multilingual)      |
| 10 | `recursive_supersession`   |  8 | drift                | supersede chain where the LATEST state matches an earlier-superseded state           |

Categories with elevated `n=16` are the ones with measurable
between-system variance from v0.2 → v0.3 runs; saturated and
zero-coverage categories stay at `n=8` because additional cases
there only tighten an already-tight CI bound.

### Why these eight

Each attack corresponds to a specific architectural weakness we
observed (or could plausibly observe) during development of Lethe and
the comparison adapters:

- **`substring_trap`** — substring-based scoring (ForgetEval's default)
  is cheap and deterministic but vulnerable to spurious matches.  A
  must-not "Stripe" can be falsely failed by a distractor about "fish
  stripes."  Catches: brittle scoring, *not* brittle systems.  We use
  this to validate the bench itself.
- **`prefix_collision`** — Lethe's lexical-purge path uses BM25, which
  ranks by IDF-weighted token overlap.  Two emails with a long shared
  prefix may rank near-identically.  Catches: insufficient
  identifier-precision.
- **`paraphrase_supersession`** — RRF fusion can over-rely on BM25
  when the new fact lexically diverges; can over-rely on vec when the
  entity tokens dominate.  Catches: shallow-only supersession matching.
- **`negation_trap`** — embeddings often blur "X" and "not X".  In
  retrieval this is usually benign, but in supersession or decay,
  releasing "does not own X" should not affect recall of "owns X."
- **`temporal_qualifier`** — drift through facts like "joined Google
  in 2020" → "moved to Meta in 2022" → "visited Google in 2024"
  requires the system to either ignore time or respect it; either
  choice can fail naively.
- **`shared_attribute`** — amnesia's hardest case: Dana and Eve both
  live in Berlin.  Releasing "everything about Dana" must drop the
  Dana-Berlin link but preserve the Eve-Berlin link.
- **`compound_fact`** — "User lives in Berlin and works at Stripe"
  carries two facts.  Superseding only the location must not delete
  the employment.
- **`identifier_obfuscation`** — `alice@acme.io` vs `Alice@Acme.io`
  vs `alice@acme.io ` (trailing whitespace) are the same identifier
  for any reasonable interpretation.  Catches: case-sensitive lexical
  paths.

We deliberately exclude attacks that depend on multi-turn LLM
reasoning (those belong in a v0.3 "agent-loop" bench).

---

## 3. Case format

Identical to the template-generated cases: each adversarial case is a
`GeneratedCase` dataclass instance.  This means the existing runner,
adapter Protocol, and scoring logic in
[`bench/forgeteval/run.py`](../bench/forgeteval/run.py) all work
unchanged.

```python
@dataclass
class GeneratedCase:
    id: str                   # e.g. "adv_substring_trap_001"
    family: str               # one of supersession / decay / amnesia / purge / drift
    setup_facts: list[str]    # inscribed in order; may include in-line distractors
    mutations: list[tuple]    # ("supersede", old_q, new_text) | ("release", q) | ("purge", q)
    final_query: str          # the question
    must_contain: list[str]   # substrings that MUST appear in top-10 recall
    must_not_contain: list[str]  # substrings that MUST NOT appear
```

Each case additionally carries (in source comments) the **design intent**:

```python
# attack_category: prefix_collision
# intent: deleting alice@acme.io should not also delete alice.smith@acme.io
#         (shared 5-char prefix "alice")
# expected lethe behavior: pass via lexical (BM25) purge
# expected mem0 behavior: fail due to vector similarity blur
```

These comments are the **annotation record** for the IAA protocol
(§5).

---

## 4. Scoring

Same as template-generated suite:
- `must_contain` $\subseteq$ top-10 blob $\wedge$ `must_not_contain` $\cap$ top-10 blob $= \emptyset$
- Per case: 1 / 0, no partial credit
- Per attack category: `pass / 8`
- Overall: `pass / 64`
- `NotImplementedError` from optional adapter methods → counted as fail,
  surfaced as N/A in reports (same convention as template)

We **deliberately use the same substring scoring** as v0.1 even on
adversarial cases.  This keeps the methodology coherent: a forgetting
failure is a substring failure regardless of how the case was
generated.

---

## 5. Annotation and inter-annotator agreement protocol

NeurIPS Datasets-and-Benchmarks reviewers will (rightly) ask: who
decided these are good adversarial cases, and how reliable is that
judgement?

We implement a two-stage IAA protocol:

### Stage A: self-IAA over a 7-day cool-off

1. Author writes the 64 cases plus design-intent annotations (the
   `attack_category` + intent comment above).
2. After 7 days of no-look, author independently re-annotates 32
   randomly-selected cases:
   - Which of the 8 attack categories does this case probe?
   - Is the `must_contain` / `must_not_contain` specification
     unambiguous?
3. Compute Cohen's $\kappa$ on the category labels.  Target $\kappa
   \geq 0.75$ ("substantial agreement").  Cases that disagree are
   rewritten or moved to the most appropriate category.

### Stage B: external partial-IAA

1. Recruit 1–2 external annotators with LLM / IR background.
2. Each annotates 20–30 cases (no overlap with the self-IAA subset).
3. Compute pairwise $\kappa$.
4. Report all $\kappa$ values in the paper's §6.10.

### Failure thresholds

- $\kappa < 0.6$ on any pair → category boundary is unclear; redesign
  category definitions in §2.
- Individual case disagreed on by $\geq 2$ annotators → discard or
  rewrite.

---

## 6. Reporting format

Per attack category, per system, pass rate.  Plus the gap to
template-only score.

```
            ┌─────────────────── ForgetEval ─────────────────┐
            │ Template (1000)     Adversarial (64)    Δ      │
Lethe v1    │   99.3 %                ?? %         (−??)    │
Mem0 2.0.2  │   88.8 %                ?? %         (−??)    │
MemPalace   │    0.0 %                ?? %         (−??)    │
```

And per attack category (8 rows × 3 systems).  This goes in §6.10 of
the paper alongside the existing tables.

---

## 7. The LLM-optional hook (Lethe's response to v0.2)

Running v0.2 against Lethe revealed two attack categories where the
default LLM-free adapter scores 0 / 8:

- `compound_fact` — a single inscribed row carries two facts joined
  by " and ".  The atomic `supersede` primitive wipes both clauses
  together because the depth axis treats the row as the smallest unit
  of forgetting.
- `identifier_obfuscation` — surface-form variations of the same
  identifier (case, whitespace, quoting, leading @, separator variants
  in phones / UUIDs / SSNs / credit cards) are not grouped by the
  default purge, which matches BM25 top-1 plus exact-text duplicates.

Both failures require **semantic understanding** of either fact
decomposition or identifier equivalence.  We considered three
architectural responses:

1. **Brittle string heuristics in the adapter** — regex split on
   " and ", explicit negation-marker whitelists, cosine-similarity
   thresholds tuned per case, length-and-digit-ratio gates for
   identifier matching.  This raised the adversarial score to ~93 %
   but at the cost of magic constants and ad-hoc pattern matching
   that the project's own CONTRIBUTING guide explicitly warns against
   (`"No regex query routers, no query-type classifiers, no 'if `:`
   in query then ...'"`).  **Rejected.**
2. **In-engine semantic logic** — push the same heuristics into
   `lethe/core.py`.  Same brittleness, plus it pollutes the engine
   with policy that's hard to swap.  **Rejected.**
3. **LLM-optional adapter hook** — keep the engine deterministic and
   primitive-only; let the LetheAdapter take an optional
   `llm: Callable[[str], str]` parameter that delegates exactly the
   two narrow semantic decisions (supersede-mode planning, identifier
   equivalence) to the model.  Recall hot path remains LLM-free.
   **Adopted.**

The contract is two prompts, both expecting a JSON-shaped response of
at most two fields:

- **`supersede` plan** — input is `(EXISTING_MEMORY, SUPERSEDE_QUERY,
  NEW_FACT)`; output is `{"mode": "atomic"}` or `{"mode": "partial",
  "merged_text": "..."}`.  In partial mode the adapter calls
  `surrender(id, mode="edit", new_text=...)` instead of the atomic
  supersede.
- **`purge` identifier grouping** — input is `(TARGET_IDENTIFIER,
  CANDIDATES)`; output is `{"matching_indices": [...]}` listing which
  candidate rows describe the same identifier.

Full prompts and adapter wiring are in
[`bench/forgeteval/adapter.py`](../bench/forgeteval/adapter.py).  An
Anthropic Claude runner ships at
`lethe-paper/scripts/run_adversarial_with_llm.py`; export
`ANTHROPIC_API_KEY` and execute to obtain the with-LLM numbers.

Architectural invariants preserved:

1. **Engine has zero new heuristics.**  `lethe/core.py` gained exactly
   one new primitive — `surrender(mode="edit", new_text=...)` — and
   nothing else.  No canonicalization helpers, no regex, no
   identifier-shape detection live in the engine.
2. **Recall is always LLM-free.**  The LLM is consulted only at
   `supersede` and `purge` time, and only once per call.  Time-travel,
   pinning, and decay never touch a model.
3. **Same primitives, two policy modes.**  `LetheAdapter(llm=None)`
   and `LetheAdapter(llm=callable)` use the same engine primitives.
   The only difference is whether semantic decisions are routed
   through the model.  This makes the comparison rigorous: any
   difference in the adversarial score is attributable purely to the
   policy layer.

## 8. Observed scores (v0.4, LLM-free)

| System         | template (1000) | adversarial (112) | wall / case |
|----------------|----------------:|------------------:|------------:|
| **Lethe v1**   |  993 (99.3 %)   |  70 (62.5 %)      |   ~48 ms    |
| Mem0 v2.0.2    |  888 (88.8 %)   |  76 (67.9 %)      |  ~527 ms (11×) |
| LangMem (LG)   |  995 (99.5 %)   |  69 (61.6 %)      |   ~56 ms (1.2×) |
| MemPalace      |    0 ( 0.0 %)   |   0 ( 0.0 %)      |  ~167 ms    |

The three deterministic systems land within 6 absolute points on
adversarial; their overall Wilson 95 % CIs overlap.  **The honest
comparison surface is per-category**, where:

- **Lethe 100 % > Mem0 50 %** on `prefix_collision` (16 cases, Wilson
  intervals do not overlap → significant at p < 0.05).  Lethe's pure-
  BM25 lexical purge avoids the prefix-similarity confusion that
  vector-similarity-based delete falls into.
- **Mem0 50 % > Lethe 0 %** on `cross_lingual_identifier` (16 cases,
  significant).  Mem0's multilingual-MiniLM embedding accidentally
  bridges some script-equivalent identifiers; Lethe's exact-text
  equality cannot.
- **Mem0 50 % > LangMem 0 %** on `cross_lingual_identifier`
  (significant) — same mechanism.
- All three deterministic systems score **0 / 8 on `compound_fact`**
  — superseding one clause of "X and Y" with a fact about X wipes
  Y with it.  This is the **deterministic ceiling** that motivates
  the LLM-optional adapter hook (§7).

These are pre-registered hypothesis tests: per-category claims that
the bench was designed to evaluate, made with explicit Wilson
intervals.  Aggregate "Lethe beats Mem0" claims are not made
because the data does not support them at this case count;
"trade-off" is the honest read of the aggregate numbers.

LLM-assisted runs (with `LetheAdapter(llm=Anthropic-Claude)`) are
out of scope for this no-LLM-environment run and will be reported
when `ANTHROPIC_API_KEY` is set via
`lethe-paper/scripts/run_adversarial_with_llm.py`.

Data: `lethe-paper/data/adversarial_results.json` (v0.4 numbers).

---

## 9. File layout

```
lethe/
├── bench/forgeteval/
│   ├── adversarial.py        ← module exporting ADVERSARIAL_TESTS: list[GeneratedCase]
│   └── run.py                ← gains --suite {smoke,template,adversarial,all} flag
├── docs/
│   ├── forgeteval.md         ← existing v0.1 methodology
│   └── forgeteval_adversarial.md  ← THIS DOCUMENT
└── data/                     (in lethe-paper/data/)
    └── adversarial_results.json
```

---

## 10. Limitations of this layer

- **Single-author origin.**  All 64 cases are initially designed by
  one person.  Stage-B external IAA partially addresses this.  A
  larger, multi-author adversarial layer is a v0.3 goal.
- **English only.**  CJK adversarial cases require native speakers and
  are not in v0.2 scope.
- **Static.**  Once the cases are published, adapters could overfit
  to them.  ForgetEval treats this as a feature for the published
  suite (it's a fair, fixed test) and plans a hidden test set for
  competition rounds.
- **64 is a small number.**  Adversarial coverage is depth, not
  breadth — we trade fewer cases for more careful design.  Template
  generation still provides the breadth (1000+).
