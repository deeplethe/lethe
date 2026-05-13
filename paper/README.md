# Paper

Working draft of the **ForgetEval** benchmark paper.  Framed as a
benchmark-track contribution rather than a system paper —
`ForgetEval: A Benchmark for Memory Deletion in AI Agent Systems`.
Lethe is presented as a reference implementation in one section, not
the headline contribution.

## Why this framing

The methodology is more reusable than the implementation.  System
papers age out (the depth axis may not be the final architecture);
benchmarks compound (other teams cite and contribute against the
metric).  By leading with the benchmark, we make the field's adoption
the metric of success, not the popularity of Lethe.

## Layout

```
paper/
├── README.md          ← this file
├── forgeteval.md      ← main draft, markdown
├── refs.bib           ← bibliography (BibTeX, ready for pandoc → LaTeX)
└── figures/           ← plots, diagrams (populated later)
```

## Target venue

**arXiv preprint first.**  Two-week turnaround once the draft is done;
no peer-review delay; gives us a citable handle for blog posts and
Hacker News.

Peer-reviewed track later (NeurIPS Datasets & Benchmarks /
ACL benchmarks workshop), after the methodology has community
adoption — multiple ForgetEval adapters submitted to the Lethe repo
become the social proof that justifies a D&B submission.

## Build workflow

While we're in markdown:

```bash
# Word count and TOC check
wc -w paper/forgeteval.md
grep -E '^#{1,3} ' paper/forgeteval.md
```

When ready for arXiv (LaTeX):

```bash
pandoc paper/forgeteval.md -o paper/forgeteval.tex \
    --bibliography=paper/refs.bib \
    --template=...   # standard arXiv template
```

## Status

- [x] Skeleton + working title
- [x] §1 Introduction (first draft)
- [ ] §2 Related work
- [ ] §3 Methodology (mostly portable from docs/forgeteval.md)
- [ ] §4 Adapter contract
- [ ] §5 Reference implementation: Lethe
- [ ] §6 Experimental results
- [ ] §7 Production failure analysis
- [ ] §8 Limitations + roadmap
- [ ] §9 Conclusion
- [ ] Bibliography
- [ ] Figures (architecture diagram, score breakdown by family, etc.)
