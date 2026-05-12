"""Find the 5 single-session-preference questions Lethe v1 missed at R@5,
print their structure so we can spot the pattern."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bench"))

from mempalace_bench import evaluate_retrieval  # type: ignore

data = {q["question_id"]: q for q in json.loads(
    (REPO / "bench" / "longmemeval_s.json").read_text(encoding="utf-8"))}

results = {}
for line in (REPO / "bench" / "results_v1_pref.jsonl").read_text(encoding="utf-8").splitlines():
    r = json.loads(line)
    if r.get("skipped"): continue
    results[r["question_id"]] = r

prefs = [qid for qid, q in data.items()
         if q["question_type"] == "single-session-preference"]

print(f"Single-session-preference: {len(prefs)} questions")
fails = []
for qid in prefs:
    r = results.get(qid)
    if not r: continue
    q = data[qid]
    answer = set(q["answer_session_ids"])
    ra, _, _ = evaluate_retrieval(r["ranked"], answer, r["corpus_ids"], 5)
    if ra < 1:
        fails.append((qid, q, r))

print(f"Failures (R@5 missed): {len(fails)}\n")
for i, (qid, q, r) in enumerate(fails, 1):
    print(f"━━━ FAIL #{i}: {qid}")
    print(f"  Q: {q['question']}")
    print(f"  Gold answer:    {q['answer'][:100]}")
    answer_sid = list(set(q['answer_session_ids']))[0]
    # Find answer session
    for s, sid in zip(q['haystack_sessions'], q['haystack_session_ids']):
        if sid == answer_sid:
            user_turns = [t['content'] for t in s if t.get('role') == 'user']
            preview = ' | '.join(t[:90] for t in user_turns[:3])
            print(f"  Answer-session content: {preview[:280]}")
            break
    # Top-5 retrieved
    top5_indices = r['ranked'][:5]
    top5_sids = [r['corpus_ids'][i] for i in top5_indices]
    print(f"  Top-5 retrieved sessions: {top5_sids}")
    print()
