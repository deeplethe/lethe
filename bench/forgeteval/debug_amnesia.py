"""Inspect similarity scores during amnesia tests to tune threshold."""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from fastembed import TextEmbedding
from lethe import Lethe

model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
def emb(t): return list(next(iter(model.embed([t]))))

def trace(setup: list[str], query: str):
    lethe = Lethe(":memory:", vector_dim=384, embedder=emb)
    for t in setup:
        lethe.inscribe(t)
    print(f"\nquery: {query!r}")
    for r in lethe.recall(query, k=10, hybrid=False):
        print(f"  sim={r.similarity:+.3f}  {r.memory.text[:80]}")
    lethe.close()


print("=== amnesia_isolated ===")
trace(
    ["Alice likes anchovy pizza.",
     "Bob likes spicy ramen.",
     "Charlie likes plain rice."],
    "Alice anchovy pizza preferences",
)

print("\n=== amnesia_multi_facts_one_subject ===")
trace(
    ["Dana works at Stripe.",
     "Dana lives in Berlin.",
     "Dana speaks Portuguese.",
     "Eve works at Notion."],
    "Dana profile information",
)

print("\n=== purge_gdpr ===")
trace(
    ["Customer Alice@example.com bought 3 items last week.",
     "Customer Bob@example.com placed an order yesterday."],
    "Alice@example.com customer history",
)
