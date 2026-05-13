"""Belief revision: a fact about an entity changes over time.

Naive memory: "User started at Google" AND "User moved to Meta" AND
"User now at Anthropic" — all retrievable.  The agent will see three
near-duplicate facts about the same axis and confabulate.

Lethe's `supersede` links new → old in the supersession edge table.
Default recall returns only the latest; `blame()` walks the chain
when you need provenance.
"""
from lethe import Lethe
from lethe.embed import hash_embed


def main() -> None:
    agent = Lethe(":memory:", vector_dim=768, embedder=hash_embed)

    # History as it actually happened
    v1 = agent.inscribe("User started at Google in 2020.")
    v2 = agent.surrender(
        {"old": v1, "new": "User moved to Meta in 2022."},
        mode="supersede",
    )
    v3 = agent.surrender(
        {"old": v2, "new": "User joined Anthropic in 2025."},
        mode="supersede",
    )

    # The agent only sees the latest belief
    hits = agent.recall("Where does the user work?", k=3)
    print(f"  OK Top-1: {hits[0].memory.text}")
    assert "Anthropic" in hits[0].memory.text

    # Intermediate beliefs are submerged (depth = 0)
    blob = " ".join(h.memory.text for h in hits).lower()
    assert "google" not in blob and "meta" not in blob
    print("  OK Old beliefs (Google, Meta) not in top-3")

    # When provenance matters, blame() walks the chain
    print("\n  blame:")
    for e in agent.blame("user's employer history", k=5):
        tag = ""
        if e.superseded_by:
            tag = f"  -> superseded by #{e.superseded_by}"
        elif e.supersedes:
            tag = f"  <- supersedes {e.supersedes}"
        print(f"    #{e.memory.id:>2} depth={e.memory.depth:>5}  "
              f"{e.memory.text[:50]:50}{tag}")

    agent.close()


if __name__ == "__main__":
    main()
