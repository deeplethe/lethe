"""Time-travel recall for debugging.

"Why did the agent say X at 14:32 yesterday?" — to answer that you
need to know what the memory store contained at that moment, not what
it contains now.  Lethe's event log makes `recall(at=T)` deterministic
and O(events).
"""
import time

from lethe import Lethe
from lethe.embed import hash_embed


def main() -> None:
    agent = Lethe(":memory:", vector_dim=768, embedder=hash_embed)

    # T0: initial belief
    agent.inscribe("The user prefers Python over Rust.")
    t0 = time.time()
    time.sleep(0.05)

    # T1: first revision
    old = agent.recall("user's language preference", k=1)[0].memory.id
    agent.surrender(
        {"old": old, "new": "The user now prefers Rust over Python."},
        mode="supersede",
    )
    t1 = time.time()
    time.sleep(0.05)

    # T2: second revision
    old = agent.recall("user's language preference", k=1)[0].memory.id
    agent.surrender(
        {"old": old, "new": "The user picked Go for this project."},
        mode="supersede",
    )
    t2 = time.time()

    # What did the agent believe at each moment?
    print("  user's language preference, over time:")
    for label, t in [("T0  (initial)", t0),
                     ("T1  (first revision)", t1),
                     ("T2  (second revision)", t2)]:
        hits = agent.recall("user's language preference", k=1, at=t)
        text = hits[0].memory.text if hits else "(empty)"
        print(f"    at {label:24} -> {text}")

    print("\n  Same query, present time:")
    hits = agent.recall("user's language preference", k=1)
    print(f"    -> {hits[0].memory.text}")

    agent.close()


if __name__ == "__main__":
    main()
