"""Pin user preferences: depth = +∞, immune to gravity.

`consolidate()` applies decay to every unpinned fact above the
waterline.  But some facts shouldn't decay — the user's name, their
accessibility settings, their primary language.  Pin them once at
first contact and they survive every consolidation pass forever.

Only `surrender(mode="purge")` can still remove a pinned fact;
decay and release skip pinned rows by design.
"""
import time

from lethe import Lethe
from lethe.embed import hash_embed


def main() -> None:
    agent = Lethe(":memory:", vector_dim=768, embedder=hash_embed)

    # Stable user identity — pin so consolidation can never erode it
    name_id = agent.inscribe("The user's name is Alice Chen.")
    lang_id = agent.inscribe("The user reads English and Mandarin.")
    a11y_id = agent.inscribe("The user prefers high-contrast UI.")
    for mid in (name_id, lang_id, a11y_id):
        agent.pin(mid)

    # Ephemeral context — will decay over time
    ctx1_id = agent.inscribe("The user is debugging a Rust borrow-checker error.")
    ctx2_id = agent.inscribe("The user mentioned being tired today.")

    # Simulate 90 days passing under Hypnos gravity
    agent.consolidate(tau_seconds=86400 * 7, now=time.time() + 86400 * 90)

    print("  After 90 days of consolidation:")
    for mid, label in [(name_id, "name"),
                       (lang_id, "lang"),
                       (a11y_id, "a11y"),
                       (ctx1_id, "ctx1"),
                       (ctx2_id, "ctx2")]:
        row = agent.conn.execute(
            "SELECT depth, text FROM memory WHERE rowid=?",
            (mid,),
        ).fetchone()
        depth = row[0]
        marker = "PINNED" if depth == float("inf") else f"{depth:>.3g}"
        print(f"    {label:6} depth={marker:>10}  {row[1][:55]}")

    # Pinned facts still at +inf; ephemeral facts decayed toward 0
    agent.close()


if __name__ == "__main__":
    main()
