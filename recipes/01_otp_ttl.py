"""OTP / verification code with TTL.

When a one-time password is consumed it should leave normal recall.
But an auditor — six months later — may need to prove the OTP was
ever issued.  Lethe's `release` does both: depth → 0 (out of recall)
while the event log remembers the inscribe.
"""
from lethe import Lethe
from lethe.embed import hash_embed


def main() -> None:
    agent = Lethe(":memory:", vector_dim=768, embedder=hash_embed)

    # User requests login → OTP generated and stored
    otp_id = agent.inscribe("Session OTP for user@example.com: 887766")

    # User submits the OTP → we look it up
    hits = agent.recall("OTP for user@example.com", k=1)
    assert "887766" in hits[0].memory.text
    print(f"  OK OTP retrievable: {hits[0].memory.text}")

    # OTP consumed → release.  Depth → 0, gone from default recall.
    agent.surrender(otp_id, mode="release")

    hits = agent.recall("OTP for user@example.com", k=1)
    assert all("887766" not in h.memory.text for h in hits)
    print("  OK OTP not recallable after release")

    # Six months later, auditor needs to prove the OTP was issued.
    # Time-travel: find the inscribe event, recall AT that timestamp.
    events = agent.log(kind="inscribe")
    otp_event = next(e for e in events if e.memory_id == otp_id)
    past_hits = agent.recall("OTP", k=1, at=otp_event.timestamp + 1e-3)
    assert past_hits and "887766" in past_hits[0].memory.text
    print(f"  OK Auditor proof: OTP existed at t={otp_event.timestamp:.0f}")

    agent.close()


if __name__ == "__main__":
    main()
