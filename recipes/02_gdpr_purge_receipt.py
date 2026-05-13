"""GDPR Article 17 right-to-be-forgotten with a cryptographic receipt.

When a user requests deletion you owe TWO things:
  1. Actually delete the data (not soft-mark it as inactive).
  2. Produce auditable proof — so the regulator can verify, months
     later and without DB access, that the deletion happened.

Lethe's `purge_with_receipt` handles both.  The receipt is an
Ed25519-signed JSON document anchored to a Merkle root over the
entire append-only event log.  Any post-hoc tampering of the log
invalidates the root, which breaks verification.

Run:    python recipes/02_gdpr_purge_receipt.py
Requires:  pip install "pylethe[crypto]"
"""
import json

from lethe import Lethe
from lethe.embed import hash_embed
from lethe.receipt import generate_key, verify_receipt


def main() -> None:
    # One-time: generate a signing key.  In production, store in an
    # HSM / KMS — receipts inherit the key's authority.
    private_key, public_key = generate_key()
    print(f"  signing pubkey: {public_key.hex()[:16]}...")

    agent = Lethe(":memory:", vector_dim=768, embedder=hash_embed,
                  signing_key=private_key)

    # Customer data lives in the store
    alice_id = agent.inscribe("Customer alice@acme.io bought 3 items.")
    bob_id = agent.inscribe("Customer bob@acme.io placed an order.")

    # Alice requests deletion under Article 17
    receipt = agent.purge_with_receipt([alice_id])

    print("  OK Receipt issued:")
    payload = receipt.to_dict()
    for k in ("purged_ids", "purged_text_hashes", "event_log_merkle_root",
              "signature"):
        v = payload[k]
        v = v if isinstance(v, list) else v[:32] + "..."
        print(f"      {k:24} {v}")

    # Verify against the live DB (root recomputes; signature checks out)
    result = verify_receipt(receipt, agent)
    assert result["valid"]
    print(f"  OK verify (with DB):   {result}")

    # Without DB access (e.g. handed to a regulator):
    result_sig_only = verify_receipt(receipt, lethe=None)
    assert result_sig_only["signature_valid"]
    print(f"  OK verify (sig only):  {result_sig_only}")

    # Bob's record is untouched
    hits = agent.recall("bob's order", k=1)
    assert "bob@acme.io" in hits[0].memory.text
    print(f"  OK Bob's record preserved: {hits[0].memory.text}")

    # Save the receipt — Alice / the regulator can verify it without
    # ever touching the database.
    with open("alice_purge_receipt.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print("  OK Receipt saved to alice_purge_receipt.json")

    agent.close()


if __name__ == "__main__":
    main()
