# Recipes

Runnable cookbook for the most common Lethe usage patterns.  Each file
is a self-contained Python script — clone the repo and run any of
them with:

```bash
python recipes/01_otp_ttl.py
```

| # | Recipe | What it shows |
|---|--------|---------------|
| 01 | [OTP / verification code TTL](01_otp_ttl.py) | `release` after use; time-travel for audit |
| 02 | [GDPR right-to-be-forgotten with receipt](02_gdpr_purge_receipt.py) | `purge_with_receipt` → Article 17 compliance |
| 03 | [Belief revision (supersede chain)](03_belief_revision.py) | only the latest fact surfaces; `blame()` walks history |
| 04 | [Pin user preferences above the surface](04_pin_preferences.py) | `depth = +∞` for stable facts |
| 05 | [Time-travel for debugging](05_time_travel_debug.py) | `recall(at=T)` reconstructs past belief |

All recipes use `hash_embed` (a deterministic mock embedder shipped
with Lethe) so they run without installing fastembed.  In production,
swap in `fastembed_local()` or any callable `(str) → list[float]` —
see the main [README](../README.md#quickstart) for setup.
