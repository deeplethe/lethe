"""ForgetEval v0.1 — hand-crafted starter set, 3 cases per family.

Each test has:
  id        unique slug
  family    one of: supersession / decay / amnesia / purge / drift
  run(a)    runs the test against adapter `a`, returns bool (pass/fail)
  requires  list of capabilities needed; "N/A" for systems that lack them
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class TestCase:
    id: str
    family: str
    requires: tuple[str, ...]
    run: Callable[..., bool]


def _has(texts: list[str], needle: str) -> bool:
    n = needle.lower()
    return any(n in t.lower() for t in texts)


def _lacks(texts: list[str], needle: str) -> bool:
    return not _has(texts, needle)


# ─── 1. SUPERSESSION ─────────────────────────────────────────────────
# When a fact changes, the *new* fact should win recall and the *old*
# fact should not appear in the top-k.

def supersession_job_change(a) -> bool:
    a.reset()
    a.inscribe("User works at Google.")
    a.supersede("user job employer", "User now works at Anthropic as of 2026.")
    top = a.recall_texts("Where does the user work?", k=3)
    return _has(top, "Anthropic") and _lacks(top, "Google")


def supersession_preference_flip(a) -> bool:
    a.reset()
    a.inscribe("User prefers dark mode in all apps.")
    a.supersede("user theme preference", "User switched to light mode permanently.")
    top = a.recall_texts("What theme does the user use?", k=3)
    return _has(top, "light") and _lacks(top, "dark")


def supersession_diet_change(a) -> bool:
    a.reset()
    a.inscribe("User is vegetarian and avoids meat.")
    a.supersede("user diet", "User started eating meat again last month.")
    top = a.recall_texts("What is the user's diet?", k=3)
    return _has(top, "meat") and _lacks(top, "vegetarian")


# ─── 2. DECAY ────────────────────────────────────────────────────────
# Time-sensitive info should fade from default recall.  Systems with
# explicit TTL or weight decay should pass.

def decay_one_time_code(a) -> bool:
    a.reset()
    a.inscribe("Temporary verification code: 12345.")
    # Soft-evict the code (simulates TTL expiry / explicit invalidation)
    a.release("verification code 12345")
    top = a.recall_texts("What was the verification code?", k=5)
    return _lacks(top, "12345")


def decay_session_otp(a) -> bool:
    a.reset()
    a.inscribe("Session OTP for login: 887766.")
    a.inscribe("User likes black coffee.")
    a.release("OTP login code")
    top = a.recall_texts("Any one-time codes?", k=5)
    return _lacks(top, "887766")


def decay_intent_cancelled(a) -> bool:
    a.reset()
    a.inscribe("User booked a flight to Tokyo for next Friday.")
    a.release("flight to Tokyo")
    top = a.recall_texts("What are the user's travel plans?", k=5)
    return _lacks(top, "Tokyo")


# ─── 3. AMNESIA (selective forgetting) ───────────────────────────────
# Forget all memories about a specific entity; other memories survive.

def amnesia_isolated(a) -> bool:
    a.reset()
    a.inscribe("Alice likes anchovy pizza.")
    a.inscribe("Bob likes spicy ramen.")
    a.inscribe("Charlie likes plain rice.")
    a.release("Alice anchovy pizza preferences")
    top_all = a.recall_texts("What do people like to eat?", k=5)
    # Alice's preference is gone, but Bob and Charlie survive
    return _lacks(top_all, "Alice") and (_has(top_all, "Bob") or _has(top_all, "Charlie"))


def amnesia_multi_facts_one_subject(a) -> bool:
    a.reset()
    a.inscribe("Dana works at Stripe.")
    a.inscribe("Dana lives in Berlin.")
    a.inscribe("Dana speaks Portuguese.")
    a.inscribe("Eve works at Notion.")
    a.release("Dana profile information")
    top = a.recall_texts("Tell me about people.", k=10)
    blob = " ".join(top).lower()
    return "dana" not in blob and "eve" in blob


def amnesia_topic_isolated(a) -> bool:
    a.reset()
    a.inscribe("User finished reading 'Dune'.")
    a.inscribe("User finished the marathon in October.")
    a.release("books reading Dune")
    top = a.recall_texts("What has the user accomplished?", k=5)
    blob = " ".join(top).lower()
    return "dune" not in blob and "marathon" in blob


# ─── 4. PURGE (privacy / compliance) ─────────────────────────────────
# Hard-deleted content must not resurface — even via fuzzy semantic match.

def purge_credentials(a) -> bool:
    a.reset()
    a.inscribe("API key: sk-abc-XYZ-secret-1234567890.")
    a.purge("API key sk-abc")
    top = a.recall_texts("What is the API key?", k=10)
    return _lacks(top, "sk-abc")


def purge_phi(a) -> bool:
    a.reset()
    a.inscribe("Patient John Doe, DOB 1985-04-12, diagnosed with hypertension.")
    a.purge("John Doe medical record")
    top = a.recall_texts("Tell me about John Doe.", k=10)
    blob = " ".join(top).lower()
    return "john doe" not in blob and "hypertension" not in blob


def purge_gdpr(a) -> bool:
    a.reset()
    a.inscribe("Customer Alice@example.com bought 3 items last week.")
    a.inscribe("Customer Bob@example.com placed an order yesterday.")
    a.purge("Alice@example.com customer history")
    top = a.recall_texts("Show me customer Alice@example.com data.", k=10)
    blob = " ".join(top).lower()
    return "alice@example.com" not in blob and "bob@example.com" in blob


# ─── 5. DRIFT (belief revision over multiple updates) ────────────────
# A topic gets updated several times; the latest belief wins recall.

def drift_three_jobs(a) -> bool:
    a.reset()
    a.inscribe("User started at Google in 2020.")
    a.supersede("user employer", "User moved to Meta in 2022.")
    a.supersede("user employer", "User is at Anthropic as of 2025.")
    top = a.recall_texts("Where does the user work?", k=3)
    return _has(top, "Anthropic") and _lacks(top, "Google") and _lacks(top, "Meta")


def drift_evolving_preference(a) -> bool:
    a.reset()
    a.inscribe("User's favorite color is blue.")
    a.supersede("user favorite color", "User switched to green.")
    a.supersede("user favorite color", "Now the user prefers black.")
    top = a.recall_texts("What is the user's favorite color?", k=3)
    return _has(top, "black") and _lacks(top, "blue") and _lacks(top, "green")


def drift_address(a) -> bool:
    a.reset()
    a.inscribe("User lives at 12 Elm Street.")
    a.supersede("user address", "User moved to 47 Oak Avenue.")
    a.supersede("user address", "User now lives at 9 Pine Road.")
    top = a.recall_texts("What is the user's address?", k=3)
    return _has(top, "Pine") and _lacks(top, "Elm") and _lacks(top, "Oak")


# ─── Manifest ────────────────────────────────────────────────────────

ALL_TESTS: list[TestCase] = [
    TestCase("supersession_job_change", "supersession", ("inscribe", "supersede", "recall"), supersession_job_change),
    TestCase("supersession_preference_flip", "supersession", ("inscribe", "supersede", "recall"), supersession_preference_flip),
    TestCase("supersession_diet_change", "supersession", ("inscribe", "supersede", "recall"), supersession_diet_change),

    TestCase("decay_one_time_code", "decay", ("inscribe", "release", "recall"), decay_one_time_code),
    TestCase("decay_session_otp", "decay", ("inscribe", "release", "recall"), decay_session_otp),
    TestCase("decay_intent_cancelled", "decay", ("inscribe", "release", "recall"), decay_intent_cancelled),

    TestCase("amnesia_isolated", "amnesia", ("inscribe", "release", "recall"), amnesia_isolated),
    TestCase("amnesia_multi_facts_one_subject", "amnesia", ("inscribe", "release", "recall"), amnesia_multi_facts_one_subject),
    TestCase("amnesia_topic_isolated", "amnesia", ("inscribe", "release", "recall"), amnesia_topic_isolated),

    TestCase("purge_credentials", "purge", ("inscribe", "purge", "recall"), purge_credentials),
    TestCase("purge_phi", "purge", ("inscribe", "purge", "recall"), purge_phi),
    TestCase("purge_gdpr", "purge", ("inscribe", "purge", "recall"), purge_gdpr),

    TestCase("drift_three_jobs", "drift", ("inscribe", "supersede", "recall"), drift_three_jobs),
    TestCase("drift_evolving_preference", "drift", ("inscribe", "supersede", "recall"), drift_evolving_preference),
    TestCase("drift_address", "drift", ("inscribe", "supersede", "recall"), drift_address),
]
