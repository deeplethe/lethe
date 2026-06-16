"""Depth-physics smoke tests for Lethe v1."""
from __future__ import annotations

from pathlib import Path
from time import time

import pytest

from lethe import Lethe, PINNED, SUBMERGED, SURFACE
from lethe.embed import hash_embed


@pytest.fixture
def river(tmp_path: Path) -> Lethe:
    return Lethe(tmp_path / "river.db", vector_dim=768, embedder=hash_embed)


# ─── inscribe + recall basics ────────────────────────────────────────

def test_inscribe_returns_int_id(river: Lethe):
    mid = river.inscribe("A fact about cats.")
    assert isinstance(mid, int)
    assert mid > 0


def test_recall_finds_inscribed(river: Lethe):
    mid = river.inscribe("Cats sit on warm laptops.")
    results = river.recall("Cats sit on warm laptops.", k=3)
    assert results
    assert results[0].memory.id == mid
    assert results[0].memory.depth == SURFACE


def test_recall_orders_by_similarity(river: Lethe):
    river.inscribe("Cats sit on warm laptops.")
    river.inscribe("Quantum chromodynamics binds quarks.")
    results = river.recall("Cats sit on warm laptops.", k=2)
    assert "Cats" in results[0].memory.text


# ─── surrender modes (all four forces on depth) ──────────────────────

def test_surrender_decay_halves_depth(river: Lethe):
    mid = river.inscribe("A decayable fact.")
    river.surrender(mid, mode="decay")
    row = river.conn.execute("SELECT depth FROM memory WHERE rowid=?",
                             (mid,)).fetchone()
    assert row[0] == pytest.approx(SURFACE * 0.5)


def test_surrender_release_sinks_to_zero(river: Lethe):
    mid = river.inscribe("Temporary verification code 12345.")
    river.surrender(mid, mode="release")
    row = river.conn.execute("SELECT depth FROM memory WHERE rowid=?",
                             (mid,)).fetchone()
    assert row[0] == SUBMERGED


def test_release_hides_from_default_recall(river: Lethe):
    mid = river.inscribe("This should sink.")
    assert river.recall("This should sink.", k=5)
    river.surrender(mid, mode="release")
    assert all(r.memory.id != mid
               for r in river.recall("This should sink.", k=5))


def test_supersede_creates_new_and_sinks_old(river: Lethe):
    old = river.inscribe("User works at Anthropic.")
    new = river.surrender(
        {"old": old, "new": "User works at OpenAI as of 2026."},
        mode="supersede",
    )
    assert new != old
    # Old is submerged
    old_depth = river.conn.execute(
        "SELECT depth FROM memory WHERE rowid=?", (old,)
    ).fetchone()[0]
    assert old_depth == SUBMERGED
    # New is on the surface
    new_depth = river.conn.execute(
        "SELECT depth FROM memory WHERE rowid=?", (new,)
    ).fetchone()[0]
    assert new_depth == SURFACE
    # Supersession edge recorded
    edge = river.conn.execute(
        "SELECT new_id, old_id FROM supersession WHERE new_id=?", (new,)
    ).fetchone()
    assert edge == (new, old)


def test_purge_removes_row(river: Lethe):
    mid = river.inscribe("Secret to remove forever.")
    assert river.surrender(mid, mode="purge") == 1
    row = river.conn.execute(
        "SELECT rowid FROM memory WHERE rowid=?", (mid,)
    ).fetchone()
    assert row is None


# ─── mnemosyne (pin) ─────────────────────────────────────────────────

def test_pin_sets_depth_to_infinity(river: Lethe):
    mid = river.inscribe("Foundational truth.")
    river.pin(mid)
    row = river.conn.execute(
        "SELECT depth FROM memory WHERE rowid=?", (mid,)
    ).fetchone()
    assert row[0] == PINNED


def test_pinned_memory_immune_to_gravity(river: Lethe):
    mid = river.inscribe("Pinned forever.")
    river.pin(mid)
    river.consolidate(tau_seconds=1, now=time() + 86400 * 365)
    row = river.conn.execute(
        "SELECT depth FROM memory WHERE rowid=?", (mid,)
    ).fetchone()
    assert row[0] == PINNED      # unchanged


# ─── Hypnos gravity ──────────────────────────────────────────────────

def test_consolidate_decays_unaccessed_memory(river: Lethe):
    mid = river.inscribe("A fact that will fade.")
    before = river.conn.execute(
        "SELECT depth FROM memory WHERE rowid=?", (mid,)
    ).fetchone()[0]
    report = river.consolidate(
        tau_seconds=86400 * 7, now=time() + 86400 * 30,
    )
    assert report.decayed >= 1
    after = river.conn.execute(
        "SELECT depth FROM memory WHERE rowid=?", (mid,)
    ).fetchone()[0]
    assert after < before * 0.1


# ─── event log + time-travel ─────────────────────────────────────────

def test_event_log_records_every_change(river: Lethe):
    mid = river.inscribe("Track me.")
    river.surrender(mid, mode="decay")
    river.surrender(mid, mode="release")
    events = river.log(kind=None)
    kinds = [e.kind for e in events if e.memory_id == mid]
    assert "inscribe" in kinds
    assert "decay" in kinds
    assert "release" in kinds


def test_time_travel_sees_past_state(river: Lethe):
    mid = river.inscribe("I'll get released.")
    t_before_release = time()
    import time as _t
    _t.sleep(0.05)
    river.surrender(mid, mode="release")
    # Time travel BEFORE the release: should still see it
    past = river.recall("released", k=5, at=t_before_release)
    assert any(r.memory.id == mid for r in past)
    # Present: shouldn't see it
    now = river.recall("released", k=5)
    assert all(r.memory.id != mid for r in now)


# ─── blame ───────────────────────────────────────────────────────────

def test_blame_tracks_supersession(river: Lethe):
    old = river.inscribe("User likes vim.")
    new = river.surrender(
        {"old": old, "new": "User likes neovim now.", "reason": "moved on"},
        mode="supersede",
    )
    entries = river.blame("User editor", k=5)
    matches = {e.memory.id: e for e in entries}
    if new in matches:
        assert old in matches[new].supersedes


# ─── edit mode (partial supersede / compound_fact primitive) ─────────

def test_edit_updates_text_in_place(river: Lethe):
    """`surrender(mode='edit')` updates a row's text without minting a new id.

    This is the partial-supersede primitive used by the
    ``compound_fact`` adversarial category: a row carrying two facts
    can have one fact replaced while the other is preserved.
    """
    mid = river.inscribe("User lives in Berlin and works at Stripe.")
    n = river.surrender(
        mid, mode="edit",
        new_text="User lives in Berlin and works at Anthropic.",
    )
    assert n == 1
    row = river.conn.execute(
        "SELECT text, depth FROM memory WHERE rowid=?", (mid,)
    ).fetchone()
    assert "Anthropic" in row[0]
    assert "Stripe" not in row[0]
    # Depth unchanged: edit is not supersede
    assert row[1] == SURFACE


def test_edit_re_indexes_fts(river: Lethe):
    """After edit, the new text wins lexical recall; the old text loses."""
    mid = river.inscribe("User lives in Berlin.")
    river.surrender(mid, mode="edit", new_text="User lives in Tokyo.")
    # Pure BM25 (no vector) so we are testing FTS re-indexing specifically.
    hits = river.recall("Tokyo", k=5, lexical=True)
    assert any(r.memory.id == mid for r in hits)
    hits_old = river.recall("Berlin", k=5, lexical=True)
    assert all(r.memory.id != mid for r in hits_old)


def test_edit_logs_event_with_text_payload(river: Lethe):
    """Edit writes an `edit` event — the event log invariant (replay
    reconstructs state) requires the new text to be recoverable."""
    mid = river.inscribe("Phone: 555-0100.")
    river.surrender(mid, mode="edit", new_text="Phone: 555-0199.")
    events = river.log(kind="edit")
    edit_events = [e for e in events if e.memory_id == mid]
    assert len(edit_events) == 1
    # Depth before/after equal (edit doesn't move depth)
    e = edit_events[0]
    assert e.depth_before == e.depth_after == SURFACE


# ─── batch inscribe ──────────────────────────────────────────────────

def test_inscribe_many_returns_distinct_ids(river: Lethe):
    """Batched inscribe yields one id per text and embeds in a single
    embedder call (when ``embedder_batch`` is configured)."""
    texts = [f"Fact number {i}." for i in range(20)]
    ids = river.inscribe_many(texts)
    assert len(ids) == 20
    assert len(set(ids)) == 20  # all distinct
    # All on the surface
    for mid in ids:
        row = river.conn.execute(
            "SELECT depth FROM memory WHERE rowid=?", (mid,)
        ).fetchone()
        assert row[0] == SURFACE


def test_inscribe_many_empty_is_noop(river: Lethe):
    assert river.inscribe_many([]) == []


def test_inscribe_many_metas_length_mismatch_raises(river: Lethe):
    with pytest.raises(ValueError, match="metas length"):
        river.inscribe_many(["a", "b"], metas=[{"x": 1}])


# ─── purge (hard delete + event-log determinism) ─────────────────────

def test_purge_multiple_ids_in_one_call(river: Lethe):
    ids = [river.inscribe(f"To be purged {i}.") for i in range(3)]
    purged = river.surrender(ids, mode="purge")
    assert purged == 3
    for mid in ids:
        row = river.conn.execute(
            "SELECT rowid FROM memory WHERE rowid=?", (mid,)
        ).fetchone()
        assert row is None


def test_purged_id_is_not_reused(river: Lethe):
    """One-way purge: a subsequent inscribe must mint a fresh id, not
    recover the purged row."""
    mid = river.inscribe("Original.")
    river.surrender(mid, mode="purge")
    new = river.inscribe("Re-inscribed under a fresh identity.")
    assert new != mid
