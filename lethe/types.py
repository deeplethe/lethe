"""Lethe v1 data types. One physical axis: depth."""
from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from typing import Optional


# Depth landmarks
SURFACE = 1.0      # just inscribed
SUBMERGED = 0.0    # released / evicted — present but unreachable
PINNED = float("inf")     # mnemosyne — immune to gravity


@dataclass
class Memory:
    """A fact in the river. Identified by integer rowid (vec0 native)."""
    id: int
    text: str
    depth: float = SURFACE
    created_at: float = field(default_factory=time)
    last_access: float = field(default_factory=time)
    access_count: int = 0
    embedding: Optional[list[float]] = None
    meta: dict = field(default_factory=dict)

    @property
    def alive(self) -> bool:
        return self.depth > SUBMERGED

    @property
    def pinned(self) -> bool:
        return self.depth >= PINNED


@dataclass
class Recalled:
    memory: Memory
    score: float
    similarity: float = 0.0                 # raw cosine ∈ [-1, 1] (vec leg only)
    via: Optional[list[str]] = None         # diagnostic: which signals fired


@dataclass
class Event:
    """A depth change in the river. Append-only log."""
    id: int
    memory_id: int
    kind: str                # inscribe | recall | decay | release | supersede | mnemosyne | purge | flow
    depth_before: Optional[float]
    depth_after: Optional[float]
    timestamp: float
    meta: dict = field(default_factory=dict)


@dataclass
class BlameEntry:
    """For Lethe.blame(query) — show a fact's supersession history."""
    memory: Memory
    superseded_by: Optional[int] = None
    superseded_at: Optional[float] = None
    supersedes: list[int] = field(default_factory=list)
