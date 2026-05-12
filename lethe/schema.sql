-- Lethe v1 schema. ONE physical axis: depth ∈ ℝ.
-- depth = 1.0   just inscribed, on the surface
-- depth ∈ (0,1) sinking under gravity (Hypnos.flow)
-- depth = 0     submerged (released) — present but unreachable
-- depth > 1     pinned by mnemosyne or promoted

-- Primary record. All status collapses into `depth`.
CREATE TABLE IF NOT EXISTS memory (
    rowid           INTEGER PRIMARY KEY,
    text            TEXT,
    depth           REAL DEFAULT 1.0,
    created_at      REAL,
    last_access     REAL,
    access_count    INTEGER DEFAULT 0,
    meta            TEXT
);

CREATE INDEX IF NOT EXISTS idx_memory_depth ON memory(depth);

-- Vector index. rowid is mirrored from memory.rowid (app-synced).
-- vec0 in 0.1.9 doesn't allow free metadata columns reliably across versions,
-- so we keep it minimal and join on rowid.
CREATE VIRTUAL TABLE IF NOT EXISTS memory_vec USING vec0(
    embedding FLOAT[{vector_dim}]
);

-- Lexical index. memory_id == memory.rowid (app-synced).
-- Not contentless because we want DELETE to work directly.
CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    text,
    tokenize='porter unicode61'
);

-- Append-only event log: drives time-travel, ForgetEval, crypto receipts.
CREATE TABLE IF NOT EXISTS event (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id       INTEGER NOT NULL,
    kind            TEXT NOT NULL,
    depth_before    REAL,
    depth_after     REAL,
    timestamp       REAL NOT NULL,
    meta            TEXT
);

CREATE INDEX IF NOT EXISTS idx_event_memory ON event(memory_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_event_kind   ON event(kind, timestamp);
CREATE INDEX IF NOT EXISTS idx_event_time   ON event(timestamp);

-- Bianhua: supersession edges (new takes the place of old).
CREATE TABLE IF NOT EXISTS supersession (
    new_id          INTEGER NOT NULL,
    old_id          INTEGER NOT NULL,
    timestamp       REAL NOT NULL,
    reason          TEXT,
    PRIMARY KEY (new_id, old_id)
);

CREATE INDEX IF NOT EXISTS idx_supersession_new ON supersession(new_id);
CREATE INDEX IF NOT EXISTS idx_supersession_old ON supersession(old_id);
