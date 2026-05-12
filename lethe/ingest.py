"""Batch ingestion: inscribe every paragraph of every matching file.

Verbatim by design — no summarization, no smart-splitting, no
LLM-driven chunking.  Lethe stores text as-is and lets recall sort it
out.  The only structural decision is paragraph boundary: chunks split
on blank lines (\\n\\n).  Anything below `min_len` chars is discarded
as noise (page numbers, single-word headings).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional


DEFAULT_PATTERNS = ("*.md", "*.txt", "*.rst")
DEFAULT_MIN_LEN = 16
DEFAULT_BATCH = 256


def iter_files(root: Path, patterns: Iterable[str]) -> Iterator[Path]:
    """Recursively walk `root` and yield each file matching any pattern.
    Each file is yielded at most once even if multiple patterns match."""
    seen: set[Path] = set()
    for pat in patterns:
        for p in root.rglob(pat):
            if p.is_file() and p not in seen:
                seen.add(p)
                yield p


def chunk_paragraphs(text: str, *, min_len: int = DEFAULT_MIN_LEN) -> Iterator[str]:
    """Split on blank-line boundaries; trim and drop anything too short."""
    for chunk in text.split("\n\n"):
        c = chunk.strip()
        if len(c) >= min_len:
            yield c


def ingest(
    lethe,
    paths: Iterable[Path],
    *,
    encoding: str = "utf-8",
    min_len: int = DEFAULT_MIN_LEN,
    batch: int = DEFAULT_BATCH,
    on_progress: Optional[Callable[[dict], None]] = None,
) -> dict:
    """Inscribe paragraph-chunks from every readable file in `paths`.

    Streams in batches of `batch` chunks so memory stays bounded even
    on large corpora.  Files that fail to decode under `encoding` are
    skipped (logged via on_progress, not raised).
    """
    started = time.perf_counter()
    stats = {"files_scanned": 0, "files_skipped": 0,
             "chunks": 0, "wall_seconds": 0.0}

    pending: list[str] = []
    pending_meta: list[Optional[dict]] = []

    def flush():
        if not pending:
            return
        lethe.inscribe_many(pending, metas=pending_meta)
        stats["chunks"] += len(pending)
        pending.clear()
        pending_meta.clear()
        if on_progress:
            on_progress(dict(stats))

    for path in paths:
        try:
            text = path.read_text(encoding=encoding)
        except (UnicodeDecodeError, OSError):
            stats["files_skipped"] += 1
            continue
        stats["files_scanned"] += 1
        rel = str(path)
        for chunk in chunk_paragraphs(text, min_len=min_len):
            pending.append(chunk)
            pending_meta.append({"source": rel})
            if len(pending) >= batch:
                flush()

    flush()
    stats["wall_seconds"] = round(time.perf_counter() - started, 3)
    return stats
