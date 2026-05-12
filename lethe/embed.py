"""Embedders. hash_embed for tests. Local + remote helpers."""
from __future__ import annotations

import hashlib
import math
from typing import Callable


def hash_embed(text: str, dim: int = 768) -> list[float]:
    """Deterministic mock embedder for tests. SHA256 tiled across dims."""
    seed = hashlib.sha256(text.encode("utf-8")).digest()
    raw: list[float] = []
    i = 0
    while len(raw) < dim:
        chunk = hashlib.sha256(seed + i.to_bytes(4, "big")).digest()
        for b in chunk:
            raw.append((b / 255.0) * 2.0 - 1.0)
            if len(raw) == dim:
                break
        i += 1
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


def fastembed_local(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Callable[[str], list[float]]:
    """Local embedder via fastembed (ONNX runtime). 0 API."""
    try:
        from fastembed import TextEmbedding
    except ImportError as e:
        raise ImportError("pip install 'lethe[embed]'") from e
    model = TextEmbedding(model_name)

    def embed(text: str) -> list[float]:
        return list(next(iter(model.embed([text]))))

    return embed
