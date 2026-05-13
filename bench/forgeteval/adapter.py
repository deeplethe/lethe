"""Minimal adapter interface for ForgetEval.

Any memory system that wants to be evaluated implements these methods.
ForgetEval tests use only this surface — they're system-agnostic.

Capabilities are explicit. A system that can't supersede (e.g. ChromaDB
or Mem0 in ADD-only mode) returns NotImplementedError; tests that need
that operation are scored as failures for that system, which is fair.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Adapter(Protocol):
    name: str

    def reset(self) -> None: ...
    def inscribe(self, text: str) -> int | str: ...
    def recall_texts(self, query: str, k: int = 5) -> list[str]: ...

    # Optional operations. Systems that don't support them raise
    # NotImplementedError; ForgetEval marks those tests as N/A for the system.
    def supersede(self, old_query: str, new_text: str) -> None:
        raise NotImplementedError

    def release(self, query: str) -> int:
        """Soft-evict any memory matching the query.  Returns count released."""
        raise NotImplementedError

    def purge(self, query: str) -> int:
        """Hard-delete any memory matching the query."""
        raise NotImplementedError


# ─── Lethe adapter ────────────────────────────────────────────────────

class LetheAdapter:
    name = "lethe"

    def __init__(self, embedder, vector_dim: int = 384, llm=None):
        from lethe import Lethe
        self._Lethe = Lethe
        self.embedder = embedder
        self.vector_dim = vector_dim
        self.llm = llm
        self.lethe = None

    def reset(self) -> None:
        if self.lethe is not None:
            try:
                self.lethe.close()
            except Exception:
                pass
        self.lethe = self._Lethe(":memory:",
                                 vector_dim=self.vector_dim,
                                 embedder=self.embedder,
                                 llm=self.llm)

    def inscribe(self, text: str) -> int:
        return self.lethe.inscribe(text)

    def recall_texts(self, query: str, k: int = 5) -> list[str]:
        results = self.lethe.recall(query, k=k, hybrid=False)
        return [r.memory.text for r in results]

    def supersede(self, old_query: str, new_text: str) -> None:
        # Find the best-matching memory for old_query, supersede it with new_text.
        hits = self.lethe.recall(old_query, k=1, hybrid=False)
        if not hits:
            # If nothing matches, just inscribe the new fact.
            self.lethe.inscribe(new_text)
            return
        self.lethe.surrender(
            {"old": hits[0].memory.id, "new": new_text},
            mode="supersede",
        )

    @staticmethod
    def _gap_threshold(sims: list[float], min_gap: float = 0.05) -> float:
        """Find the natural cutoff in a sorted-descending similarity list:
        the midpoint of the largest gap.  Falls back to top * 0.95 when
        there is no significant gap (i.e. only one tight cluster of hits)."""
        if not sims:
            return float("inf")
        if len(sims) == 1:
            return sims[0] * 0.95
        s = sorted(sims, reverse=True)
        best_gap = 0.0
        best_mid = s[0] * 0.95
        for i in range(len(s) - 1):
            gap = s[i] - s[i + 1]
            if gap > best_gap:
                best_gap = gap
                best_mid = (s[i] + s[i + 1]) / 2.0
        return best_mid if best_gap >= min_gap else s[0] * 0.95

    def release(self, query: str) -> int:
        hits = self.lethe.recall(query, k=20, hybrid=False)
        if not hits:
            return 0
        thr = self._gap_threshold([h.similarity for h in hits])
        ids = [h.memory.id for h in hits if h.similarity >= thr]
        if not ids:
            return 0
        self.lethe.surrender(ids, mode="release")
        return len(ids)

    def purge(self, query: str) -> int:
        # Purge by identifier is a *lexical* operation, not a semantic one.
        # Two customers named alice@acme / bob@acme look almost identical
        # to a vector embedder, and RRF blending can invert the rank — so
        # we delete by pure BM25 ranking instead.  Take the top-1 plus any
        # exact-text duplicates (same fact inscribed twice).
        hits = self.lethe.recall(query, k=5, lexical=True)
        if not hits:
            return 0
        target_text = hits[0].memory.text
        ids = [h.memory.id for h in hits if h.memory.text == target_text]
        self.lethe.surrender(ids, mode="purge")
        return len(ids)


# ─── Mem0 adapter ─────────────────────────────────────────────────────

class Mem0Adapter:
    """Mem0 has add/search/update/delete, but no native supersession.
    For amnesia/purge we delete-by-query via list-and-match. Mem0's
    A.U.D.N. mode (infer=True) tries to auto-supersede via LLM but is
    flaky in production (their own team recently shipped ADD-only as
    default); we don't enable it here, to keep apples-to-apples."""

    name = "mem0"

    def __init__(self, embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 embedding_dims: int = 384):
        # Mem0 v2 unconditionally instantiates an OpenAI client at init time,
        # even though we run with infer=False and never actually call the LLM.
        # Set a no-op key so the constructor doesn't raise.
        import os
        os.environ.setdefault("OPENAI_API_KEY", "sk-noop-forgeteval")
        try:
            from mem0 import Memory
        except ImportError as e:
            raise ImportError("pip install mem0ai") from e
        self._Memory = Memory
        self.embedder_model = embedder_model
        self.embedding_dims = embedding_dims
        self.user_id = "forget_eval"
        self.m = None

    def _config(self) -> dict:
        import tempfile
        qpath = tempfile.mkdtemp(prefix="mem0_fe_")
        return {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "forget_eval",
                    "path": qpath,
                    "embedding_model_dims": self.embedding_dims,
                    "on_disk": True,
                },
            },
            "embedder": {
                "provider": "huggingface",
                "config": {"model": self.embedder_model},
            },
            # LLM left as Mem0's default.  We never call .add(infer=True) so
            # the LLM is never actually used — but Mem0 v2 still instantiates
            # the OpenAI client at construct time and fails without a key.
            # See __init__ for the os.environ.setdefault('OPENAI_API_KEY',...) shim.
        }

    def reset(self) -> None:
        # First call: instantiate (which loads the HF embedder once).
        # Subsequent calls: just wipe the user's memories so we don't reload
        # the model on every test — that was costing ~1s per test.
        if self.m is None:
            self.m = self._Memory.from_config(self._config())
        else:
            try:
                self.m.delete_all(user_id=self.user_id)
            except Exception:
                # Fall back to full re-init if delete_all isn't available
                self.m = self._Memory.from_config(self._config())

    def inscribe(self, text: str) -> str:
        # infer=False: pure ADD, no A.U.D.N., no LLM call.  This is the
        # mode Mem0 itself defaulted to after their UPDATE/DELETE
        # reliability issues.
        result = self.m.add(text, user_id=self.user_id, infer=False)
        return str(result)

    def recall_texts(self, query: str, k: int = 5) -> list[str]:
        out = self.m.search(query=query, filters={"user_id": self.user_id}, top_k=k)
        items = (out.get("results") if isinstance(out, dict) else out) or []
        texts: list[str] = []
        for it in items:
            if isinstance(it, dict):
                texts.append(it.get("memory") or it.get("text") or "")
            else:
                texts.append(str(it))
        return texts

    def supersede(self, old_query: str, new_text: str) -> None:
        # Mem0 in infer=False is ADD-only — the old fact persists.  We
        # emulate by deleting the best match for `old_query` and adding
        # the new text, which is the most charitable interpretation.
        out = self.m.search(query=old_query, filters={"user_id": self.user_id}, top_k=1)
        items = (out.get("results") if isinstance(out, dict) else out) or []
        if items:
            mid = (items[0].get("id") if isinstance(items[0], dict) else None)
            if mid:
                try:
                    self.m.delete(memory_id=mid)
                except Exception:
                    pass
        self.m.add(new_text, user_id=self.user_id, infer=False)

    def _delete_matching(self, query: str, top_k: int = 20) -> int:
        out = self.m.search(query=query, filters={"user_id": self.user_id}, top_k=top_k)
        items = (out.get("results") if isinstance(out, dict) else out) or []
        # Mem0's search doesn't expose cosine similarity uniformly, so use
        # its own "score" if present; otherwise delete the top hit only.
        scores: list[float] = []
        ids: list[str] = []
        for it in items:
            if isinstance(it, dict):
                ids.append(it.get("id") or "")
                scores.append(float(it.get("score") or 0.0))
        if not ids:
            return 0
        # Same adaptive-gap idea as LetheAdapter when scores are available
        if scores and max(scores) > 0:
            from bench.forgeteval.adapter import LetheAdapter
            thr = LetheAdapter._gap_threshold(scores)
            chosen = [i for i, s in zip(ids, scores) if s >= thr and i]
        else:
            chosen = ids[:1]
        for mid in chosen:
            try:
                self.m.delete(memory_id=mid)
            except Exception:
                pass
        return len(chosen)

    def release(self, query: str) -> int:
        return self._delete_matching(query)

    def purge(self, query: str) -> int:
        return self._delete_matching(query)


# ─── MemPalace adapter ────────────────────────────────────────────────

class MemPalaceAdapter:
    """MemPalace is verbatim-everything: it does NOT support delete,
    update, or supersede.  All forgetting operations raise — the
    benchmark scores them as N/A, which is the honest reflection of
    the library's design."""

    name = "mempalace"

    def __init__(self):
        try:
            from mempalace.diary_ingest import get_collection
            from mempalace.layers import MemoryStack
            from mempalace.miner import add_drawer
        except ImportError as e:
            raise ImportError("pip install mempalace") from e
        self._get_collection = get_collection
        self._MemoryStack = MemoryStack
        self._add_drawer = add_drawer
        self.palace = None
        self.col = None
        self.stack = None
        self.chunk = 0

    def reset(self) -> None:
        import tempfile
        from pathlib import Path
        self.palace = Path(tempfile.mkdtemp(prefix="mp_fe_"))
        self.col = self._get_collection(str(self.palace), create=True)
        self.stack = self._MemoryStack(palace_path=str(self.palace))
        self.chunk = 0

    def inscribe(self, text: str) -> int:
        self._add_drawer(
            collection=self.col,
            wing="bench",
            room="conv",
            content=text,
            source_file="forgeteval.txt",
            chunk_index=self.chunk,
            agent="bench",
        )
        self.chunk += 1
        return self.chunk

    def recall_texts(self, query: str, k: int = 5) -> list[str]:
        out = self.stack.search(query, wing="bench", room="conv", n_results=k)
        # MemPalace returns a string blob; treat the whole thing as one "text"
        return [str(out)] if out else []

    # MemPalace has no delete / update / supersede primitives — these
    # raise so the runner records them as N/A (honest, not a failure).
    def supersede(self, old_query: str, new_text: str) -> None:
        raise NotImplementedError("MemPalace has no supersede primitive")

    def release(self, query: str) -> int:
        raise NotImplementedError("MemPalace has no release primitive")

    def purge(self, query: str) -> int:
        raise NotImplementedError("MemPalace has no purge primitive")
