"""
Lore Memory — the public API.

Simple interface wrapping the Engine for end-user consumption.
All internals hidden behind clean methods with docstrings and type hints.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from lore_memory.engine import Engine, Config, WriteResult
from lore_memory.retrieval import SearchResult


class MemoryResult:
    """A single memory retrieval result."""

    __slots__ = ("id", "text", "subject", "predicate", "object", "confidence",
                 "is_negation", "score", "scope", "created_at")

    def __init__(self, sr: SearchResult):
        self.id: str = sr.memory.id
        self.text: str = sr.memory.source_text or sr.memory.object_value
        self.subject: str = sr.memory.subject
        self.predicate: str = sr.memory.predicate
        self.object: str = sr.memory.object_value
        self.confidence: float = round(sr.memory.posterior, 3)
        self.is_negation: bool = sr.memory.is_negation
        self.score: float = round(sr.score, 4)
        self.scope: str = sr.source_scope
        self.created_at: float = sr.memory.created_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "text": self.text, "subject": self.subject,
            "predicate": self.predicate, "object": self.object,
            "confidence": self.confidence, "is_negation": self.is_negation,
            "score": self.score, "scope": self.scope,
        }

    def __repr__(self) -> str:
        return f"MemoryResult({self.predicate}={self.object}, conf={self.confidence})"


class Memory:
    """Persistent AI memory that learns from every interaction.

    Args:
        user_id: Identifier for the user. Defaults to "default".
        org_id: Organization for shared memories. Optional.
        data_dir: Where to store data. Defaults to ~/.lore-memory.
        embedding_dims: Embedding vector size. Defaults to 64 (hash).
        embed_fn: Custom embedding function. Defaults to hash embeddings.
            Pass a real model for better accuracy:
            ``embed_fn=SentenceTransformer('all-MiniLM-L6-v2').encode``

    Examples:
        >>> m = Memory()
        >>> m.store("I work at Google")
        >>> m.query("where do I work?")
        [MemoryResult(works_at=Google, conf=0.867)]
    """

    def __init__(
        self,
        user_id: str = "default",
        org_id: str = "",
        data_dir: str | Path | None = None,
        embedding_dims: int = 64,
        embed_fn: Callable | None = None,
    ) -> None:
        if data_dir is None:
            data_dir = str(Path.home() / ".lore-memory")
        self._user_id = user_id
        self._org_id = org_id
        self._engine = Engine(Config(data_dir=str(data_dir),
                                     embedding_dims=embedding_dims))
        if embed_fn is not None:
            self._engine._embed = embed_fn
            self._engine._retriever.embed = embed_fn

    # --- Core API ---

    def store(self, text: str, scope: str = "private") -> dict[str, Any]:
        """Store a memory from natural text.

        Automatically extracts facts using grammar-based parsing.
        Raw text is always stored and searchable via FTS.

        Args:
            text: Natural language text to remember.
            scope: "private" (user-only) or "shared" (org-visible).

        Returns:
            Dict with keys: created, deduplicated, contradictions.
        """
        if scope == "shared" and self._org_id:
            r = self._engine.store_company(self._user_id, self._org_id, text)
        else:
            r = self._engine.store_personal(self._user_id, text)
        return {"created": r.created, "deduplicated": r.deduplicated,
                "contradictions": r.contradictions}

    def query(self, query: str, limit: int = 10) -> list[MemoryResult]:
        """Query memory using natural language.

        Uses 7-channel retrieval: semantic, keyword, temporal, belief,
        frequency, graph (spreading activation), and resonance.

        Args:
            query: Natural language question or keywords.
            limit: Maximum results to return.

        Returns:
            List of MemoryResult objects, ranked by relevance.
        """
        results = self._engine.recall(
            self._user_id, query, org_id=self._org_id, top_k=limit)
        return [MemoryResult(r) for r in results]

    def forget(self, memory_id: str | None = None,
               subject: str | None = None,
               predicate: str | None = None) -> bool:
        """Remove a memory or category of memories.

        Args:
            memory_id: Specific memory ID to delete.
            subject: Delete all memories about this subject.
            predicate: Combined with subject, delete specific predicate.

        Returns:
            True if any memories were deleted.
        """
        if memory_id:
            db, mem = self._engine._resolve_db(
                memory_id, scope_hint="private", user_id=self._user_id)
            if db and mem:
                db.update_state(memory_id, "deleted")
                # Also delete sibling memories from the same source text
                # (raw text + extracted facts are stored together)
                if mem.source_text:
                    siblings = db.query_by_subject(mem.subject)
                    for sib in siblings:
                        if sib.id != memory_id and sib.source_text == mem.source_text:
                            db.update_state(sib.id, "deleted")
                return True
            return False

        if subject:
            from lore_memory.scopes import Scope
            db = self._engine._db(Scope.PRIVATE, user_id=self._user_id)
            mems = db.query_by_subject(subject)
            deleted = 0
            for m in mems:
                if predicate and m.predicate != predicate:
                    continue
                db.update_state(m.id, "deleted")
                deleted += 1
            return deleted > 0

        return False

    def forget_all(self) -> bool:
        """Delete ALL memories for this user. Irreversible."""
        return self._engine.purge_user(self._user_id)

    # --- Advanced API ---

    def store_triple(self, subject: str, predicate: str, object: str,
                     confidence: float = 0.85, scope: str = "private") -> dict:
        """Store a structured fact directly as a subject-predicate-object triple.

        Args:
            subject: Who/what the fact is about.
            predicate: The relationship (e.g., "works_at", "likes").
            object: The value.
            confidence: How confident (0.0 to 1.0).
            scope: "private" or "shared".
        """
        ctx = "company" if scope == "shared" else "personal"
        r = self._engine.store_fact(
            scope=scope, context=ctx, subject=subject, predicate=predicate,
            object_value=object, user_id=self._user_id,
            org_id=self._org_id if scope == "shared" else "",
            confidence=confidence)
        return {"created": r.created, "contradictions": r.contradictions}

    def profile(self) -> dict[str, list[dict]]:
        """Get the user's profile — all known facts organized by predicate.

        Returns:
            Dict mapping predicate names to lists of values with confidence.
        """
        return self._engine.profile(self._user_id, self._user_id,
                                    org_id=self._org_id)

    def profile_compact(self, max_tokens: int = 200) -> str:
        """Get a token-budgeted summary for LLM context injection.

        Args:
            max_tokens: Approximate token budget.

        Returns:
            Compact string of top facts, suitable for system prompts.
        """
        return self._engine.profile_compact(self._user_id, max_tokens)

    def stats(self) -> dict[str, int]:
        """Return memory statistics.

        Returns:
            Dict with counts: private_total, private_personal, private_chat, etc.
        """
        return self._engine.stats(self._user_id, org_id=self._org_id)

    def feedback(self, memory_id: str, helpful: bool) -> bool:
        """Give feedback on a memory result.

        This drives adaptive learning — channel weights shift based on
        which types of matches are helpful.

        Args:
            memory_id: The ID from a MemoryResult.
            helpful: True if the result was relevant, False if not.
        """
        return self._engine.feedback(self._user_id, memory_id, helpful)

    def consolidate(self) -> dict:
        """Run memory maintenance: decay old facts, replay traces, archive junk.

        Should be called periodically (e.g., daily) for long-running systems.
        """
        return self._engine.consolidate(self._user_id, org_id=self._org_id)

    def close(self) -> None:
        """Close the memory engine and persist state to disk."""
        self._engine.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self) -> str:
        s = self.stats()
        return f"Memory(user={self._user_id}, facts={s.get('private_total', 0)})"
