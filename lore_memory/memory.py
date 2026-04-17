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
from lore_memory.schema import Schema, PERSONAL_LIFE_SCHEMA


# Minimum score for the top result to be considered a real answer.
# Below this the retriever has basically no signal — forcing
# needs_clarification=True prevents the "I returned something random"
# failure mode when there's no matching fact in memory.
_MIN_ANSWER_SCORE = 0.02


def _build_query_result(results: list, clarification_threshold: float) -> "QueryResult":
    """Compute certainty and clarification hint from a ranked result list."""
    if not results:
        return QueryResult(answer=None, certainty=0.0,
                           needs_clarification=False, alternatives=[])
    top1 = results[0]
    # If even the top result is weak, the system shouldn't pretend to
    # have an answer: flag it as unclear so consumers ask the user.
    if top1.score < _MIN_ANSWER_SCORE:
        return QueryResult(answer=top1, certainty=0.0,
                           needs_clarification=True,
                           alternatives=results[:5])
    if len(results) == 1:
        return QueryResult(answer=top1, certainty=1.0,
                           needs_clarification=False, alternatives=[top1])
    top2 = results[1]
    # Margin-based certainty: how much top-1 leads top-2 as a fraction
    # of top-1's score. Range [0, 1]. Guards against zero / negative
    # top-1 scores (can happen after heavy multiplicative penalties).
    denom = max(abs(top1.score), 1e-6)
    margin = (top1.score - top2.score) / denom
    certainty = max(0.0, min(1.0, margin))
    alternatives = results[:5]
    needs_clarification = certainty < clarification_threshold
    return QueryResult(answer=top1, certainty=certainty,
                       needs_clarification=needs_clarification,
                       alternatives=alternatives)


class QueryResult:
    """Single-answer retrieval result with a margin-based certainty signal.

    Produced by `Memory.query_one()`. The contract is designed so a
    downstream LLM can route behavior by certainty:

        r = m.query_one("where do I live?")
        if r.needs_clarification:
            ask_user_to_disambiguate(r.alternatives)
        elif r.answer:
            use(r.answer.text)

    Fields:
      answer             — the top-ranked MemoryResult, or None if empty.
      certainty          — margin of top-1 over top-2, in [0.0, 1.0].
                           1.0 means top-2 scored zero; 0.0 means tie.
                           Specifically: (top1 - top2) / max(top1, eps).
      needs_clarification — True when certainty < 0.15 AND top-2 exists.
                           At this margin top-1 is within 15% of top-2
                           and should not be treated as a confident answer.
      alternatives       — top-k results (up to 5), including the answer
                           in position 0. Empty list if no matches.
    """
    __slots__ = ("answer", "certainty", "needs_clarification", "alternatives")

    def __init__(self, answer, certainty: float,
                 needs_clarification: bool,
                 alternatives: list) -> None:
        self.answer = answer
        self.certainty: float = certainty
        self.needs_clarification: bool = needs_clarification
        self.alternatives: list = alternatives

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer.to_dict() if self.answer else None,
            "certainty": round(self.certainty, 3),
            "needs_clarification": self.needs_clarification,
            "alternatives": [a.to_dict() for a in self.alternatives],
        }

    def __repr__(self) -> str:
        status = "unclear" if self.needs_clarification else "confident"
        a = self.answer
        if a is None:
            return f"QueryResult(empty, {status})"
        return f"QueryResult({a.predicate}={a.object}, certainty={self.certainty:.2f}, {status})"


class MemoryResult:
    """A single memory retrieval result."""

    __slots__ = ("id", "text", "subject", "predicate", "object", "confidence",
                 "is_negation", "score", "scope", "created_at", "source_type")

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
        self.source_type: str = sr.memory.source_type

    @property
    def is_suspicious(self) -> bool:
        """True if this memory was flagged as a prompt-injection candidate on write."""
        from lore_memory.safety import SUSPICIOUS_SOURCE_TYPE
        return self.source_type == SUSPICIOUS_SOURCE_TYPE

    def to_llm_context(self) -> str:
        """Render the result's text for safe inclusion in an LLM prompt.

        Suspicious memories (flagged as possible prompt injection on write)
        are wrapped in <user_stated_untrusted>...</user_stated_untrusted>
        delimiters so a consuming LLM can treat them as data, not
        instructions.
        """
        from lore_memory.safety import wrap_untrusted
        return wrap_untrusted(self.text) if self.is_suspicious else self.text

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "text": self.text, "subject": self.subject,
            "predicate": self.predicate, "object": self.object,
            "confidence": self.confidence, "is_negation": self.is_negation,
            "score": self.score, "scope": self.scope,
            "source_type": self.source_type,
        }

    def __repr__(self) -> str:
        suffix = " [suspicious]" if self.is_suspicious else ""
        return f"MemoryResult({self.predicate}={self.object}, conf={self.confidence}){suffix}"


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
        schema: Schema | None = None,
    ) -> None:
        if data_dir is None:
            data_dir = str(Path.home() / ".lore-memory")
        self._user_id = user_id
        self._org_id = org_id
        self._engine = Engine(
            Config(data_dir=str(data_dir), embedding_dims=embedding_dims),
            schema=schema or PERSONAL_LIFE_SCHEMA,
        )
        if embed_fn is not None:
            self._engine._embed = embed_fn
            self._engine._retriever.embed = embed_fn
        # Persist schema hash and warn on mismatch with the stored one.
        self._engine._ensure_schema_compatible(self._user_id)

    # --- Core API ---

    _MAX_INPUT_CHARS = 50_000  # ~10K tokens; guards pathological pastes.

    def store(self, text: str, scope: str = "private",
              facts: list[dict] | None = None) -> dict[str, Any]:
        """Store a memory from natural text.

        Two paths:
          1. ``store(text)`` — grammar parser extracts facts. Use this when
             the caller doesn't know the structure (curl, bash, dumb pipes).
          2. ``store(text, facts=[{subject, predicate, object}, ...])`` —
             LLM caller supplies structured triples. The grammar extractor
             is skipped; the LLM's S-P-O is written directly. Raw text is
             still saved as a 'stated' row so FTS keyword search and
             journal continuity work the same.

        Pass ``facts`` from any LLM that can produce JSON: Claude, GPT,
        Gemini, Llama via Ollama. The schema is the contract — see the
        MCP and REST tool descriptions for canonical predicates.

        Args:
            text: Natural-language utterance. Always stored as the raw row.
            scope: "private" (user-only) or "shared" (org-visible).
            facts: Optional list of {subject, predicate, object,
                confidence?, is_negation?} dicts. Subject "user" maps to
                the caller's user_id. When omitted, grammar extraction runs.

        Returns:
            Dict with keys: created, deduplicated, contradictions.
        """
        if text is None:
            return {"created": 0, "deduplicated": 0, "contradictions": 0}
        if not isinstance(text, str):
            try:
                text = str(text)
            except Exception:
                return {"created": 0, "deduplicated": 0, "contradictions": 0}
        text = text.strip()
        if not text:
            return {"created": 0, "deduplicated": 0, "contradictions": 0}
        if len(text) > self._MAX_INPUT_CHARS:
            text = text[: self._MAX_INPUT_CHARS]
        if scope == "shared" and self._org_id:
            r = self._engine.store_company(self._user_id, self._org_id, text,
                                           facts=facts)
        else:
            r = self._engine.store_personal(self._user_id, text, facts=facts)
        return {"created": r.created, "deduplicated": r.deduplicated,
                "contradictions": r.contradictions}

    def query(self, query: str, limit: int = 10,
              predicate_hint: str | list[str] | None = None,
              subject_hint: str | None = None) -> list[MemoryResult]:
        """Query memory using natural language, optionally with LLM hints.

        Two paths:
          1. ``query(text)`` — current 7-channel retrieval. Works for any
             caller including raw scripts.
          2. ``query(text, predicate_hint=..., subject_hint=...)`` — LLM
             caller hints which canonical predicate / subject the answer
             lives under. Hints are *boosts*, not filters: a wrong hint
             never hides a correct answer; a right hint surfaces it
             instantly without paying for full semantic ranking noise.

        Args:
            query: Natural language question or keywords.
            limit: Maximum results to return.
            predicate_hint: Canonical predicate string or list of strings
                (e.g. "lives_in", ["job_title", "works_at"]). Matched
                against canonicalized predicates via Schema.aliases.
            subject_hint: Subject to favor. Pass "user" to favor the
                caller's own facts.

        Returns:
            List of MemoryResult objects, ranked by relevance.
        """
        if query is None or not isinstance(query, str) or not query.strip():
            return []
        results = self._engine.recall(
            self._user_id, query.strip(), org_id=self._org_id, top_k=limit,
            predicate_hint=predicate_hint, subject_hint=subject_hint)
        return [MemoryResult(r) for r in results]

    def query_one(self, query: str,
                  clarification_threshold: float = 0.15) -> QueryResult:
        """Single-answer query with a margin-based certainty signal.

        Returns a QueryResult with `.answer` (top-1 MemoryResult or None),
        `.certainty` (top1-vs-top2 margin in [0, 1]), and
        `.needs_clarification` (True when certainty < threshold and a
        top-2 exists). Use this from an LLM when you want the model to
        ask the user to disambiguate rather than guess.

        Args:
            query: Natural language question.
            clarification_threshold: certainty below this triggers
                needs_clarification=True. Default 0.15 — top-1 within
                ~15% of top-2 is considered ambiguous.
        """
        results = self.query(query, limit=5)
        return _build_query_result(results, clarification_threshold)

    def forget(self, memory_id: str | None = None,
               subject: str | None = None,
               predicate: str | None = None,
               hard: bool = False) -> bool:
        """Remove a memory or category of memories.

        Args:
            memory_id: Specific memory ID to delete.
            subject: Delete all memories about this subject.
            predicate: Combined with subject, delete specific predicate.
            hard: If True, actually DELETE the rows (not a state flag) and
                  VACUUM the DB. Required for GDPR-style erasure. Default
                  is a soft delete (state='deleted') so facts are recoverable.

        Returns:
            True if any memories were deleted.
        """
        def _apply(db, mem_id: str) -> None:
            if hard:
                db.delete_hard(mem_id)
            else:
                db.update_state(mem_id, "deleted")

        if memory_id:
            db, mem = self._engine._resolve_db(
                memory_id, scope_hint="private", user_id=self._user_id)
            if db and mem:
                _apply(db, memory_id)
                # Also delete sibling memories from the same source text
                # (raw text + extracted facts are stored together)
                if mem.source_text:
                    siblings = db.query_by_subject(mem.subject)
                    for sib in siblings:
                        if sib.id != memory_id and sib.source_text == mem.source_text:
                            _apply(db, sib.id)
                if hard:
                    db.vacuum()
                return True
            return False

        if subject:
            from lore_memory.scopes import Scope
            db = self._engine._db(Scope.PRIVATE, user_id=self._user_id)
            if hard:
                # Erase all rows for this (subject, predicate?) — any state.
                # Required for GDPR: leaving soft-deleted rows around is
                # incomplete erasure.
                n = db.delete_hard_by_subject(subject, predicate=predicate)
                if n:
                    db.vacuum()
                return n > 0
            # Soft path: only visible (active) rows get flagged.
            mems = db.query_by_subject(subject)
            deleted = 0
            for m in mems:
                if predicate and m.predicate != predicate:
                    continue
                db.update_state(m.id, "deleted")
                deleted += 1
            return deleted > 0

        return False

    def forget_all(self, hard: bool = True) -> bool:
        """Delete ALL memories for this user. Irreversible.

        hard=True (default) removes the user's SQLite file entirely, which
        is the strongest available guarantee on a per-tenant filesystem
        layout. hard=False is a soft no-op placeholder for API symmetry with
        forget() — it currently behaves identically to hard=True because the
        storage layout is file-per-tenant.
        """
        return self._engine.purge_user(self._user_id)

    def export_all(self) -> list[dict[str, Any]]:
        """Return every memory ever stored for this user, in insertion order.

        Includes active, superseded, archived, and soft-deleted rows — this
        is the GDPR-portability surface: the user can see and take
        everything. Embeddings are excluded (not human-portable).
        Returns a list of plain dicts, safe to json.dump().
        """
        from lore_memory.scopes import Scope
        db = self._engine._db(Scope.PRIVATE, user_id=self._user_id)
        return db.export_all()

    def export_to_jsonl(self, path: str | Path) -> int:
        """Write export_all() output to a JSONL file. Returns row count."""
        import json
        rows = self.export_all()
        p = Path(path)
        with p.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, default=str, ensure_ascii=False) + "\n")
        return len(rows)

    def import_from_jsonl(self, path: str | Path, skip_existing: bool = True) -> int:
        """Import memories from a JSONL export (e.g. from export_to_jsonl).

        Useful for portability / restore across machines. The JSON must
        match the export_all() shape. Existing rows with the same id are
        skipped by default (idempotent restore); pass skip_existing=False
        to overwrite state. Returns the number of rows imported.
        """
        import json
        from lore_memory.scopes import Scope
        from lore_memory.store import Memory as _MemoryRow
        db = self._engine._db(Scope.PRIVATE, user_id=self._user_id)
        n = 0
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if skip_existing:
                    try:
                        existing = db.conn.execute(
                            "SELECT 1 FROM memories WHERE id=?",
                            (d.get("id", ""),)).fetchone()
                    except Exception:
                        existing = None
                    if existing:
                        continue
                # Rebuild Memory from the export dict. Embedding is not
                # exported (not portable), so is_active will use existing
                # posterior values from the row.
                md = d.get("metadata") or {}
                if isinstance(md, str):
                    try: md = json.loads(md)
                    except Exception: md = {}
                try:
                    now = time.time()
                    mem = _MemoryRow(
                        id=d["id"],
                        scope=d.get("scope","private"),
                        context=d.get("context","personal"),
                        user_id=d.get("user_id", self._user_id),
                        org_id=d.get("org_id",""),
                        repo_id=d.get("repo_id",""),
                        subject=d.get("subject",""),
                        predicate=d.get("predicate","stated"),
                        object_value=d.get("object_value",""),
                        source_text=d.get("source_text","") or "",
                        is_negation=bool(d.get("is_negation", False)),
                        source_type=d.get("source_type","imported"),
                        embedding=[],
                        confidence=float(d.get("confidence", 0.7)),
                        evidence_count=int(d.get("evidence_count", 1)),
                        contradiction_count=int(d.get("contradiction_count", 0)),
                        created_at=float(d.get("created_at") or now),
                        updated_at=float(d.get("updated_at") or now),
                        last_accessed=float(d.get("last_accessed") or now),
                        access_count=int(d.get("access_count", 0)),
                        valid_until=d.get("valid_until"),
                        state=d.get("state","active"),
                        deleted_at=d.get("deleted_at"),
                        metadata=md or {},
                    )
                    db.put(mem)
                    n += 1
                except Exception:
                    continue
        return n

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

    def warmup(self) -> None:
        """Force the embedding model to load now.

        Memory is cheap to construct (~10 ms) because the sentence-
        transformers model is loaded lazily on first embed. Call this
        method to pay that cost up-front — useful when you want
        predictable first-query latency in an interactive app.
        """
        # Touching the embed function with any input triggers the load.
        self._engine._embed("warmup")

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
