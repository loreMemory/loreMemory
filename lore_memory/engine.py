"""
Lore Engine — text-first, extraction-optional.

Architecture:
  Text in → stored as raw text memory (FTS-indexed) → searchable immediately
  Optional extractor → produces additional structured SPO memories
  No hardcoded regex. No LLM required. Works with any language.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from lore_memory.belief import check_contradictions, check_cross_scope_contradictions, consolidate as run_consolidation, canon
from lore_memory.extraction import extract_personal, extract_chat, extract_company, _norm, Extractor
from lore_memory.graph import GraphCache
from lore_memory.repo import recent_commits, file_tree, commits_to_memories, tree_to_memories, repo_id
from lore_memory.retrieval import Retriever, SearchResult
from lore_memory.scopes import Scope, Context, scope_db_path, can_access
from lore_memory.store import Memory, MemoryDB


DB_CACHE_MAX = 100  # Max open MemoryDB connections before LRU eviction


@dataclass
class Config:
    data_dir: str = "./lore_data"
    embedding_dims: int = 384
    db_cache_max: int = DB_CACHE_MAX


@dataclass
class WriteResult:
    created: int = 0
    deduplicated: int = 0
    contradictions: int = 0


class Engine:
    """
    Lore — text-first multi-context persistent memory engine.

    Default: stores raw text, retrieves via FTS5 + embeddings. No extraction.
    Optional: pass an extractor for structured SPO enrichment.

    Usage:
        e = Engine(Config(data_dir="./data"))
        e.store_personal("user1", "I live in Amsterdam and work at Google")
        e.store_chat("user1", "We decided to use React", session_id="s1")
        e.ingest_repo("user1", "/path/to/repo")
        e.store_company("user1", "acme", "Our mission is to democratize AI")
        results = e.recall("user1", "Where do I live?", org_id="acme")
        e.feedback("user1", results[0].memory.id, helpful=True)
    """

    def __init__(self, config: Config | None = None,
                 extractor: Extractor | None = None) -> None:
        self.config = config or Config()
        self._data = Path(self.config.data_dir)
        self._data.mkdir(parents=True, exist_ok=True)

        # Embedding: auto-detect best available, no user config needed
        self._dims = self.config.embedding_dims
        self._embed = self._init_embeddings()

        # Optional extractor — produces structured SPO in addition to raw text
        self._extractor = extractor

        self._retriever = Retriever(self._embed)
        self._graph = GraphCache(cache_path=self._data / "graph_cache.json")
        self._dbs: OrderedDict[str, MemoryDB] = OrderedDict()
        self._dbs_lock = threading.Lock()  # Protects _dbs mutations
        self._db_cache_max = self.config.db_cache_max
        self._purged_paths: set[str] = set()

    def _init_embeddings(self):
        """Auto-detect best available embedding model. No user config needed."""
        try:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer('all-MiniLM-L6-v2')
            self._dims = 384
            return self._st_embed
        except ImportError:
            return self._hash_embed

    def _st_embed(self, text: str) -> list[float]:
        """Real semantic embeddings via sentence-transformers."""
        return self._st_model.encode(text).tolist()

    def _hash_embed(self, text: str) -> list[float]:
        import struct, math
        vecs: list[float] = []
        seed = text.encode("utf-8")
        for i in range(0, self._dims, 8):
            h = hashlib.sha256(seed + struct.pack(">I", i)).digest()
            for j in range(min(8, self._dims - i)):
                vecs.append((h[j] / 127.5) - 1.0)
        norm = math.sqrt(sum(v * v for v in vecs))
        return [v / norm for v in vecs] if norm > 0 else vecs[:self._dims]

    def _db(self, scope: Scope, *, user_id: str = "", org_id: str = "", repo_id: str = "") -> MemoryDB:
        path = scope_db_path(self._data, scope, user_id=user_id, org_id=org_id, repo_id=repo_id)
        key = str(path)
        with self._dbs_lock:
            if key in self._purged_paths:
                if key not in self._dbs:
                    self._dbs[key] = MemoryDB(":memory:", self._dims)
                else:
                    self._dbs.move_to_end(key)
                return self._dbs[key]
            if key in self._dbs:
                self._dbs.move_to_end(key)
            else:
                self._evict_lru()
                self._dbs[key] = MemoryDB(path, self._dims)
            return self._dbs[key]

    def _evict_lru(self) -> None:
        """Evict least recently used databases when cache exceeds max."""
        while len(self._dbs) >= self._db_cache_max:
            oldest_key, oldest_db = self._dbs.popitem(last=False)
            if oldest_db.db_path == ":memory:":
                self._dbs[oldest_key] = oldest_db
                self._dbs.move_to_end(oldest_key, last=False)
                break
            try:
                oldest_db.close()
            except Exception:
                pass  # Don't let close failure block eviction

    # --- Write API ---

    def store_personal(self, user_id: str, text: str, subject: str = "") -> WriteResult:
        db = self._db(Scope.PRIVATE, user_id=user_id)
        mems = extract_personal(text, user_id, subject, extractor=self._extractor)
        return self._store(db, mems, cross_scope_dbs=[])

    def store_chat(self, user_id: str, text: str, speaker: str = "", session_id: str = "") -> WriteResult:
        db = self._db(Scope.PRIVATE, user_id=user_id)
        mems = extract_chat(text, user_id, speaker, session_id, extractor=self._extractor)
        return self._store(db, mems, cross_scope_dbs=[])

    def store_company(self, user_id: str, org_id: str, text: str, subject: str = "") -> WriteResult:
        db = self._db(Scope.SHARED, org_id=org_id)
        mems = extract_company(text, user_id, org_id, subject, extractor=self._extractor)
        private_db = self._db(Scope.PRIVATE, user_id=user_id)
        return self._store(db, mems, cross_scope_dbs=[private_db])

    _VALID_SCOPE_CONTEXT = {
        ("private", "personal"), ("private", "chat"),
        ("shared", "company"), ("repo", "repo"),
    }

    def store_fact(self, scope: str, context: str, subject: str, predicate: str,
                   object_value: str, user_id: str = "", org_id: str = "",
                   repo_id_val: str = "", confidence: float = 0.7,
                   source_text: str = "", is_negation: bool = False,
                   source_type: str = "user_stated", metadata: dict | None = None) -> WriteResult:
        sc = Scope(scope)
        if (scope, context) not in self._VALID_SCOPE_CONTEXT:
            raise ValueError(f"Invalid scope/context combination: {scope}/{context}")
        db = self._db(sc, user_id=user_id, org_id=org_id, repo_id=repo_id_val)
        mem = Memory(
            scope=scope, context=context, user_id=user_id, org_id=org_id,
            repo_id=repo_id_val, subject=subject, predicate=_norm(predicate),
            object_value=object_value, source_text=source_text,
            is_negation=is_negation, source_type=source_type,
            confidence=confidence, metadata=metadata or {},
        )
        return self._store(db, [mem], cross_scope_dbs=[])

    def ingest_repo(self, user_id: str, repo_path: str | Path, commit_limit: int = 50) -> WriteResult:
        rid = repo_id(repo_path)
        db = self._db(Scope.REPO, repo_id=rid)
        mems = commits_to_memories(recent_commits(repo_path, commit_limit), rid)
        mems += tree_to_memories(file_tree(repo_path), rid)
        return self._store(db, mems, cross_scope_dbs=[])

    # --- Read API ---

    def recall(self, user_id: str, query: str, org_id: str = "", repo_id_val: str = "",
               context: str | None = None, top_k: int = 20) -> list[SearchResult]:
        dbs: list[tuple[MemoryDB, str]] = []
        dbs.append((self._db(Scope.PRIVATE, user_id=user_id), f"private:{user_id}"))
        if org_id:
            dbs.append((self._db(Scope.SHARED, org_id=org_id), f"shared:{org_id}"))
        if repo_id_val:
            dbs.append((self._db(Scope.REPO, repo_id=repo_id_val), f"repo:{repo_id_val}"))

        # Graph build only on cold start (first query ever).
        # After that, incremental add_edge in _store keeps it current.
        if self._graph.is_dirty:
            self._graph.build([db for db, _ in dbs])
            self._retriever.set_graph_cache(self._graph)
        elif not self._retriever._graph_cache:
            self._retriever.set_graph_cache(self._graph)

        results = self._retriever.search(
            query, dbs, context=context, top_k=top_k,
            weight_key=f"{user_id}:{context or 'all'}",
        )
        return results

    def profile(self, user_id: str, subject: str, org_id: str = "") -> dict[str, list[dict]]:
        dbs = [(self._db(Scope.PRIVATE, user_id=user_id), "private")]
        if org_id:
            dbs.append((self._db(Scope.SHARED, org_id=org_id), "shared"))
        prof: dict[str, list[dict]] = {}
        for db, label in dbs:
            for f in db.query_by_subject(subject):
                if not f.is_active:
                    continue
                prof.setdefault(f.predicate, []).append({
                    "value": f.object_value, "confidence": round(f.posterior, 3),
                    "evidence": f.evidence_count, "negation": f.is_negation,
                    "scope": label,
                })
        for pred in prof:
            prof[pred].sort(key=lambda x: x["confidence"], reverse=True)
        return prof

    # --- Feedback API ---

    def _resolve_db(self, memory_id: str, scope_hint: str | None = None,
                    user_id: str = "", org_id: str = "",
                    repo_id_val: str = "") -> tuple[MemoryDB | None, Memory | None]:
        """Find the database containing a memory, optionally using scope_hint
        to avoid iterating all open databases."""
        if scope_hint:
            try:
                sc = Scope(scope_hint)
                db = self._db(sc, user_id=user_id, org_id=org_id, repo_id=repo_id_val)
                mem = db.get(memory_id)
                if mem:
                    return db, mem
            except ValueError:
                pass
        # Fallback: snapshot DB list under lock, iterate outside lock
        with self._dbs_lock:
            dbs_snapshot = list(self._dbs.values())
        for db in dbs_snapshot:
            mem = db.get(memory_id)
            if mem:
                return db, mem
        return None, None

    def feedback(self, user_id: str, memory_id: str, helpful: bool,
                 channel_scores: dict[str, float] | None = None,
                 context: str | None = None,
                 scope_hint: str | None = None,
                 org_id: str = "", repo_id_val: str = "") -> bool:
        db, mem = self._resolve_db(memory_id, scope_hint=scope_hint,
                                   user_id=user_id, org_id=org_id,
                                   repo_id_val=repo_id_val)
        if not db or not mem:
            return False
        if not can_access(user_id, Scope(mem.scope), mem.user_id):
            return False
        db.record_feedback(memory_id, helpful)
        if helpful and channel_scores:
            result = SearchResult(memory=mem, score=0.0, channel_scores=channel_scores)
            weight_key = f"{user_id}:{context or 'all'}"
            self._retriever.learn_from_feedback(result, weight_key)
        return True

    def recover(self, user_id: str, memory_id: str,
                scope_hint: str | None = None,
                org_id: str = "", repo_id_val: str = "") -> bool:
        db, mem = self._resolve_db(memory_id, scope_hint=scope_hint,
                                   user_id=user_id, org_id=org_id,
                                   repo_id_val=repo_id_val)
        if not db or not mem:
            return False
        if not can_access(user_id, Scope(mem.scope), mem.user_id):
            return False
        result = db.recover(memory_id)
        if result and mem.subject and mem.object_value:
            # Re-add edge for recovered memory
            self._graph.add_edge(mem.subject, mem.object_value)
        return result

    # --- Data Deletion ---

    def purge_user(self, user_id: str) -> bool:
        import os
        path = scope_db_path(self._data, Scope.PRIVATE, user_id=user_id)
        key = str(path)
        if key in self._dbs:
            self._dbs[key].close()
            del self._dbs[key]
        self._purged_paths.add(key)
        # Force graph rebuild — purged user's edges are invalid
        self._graph.force_rebuild()
        if path.exists():
            os.remove(path)
            return True
        return False

    # --- Maintenance ---

    def consolidate(self, user_id: str, org_id: str = "") -> dict:
        """Run consolidation with memory replay from activation traces."""
        total = {"archived": 0, "decayed": 0, "replayed": 0, "scopes": 0, "duration_ms": 0.0}

        # Collect replay traces from retriever (which memories were frequently returned)
        replay_traces: dict[str, int] = {}
        for mem_id, coact in self._retriever._traces.items():
            total_coact = sum(coact.values())
            if total_coact >= 3:
                replay_traces[mem_id] = int(total_coact)

        dbs = [self._db(Scope.PRIVATE, user_id=user_id)]
        if org_id:
            dbs.append(self._db(Scope.SHARED, org_id=org_id))
        for db in dbs:
            r = run_consolidation(db, replay_traces=replay_traces)
            total["archived"] += r["archived"]
            total["decayed"] += r["decayed"]
            total["replayed"] += r.get("replayed", 0)
            total["duration_ms"] += r["duration_ms"]
            total["scopes"] += 1
        return total

    def stats(self, user_id: str, org_id: str = "", repo_id_val: str = "") -> dict:
        r: dict[str, int] = {}
        db = self._db(Scope.PRIVATE, user_id=user_id)
        r["private_total"] = db.count()
        r["private_personal"] = db.count("personal")
        r["private_chat"] = db.count("chat")
        if org_id:
            r["shared_total"] = self._db(Scope.SHARED, org_id=org_id).count()
        if repo_id_val:
            r["repo_total"] = self._db(Scope.REPO, repo_id=repo_id_val).count()
        return r

    def profile_compact(self, user_id: str, max_tokens: int = 200) -> str:
        """Progressive loading: return a token-budgeted identity string.
        L0 (~50 tokens): name, role
        L1 (~150 tokens): top facts by evidence count
        Inspired by MemPalace's L0-L3 loading."""
        db = self._db(Scope.PRIVATE, user_id=user_id)
        facts = db.query_by_subject(user_id, limit=50)
        if not facts:
            return ""

        # Sort by evidence (most confirmed first), then by posterior
        facts.sort(key=lambda f: (f.evidence_count, f.posterior), reverse=True)

        lines: list[str] = []
        token_est = 0  # rough: 1 token ≈ 4 chars

        for f in facts:
            if not f.is_active or f.predicate == "stated":
                continue
            neg = "not " if f.is_negation else ""
            line = f"{f.predicate}: {neg}{f.object_value}"
            est = len(line) // 4 + 1
            if token_est + est > max_tokens:
                break
            lines.append(line)
            token_est += est

        return "\n".join(lines)

    def get_weights(self, user_id: str, context: str = "all") -> dict:
        return self._retriever.get_weights(f"{user_id}:{context}").to_dict()

    # --- Internal ---

    def _store(self, db: MemoryDB, mems: list[Memory], cross_scope_dbs: list[MemoryDB]) -> WriteResult:
        result = WriteResult()
        for mem in mems:
            if not mem.embedding:
                mem.embedding = self._embed(mem.triplet_text)
            # Skip contradiction checks for "stated" predicate — raw text
            # memories are never single-valued and never contradict each other.
            # This avoids O(n) subject queries for every text memory.
            if mem.predicate != "stated":
                contradicted = check_contradictions(db, mem)
                result.contradictions += len(contradicted)
                if cross_scope_dbs:
                    cross = check_cross_scope_contradictions(
                        cross_scope_dbs, mem.subject, mem.predicate, mem.object_value)
                    result.contradictions += len(cross)
            dup = db.put(mem)
            if dup:
                result.deduplicated += 1
            else:
                result.created += 1
                # Incrementally add edge to graph (O(1), no full rebuild)
                if mem.subject and mem.object_value:
                    self._graph.add_edge(mem.subject, mem.object_value)
        return result

    def close(self) -> None:
        # Persist graph cache for fast restart
        self._graph.save_cache()
        for db in self._dbs.values():
            db.close()
        self._dbs.clear()
