"""
Lore Engine — text-first, extraction-optional.

Architecture:
  Text in → stored as raw text memory (FTS-indexed) → searchable immediately
  Optional extractor → produces additional structured SPO memories
  No hardcoded regex. No LLM required. Works with any language.
"""

from __future__ import annotations

import hashlib
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from lore_memory.belief import check_contradictions, check_cross_scope_contradictions, check_loss_events, consolidate as run_consolidation, canon, ALIASES as BELIEF_ALIASES
from lore_memory.schema import Schema, PERSONAL_LIFE_SCHEMA
from lore_memory.dedup import DedupEngine, DedupResult, ProvenanceTracker
from lore_memory.extraction import extract_personal, extract_chat, extract_company, _norm, Extractor
from lore_memory.extraction_gf import GrammarFreeExtractor
from lore_memory.graph import GraphCache
from lore_memory.normalization import NormalizationPipeline
from lore_memory.repo import recent_commits, file_tree, commits_to_memories, tree_to_memories, repo_id
from lore_memory.retrieval import Retriever, SearchResult
from lore_memory.scopes import Scope, Context, scope_db_path, can_access
from lore_memory.store import Memory, MemoryDB


from lore_memory.lexicons import COMMIT_TYPE_PREFIXES as _COMMIT_PREFIXES
from lore_memory.lexicons import COMMIT_MSG_RE as _COMMIT_MSG_RE

_VOWELS = set("aeiouAEIOU")


def _is_profile_worthy(f) -> bool:
    """Return True if a fact should appear in a user profile.

    Structural garbage filter only: reject parser debris (length extremes,
    vowel-less predicates, commit artifacts). Do NOT gate on a vocabulary
    allowlist — rank-by-evidence is the ranking layer's job.
    """
    pred = f.predicate
    # Skip raw-text bucket
    if pred == "stated":
        return False
    # Predicates shorter than 3 chars are almost always parser noise
    # (e.g. "se", "am", "is").
    if len(pred) <= 2:
        return False
    # Overly long predicates are parsing artifacts
    if len(pred) > 25:
        return False
    # Commit-like compound predicates (e.g. 'fixredi', 'featauth')
    if any(pred.startswith(p) and len(pred) > len(p) and pred != p
           for p in _COMMIT_PREFIXES):
        return False
    # Predicates with no vowels are likely garbage (e.g. 'te_bk_s')
    if not any(ch in _VOWELS for ch in pred):
        return False
    # Source text that looks like a conventional commit message
    if f.source_text and _COMMIT_MSG_RE.match(f.source_text):
        return False
    # Object sanity: empty/too-short/too-long objects are parser debris
    obj = (f.object_value or "").strip()
    if len(obj) < 2 or len(obj) > 80:
        return False
    return True


DB_CACHE_MAX = 100  # Max open MemoryDB connections before LRU eviction


@dataclass
class Config:
    data_dir: str = "./lore_data"
    embedding_dims: int = 384
    db_cache_max: int = DB_CACHE_MAX
    # Prompt-injection defense: classify every store() and flag suspicious
    # text as source_type="suspicious", excluding it from profile_compact
    # and wrapping it in untrusted delimiters when recalled to LLM context.
    # Adds ~1ms per insert. Disable only for trusted batch-ingest pipelines
    # where the input is guaranteed safe.
    injection_defense: bool = True
    # Hypothetical detection: classify text as a hedge/conditional/
    # speculation and, if so, store each resulting memory with lower
    # posterior and metadata["hypothetical"]=True, and skip the
    # supersession check. Keeps "if I get the offer I'll move" and
    # "maybe I'll quit" from overwriting real facts.
    hypothetical_detection: bool = True
    # Use spaCy-based parser instead of the hand-rolled grammar.
    # Default since validation: spaCy beat grammar on compound clauses,
    # compound objects, particle verbs ("Luna passed away"), and "named X"
    # patterns, with no regression on the 7 documented-bug black-box tests.
    # Set to False for the legacy grammar parser (faster ~1ms p50 vs 4ms,
    # but missing the above patterns).
    use_spacy: bool = True


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
                 extractor: Extractor | None = None,
                 schema: Schema | None = None) -> None:
        self.config = config or Config()
        self._data = Path(self.config.data_dir)
        self._data.mkdir(parents=True, exist_ok=True)
        # Identity schema: which predicates supersede, alias, and decay.
        # Default is the personal-life / English knowledge-worker schema —
        # exactly the behavior prior to Phase 2 Step 6.
        self._schema: Schema = schema or PERSONAL_LIFE_SCHEMA

        # Embedding: auto-detect best available, no user config needed
        self._dims = self.config.embedding_dims
        self._embed_cache: dict[str, list[float]] = {}
        self._EMBED_CACHE_MAX = 10000
        self._embed = self._init_embeddings()

        # Optional extractor — produces structured SPO in addition to raw text
        self._extractor = extractor

        # Normalization pipeline — subject, predicate, object canonicalization
        self._normalizers: dict[str, NormalizationPipeline] = {}

        # Dedup engine — per-user, initialized lazily with normalizer
        self._dedup_engines: dict[str, DedupEngine] = {}

        # Provenance tracker — tracks origin and transformation of facts
        self._provenance = ProvenanceTracker()

        # Grammar-free extractor
        self._gf_extractor = GrammarFreeExtractor(embed_fn=self._embed)

        self._retriever = Retriever(self._embed)
        self._graph = GraphCache(cache_path=self._data / "graph_cache.json")
        self._dbs: OrderedDict[str, MemoryDB] = OrderedDict()
        self._dbs_lock = threading.Lock()  # Protects _dbs mutations
        self._db_cache_max = self.config.db_cache_max
        self._purged_paths: set[str] = set()

        # Prompt-injection classifier — tags suspicious-looking writes so
        # they are never surfaced verbatim into LLM context. Use a lambda
        # closure so the classifier always sees the current _embed (which
        # callers may override on the public Memory wrapper).
        from lore_memory.safety import InjectionClassifier, HypotheticalClassifier
        self._injection_clf = InjectionClassifier(lambda t: self._embed(t))
        # Hypothetical classifier — flags conditional / speculative text so
        # it's stored with lower posterior and never supersedes a fact.
        self._hypothetical_clf = HypotheticalClassifier(lambda t: self._embed(t))

    # Process-level singleton — loading sentence-transformers is expensive
    # (~5 s, ~500 MB RAM) and every Engine instance wants the same model.
    # Load once per process; subsequent Engines reuse it for free.
    _ST_MODEL_CACHE: "object | None" = None
    _ST_PROBED: bool = False
    _ST_LOCK = threading.Lock()

    def _init_embeddings(self):
        """Return the embed fn. The sentence-transformers model is cached
        at the class level so the first Engine in a process pays the ~5 s
        load cost and every subsequent Engine is instant. If the library
        isn't installed, fall back to a fast hash embed transparently.
        """
        self._dims = 384
        return self._st_embed

    def _ensure_st_model(self) -> bool:
        """Load the process-level sentence-transformer singleton.

        Returns True if the real model is ready, False if we fell back
        to the hash embed (library not installed).
        """
        if Engine._ST_PROBED:
            return Engine._ST_MODEL_CACHE is not None
        with Engine._ST_LOCK:
            if Engine._ST_PROBED:
                return Engine._ST_MODEL_CACHE is not None
            try:
                from sentence_transformers import SentenceTransformer
                Engine._ST_MODEL_CACHE = SentenceTransformer('all-MiniLM-L6-v2')
            except ImportError:
                Engine._ST_MODEL_CACHE = None
            Engine._ST_PROBED = True
            return Engine._ST_MODEL_CACHE is not None

    def _st_embed(self, text: str) -> list[float]:
        """Real semantic embeddings via the shared sentence-transformers
        singleton (with per-Engine LRU cache). Falls back to hash embed if
        the library isn't available in this environment.
        """
        key = text[:200]
        cached = self._embed_cache.get(key)
        if cached is not None:
            return cached

        if not self._ensure_st_model():
            # Library not installed anywhere — fall back permanently.
            self._embed = self._hash_embed
            return self._hash_embed(text)

        emb = Engine._ST_MODEL_CACHE.encode(text).tolist()
        if len(self._embed_cache) >= self._EMBED_CACHE_MAX:
            oldest = next(iter(self._embed_cache))
            del self._embed_cache[oldest]
        self._embed_cache[key] = emb
        return emb

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

    def _get_normalizer(self, user_id: str) -> NormalizationPipeline:
        """Get or create the normalization pipeline for a user."""
        if user_id not in self._normalizers:
            normalizer = NormalizationPipeline(
                embed_fn=self._embed, user_id=user_id)
            # Seed with existing predicate aliases from belief.py
            normalizer.seed_predicate_aliases(BELIEF_ALIASES)
            self._normalizers[user_id] = normalizer
        return self._normalizers[user_id]

    def _get_dedup(self, user_id: str) -> DedupEngine:
        """Get or create the dedup engine for a user."""
        if user_id not in self._dedup_engines:
            normalizer = self._get_normalizer(user_id)
            self._dedup_engines[user_id] = DedupEngine(normalizer)
        return self._dedup_engines[user_id]

    # --- Write API ---

    def store_personal(self, user_id: str, text: str, subject: str = "",
                        source_tool: str = "chat_parser",
                        facts: list[dict] | None = None) -> WriteResult:
        """Store personal-scope memory.

        If ``facts`` is provided (LLM-supplied S-P-O list), the grammar
        extractor is skipped and the supplied triples are written directly
        alongside the standard raw-text 'stated' row. Each fact still goes
        through the full dedup / canonicalization / supersession pipeline,
        so an LLM can't bypass schema discipline by writing whatever it
        wants — predicates are normalized via Schema.aliases on read.
        """
        db = self._db(Scope.PRIVATE, user_id=user_id)
        normalizer = self._get_normalizer(user_id)
        normalizer.learn_from_text(text)

        if facts is not None:
            mems = self._build_mems_from_facts(
                user_id=user_id, text=text, facts=facts,
                scope="private", context="personal", source_tool=source_tool)
        else:
            # Grammar-based extraction (primary for sentence-like text)
            mems = extract_personal(text, user_id, subject, extractor=self._extractor,
                                    use_spacy=self.config.use_spacy)

            # Grammar-free extraction only for non-sentence formats
            # Avoids double-extraction and uncapped objects on normal sentences
            from lore_memory.extraction_gf import detect_source_type
            st = detect_source_type(text)
            if st.type != "sentence":
                gf_mems = self._gf_extractor.extract(
                    text, user_id, source_tool=source_tool,
                    scope="private", context="personal", subject=subject or user_id)
                mems.extend(gf_mems)

        self._apply_write_flags(mems, text)

        return self._store(db, mems, cross_scope_dbs=[], user_id=user_id,
                          source_tool=source_tool)

    def _build_mems_from_facts(self, *, user_id: str, text: str,
                                facts: list[dict], scope: str, context: str,
                                source_tool: str) -> list:
        """Build a stated row + one Memory per LLM-supplied fact dict.

        Fact dict shape:
          {subject, predicate, object, confidence?, is_negation?}
        Subject 'user' (case-insensitive) is rewritten to user_id.
        Predicate is normalized; canonicalization is deferred to read time.
        """
        from lore_memory.extraction import _norm
        mems: list = []
        # Always preserve raw text — same as the grammar path.
        mems.append(Memory(
            scope=scope, context=context, user_id=user_id,
            subject=user_id, predicate="stated",
            object_value=text[:500], source_text=text[:500],
            source_type="user_stated", confidence=1.0,
            metadata={"source_tool": source_tool, "extraction_source": "llm"},
        ))
        for f in facts or []:
            try:
                raw_pred = (f.get("predicate") or "").strip()
                obj = (f.get("object") or "").strip()
                # Reject before _norm — _norm returns "stated" for empty
                # input, which would silently turn malformed facts into
                # extra journal rows.
                if not raw_pred or not obj:
                    continue
                subj = (f.get("subject") or user_id).strip()
                if subj.lower() == "user":
                    subj = user_id
                pred = _norm(raw_pred)
                mems.append(Memory(
                    scope=scope, context=context, user_id=user_id,
                    subject=subj, predicate=pred, object_value=obj[:500],
                    source_text=text[:500],
                    is_negation=bool(f.get("is_negation", False)),
                    source_type="user_stated",
                    confidence=float(f.get("confidence", 0.9)),
                    metadata={"source_tool": source_tool, "extraction_source": "llm"},
                ))
            except (AttributeError, TypeError, ValueError):
                continue
        return mems

    def store_chat(self, user_id: str, text: str, speaker: str = "",
                    session_id: str = "", source_tool: str = "chat_parser") -> WriteResult:
        db = self._db(Scope.PRIVATE, user_id=user_id)
        normalizer = self._get_normalizer(user_id)
        normalizer.learn_from_text(text)
        mems = extract_chat(text, user_id, speaker, session_id, extractor=self._extractor)
        self._apply_write_flags(mems, text)
        return self._store(db, mems, cross_scope_dbs=[], user_id=user_id,
                          source_tool=source_tool)

    def store_company(self, user_id: str, org_id: str, text: str, subject: str = "",
                       source_tool: str = "company_parser",
                       facts: list[dict] | None = None) -> WriteResult:
        db = self._db(Scope.SHARED, org_id=org_id)
        if facts is not None:
            mems = self._build_mems_from_facts(
                user_id=user_id, text=text, facts=facts,
                scope="shared", context="company", source_tool=source_tool)
            for m in mems:
                m.org_id = org_id
        else:
            mems = extract_company(text, user_id, org_id, subject, extractor=self._extractor)
        private_db = self._db(Scope.PRIVATE, user_id=user_id)
        return self._store(db, mems, cross_scope_dbs=[private_db], user_id=user_id,
                          source_tool=source_tool)

    _VALID_SCOPE_CONTEXT = {
        ("private", "personal"), ("private", "chat"),
        ("shared", "company"), ("repo", "repo"),
    }

    def store_fact(self, scope: str, context: str, subject: str, predicate: str,
                   object_value: str, user_id: str = "", org_id: str = "",
                   repo_id_val: str = "", confidence: float = 0.7,
                   source_text: str = "", is_negation: bool = False,
                   source_type: str = "user_stated", metadata: dict | None = None,
                   source_tool: str = "direct") -> WriteResult:
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
        return self._store(db, [mem], cross_scope_dbs=[], user_id=user_id,
                          source_tool=source_tool)

    def ingest_repo(self, user_id: str, repo_path: str | Path, commit_limit: int = 50) -> WriteResult:
        rid = repo_id(repo_path)
        db = self._db(Scope.REPO, repo_id=rid)
        mems = commits_to_memories(recent_commits(repo_path, commit_limit), rid)
        mems += tree_to_memories(file_tree(repo_path), rid)
        return self._store(db, mems, cross_scope_dbs=[], user_id=user_id,
                          source_tool="repo_parser")

    def store_from_tool(self, user_id: str, text: str,
                         source_tool: str, **kwargs) -> WriteResult:
        """Store facts from any tool. Uses grammar-free extraction and
        full normalization + dedup pipeline.

        This is the preferred entry point for multi-tool integrations.
        """
        db = self._db(Scope.PRIVATE, user_id=user_id)
        normalizer = self._get_normalizer(user_id)
        normalizer.learn_from_text(text)

        # Grammar-free extraction (handles any text format)
        mems = self._gf_extractor.extract(
            text, user_id, source_tool=source_tool,
            scope="private", context="personal",
            subject=kwargs.get("subject", user_id))

        # Also run grammar-based extraction for sentence-like text
        grammar_mems = extract_personal(text, user_id,
                                        kwargs.get("subject", ""),
                                        extractor=self._extractor,
                                        use_spacy=self.config.use_spacy)
        mems.extend(grammar_mems)

        return self._store(db, mems, cross_scope_dbs=[], user_id=user_id,
                          source_tool=source_tool)

    def get_provenance(self, fact_id: str) -> list:
        """Get provenance records for a fact."""
        return self._provenance.get_provenance(fact_id)

    def normalization_stats(self, user_id: str) -> dict:
        """Get normalization statistics for a user."""
        normalizer = self._get_normalizer(user_id)
        return normalizer.stats()

    def dedup_stats(self, user_id: str) -> dict:
        """Get deduplication statistics for a user."""
        dedup = self._get_dedup(user_id)
        return dedup.stats.to_dict()

    # --- Read API ---

    def recall(self, user_id: str, query: str, org_id: str = "", repo_id_val: str = "",
               context: str | None = None, top_k: int = 20,
               predicate_hint: str | list[str] | None = None,
               subject_hint: str | None = None) -> list[SearchResult]:
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
            user_id=user_id,
            predicate_hint=predicate_hint,
            subject_hint=subject_hint,
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
                if not _is_profile_worthy(f):
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
            r = run_consolidation(db, replay_traces=replay_traces, schema=self._schema)
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

        from lore_memory.safety import SUSPICIOUS_SOURCE_TYPE
        for f in facts:
            if not f.is_active or not _is_profile_worthy(f):
                continue
            # Injection-flagged memories never enter an LLM context directly.
            if f.source_type == SUSPICIOUS_SOURCE_TYPE:
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

    @property
    def schema(self) -> Schema:
        """The schema governing identity rewrites, supersession, and decay."""
        return self._schema

    def _apply_write_flags(self, mems: list[Memory], raw_text: str) -> None:
        """Tag each memory with write-time safety + provenance flags.

        Three tags can be applied, mutating mems in place:
          * source_type = "suspicious" — the raw input looks like a
            prompt-injection payload.
          * metadata["hypothetical"] = True — the input is a hedge /
            conditional / speculation; confidence is halved and the
            supersession check is skipped downstream.
          * metadata["source_speaker"] = <speaker> — the grammar
            parser's subject contained a reported-speech verb; the fact
            is attributed, not the user's own assertion.
        """
        from lore_memory.safety import SUSPICIOUS_SOURCE_TYPE, detect_speaker

        is_injection = (self.config.injection_defense
                        and self._injection_clf.is_injection(raw_text))
        is_hypothetical = (self.config.hypothetical_detection
                           and self._hypothetical_clf.is_hypothetical(raw_text))

        # When injection or hypothetical is detected, drop extracted triples —
        # keep only the raw stated row for audit/recall. Structured triples
        # from these inputs would otherwise outrank legitimate facts via
        # recency even with supersession skipped. The journal row remains so
        # the user can search for what they wrote; it's tagged accordingly.
        if is_injection:
            kept: list[Memory] = []
            for m in mems:
                if m.predicate == "stated":
                    m.source_type = SUSPICIOUS_SOURCE_TYPE
                    kept.append(m)
            mems[:] = kept
        elif is_hypothetical:
            kept2: list[Memory] = []
            for m in mems:
                if m.predicate == "stated":
                    m.metadata = dict(m.metadata or {})
                    m.metadata["hypothetical"] = True
                    m.confidence = min(m.confidence, 0.35)
                    kept2.append(m)
            mems[:] = kept2

        for m in mems:
            if is_hypothetical:
                m.metadata = dict(m.metadata or {})
                m.metadata["hypothetical"] = True
                # Halve the stored confidence — posterior follows.
                m.confidence = min(m.confidence, 0.35)
            # Reported-speech attribution: "My wife said we..." → speaker
            # = "My wife", fact is about the speaker's claim, not user.
            speaker = detect_speaker(m.subject) if m.subject else None
            if speaker:
                m.metadata = dict(m.metadata or {})
                m.metadata["source_speaker"] = speaker
                m.subject = speaker
                m.confidence = min(m.confidence, 0.45)

    def _ensure_schema_compatible(self, user_id: str) -> None:
        """Record the current schema hash on first open; warn on mismatch.

        This does not block: past data remains readable and new writes follow
        the new rules. It just surfaces a schema change in the logs so the
        caller isn't silently confused about supersession/decay behavior.
        """
        import logging
        db = self._db(Scope.PRIVATE, user_id=user_id)
        stored = db.get_meta("schema_hash")
        current = self._schema.hash_key()
        if stored is None:
            db.set_meta("schema_hash", current)
            return
        if stored != current:
            logging.getLogger("lore_memory").warning(
                "Schema hash changed for user_id=%s (was %s, now %s). "
                "Past facts keep their stored predicates; new writes follow "
                "the new schema. If this is unintentional, re-open with the "
                "previous Schema to avoid behavioral drift.",
                user_id, stored, current)
            db.set_meta("schema_hash", current)

    # --- Internal ---

    def _store(self, db: MemoryDB, mems: list[Memory],
               cross_scope_dbs: list[MemoryDB],
               user_id: str = "", source_tool: str = "") -> WriteResult:
        result = WriteResult()
        normalizer = self._get_normalizer(user_id) if user_id else None
        dedup = self._get_dedup(user_id) if user_id else None

        for mem in mems:
            # --- Normalization ---
            # Preserve original surface forms for provenance
            orig_subject = mem.subject
            orig_predicate = mem.predicate
            orig_object = mem.object_value

            if normalizer and mem.predicate != "stated":
                canon_s, canon_p, canon_o = normalizer.normalize_triple(
                    mem.subject, mem.predicate, mem.object_value)
                # Normalize subject and object at write time
                mem.subject = canon_s
                mem.object_value = canon_o
                # Predicate: register in normalizer for clustering but keep
                # the grammar-extracted form for backward compatibility.
                # The dedup engine and belief system handle predicate aliases
                # via their own canon() at comparison time.

            if not mem.embedding:
                mem.embedding = self._embed(mem.triplet_text)

            # --- Advanced dedup check ---
            if dedup and mem.predicate != "stated":
                dedup_result = dedup.check(mem, db, embed_fn=self._embed)
                if dedup_result.is_duplicate:
                    if dedup_result.action == "reject":
                        result.deduplicated += 1
                        # Record provenance even for rejected duplicates
                        self._provenance.record(
                            dedup_result.existing_id, source_tool,
                            orig_subject, orig_predicate, orig_object,
                            mem.source_text)
                        continue
                    elif dedup_result.action == "merge":
                        result.deduplicated += 1
                        # Boost evidence on existing fact
                        db.record_feedback(dedup_result.existing_id, helpful=True)
                        self._provenance.record(
                            dedup_result.existing_id, source_tool,
                            orig_subject, orig_predicate, orig_object,
                            mem.source_text)
                        continue
                    elif dedup_result.action == "supersede":
                        # Mark old as superseded, store new
                        db.update_state(dedup_result.existing_id, "superseded")
                        self._provenance.mark_superseded(
                            dedup_result.existing_id, mem.id)
                        result.contradictions += 1

            # Skip contradiction checks for "stated" predicate — raw text
            # memories are never single-valued and never contradict each other.
            # Skip supersession for hypothetical / attributed / suspicious facts:
            #   * conditional ("if I quit") — user hasn't committed
            #   * reported claim ("my wife said we should move") — not user's own
            #   * suspicious (prompt-injection-shaped) — must not overwrite real
            #     facts even though the row is stored for audit
            from lore_memory.safety import SUSPICIOUS_SOURCE_TYPE as _SUSP
            _md = mem.metadata or {}
            _skip_supersede = (_md.get("hypothetical")
                               or _md.get("source_speaker")
                               or mem.source_type == _SUSP)
            if mem.predicate != "stated" and not _skip_supersede:
                contradicted = check_contradictions(db, mem, schema=self._schema)
                result.contradictions += len(contradicted)
                if cross_scope_dbs:
                    cross = check_cross_scope_contradictions(
                        cross_scope_dbs, mem.subject, mem.predicate, mem.object_value,
                        schema=self._schema)
                    result.contradictions += len(cross)
                # Loss events ("Luna passed away") supersede first-person
                # possession facts that name the affected entity.
                lost = check_loss_events(db, mem)
                result.contradictions += len(lost)

            dup = db.put(mem)
            if dup:
                result.deduplicated += 1
                # Record provenance for duplicate
                self._provenance.record(
                    dup, source_tool, orig_subject, orig_predicate,
                    orig_object, mem.source_text)
            else:
                result.created += 1
                # Record provenance for new fact
                self._provenance.record(
                    mem.id, source_tool, orig_subject, orig_predicate,
                    orig_object, mem.source_text)
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
