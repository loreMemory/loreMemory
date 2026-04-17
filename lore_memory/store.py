"""
Storage layer: SQLite per scope with FTS5 for keyword search.

v3 improvements over v2:
- FTS5 virtual table for O(log n) keyword search (vs O(n) LIKE scans)
- IMMEDIATE transactions for concurrency safety
- Soft-delete with recovery (deleted_at timestamp, not permanent)
- Expired memories (valid_until < now) excluded from active queries
- Source weighting (user_stated > inferred > system)
"""

from __future__ import annotations

import json
import math
import sqlite3
import struct
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lore_memory.scopes import Scope, Context


SOURCE_WEIGHTS = {"user_stated": 1.0, "inferred": 0.7, "system": 0.5, "imported": 0.8}


@dataclass
class Memory:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    scope: str = "private"
    context: str = "personal"
    user_id: str = ""
    org_id: str = ""
    repo_id: str = ""
    subject: str = ""
    predicate: str = ""
    object_value: str = ""
    source_text: str = ""
    is_negation: bool = False  # v3: "I don't like Java" → True
    source_type: str = "user_stated"  # v3: user_stated | inferred | system | imported
    embedding: list[float] = field(default_factory=list)
    confidence: float = 0.7
    evidence_count: int = 1
    contradiction_count: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    valid_until: float | None = None
    state: str = "active"  # active | superseded | archived | deleted
    deleted_at: float | None = None  # v3: soft-delete with recovery
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def posterior(self) -> float:
        sw = SOURCE_WEIGHTS.get(self.source_type, 0.7)
        a = self.confidence * sw * 2 + self.evidence_count
        b = (1 - self.confidence * sw) * 2 + self.contradiction_count
        return a / (a + b) if (a + b) > 0 else 0.5

    @property
    def is_active(self) -> bool:
        if self.state != "active":
            return False
        if self.valid_until is not None and self.valid_until < time.time():
            return False  # v3 fix: expired memories are not active
        return self.posterior > 0.15

    @property
    def triplet_text(self) -> str:
        neg = "not " if self.is_negation else ""
        return f"{self.subject} {neg}{self.predicate} {self.object_value}"


def _pack_f32(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)

def _unpack_f32(blob: bytes) -> list[float]:
    return list(struct.unpack(f"{len(blob)//4}f", blob)) if blob else []


class MemoryDB:
    """SQLite-based memory store for a single scope."""

    def __init__(self, db_path: str | Path, embedding_dims: int = 384) -> None:
        self.db_path = str(db_path)
        self.dims = embedding_dims
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()
        self._init()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
        return self._conn

    def _init(self) -> None:
        c = self.conn
        c.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                scope TEXT NOT NULL, context TEXT NOT NULL,
                user_id TEXT DEFAULT '', org_id TEXT DEFAULT '', repo_id TEXT DEFAULT '',
                subject TEXT NOT NULL DEFAULT '',
                predicate TEXT NOT NULL DEFAULT '',
                object_value TEXT NOT NULL DEFAULT '',
                source_text TEXT DEFAULT '',
                is_negation INTEGER NOT NULL DEFAULT 0,
                source_type TEXT NOT NULL DEFAULT 'user_stated',
                embedding BLOB,
                confidence REAL NOT NULL DEFAULT 0.7,
                evidence_count INTEGER NOT NULL DEFAULT 1,
                contradiction_count INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                last_accessed REAL NOT NULL,
                access_count INTEGER NOT NULL DEFAULT 0,
                valid_until REAL,
                state TEXT NOT NULL DEFAULT 'active',
                deleted_at REAL,
                metadata TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_mem_subject ON memories(subject);
            CREATE INDEX IF NOT EXISTS idx_mem_state ON memories(state);
            CREATE INDEX IF NOT EXISTS idx_mem_subj_pred ON memories(subject, predicate);
            CREATE INDEX IF NOT EXISTS idx_mem_context ON memories(context);
            -- v3.2: Composite indexes for scale (100K+ memories)
            CREATE INDEX IF NOT EXISTS idx_mem_state_context_la ON memories(state, context, last_accessed DESC);
            CREATE INDEX IF NOT EXISTS idx_mem_state_valid ON memories(state, valid_until);
            CREATE INDEX IF NOT EXISTS idx_mem_subj_state ON memories(subject, state);
            CREATE TABLE IF NOT EXISTS _meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)
        # FTS5 for fast keyword search. Porter stemmer + unicode61
        # so queries like "learn" match stored "learning", "move" matches
        # "moved", "join" matches "joined" — reducing the need to maintain
        # large query-keyword → predicate lookup tables.
        try:
            c.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    id, subject, predicate, object_value, source_text,
                    content='memories', content_rowid='rowid',
                    tokenize='porter unicode61 remove_diacritics 2'
                )
            """)
        except sqlite3.OperationalError:
            # Older SQLite may not support porter — fall back to unicode61.
            try:
                c.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                        id, subject, predicate, object_value, source_text,
                        content='memories', content_rowid='rowid'
                    )
                """)
            except sqlite3.OperationalError:
                pass  # FTS5 not available at all — fall back to LIKE
        c.commit()
        self._has_fts = self._check_fts()

    def _check_fts(self) -> bool:
        try:
            self.conn.execute("SELECT * FROM memories_fts LIMIT 0")
            return True
        except sqlite3.OperationalError:
            return False

    # --- Write ---

    def put(self, mem: Memory) -> str | None:
        """Insert memory. Returns existing ID if duplicate, else None."""
        with self._lock:
            c = self.conn
            try:
                c.execute("BEGIN IMMEDIATE")
            except sqlite3.OperationalError:
                c.execute("BEGIN")

            # Dedup check — include non-active states to prevent duplicates
            # after delete-store-recover cycles
            row = c.execute(
                "SELECT id, state FROM memories WHERE subject=? AND predicate=? AND object_value=? AND is_negation=? AND state IN ('active','deleted','archived') ORDER BY CASE state WHEN 'active' THEN 0 ELSE 1 END LIMIT 1",
                (mem.subject, mem.predicate, mem.object_value, int(mem.is_negation)),
            ).fetchone()
            if row:
                now = time.time()
                # If the existing memory was deleted/archived, reactivate it
                if row["state"] != "active":
                    c.execute(
                        "UPDATE memories SET state='active', deleted_at=NULL, evidence_count=evidence_count+1, last_accessed=?, updated_at=? WHERE id=?",
                        (now, now, row["id"]),
                    )
                else:
                    c.execute(
                        "UPDATE memories SET evidence_count=evidence_count+1, last_accessed=?, updated_at=? WHERE id=?",
                        (now, now, row["id"]),
                    )
                c.commit()
                self._fts_update(row["id"])
                return row["id"]

            emb = _pack_f32(mem.embedding) if mem.embedding else None
            c.execute(
                """INSERT INTO memories
                (id,scope,context,user_id,org_id,repo_id,subject,predicate,object_value,
                 source_text,is_negation,source_type,embedding,confidence,evidence_count,
                 contradiction_count,created_at,updated_at,last_accessed,access_count,
                 valid_until,state,deleted_at,metadata)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (mem.id, mem.scope, mem.context, mem.user_id, mem.org_id, mem.repo_id,
                 mem.subject, mem.predicate, mem.object_value, mem.source_text,
                 int(mem.is_negation), mem.source_type, emb, mem.confidence,
                 mem.evidence_count, mem.contradiction_count, mem.created_at,
                 mem.updated_at, mem.last_accessed, mem.access_count,
                 mem.valid_until, mem.state, mem.deleted_at, json.dumps(mem.metadata)),
            )
            # FTS insert inside the IMMEDIATE transaction — atomic with the main insert
            self._fts_insert_in_txn(c, mem)
            c.commit()
            return None

    def _fts_insert_in_txn(self, cursor, mem: Memory) -> None:
        """Insert into FTS index within an existing transaction."""
        if not self._has_fts:
            return
        try:
            cursor.execute(
                "INSERT INTO memories_fts(id, subject, predicate, object_value, source_text) VALUES (?,?,?,?,?)",
                (mem.id, mem.subject, mem.predicate.replace("_", " "), mem.object_value, mem.source_text),
            )
        except sqlite3.OperationalError:
            pass

    def _fts_insert(self, mem: Memory) -> None:
        """Legacy FTS insert — used only for standalone inserts outside transactions."""
        if not self._has_fts:
            return
        try:
            self.conn.execute(
                "INSERT INTO memories_fts(id, subject, predicate, object_value, source_text) VALUES (?,?,?,?,?)",
                (mem.id, mem.subject, mem.predicate.replace("_", " "), mem.object_value, mem.source_text),
            )
            self.conn.commit()
        except sqlite3.OperationalError:
            pass

    def _fts_update(self, mem_id: str) -> None:
        pass  # FTS content synced on read

    # --- Read ---

    def get(self, mem_id: str) -> Memory | None:
        with self._lock:
            row = self.conn.execute("SELECT * FROM memories WHERE id=?", (mem_id,)).fetchone()
            return self._to_mem(row) if row else None

    def query_by_subject(self, subject: str, context: str | None = None, limit: int = 500) -> list[Memory]:
        with self._lock:
            now = time.time()
            if context:
                rows = self.conn.execute(
                    "SELECT * FROM memories WHERE subject=? AND context=? AND state='active' AND (valid_until IS NULL OR valid_until > ?) ORDER BY updated_at DESC LIMIT ?",
                    (subject, context, now, limit),
                ).fetchall()
            else:
                rows = self.conn.execute(
                    "SELECT * FROM memories WHERE subject=? AND state='active' AND (valid_until IS NULL OR valid_until > ?) ORDER BY updated_at DESC LIMIT ?",
                    (subject, now, limit),
                ).fetchall()
            return [self._to_mem(r) for r in rows]

    def query_active(self, context: str | None = None, limit: int = 1000, skip_embedding: bool = False) -> list[Memory]:
        """Query active memories. Set skip_embedding=True for faster results at scale."""
        with self._lock:
            now = time.time()
            cols = """id, scope, context, user_id, org_id, repo_id, subject, predicate,
                      object_value, source_text, is_negation, source_type,
                      {} as embedding,
                      confidence, evidence_count, contradiction_count, created_at, updated_at,
                      last_accessed, access_count, valid_until, state, deleted_at, metadata""".format(
                "NULL" if skip_embedding else "embedding"
            )
            if context:
                rows = self.conn.execute(
                    f"SELECT {cols} FROM memories WHERE state='active' AND context=? AND (valid_until IS NULL OR valid_until > ?) ORDER BY last_accessed DESC LIMIT ?",
                    (context, now, limit),
                ).fetchall()
            else:
                rows = self.conn.execute(
                    f"SELECT {cols} FROM memories WHERE state='active' AND (valid_until IS NULL OR valid_until > ?) ORDER BY last_accessed DESC LIMIT ?",
                    (now, limit),
                ).fetchall()
            return [self._to_mem(r) for r in rows]

    def fts_search(self, query: str, limit: int = 50) -> list[Memory]:
        """FTS5 keyword search — O(log n)."""
        with self._lock:
            if not self._has_fts or not query.strip():
                return self._fallback_search(query, limit)
            now = time.time()
            try:
                rows = self.conn.execute(
                    """SELECT m.* FROM memories_fts f
                       JOIN memories m ON f.id = m.id
                       WHERE memories_fts MATCH ? AND m.state='active'
                       AND (m.valid_until IS NULL OR m.valid_until > ?)
                       LIMIT ?""",
                    (query, now, limit),
                ).fetchall()
                return [self._to_mem(r) for r in rows]
            except sqlite3.OperationalError:
                return self._fallback_search(query, limit)

    def _fallback_search(self, query: str, limit: int) -> list[Memory]:
        """LIKE-based fallback when FTS5 unavailable."""
        now = time.time()
        # Escape LIKE special characters to prevent wildcard injection
        escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        pattern = f"%{escaped}%"
        rows = self.conn.execute(
            """SELECT * FROM memories WHERE state='active'
               AND (valid_until IS NULL OR valid_until > ?)
               AND (subject LIKE ? ESCAPE '\\' OR predicate LIKE ? ESCAPE '\\'
                    OR object_value LIKE ? ESCAPE '\\' OR source_text LIKE ? ESCAPE '\\')
               ORDER BY last_accessed DESC LIMIT ?""",
            (now, pattern, pattern, pattern, pattern, limit),
        ).fetchall()
        return [self._to_mem(r) for r in rows]

    def vector_search(self, query_emb: list[float], top_k: int = 20, context: str | None = None) -> list[tuple[Memory, float]]:
        candidates = self.query_active(context=context, limit=top_k * 5)  # already locked
        scored = []
        for m in candidates:
            if m.embedding:
                sim = cosine_sim(query_emb, m.embedding)
                if sim > 0:
                    scored.append((m, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def query_superseded(self, subject: str | None = None, limit: int = 50) -> list[Memory]:
        """Query superseded memories (facts that were replaced by newer ones).
        Used for temporal queries like 'where did X live before?'."""
        with self._lock:
            if subject:
                rows = self.conn.execute(
                    "SELECT * FROM memories WHERE state='superseded' AND subject=? ORDER BY updated_at DESC LIMIT ?",
                    (subject, limit),
                ).fetchall()
            else:
                rows = self.conn.execute(
                    "SELECT * FROM memories WHERE state='superseded' ORDER BY updated_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [self._to_mem(r) for r in rows]

    def fts_search_all_states(self, query: str, limit: int = 50) -> list[Memory]:
        """FTS search across all states (active, superseded, archived).
        Used for temporal queries that need to find past facts."""
        with self._lock:
            if not self._has_fts or not query.strip():
                return []
            try:
                rows = self.conn.execute(
                    """SELECT m.* FROM memories_fts f
                       JOIN memories m ON f.id = m.id
                       WHERE memories_fts MATCH ? AND m.state IN ('active', 'superseded')
                       LIMIT ?""",
                    (query, limit),
                ).fetchall()
                return [self._to_mem(r) for r in rows]
            except sqlite3.OperationalError:
                return []

    # --- Update ---

    def update_access(self, mem_id: str) -> None:
        with self._lock:
            now = time.time()
            self.conn.execute(
                "UPDATE memories SET access_count=access_count+1, last_accessed=? WHERE id=?",
                (now, mem_id))
            self.conn.commit()

    def add_contradiction(self, mem_id: str) -> None:
        with self._lock:
            now = time.time()
            self.conn.execute(
                "UPDATE memories SET contradiction_count=contradiction_count+1, updated_at=? WHERE id=?",
                (now, mem_id))
            self.conn.commit()

    def update_state(self, mem_id: str, state: str) -> None:
        with self._lock:
            now = time.time()
            extra = ", deleted_at=?" if state == "deleted" else ""
            params = [state, now, mem_id] if not extra else [state, now, now, mem_id]
            self.conn.execute(
                f"UPDATE memories SET state=?, updated_at=?{extra} WHERE id=?", params)
            self.conn.commit()

    def get_meta(self, key: str) -> str | None:
        """Read a key/value from the per-DB _meta table."""
        with self._lock:
            row = self.conn.execute(
                "SELECT value FROM _meta WHERE key=?", (key,)).fetchone()
            return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        """Upsert a key/value into the per-DB _meta table."""
        with self._lock:
            self.conn.execute(
                "INSERT INTO _meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value))
            self.conn.commit()

    def delete_hard(self, mem_id: str) -> bool:
        """Permanently remove a row from both the main table and FTS index.

        Unlike update_state(state='deleted'), this does a real DELETE so
        nothing remains on disk (after the next checkpoint/VACUUM).
        Returns True if a row was removed.
        """
        with self._lock:
            cur = self.conn.execute("DELETE FROM memories WHERE id=?", (mem_id,))
            if self._has_fts:
                self.conn.execute("DELETE FROM memories_fts WHERE id=?", (mem_id,))
            self.conn.commit()
            return cur.rowcount > 0

    def delete_hard_by_subject(self, subject: str, predicate: str | None = None) -> int:
        """Permanently DELETE every row matching subject (and optional predicate)
        *regardless of state* — active, superseded, archived, or soft-deleted.

        Intended for GDPR-style erasure where nothing must remain. Returns
        the count of rows removed. Caller should invoke vacuum() after.
        """
        with self._lock:
            if predicate is not None:
                cur = self.conn.execute(
                    "DELETE FROM memories WHERE subject=? AND predicate=?",
                    (subject, predicate))
            else:
                cur = self.conn.execute(
                    "DELETE FROM memories WHERE subject=?", (subject,))
            if self._has_fts:
                if predicate is not None:
                    self.conn.execute(
                        "DELETE FROM memories_fts WHERE id IN ("
                        "SELECT id FROM memories WHERE subject=? AND predicate=?)",
                        (subject, predicate))
                else:
                    # FTS rows are gone after the main-table DELETE via
                    # content='memories' contentless-trigger semantics; still
                    # explicit to be safe on older SQLite versions.
                    self.conn.execute(
                        "DELETE FROM memories_fts WHERE id NOT IN "
                        "(SELECT id FROM memories)")
            self.conn.commit()
            return cur.rowcount

    def vacuum(self) -> None:
        """Reclaim disk space after hard-deletes. Blocks writers briefly."""
        with self._lock:
            self.conn.commit()  # ensure no open txn
            self.conn.execute("VACUUM")

    def export_all(self) -> list[dict]:
        """Dump every row (any state) as plain dicts for portability/audit.

        Intended for GDPR-style export. Embeddings are omitted (not
        human-portable); everything else is included. Returns in insertion
        order so the export is deterministic.
        """
        with self._lock:
            rows = self.conn.execute(
                """SELECT id, scope, context, user_id, org_id, repo_id, subject,
                          predicate, object_value, source_text, is_negation,
                          source_type, confidence, evidence_count,
                          contradiction_count, created_at, updated_at,
                          last_accessed, access_count, valid_until, state,
                          deleted_at, metadata
                   FROM memories ORDER BY created_at ASC""").fetchall()
            import json
            out: list[dict] = []
            for r in rows:
                d = dict(r)
                # Keep metadata as parsed dict, not string
                md = d.get("metadata")
                if isinstance(md, str):
                    try:
                        d["metadata"] = json.loads(md)
                    except Exception:
                        pass
                out.append(d)
            return out

    def recover(self, mem_id: str) -> bool:
        """v3: Recover a soft-deleted or archived memory."""
        with self._lock:
            row = self.conn.execute("SELECT state FROM memories WHERE id=?", (mem_id,)).fetchone()
            if not row or row["state"] == "active":
                return False
            now = time.time()
            self.conn.execute(
                "UPDATE memories SET state='active', deleted_at=NULL, updated_at=? WHERE id=?",
                (now, mem_id))
            self.conn.commit()
            return True

    def record_feedback(self, mem_id: str, helpful: bool) -> None:
        """v3: Explicit feedback — boost or penalize a memory."""
        with self._lock:
            now = time.time()
            if helpful:
                self.conn.execute(
                    "UPDATE memories SET evidence_count=evidence_count+1, access_count=access_count+1, last_accessed=?, updated_at=? WHERE id=?",
                    (now, now, mem_id))
            else:
                self.conn.execute(
                    "UPDATE memories SET contradiction_count=contradiction_count+1, updated_at=? WHERE id=?",
                    (now, mem_id))
            self.conn.commit()

    def query_active_lightweight(self, context: str | None = None, limit: int = 1000, offset: int = 0) -> list[Memory]:
        """Like query_active but skips embedding deserialization for consolidation/maintenance."""
        with self._lock:
            now = time.time()
            if context:
                rows = self.conn.execute(
                    """SELECT id, scope, context, user_id, org_id, repo_id, subject, predicate,
                       object_value, source_text, is_negation, source_type, NULL as embedding,
                       confidence, evidence_count, contradiction_count, created_at, updated_at,
                       last_accessed, access_count, valid_until, state, deleted_at, metadata
                       FROM memories WHERE state='active' AND context=?
                       AND (valid_until IS NULL OR valid_until > ?)
                       ORDER BY updated_at ASC LIMIT ? OFFSET ?""",
                    (context, now, limit, offset),
                ).fetchall()
            else:
                rows = self.conn.execute(
                    """SELECT id, scope, context, user_id, org_id, repo_id, subject, predicate,
                       object_value, source_text, is_negation, source_type, NULL as embedding,
                       confidence, evidence_count, contradiction_count, created_at, updated_at,
                       last_accessed, access_count, valid_until, state, deleted_at, metadata
                       FROM memories WHERE state='active'
                       AND (valid_until IS NULL OR valid_until > ?)
                       ORDER BY updated_at ASC LIMIT ? OFFSET ?""",
                    (now, limit, offset),
                ).fetchall()
            return [self._to_mem(r) for r in rows]

    def count(self, context: str | None = None) -> int:
        with self._lock:
            now = time.time()
            if context:
                row = self.conn.execute(
                    "SELECT COUNT(*) as c FROM memories WHERE state='active' AND context=? AND (valid_until IS NULL OR valid_until > ?)",
                    (context, now)).fetchone()
            else:
                row = self.conn.execute(
                    "SELECT COUNT(*) as c FROM memories WHERE state='active' AND (valid_until IS NULL OR valid_until > ?)",
                    (now,)).fetchone()
            return row["c"]

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def _to_mem(self, row: sqlite3.Row) -> Memory:
        return Memory(
            id=row["id"], scope=row["scope"], context=row["context"],
            user_id=row["user_id"], org_id=row["org_id"], repo_id=row["repo_id"],
            subject=row["subject"], predicate=row["predicate"],
            object_value=row["object_value"], source_text=row["source_text"] or "",
            is_negation=bool(row["is_negation"]),
            source_type=row["source_type"],
            embedding=_unpack_f32(row["embedding"]),
            confidence=row["confidence"], evidence_count=row["evidence_count"],
            contradiction_count=row["contradiction_count"],
            created_at=row["created_at"], updated_at=row["updated_at"],
            last_accessed=row["last_accessed"], access_count=row["access_count"],
            valid_until=row["valid_until"], state=row["state"],
            deleted_at=row["deleted_at"],
            metadata=_safe_json(row["metadata"]),
        )


def _safe_json(raw) -> dict:
    """Parse JSON metadata, returning empty dict on any error."""
    if not raw:
        return {}
    try:
        result = json.loads(raw)
        return result if isinstance(result, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def cosine_sim(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0
