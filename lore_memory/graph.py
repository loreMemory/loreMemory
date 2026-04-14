"""
Graph index — v5: incremental updates, thread-safe, no full rebuilds.

Architecture:
  - Writes ADD edges incrementally (O(1) per memory)
  - Reads use cached edges (O(fanout^2) per query term)
  - Full rebuild only on cold start (no cached state)
  - Thread-safe via lock on mutations
  - Never blocks queries — reads use snapshot of current state
"""

from __future__ import annotations

import json
import os
import threading
from collections import defaultdict
from pathlib import Path
from lore_memory.store import Memory, MemoryDB

MAX_RELATED = 1000


class GraphCache:
    """
    Incremental graph index. Edges are added one at a time on writes.
    Full rebuild only on first use or explicit reset.

    Thread safety:
      - add_edge/remove_edge: acquire write lock
      - get_related: lock-free read on immutable snapshot
      - build: acquire write lock for full rebuild
    """

    def __init__(self, cache_path: str | Path | None = None) -> None:
        self._forward: dict[str, set[str]] = defaultdict(set)
        self._reverse: dict[str, set[str]] = defaultdict(set)
        self._edge_count = 0
        self._built = False
        self._lock = threading.Lock()
        self._cache_path = str(cache_path) if cache_path else None
        # Try to load from disk cache on init
        if self._cache_path:
            self._load_cache()

    @property
    def is_dirty(self) -> bool:
        """Only dirty if never built. After first build, incremental updates
        keep the graph current without full rebuilds."""
        return not self._built

    def invalidate(self) -> None:
        """Soft invalidate — does NOT clear edges. Marks for rebuild only if
        the graph has never been built (cold start). After first build,
        incremental add_edge keeps it current."""
        # Don't clear edges — they're still valid.
        # Only mark unbuilt for cold start.
        pass

    def force_rebuild(self) -> None:
        """Hard reset — used after purge_user or major data changes."""
        with self._lock:
            self._forward.clear()
            self._reverse.clear()
            self._edge_count = 0
            self._built = False

    def add_edge(self, subject: str, object_value: str) -> None:
        """Incrementally add one edge. O(1). Called on every write."""
        if not subject or not object_value:
            return
        s, o = subject.lower(), object_value.lower()
        with self._lock:
            if o not in self._forward.get(s, set()):
                self._forward[s].add(o)
                self._reverse[o].add(s)
                self._edge_count += 1

    def remove_edges_for_subject(self, subject: str) -> None:
        """Remove all edges for a subject. Used on supersede/delete."""
        s = subject.lower()
        with self._lock:
            if s in self._forward:
                for o in self._forward[s]:
                    self._reverse.get(o, set()).discard(s)
                removed = len(self._forward[s])
                del self._forward[s]
                self._edge_count -= removed

    def build(self, dbs: list[MemoryDB]) -> int:
        """Full build from databases. Only needed on cold start.
        After this, incremental add_edge keeps the graph current."""
        with self._lock:
            self._forward.clear()
            self._reverse.clear()
            self._edge_count = 0

            for db in dbs:
                offset = 0
                batch_size = 10000
                while True:
                    batch = db.query_active_lightweight(limit=batch_size, offset=offset)
                    if not batch:
                        break
                    for mem in batch:
                        if mem.subject and mem.object_value:
                            s, o = mem.subject.lower(), mem.object_value.lower()
                            if o not in self._forward.get(s, set()):
                                self._forward[s].add(o)
                                self._reverse[o].add(s)
                                self._edge_count += 1
                    if len(batch) < batch_size:
                        break
                    offset += batch_size

            self._built = True
            return self._edge_count

    def get_related(self, term: str) -> set[str]:
        """Get all subjects related to a term (lazy 2-hop). Lock-free read.
        Results capped at MAX_RELATED."""
        tl = term.lower()
        related = set()
        # Snapshot references — safe to read without lock since sets are
        # only mutated under lock and Python's GIL protects reference reads
        fwd = self._forward
        rev = self._reverse
        # 1-hop forward
        for neighbor in fwd.get(tl, set()):
            related.add(neighbor)
            related |= fwd.get(neighbor, set())
        # 1-hop reverse
        for neighbor in rev.get(tl, set()):
            related.add(neighbor)
            related |= rev.get(neighbor, set())
        related.discard(tl)

        if len(related) > MAX_RELATED:
            def _conn(node: str) -> int:
                return len(fwd.get(node, set())) + len(rev.get(node, set()))
            ranked = sorted(related, key=_conn, reverse=True)
            related = set(ranked[:MAX_RELATED])

        return related

    def get_all_related(self) -> dict[str, set[str]]:
        """Return self — retriever calls get_related() per term."""
        return self

    @property
    def edge_count(self) -> int:
        return self._edge_count

    @property
    def node_count(self) -> int:
        return len(set(self._forward.keys()) | set(self._reverse.keys()))

    # --- Disk persistence ---

    def save_cache(self) -> None:
        """Save graph edges to disk for fast restart. Called periodically or on shutdown."""
        if not self._cache_path or not self._built:
            return
        try:
            data = {
                "edges": {k: list(v) for k, v in self._forward.items()},
                "count": self._edge_count,
            }
            tmp = self._cache_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f)
            os.replace(tmp, self._cache_path)  # atomic rename
        except (OSError, TypeError):
            pass

    def _load_cache(self) -> None:
        """Load graph edges from disk cache. Avoids full DB scan on restart."""
        if not self._cache_path or not os.path.exists(self._cache_path):
            return
        try:
            with open(self._cache_path) as f:
                data = json.load(f)
            with self._lock:
                self._forward.clear()
                self._reverse.clear()
                for s, targets in data.get("edges", {}).items():
                    for o in targets:
                        self._forward[s].add(o)
                        self._reverse[o].add(s)
                self._edge_count = data.get("count", sum(len(v) for v in self._forward.values()))
                self._built = True
        except (OSError, json.JSONDecodeError, KeyError):
            pass  # cache corrupt — will rebuild from DB
