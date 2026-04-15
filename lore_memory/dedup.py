"""
Deduplication engine — 5-type duplicate detection and resolution.

Handles:
  Type 1: Exact duplicate (same S+P+O+negation) — O(1) hash
  Type 2: Near duplicate (same meaning, different surface) — embedding similarity
  Type 3: Same S+P, different O (one outdated) — single-valued check
  Type 4: Cross-tool duplicate (same fact, different predicates) — predicate clustering
  Type 5: Temporal duplicate (fact was true, now stale) — timestamp + expiry

Conservative: false positive dedup (merging different facts) is treated as
worse than false negative (missing a duplicate). Tune thresholds accordingly.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from lore_memory.store import Memory, MemoryDB, cosine_sim
from lore_memory.normalization import NormalizationPipeline
from lore_memory.belief import is_single_valued


@dataclass
class DedupResult:
    """Result of deduplication check on a single memory."""
    is_duplicate: bool = False
    duplicate_type: str = ""  # "exact", "near", "same_sp", "cross_tool", "temporal"
    existing_id: str = ""  # ID of the existing memory this duplicates
    confidence: float = 0.0  # How confident the dedup decision is (0-1)
    action: str = ""  # "reject", "merge", "supersede", "conflict"
    original_surface: dict[str, str] = field(default_factory=dict)  # Pre-normalization values


@dataclass
class DedupStats:
    exact: int = 0
    near: int = 0
    same_sp: int = 0
    cross_tool: int = 0
    temporal: int = 0
    false_positives: int = 0  # Tracked by user corrections
    total_checked: int = 0

    @property
    def total_deduped(self) -> int:
        return self.exact + self.near + self.same_sp + self.cross_tool + self.temporal

    def to_dict(self) -> dict:
        return {
            "exact": self.exact,
            "near": self.near,
            "same_sp": self.same_sp,
            "cross_tool": self.cross_tool,
            "temporal": self.temporal,
            "false_positives": self.false_positives,
            "total_checked": self.total_checked,
            "total_deduped": self.total_deduped,
        }


class DedupEngine:
    """Multi-tier deduplication engine.

    Checks duplicates in order from cheapest to most expensive:
    1. Exact match (hash-based, O(1))
    2. Near duplicate (embedding similarity, O(n) on active memories)
    3. Same S+P different O (single-valued predicate check)
    4. Cross-tool (predicate cluster + subject/object normalization)
    5. Temporal (timestamp comparison)
    """

    def __init__(self, normalizer: NormalizationPipeline,
                 near_threshold: float = 0.92,
                 cross_tool_threshold: float = 0.88) -> None:
        self._normalizer = normalizer
        self._near_threshold = near_threshold
        self._cross_tool_threshold = cross_tool_threshold
        self.stats = DedupStats()

    def check(self, new_mem: Memory, db: MemoryDB,
              embed_fn=None) -> DedupResult:
        """Run all dedup checks on a new memory before storage.

        Returns DedupResult indicating if this is a duplicate and what to do.
        """
        self.stats.total_checked += 1

        if new_mem.predicate == "stated":
            # Raw text memories use exact dedup only (via db.put)
            return DedupResult()

        # Type 1: Exact duplicate
        result = self._check_exact(new_mem, db)
        if result.is_duplicate:
            self.stats.exact += 1
            return result

        # Type 2: Near duplicate (embedding similarity)
        if embed_fn and new_mem.embedding:
            result = self._check_near(new_mem, db)
            if result.is_duplicate:
                self.stats.near += 1
                return result

        # Type 3: Same subject+predicate, different object (single-valued)
        result = self._check_same_sp(new_mem, db)
        if result.is_duplicate:
            self.stats.same_sp += 1
            return result

        # Type 4: Cross-tool duplicate (different predicate, same meaning)
        result = self._check_cross_tool(new_mem, db)
        if result.is_duplicate:
            self.stats.cross_tool += 1
            return result

        # Type 5: Temporal duplicate
        result = self._check_temporal(new_mem, db)
        if result.is_duplicate:
            self.stats.temporal += 1
            return result

        return DedupResult()

    def _check_exact(self, new_mem: Memory, db: MemoryDB) -> DedupResult:
        """Type 1: Exact (S, P, O, negation) match."""
        existing = db.query_by_subject(new_mem.subject, limit=200)
        for ex in existing:
            if (ex.predicate == new_mem.predicate
                    and ex.object_value == new_mem.object_value
                    and ex.is_negation == new_mem.is_negation
                    and ex.state == "active"):
                return DedupResult(
                    is_duplicate=True,
                    duplicate_type="exact",
                    existing_id=ex.id,
                    confidence=1.0,
                    action="reject",
                )
        return DedupResult()

    def _check_near(self, new_mem: Memory, db: MemoryDB) -> DedupResult:
        """Type 2: Same meaning, different surface form.

        Uses embedding similarity on the full triplet text.
        Only matches within the same subject (after normalization).
        """
        if not new_mem.embedding:
            return DedupResult()

        existing = db.query_by_subject(new_mem.subject, limit=200)
        for ex in existing:
            if ex.state != "active" or not ex.embedding:
                continue
            if ex.predicate == "stated":
                continue  # Don't dedup structured against raw text

            # Never dedup across negation boundaries
            if ex.is_negation != new_mem.is_negation:
                continue

            # Same subject+predicate but different objects on multi-valued predicates → not duplicate
            if (ex.subject == new_mem.subject
                    and ex.predicate == new_mem.predicate
                    and ex.object_value.lower() != new_mem.object_value.lower()
                    and not is_single_valued(new_mem.predicate)):
                continue

            sim = cosine_sim(new_mem.embedding, ex.embedding)
            if sim >= self._near_threshold:
                # Additional check: predicates should be related
                pred_cluster = self._normalizer.predicate_normalizer.get_cluster_members(
                    new_mem.predicate)
                if ex.predicate in pred_cluster or new_mem.predicate == ex.predicate:
                    return DedupResult(
                        is_duplicate=True,
                        duplicate_type="near",
                        existing_id=ex.id,
                        confidence=sim,
                        action="merge",
                    )
        return DedupResult()

    def _check_same_sp(self, new_mem: Memory, db: MemoryDB) -> DedupResult:
        """Type 3: Same subject + predicate, different object.

        For single-valued predicates, the new value supersedes the old.
        For multi-valued predicates, both can coexist.
        """
        from lore_memory.belief import is_single_valued, canon

        if not is_single_valued(new_mem.predicate):
            return DedupResult()

        canon_pred = canon(new_mem.predicate)
        existing = db.query_by_subject(new_mem.subject, limit=200)
        for ex in existing:
            if ex.state != "active":
                continue
            if canon(ex.predicate) == canon_pred and ex.object_value != new_mem.object_value:
                if not new_mem.is_negation:
                    return DedupResult(
                        is_duplicate=True,
                        duplicate_type="same_sp",
                        existing_id=ex.id,
                        confidence=0.9,
                        action="supersede",
                    )
        return DedupResult()

    def _check_cross_tool(self, new_mem: Memory, db: MemoryDB) -> DedupResult:
        """Type 4: Same fact from different tools with different predicates.

        Uses predicate clustering + object canonicalization to detect
        that (user, works_at, Google) and (user, employer, Google Inc.)
        are the same fact.
        """
        # Get all predicates in the same cluster as the new predicate
        pred_cluster = self._normalizer.predicate_normalizer.get_cluster_members(
            new_mem.predicate)

        # Canonicalize the new object
        canon_obj = self._normalizer.object_canonicalizer.canonicalize(
            new_mem.object_value)

        existing = db.query_by_subject(new_mem.subject, limit=200)
        for ex in existing:
            if ex.state != "active" or ex.predicate == "stated":
                continue

            # Never dedup across negation boundaries
            if ex.is_negation != new_mem.is_negation:
                continue

            # Same subject+predicate but different objects on multi-valued predicates → not duplicate
            if (ex.subject == new_mem.subject
                    and ex.predicate == new_mem.predicate
                    and ex.object_value.lower() != new_mem.object_value.lower()
                    and not is_single_valued(new_mem.predicate)):
                continue

            # Check if existing predicate is in the same cluster
            if ex.predicate not in pred_cluster:
                ex_canon = self._normalizer.predicate_normalizer.get_canonical(
                    ex.predicate)
                new_canon = self._normalizer.predicate_normalizer.get_canonical(
                    new_mem.predicate)
                if ex_canon != new_canon:
                    continue

            # Check if objects refer to the same entity
            ex_canon_obj = self._normalizer.object_canonicalizer.canonicalize(
                ex.object_value)
            if ex_canon_obj == canon_obj:
                return DedupResult(
                    is_duplicate=True,
                    duplicate_type="cross_tool",
                    existing_id=ex.id,
                    confidence=0.85,
                    action="merge",
                )

        return DedupResult()

    def _check_temporal(self, new_mem: Memory, db: MemoryDB) -> DedupResult:
        """Type 5: Fact was true but is now stale.

        Detects when an existing fact has expired or been superseded
        by a newer version of the same fact.
        """
        now = time.time()
        existing = db.query_by_subject(new_mem.subject, limit=200)
        for ex in existing:
            if ex.state != "active":
                continue
            if ex.predicate != new_mem.predicate:
                continue
            if ex.object_value != new_mem.object_value:
                continue

            # Same fact but the existing one has expired
            if ex.valid_until is not None and ex.valid_until < now:
                return DedupResult(
                    is_duplicate=True,
                    duplicate_type="temporal",
                    existing_id=ex.id,
                    confidence=0.95,
                    action="supersede",
                )

        return DedupResult()


# ---------------------------------------------------------------------------
#  Provenance tracking
# ---------------------------------------------------------------------------

@dataclass
class ProvenanceRecord:
    """Tracks the origin and transformation of a fact."""
    fact_id: str  # ID of the canonical fact
    source_tool: str  # Which tool wrote this
    original_subject: str  # Pre-normalization subject
    original_predicate: str  # Pre-normalization predicate
    original_object: str  # Pre-normalization object
    original_text: str  # Original source text
    timestamp: float = field(default_factory=time.time)
    superseded_by: str = ""  # ID of fact that replaced this one


class ProvenanceTracker:
    """Maintains provenance records for all facts.

    For any canonical fact, you can answer:
    - Which tool wrote the original?
    - What was the original surface form?
    - When was it written?
    - Has it been superseded?
    """

    def __init__(self) -> None:
        self._records: dict[str, list[ProvenanceRecord]] = {}
        self._lock = threading.Lock()

    def record(self, fact_id: str, source_tool: str,
               original_subject: str, original_predicate: str,
               original_object: str, original_text: str = "") -> None:
        """Record provenance for a fact."""
        rec = ProvenanceRecord(
            fact_id=fact_id,
            source_tool=source_tool,
            original_subject=original_subject,
            original_predicate=original_predicate,
            original_object=original_object,
            original_text=original_text,
        )
        with self._lock:
            if fact_id not in self._records:
                self._records[fact_id] = []
            self._records[fact_id].append(rec)

    def get_provenance(self, fact_id: str) -> list[ProvenanceRecord]:
        """Get all provenance records for a fact."""
        with self._lock:
            return list(self._records.get(fact_id, []))

    def mark_superseded(self, old_id: str, new_id: str) -> None:
        """Mark a fact as superseded by another."""
        with self._lock:
            for rec in self._records.get(old_id, []):
                rec.superseded_by = new_id

    def stats(self) -> dict:
        with self._lock:
            total_facts = len(self._records)
            total_records = sum(len(v) for v in self._records.values())
            multi_source = sum(1 for v in self._records.values() if len(v) > 1)
            return {
                "total_facts_tracked": total_facts,
                "total_provenance_records": total_records,
                "multi_source_facts": multi_source,
            }
