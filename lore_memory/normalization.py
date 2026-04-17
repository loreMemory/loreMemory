"""
Normalization engine — subject, predicate, and object canonicalization.

No hardcoded synonym lists. Uses embedding similarity and string distance
to cluster entities without manual maintenance.

Architecture:
  Subject normalization:  First-person pronouns + learned name aliases -> canonical user_id
  Predicate normalization: Embedding-based clustering of semantically similar predicates
  Object canonicalization: Fuzzy string matching + case normalization
"""

from __future__ import annotations

import math
import re
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

from lore_memory.store import cosine_sim


# ---------------------------------------------------------------------------
#  Subject Normalization
# ---------------------------------------------------------------------------

# First-person references that always resolve to the session user
from lore_memory.lexicons import (
    FIRST_PERSON_SUBJECTS,
    FIRST_PERSON_ALIASES,
    SUBJECT_RESOLVER_RELATIONSHIP_NOUNS as _RELATIONSHIP_NOUNS,
)
_FIRST_PERSON = FIRST_PERSON_SUBJECTS | FIRST_PERSON_ALIASES


@dataclass
class SubjectResolver:
    """Resolves subject variants to a canonical user identifier.

    In single-user context, first-person pronouns and known names
    all resolve to the canonical user_id. Names are learned from
    context, not hardcoded.
    """
    canonical_id: str = ""
    _aliases: set[str] = field(default_factory=set)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def register_alias(self, alias: str) -> None:
        """Register a new alias for the canonical user."""
        normalized = alias.strip().lower()
        if normalized and len(normalized) > 1:
            with self._lock:
                self._aliases.add(normalized)

    def resolve(self, subject: str) -> str:
        """Resolve a subject string to its canonical form.

        Returns canonical_id for first-person refs and known aliases.
        Returns the original subject (lowercased, stripped) otherwise.
        """
        if not subject:
            return self.canonical_id

        s = subject.strip().lower()

        # Possessive + relationship noun → third-party entity, not the user
        if s.startswith("my "):
            remainder = s[3:].strip()
            if any(word in _RELATIONSHIP_NOUNS for word in remainder.split()):
                return subject  # preserve original case

        # First-person pronouns always resolve to canonical user
        if s in _FIRST_PERSON:
            return self.canonical_id

        # Check learned aliases
        with self._lock:
            if s in self._aliases:
                return self.canonical_id

        # Check if subject is a substring match of canonical_id or vice versa
        canon_lower = self.canonical_id.lower()
        if canon_lower and (s in canon_lower or canon_lower in s):
            return self.canonical_id

        return subject  # preserve original case for third-party subjects

    def learn_from_text(self, text: str, user_id: str) -> None:
        """Learn user name aliases from statements like 'My name is Alice'."""
        lower = text.lower()
        patterns = [
            r"my name is\s+(.+?)(?:\.|,|$)",
            r"i(?:'m| am)\s+(.+?)(?:\.|,|$)",
            r"call me\s+(.+?)(?:\.|,|$)",
        ]
        for pat in patterns:
            m = re.search(pat, lower)
            if m:
                name = m.group(1).strip()
                # Filter out verbs/descriptions (keep short names)
                words = name.split()
                if 1 <= len(words) <= 4:
                    # Don't register common descriptions as names
                    skip = {"a ", "an ", "the ", "working", "living", "based"}
                    if not any(name.startswith(s) for s in skip):
                        self.register_alias(name)
                        # Also register individual name parts
                        for w in words:
                            if len(w) > 2:
                                self.register_alias(w)

    def is_self_reference(self, subject: str) -> bool:
        """Check if a subject refers to the canonical user."""
        return self.resolve(subject) == self.canonical_id


# ---------------------------------------------------------------------------
#  Predicate Normalization
# ---------------------------------------------------------------------------

@dataclass
class PredicateCluster:
    canonical: str  # The canonical form for this cluster
    members: set[str] = field(default_factory=set)  # All predicates in cluster
    embedding: list[float] = field(default_factory=list)  # Centroid embedding


class PredicateNormalizer:
    """Clusters semantically similar predicates without a hardcoded synonym list.

    Uses embedding similarity to detect that 'works_at', 'employed_by',
    'employer' all express the same relationship. Builds clusters
    incrementally as new predicates are encountered.
    """

    def __init__(self, embed_fn: Callable[[str], list[float]],
                 similarity_threshold: float = 0.82) -> None:
        self._embed = embed_fn
        self._threshold = similarity_threshold
        self._clusters: list[PredicateCluster] = []
        self._pred_to_cluster: dict[str, int] = {}  # predicate -> cluster index
        self._pred_embeddings: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def _embed_predicate(self, predicate: str) -> list[float]:
        """Embed a predicate by converting underscores to spaces for better semantics."""
        readable = predicate.replace("_", " ")
        return self._embed(readable)

    def normalize(self, predicate: str) -> str:
        """Return the canonical form for a predicate.

        If the predicate matches an existing cluster, return that cluster's
        canonical form. Otherwise, create a new cluster or return as-is.
        """
        pred = predicate.lower().strip()
        if not pred or pred == "stated":
            return pred

        with self._lock:
            # Fast path: already in a cluster
            if pred in self._pred_to_cluster:
                idx = self._pred_to_cluster[pred]
                return self._clusters[idx].canonical

            # Compute embedding
            emb = self._embed_predicate(pred)
            self._pred_embeddings[pred] = emb

            # Find best matching cluster
            best_sim = 0.0
            best_idx = -1
            for i, cluster in enumerate(self._clusters):
                if cluster.embedding:
                    sim = cosine_sim(emb, cluster.embedding)
                    if sim > best_sim:
                        best_sim = sim
                        best_idx = i

            if best_sim >= self._threshold and best_idx >= 0:
                # Join existing cluster
                cluster = self._clusters[best_idx]
                cluster.members.add(pred)
                self._pred_to_cluster[pred] = best_idx
                return cluster.canonical
            else:
                # Create new cluster with this predicate as canonical
                new_cluster = PredicateCluster(
                    canonical=pred,
                    members={pred},
                    embedding=emb,
                )
                idx = len(self._clusters)
                self._clusters.append(new_cluster)
                self._pred_to_cluster[pred] = idx
                return pred

    def get_canonical(self, predicate: str) -> str:
        """Get canonical form without creating new clusters."""
        pred = predicate.lower().strip()
        with self._lock:
            if pred in self._pred_to_cluster:
                idx = self._pred_to_cluster[pred]
                return self._clusters[idx].canonical
        return pred

    def get_cluster_members(self, predicate: str) -> set[str]:
        """Get all predicates in the same cluster as the given predicate."""
        pred = predicate.lower().strip()
        with self._lock:
            if pred in self._pred_to_cluster:
                idx = self._pred_to_cluster[pred]
                return set(self._clusters[idx].members)
        return {pred}

    def seed_cluster(self, canonical: str, members: list[str]) -> None:
        """Seed a predicate cluster with known synonyms.
        Used to bootstrap from the existing ALIASES map in belief.py."""
        with self._lock:
            emb = self._embed_predicate(canonical)
            cluster = PredicateCluster(
                canonical=canonical,
                members=set(members) | {canonical},
                embedding=emb,
            )
            idx = len(self._clusters)
            self._clusters.append(cluster)
            for m in cluster.members:
                self._pred_to_cluster[m] = idx

    @property
    def cluster_count(self) -> int:
        return len(self._clusters)

    @property
    def unique_predicates(self) -> int:
        return len(self._pred_to_cluster)

    def stats(self) -> dict:
        """Return normalization statistics."""
        with self._lock:
            total_preds = len(self._pred_to_cluster)
            total_clusters = len(self._clusters)
            multi_member = sum(1 for c in self._clusters if len(c.members) > 1)
            return {
                "total_predicates": total_preds,
                "total_clusters": total_clusters,
                "multi_member_clusters": multi_member,
                "compression_ratio": 1.0 - (total_clusters / max(total_preds, 1)),
            }


# ---------------------------------------------------------------------------
#  Object Canonicalization
# ---------------------------------------------------------------------------

def _levenshtein(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


def _normalize_object_string(obj: str) -> str:
    """Normalize an object string for comparison."""
    s = obj.strip()
    # Remove common suffixes that don't change identity
    for suffix in (" Inc.", " Inc", " LLC", " Ltd.", " Ltd",
                   " Corp.", " Corp", " Co.", " Co",
                   " (company)", " (the company)"):
        if s.endswith(suffix):
            s = s[:-len(suffix)].strip()
    return s


class ObjectCanonicalizer:
    """Canonicalizes object values to prevent namespace explosion.

    Uses fuzzy string matching to detect that 'Google', 'Google Inc.',
    'Google LLC', 'google' all refer to the same entity.

    Conservative: only merges when evidence is strong. Never silently
    merges ambiguous cases.
    """

    def __init__(self, similarity_threshold: float = 0.85) -> None:
        self._threshold = similarity_threshold
        self._canonicals: dict[str, str] = {}  # normalized_lower -> canonical form
        self._lock = threading.Lock()

    def canonicalize(self, obj: str) -> str:
        """Return the canonical form for an object value.

        Checks for:
        1. Case-insensitive exact match
        2. Suffix-stripped match (Google Inc. -> Google)
        3. Levenshtein distance for near-identical strings
        4. Substring containment
        """
        if not obj or len(obj.strip()) == 0:
            return obj

        obj_stripped = obj.strip()
        normalized = _normalize_object_string(obj_stripped)
        norm_lower = normalized.lower()

        with self._lock:
            # Case-insensitive exact match
            if norm_lower in self._canonicals:
                return self._canonicals[norm_lower]

            # Check for fuzzy matches against existing canonicals
            best_match = None
            best_score = 0.0

            for existing_lower, existing_canonical in self._canonicals.items():
                # Substring containment (bidirectional)
                if norm_lower in existing_lower or existing_lower in norm_lower:
                    # Use the shorter form as canonical (more general)
                    shorter = existing_canonical if len(existing_lower) <= len(norm_lower) else obj_stripped
                    best_match = shorter if len(shorter) <= len(existing_canonical) else existing_canonical
                    best_score = 1.0
                    break

                # Levenshtein distance relative to string length
                dist = _levenshtein(norm_lower, existing_lower)
                max_len = max(len(norm_lower), len(existing_lower))
                if max_len > 0:
                    similarity = 1.0 - (dist / max_len)
                    if similarity > self._threshold and similarity > best_score:
                        best_score = similarity
                        best_match = existing_canonical

            if best_match:
                self._canonicals[norm_lower] = best_match
                return best_match
            else:
                # New canonical entry
                self._canonicals[norm_lower] = obj_stripped
                return obj_stripped

    def register_canonical(self, obj: str) -> None:
        """Explicitly register an object as a canonical form."""
        normalized = _normalize_object_string(obj.strip())
        with self._lock:
            self._canonicals[normalized.lower()] = obj.strip()

    @property
    def canonical_count(self) -> int:
        with self._lock:
            return len(set(self._canonicals.values()))

    @property
    def total_entries(self) -> int:
        with self._lock:
            return len(self._canonicals)

    def stats(self) -> dict:
        with self._lock:
            unique_canonicals = len(set(self._canonicals.values()))
            total_entries = len(self._canonicals)
            return {
                "total_entries": total_entries,
                "unique_canonicals": unique_canonicals,
                "compression_ratio": 1.0 - (unique_canonicals / max(total_entries, 1)),
            }


# ---------------------------------------------------------------------------
#  Unified Normalization Pipeline
# ---------------------------------------------------------------------------

class NormalizationPipeline:
    """Unified normalization for subject, predicate, and object.

    Call normalize_triple() before storing any fact. This ensures
    all variants are resolved to canonical forms at write time.
    """

    def __init__(self, embed_fn: Callable[[str], list[float]],
                 user_id: str = "",
                 pred_threshold: float = 0.82,
                 obj_threshold: float = 0.85) -> None:
        self.subject_resolver = SubjectResolver(canonical_id=user_id)
        self.predicate_normalizer = PredicateNormalizer(
            embed_fn, similarity_threshold=pred_threshold)
        self.object_canonicalizer = ObjectCanonicalizer(
            similarity_threshold=obj_threshold)
        self._embed_fn = embed_fn

    def normalize_triple(self, subject: str, predicate: str,
                         object_value: str) -> tuple[str, str, str]:
        """Normalize all three components of an SPO triple.

        Returns (canonical_subject, canonical_predicate, canonical_object).
        """
        canon_subject = self.subject_resolver.resolve(subject)
        canon_predicate = self.predicate_normalizer.normalize(predicate)
        canon_object = self.object_canonicalizer.canonicalize(object_value)
        return canon_subject, canon_predicate, canon_object

    def learn_from_text(self, text: str) -> None:
        """Learn aliases from text (e.g., 'My name is Alice')."""
        self.subject_resolver.learn_from_text(text, self.subject_resolver.canonical_id)

    def seed_predicate_aliases(self, aliases: dict[str, str]) -> None:
        """Seed predicate clusters from existing alias map.

        Input: {variant: canonical} like belief.py's ALIASES.
        Groups by canonical form and seeds clusters.
        """
        from collections import defaultdict
        groups: dict[str, list[str]] = defaultdict(list)
        for variant, canonical in aliases.items():
            groups[canonical].append(variant)
        for canonical, members in groups.items():
            self.predicate_normalizer.seed_cluster(canonical, members)

    def stats(self) -> dict:
        return {
            "subject": {
                "canonical_id": self.subject_resolver.canonical_id,
                "alias_count": len(self.subject_resolver._aliases),
            },
            "predicate": self.predicate_normalizer.stats(),
            "object": self.object_canonicalizer.stats(),
        }
