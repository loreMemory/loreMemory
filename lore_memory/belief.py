"""
Belief revision — v3 improvements:
- Cross-scope contradiction detection (personal vs company)
- Source-weighted posterior (user_stated > inferred)
- Negation-aware: "likes Java" contradicts "doesn't like Java"
- Consolidation with soft-delete recovery
"""

from __future__ import annotations

import math
import time
from lore_memory.store import Memory, MemoryDB

ALIASES: dict[str, str] = {
    # Grammar parser produces base-form predicates (live_in, work_at)
    # Map them to the canonical forms used in SINGLE_VALUED
    "live_in": "lives_in", "live_at": "lives_in", "reside_in": "lives_in",
    "lives_at": "lives_in", "resides_in": "lives_in", "resides_at": "lives_in",
    "home_city": "lives_in", "located_in": "lives_in",
    "base_in": "lives_in", "move_to": "lives_in", "relocate_to": "lives_in",
    "work_at": "works_at", "work_for": "works_at",
    "employed_at": "works_at", "employed_by": "works_at",
    "works_for": "works_at", "company": "works_at", "employer": "works_at",
    "role": "job_title", "position": "job_title", "occupation": "job_title", "job": "job_title",
    "born_on": "birthday", "date_of_birth": "birthday",
    "email_address": "email", "phone_number": "phone", "mobile": "phone",
    "citizen_of": "nationality", "citizenship": "nationality",
    "mother_tongue": "native_language", "first_language": "native_language",
}

SINGLE_VALUED: set[str] = {
    "lives_in", "works_at", "job_title", "email", "phone", "age",
    "birthday", "nationality", "native_language", "current_project",
    "relationship_status", "timezone", "name",
}

DECAY_FACTORS: dict[str, float] = {
    "birthday": 0.0, "nationality": 0.0, "native_language": 0.0, "name": 0.0,
    "lives_in": 0.3, "works_at": 0.4, "job_title": 0.4,
    "likes": 0.2, "dislikes": 0.2, "prefers": 0.3,
    "current_project": 0.7, "decided": 0.5,
    "committed": 0.1, "changed_by": 0.1,
}


def canon(pred: str) -> str:
    return ALIASES.get(pred, pred)


def is_single_valued(pred: str) -> bool:
    return canon(pred) in SINGLE_VALUED


def check_contradictions(db: MemoryDB, new_mem: Memory) -> list[str]:
    """Check new memory against existing for contradictions. Returns contradicted IDs."""
    canon_new = canon(new_mem.predicate)
    existing = db.query_by_subject(new_mem.subject)
    contradicted = []

    for ex in existing:
        if ex.state != "active":
            continue
        if canon(ex.predicate) != canon_new:
            continue

        # Case 1: Same predicate, same value, opposite negation → contradiction
        if (ex.object_value == new_mem.object_value
                and ex.is_negation != new_mem.is_negation):
            db.add_contradiction(ex.id)
            db.update_state(ex.id, "superseded")
            contradicted.append(ex.id)
        # Case 2: Single-valued predicate, different value → supersede old
        # Only when the new memory is a positive assertion (not a negation).
        # "I used to live in Berlin" should NOT supersede "I live in Amsterdam"
        # because the negation is a retraction, not a new location assertion.
        elif (is_single_valued(new_mem.predicate)
              and ex.object_value != new_mem.object_value
              and not new_mem.is_negation):
            db.add_contradiction(ex.id)
            db.update_state(ex.id, "superseded")
            contradicted.append(ex.id)

    return contradicted


def check_cross_scope_contradictions(
    dbs: list[MemoryDB],
    subject: str,
    predicate: str,
    object_value: str,
) -> list[tuple[str, str]]:
    """
    v3: Check for contradictions across multiple scope databases.

    Returns list of (db_path, memory_id) for contradicted memories.
    """
    if not is_single_valued(predicate):
        return []

    canon_pred = canon(predicate)
    contradicted = []
    for db in dbs:
        for ex in db.query_by_subject(subject):
            if ex.state != "active":
                continue
            if canon(ex.predicate) == canon_pred and ex.object_value != object_value:
                db.add_contradiction(ex.id)
                contradicted.append((db.db_path, ex.id))
    return contradicted


STATED_PROTECTION_DAYS = 7  # Raw text memories ("stated") are protected for this many days


def consolidate(db: MemoryDB, batch_size: int = 1000,
                replay_traces: dict | None = None) -> dict:
    """Run consolidation with memory replay (from NRM).

    Phases:
      1. Replay: boost evidence on memories that appear in activation traces
      2. Decay: apply Ebbinghaus forgetting curve
      3. Archive: remove low-posterior memories

    replay_traces: dict of {memory_id: co-activation_count} from retriever.
    Memories with high trace counts get evidence boosts (they're important).
    """
    now = time.time()
    stats = {"archived": 0, "decayed": 0, "replayed": 0, "duration_ms": 0.0}
    start = time.perf_counter()

    # Phase 0: Memory replay — boost frequently co-activated memories
    if replay_traces:
        for mem_id, trace_count in replay_traces.items():
            if trace_count >= 3:  # appeared in 3+ result sets
                # Boost evidence (makes it harder to decay)
                db.record_feedback(mem_id, helpful=True)
                stats["replayed"] += 1

    offset = 0
    while True:
        batch = db.query_active_lightweight(limit=batch_size, offset=offset)
        if not batch:
            break
        archived_this_batch = 0
        for mem in batch:
            # Protect "stated" (raw text) memories for STATED_PROTECTION_DAYS
            # These form the foundation of the system and should not be
            # archived prematurely regardless of confidence
            if mem.predicate == "stated":
                age_days = (now - mem.created_at) / 86400
                if age_days < STATED_PROTECTION_DAYS:
                    continue

            decay = DECAY_FACTORS.get(canon(mem.predicate), 0.3)
            if decay > 0:
                days = (now - mem.updated_at) / 86400
                strength = max(0.1, mem.access_count * mem.posterior)
                forgetting = math.exp(-days / (strength * (1.0 / decay) * 30))

                if forgetting < 0.1 and mem.posterior < 0.3:
                    db.update_state(mem.id, "archived")
                    stats["archived"] += 1
                    stats["decayed"] += 1
                    archived_this_batch += 1
                    continue

            if mem.posterior < 0.15 or (mem.confidence < 0.1 and mem.evidence_count <= 1):
                db.update_state(mem.id, "archived")
                stats["archived"] += 1
                archived_this_batch += 1

        # Adjust offset: skip forward by batch_size minus archived items
        # (archived items are no longer 'active' so won't appear in next query)
        offset += batch_size - archived_this_batch
        if len(batch) < batch_size:
            break

    stats["duration_ms"] = (time.perf_counter() - start) * 1000
    return stats
