"""
Belief revision — v3 improvements:
- Cross-scope contradiction detection (personal vs company)
- Source-weighted posterior (user_stated > inferred)
- Negation-aware: "likes Java" contradicts "doesn't like Java"
- Consolidation with soft-delete recovery

Schema awareness (Phase 2 Step 6):
The ALIASES / SINGLE_VALUED / DECAY_FACTORS module-level names are kept
as back-compat shims that read from PERSONAL_LIFE_SCHEMA. All new code
should pass a Schema explicitly through check_contradictions /
check_cross_scope_contradictions / consolidate. Defaulting to the personal
schema preserves pre-refactor behavior for every existing caller.
"""

from __future__ import annotations

import math
import time
from lore_memory.store import Memory, MemoryDB
from lore_memory.schema import Schema, PERSONAL_LIFE_SCHEMA

# Back-compat aliases: old code imports these by name from belief.
# They reflect the default personal schema and are safe to read, but
# mutating them does NOT affect runtime — schemas are frozen. Prefer
# passing a Schema explicitly to the functions below.
ALIASES: dict[str, str] = dict(PERSONAL_LIFE_SCHEMA.aliases)
SINGLE_VALUED: frozenset[str] = PERSONAL_LIFE_SCHEMA.single_valued
DECAY_FACTORS: dict[str, float] = dict(PERSONAL_LIFE_SCHEMA.decay_factors)


def canon(pred: str, schema: Schema | None = None) -> str:
    return (schema or PERSONAL_LIFE_SCHEMA).canon(pred)


def is_single_valued(pred: str, schema: Schema | None = None) -> bool:
    return (schema or PERSONAL_LIFE_SCHEMA).is_single_valued(pred)


def check_contradictions(db: MemoryDB, new_mem: Memory,
                          schema: Schema | None = None) -> list[str]:
    """Check new memory against existing for contradictions. Returns contradicted IDs."""
    s = schema or PERSONAL_LIFE_SCHEMA
    canon_new = s.canon(new_mem.predicate)
    # Exclude 'stated' (journal) rows — contradiction checks only compare
    # structured facts, and journal rows swamp the LIMIT window at scale.
    existing = db.query_by_subject(new_mem.subject, exclude_stated=True)
    contradicted = []

    for ex in existing:
        if ex.state != "active":
            continue
        if s.canon(ex.predicate) != canon_new:
            continue

        # Case 1: Same predicate, same value, opposite negation → contradiction
        if (ex.object_value == new_mem.object_value
                and ex.is_negation != new_mem.is_negation):
            db.add_contradiction(ex.id)
            db.update_state(ex.id, "superseded")
            contradicted.append(ex.id)
        # Case 2: Single-valued predicate, different value → supersede old.
        # Two paths into "single-valued":
        #   a) the schema's static set (lives_in, works_at, …)
        #   b) extraction-time inference (e.g. spaCy's "My X is Y" pattern
        #      sets metadata['single_valued']=True for identity-style facts
        #      like favorite_color, hobby, current_project — without us
        #      having to enumerate every possible head noun in the schema)
        # Only when the new memory is a positive assertion (not a negation).
        _new_md = new_mem.metadata or {}
        _is_sv = (s.is_single_valued(new_mem.predicate)
                  or _new_md.get("single_valued") is True)
        if (_is_sv
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
    schema: Schema | None = None,
) -> list[tuple[str, str]]:
    """
    v3: Check for contradictions across multiple scope databases.

    Returns list of (db_path, memory_id) for contradicted memories.
    """
    s = schema or PERSONAL_LIFE_SCHEMA
    if not s.is_single_valued(predicate):
        return []

    canon_pred = s.canon(predicate)
    contradicted = []
    for db in dbs:
        for ex in db.query_by_subject(subject, exclude_stated=True):
            if ex.state != "active":
                continue
            if s.canon(ex.predicate) == canon_pred and ex.object_value != object_value:
                db.add_contradiction(ex.id)
                contradicted.append((db.db_path, ex.id))
    return contradicted


STATED_PROTECTION_DAYS = PERSONAL_LIFE_SCHEMA.stated_protection_days  # back-compat


# Predicates that signal an entity is no longer accessible to the user.
# Triggers supersession of `have`/`own`-family facts whose object names the
# affected entity. Kept tiny — when a phrase isn't on this list, the user
# can supersede explicitly via "I no longer have X".
_LOSS_PREDICATES: frozenset[str] = frozenset({
    "pass_away", "die", "die_away", "die_off",
    "leave",  # "Sarah left the company" — paired with works_at upstream
})

_POSSESSION_PREDICATES: frozenset[str] = frozenset({
    "have", "own", "has", "owns",
})


def check_loss_events(db: MemoryDB, new_mem: Memory) -> list[str]:
    """Supersede possession facts when the named entity is reported as lost.

    Trigger: `new_mem` has a loss-signaling predicate (pass_away, die, …)
    and a non-first-person subject. Action: find existing first-person
    `have`/`own` facts whose object value equals `new_mem.subject`
    (case-insensitive) and mark them superseded.

    Example:
      stored: I → have → Luna   (active)
      stored: Luna → pass_away → ""   (triggers this)
      result: I → have → Luna   becomes superseded.

    Returns the list of superseded memory IDs.
    """
    if new_mem.predicate not in _LOSS_PREDICATES:
        return []
    target = (new_mem.subject or "").strip().lower()
    if not target or target in ("i", "we", "me", "us", "my"):
        return []

    superseded: list[str] = []
    # Scan possession facts and match by exact object_value (case-insensitive).
    # Cheap: each loss event triggers at most one DB scan over active rows.
    rows = db.query_active_lightweight(limit=2000)
    for ex in rows:
        if ex.state != "active":
            continue
        if ex.predicate not in _POSSESSION_PREDICATES:
            continue
        if (ex.object_value or "").strip().lower() == target:
            db.update_state(ex.id, "superseded")
            db.add_contradiction(ex.id)
            superseded.append(ex.id)
    return superseded


def consolidate(db: MemoryDB, batch_size: int = 1000,
                replay_traces: dict | None = None,
                schema: Schema | None = None) -> dict:
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
        s = schema or PERSONAL_LIFE_SCHEMA
        for mem in batch:
            # Protect "stated" (raw text) memories for stated_protection_days
            # These form the foundation of the system and should not be
            # archived prematurely regardless of confidence
            if mem.predicate == "stated":
                age_days = (now - mem.created_at) / 86400
                if age_days < s.stated_protection_days:
                    continue

            decay = s.decay_for(mem.predicate)
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
