"""
Schema — per-tenant identity vocabulary.

A Schema tells the memory engine three things:

  1. ``aliases`` — map from extracted predicates to canonical ones
     (``move_to`` → ``lives_in``, ``joined`` → ``works_at``).
  2. ``single_valued`` — canonical predicates where a new positive value
     supersedes the previous one (address, current employer, age).
  3. ``decay_factors`` — per-canonical-predicate Ebbinghaus forgetting
     rate. 0.0 means never decay (name, birthday). Higher means fades
     faster without reinforcement.

Why per-tenant. The default ``PERSONAL_LIFE_SCHEMA`` is built for an
English-speaking knowledge worker: ``lives_in``, ``works_at``,
``job_title``, ``email``, ``phone``, etc. A caregiver tracking a parent's
health wants ``on_medication`` and ``dose`` to supersede like the user's
city supersedes. A researcher wants ``argues`` and ``cites`` to accumulate
instead. Same English, different identity model. The schema is the knob.

Schema changes are recorded as a short stable hash in each DB's ``_meta``
table; a mismatch on open logs a warning but does not abort — past data is
still readable under the old rules; new writes follow the new rules.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Schema:
    """Per-tenant identity schema. Frozen so it can safely be shared across threads."""

    # Raw extracted predicate → canonical predicate.
    # The grammar parser produces verb-based preds like ``move_to`` or ``joined``;
    # canonical names are what SINGLE_VALUED keys and DECAY_FACTORS reference.
    aliases: dict[str, str] = field(default_factory=dict)

    # Canonical predicates that may have at most one active value per subject.
    # A new positive fact with the same (subject, canon-predicate) supersedes
    # the old one. Multi-valued predicates (``likes``, ``uses``) accumulate.
    single_valued: frozenset[str] = field(default_factory=frozenset)

    # Per-canonical-predicate Ebbinghaus decay rate. Missing keys fall back
    # to the default (0.3 in consolidate()). 0.0 means the fact never decays.
    decay_factors: dict[str, float] = field(default_factory=dict)

    # How many days "stated" (raw text) memories are protected from decay.
    stated_protection_days: int = 7

    # --- Derived helpers ---

    def canon(self, predicate: str) -> str:
        """Map an extracted predicate to its canonical form."""
        return self.aliases.get(predicate, predicate)

    def is_single_valued(self, predicate: str) -> bool:
        """True if this predicate (by canonical form) supersedes on rewrite."""
        return self.canon(predicate) in self.single_valued

    def decay_for(self, predicate: str) -> float:
        """Ebbinghaus decay rate for a predicate. 0.3 default if unknown."""
        return self.decay_factors.get(self.canon(predicate), 0.3)

    # --- Persistence / identity ---

    def hash_key(self) -> str:
        """Short stable hash of the schema contents.

        Stored in the DB's _meta table; mismatches on open trigger a warning.
        Content is sorted to guarantee reproducibility across runs.
        """
        payload = json.dumps({
            "aliases": dict(sorted(self.aliases.items())),
            "single_valued": sorted(self.single_valued),
            "decay_factors": dict(sorted(self.decay_factors.items())),
            "stated_protection_days": self.stated_protection_days,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
#  Default: personal life (English knowledge-worker)
# ---------------------------------------------------------------------------

# These were previously module-level globals in belief.py. Preserved exactly
# so migrating is a no-op for existing users.

_PERSONAL_ALIASES: dict[str, str] = {
    # Grammar parser produces base-form predicates (live_in, work_at).
    # Map them to the canonical forms used in SINGLE_VALUED.
    "live_in": "lives_in", "live_at": "lives_in", "reside_in": "lives_in",
    "lives_at": "lives_in", "resides_in": "lives_in", "resides_at": "lives_in",
    "home_city": "lives_in", "located_in": "lives_in",
    "base_in": "lives_in", "move_to": "lives_in", "relocate_to": "lives_in",
    "work_at": "works_at", "work_for": "works_at",
    "employed_at": "works_at", "employed_by": "works_at",
    "works_for": "works_at", "company": "works_at", "employer": "works_at",
    "join": "works_at", "joined": "works_at", "join_at": "works_at",
    "start_at": "works_at",
    "role": "job_title", "position": "job_title",
    "occupation": "job_title", "job": "job_title",
    "born_on": "birthday", "date_of_birth": "birthday",
    "email_address": "email", "phone_number": "phone", "mobile": "phone",
    "citizen_of": "nationality", "citizenship": "nationality",
    "mother_tongue": "native_language", "first_language": "native_language",
}

_PERSONAL_SINGLE_VALUED: frozenset[str] = frozenset({
    "lives_in", "works_at", "job_title", "email", "phone", "age",
    "birthday", "nationality", "native_language", "current_project",
    "relationship_status", "timezone", "name",
})

_PERSONAL_DECAY: dict[str, float] = {
    "birthday": 0.0, "nationality": 0.0, "native_language": 0.0, "name": 0.0,
    "lives_in": 0.3, "works_at": 0.4, "job_title": 0.4,
    "likes": 0.2, "dislikes": 0.2, "prefers": 0.3,
    "current_project": 0.7, "decided": 0.5,
    "committed": 0.1, "changed_by": 0.1,
}

PERSONAL_LIFE_SCHEMA: Schema = Schema(
    aliases=_PERSONAL_ALIASES,
    single_valued=_PERSONAL_SINGLE_VALUED,
    decay_factors=_PERSONAL_DECAY,
)


# ---------------------------------------------------------------------------
#  Preset: caregiver / medical tracking
# ---------------------------------------------------------------------------

_CARE_ALIASES: dict[str, str] = {
    **_PERSONAL_ALIASES,
    # Medication
    "prescribed":     "on_medication",
    "taking":         "on_medication",
    "takes":          "on_medication",
    "started_on":     "on_medication",
    # Dose
    "dosage":         "dose",
    "dosed_at":       "dose",
    # Symptoms (multi-valued; not in single_valued)
    "experiencing":   "has_symptom",
    "suffering_from": "has_symptom",
    # Diagnosis (single-valued per condition)
    "diagnosed_with": "has_diagnosis",
    "dx":             "has_diagnosis",
    # Appointments
    "appointment_on": "appointment_on",
    "scheduled_for":  "appointment_on",
}

CARE_TRACKING_SCHEMA: Schema = Schema(
    aliases=_CARE_ALIASES,
    single_valued=_PERSONAL_SINGLE_VALUED | frozenset({
        "on_medication", "dose", "has_diagnosis", "appointment_on",
        "primary_caregiver", "insurance_plan",
    }),
    decay_factors={
        **_PERSONAL_DECAY,
        "on_medication": 0.5,
        "dose":          0.6,
        "has_symptom":   0.4,
        "appointment_on": 0.8,
        # Diagnoses usually persist — low decay.
        "has_diagnosis": 0.1,
    },
)


# ---------------------------------------------------------------------------
#  Preset: research notes (multi-valued claims about sources)
# ---------------------------------------------------------------------------

_RESEARCH_ALIASES: dict[str, str] = {
    **_PERSONAL_ALIASES,
    "authored_by":    "written_by",
    "published_in":   "published_in",
    "cited_by":       "cited_by",
    "refers_to":      "cites",
    "references":     "cites",
}

RESEARCH_NOTES_SCHEMA: Schema = Schema(
    aliases=_RESEARCH_ALIASES,
    # Deliberately multi-valued: a note can argue many things, cite many
    # sources, contradict several claims. Only the personal-identity
    # predicates stay single-valued.
    single_valued=_PERSONAL_SINGLE_VALUED,
    decay_factors={
        **_PERSONAL_DECAY,
        # Academic claims don't decay — they're the primary content.
        "argues":       0.0,
        "cites":        0.0,
        "defines":      0.0,
        "contradicts":  0.0,
        "extends":      0.0,
        "written_by":   0.0,
        "published_in": 0.0,
    },
)


# Registry for debug/exploration. Not load-bearing.
PRESETS: dict[str, Schema] = {
    "personal_life":  PERSONAL_LIFE_SCHEMA,
    "care_tracking":  CARE_TRACKING_SCHEMA,
    "research_notes": RESEARCH_NOTES_SCHEMA,
}
