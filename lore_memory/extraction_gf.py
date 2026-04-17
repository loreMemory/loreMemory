"""
Grammar-free extraction — extract SPO triples without hardcoded rules.

Five approaches, used in combination:
  A. Entity-relationship co-occurrence patterns
  B. Contextual predicate inference from entity type pairs
  C. Correction-driven learning
  D. Source-type-aware extraction (chat, email signature, commit, etc.)
  E. Confidence-weighted multi-candidate extraction

Architecture:
  Input text → source type detection → entity extraction →
  relationship inference → candidate ranking → normalized triple output

No verb dictionary. No grammar ruleset. No fixed predicate list.
No hardcoded sentence parser. No vocabulary that needs maintenance.
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

from lore_memory.store import Memory


# ---------------------------------------------------------------------------
#  Entity detection (lightweight, no grammar)
# ---------------------------------------------------------------------------

# These are entity boundary markers, not grammar rules
_BOUNDARY_CHARS = frozenset(".,;:!?()[]{}\"'")

# Common first-person markers for subject detection (with contractions)
from lore_memory.lexicons import FIRST_PERSON_WITH_CONTRACTIONS as _SELF_MARKERS


def _extract_entities(text: str) -> list[str]:
    """Extract potential entities from text.

    Uses capitalization, quoted strings, and known patterns
    rather than grammar rules.
    """
    entities = []

    # Capitalized sequences (proper nouns)
    for m in re.finditer(r'\b([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)\b', text):
        ent = m.group(1).strip()
        # Filter out sentence-start capitalization for common words
        if len(ent) > 1 and ent.lower() not in _SELF_MARKERS:
            entities.append(ent)

    # Quoted strings
    for m in re.finditer(r'"([^"]+)"', text):
        entities.append(m.group(1).strip())
    for m in re.finditer(r"'([^']+)'", text):
        ent = m.group(1).strip()
        if len(ent) > 1:
            entities.append(ent)

    # Technical terms (camelCase, snake_case, kebab-case)
    for m in re.finditer(r'\b([a-z]+(?:[A-Z][a-z]+)+)\b', text):
        entities.append(m.group(1))
    for m in re.finditer(r'\b([a-z]+(?:_[a-z]+)+)\b', text):
        entities.append(m.group(1))

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for e in entities:
        el = e.lower()
        if el not in seen:
            seen.add(el)
            unique.append(e)
    return unique


def _extract_between_text(text: str, ent1: str, ent2: str) -> str:
    """Extract the text between two entities in a string."""
    idx1 = text.lower().find(ent1.lower())
    idx2 = text.lower().find(ent2.lower())
    if idx1 < 0 or idx2 < 0:
        return ""
    if idx1 < idx2:
        between = text[idx1 + len(ent1):idx2].strip()
    else:
        between = text[idx2 + len(ent2):idx1].strip()
    # Clean boundary chars
    between = between.strip(".,;:!?-–—")
    return between.strip()


# ---------------------------------------------------------------------------
#  Source type detection
# ---------------------------------------------------------------------------

@dataclass
class SourceType:
    type: str  # "chat", "email_signature", "commit", "kv_pairs", "social", "fragment", "sentence"
    confidence: float = 0.5


def detect_source_type(text: str) -> SourceType:
    """Detect the source type of input text without hardcoded grammar."""
    text_stripped = text.strip()

    # Key-value format: "key: value" or "key = value"
    kv_count = len(re.findall(r'[a-zA-Z_]+\s*[:=]\s*\S+', text_stripped))
    if kv_count >= 2:
        return SourceType("kv_pairs", 0.9)

    # Email signature: name | title | company pattern — but NOT tables
    if "|" in text_stripped:
        lines = text_stripped.split("\n")
        parts = [p.strip() for p in lines[0].split("|")]
        if len(parts) >= 3 and len(lines) <= 2:
            return SourceType("email_signature", 0.85)

    # Git commit: starts with type(scope): or type:
    from lore_memory.lexicons import COMMIT_MSG_RE as _COMMIT_RE
    if _COMMIT_RE.match(text_stripped):
        return SourceType("commit", 0.9)

    # Social media: has @ or # markers
    if re.search(r'[@#]\w+', text_stripped):
        return SourceType("social", 0.7)

    # Arrow/structured: contains -> or => or ->
    if re.search(r'[→\->=>]', text_stripped):
        return SourceType("fragment", 0.7)

    # Resume format: section headers + dash-separated entries
    if re.search(r'^[A-Z]{3,}', text_stripped) and ('—' in text_stripped or '–' in text_stripped or ' - ' in text_stripped):
        return SourceType("resume", 0.8)

    # Fragment: very short, no verb-like structure
    words = text_stripped.split()

    # Check for subject+verb pattern before declaring fragment (F1 fix)
    _COMMON_VERBS = {"like", "love", "hate", "use", "want", "need", "know", "think",
                     "work", "live", "speak", "read", "write", "build", "run", "play",
                     "prefer", "enjoy", "study", "teach", "learn", "make", "go", "come",
                     "eat", "drink", "sleep", "drive", "fly", "swim", "cook"}
    _SUBJECT_WORDS = {"i", "you", "he", "she", "we", "they", "it"}
    words_lower = [w.lower() for w in words]
    if len(words) >= 2 and len(words) <= 4:
        if words_lower[0] in _SUBJECT_WORDS and any(w in _COMMON_VERBS for w in words_lower[1:]):
            return SourceType("sentence", 0.5)

    if len(words) <= 4 and not any(w.lower() in ("is", "am", "are", "was", "were",
                                                    "have", "has", "had", "do", "does")
                                   for w in words):
        return SourceType("fragment", 0.6)

    # Chat/casual: starts with lowercase, has informal markers
    if text_stripped and text_stripped[0].islower():
        informal = {"lol", "haha", "yeah", "yep", "nope", "btw", "fyi",
                    "tbh", "imo", "ikr", "omg", "brb"}
        if any(w in text_stripped.lower().split() for w in informal):
            return SourceType("chat", 0.8)

    return SourceType("sentence", 0.5)


# ---------------------------------------------------------------------------
#  Source-type-specific extractors (Approach D)
# ---------------------------------------------------------------------------

def _extract_kv_pairs(text: str, user_id: str) -> list[dict]:
    """Extract from key-value format: 'employer: Google | role: SWE'."""
    results = []
    # Split on | or newlines
    parts = re.split(r'[|\n]+', text)
    for part in parts:
        # Try colon separator
        m = re.match(r'\s*([a-zA-Z_\s]+?)\s*[:=]\s*(.+?)\s*$', part.strip())
        if m:
            key = m.group(1).strip().lower().replace(" ", "_")
            value = m.group(2).strip()
            results.append({
                "subject": user_id,
                "predicate": key,
                "object": value,
                "confidence": 0.8,
            })
    return results


def _extract_email_signature(text: str, user_id: str) -> list[dict]:
    """Extract from email signature: 'Name | Title | Company'."""
    parts = [p.strip() for p in text.split("|")]
    results = []
    if len(parts) >= 3:
        # First part is usually name
        results.append({
            "subject": user_id,
            "predicate": "name",
            "object": parts[0],
            "confidence": 0.75,
        })
        # Second part is usually title/role
        results.append({
            "subject": user_id,
            "predicate": "job_title",
            "object": parts[1],
            "confidence": 0.75,
        })
        # Third part is usually company
        results.append({
            "subject": user_id,
            "predicate": "works_at",
            "object": parts[2],
            "confidence": 0.75,
        })
        # Additional parts are bonus info
        for part in parts[3:]:
            results.append({
                "subject": user_id,
                "predicate": "affiliated_with",
                "object": part,
                "confidence": 0.6,
            })
    return results


def _extract_commit(text: str, user_id: str) -> list[dict]:
    """Extract from git commit message: 'fix(auth): resolve login bug'."""
    results = []
    m = re.match(r'^(\w+)\s*(?:\(([^)]+)\))?\s*:\s*(.+)', text.strip())
    if m:
        action = m.group(1)  # feat, fix, etc.
        scope = m.group(2)   # module/area
        description = m.group(3).strip()

        results.append({
            "subject": "commit",
            "predicate": action,
            "object": description,
            "confidence": 0.8,
        })
        if scope:
            results.append({
                "subject": "commit",
                "predicate": "affects",
                "object": scope,
                "confidence": 0.8,
            })
    return results


def _extract_social(text: str, user_id: str) -> list[dict]:
    """Extract from social media format: '@user works_at Google #tech'."""
    results = []
    # Extract @mentions
    for m in re.finditer(r'@(\w+)', text):
        results.append({
            "subject": m.group(1),
            "predicate": "mentioned",
            "object": text[:100],
            "confidence": 0.5,
        })
    # Extract ticket references (e.g. PAYMENTS-1234)
    for m in re.finditer(r'([A-Z]+-\d+)', text):
        results.append({
            "subject": user_id,
            "predicate": "works_on",
            "object": m.group(1),
            "confidence": 0.6,
        })
    # Extract #hashtags
    for m in re.finditer(r'#(\w+)', text):
        results.append({
            "subject": user_id,
            "predicate": "tagged",
            "object": m.group(1),
            "confidence": 0.5,
        })
    return results


def _extract_resume(text: str, user_id: str) -> list[dict]:
    """Extract from resume/CV format: 'EXPERIENCE\\nStartupCo — Tech Lead (2022–present)'."""
    results = []
    for line in text.strip().split("\n"):
        # Match: Company — Role (dates)
        m = re.match(r'(.+?)\s*[—–\-]\s*(.+?)(?:\s*\((.+?)\))?\s*$', line.strip())
        if m and not line.strip().isupper():  # Skip section headers
            company = m.group(1).strip()
            role = m.group(2).strip()
            dates = m.group(3).strip() if m.group(3) else ""
            results.append({"subject": user_id, "predicate": "works_at", "object": company, "confidence": 0.75})
            results.append({"subject": user_id, "predicate": "job_title", "object": role, "confidence": 0.75})
            if dates:
                results.append({"subject": user_id, "predicate": "period", "object": f"{company}: {dates}", "confidence": 0.6})
    return results


def _extract_fragment(text: str, user_id: str) -> list[dict]:
    """Extract from fragments: 'Python. 8 years.' or 'Google -> SWE -> 2022'."""
    results = []
    text_stripped = text.strip()

    # Arrow format: "A -> B -> C"
    if re.search(r'[→\->=>]+', text_stripped):
        # Normalize arrow types
        normalized = re.sub(r'[→\->=]+>', '->', text_stripped)
        normalized = re.sub(r'→', '->', normalized)
        parts = [p.strip() for p in re.split(r'\s*->\s*', normalized) if p.strip()]
        if len(parts) >= 2:
            results.append({
                "subject": user_id,
                "predicate": "affiliated_with",
                "object": parts[0],
                "confidence": 0.6,
            })
            for i in range(1, len(parts)):
                results.append({
                    "subject": parts[0] if i == 1 else user_id,
                    "predicate": "has",
                    "object": parts[i],
                    "confidence": 0.55,
                })
        return results

    # Period-separated fragments: "Python. 8 years."
    fragments = [f.strip().rstrip(".") for f in text_stripped.split(".") if f.strip()]
    if len(fragments) >= 2:
        # First fragment is likely the topic
        topic = fragments[0]
        for frag in fragments[1:]:
            # Check for duration pattern
            dur_m = re.match(r'(\d+)\s*(years?|months?|weeks?|days?)', frag, re.IGNORECASE)
            if dur_m:
                results.append({
                    "subject": user_id,
                    "predicate": "experience_with",
                    "object": f"{topic} ({frag})",
                    "confidence": 0.7,
                })
            else:
                results.append({
                    "subject": user_id,
                    "predicate": "related_to",
                    "object": f"{topic}: {frag}",
                    "confidence": 0.5,
                })
    elif len(fragments) == 1 and len(fragments[0].split()) <= 3:
        # Single short fragment — store as stated, can't extract reliable SPO
        pass

    # Single word/very short — store as a skill/topic mention
    if not results:
        words = text_stripped.split()
        if len(words) == 1:
            word = words[0].strip(".,;:!?")
            if word and len(word) > 1:
                results.append({
                    "subject": user_id,
                    "predicate": "mentions",
                    "object": word,
                    "confidence": 0.4,
                })

    return results


# ---------------------------------------------------------------------------
#  Approach A: Entity-relationship co-occurrence
# ---------------------------------------------------------------------------

class CooccurrenceTracker:
    """Track what text appears between entity pairs across many inputs.

    Over time, builds clusters of relationship words that mean the same thing
    without a verb dictionary.
    """

    def __init__(self) -> None:
        # (entity_type_pair) -> {between_text: count}
        self._patterns: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._lock = __import__("threading").Lock()

    def record(self, ent1: str, ent2: str, between: str) -> None:
        """Record text found between two entities."""
        if not between or len(between) < 2:
            return
        between_clean = between.lower().strip()
        with self._lock:
            self._patterns[f"{ent1.lower()}:{ent2.lower()}"][between_clean] += 1

    def get_common_relations(self, min_count: int = 2) -> dict[str, list[str]]:
        """Get commonly occurring relationship patterns."""
        with self._lock:
            result = {}
            for pair, relations in self._patterns.items():
                common = [r for r, c in relations.items() if c >= min_count]
                if common:
                    result[pair] = sorted(common, key=lambda r: relations[r], reverse=True)
            return result


# ---------------------------------------------------------------------------
#  Approach C: Correction-driven learning
# ---------------------------------------------------------------------------

class CorrectionLearner:
    """Learn from user corrections to improve future extractions.

    When a user says 'that should be prefers, not into', the system
    learns that 'into' in the context of technology objects maps to 'prefers'.
    """

    def __init__(self) -> None:
        # (wrong_predicate, object_context) -> correct_predicate
        self._corrections: dict[tuple[str, str], str] = {}
        # predicate -> predicate mapping (context-free)
        self._pred_corrections: dict[str, str] = {}
        self._lock = __import__("threading").Lock()

    def learn(self, wrong_pred: str, correct_pred: str,
              object_context: str = "") -> None:
        """Learn a correction."""
        with self._lock:
            if object_context:
                self._corrections[(wrong_pred.lower(), object_context.lower())] = correct_pred.lower()
            self._pred_corrections[wrong_pred.lower()] = correct_pred.lower()

    def apply(self, predicate: str, object_value: str = "") -> str:
        """Apply learned corrections to a predicate."""
        pred_lower = predicate.lower()
        with self._lock:
            # Context-specific correction first
            if object_value:
                key = (pred_lower, object_value.lower())
                if key in self._corrections:
                    return self._corrections[key]
            # Context-free correction
            if pred_lower in self._pred_corrections:
                return self._pred_corrections[pred_lower]
        return predicate

    def has_correction(self, predicate: str) -> bool:
        with self._lock:
            return predicate.lower() in self._pred_corrections


# ---------------------------------------------------------------------------
#  Approach E: Confidence-weighted multi-candidate extraction
# ---------------------------------------------------------------------------

@dataclass
class ExtractionCandidate:
    subject: str
    predicate: str
    object_value: str
    confidence: float
    source: str = ""  # Which approach produced this candidate
    is_negation: bool = False


# ---------------------------------------------------------------------------
#  Grammar-Free Extractor (combines all approaches)
# ---------------------------------------------------------------------------

class GrammarFreeExtractor:
    """Extract SPO triples from arbitrary text without grammar rules.

    Combines five approaches:
    A. Entity co-occurrence patterns
    B. Contextual predicate inference
    C. Correction-driven learning
    D. Source-type-aware extraction
    E. Confidence-weighted candidates
    """

    def __init__(self, embed_fn: Callable[[str], list[float]] | None = None) -> None:
        self._embed_fn = embed_fn
        self._cooccurrence = CooccurrenceTracker()
        self._corrections = CorrectionLearner()
        self._candidate_threshold = 0.4  # Minimum confidence to keep a candidate

    def extract(self, text: str, user_id: str,
                source_tool: str = "unknown",
                **kwargs) -> list[Memory]:
        """Extract memories from arbitrary text.

        This is the main entry point. Detects source type and applies
        the appropriate extraction approach(es).
        """
        text = text.strip()
        if not text:
            return []

        now = time.time()
        scope = kwargs.get("scope", "private")
        context = kwargs.get("context", "personal")
        subject = kwargs.get("subject", user_id)

        mems: list[Memory] = []
        candidates: list[ExtractionCandidate] = []

        # Detect source type (Approach D)
        source_type = detect_source_type(text)

        # Apply source-specific extractor
        raw_extractions = self._extract_by_source(text, user_id, source_type)
        for ext in raw_extractions:
            # Apply corrections (Approach C)
            corrected_pred = self._corrections.apply(
                ext["predicate"], ext.get("object", ""))
            candidates.append(ExtractionCandidate(
                subject=ext.get("subject", user_id),
                predicate=corrected_pred,
                object_value=ext["object"],
                confidence=ext.get("confidence", 0.6),
                source=f"source_type:{source_type.type}",
            ))

        # Approach A: Entity co-occurrence
        entities = _extract_entities(text)
        if len(entities) >= 2:
            for i in range(len(entities)):
                for j in range(i + 1, min(i + 3, len(entities))):
                    between = _extract_between_text(text, entities[i], entities[j])
                    if between and len(between.split()) <= 5:
                        self._cooccurrence.record(entities[i], entities[j], between)
                        candidates.append(ExtractionCandidate(
                            subject=entities[i],
                            predicate=_normalize_predicate(between),
                            object_value=entities[j],
                            confidence=0.6,
                            source="cooccurrence",
                        ))

        # Approach B: Contextual inference from self-reference patterns
        lower = text.lower()
        is_self = any(lower.startswith(m) or f" {m} " in f" {lower} "
                      for m in _SELF_MARKERS)
        if is_self:
            for ent in entities:
                between = _extract_between_text(text, self._get_self_ref(lower), ent)
                if between and 1 <= len(between.split()) <= 4:
                    candidates.append(ExtractionCandidate(
                        subject=user_id,
                        predicate=_normalize_predicate(between),
                        object_value=ent,
                        confidence=0.7,
                        source="self_reference",
                    ))

        # Approach E: Rank candidates by confidence, keep best
        candidates.sort(key=lambda c: c.confidence, reverse=True)
        seen: set[tuple[str, str, str]] = set()  # (subject, predicate, object)

        for cand in candidates:
            if cand.confidence < self._candidate_threshold:
                continue
            key = (cand.subject.lower(), cand.predicate.lower(),
                   cand.object_value.lower())
            if key in seen:
                continue
            seen.add(key)

            mems.append(Memory(
                scope=scope,
                context=context,
                user_id=user_id,
                subject=cand.subject,
                predicate=cand.predicate,
                object_value=cand.object_value[:500],
                source_text=text[:500],
                is_negation=cand.is_negation,
                source_type="user_stated",
                confidence=cand.confidence,
                created_at=now,
                updated_at=now,
                last_accessed=now,
                metadata={"source_tool": source_tool, "extraction_source": cand.source},
            ))

        return mems

    def _extract_by_source(self, text: str, user_id: str,
                           source_type: SourceType) -> list[dict]:
        """Route to source-specific extractor."""
        if source_type.type == "kv_pairs":
            return _extract_kv_pairs(text, user_id)
        elif source_type.type == "email_signature":
            return _extract_email_signature(text, user_id)
        elif source_type.type == "commit":
            return _extract_commit(text, user_id)
        elif source_type.type == "social":
            return _extract_social(text, user_id)
        elif source_type.type == "resume":
            return _extract_resume(text, user_id)
        elif source_type.type == "fragment":
            return _extract_fragment(text, user_id)
        else:
            return []  # Sentence type falls back to grammar parser or entity co-occurrence

    def _get_self_ref(self, lower_text: str) -> str:
        """Find the self-reference token in text."""
        for marker in ("i'm", "i am", "i've", "i have", "i"):
            if lower_text.startswith(marker):
                return marker
            idx = lower_text.find(f" {marker} ")
            if idx >= 0:
                return marker
        return "i"

    def learn_correction(self, wrong_pred: str, correct_pred: str,
                         context: str = "") -> None:
        """Learn from a user correction."""
        self._corrections.learn(wrong_pred, correct_pred, context)


def _normalize_predicate(text: str) -> str:
    """Convert a between-text fragment into a predicate string."""
    # Remove common articles and prepositions from start
    cleaned = text.strip().lower()
    for prefix in ("is ", "am ", "are ", "was ", "were ", "have ", "has ", "had ",
                   "do ", "does ", "did ", "will ", "would ", "can ", "could ",
                   "the ", "a ", "an "):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]

    # Replace spaces with underscores
    cleaned = re.sub(r'\s+', '_', cleaned)
    # Remove non-alphanumeric except underscores
    cleaned = re.sub(r'[^a-z0-9_]', '', cleaned)
    # Collapse multiple underscores
    cleaned = re.sub(r'_+', '_', cleaned)
    return cleaned.strip('_') or "related_to"
