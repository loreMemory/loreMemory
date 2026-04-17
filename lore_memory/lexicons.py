"""
Canonical English lexicons.

Single source of truth for every hand-curated word list and domain-knowledge
regex in the codebase. Previously these were inlined — and drifted — across
extraction.py, extraction_gf.py, normalization.py, retrieval.py, engine.py.
One canonical definition per concept; call sites compose by union.

Scope: English. These are **product vocabulary**, not language-agnostic
structure. Intentionally kept in one file so the domain boundary is explicit
and replaceable.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
#  First-person tokens
# ---------------------------------------------------------------------------

# Core first-person singular — these tokens always resolve to "the user"
# when they appear as a subject.
FIRST_PERSON_SUBJECTS: frozenset[str] = frozenset({
    "i", "me", "my", "mine", "myself",
})

# Synonymic aliases that some call sites treat as first-person.
FIRST_PERSON_ALIASES: frozenset[str] = frozenset({
    "user", "current_user", "the user",
})

# Expanded form with contracted-verb variants; used when detecting subject
# in raw untokenized chat text where contractions are common.
FIRST_PERSON_WITH_CONTRACTIONS: frozenset[str] = FIRST_PERSON_SUBJECTS | frozenset({
    "i'm", "im", "i've", "ive", "i'll", "ill", "i'd", "id",
})


# ---------------------------------------------------------------------------
#  Relationship and attribute nouns
# ---------------------------------------------------------------------------

# Kinship terms — family and romantic partnership.
KINSHIP_NOUNS: frozenset[str] = frozenset({
    "wife", "husband", "partner", "spouse", "girlfriend", "boyfriend",
    "fiancee", "fiance",
    "brother", "sister", "mother", "father", "mom", "dad",
    "son", "daughter", "aunt", "uncle", "cousin",
    "grandmother", "grandfather", "grandma", "grandpa",
    "parent", "parents", "kids", "children", "family",
})

# Social / professional roles the user may mention someone by.
PROFESSIONAL_NOUNS: frozenset[str] = frozenset({
    "manager", "boss", "supervisor", "director",
    "coworker", "colleague", "friend", "neighbor", "mentor",
    "teacher", "professor", "therapist", "doctor", "lawyer", "dentist",
})

# Pet nouns — treated like relationships in the "My X is Y" pattern.
PET_NOUNS: frozenset[str] = frozenset({"pet", "dog", "cat"})

# Personal attribute nouns — "my name/birthday/email is Y" → predicate=Y.
PERSONAL_ATTRIBUTE_NOUNS: frozenset[str] = frozenset({
    "name", "birthday", "email", "phone", "age", "salary",
    "favorite",
})

# Convenient unions for the common call-site mixes.

# The "My X is Y" / "my X's name is" patterns in the grammar parser — rewrite
# the predicate when X is any of these.
REL_NOUNS_FOR_MY_PATTERN: frozenset[str] = (
    KINSHIP_NOUNS | PROFESSIONAL_NOUNS | PET_NOUNS | PERSONAL_ATTRIBUTE_NOUNS
)

# Third-party subject detection: "My <X>" → X is about someone else.
# Preserves the original extraction.py set exactly — no personal attributes
# (those are about the user, not a third party), no "team" (too
# ambiguous), keeps "fiancee"/"fiance"/"parent" as useful additions.
THIRD_PARTY_SUBJECT_NOUNS: frozenset[str] = (
    KINSHIP_NOUNS | PROFESSIONAL_NOUNS | PET_NOUNS
)

# Retrieval subject-alignment boost list — kinship + the most common
# professional role words users actually query on.
RETRIEVAL_RELATIONSHIP_NOUNS: frozenset[str] = KINSHIP_NOUNS | frozenset({
    "manager", "boss", "friend", "colleague",
})

# Normalization's relationship set — kinship + common pros + team.
SUBJECT_RESOLVER_RELATIONSHIP_NOUNS: frozenset[str] = (
    KINSHIP_NOUNS | frozenset({
        "manager", "boss", "supervisor", "director",
        "coworker", "colleague", "friend", "neighbor", "mentor",
        "teacher", "professor",
    }) | PET_NOUNS | frozenset({"team"})
)


# ---------------------------------------------------------------------------
#  Stopwords
# ---------------------------------------------------------------------------

# Stopwords used during FTS query tokenization. Richer set — these words
# are stripped from lexical match terms so a 2-word query like "my manager"
# doesn't return everything with "my" in it.
FTS_STOPWORDS: frozenset[str] = frozenset({
    "the", "is", "at", "in", "on", "of", "to", "and", "or", "an", "it", "be",
    "as", "do", "by", "for", "was", "are", "has", "had", "not", "but", "its",
    "he", "she", "we", "my", "what", "which", "who", "this", "that", "am",
    "been", "have", "does", "did", "will", "would", "can", "could",
    "about", "with", "from", "into", "where", "when", "how", "all",
    "me", "him", "them", "you", "your", "our", "their", "tell", "know",
})

# Stopwords used when extracting "content words" from a user query to
# compare against stored memories. Narrower than FTS stopwords — keeps
# words like "for", "of", "and" out of overlap scoring but leaves e.g.
# "about", "from" in since they can signal intent.
QUERY_CONTENT_STOPWORDS: frozenset[str] = frozenset({
    "what", "where", "who", "when", "how", "does", "do", "is", "are",
    "the", "a", "an", "my", "i", "me", "to", "in", "at",
    "for", "of", "and", "or", "on", "with", "s",
})

# Narrow stopword set for query-intent detection (is this a specific
# question vs. a broad summary?). Keep only the universal question
# words + determiners.
QUERY_INTENT_STOPWORDS: frozenset[str] = frozenset({
    "what", "where", "who", "when", "how", "does", "do", "is", "are",
    "the", "a", "an", "my", "i",
})


# ---------------------------------------------------------------------------
#  Commit-message detection (for repo-scope ingestion and profile filtering)
# ---------------------------------------------------------------------------

# Type tokens the conventional-commits spec recognises.
COMMIT_TYPE_PREFIXES: tuple[str, ...] = (
    "fix", "feat", "chore", "refactor", "docs",
    "style", "perf", "ci", "build", "revert", "test",
)

# Regex matching `<type>(scope)?: ...` at line start.
COMMIT_MSG_RE: re.Pattern[str] = re.compile(
    r'^(feat|fix|docs|style|refactor|test|chore|perf|ci|build|revert)'
    r'\s*(\([^)]+\))?\s*:'
)
