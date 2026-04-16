"""
Extraction pipeline — v5: grammar-based, zero dictionaries.

Architecture:
  1. Always store raw text as a searchable memory (FTS-indexed)
  2. Parse each sentence by English grammar structure:
     [Subject] [Negation?] [Auxiliary?] [Verb] [Preposition?] [Object]
  3. The verb+preposition becomes the predicate automatically
  4. No verb map, no regex patterns, no predefined lists

The verb is identified by GRAMMAR POSITION, not by dictionary lookup.
The predicate is the verb itself (lemmatized + preposition).
Adding a new phrasing requires NOTHING — the parser handles it structurally.

No regex. No LLM. No dictionaries. English-only. Pure grammar.
"""

from __future__ import annotations

import re
import time
from typing import Any, Protocol

from lore_memory.store import Memory


# ---------------------------------------------------------------------------
#  Predicate normalization
# ---------------------------------------------------------------------------

def _norm(pred: str) -> str:
    p = pred.lower().strip()
    p = re.sub(r"[-\s]+", "_", p)
    p = re.sub(r"[^a-z0-9_]", "", p)
    p = re.sub(r"_+", "_", p)
    return p.strip("_") or "stated"


# ---------------------------------------------------------------------------
#  English grammar constants (closed-class words — these are FINITE by
#  definition in any language, not an extensibility problem)
# ---------------------------------------------------------------------------

_PRONOUNS = frozenset({
    "i", "you", "he", "she", "it", "we", "they",
    "me", "him", "her", "us", "them",
})

_POSSESSIVES = frozenset({
    "my", "your", "his", "her", "its", "our", "their",
})

_COPULAS = frozenset({
    "am", "is", "are", "was", "were", "be",
})

_POSSESSIVE_VERBS = frozenset({
    "have", "has", "had",
})

# Modals are ALWAYS followed by a base verb — always skip them
_MODALS = frozenset({
    "will", "would", "shall", "should",
    "can", "could", "may", "might", "must",
})

# Full set for reference
_AUXILIARIES = _COPULAS | _POSSESSIVE_VERBS | _MODALS | frozenset({
    "do", "does", "did", "been", "being", "having",
})

_NEGATIONS = frozenset({
    "not", "n't", "never", "no", "hardly", "barely", "rarely", "seldom",
})

_PREPOSITIONS = frozenset({
    "in", "at", "for", "to", "on", "up", "out", "off", "with", "from",
    "about", "of", "by", "into", "through", "over", "under", "between",
    "after", "before", "during", "without", "within", "across",
})

_DETERMINERS = frozenset({
    "a", "an", "the", "this", "that", "these", "those",
    "some", "any", "every", "each", "all", "both", "few",
    "many", "much", "several", "no",
})

_CONJUNCTIONS = frozenset({
    "and", "but", "or", "yet", "so", "because", "since",
    "although", "though", "while", "when", "if", "unless",
    "that", "which", "who", "whom", "whose", "where",
})

_FILLERS = frozenset({
    "well", "yeah", "yep", "yes", "nah", "nope", "actually", "honestly",
    "basically", "really", "just", "also", "so", "okay", "ok",
    "lol", "haha", "idk", "tbh", "imo", "btw", "hmm", "oh", "wow",
    "mostly", "probably", "maybe", "perhaps", "definitely", "certainly",
})


# ---------------------------------------------------------------------------
#  Trivial / noise message filter
# ---------------------------------------------------------------------------

def _is_trivial(text: str) -> bool:
    """Return True if text is trivial chat noise that shouldn't be stored as raw text.

    Filters: greetings, acknowledgements, filler, very short non-informational text.
    Does NOT filter: short but factual text like "I use Python" or "Berlin. 3 years."
    """
    t = text.strip().lower()

    # Too short to be meaningful (under 3 words and no factual content)
    words = t.split()
    if len(words) <= 2:
        # Exception: short but factual ("I'm vegetarian", "Berlin. 3 years.")
        _FACTUAL_MARKERS = {"i", "my", "we", "our", "i'm", "i've", "im"}
        if any(w in _FACTUAL_MARKERS for w in words):
            return False
        # Check for proper nouns or numbers (likely factual)
        if any(w[0].isupper() for w in text.strip().split() if w):
            return False
        if any(c.isdigit() for c in t):
            return False
        return True

    # Known noise patterns
    _NOISE_STARTS = {
        "ok", "okay", "sure", "yeah", "yep", "nope", "nah",
        "lol", "haha", "heh", "lmao", "rofl",
        "nice", "cool", "great", "thanks", "thank",
        "brb", "gtg", "ttyl", "bye", "hi", "hey",
        "happy friday", "happy monday", "good morning", "good night",
        "sounds good", "looks good", "lgtm",
        "can someone", "anyone want", "reminder:",
        "back from", "gonna be ooo",
    }
    for noise in _NOISE_STARTS:
        if t.startswith(noise):
            # But not if it continues with factual content
            rest = t[len(noise):].strip()
            if len(rest.split()) <= 3:
                return True

    # Very short with no verb-like content
    if len(words) <= 4:
        _INFO_VERBS = {"work", "live", "use", "like", "have", "am", "is", "was",
                       "graduated", "moved", "started", "learned", "built", "speak",
                       "code", "prefer", "love", "hate", "studying", "reading",
                       "drink", "eat", "run", "play", "drive"}
        if not any(w in _INFO_VERBS for w in words):
            # No informational verb -> likely noise
            # Exception: contains proper noun or number
            if not any(w[0].isupper() for w in text.strip().split() if w):
                if not any(c.isdigit() for c in t):
                    return True

    return False

# Adverbs that can appear between auxiliary and main verb — skip in verb search
_ADVERBS = frozenset({
    "currently", "still", "now", "already", "always", "usually",
    "sometimes", "often", "recently", "lately", "presently",
    "mostly", "probably", "maybe", "perhaps", "definitely", "certainly",
    "never", "ever", "just", "really", "actually", "also",
})

# Multi-word modifiers that should be consumed as a unit
_MULTI_WORD_NEGATIONS = [
    "no longer", "not anymore", "used to", "no more",
]

# Single-valued predicates (for contradiction detection)
# These are semantically single-valued — a person has one current city, one current job, etc.
SINGLE_VALUED: set[str] = {
    "live_in", "lives_in", "based_in", "moved_to", "relocated_to", "reside_in",
    "work_at", "works_at", "work_for", "works_for", "employed_at", "joined",
    "am", "is",  # "I am a X" — identity is single-valued
    "name", "age", "email", "phone", "birthday", "nationality",
}


# ---------------------------------------------------------------------------
#  Verb lemmatization (no external library)
# ---------------------------------------------------------------------------

def _lemmatize(verb: str) -> str:
    """Reduce English verb to approximate base form.
    Uses the verb itself as predicate when unsure — better to be consistent
    than to guess wrong."""
    v = verb.lower()
    if len(v) <= 2:
        return v

    # Irregular common verbs (finite closed set — these are ALL of the
    # common English irregulars. Not a scalability problem.)
    _IRR = {
        "am": "be", "is": "be", "are": "be", "was": "be", "were": "be",
        "has": "have", "had": "have",
        "does": "do", "did": "do",
        "went": "go", "gone": "go",
        "said": "say", "told": "tell",
        "knew": "know", "known": "know",
        "spoke": "speak", "spoken": "speak",
        "drove": "drive", "driven": "drive",
        "wrote": "write", "written": "write",
        "built": "build", "made": "make",
        "left": "leave", "ran": "run",
        "taught": "teach", "brought": "bring",
        "thought": "think", "bought": "buy",
        "got": "get", "been": "be",
        "took": "take", "taken": "take",
        "gave": "give", "given": "give",
        "saw": "see", "seen": "see",
        "came": "come", "found": "find",
        "sat": "sit", "stood": "stand",
        "met": "meet", "held": "hold",
        "kept": "keep", "felt": "feel",
        "lost": "lose", "sent": "send",
        "read": "read", "led": "lead",
        "spent": "spend", "grew": "grow", "grown": "grow",
        "began": "begin", "begun": "begin",
        "chose": "choose", "chosen": "choose",
        "ate": "eat", "eaten": "eat",
        "fell": "fall", "fallen": "fall",
        "forgot": "forget", "forgotten": "forget",
        "hid": "hide", "hidden": "hide",
        "rode": "ride", "ridden": "ride",
        "rose": "rise", "risen": "rise",
        "shook": "shake", "shaken": "shake",
        "stole": "steal", "stolen": "steal",
        "swore": "swear", "sworn": "swear",
        "woke": "wake", "woken": "wake",
        "wore": "wear", "worn": "wear",
    }
    if v in _IRR:
        return _IRR[v]

    # -ying: strip -ing, keep the y (studying→study, playing→play)
    if v.endswith("ying"):
        return v[:-3]  # studying → study (stem already has 'y')

    # -ied / -ies: y-stem
    if v.endswith("ied"):
        return v[:-3] + "y"
    if v.endswith("ies"):
        return v[:-3] + "y"

    # -ing: strip suffix, handle doubled consonant and silent-e
    if v.endswith("ing") and len(v) >= 5:
        stem = v[:-3]
        if len(stem) >= 2 and stem[-1] == stem[-2]:
            return stem[:-1]  # running → run, sitting → sit
        # CVC rule: if stem ends consonant-vowel-consonant pattern, add 'e'
        # liv→live, mak→make, driv→drive, manag→manage
        # But NOT work, speak, build, learn (consonant clusters at end)
        _V = set("aeiou")
        if (len(stem) >= 3
                and stem[-1] not in _V           # ends in consonant
                and stem[-2] in _V               # preceded by vowel
                and stem[-3] not in _V):         # that vowel preceded by consonant
            return stem + "e"
        return stem

    # -ed: handle words where base ends in 'e' (graduated→graduate)
    if v.endswith("ed") and len(v) >= 5:
        stem_ed = v[:-2]  # strip -ed
        if len(stem_ed) >= 2 and stem_ed[-1] == stem_ed[-2]:
            return stem_ed[:-1]  # stopped → stop
        # Words ending in -ated/-eted/-ited/-oted/-uted: base ends in -ate etc
        # graduated→graduate, relocated→relocate, completed→complete
        if v.endswith(("ated", "eted", "ited", "oted", "uted")):
            return v[:-1]  # strip just -d
        # CVC rule for short stems: moved→move, liked→like
        _V = set("aeiou")
        if (len(stem_ed) >= 3
                and stem_ed[-1] not in _V
                and stem_ed[-2] in _V
                and stem_ed[-3] not in _V):
            return v[:-1]  # strip -d, keep the 'e'
        return stem_ed  # worked→work, helped→help

    # -es: uses→use, watches→watch, fixes→fix
    if v.endswith("es") and len(v) >= 4 and not v.endswith("ies"):
        # If stripping -s gives a word ending in 'e', prefer that: uses→use
        if v[-3] == "s" and v.endswith("ses"):
            return v[:-1]  # uses→use (not "us")
        if v[-3] in "sxzh":
            return v[:-2]
        return v[:-1]

    # -s: likes → like, works → work
    if v.endswith("s") and len(v) >= 4 and not v.endswith("ss") and not v.endswith("us"):
        return v[:-1]

    return v


# ---------------------------------------------------------------------------
#  Grammar-based SVO parser
# ---------------------------------------------------------------------------

def _looks_like_verb_form(word: str) -> bool:
    """Heuristic: does this word look like an English verb form?
    Checks for -ing, -ed, -en endings, but avoids false positives
    on proper nouns, common nouns, and adjectives.

    Capitalized words are accepted as verb forms ONLY for unambiguous
    endings like -ing (Working, Developing). For -ed/-en endings,
    capitalized words are rejected to avoid false positives on proper
    nouns (Marcus, Ahmed, Alfred).
    """
    w = word.lower()
    if w in _AUXILIARIES:
        return True
    # Too short to be a verb form
    if len(w) < 4:
        return False
    # -ing is unambiguous — always a verb form, even when capitalized
    # "Working", "Developing" at sentence start are valid verb forms
    if w.endswith("ing") and len(w) >= 5:
        return True
    # -ed and -en: reject capitalized words to avoid false positives
    # on proper nouns (Marcus, Ahmed, Alfred, hundred)
    if word[0].isupper():
        return False
    if w.endswith("ed") and len(w) >= 5:
        return True
    if w.endswith("en") and len(w) >= 5:
        if w in ("been", "seen", "taken", "given", "written", "spoken",
                 "broken", "chosen", "driven", "eaten", "fallen",
                 "forgotten", "frozen", "gotten", "hidden", "proven",
                 "risen", "shaken", "stolen", "sworn", "woken", "worn"):
            return True
        return False
    return False


def _clean_token(word: str) -> str:
    """Strip punctuation from a token."""
    return word.strip(".,!?;:\"'()[]{}").lower()


# ---------------------------------------------------------------------------
#  Temporal markers — detect time-bound facts
# ---------------------------------------------------------------------------

_TEMPORAL_PAST = re.compile(
    r'\b(?:last\s+(?:week|month|year|night|monday|tuesday|wednesday|thursday|friday|saturday|sunday)'
    r'|yesterday|(?:a\s+)?(?:few|couple)\s+(?:days?|weeks?|months?)\s+ago'
    r'|previously|formerly|back\s+in\s+\d{4})\b',
    re.IGNORECASE,
)

_TEMPORAL_UNTIL = re.compile(
    r'\buntil\s+(\w+(?:\s+\w+){0,2})\b',
    re.IGNORECASE,
)

_TEMPORAL_DURATION = re.compile(
    r'\bfor\s+(\d+)\s+(days?|weeks?|months?|years?)\b',
    re.IGNORECASE,
)

# Map duration units to seconds
_DURATION_SECONDS = {
    "day": 86400, "days": 86400,
    "week": 604800, "weeks": 604800,
    "month": 2592000, "months": 2592000,
    "year": 31536000, "years": 31536000,
}


def _detect_temporal(text: str) -> tuple[float | None, str]:
    """Detect temporal markers in text.
    Returns (valid_until_timestamp_or_None, cleaned_text_without_temporal)."""
    import time as _time

    # "until Friday" / "until next month"
    m = _TEMPORAL_UNTIL.search(text)
    if m:
        # Approximate: "until" implies a few days
        cleaned = text[:m.start()].strip() + " " + text[m.end():].strip()
        return _time.time() + 604800, cleaned.strip()  # ~1 week

    # "for 3 months"
    m = _TEMPORAL_DURATION.search(text)
    if m:
        count = int(m.group(1))
        unit = m.group(2).lower()
        secs = _DURATION_SECONDS.get(unit, 86400) * count
        cleaned = text[:m.start()].strip() + " " + text[m.end():].strip()
        return _time.time() + secs, cleaned.strip()

    # "last week" / "yesterday" / "a few days ago" → already past
    m = _TEMPORAL_PAST.search(text)
    if m:
        cleaned = text[:m.start()].strip() + " " + text[m.end():].strip()
        return _time.time() - 1, cleaned.strip()  # already expired

    return None, text


def _expand_compound_verbs(sentence: str) -> list[str]:
    """Split compound verb sentences into separate clauses.
    'I use Python and love Rust' → ['I use Python', 'I love Rust']
    'I like cats and dogs' → ['I like cats and dogs'] (no split — 'dogs' is noun)
    """
    # Find "and" positions
    words = sentence.split()
    results = []
    i = 0
    last_split = 0

    # Find the subject (first pronoun or word before first verb)
    subject = ""
    for w in words:
        wl = w.lower().strip(".,!?;:")
        if wl in ("i", "we", "he", "she", "they", "you") or wl in _POSSESSIVES:
            subject = w
            break
        if w[0].isupper() and wl not in _DETERMINERS and wl not in _FILLERS:
            subject = w
            break

    for idx in range(len(words)):
        wl = words[idx].lower().strip(".,!?;:")
        if wl != "and" or idx == 0 or idx >= len(words) - 1:
            continue

        # Check if word after "and" looks like a verb
        next_word = words[idx + 1]
        next_lower = next_word.lower().strip(".,!?;:")

        is_verb = (
            _looks_like_verb_form(next_word)
            or next_lower in _AUXILIARIES
            or next_lower in _MODALS
            or next_lower in _NEGATIONS
            # Common base verbs are short, followed by a noun/prep
            or (len(next_lower) >= 3 and next_lower not in _DETERMINERS
                and next_lower not in _PREPOSITIONS
                and next_lower not in _CONJUNCTIONS
                and next_lower not in _FILLERS
                # Heuristic: if the word before "and" is NOT a determiner/adjective,
                # the previous part was Object, so this is a new verb
                and idx >= 2 and words[idx - 1].lower().strip(".,!?;:") not in _DETERMINERS)
        )

        if is_verb:
            # Split here: [... Python] and [love Rust]
            clause1 = " ".join(words[last_split:idx]).strip()
            if clause1:
                results.append(clause1)
            # Prepend subject to second clause: "I love Rust"
            remaining_words = words[idx + 1:]
            # Check if subject is already there
            first_remaining = remaining_words[0].lower().strip(".,!?;:") if remaining_words else ""
            if first_remaining not in ("i", "we", "he", "she", "they", "you") and subject:
                clause2 = subject + " " + " ".join(remaining_words)
            else:
                clause2 = " ".join(remaining_words)
            results.append(clause2.strip())
            return results  # Only split on first compound verb

    # No compound verb found — return as-is
    return [sentence]


_ABBREVIATIONS = re.compile(
    r'\b(?:Dr|Mr|Mrs|Ms|Prof|Jr|Sr|St|Ave|Blvd|Dept|Gen|Gov|Sgt|Corp|Inc|Ltd|vs|etc|approx|i\.e|e\.g)\.\s',
    re.IGNORECASE,
)

def _split_sentences(text: str) -> list[str]:
    """Split text into sentences.
    Avoids splitting on '.' inside abbreviations (Dr., Mr., U.S.) or
    decimal numbers (3.14, v2.0)."""
    # Protect abbreviations and decimals by replacing their ". " with a placeholder
    protected = text
    # Protect abbreviations: "Dr. Smith" → "Dr.\x00Smith"
    for m in reversed(list(_ABBREVIATIONS.finditer(protected))):
        s, e = m.start(), m.end()
        inner = protected[s:e]
        protected = protected[:s] + inner.replace(". ", ".\x00") + protected[e:]

    parts = re.split(r'(?<=[.!?])\s+|\n+', protected)
    # Restore placeholders
    parts = [p.replace('\x00', ' ') for p in parts]
    # Split on clause boundaries
    result = []
    for p in parts:
        # Split on coordinating conjunctions before pronouns
        sub = re.split(r'\b(?:and|but|although|however)\s+(?=I\s)', p, flags=re.IGNORECASE)
        for s in sub:
            # Split on "that" introducing a subordinate clause
            that_split = re.split(r'\bthat\s+(?=[A-Z])', s)
            for part in that_split:
                part = part.strip()
                if not part:
                    continue
                # Split compound verbs: "I use Python and love Rust"
                # → ["I use Python", "I love Rust"]
                # Detect: "X and [verb]" where the word after "and" looks like a verb
                expanded = _expand_compound_verbs(part)
                result.extend(expanded)
    return result


def parse_sentence(sentence: str) -> dict | None:
    """Parse an English sentence into structured components.

    Returns dict with: subject, verb, preposition, object, is_negation, predicate
    Or None if the sentence can't be parsed.

    The parser identifies grammar roles by POSITION, not by dictionary:
    1. Subject = words before the verb (pronouns, possessives + nouns, proper nouns)
    2. Verb = first word after subject that is not auxiliary/negation/filler
    3. Preposition = word immediately after verb if it's a known preposition
    4. Object = everything after verb+preposition, up to clause boundary
    """
    # Expand contractions before parsing
    sentence = re.sub(r"\bI'm\b", "I am", sentence, flags=re.IGNORECASE)
    sentence = re.sub(r"\bI've\b", "I have", sentence, flags=re.IGNORECASE)
    sentence = re.sub(r"\bI'll\b", "I will", sentence, flags=re.IGNORECASE)
    sentence = re.sub(r"\bI'd\b", "I would", sentence, flags=re.IGNORECASE)
    sentence = re.sub(r"\b(\w+)'re\b", r"\1 are", sentence)
    # Expand "'s" to "is" only for pronouns and similar; keep possessive for nouns
    _CONTRACTION_PRONOUNS = {"he", "she", "it", "that", "what", "who", "there",
                             "here", "everyone", "somebody", "someone",
                             "everything", "anything", "nothing", "nobody"}
    def _expand_s(m: re.Match) -> str:
        word = m.group(1)
        if word.lower() in _CONTRACTION_PRONOUNS:
            return word + " is"
        return m.group(0)  # keep as possessive
    sentence = re.sub(r"\b(\w+)'s\b", _expand_s, sentence)
    sentence = re.sub(r"\b(\w+)'ve\b", r"\1 have", sentence)
    sentence = re.sub(r"\b(\w+)'ll\b", r"\1 will", sentence)

    words = sentence.split()
    if len(words) < 2:
        return None

    tokens = [_clean_token(w) for w in words]

    # Pre-check: detect "My X is Y" pattern where X is a relationship/role noun
    # This prevents the copula from being skipped when the object starts with a verb-form word
    _MY_REL_NOUNS_EARLY = frozenset({
        "manager", "boss", "supervisor", "director",
        "wife", "husband", "partner", "spouse", "girlfriend", "boyfriend",
        "brother", "sister", "mother", "father", "mom", "dad",
        "son", "daughter", "aunt", "uncle", "cousin",
        "grandmother", "grandfather", "grandma", "grandpa",
        "friend", "coworker", "colleague", "neighbor", "mentor",
        "teacher", "professor", "therapist", "doctor", "lawyer", "dentist",
        "pet", "dog", "cat", "favorite",
        "name", "birthday", "email", "phone", "age", "salary",
    })
    _is_my_rel_pattern = False
    if tokens[0] == "my" and len(tokens) >= 3:
        for k in range(1, min(4, len(tokens))):
            tk = re.sub(r"'s$", "", tokens[k]) if tokens[k] else tokens[k]
            if tk in _MY_REL_NOUNS_EARLY:
                # Check there's a copula somewhere after
                for c in range(k + 1, min(k + 4, len(tokens))):
                    if tokens[c] in _COPULAS:
                        _is_my_rel_pattern = True
                        break
                break

    # Phase 1: Identify subject span
    subj_end = 0
    is_negation = False
    found_subject_word = False

    for i, tok in enumerate(tokens):
        if not tok:
            continue

        # Subject words: pronouns, possessives, and the nouns they modify
        if tok in _PRONOUNS or tok in _POSSESSIVES:
            found_subject_word = True
            subj_end = i + 1
            # After possessive, grab the following noun(s) until we hit a verb
            if tok in _POSSESSIVES:
                for j in range(i + 1, min(i + 5, len(tokens))):
                    jt = tokens[j]
                    if jt in _AUXILIARIES or jt in _PREPOSITIONS or jt in _CONJUNCTIONS:
                        break
                    # Stop if this looks like a verb (not part of the subject noun phrase)
                    if (found_subject_word and j > i + 1
                            and _looks_like_verb_form(words[j])):
                        break
                    # Stop if it ends in -s and we already have a noun (likely a verb)
                    if (found_subject_word and j > i + 1 and len(jt) >= 4
                            and jt.endswith("s") and not jt.endswith("ss")
                            and not jt.endswith("us")):
                        break
                    # Stop if this word is followed by a preposition (likely a verb)
                    if j > i + 1:
                        next_after = tokens[j + 1] if j + 1 < len(tokens) else ""
                        if next_after in _PREPOSITIONS:
                            break
                    subj_end = j + 1
            continue

        # Capitalized word at start (proper noun) — part of subject
        # Skip determiners, verb forms (Decided, Started, Working)
        if i == 0 and words[i][0].isupper() and tok not in _FILLERS and tok not in _DETERMINERS:
            # Check if this is actually a verb at sentence start (implicit subject)
            # "Decided to use X" → verb. "Marcus lives in X" → proper noun.
            # For -ed words: only verb if followed by preposition/to (confirms verb phrase)
            next_tok = tokens[i + 1] if i + 1 < len(tokens) else ""
            is_verb_at_start = (
                _looks_like_verb_form(words[i])  # catches -ing always
                or (len(tok) >= 5 and tok.endswith("ed")
                    and (next_tok in _PREPOSITIONS or next_tok.endswith("ing")))
                or tok in _AUXILIARIES
            )
            if is_verb_at_start:
                pass  # don't treat as subject — fall through
            else:
                found_subject_word = True
                subj_end = i + 1
            # Grab multi-word proper nouns
            for j in range(i + 1, min(i + 3, len(tokens))):
                if words[j][0].isupper() and tokens[j] not in _AUXILIARIES:
                    subj_end = j + 1
                else:
                    break
            continue

        # "The X" at start — "The" is determiner, next word(s) are the subject
        if i == 0 and tok in _DETERMINERS:
            for j in range(i + 1, min(i + 4, len(tokens))):
                jt = tokens[j]
                if jt in _AUXILIARIES or jt in _PREPOSITIONS or jt in _CONJUNCTIONS:
                    break
                # If we already have a subject word and this one looks like
                # a verb form (-s/-es ending), stop — it's the verb, not subject
                if found_subject_word and _looks_like_verb_form(words[j]):
                    break
                if found_subject_word and len(jt) >= 4 and (jt.endswith("s") and not jt.endswith("ss") and not jt.endswith("us")):
                    break
                found_subject_word = True
                subj_end = j + 1
            continue

        # Filler at start — skip
        if not found_subject_word and tok in _FILLERS:
            continue

        break

    # If no explicit subject found, check if sentence starts with a verb
    # (implicit subject: "Decided to use PostgreSQL" → subject is contextual)
    if not found_subject_word:
        # Try to parse as verb-first (imperative/implicit subject)
        first_tok = tokens[0] if tokens else ""
        if first_tok and first_tok not in _FILLERS and first_tok not in _DETERMINERS:
            # Treat position 0 as verb start, subject will be filled by caller
            subj_end = 0
            found_subject_word = True  # mark as parseable
        else:
            return None

    # Phase 2: Handle multi-word negation/retraction markers
    # "I used to work at X" → skip "used to", verb is "work"
    # "I no longer live in X" → skip "no longer", verb is "live"
    remaining_text = " ".join(tokens[subj_end:])
    skip_words = 0
    for mwn in _MULTI_WORD_NEGATIONS:
        if remaining_text.startswith(mwn):
            is_negation = True
            skip_words = len(mwn.split())
            break

    # Phase 3: Find verb (skip negations, auxiliaries, fillers)
    verb_idx = -1
    aux_idx = -1
    search_start = subj_end + skip_words

    for i in range(search_start, len(tokens)):
        tok = tokens[i]
        if not tok:
            continue

        # Contraction handling: "don't" → negation
        if tok.endswith("n't") or tok in ("dont", "doesnt", "didnt", "cant",
                                          "wont", "wouldnt", "shouldnt"):
            is_negation = True
            continue

        if tok in _NEGATIONS:
            is_negation = True
            continue

        if tok in _FILLERS or tok in _ADVERBS:
            continue

        # Modals (can, will, must, etc.) — ALWAYS skip, next word is the verb
        if tok in _MODALS:
            continue

        # "do/does/did" as auxiliary — skip (the next word is the main verb)
        if tok in ("do", "does", "did"):
            continue

        # "been/being/having" — auxiliary forms, skip
        if tok in ("been", "being", "having"):
            continue

        if tok in _COPULAS or tok in _POSSESSIVE_VERBS:
            aux_idx = i
            # Copula/have: check if next word is a verb form (-ing, -ed, -en)
            # If yes: this is an auxiliary ("I am working", "I have finished")
            # If no: this IS the main verb ("I am a dev", "I have kids")
            next_main_original = None
            for j in range(i + 1, min(i + 6, len(tokens))):
                jt = tokens[j]
                if jt in _DETERMINERS or jt in _FILLERS or jt in _ADVERBS:
                    continue
                next_main_original = words[j]  # original case for proper noun detection
                break
            if (next_main_original and _looks_like_verb_form(next_main_original)
                    and not _is_my_rel_pattern):
                continue  # skip auxiliary, main verb is next
            else:
                verb_idx = i  # copula/have IS the verb
                break

        # This is the main verb
        verb_idx = i
        break

    if verb_idx == -1:
        return None

    verb = tokens[verb_idx]

    # Fix: if verb is a preposition and there was a copula/auxiliary, combine
    # "been at" → "be_at", "been to" → "be_to"
    if verb in _PREPOSITIONS and aux_idx >= 0:
        verb = f"be_{verb}"

    # Phase 3a: Copula + adjective + preposition pattern
    # "I am passionate about ML" → pred=passionate_about, obj=ML
    # "I am good at Python" → pred=good_at, obj=Python
    # When copula is followed by [adjective] [preposition], merge into predicate
    if verb in _COPULAS:
        adj_idx = verb_idx + 1
        if adj_idx < len(tokens):
            adj_tok = tokens[adj_idx]
            # Check if next word is an adjective (not a det, not a prep, not a verb form)
            if (adj_tok not in _DETERMINERS
                    and adj_tok not in _PREPOSITIONS
                    and adj_tok not in _AUXILIARIES
                    and adj_tok not in _CONJUNCTIONS
                    and not _looks_like_verb_form(words[adj_idx])):
                prep_after_adj = adj_idx + 1
                if prep_after_adj < len(tokens) and tokens[prep_after_adj] in _PREPOSITIONS:
                    # Merge: verb becomes the adjective, prep becomes the preposition
                    verb = adj_tok
                    verb_idx = adj_idx  # shift verb position for object extraction

    # Phase 3: Check for preposition after verb
    prep = None
    obj_start = verb_idx + 1

    if obj_start < len(tokens):
        next_tok = tokens[obj_start]
        if next_tok in _PREPOSITIONS:
            prep = next_tok
            obj_start = verb_idx + 2

    # Phase 4: Extract object (up to clause boundary)
    obj_tokens = []
    for i in range(obj_start, len(tokens)):
        tok = tokens[i]
        # Clause boundary
        if tok in _CONJUNCTIONS and i > obj_start:
            break
        obj_tokens.append(words[i].strip(".,!?;:\"'"))

    obj = " ".join(obj_tokens).strip()
    if not obj or len(obj) < 1:
        return None

    # Phase 5: Detect temporal markers in object
    valid_until, cleaned_obj = _detect_temporal(obj)
    if cleaned_obj and len(cleaned_obj) >= 1:
        obj = cleaned_obj
    # Past tense + duration = already expired (not future expiry)
    # "I lived in Paris for 3 years" → expired, not "expires in 3 years"
    if valid_until and verb.endswith("ed") and valid_until > time.time():
        valid_until = time.time() - 1

    # Build predicate from verb + preposition
    lemma = _lemmatize(verb)
    if prep:
        predicate = _norm(f"{lemma}_{prep}")
    else:
        predicate = _norm(lemma)

    # Normalize copula + "a/an" to "is_a" (identity predicate)
    # "I am a software engineer" → predicate=is_a, object="software engineer"
    if predicate in ("am", "is", "be") and obj_tokens:
        first_obj = obj_tokens[0].lower().strip(".,!?;:\"'")
        if first_obj in ("a", "an"):
            predicate = "is_a"
            obj = " ".join(obj_tokens[1:]).strip(".,!?;:\"'")
            if not obj:
                obj = " ".join(obj_tokens).strip(".,!?;:\"'")

    # Build subject text
    subj_text = " ".join(words[:subj_end]).strip(".,!?;:\"'")

    # --- "My X is Y" → predicate=X when X is a relationship/role noun ---
    _RELATIONSHIP_NOUNS = frozenset({
        "manager", "boss", "supervisor", "director",
        "wife", "husband", "partner", "spouse", "girlfriend", "boyfriend",
        "brother", "sister", "mother", "father", "mom", "dad",
        "son", "daughter", "aunt", "uncle", "cousin",
        "grandmother", "grandfather", "grandma", "grandpa",
        "friend", "coworker", "colleague", "neighbor", "mentor",
        "teacher", "professor", "therapist", "doctor", "lawyer", "dentist",
        "pet", "dog", "cat",
        "name", "birthday", "email", "phone", "age", "salary",
        "favorite",
    })
    _ATTRIBUTE_NOUNS = {"name", "birthday", "job", "role", "age", "email",
                        "phone", "title", "address", "salary"}

    subj_lower_stripped = subj_text.lower().strip()
    if (predicate in ("am", "is", "be", "is_a")
            and subj_lower_stripped.startswith("my ")):
        rest = subj_lower_stripped[3:].strip()
        rest_words = rest.split()
        # Handle possessive form: "My wife's name" → rest_words = ["wife's", "name"]
        # or after _clean_token: subj tokens are ["my", "wifes", "name"]
        # Check if any word in the rest is a relationship noun
        found_rel = None
        found_attr = None
        for rw in rest_words:
            rw_clean = re.sub(r"'s$", "", rw)  # strip possessive
            rw_clean = re.sub(r"s$", "", rw_clean) if rw_clean.endswith("s") and rw_clean not in _RELATIONSHIP_NOUNS else rw_clean
            if rw_clean in _RELATIONSHIP_NOUNS:
                found_rel = rw_clean
            if rw in _ATTRIBUTE_NOUNS or rw_clean in _ATTRIBUTE_NOUNS:
                found_attr = rw
        # "My favorite X is Y" → predicate=favorite_X
        if found_rel == "favorite" and len(rest_words) >= 2:
            # "my favorite book" → predicate="favorite_book"
            fav_obj = rest_words[-1]  # the noun after "favorite"
            fav_obj_clean = re.sub(r"'s$", "", fav_obj)
            if fav_obj_clean in _ATTRIBUTE_NOUNS:
                predicate = _norm(fav_obj_clean)
            else:
                predicate = _norm(f"favorite_{fav_obj_clean}")
        elif found_attr and found_rel and found_attr != found_rel:
            # "My wife's name is Emma" → predicate=name
            predicate = _norm(found_attr)
        elif found_rel:
            # "My manager is Sarah Chen" → predicate=manager
            predicate = _norm(found_rel)

    return {
        "subject": subj_text,
        "verb": verb,
        "preposition": prep,
        "object": obj,
        "is_negation": is_negation,
        "predicate": predicate,
        "raw_verb": f"{verb} {prep}" if prep else verb,
        "valid_until": valid_until,
    }


# ---------------------------------------------------------------------------
#  Extractor protocol
# ---------------------------------------------------------------------------

class Extractor(Protocol):
    def __call__(self, text: str, user_id: str, **kwargs: Any) -> list[Memory]:
        ...


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def extract_personal(text: str, user_id: str, subject: str = "",
                     extractor: Extractor | None = None) -> list[Memory]:
    text = text.strip()
    if not text:
        return []

    subject = subject or user_id
    now = time.time()
    mems: list[Memory] = []
    seen: set[tuple[str, str]] = set()  # (predicate, object) dedup

    # Layer 1: Raw text memory (skip trivial noise)
    if not _is_trivial(text):
        mems.append(Memory(
            scope="private", context="personal", user_id=user_id,
            subject=subject, predicate="stated", object_value=text[:500],
            source_text=text, confidence=0.80, source_type="user_stated",
            created_at=now, updated_at=now, last_accessed=now,
        ))

    # Layer 2: Grammar-based extraction
    for sentence in _split_sentences(text):
        parsed = parse_sentence(sentence)
        if not parsed:
            continue

        subj_lower = parsed["subject"].lower().strip()
        is_first_person = (subj_lower in ("i", "i'm", "im", "my", "", subject.lower())
                           or subj_lower.startswith("my "))

        pred = parsed["predicate"]
        obj = parsed["object"]

        # Check if subject is a third-party reference ("My brother", "My manager")
        _TP_RELATIONSHIP_NOUNS = frozenset({
            "manager", "boss", "supervisor", "director",
            "wife", "husband", "partner", "spouse", "girlfriend", "boyfriend",
            "brother", "sister", "mother", "father", "mom", "dad",
            "son", "daughter", "aunt", "uncle", "cousin",
            "grandmother", "grandfather", "grandma", "grandpa",
            "friend", "coworker", "colleague", "neighbor", "mentor",
            "teacher", "professor", "therapist", "doctor", "lawyer", "dentist",
            "pet", "dog", "cat", "parents", "kids", "children", "family",
        })
        is_third_party_my = False
        if subj_lower.startswith("my "):
            rest_words = subj_lower[3:].split()
            for rw in rest_words:
                rw_clean = re.sub(r"'s?$", "", rw)
                if rw_clean in _TP_RELATIONSHIP_NOUNS:
                    is_third_party_my = True
                    break

        # For implicit subject (empty), use the context subject
        # But keep "My brother" etc. as the subject (third-party reference)
        if is_third_party_my:
            mem_subject = parsed["subject"]
        elif is_first_person or not parsed["subject"]:
            mem_subject = subject
        else:
            mem_subject = parsed["subject"]
        # Third-person facts from nested clauses get lower confidence
        conf = 0.85 if is_first_person else 0.65

        key = (pred, obj.lower())
        if key in seen:
            continue
        seen.add(key)

        mems.append(Memory(
            scope="private", context="personal", user_id=user_id,
            subject=mem_subject, predicate=pred, object_value=obj[:500],
            source_text=sentence[:500], is_negation=parsed["is_negation"],
            source_type="user_stated", confidence=conf,
            valid_until=parsed.get("valid_until"),
            created_at=now, updated_at=now, last_accessed=now,
        ))

    # Layer 3: Optional plugin extractor
    if extractor:
        mems.extend(extractor(text, user_id, subject=subject,
                              scope="private", context="personal"))

    return mems


def extract_chat(text: str, user_id: str, speaker: str = "",
                 session_id: str = "",
                 extractor: Extractor | None = None) -> list[Memory]:
    text = text.strip()
    if not text:
        return []

    speaker = speaker or user_id
    now = time.time()
    meta = {"session_id": session_id, "speaker": speaker}
    mems: list[Memory] = []

    # Raw text memory (skip trivial noise)
    if not _is_trivial(text):
        mems.append(Memory(
            scope="private", context="chat", user_id=user_id,
            subject=speaker, predicate="stated", object_value=text[:500],
            source_text=text, confidence=0.80, source_type="user_stated",
            created_at=now, updated_at=now, last_accessed=now,
            metadata=meta,
        ))

    # Grammar-based extraction — both first and third person
    for sentence in _split_sentences(text):
        parsed = parse_sentence(sentence)
        if not parsed:
            continue

        subj_lower = parsed["subject"].lower().strip()
        if subj_lower in ("i", "i'm", "im", "my") or subj_lower.startswith("my "):
            mem_subject = speaker
        else:
            mem_subject = parsed["subject"]

        mems.append(Memory(
            scope="private", context="chat", user_id=user_id,
            subject=mem_subject, predicate=parsed["predicate"],
            object_value=parsed["object"][:500],
            source_text=sentence[:500], is_negation=parsed["is_negation"],
            source_type="user_stated", confidence=0.75,
            created_at=now, updated_at=now, last_accessed=now,
            metadata=meta,
        ))

    # Decision detection (keyword-based, not dictionary)
    lower = text.lower()
    for marker in ("we decided", "we agreed", "let's go with",
                   "let's use", "the plan is", "the decision is"):
        idx = lower.find(marker)
        if idx == -1:
            continue
        after = text[idx + len(marker):].strip()
        after = re.sub(r'^[\s,]*(to\s+)?', '', after).strip()
        after = re.sub(r'[.!?,;:]+$', '', after).strip()
        if len(after) >= 5:
            mems.append(Memory(
                scope="private", context="chat", user_id=user_id,
                subject=speaker, predicate="decided", object_value=after[:200],
                source_text=text, source_type="user_stated", confidence=0.80,
                created_at=now, updated_at=now, last_accessed=now,
                metadata={**meta, "type": "decision"},
            ))

    if extractor:
        mems.extend(extractor(text, user_id, subject=speaker,
                              scope="private", context="chat"))

    return mems


def extract_company(text: str, user_id: str, org_id: str,
                    subject: str = "",
                    extractor: Extractor | None = None) -> list[Memory]:
    text = text.strip()
    if not text:
        return []

    subj = subject or "team"
    now = time.time()
    meta = {"attributed_to": user_id}
    mems: list[Memory] = []

    # Raw text memory
    mems.append(Memory(
        scope="shared", context="company", user_id=user_id, org_id=org_id,
        subject=subj, predicate="stated", object_value=text[:500],
        source_text=text, confidence=0.80, source_type="user_stated",
        created_at=now, updated_at=now, last_accessed=now,
        metadata=meta,
    ))

    # Grammar extraction — only third person / "we" / "our"
    for sentence in _split_sentences(text):
        parsed = parse_sentence(sentence)
        if not parsed:
            continue

        subj_lower = parsed["subject"].lower().strip()
        # Skip first-person facts — belong in personal scope
        if subj_lower in ("i", "i'm", "im", "my") or subj_lower.startswith("my "):
            continue

        mems.append(Memory(
            scope="shared", context="company", user_id=user_id, org_id=org_id,
            subject=subj, predicate=parsed["predicate"],
            object_value=parsed["object"][:500],
            source_text=sentence[:500], source_type="user_stated", confidence=0.75,
            created_at=now, updated_at=now, last_accessed=now,
            metadata=meta,
        ))

    if extractor:
        mems.extend(extractor(text, user_id, subject=subj,
                              scope="shared", context="company", org_id=org_id))

    return mems
