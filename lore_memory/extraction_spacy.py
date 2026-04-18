"""
spaCy-based sentence parser — Phase 1 replacement for the hand-rolled
grammar in extraction.py.

Why: the grammar parser hard-codes English vocabulary (irregular verbs,
adverbs, possessives, multi-word negations) across ~800 LOC. spaCy's
en_core_web_sm gives us POS tags, lemmas, and dependency arcs for free,
and stays correct on phrasings the grammar hasn't been taught.

Status: opt-in. The default `extract_personal` still uses the grammar
parser. Switch via `Memory(use_spacy=True)`. Phase 2 runs the critique
tier table on both and only deletes the grammar after parity at every tier.

Returns the same dict shape as `extraction.parse_sentence`:
    {subject, predicate, object, is_negation, valid_until, valid_from}
or None if no triple was extracted.
"""

from __future__ import annotations

import re
import time
from typing import Any

# spaCy import is deferred to first use so the module stays cheap to import
# in environments that don't enable the spaCy path.
_NLP: Any = None


_SPACY_MODEL = "en_core_web_sm"


def _get_nlp() -> Any:
    """Load the spaCy English model, downloading it on first use if missing.

    Why auto-download: spaCy's models aren't on PyPI; the canonical install
    is `python -m spacy download en_core_web_sm`. Forcing users to run that
    after `pip install loremem-ai` is friction we don't need — instead we
    fetch the ~12MB model the first time it's required, print one line so
    the user knows what's happening, and cache the loaded pipeline for the
    process. Set LORE_SKIP_MODEL_DOWNLOAD=1 to disable the auto-fetch and
    raise a clear error instead (useful in air-gapped environments where
    the model should be pre-installed via your package manager).
    """
    global _NLP
    if _NLP is not None:
        return _NLP
    import spacy
    try:
        _NLP = spacy.load(_SPACY_MODEL, disable=["ner"])
        return _NLP
    except OSError:
        pass
    import os
    if os.environ.get("LORE_SKIP_MODEL_DOWNLOAD"):
        raise RuntimeError(
            f"spaCy model {_SPACY_MODEL!r} not found and "
            "LORE_SKIP_MODEL_DOWNLOAD is set. Install it with: "
            f"python -m spacy download {_SPACY_MODEL}"
        )
    print(f"[lore-memory] downloading spaCy model {_SPACY_MODEL!r} "
          f"(~12 MB, one-time)...", flush=True)
    from spacy.cli import download as _download
    _download(_SPACY_MODEL)
    _NLP = spacy.load(_SPACY_MODEL, disable=["ner"])
    return _NLP


# Map spaCy dep labels to roles we care about. Kept tiny on purpose —
# anything not listed is ignored by the parser, which is the right default.
_SUBJ_DEPS = frozenset({"nsubj", "nsubjpass"})
_OBJ_DEPS = frozenset({"dobj", "attr", "acomp"})
_MOD_DEPS_FOR_SPAN = frozenset({"det", "amod", "compound", "poss", "nummod"})

_RELATIVE_TEMPORAL = re.compile(
    r"\b(?:last\s+(?:week|month|year|night|monday|tuesday|wednesday|thursday|"
    r"friday|saturday|sunday)|yesterday|(?:a\s+)?(?:few|couple)\s+"
    r"(?:days?|weeks?|months?)\s+ago|recently|the\s+other\s+day)\b",
    re.IGNORECASE,
)

_REL_DELTA_SECONDS = {
    "yesterday": 86400,
    "last week": 7 * 86400,
    "last month": 30 * 86400,
    "last year": 365 * 86400,
    "recently": 7 * 86400,
    "the other day": 3 * 86400,
}


def _resolve_relative_date(phrase: str, now: float) -> float | None:
    """Map a relative-time phrase to an absolute timestamp in the past.
    Returns None when the phrase doesn't resolve to a known offset.

    This is the cheap version: covers the high-frequency cases that
    silently rot in the current store. Phase 2 can add a real chrono
    parser if needed.
    """
    p = phrase.lower().strip()
    if p in _REL_DELTA_SECONDS:
        return now - _REL_DELTA_SECONDS[p]
    if p.startswith("last "):
        return now - 7 * 86400  # default: ~one week back
    m = re.match(r"(?:a\s+)?(few|couple)\s+(days?|weeks?|months?)\s+ago", p)
    if m:
        unit = m.group(2).rstrip("s")
        per = {"day": 86400, "week": 7 * 86400, "month": 30 * 86400}[unit]
        count = 3 if m.group(1) == "few" else 2
        return now - per * count
    return None


def _resolve_named_entity(head: Any) -> Any:
    """If `head` is introduced via an "X named/called Y" clause, return Y.
    Otherwise return `head` unchanged.

    spaCy parses "a dog named Luna" as dog→acl(named)→oprd(Luna). The user
    cares about Luna, not "a dog" — promote the named entity to the
    extracted object so `have:Luna` is what gets stored.
    """
    for child in head.children:
        if child.dep_ == "acl" and child.lemma_ in ("name", "call"):
            for gc in child.children:
                if gc.dep_ in ("oprd", "dobj", "attr") and gc.pos_ in ("PROPN", "NOUN"):
                    return gc
    return head


def _span_text(tok: Any) -> str:
    """Reconstruct the surface span for a head token: include determiners,
    adjectives, compounds, possessives, numeric modifiers — exclude relative
    clauses and prepositional phrases that hang off the head.
    """
    keep = [tok]
    for child in tok.children:
        if child.dep_ in _MOD_DEPS_FOR_SPAN:
            keep.extend(child.subtree)
    keep_sorted = sorted(set(keep), key=lambda t: t.i)
    return " ".join(t.text for t in keep_sorted).strip()


def _extract_one(verb: Any, subj_tok: Any, doc: Any, now: float) -> list[dict]:
    """Extract one or more triples from (verb, subject) — handles compound
    objects ("Python, Go, Rust") by walking object `conj` chains.

    Returns a list of triple dicts (length >= 0).
    """
    subject = _span_text(subj_tok)
    is_negation = any(c.dep_ == "neg" for c in verb.children)
    particle = next((c for c in verb.children if c.dep_ == "prt"), None)

    obj_heads: list[tuple[str, Any]] = []  # (predicate, head_tok)

    # 1. Direct object + conjuncts. Preferred when both exist:
    #    "I joined Spotify in January" → dobj=Spotify (meaning), prep=January
    #    (modifier). Picking prep first would mis-extract January as the
    #    object and lose the works_at link.
    dobj = next((c for c in verb.children if c.dep_ == "dobj"), None)
    if dobj is not None:
        pred = verb.lemma_
        obj_heads.append((pred, _resolve_named_entity(dobj)))
        obj_heads.extend((pred, _resolve_named_entity(c)) for c in dobj.conjuncts)

    # 2. Prepositional object — used when there's no dobj.
    #    "I work at Spotify", "I moved to Amsterdam" → predicate = verb_prep.
    if not obj_heads:
        for child in verb.children:
            if child.dep_ == "prep":
                pobj = next((c for c in child.children if c.dep_ == "pobj"), None)
                if pobj is not None:
                    pred = f"{verb.lemma_}_{child.text.lower()}"
                    obj_heads.append((pred, _resolve_named_entity(pobj)))
                    obj_heads.extend((pred, _resolve_named_entity(c)) for c in pobj.conjuncts)
                    break

    # 3. advmod fallback for capitalized single-word objects.
    #    spaCy frequently mis-tags single-word capitalized objects as ADV
    #    ("I like Java" → Java/ADV/advmod). Accept any noun-like child as
    #    the object when no proper dobj exists. Skip copular verbs — the
    #    copula path below handles them; running this for "is" would catch
    #    sentence-initial adverbs like "Actually" as the object.
    if not obj_heads and verb.lemma_ != "be":
        for child in verb.children:
            if child.dep_ in ("advmod", "npadvmod", "oprd"):
                tok = child.text
                looks_nominal = (child.pos_ in ("NOUN", "PROPN")
                                 or (tok[:1].isupper() and tok.isalpha()
                                     and len(tok) >= 2))
                if looks_nominal:
                    obj_heads.append((verb.lemma_, child))
                    break

    # 4. Copula attribute: "I am a software engineer" — root is the noun.
    copula_pred: str | None = None
    copula_obj: str | None = None
    if not obj_heads:
        has_cop = any(c.dep_ in ("cop", "aux") and c.lemma_ == "be"
                      for c in verb.children)
        if has_cop and verb.pos_ in ("NOUN", "PROPN", "ADJ"):
            copula_obj = _span_text(verb)
            first = next((t for t in verb.subtree if t.dep_ == "det"), None)
            if first is not None and first.text.lower() in ("a", "an"):
                copula_pred = "is_a"
            elif verb.pos_ in ("NOUN", "PROPN"):
                copula_pred = "is_a"
            else:
                copula_pred = "is"

    # 5. attr/acomp under a copular root: "My job is engineer".
    if not obj_heads and copula_obj is None:
        attr = next((c for c in verb.children if c.dep_ in _OBJ_DEPS), None)
        if attr is not None:
            first = next((t for t in attr.subtree if t.dep_ == "det"), None)
            if first is not None and first.text.lower() in ("a", "an"):
                pred = "is_a"
            else:
                pred = verb.lemma_ if verb.lemma_ != "be" else "is"
            obj_heads.append((pred, attr))
            obj_heads.extend((pred, c) for c in attr.conjuncts)

    # 6. Zero-object root verb with particle: "Luna passed away".
    if not obj_heads and copula_obj is None and particle is not None:
        return [_build_triple(subject, f"{verb.lemma_}_{particle.text.lower()}",
                              "", is_negation, doc, now)]

    triples: list[dict] = []
    if copula_obj is not None and copula_pred is not None:
        triples.append(_build_triple(subject, copula_pred, copula_obj,
                                     is_negation, doc, now, particle))
    for pred, head in obj_heads:
        obj_text = _span_text(head)
        if not obj_text:
            continue
        triples.append(_build_triple(subject, pred, obj_text,
                                     is_negation, doc, now, particle))
    return triples


def _build_triple(subject: str, predicate: str, obj_text: str,
                  is_negation: bool, doc: Any, now: float,
                  particle: Any = None) -> dict:
    """Apply post-processing common to every extracted triple."""
    # Particle suffix when not already present.
    if particle is not None and not predicate.endswith(f"_{particle.text.lower()}"):
        predicate = f"{predicate}_{particle.text.lower()}"

    # "My X is Y" → predicate=X, subject=I. These are identity-style facts
    # (favorite_color, current_project, hobby, …) and the user expects a
    # follow-up "actually my X is Y2" to replace, not accumulate. Mark as
    # single-valued so belief.check_contradictions supersedes regardless of
    # whether the predicate happens to appear in the schema's static set.
    single_valued = False
    if (predicate in ("is", "is_a") and subject.lower().startswith("my ")
            and obj_text):
        rest = subject[3:].strip().split()
        if rest:
            head = rest[-1].lower().rstrip("'s").rstrip("s")
            if head and head not in ("i", "me", "we"):
                predicate = head
                subject = "I"
                single_valued = True

    # Strip leading determiner from is_a objects.
    if predicate == "is_a" and obj_text:
        toks = obj_text.split()
        if toks and toks[0].lower() in ("a", "an", "the"):
            obj_text = " ".join(toks[1:])

    # Temporal: detect a relative-time phrase anywhere in the sentence.
    valid_from: float | None = None
    m = _RELATIVE_TEMPORAL.search(doc.text)
    if m:
        valid_from = _resolve_relative_date(m.group(0), now)

    return {
        "subject": subject,
        "predicate": predicate.lower(),
        "object": obj_text,
        "is_negation": is_negation,
        "valid_until": None,
        "valid_from": valid_from,
        "single_valued": single_valued,
    }


_RETRACTION_VERBS = frozenset({"leave", "quit", "resign"})


def extract_triples_from_sentence(sentence: str, now: float | None = None) -> list[dict]:
    """Return ALL triples in a sentence — handles compound clauses
    ("I use Python and love Rust") via root.conjuncts and compound objects
    ("Python, Go, and Rust") via obj.conjuncts.

    Use this from extract_personal/chat/company in the spaCy path.
    """
    if not sentence or not sentence.strip():
        return []
    now = now or time.time()
    doc = _get_nlp()(sentence.strip())
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if root is None:
        return []

    # Each verb in (root + root.conjuncts) is a separate clause.
    # Subject is on the outer root; conjunct verbs share it unless they
    # carry their own nsubj.
    out: list[dict] = []
    outer_subj = next((c for c in root.children if c.dep_ in _SUBJ_DEPS), None)
    if outer_subj is None:
        return []

    verbs: list[Any] = [root] + list(root.conjuncts)
    for v in verbs:
        # xcomp chain: "started learning Rust" — use inner verb for pred/obj.
        effective = v
        for child in v.children:
            if child.dep_ == "xcomp" and child.pos_ == "VERB":
                effective = child
                break
        v_subj = next((c for c in v.children if c.dep_ in _SUBJ_DEPS), outer_subj)
        triples = _extract_one(effective, v_subj, doc, now)
        # First-person "I left/quit/resigned <Org>" → retract works_at.
        # Without this, "I left Google" creates leave:Google but does not
        # tell belief.py to supersede the existing work_at:Google fact.
        if (effective.lemma_ in _RETRACTION_VERBS
                and v_subj.text.lower() in ("i", "we")):
            for t in triples:
                obj = t.get("object", "")
                if obj and obj[0:1].isupper():
                    out.append({**t, "predicate": "works_at",
                                "is_negation": True})
        out.extend(triples)
    return out


def parse_sentence_spacy(sentence: str, now: float | None = None) -> dict | None:
    """Single-triple wrapper around extract_triples_from_sentence.

    Kept for tests and backward compatibility. Returns the first triple
    or None. New code should use extract_triples_from_sentence.
    """
    triples = extract_triples_from_sentence(sentence, now)
    return triples[0] if triples else None


def split_sentences_spacy(text: str) -> list[str]:
    """spaCy sentence splitter. Replaces the regex-based _split_sentences
    when the spaCy path is enabled."""
    if not text or not text.strip():
        return []
    doc = _get_nlp()(text.strip())
    return [s.text.strip() for s in doc.sents if s.text.strip()]
