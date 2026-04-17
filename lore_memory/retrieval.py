"""
Retrieval engine — v6: NRM-grade learning.

Channels:
  1. Semantic: cosine similarity (embedding)
  2. Keyword: BM25-style term overlap
  3. Temporal: exponential recency decay
  4. Belief: Bayesian posterior
  5. Frequency: log-scaled access count
  6. Graph: spreading activation with fan-effect (3-hop)
  7. Resonance: co-activation frequency from traces

Learning mechanisms:
  - Adaptive channel weights (EMA from feedback)
  - Hebbian synapse strengthening (co-activated facts)
  - Activation traces (which facts appear together)
  - Lateral inhibition (suppress redundant results)
"""

from __future__ import annotations

import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field

from lore_memory.store import Memory, MemoryDB, cosine_sim


@dataclass
class SearchResult:
    memory: Memory
    score: float
    channel_scores: dict[str, float] = field(default_factory=dict)
    source_scope: str = ""


@dataclass
class Weights:
    semantic: float = 0.30
    keyword: float = 0.15
    temporal: float = 0.10
    belief: float = 0.15
    frequency: float = 0.10
    graph: float = 0.10
    resonance: float = 0.10
    _alpha: float = 0.1
    _updates: int = 0

    def update(self, scores: dict[str, float]) -> None:
        total = sum(scores.values())
        if total <= 0:
            return
        a = self._alpha
        for ch in ("semantic", "keyword", "temporal", "belief",
                    "frequency", "graph", "resonance"):
            if ch in scores:
                old = getattr(self, ch)
                setattr(self, ch, (1 - a) * old + a * (scores[ch] / total))
        self._normalize()
        self._updates += 1

    def _normalize(self) -> None:
        channels = ["semantic", "keyword", "temporal", "belief",
                    "frequency", "graph", "resonance"]
        vals = [getattr(self, ch) for ch in channels]
        total = sum(vals)
        if total <= 0:
            self.semantic, self.keyword, self.temporal = 0.30, 0.15, 0.10
            self.belief, self.frequency, self.graph, self.resonance = 0.15, 0.10, 0.10, 0.10
            return
        floor = 0.02
        vals = [max(floor, v / total) for v in vals]
        total = sum(vals)
        normed = [v / total for v in vals]
        for ch, v in zip(channels, normed):
            setattr(self, ch, v)

    def to_dict(self) -> dict:
        return {
            "semantic": round(self.semantic, 4), "keyword": round(self.keyword, 4),
            "temporal": round(self.temporal, 4), "belief": round(self.belief, 4),
            "frequency": round(self.frequency, 4), "graph": round(self.graph, 4),
            "resonance": round(self.resonance, 4), "updates": self._updates,
        }


from lore_memory.lexicons import (
    FTS_STOPWORDS as _STOPS,
    QUERY_CONTENT_STOPWORDS,
    QUERY_INTENT_STOPWORDS,
    RETRIEVAL_RELATIONSHIP_NOUNS,
)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in re.split(r"[\s\-_/\\.,;:!?'\"()\[\]{}]+", text)
            if t and len(t) > 1 and t.lower() not in _STOPS]


# Predicates that are ultra-generic and match almost any query semantically
_GENERIC_PREDICATES = frozenset({
    "is_a", "name", "am", "is", "stated", "be",
})

# Query synonym expansion: broaden FTS recall for near-synonyms
_QUERY_SYNONYMS = {
    "fiancee": ["girlfriend", "boyfriend", "partner", "engaged", "fiance"],
    "fiance": ["girlfriend", "boyfriend", "partner", "engaged", "fiancee"],
    "wife": ["married", "spouse", "partner", "husband"],
    "husband": ["married", "spouse", "partner", "wife"],
    "car": ["drive", "vehicle", "motorcycle", "bmw", "tesla"],
    "vehicle": ["drive", "car", "motorcycle"],
    "phone": ["iphone", "pixel", "samsung", "mobile"],
    "laptop": ["macbook", "thinkpad", "computer"],
    "drink": ["coffee", "tea", "matcha", "alcohol", "water"],
    "salary": ["pay", "compensation", "income", "earn"],
    "left": ["leaving", "departed", "quit", "resigned"],
    "departed": ["left", "leaving", "quit"],
}


class Retriever:
    def __init__(self, embed_fn) -> None:
        self.embed = embed_fn
        self._weights: dict[str, Weights] = {}
        self._graph_cache = None

        # Activation traces: which memory IDs co-appear in results
        # Key: memory_id, Value: set of memory_ids it co-appeared with
        self._traces: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._trace_count = 0
        self._MAX_TRACES = 50000  # per-entry cap

        # Hebbian synapse strengths (co-activation reinforcement)
        # Key: (mem_id_a, mem_id_b) → strength
        self._synapses: dict[tuple[str, str], float] = {}
        self._MAX_SYNAPSES = 100000

        # Predicate embedding cache: predicate_str → embedding vector
        # Predicates are embedded as natural language phrases for semantic matching
        self._pred_embeddings: dict[str, list[float]] = {}

        # Query intent embeddings: precomputed prototype vectors for common intents
        # Lazily initialized on first query
        self._intent_protos: dict[str, list[float]] = {}
        self._intents_initialized = False

    def set_graph_cache(self, graph_cache) -> None:
        self._graph_cache = graph_cache

    _MAX_WEIGHT_ENTRIES = 10000

    def get_weights(self, key: str) -> Weights:
        if key not in self._weights:
            if len(self._weights) >= self._MAX_WEIGHT_ENTRIES:
                oldest = next(iter(self._weights))
                del self._weights[oldest]
            self._weights[key] = Weights()
        return self._weights[key]

    def search(
        self,
        query: str,
        dbs: list[tuple[MemoryDB, str]],
        context: str | None = None,
        top_k: int = 20,
        weight_key: str = "default",
        user_id: str = "",
        predicate_hint: str | list[str] | None = None,
        subject_hint: str | None = None,
    ) -> list[SearchResult]:
        # Normalize hints into canonical sets the scorer can match against.
        # Boost — not filter — so a wrong hint never hides a correct answer.
        from lore_memory.belief import canon as _canon
        _pred_hint_canon: set[str] = set()
        if predicate_hint:
            preds = [predicate_hint] if isinstance(predicate_hint, str) else list(predicate_hint)
            for p in preds:
                if not p:
                    continue
                p_norm = p.lower().strip().replace(" ", "_")
                _pred_hint_canon.add(p_norm)
                _pred_hint_canon.add(_canon(p_norm))
        _subj_hint = (subject_hint or "").strip()
        if _subj_hint.lower() == "user" and user_id:
            _subj_hint = user_id

        now = time.time()
        q_emb = self.embed(query)
        q_terms = tokenize(query)

        # First-person intent: when the query is about the *user*, the
        # user's own active facts must always be in the candidate pool
        # regardless of FTS/vector shortlist — otherwise, as the corpus
        # grows, a user's own live_in / works_at / likes fact gets
        # drowned by third-party facts that share the same predicate.
        _FIRST_PERSON_Q = {"i", "my", "me", "mine", "myself"}
        _ql_tokens = set(re.findall(r"\b\w+\b", query.lower()))
        _is_first_person_q = bool(_ql_tokens & _FIRST_PERSON_Q)

        # Expand query terms with synonyms for broader FTS recall
        expanded_terms = list(q_terms)
        for term in q_terms:
            if term in _QUERY_SYNONYMS:
                expanded_terms.extend(_QUERY_SYNONYMS[term])

        w = self.get_weights(weight_key)
        results: list[SearchResult] = []

        # Detect temporal intent: semantic prototype + keyword fallback.
        # "old" is deliberately NOT in the bare keyword set because it
        # mis-fires on "how old am I?" (age question, not temporal). We
        # require a possessive/article before "old" to count it temporal
        # ("my old manager", "the old system") — same for "former" only
        # as a standalone word (not substring of "performer").
        self._ensure_intents()
        q_lower = query.lower()
        _temp_sim = cosine_sim(q_emb, self._intent_protos["temporal"])
        _TEMPORAL_KEYWORDS = {"before", "previous", "previously", "formerly",
                              "used to", "earlier", "prior", "ago", "past",
                              "former", "history"}
        _TEMPORAL_OLD_RE = re.compile(
            r"\b(?:my|the|our|their|his|her|a|an)\s+old\b")
        is_temporal_query = (_temp_sim > 0.25
                             or any(kw in q_lower for kw in _TEMPORAL_KEYWORDS)
                             or bool(_TEMPORAL_OLD_RE.search(q_lower)))

        # Scope alignment: boost shared scope for company/team queries
        _COMPANY_KEYWORDS = {"company", "team", "org", "organization", "startup",
                             "we", "our", "engineers", "employees", "cto",
                             "mission", "deploy", "infrastructure", "stack",
                             "process", "methodology", "market", "product",
                             "payflow", "revenue", "volume"}
        is_company_query = any(kw in q_lower for kw in _COMPANY_KEYWORDS)

        for db, scope_label in dbs:
            seen_ids: set[str] = set()
            cands: list[Memory] = []

            # Primary: FTS keyword search (O(log n))
            if q_terms:
                safe_terms = [t.replace('"', '').replace("'", "").replace("*", "")
                              .replace(":", "").replace("(", "").replace(")", "")
                              for t in expanded_terms if t.replace('"', '').strip()]
                # Use prefix matching (term*) so "auth" matches "authentication"
                fts_parts = [f"{t}*" for t in safe_terms if len(t) >= 2]
                fts_query = " OR ".join(fts_parts) if fts_parts else ""
                try:
                    # For temporal queries, search all states (active + superseded)
                    if is_temporal_query and fts_query:
                        fts_hits = db.fts_search_all_states(fts_query, limit=top_k * 5)
                    else:
                        fts_hits = db.fts_search(fts_query, limit=top_k * 5) if fts_query else []
                except Exception:
                    fts_hits = []
                for m in fts_hits:
                    if m.is_active or (is_temporal_query and m.state == "superseded"):
                        cands.append(m)
                        seen_ids.add(m.id)

            # Secondary: vector similarity search (finds semantic matches FTS
            # misses). Depth is bounded above top_k so we always consider
            # predicate-alignment-only matches (e.g. "I joined Anthropic"
            # for "where do I work?") whose text lacks the query keywords.
            # top_k * 3 clamped to 25 balances recall (big enough to catch
            # those) against noise (not so big that unrelated facts outrank
            # the right answer via diversity penalties).
            vec_hits = db.vector_search(q_emb,
                                         top_k=max(min(top_k * 3, 25), top_k),
                                         context=context)
            for m, sim in vec_hits:
                if m.id not in seen_ids and m.is_active:
                    cands.append(m)
                    seen_ids.add(m.id)

            # For temporal queries, also pull superseded facts
            if is_temporal_query:
                superseded = db.query_superseded(limit=top_k)
                for m in superseded:
                    if m.id not in seen_ids:
                        cands.append(m)
                        seen_ids.add(m.id)

            # Tertiary: small recency sample if still short
            if len(cands) < top_k:
                recent = db.query_active(context=context, limit=top_k,
                                         skip_embedding=True)
                for m in recent:
                    if m.id not in seen_ids:
                        cands.append(m)
                        seen_ids.add(m.id)

            # First-person augmentation: if the query is phrased as
            # first-person ("where do I live?"), pull in the user's own
            # active facts directly. This guarantees the user's own
            # identity facts are always considered, no matter how many
            # third-party facts with the same predicate have accumulated.
            if _is_first_person_q and user_id:
                try:
                    own = db.query_by_subject(user_id, context=context, limit=50)
                    for m in own:
                        if m.id not in seen_ids and m.is_active:
                            cands.append(m)
                            seen_ids.add(m.id)
                except Exception:
                    pass

            if not cands:
                continue

            # --- Score all 7 channels ---
            sem = {m.id: max(0.0, cosine_sim(q_emb, m.embedding))
                   for m in cands if m.embedding}
            kw = self._kw_scores(q_terms, cands)
            tmp = self._temporal(cands, now)
            bel = {m.id: m.posterior for m in cands}
            freq = self._freq(cands)
            graph = self._spreading_activation(q_terms, cands)
            res = self._resonance(cands)

            channels = [
                (sem, w.semantic), (kw, w.keyword), (tmp, w.temporal),
                (bel, w.belief), (freq, w.frequency),
                (graph, w.graph), (res, w.resonance),
            ]
            names = ["semantic", "keyword", "temporal", "belief",
                     "frequency", "graph", "resonance"]

            # Precompute ranks per channel
            rank_maps: list[dict[str, int]] = []
            for scores, _ in channels:
                ranked_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
                rank_maps.append({sid: j for j, sid in enumerate(ranked_ids)})

            # Predicate alignment: detect query intent and match predicates
            pred_boost = self._predicate_alignment(query, cands)

            # Negation alignment: boost negation facts for negation queries
            neg_boost = self._negation_alignment(query, cands)

            # Temporal alignment: boost older/retracted facts for temporal queries
            temporal_boost = self._temporal_query_alignment(query, cands)

            # Broad query detection: boost high-evidence facts
            q_lower = query.lower()
            broad_query_markers = {"tell me about", "what do you know", "what about", "describe", "summarize"}
            is_broad = any(marker in q_lower for marker in broad_query_markers)

            # Check if query has specific intent (from pred map or content words)
            has_specific_intent = bool(pred_boost) or any(
                w not in QUERY_INTENT_STOPWORDS and len(w) > 2
                for w in q_lower.split())

            # Entity boost: hard boost for memories containing query entities
            entity_scores = self._entity_boost(query, cands)

            # Scoring: RRF base (k=20) + alignment boosts
            for m in cands:
                # RRF base score from all 7 channels
                rrf = 0.0
                ch: dict[str, float] = {}
                for i, (scores, weight) in enumerate(channels):
                    raw = scores.get(m.id, 0.0)
                    ch[names[i]] = raw
                    rank = rank_maps[i].get(m.id, len(scores))
                    rrf += weight * (1.0 / (20 + rank + 1))

                # Additive semantic score — break ties when RRF is close
                sem_raw = sem.get(m.id, 0.0)
                rrf += 0.05 * sem_raw

                # Predicate alignment: strong multiplicative boost
                pa = pred_boost.get(m.id, 0.0)
                if pa > 0:
                    rrf *= (1.0 + pa)

                # Negation alignment: boost/penalize based on negation match
                na = neg_boost.get(m.id, 0.0)
                if na > 0:
                    rrf *= (1.0 + na)
                elif na < 0:
                    rrf *= max(0.1, 1.0 + na)  # penalize but don't zero out

                # Temporal alignment: boost older/retracted facts
                ta = temporal_boost.get(m.id, 0.0)
                if ta > 0:
                    rrf *= (1.0 + ta)

                # For broad queries, boost high-evidence facts
                if is_broad:
                    rrf *= (1.0 + 0.3 * min(m.evidence_count, 5) / 5)

                # Structured SPO bonus: prefer extracted triples over raw text
                if m.predicate == "stated":
                    rrf *= 0.75  # raw text is fallback, not primary
                else:
                    rrf *= 1.3  # structured facts are preferred

                # Superseded facts get a penalty unless temporal query
                if m.state == "superseded":
                    if is_temporal_query:
                        rrf *= 1.5  # boost superseded for temporal queries
                    else:
                        rrf *= 0.1  # heavily penalize for non-temporal

                # Hypothetical / attributed facts are recall-accessible
                # (the user may search for "what did my wife say about
                # Tokyo?") but must not dominate factual answers.
                md = m.metadata or {}
                if md.get("hypothetical"):
                    rrf *= 0.4
                if md.get("source_speaker"):
                    rrf *= 0.6

                # Subject alignment: boost relationship matches, penalize mismatches
                sa = self._subject_alignment(query, m)
                if sa > 0:
                    rrf *= (1.0 + sa)
                elif sa < 0:
                    rrf *= max(0.1, 1.0 + sa)

                # Generic predicate penalty: penalize identity facts when query is specific
                if m.predicate in _GENERIC_PREDICATES and has_specific_intent:
                    # Query has specific intent but this fact is generic
                    rrf *= 0.5

                # Entity boost: boost memories containing query entities
                eb = entity_scores.get(m.id, 0.0)
                if eb > 0:
                    rrf *= (1.0 + eb)

                # Scope alignment: boost shared scope for company/team queries
                if is_company_query and "shared" in scope_label:
                    rrf *= 1.8

                # LLM-supplied hints (boost, not filter — wrong hints don't
                # hide answers because the underlying multi-channel score is
                # still computed). When an LLM client knows the user is
                # asking about a specific predicate/subject ("what is my
                # job?" → predicate_hint="job_title"), this lets the
                # right candidate jump to the top without us having to
                # guess via embedding similarity.
                if _pred_hint_canon:
                    if (m.predicate in _pred_hint_canon
                            or _canon(m.predicate) in _pred_hint_canon):
                        rrf *= 3.0
                if _subj_hint and m.subject == _subj_hint:
                    rrf *= 2.0

                if rrf > 0:
                    results.append(SearchResult(
                        memory=m, score=rrf, channel_scores=ch,
                        source_scope=scope_label))

        results.sort(key=lambda r: r.score, reverse=True)

        # --- Lateral inhibition: suppress same subject:predicate duplicates ---
        # Only for single-valued predicates. Multi-valued (like, uses) keep all.
        from lore_memory.extraction import SINGLE_VALUED
        seen_sp: dict[tuple[str, str], SearchResult] = {}
        seen_ids_set: set[str] = set()
        pred_counts: dict[str, int] = {}  # predicate diversity tracking
        obj_counts: dict[str, int] = {}   # object diversity tracking
        final: list[SearchResult] = []
        for r in results:
            if r.memory.id in seen_ids_set:
                continue
            seen_ids_set.add(r.memory.id)
            pred = r.memory.predicate
            if pred != "stated" and pred in SINGLE_VALUED:
                sp_key = (r.memory.subject.lower(), pred)
                if sp_key in seen_sp:
                    continue  # suppress — single-valued duplicate
                seen_sp[sp_key] = r
            # Predicate diversity: penalize 3rd+ result with the same predicate
            count = pred_counts.get(pred, 0)
            if count >= 2:
                r = SearchResult(
                    memory=r.memory,
                    score=r.score * 0.5,
                    channel_scores=r.channel_scores,
                    source_scope=r.source_scope,
                )
            pred_counts[pred] = count + 1
            # Object diversity: penalize 3rd+ result with the same object_value
            obj_key = r.memory.object_value.lower().strip()[:50]
            obj_count = obj_counts.get(obj_key, 0)
            if obj_count >= 2:
                r = SearchResult(
                    memory=r.memory,
                    score=r.score * 0.3,
                    channel_scores=r.channel_scores,
                    source_scope=r.source_scope,
                )
            obj_counts[obj_key] = obj_count + 1
            final.append(r)
            if len(final) >= top_k:
                break
        # Re-sort after diversity penalties to ensure best results float up
        final.sort(key=lambda r: r.score, reverse=True)

        # --- Record activation trace (Hebbian learning) ---
        if len(final) >= 2:
            self._record_trace(final)

        return final

    def _record_trace(self, results: list[SearchResult]) -> None:
        """Record co-activation: facts that appeared together in results.
        This feeds the resonance channel and Hebbian synapse strengthening."""
        ids = [r.memory.id for r in results[:10]]  # top 10 only
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                # Resonance trace: symmetric co-occurrence count
                self._traces[a][b] += 1.0
                self._traces[b][a] += 1.0

                # Hebbian synapse: strengthen co-activated pairs
                key = (min(a, b), max(a, b))
                old = self._synapses.get(key, 0.0)
                # Hebbian delta: pre * post activation (both scored > 0)
                delta = 0.05
                self._synapses[key] = min(1.0, old + delta)

        self._trace_count += 1
        # Evict old synapses if too many
        if len(self._synapses) > self._MAX_SYNAPSES:
            # Remove weakest 10%
            cutoff = sorted(self._synapses.values())[len(self._synapses) // 10]
            self._synapses = {k: v for k, v in self._synapses.items() if v > cutoff}

    def learn_from_feedback(self, result: SearchResult, weight_key: str) -> None:
        """Update adaptive weights from explicit user feedback."""
        if result.channel_scores:
            self.get_weights(weight_key).update(result.channel_scores)

    # --- Subject & Predicate alignment ---

    def _subject_alignment(self, query: str, mem: Memory) -> float:
        """Score how well the memory relates to what the query is asking about.

        Three-phase approach:
        0. First-person alignment: when the query is phrased in first person
           (I / my / me) and doesn't name a specific relationship, additive
           boost for memories whose subject is the user's id (their own facts).
           Additive — does not penalise third-person facts, only lifts the
           user's own into contention above a growing noise floor.
        1. Relationship entity detection: when query mentions a relationship
           (sister, wife, manager), boost memories whose source text or
           subject mentions that relationship.
        2. Content keyword overlap: boost when query content words appear
           in the memory's combined text.
        """
        ql = query.lower()
        q_tokens = set(re.findall(r"\b\w+\b", ql))

        _RELATIONSHIPS = RETRIEVAL_RELATIONSHIP_NOUNS
        _FIRST_PERSON_Q = {"i", "my", "me", "mine", "myself"}
        has_first_person_q = bool(q_tokens & _FIRST_PERSON_Q)
        has_relationship_q = any(rel in ql for rel in _RELATIONSHIPS)

        # Phase 0: the user's own fact on a first-person query, when no
        # specific relationship is named. Matching-user check is structural:
        # first-person facts are stored with subject == user_id.
        if (has_first_person_q and not has_relationship_q
                and mem.user_id and mem.subject == mem.user_id):
            return 1.5

        # Phase 1: relationship entity detection via source text matching
        # This handles "fiancee"↔"girlfriend", "sister"↔"Maya" etc.
        # by checking if any query content word appears in the source text
        for rel in _RELATIONSHIPS:
            if rel in ql:
                combined = f"{mem.subject} {mem.object_value} {mem.source_text}".lower()
                # Check for the relationship word OR synonyms in the memory
                if rel in combined:
                    return 1.5
                # Also check related words (fiancee↔engaged, wife↔married)
                _SYNONYMS = {
                    "fiancee": ["girlfriend", "engaged", "fiancé"],
                    "fiance": ["boyfriend", "engaged", "fiancé"],
                    "wife": ["married", "spouse", "partner"],
                    "husband": ["married", "spouse", "partner"],
                    "mom": ["mother"], "dad": ["father"],
                }
                for syn in _SYNONYMS.get(rel, []):
                    if syn in combined:
                        return 1.2
                # Don't penalize — just don't boost

        # Phase 2: content keyword overlap (generic boost)
        q_content = set(w.strip("'?.,!") for w in ql.split()) - QUERY_CONTENT_STOPWORDS
        q_content = {w for w in q_content if len(w) > 2}

        if not q_content:
            return 0.0

        combined = f"{mem.subject} {mem.object_value} {mem.source_text}".lower()
        hits = sum(1 for w in q_content if w in combined)
        if hits >= 2:
            return 0.6
        return 0.0

    # --- Hybrid predicate alignment: curated map + semantic fallback ---

    # Curated map for common query patterns (fast, precise)
    _QUERY_PRED_MAP: dict[str, set[str]] = {
        "work": {"work_at", "works_at", "work_for", "employed_at", "be_at", "start_at"},
        "job": {"work_at", "works_at", "work_for", "be_at", "role", "is_a"},
        "live": {"live_in", "lives_in", "move_to", "base_in", "reside_in"},
        "located": {"live_in", "lives_in", "base_in"},
        "use": {"use", "prefer", "run_on"},
        "language": {"use", "speak", "know", "code_in", "write",
                     "learn", "learning", "study"},
        "speak": {"speak", "know"},
        "learning": {"learn", "learning", "study", "studying"},
        "like": {"like", "love", "enjoy", "prefer", "fan_of"},
        "read": {"read", "reading"},
        "study": {"study", "graduate_from", "major_in", "attend"},
        "school": {"graduate_from", "study_at", "attend", "major_in"},
        "university": {"graduate_from", "study_at", "attend"},
        "pet": {"have", "own"},
        "dog": {"have", "own"},
        "cat": {"have", "own"},
        "database": {"use", "is", "run_on", "switch_from"},
        "team": {"have", "is", "manage", "build", "lead", "hire", "to", "grow"},
        "build": {"build", "work_on", "develop"},
        "deploy": {"deploy_to", "use", "host_on"},
        "coffee": {"drink", "prefer", "like", "switch_from"},
        "tea": {"drink", "prefer", "like", "switch_from"},
        "learn": {"think_about", "want_to", "plan_to", "interested_in", "study"},
        "planning": {"think_about", "want_to", "plan_to", "work_on"},
        "reading": {"am", "read"},
        "allergic": {"allergic_to"},
        "birthday": {"is", "birthday", "born_on"},
        "name": {"is", "name", "call"},
        "children": {"have", "own"},
        "kids": {"have", "own"},
        "cloud": {"use", "deploy_to", "host_on", "run_on"},
        "drink": {"drink", "switch_from", "prefer"},
        "hobby": {"love", "like", "enjoy", "into", "start", "do"},
        "hobbies": {"love", "like", "enjoy", "into", "start", "do"},
        "market": {"launch_in", "expand", "enter"},
        "hire": {"hire", "recruit"},
        "size": {"to", "have", "is", "grow"},
        "volume": {"hit", "handle", "process", "reach"},
        "process": {"follow", "use", "do"},
        "methodology": {"follow", "use", "do"},
    }

    def _get_pred_embedding(self, predicate: str) -> list[float]:
        """Get or compute the embedding for a predicate."""
        if predicate in self._pred_embeddings:
            return self._pred_embeddings[predicate]
        natural = predicate.replace("_", " ")
        emb = self.embed(natural)
        self._pred_embeddings[predicate] = emb
        if len(self._pred_embeddings) > 5000:
            oldest = next(iter(self._pred_embeddings))
            del self._pred_embeddings[oldest]
        return emb

    def _predicate_alignment(self, query: str, cands: list[Memory]) -> dict[str, float]:
        """Score memories whose predicate aligns with the query intent.

        Semantic-primary hybrid:
          1. Semantic cosine between the query embedding and each candidate's
             predicate embedding is computed for *every* candidate. This is
             the base signal and generalises to any English vocabulary.
          2. The curated `_QUERY_PRED_MAP` acts as an opt-in precision
             override for canned query patterns (e.g. "live" → {live_in,
             lives_in, ...}). When it fires, it promotes the match to 1.0.
             It is no longer a gate — out-of-map queries still get scored.
          3. Predicate-token overlap and object/source token overlap are
             cheap tie-breakers, applied as max() with the semantic base.
        """
        q_lower = query.lower()

        # Optional curated map (precision override)
        target_preds: set[str] = set()
        for keyword, preds in self._QUERY_PRED_MAP.items():
            if keyword in q_lower:
                target_preds.update(preds)

        q_content = set(w.strip("'?.,!") for w in q_lower.split()) - QUERY_CONTENT_STOPWORDS
        q_content = {w for w in q_content if len(w) > 2}

        # Query embedding computed once for the whole candidate set
        q_emb = self.embed(query)

        scores: dict[str, float] = {}
        for m in cands:
            if m.predicate == "stated":
                continue

            # Base: semantic predicate alignment, always.
            # Cosine threshold 0.3; mapped linearly to [0, 1.0] up to 0.8.
            pred_emb = self._get_pred_embedding(m.predicate)
            sim = cosine_sim(q_emb, pred_emb)
            if sim >= 0.3:
                base = min(1.0, (sim - 0.3) / 0.5)
            else:
                base = 0.0

            score = base

            # Curated map: precision override — when the user's query uses
            # a keyword we have a canned mapping for, elevate matches of the
            # mapped predicates to the top tier. Match on either the raw
            # predicate or its canonical form so "join" / "joined" etc.
            # (which canonicalize to works_at) fire on a "work" query.
            if target_preds:
                from lore_memory.belief import canon as _canon
                if m.predicate in target_preds or _canon(m.predicate) in target_preds:
                    score = max(score, 1.0)

            # Predicate-token overlap (e.g. query "manager" ∩ pred "manager")
            pred_words = set(m.predicate.replace("_", " ").split())
            pred_overlap = pred_words & q_content
            if pred_overlap:
                score = max(score, min(1.0, len(pred_overlap) * 0.5))

            # Object/source-text overlap (weak signal, last resort)
            if q_content and score < 0.4:
                combined = f"{m.object_value} {m.source_text}".lower()
                obj_hits = sum(1 for w in q_content if w in combined)
                if obj_hits >= 2:
                    score = max(score, 0.3)

            if score > 0:
                scores[m.id] = score

        return scores

    def _negation_alignment(self, query: str, cands: list[Memory]) -> dict[str, float]:
        """Boost negation facts when query asks about negation.

        Uses semantic similarity to a precomputed negation prototype (1 cosine
        op per query, not per candidate). Falls back to keyword detection.
        """
        self._ensure_intents()
        q_emb = self.embed(query)
        neg_sim = cosine_sim(q_emb, self._intent_protos["negation"])

        if neg_sim < 0.35:
            return {}

        strength = min(2.0, (neg_sim - 0.25) * 4)
        scores: dict[str, float] = {}
        for m in cands:
            if m.is_negation:
                scores[m.id] = strength
        return scores

    def _temporal_query_alignment(self, query: str, cands: list[Memory]) -> dict[str, float]:
        """Boost retracted/older facts when query asks about the past.

        Uses semantic similarity + keyword fallback to detect temporal intent.
        """
        self._ensure_intents()
        q_emb = self.embed(query)
        q_lower = query.lower()
        temp_sim = cosine_sim(q_emb, self._intent_protos["temporal"])
        _TEMPORAL_KW = {"before", "previous", "previously", "formerly",
                        "used to", "earlier", "prior", "ago", "past"}
        has_kw = any(kw in q_lower for kw in _TEMPORAL_KW)

        if temp_sim < 0.2 and not has_kw:
            return {}

        strength = max(1.5, min(2.5, (temp_sim - 0.1) * 5)) if has_kw else max(0.5, min(2.5, (temp_sim - 0.15) * 5))
        _CURRENT_STATE_PREDS = frozenset({
            "live_in", "lives_in", "work_at", "works_at", "move_to",
            "employed_at", "reside_in", "base_in",
        })
        scores: dict[str, float] = {}
        for m in cands:
            if m.is_negation or m.state == "superseded":
                scores[m.id] = strength  # boost past/superseded facts
            elif m.state == "active" and m.predicate in _CURRENT_STATE_PREDS:
                scores[m.id] = scores.get(m.id, 0) - 0.6  # penalize current state
        return scores

    def _ensure_intents(self) -> None:
        """Lazily initialize intent prototype embeddings (3 embeds total, once)."""
        if self._intents_initialized:
            return
        self._intent_protos["negation"] = self.embed(
            "what doesn't the person like, dislike, hate, avoid, can't, not")
        self._intent_protos["temporal"] = self.embed(
            "what did the person do before, previously, in the past, "
            "former, used to, earlier, prior history")
        self._intents_initialized = True

    # --- Entity boost ---

    def _entity_boost(self, query: str, cands: list[Memory]) -> dict[str, float]:
        """Hard boost for memories containing entities mentioned in the query."""
        # Extract entities: capitalized words, quoted strings, specific terms
        entities = set()
        for word in query.split():
            clean = word.strip("?.,!\"'")
            if clean and (clean[0].isupper() or len(clean) > 3):
                entities.add(clean.lower())

        if not entities:
            return {}

        scores: dict[str, float] = {}
        for m in cands:
            combined = f"{m.object_value} {m.source_text}".lower()
            hits = sum(1 for ent in entities if ent in combined)
            if hits > 0:
                scores[m.id] = 0.8 * (hits / len(entities))
        return scores

    # --- Channel scoring functions ---

    def _kw_scores(self, terms: list[str], cands: list[Memory]) -> dict[str, float]:
        if not terms:
            return {}
        scores: dict[str, float] = {}
        for m in cands:
            # Score source_text (raw) and structured fields separately
            src_doc = tokenize(m.source_text)
            obj_doc = tokenize(m.object_value)
            pred_doc = tokenize(m.predicate.replace("_", " "))
            all_doc = set(src_doc + obj_doc + pred_doc)
            if not all_doc:
                continue
            hits = sum(1 for qt in terms if any(
                d.startswith(qt) or qt.startswith(d) for d in all_doc))
            if hits > 0:
                score = hits / len(terms)
                # Bonus: if query terms appear in the object_value specifically
                # (more targeted than appearing in source_text)
                obj_hits = sum(1 for qt in terms if any(
                    d.startswith(qt) or qt.startswith(d) for d in obj_doc))
                if obj_hits > 0:
                    score += 0.3 * (obj_hits / len(terms))
                # Exact word match in object_value gets a strong bonus
                exact_obj_hits = sum(1 for qt in terms if qt in obj_doc)
                if exact_obj_hits > 0:
                    score += 0.5 * (exact_obj_hits / len(terms))
                scores[m.id] = min(1.0, score)
        return scores

    def _temporal(self, cands: list[Memory], now: float) -> dict[str, float]:
        hl = 30 * 86400
        out: dict[str, float] = {}
        for m in cands:
            age = now - m.updated_at
            rec = math.exp(-0.693 * age / hl)
            if m.valid_until is not None and m.valid_until < now:
                out[m.id] = 0.0
            else:
                out[m.id] = min(1.0, rec)
        return out

    def _freq(self, cands: list[Memory]) -> dict[str, float]:
        mx = max((m.access_count for m in cands), default=1) or 1
        return {m.id: math.log2(1 + m.access_count) / math.log2(1 + mx)
                for m in cands}

    def _spreading_activation(self, terms: list[str],
                              cands: list[Memory]) -> dict[str, float]:
        """Spreading activation with fan-effect dilution (from NRM).
        3-hop traversal with decay per hop."""
        if not self._graph_cache or not terms:
            return {}

        # Activation map: node → activation level
        activation: dict[str, float] = {}
        decay = 0.5  # activation decay per hop

        for t in terms:
            tl = t.lower()
            # Seed activation
            activation[tl] = 1.0

            # 1-hop
            try:
                fwd = self._graph_cache._forward
                rev = self._graph_cache._reverse
            except AttributeError:
                # Fallback for non-GraphCache objects
                try:
                    related = self._graph_cache.get_related(t)
                    return {m.id: 0.8 for m in cands
                            if m.subject.lower() in related
                            or m.object_value.lower() in related}
                except Exception:
                    return {}

            hop1 = set()
            for neighbor in fwd.get(tl, set()):
                fan = 1.0 / math.log2(len(fwd.get(tl, set())) + 2)  # fan-effect
                act = decay * fan
                activation[neighbor] = max(activation.get(neighbor, 0), act)
                hop1.add(neighbor)
            for neighbor in rev.get(tl, set()):
                fan = 1.0 / math.log2(len(rev.get(tl, set())) + 2)
                act = decay * fan
                activation[neighbor] = max(activation.get(neighbor, 0), act)
                hop1.add(neighbor)

            # 2-hop
            hop2 = set()
            for n1 in hop1:
                for n2 in fwd.get(n1, set()):
                    if n2 == tl:
                        continue
                    fan = 1.0 / math.log2(len(fwd.get(n1, set())) + 2)
                    act = activation.get(n1, 0) * decay * fan
                    if act > 0.01:  # early stopping
                        activation[n2] = max(activation.get(n2, 0), act)
                        hop2.add(n2)

            # 3-hop (light)
            for n2 in hop2:
                for n3 in fwd.get(n2, set()):
                    if n3 == tl:
                        continue
                    fan = 1.0 / math.log2(len(fwd.get(n2, set())) + 2)
                    act = activation.get(n2, 0) * decay * fan
                    if act > 0.005:
                        activation[n3] = max(activation.get(n3, 0), act)

        # Score candidates by activation level
        scores: dict[str, float] = {}
        for m in cands:
            sl = m.subject.lower()
            ol = m.object_value.lower()
            act = max(activation.get(sl, 0), activation.get(ol, 0))
            if act > 0:
                # Boost by Hebbian synapse strength if available
                synapse_boost = 0.0
                for other_id in [r for r in activation if r != sl and r != ol]:
                    key = (min(m.id, other_id), max(m.id, other_id))
                    synapse_boost = max(synapse_boost,
                                       self._synapses.get(key, 0.0))
                scores[m.id] = min(1.0, act + synapse_boost * 0.3)
        return scores

    def _resonance(self, cands: list[Memory]) -> dict[str, float]:
        """Resonance channel: score based on co-activation frequency.
        Facts that frequently appear together in results get boosted."""
        if not self._traces or len(cands) < 2:
            return {}
        scores: dict[str, float] = {}
        cand_ids = {m.id for m in cands}
        for m in cands:
            if m.id not in self._traces:
                continue
            # Sum co-occurrence with other candidates
            coact = sum(self._traces[m.id].get(other, 0)
                        for other in cand_ids if other != m.id)
            if coact > 0:
                scores[m.id] = min(1.0, math.log2(1 + coact) / 5.0)
        return scores
