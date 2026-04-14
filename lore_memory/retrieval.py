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


_STOPS = frozenset({
    "the", "is", "at", "in", "on", "of", "to", "and", "or", "an", "it", "be",
    "as", "do", "by", "for", "was", "are", "has", "had", "not", "but", "its",
    "he", "she", "we", "my", "what", "which", "who", "this", "that", "am",
    "been", "have", "does", "did", "will", "would", "can", "could",
    "about", "with", "from", "into", "where", "when", "how", "all",
    "me", "him", "them", "you", "your", "our", "their", "tell", "know",
})


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in re.split(r"[\s\-_/\\.,;:!?'\"()\[\]{}]+", text)
            if t and len(t) > 1 and t.lower() not in _STOPS]


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
    ) -> list[SearchResult]:
        now = time.time()
        q_emb = self.embed(query)
        q_terms = tokenize(query)
        w = self.get_weights(weight_key)
        results: list[SearchResult] = []

        for db, scope_label in dbs:
            seen_ids: set[str] = set()
            cands: list[Memory] = []

            # Primary: FTS keyword search (O(log n))
            if q_terms:
                safe_terms = [t.replace('"', '').replace("'", "").replace("*", "")
                              .replace(":", "").replace("(", "").replace(")", "")
                              for t in q_terms if t.replace('"', '').strip()]
                # Use prefix matching (term*) so "auth" matches "authentication"
                fts_parts = [f"{t}*" for t in safe_terms if len(t) >= 2]
                fts_query = " OR ".join(fts_parts) if fts_parts else ""
                try:
                    fts_hits = db.fts_search(fts_query, limit=top_k * 5) if fts_query else []
                except Exception:
                    fts_hits = []
                for m in fts_hits:
                    if m.is_active:
                        cands.append(m)
                        seen_ids.add(m.id)

            # Secondary: vector similarity search (finds semantic matches FTS misses)
            # "pets" → "cat", "exercise" → "run", "live" → "based in"
            vec_hits = db.vector_search(q_emb, top_k=top_k, context=context)
            for m, sim in vec_hits:
                if m.id not in seen_ids and m.is_active:
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

            # RRF merge
            for m in cands:
                rrf = 0.0
                ch: dict[str, float] = {}
                for i, (scores, weight) in enumerate(channels):
                    raw = scores.get(m.id, 0.0)
                    ch[names[i]] = raw
                    rank = rank_maps[i].get(m.id, len(scores))
                    rrf += weight * (1.0 / (60 + rank + 1))
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
            final.append(r)
            if len(final) >= top_k:
                break

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

    # --- Channel scoring functions ---

    def _kw_scores(self, terms: list[str], cands: list[Memory]) -> dict[str, float]:
        if not terms:
            return {}
        scores: dict[str, float] = {}
        for m in cands:
            text = f"{m.subject} {m.predicate.replace('_', ' ')} {m.object_value} {m.source_text}"
            doc = tokenize(text)
            if not doc:
                continue
            hits = sum(1 for qt in terms if any(
                d.startswith(qt) or qt.startswith(d) for d in doc))
            if hits > 0:
                scores[m.id] = hits / len(terms)
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
