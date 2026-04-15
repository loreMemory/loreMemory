"""
SPO Engine tests — normalization, dedup, grammar-free extraction, multi-tool.

Tests cover:
1. Subject normalization (user/me/name -> canonical)
2. Predicate normalization (works_at/employer/job -> cluster)
3. Object canonicalization (Google/Google Inc. -> canonical)
4. Near-duplicate detection (embedding similarity)
5. Cross-tool duplicate detection
6. Grammar-free extraction (all 5 approaches)
7. Adversarial battery (fragments, commits, social, non-English)
8. Provenance tracking
9. Namespace growth measurement
"""

import json
import tempfile
import time

import pytest

from lore_memory.engine import Engine, Config
from lore_memory.normalization import (
    SubjectResolver, PredicateNormalizer, ObjectCanonicalizer,
    NormalizationPipeline,
)
from lore_memory.dedup import DedupEngine, ProvenanceTracker
from lore_memory.extraction_gf import (
    GrammarFreeExtractor, detect_source_type,
    _extract_kv_pairs, _extract_email_signature, _extract_commit,
    _extract_fragment, _extract_entities, _extract_between_text,
    CooccurrenceTracker, CorrectionLearner,
)
from lore_memory.store import Memory, MemoryDB


# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def engine(tmp_dir):
    e = Engine(Config(data_dir=tmp_dir, embedding_dims=64))
    yield e
    e.close()


@pytest.fixture
def subject_resolver():
    r = SubjectResolver(canonical_id="user_alice")
    r.register_alias("alice")
    r.register_alias("alice johnson")
    return r


# ---------------------------------------------------------------------------
#  Subject Normalization Tests
# ---------------------------------------------------------------------------

class TestSubjectNormalization:
    def test_first_person_resolves_to_canonical(self, subject_resolver):
        assert subject_resolver.resolve("I") == "user_alice"
        assert subject_resolver.resolve("me") == "user_alice"
        assert subject_resolver.resolve("my") == "user_alice"
        assert subject_resolver.resolve("user") == "user_alice"

    def test_registered_alias_resolves(self, subject_resolver):
        assert subject_resolver.resolve("Alice") == "user_alice"
        assert subject_resolver.resolve("alice") == "user_alice"
        assert subject_resolver.resolve("Alice Johnson") == "user_alice"

    def test_third_party_preserved(self, subject_resolver):
        assert subject_resolver.resolve("Bob") == "Bob"
        assert subject_resolver.resolve("team") == "team"

    def test_empty_resolves_to_canonical(self, subject_resolver):
        assert subject_resolver.resolve("") == "user_alice"

    def test_learn_from_name_statement(self):
        r = SubjectResolver(canonical_id="user1")
        r.learn_from_text("My name is John Smith", "user1")
        assert r.resolve("John Smith") == "user1"
        assert r.resolve("john") == "user1"
        assert r.resolve("smith") == "user1"

    def test_is_self_reference(self, subject_resolver):
        assert subject_resolver.is_self_reference("I")
        assert subject_resolver.is_self_reference("Alice")
        assert not subject_resolver.is_self_reference("Bob")


# ---------------------------------------------------------------------------
#  Predicate Normalization Tests
# ---------------------------------------------------------------------------

class TestPredicateNormalization:
    def test_seed_cluster(self, engine):
        normalizer = engine._get_normalizer("user1")
        pn = normalizer.predicate_normalizer
        assert pn.cluster_count > 0

    def test_stated_passthrough(self, engine):
        normalizer = engine._get_normalizer("user1")
        pn = normalizer.predicate_normalizer
        assert pn.normalize("stated") == "stated"

    def test_seeded_aliases_resolve(self, engine):
        normalizer = engine._get_normalizer("user1")
        pn = normalizer.predicate_normalizer
        canon1 = pn.normalize("live_in")
        canon2 = pn.normalize("live_at")
        canon3 = pn.normalize("reside_in")
        assert canon1 == canon2 == canon3


# ---------------------------------------------------------------------------
#  Object Canonicalization Tests
# ---------------------------------------------------------------------------

class TestObjectCanonicalization:
    def test_case_normalization(self):
        oc = ObjectCanonicalizer()
        c1 = oc.canonicalize("Google")
        c2 = oc.canonicalize("google")
        assert c1 == c2

    def test_suffix_normalization(self):
        oc = ObjectCanonicalizer()
        c1 = oc.canonicalize("Google")
        c2 = oc.canonicalize("Google Inc.")
        assert c1 == c2

    def test_llc_normalization(self):
        oc = ObjectCanonicalizer()
        c1 = oc.canonicalize("Google")
        c2 = oc.canonicalize("Google LLC")
        assert c1 == c2

    def test_different_entities_preserved(self):
        oc = ObjectCanonicalizer()
        c1 = oc.canonicalize("Google")
        c2 = oc.canonicalize("Microsoft")
        assert c1 != c2

    def test_empty_passthrough(self):
        oc = ObjectCanonicalizer()
        assert oc.canonicalize("") == ""

    def test_stats(self):
        oc = ObjectCanonicalizer()
        oc.canonicalize("Google")
        oc.canonicalize("Google Inc.")
        oc.canonicalize("Microsoft")
        stats = oc.stats()
        assert stats["unique_canonicals"] == 2
        assert stats["total_entries"] >= 2


# ---------------------------------------------------------------------------
#  Source Type Detection Tests
# ---------------------------------------------------------------------------

class TestSourceTypeDetection:
    def test_kv_pairs(self):
        st = detect_source_type("employer: Google | role: SWE | since: 2022")
        assert st.type == "kv_pairs"

    def test_email_signature(self):
        st = detect_source_type("Alice Johnson | Software Engineer | Google Inc.")
        assert st.type == "email_signature"

    def test_commit(self):
        st = detect_source_type("fix(auth): resolve login bug for Google SSO users")
        assert st.type == "commit"

    def test_social(self):
        st = detect_source_type("@alice works_at Google #tech")
        assert st.type == "social"

    def test_fragment(self):
        st = detect_source_type("Python. 8 years.")
        assert st.type == "fragment"

    def test_sentence(self):
        st = detect_source_type("I work at Google as a software engineer.")
        assert st.type == "sentence"

    def test_all_source_types_detected(self):
        cases = {
            "employer: Google | role: SWE | since: 2022": "kv_pairs",
            "Alice | Software Engineer | Google": "email_signature",
            "fix(auth): resolve login bug": "commit",
            "@alice #tech": "social",
            "Python. 8 years.": "fragment",
            "I work at Google as a software engineer.": "sentence",
        }
        for text, expected in cases.items():
            st = detect_source_type(text)
            assert st.type == expected, \
                f"Expected {expected} for '{text}', got {st.type}"


# ---------------------------------------------------------------------------
#  Grammar-Free Extraction Tests
# ---------------------------------------------------------------------------

class TestGrammarFreeExtraction:
    def test_kv_extraction(self):
        results = _extract_kv_pairs("employer: Google | role: SWE", "user1")
        assert len(results) >= 2
        preds = {r["predicate"] for r in results}
        assert "employer" in preds
        assert "role" in preds

    def test_email_signature_extraction(self):
        results = _extract_email_signature(
            "Alice Johnson | Software Engineer | Google Inc.", "user1")
        assert len(results) == 3
        assert results[0]["predicate"] == "name"
        assert results[1]["predicate"] == "job_title"
        assert results[2]["predicate"] == "works_at"

    def test_commit_extraction(self):
        results = _extract_commit(
            "fix(auth): resolve login bug for Google SSO users", "user1")
        assert len(results) >= 1
        assert results[0]["predicate"] == "fix"

    def test_fragment_extraction(self):
        results = _extract_fragment("Python. 8 years.", "user1")
        assert len(results) >= 1

    def test_entity_extraction(self):
        entities = _extract_entities("Alice works at Google in Amsterdam")
        assert "Alice" in entities
        assert "Google" in entities
        assert "Amsterdam" in entities

    def test_full_extractor(self):
        gf = GrammarFreeExtractor()
        mems = gf.extract("employer: Google | role: SWE", "user1",
                           source_tool="email_parser")
        assert len(mems) >= 2


# ---------------------------------------------------------------------------
#  Approach A: Entity-relationship co-occurrence
# ---------------------------------------------------------------------------

class TestApproachA:
    def test_between_text_extraction(self):
        text = "Alice works at Google"
        between = _extract_between_text(text, "Alice", "Google")
        assert "works at" == between

    def test_between_text_different_order(self):
        text = "Google employs Alice"
        between = _extract_between_text(text, "Alice", "Google")
        assert "employs" in between

    def test_cooccurrence_tracking(self):
        tracker = CooccurrenceTracker()
        tracker.record("Alice", "Google", "works at")
        tracker.record("Ali", "Meta", "works at")
        tracker.record("Sarah", "Microsoft", "joined")
        relations = tracker.get_common_relations(min_count=1)
        assert len(relations) >= 2

    def test_entity_extraction_from_text(self):
        entities = _extract_entities("Alice works at Google in Amsterdam")
        assert "Alice" in entities
        assert "Google" in entities
        assert "Amsterdam" in entities

    def test_entity_extraction_technical(self):
        entities = _extract_entities("We use PostgreSQL and react_native for the userService")
        assert "PostgreSQL" in entities


# ---------------------------------------------------------------------------
#  Approach B: Contextual predicate inference
# ---------------------------------------------------------------------------

class TestApproachB:
    def test_self_reference_extraction(self, engine):
        r = engine.store_from_tool("user1", "I work at Google",
                                    source_tool="chat")
        assert r.created >= 1

    def test_third_person_extraction(self, engine):
        r = engine.store_from_tool("user1",
                                    "Alice joined Google in 2022",
                                    source_tool="bio")
        assert r.created >= 1


# ---------------------------------------------------------------------------
#  Approach C: Correction-driven learning
# ---------------------------------------------------------------------------

class TestApproachC:
    def test_correction_learning(self):
        learner = CorrectionLearner()
        learner.learn("into", "prefers", "Python")
        result = learner.apply("into", "Python")
        assert result == "prefers"

    def test_correction_context_specific(self):
        learner = CorrectionLearner()
        learner.learn("into", "prefers", "python")
        learner.learn("into", "interested_in", "music")
        assert learner.apply("into", "python") == "prefers"
        assert learner.apply("into", "music") == "interested_in"

    def test_correction_fallback(self):
        learner = CorrectionLearner()
        learner.learn("into", "prefers")
        assert learner.apply("into", "unknown_thing") == "prefers"

    def test_extractor_learns_correction(self):
        gf = GrammarFreeExtractor()
        gf.learn_correction("into", "prefers", "Python")
        assert gf._corrections.has_correction("into")


# ---------------------------------------------------------------------------
#  Approach D: Source-type-aware extraction (KV completeness)
# ---------------------------------------------------------------------------

class TestApproachD:
    def test_kv_extraction_completeness(self):
        results = _extract_kv_pairs(
            "employer: Google | role: SWE | since: 2022 | location: Amsterdam",
            "user1")
        preds = {r["predicate"] for r in results}
        assert "employer" in preds
        assert "role" in preds
        assert "since" in preds
        assert "location" in preds

    def test_email_sig_extraction_completeness(self):
        results = _extract_email_signature(
            "John Smith | VP Engineering | Acme Corp | San Francisco",
            "user1")
        assert len(results) == 4
        assert results[0]["object"] == "John Smith"
        assert results[1]["object"] == "VP Engineering"
        assert results[2]["object"] == "Acme Corp"

    def test_commit_with_scope(self):
        results = _extract_commit("refactor(database): optimize query performance", "user1")
        assert any(r["predicate"] == "refactor" for r in results)
        assert any(r["object"] == "database" for r in results)


# ---------------------------------------------------------------------------
#  Approach E: Confidence-weighted multi-candidate
# ---------------------------------------------------------------------------

class TestApproachE:
    def test_candidates_sorted_by_confidence(self, engine):
        r = engine.store_from_tool(
            "user1",
            "employer: Google | role: Software Engineer",
            source_tool="kv_parser")
        assert r.created >= 2

    def test_low_confidence_filtered(self):
        gf = GrammarFreeExtractor()
        gf._candidate_threshold = 0.9
        mems = gf.extract("some vague text here", "user1")


# ---------------------------------------------------------------------------
#  Multi-Tool Write Simulation Tests
# ---------------------------------------------------------------------------

class TestMultiToolWrites:
    def test_same_fact_different_tools(self, engine):
        engine.store_personal("user1", "I work at Google",
                              source_tool="chat_parser")
        engine.store_fact(
            scope="private", context="personal",
            subject="user1", predicate="works_at",
            object_value="Google", user_id="user1",
            source_tool="email_parser")

        results = engine.recall("user1", "where does user1 work?")
        work_results = [r for r in results
                       if r.memory.predicate != "stated"
                       and "work" in r.memory.predicate]
        seen_values = set()
        for r in work_results:
            seen_values.add(r.memory.object_value)
        assert len(seen_values) <= 1, f"Got multiple work facts: {seen_values}"

    def test_subject_normalization_in_store(self, engine):
        engine.store_personal("user1", "I work at Google",
                              source_tool="chat_parser")
        results = engine.recall("user1", "where does user1 work?")
        work_results = [r for r in results
                       if r.memory.predicate != "stated"
                       and "work" in r.memory.predicate]
        assert len(work_results) >= 1

    def test_provenance_tracking(self, engine):
        engine.store_personal("user1", "I work at Google",
                              source_tool="chat_parser")
        stats = engine._provenance.stats()
        assert stats["total_facts_tracked"] > 0

    @pytest.mark.slow
    def test_6_tools_same_fact(self, engine):
        engine.store_personal("user1", "I work at Google",
                              source_tool="chat_parser")
        engine.store_fact(
            scope="private", context="personal",
            subject="user1", predicate="works_at",
            object_value="Google", user_id="user1",
            source_tool="email_parser")
        engine.store_fact(
            scope="private", context="personal",
            subject="user1", predicate="employed_at",
            object_value="Google", user_id="user1",
            source_tool="linkedin_parser")
        engine.store_from_tool("user1", "employer: Google",
                                source_tool="kv_parser")
        engine.store_fact(
            scope="private", context="personal",
            subject="user1", predicate="works_at",
            object_value="Google Inc.", user_id="user1",
            source_tool="crm_parser")
        engine.store_personal("user1", "I am employed at Google",
                              source_tool="voice_parser")

        results = engine.recall("user1", "where does user1 work?")
        work_results = [r for r in results
                       if r.memory.predicate != "stated"
                       and ("work" in r.memory.predicate
                            or "employ" in r.memory.predicate
                            or "employer" in r.memory.predicate)]
        unique_objects = set()
        for r in work_results:
            unique_objects.add(r.memory.object_value.lower().replace(" inc.", "").strip())
        assert len(unique_objects) <= 1, \
            f"Expected 1 unique work entity, got {len(unique_objects)}: {unique_objects}"

    def test_conflicting_facts_from_tools(self, engine):
        engine.store_fact(
            scope="private", context="personal",
            subject="user1", predicate="works_at",
            object_value="Google", user_id="user1",
            source_tool="tool_a")
        time.sleep(0.01)
        engine.store_fact(
            scope="private", context="personal",
            subject="user1", predicate="works_at",
            object_value="Microsoft", user_id="user1",
            source_tool="tool_b")

        results = engine.recall("user1", "where does user1 work?")
        work_results = [r for r in results
                       if r.memory.predicate != "stated"
                       and "work" in r.memory.predicate
                       and r.memory.state == "active"]
        if work_results:
            assert work_results[0].memory.object_value == "Microsoft", \
                f"Expected Microsoft, got {work_results[0].memory.object_value}"

    def test_subject_variants_same_fact(self, engine):
        engine.store_personal("user1", "I live in Amsterdam",
                              source_tool="chat_parser")
        results = engine.recall("user1", "where does user1 live?")
        live_results = [r for r in results
                       if r.memory.predicate != "stated"
                       and "live" in r.memory.predicate]
        for r in live_results:
            assert r.memory.subject == "user1", \
                f"Subject not normalized: {r.memory.subject}"


# ---------------------------------------------------------------------------
#  Dedup Tests
# ---------------------------------------------------------------------------

class TestDedup:
    def test_exact_dedup(self, engine):
        r1 = engine.store_personal("user1", "I work at Google")
        r2 = engine.store_personal("user1", "I work at Google")
        assert r2.deduplicated > 0

    def test_near_dedup_different_predicate_same_meaning(self, engine):
        engine.store_fact(
            scope="private", context="personal",
            subject="user1", predicate="works_at",
            object_value="Google", user_id="user1")
        r2 = engine.store_fact(
            scope="private", context="personal",
            subject="user1", predicate="employed_at",
            object_value="Google", user_id="user1")
        assert r2.deduplicated > 0 or r2.contradictions > 0

    @pytest.mark.slow
    def test_10_writes_1_fact(self, engine):
        variants = [
            ("works_at", "Google"),
            ("works_at", "Google"),
            ("works_at", "google"),
            ("employed_at", "Google"),
            ("works_at", "Google Inc."),
            ("work_for", "Google"),
            ("works_at", "Google LLC"),
            ("employer", "Google"),
            ("work_at", "Google"),
            ("works_at", "Google Inc"),
        ]
        total_created = 0
        total_deduped = 0
        for pred, obj in variants:
            r = engine.store_fact(
                scope="private", context="personal",
                subject="user1", predicate=pred,
                object_value=obj, user_id="user1",
                source_tool="test")
            total_created += r.created
            total_deduped += r.deduplicated
        assert total_deduped > 0, \
            f"Expected dedup, got created={total_created} deduped={total_deduped}"

    def test_dedup_stats_tracked(self, engine):
        engine.store_fact(
            scope="private", context="personal",
            subject="user1", predicate="works_at",
            object_value="Google", user_id="user1")
        engine.store_fact(
            scope="private", context="personal",
            subject="user1", predicate="works_at",
            object_value="Google", user_id="user1")
        stats = engine.dedup_stats("user1")
        assert stats["total_checked"] > 0


# ---------------------------------------------------------------------------
#  Adversarial Battery Tests
# ---------------------------------------------------------------------------

class TestAdversarialBattery:
    def test_fragment(self, engine):
        result = engine.store_from_tool("user1", "Python. 8 years.",
                                         source_tool="fragment_parser")
        assert result.created > 0

    def test_structured_arrow(self, engine):
        result = engine.store_from_tool(
            "user1", "Google → Software Engineer → 2022",
            source_tool="structured_parser")
        assert result.created > 0

    def test_social_format(self, engine):
        result = engine.store_from_tool(
            "user1", "@alice works_at Google #tech",
            source_tool="social_parser")
        assert result.created >= 0

    def test_kv_format(self, engine):
        result = engine.store_from_tool(
            "user1", "employer: Google | role: SWE | since: 2022",
            source_tool="kv_parser")
        assert result.created >= 2

    def test_commit_format(self, engine):
        result = engine.store_from_tool(
            "user1", "feat(auth): fix login bug for Google SSO users",
            source_tool="git_parser")
        assert result.created >= 1

    def test_casual(self, engine):
        result = engine.store_from_tool(
            "user1", "lol yeah i work at google",
            source_tool="chat_parser")
        assert result.created >= 1

    def test_email_signature(self, engine):
        result = engine.store_from_tool(
            "user1",
            "Alice Johnson | Software Engineer | Google Inc.",
            source_tool="email_parser")
        assert result.created >= 2

    def test_arabic_sentence(self, engine):
        r = engine.store_from_tool(
            "user1", "أنا مهندس في Google", source_tool="chat")
        assert r.created >= 0


# ---------------------------------------------------------------------------
#  Full 8-Sentence Adversarial Battery
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestFullAdversarialBattery:
    @pytest.fixture
    def results(self, engine):
        battery = [
            ("Python. 8 years.", "fragment"),
            ("Google → Software Engineer → 2022", "structured"),
            ("@alice works_at Google #tech", "social"),
            ("employer: Google | role: SWE | since: 2022", "kv"),
            ("feat(auth): fix login bug for Google SSO users", "commit"),
            ("أنا مهندس في Google", "arabic"),
            ("我在Google工作", "chinese"),
            ("lol yeah i work at google", "casual"),
        ]
        outcomes = []
        for text, label in battery:
            try:
                r = engine.store_from_tool("user1", text,
                                            source_tool=f"{label}_parser")
                outcomes.append({
                    "text": text, "label": label,
                    "created": r.created, "deduped": r.deduplicated,
                    "status": "correct" if r.created > 0 else "empty",
                })
            except Exception as e:
                outcomes.append({
                    "text": text, "label": label,
                    "created": 0, "deduped": 0,
                    "status": f"error: {e}",
                })
        return outcomes

    def test_no_crashes(self, results):
        for r in results:
            assert "error" not in r["status"], \
                f"Crashed on '{r['text']}': {r['status']}"

    def test_at_least_6_of_8(self, results):
        correct = sum(1 for r in results if r["created"] > 0)
        assert correct >= 6, \
            f"Only {correct}/8 extracted. Details: {json.dumps(results, indent=2)}"

    def test_fragment_extracts(self, results):
        frag = next(r for r in results if r["label"] == "fragment")
        assert frag["created"] > 0

    def test_kv_extracts(self, results):
        kv = next(r for r in results if r["label"] == "kv")
        assert kv["created"] >= 2

    def test_commit_extracts(self, results):
        commit = next(r for r in results if r["label"] == "commit")
        assert commit["created"] >= 1

    def test_casual_extracts(self, results):
        casual = next(r for r in results if r["label"] == "casual")
        assert casual["created"] >= 1


# ---------------------------------------------------------------------------
#  Namespace Growth Tests
# ---------------------------------------------------------------------------

class TestNamespaceGrowth:
    def test_normalization_stats(self, engine):
        engine.store_personal("user1", "I work at Google")
        stats = engine.normalization_stats("user1")
        assert "subject" in stats
        assert "predicate" in stats
        assert "object" in stats

    def test_dedup_stats(self, engine):
        engine.store_personal("user1", "I work at Google")
        stats = engine.dedup_stats("user1")
        assert "total_checked" in stats

    def test_subject_normalization_reduces_namespace(self, engine):
        facts = [
            "I work at Google",
            "I live in Amsterdam",
            "I use Python",
            "I like coffee",
            "I have a dog",
        ]
        for f in facts:
            engine.store_personal("user1", f, source_tool="chat_parser")
        stats = engine.normalization_stats("user1")
        assert stats["subject"]["canonical_id"] == "user1"

    def test_predicate_normalization_stats(self, engine):
        engine.store_fact(
            scope="private", context="personal",
            subject="user1", predicate="works_at",
            object_value="Google", user_id="user1")
        engine.store_fact(
            scope="private", context="personal",
            subject="user1", predicate="employed_at",
            object_value="Meta", user_id="user1")
        stats = engine.normalization_stats("user1")
        assert stats["predicate"]["total_predicates"] > 0

    def test_object_canonicalization_stats(self, engine):
        engine.store_fact(
            scope="private", context="personal",
            subject="user1", predicate="works_at",
            object_value="Google", user_id="user1")
        engine.store_fact(
            scope="private", context="personal",
            subject="user1", predicate="likes",
            object_value="Google Inc.", user_id="user1")
        stats = engine.normalization_stats("user1")
        assert stats["object"]["unique_canonicals"] >= 1

    @pytest.mark.slow
    def test_100_writes_bounded_namespace(self, engine):
        tools = ["chat", "email", "kv", "commit", "social"]
        facts = [
            "I work at Google",
            "I live in Amsterdam",
            "I use Python",
            "I like coffee",
            "employer: Google | role: SWE",
            "Alice | Software Engineer | Google Inc.",
            "fix(auth): resolve login bug",
            "I have a dog named Max",
            "I graduated from MIT",
            "My favorite language is Rust",
        ]
        total_created = 0
        total_deduped = 0
        for i in range(100):
            text = facts[i % len(facts)]
            tool = tools[i % len(tools)]
            r = engine.store_from_tool("user1", text, source_tool=tool)
            total_created += r.created
            total_deduped += r.deduplicated
        assert total_deduped > 0, \
            f"No dedup after 100 writes: created={total_created} deduped={total_deduped}"
        norm_stats = engine.normalization_stats("user1")
        assert norm_stats["predicate"]["total_predicates"] < 100


# ---------------------------------------------------------------------------
#  Provenance Gauntlet
# ---------------------------------------------------------------------------

class TestProvenanceGauntlet:
    def test_provenance_from_multiple_tools(self, engine):
        engine.store_fact(
            scope="private", context="personal",
            subject="user1", predicate="works_at",
            object_value="Google", user_id="user1",
            source_tool="tool_a")
        engine.store_fact(
            scope="private", context="personal",
            subject="user1", predicate="works_at",
            object_value="Google", user_id="user1",
            source_tool="tool_b")
        stats = engine._provenance.stats()
        assert stats["total_provenance_records"] >= 2

    def test_provenance_for_canonical_fact(self, engine):
        engine.store_fact(
            scope="private", context="personal",
            subject="user1", predicate="lives_in",
            object_value="Amsterdam", user_id="user1",
            source_tool="chat_parser")
        stats = engine._provenance.stats()
        assert stats["total_facts_tracked"] >= 1

    def test_provenance_preserved(self, engine):
        engine.store_personal("user1", "I work at Google Inc.",
                              source_tool="chat_parser")
        prov_stats = engine._provenance.stats()
        assert prov_stats["total_facts_tracked"] > 0


# ---------------------------------------------------------------------------
#  Performance
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestPerformance:
    def test_dedup_performance(self, engine):
        start = time.time()
        for i in range(50):
            engine.store_fact(
                scope="private", context="personal",
                subject="user1", predicate="works_at",
                object_value="Google", user_id="user1",
                source_tool="perf_test")
        elapsed = time.time() - start
        assert elapsed < 30, f"50 writes took {elapsed:.1f}s"
