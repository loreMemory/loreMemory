"""Tests for all four memory contexts.
v4: Updated for text-first architecture — tests verify retrieval works,
not that specific regex patterns extracted specific predicates."""

import os, pytest
from lore_memory.engine import Engine, Config


class TestPersonal:
    def test_store_recall(self, engine):
        engine.store_personal("u1", "I live in Amsterdam")
        r = engine.recall("u1", "Where do I live?")
        assert any("Amsterdam" in x.memory.object_value for x in r)

    def test_multiple_facts(self, engine):
        engine.store_personal("u1", "I live in Amsterdam")
        engine.store_personal("u1", "I work at Google")
        r = engine.recall("u1", "Tell me about myself")
        assert len(r) >= 2

    def test_profile(self, engine):
        """Profile groups by predicate. Text-first uses 'stated' predicate.
        store_fact still creates structured predicates."""
        engine.store_fact("private", "personal", "u1", "lives_in", "Amsterdam", user_id="u1")
        engine.store_fact("private", "personal", "u1", "works_at", "Google", user_id="u1")
        p = engine.profile("u1", "u1")
        assert "lives_in" in p
        assert "works_at" in p

    def test_dedup(self, engine):
        r1 = engine.store_personal("u1", "I live in Amsterdam")
        r2 = engine.store_personal("u1", "I live in Amsterdam")
        assert r1.created >= 1
        assert r2.deduplicated >= 1

    def test_contradiction_via_store_fact(self, engine):
        """Contradiction detection works via structured store_fact API."""
        engine.store_fact("private", "personal", "u1", "lives_in", "Amsterdam", user_id="u1")
        r = engine.store_fact("private", "personal", "u1", "lives_in", "Berlin", user_id="u1")
        assert r.contradictions >= 1

    def test_any_language(self, engine):
        """Text-first handles any language without regex."""
        engine.store_personal("u1", "أنا أعيش في أمستردام")
        r = engine.recall("u1", "أمستردام")
        assert any("أمستردام" in x.memory.object_value for x in r)

    def test_any_phrasing(self, engine):
        """Text-first handles any phrasing without predefined patterns."""
        engine.store_personal("u1", "Amsterdam is where I call home these days")
        r = engine.recall("u1", "Amsterdam")
        assert any("Amsterdam" in x.memory.object_value for x in r)


class TestChat:
    def test_extract_facts(self, engine):
        engine.store_chat("u1", "I prefer Python over JavaScript", session_id="s1")
        r = engine.recall("u1", "Python prefer")
        assert len(r) > 0

    def test_extract_decisions(self, engine):
        engine.store_chat("u1", "We decided to use PostgreSQL for the database", session_id="s1")
        r = engine.recall("u1", "PostgreSQL")
        assert any("PostgreSQL" in x.memory.object_value for x in r)

    def test_third_person(self, engine):
        engine.store_chat("u1", "Alice lives in Tokyo", session_id="s1")
        r = engine.recall("u1", "Alice Tokyo")
        assert len(r) > 0

    def test_negation_via_store_fact(self, engine):
        """Negation flag works via structured store_fact API."""
        engine.store_fact("private", "chat", "u1", "likes", "Java",
                          user_id="u1", is_negation=True)
        r = engine.recall("u1", "Java")
        found = [x for x in r if "Java" in x.memory.object_value]
        assert found[0].memory.is_negation


class TestRepo:
    def test_ingest_current_repo(self, engine):
        repo_path = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
        if not os.path.isdir(os.path.join(repo_path, ".git")):
            pytest.skip("Not in git repo")
        from lore_memory.repo import repo_id as rid_fn
        rid = rid_fn(repo_path)
        result = engine.ingest_repo("u1", repo_path, commit_limit=10)
        assert result.created > 0
        r = engine.recall("u1", "What changed recently?", repo_id_val=rid)
        assert len(r) > 0

    def test_store_repo_fact(self, engine):
        engine.store_fact("repo", "repo", "main.py", "implements", "FastAPI app", repo_id_val="test")
        r = engine.recall("u1", "main.py implements", repo_id_val="test")
        assert len(r) > 0

    def test_empty_repo(self, engine, tmp_dir):
        os.makedirs(os.path.join(tmp_dir, "empty_repo", ".git"))
        result = engine.ingest_repo("u1", os.path.join(tmp_dir, "empty_repo"))
        assert result.created == 0


class TestCompany:
    def test_shared_fact(self, engine):
        engine.store_company("u1", "acme", "Our mission is to democratize AI")
        r = engine.recall("u1", "mission democratize", org_id="acme")
        assert len(r) > 0

    def test_multi_user_shared(self, engine):
        engine.store_company("alice", "acme", "The roadmap includes mobile support")
        r_a = engine.recall("alice", "roadmap", org_id="acme")
        r_b = engine.recall("bob", "roadmap", org_id="acme")
        assert len(r_a) > 0
        assert len(r_b) > 0

    def test_attribution(self, engine):
        engine.store_company("alice", "acme", "We prefer microservices")
        r = engine.recall("alice", "microservices", org_id="acme")
        if r:
            assert r[0].memory.user_id == "alice"


class TestCrossContext:
    def test_merge_private_shared(self, engine):
        engine.store_personal("u1", "I am a Python developer")
        engine.store_company("u1", "acme", "Our stack includes Python and React")
        r = engine.recall("u1", "Python", org_id="acme")
        assert len(r) >= 1

    def test_stats(self, engine):
        engine.store_personal("u1", "I like cats")
        engine.store_company("u1", "acme", "Our mascot is a dog")
        s = engine.stats("u1", org_id="acme")
        assert s["private_total"] >= 1
        assert s["shared_total"] >= 1


class TestFeedback:
    def test_helpful_feedback(self, engine):
        engine.store_personal("u1", "I like Python")
        r = engine.recall("u1", "Python")
        assert len(r) > 0
        mid = r[0].memory.id
        ok = engine.feedback("u1", mid, helpful=True)
        assert ok
        mem = engine._db(Scope.PRIVATE, user_id="u1").get(mid)
        assert mem.evidence_count >= 2

    def test_unhelpful_feedback(self, engine):
        engine.store_personal("u1", "I like Java")
        r = engine.recall("u1", "Java")
        if r:
            mid = r[0].memory.id
            engine.feedback("u1", mid, helpful=False)
            mem = engine._db(Scope.PRIVATE, user_id="u1").get(mid)
            assert mem.contradiction_count >= 1

    def test_feedback_respects_scope(self, engine):
        engine.store_personal("alice", "I like cats")
        r = engine.recall("alice", "cats")
        if r:
            ok = engine.feedback("bob", r[0].memory.id, helpful=True)
            assert not ok


class TestRecovery:
    def test_recover_archived(self, engine):
        engine.store_personal("u1", "My temporary fact is XYZ123")
        r = engine.recall("u1", "XYZ123")
        if r:
            mid = r[0].memory.id
            db = engine._db(Scope.PRIVATE, user_id="u1")
            db.update_state(mid, "archived")
            r2 = engine.recall("u1", "XYZ123")
            assert not any(x.memory.id == mid for x in r2)
            ok = engine.recover("u1", mid)
            assert ok
            r3 = engine.recall("u1", "XYZ123")
            assert any(x.memory.id == mid for x in r3)


from lore_memory.scopes import Scope
