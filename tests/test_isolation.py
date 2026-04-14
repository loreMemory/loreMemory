"""Zero scope leakage tests — any failure is automatic disqualification."""

import os, pytest
from lore_memory.engine import Engine, Config
from lore_memory.scopes import Scope, scope_db_path, can_access


class TestPhysicalIsolation:
    def test_separate_files_per_user(self, tmp_dir):
        from pathlib import Path
        p = Path(tmp_dir)
        a = scope_db_path(p, Scope.PRIVATE, user_id="alice")
        b = scope_db_path(p, Scope.PRIVATE, user_id="bob")
        assert a != b

    def test_separate_files_per_repo(self, tmp_dir):
        from pathlib import Path
        p = Path(tmp_dir)
        a = scope_db_path(p, Scope.REPO, repo_id="repo_a")
        b = scope_db_path(p, Scope.REPO, repo_id="repo_b")
        assert a != b

    def test_same_org_same_file(self, tmp_dir):
        from pathlib import Path
        p = Path(tmp_dir)
        a = scope_db_path(p, Scope.SHARED, org_id="acme")
        b = scope_db_path(p, Scope.SHARED, org_id="acme")
        assert a == b

    def test_different_orgs_different_files(self, tmp_dir):
        from pathlib import Path
        p = Path(tmp_dir)
        a = scope_db_path(p, Scope.SHARED, org_id="acme")
        b = scope_db_path(p, Scope.SHARED, org_id="globex")
        assert a != b


class TestAccessControl:
    def test_own_private(self, tmp_dir):
        assert can_access("alice", Scope.PRIVATE, "alice") is True

    def test_other_private_blocked(self, tmp_dir):
        assert can_access("alice", Scope.PRIVATE, "bob") is False


class TestZeroLeakage:
    def test_private_invisible_to_other_user(self, engine):
        engine.store_personal("alice", "I live in Amsterdam and work at Google")
        engine.store_personal("bob", "I live in Berlin and work at Meta")

        alice_r = engine.recall("alice", "Where do I live?")
        bob_r = engine.recall("bob", "Where do I live?")

        alice_vals = {r.memory.object_value for r in alice_r}
        bob_vals = {r.memory.object_value for r in bob_r}

        assert "Amsterdam" not in bob_vals, "LEAK: Bob sees Alice's city"
        assert "Google" not in bob_vals, "LEAK: Bob sees Alice's employer"
        assert "Berlin" not in alice_vals, "LEAK: Alice sees Bob's city"

    def test_chat_invisible_to_other_user(self, engine):
        engine.store_chat("alice", "We decided to use React", session_id="s1")
        engine.store_chat("bob", "We decided to use Vue", session_id="s2")

        for r in engine.recall("bob", "What framework?"):
            assert "React" not in r.memory.object_value, "LEAK: Bob sees Alice's decision"
        for r in engine.recall("alice", "What framework?"):
            assert "Vue" not in r.memory.object_value, "LEAK: Alice sees Bob's decision"

    def test_shared_visible_to_org(self, engine):
        engine.store_company("alice", "acme", "Our mission is to democratize AI")
        results = engine.recall("bob", "mission", org_id="acme")
        assert len(results) > 0, "Shared memory should be visible to org members"

    def test_shared_invisible_to_other_org(self, engine):
        engine.store_company("alice", "acme", "Our secret sauce is AI")
        results = engine.recall("bob", "secret sauce", org_id="globex")
        for r in results:
            assert "AI" not in r.memory.object_value, "LEAK: Globex sees Acme secrets"

    def test_repo_isolated(self, engine):
        engine.store_fact("repo", "repo", "main.py", "contains", "FastAPI", repo_id_val="repo_a")
        engine.store_fact("repo", "repo", "main.py", "contains", "Django", repo_id_val="repo_b")
        results = engine.recall("u1", "main.py contains", repo_id_val="repo_a")
        vals = {r.memory.object_value for r in results}
        assert "Django" not in vals, "LEAK: repo_a sees repo_b"

    def test_private_not_in_shared(self, engine):
        engine.store_personal("alice", "My salary is 200k")
        engine.store_company("alice", "acme", "Our revenue is 10M")
        for r in engine.recall("bob", "salary", org_id="acme"):
            assert "200k" not in r.memory.object_value, "LEAK: Private salary in shared scope"

    def test_db_files_separate(self, engine, tmp_dir):
        engine.store_personal("alice", "I like cats")
        engine.store_personal("bob", "I like dogs")
        priv = os.path.join(tmp_dir, "private")
        dbs = [f for f in os.listdir(priv) if f.endswith(".db")]
        assert len(dbs) == 2
        assert "user_alice.db" in dbs
        assert "user_bob.db" in dbs

    def test_10_user_stress(self, engine):
        """10 users with unique secrets — no cross-user visibility."""
        for i in range(10):
            engine.store_personal(f"user_{i}", f"My secret code is CODE{i}X")
        for i in range(10):
            results = engine.recall(f"user_{i}", "secret code")
            for r in results:
                # Verify this result belongs to the querying user
                assert r.memory.user_id == f"user_{i}", \
                    f"LEAK: user_{i} sees memory owned by {r.memory.user_id}"
