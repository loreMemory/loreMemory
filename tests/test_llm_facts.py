"""LLM-supplied facts path: store(text, facts=...) and query hints.

Locks in that:
  - facts=[...] writes structured triples without grammar parsing
  - the raw text is still saved as a 'stated' row (FTS recall preserved)
  - subject "user" is rewritten to user_id
  - predicate_hint / subject_hint act as boosts, not filters
"""
from __future__ import annotations

from lore_memory import Memory


def test_store_with_facts_writes_stated_plus_triples(tmp_path):
    m = Memory(user_id="u1", data_dir=str(tmp_path))
    r = m.store(
        "I have a cat named Luna and I climb on weekends",
        facts=[
            {"subject": "user", "predicate": "pet", "object": "Luna"},
            {"subject": "user", "predicate": "hobby", "object": "climbing"},
        ],
    )
    # 1 stated row + 2 facts
    assert r["created"] == 3

    # Retrievable by canonical predicate
    pet = m.query("what's my pet?", predicate_hint="pet", subject_hint="user", limit=1)
    assert pet and pet[0].object == "Luna"

    hobby = m.query("what's my hobby?", predicate_hint="hobby", subject_hint="user", limit=1)
    assert hobby and hobby[0].object == "climbing"


def test_facts_skip_grammar_parser(tmp_path):
    """When facts is supplied, the grammar parser should NOT also extract."""
    m = Memory(user_id="u2", data_dir=str(tmp_path))
    m.store(
        "I am a software engineer at Stripe",
        facts=[{"subject": "user", "predicate": "job_title", "object": "software engineer"}],
    )
    # Profile should have job_title=software engineer, not is_a or work_at
    # (since facts bypass grammar entirely)
    prof = m.profile()
    flat_predicates = set(prof.keys())
    assert "job_title" in flat_predicates
    # No is_a row should exist for this text — grammar was skipped
    assert "is_a" not in flat_predicates


def test_subject_user_rewrites_to_user_id(tmp_path):
    m = Memory(user_id="alice", data_dir=str(tmp_path))
    m.store("I live in Berlin",
            facts=[{"subject": "user", "predicate": "lives_in", "object": "Berlin"}])
    # The fact should be stored with subject=alice (not "user")
    rs = m.query("where do I live?", limit=5)
    assert any(r.subject == "alice" and r.object == "Berlin"
               for r in rs)


def test_hints_are_boosts_not_filters(tmp_path):
    """A wrong predicate_hint must never hide a correct answer."""
    m = Memory(user_id="u3", data_dir=str(tmp_path))
    m.store("I love sushi",
            facts=[{"subject": "user", "predicate": "likes", "object": "sushi"}])

    # Hint with the WRONG predicate. Should still find the fact.
    rs = m.query("what do I like to eat?",
                 predicate_hint="absolutely_not_a_real_predicate",
                 subject_hint="user", limit=5)
    # The fact must still surface even with a wrong hint
    assert any(r.predicate == "likes" and r.object == "sushi" for r in rs)


def test_no_facts_falls_back_to_grammar(tmp_path):
    """Calling store(text) without facts uses grammar extraction (no regression)."""
    m = Memory(user_id="u4", data_dir=str(tmp_path))
    r = m.store("I live in Berlin")
    # Grammar extracts at least one fact + the stated row
    assert r["created"] >= 2


def test_invalid_fact_dict_is_skipped_not_raised(tmp_path):
    """Malformed fact entries shouldn't blow up the whole store call."""
    m = Memory(user_id="u5", data_dir=str(tmp_path))
    r = m.store(
        "I live in Berlin",
        facts=[
            {"subject": "user", "predicate": "lives_in", "object": "Berlin"},
            {"subject": "user", "predicate": "", "object": "missing pred"},
            {"object": "missing subject and predicate"},
            None,
        ],
    )
    # Stated row + the one valid fact
    assert r["created"] == 2
