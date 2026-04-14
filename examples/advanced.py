"""
Advanced features: triples, temporal queries, and feedback.

Demonstrates:
- Storing structured triples directly
- Temporal memory consolidation
- Feedback-driven learning
"""

from lore_memory import Memory

m = Memory(user_id="advanced_demo")

# --- Structured triples ---
# Store facts as explicit subject-predicate-object triples
m.store_triple("Alice", "works_at", "Google", confidence=0.95)
m.store_triple("Alice", "speaks", "Python", confidence=0.9)
m.store_triple("Alice", "lives_in", "San Francisco", confidence=0.8)

print("Stored 3 structured triples.\n")

# Query using natural language — retrieval works across both
# natural text memories and structured triples
results = m.query("where does Alice work?")
print("Where does Alice work?")
for r in results:
    print(f"  {r.subject} -> {r.predicate} -> {r.object}  "
          f"(confidence: {r.confidence}, score: {r.score})")

# --- Feedback ---
# Tell the engine which results were helpful.
# This adjusts retrieval channel weights over time.
if results:
    m.feedback(results[0].id, helpful=True)
    print(f"\nMarked result '{results[0].text}' as helpful.")

# --- Temporal consolidation ---
# Over time, memories decay. Run consolidation to:
#   - Decay old, unaccessed memories
#   - Replay access traces
#   - Archive low-value memories
result = m.consolidate()
print(f"\nConsolidation: {result}")

# --- Profile: everything known about this user ---
print("\nFull profile:")
profile = m.profile()
for predicate, entries in profile.items():
    print(f"  {predicate}: {entries}")

# --- Clean up ---
m.close()
