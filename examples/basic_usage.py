"""
Basic usage of Lore Memory.

Just 3 lines to get started:
    1. Create a Memory instance
    2. Store something
    3. Query it back
"""

from lore_memory import Memory

# Create a memory instance (stores data in ~/.lore-memory by default)
m = Memory()

# Store some facts
m.store("I work at Google and love Python")
m.store("My favorite color is blue")
m.store("I have a dog named Max")

# Query memories
results = m.query("where do I work?")
for r in results:
    print(f"  {r.text}  (score: {r.score})")

# Check what's stored
print("\nProfile:")
profile = m.profile()
for key, values in profile.items():
    print(f"  {key}: {values}")

# Stats
print("\nStats:", m.stats())

# Clean up
m.close()
