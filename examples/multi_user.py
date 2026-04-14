"""
Multi-user memory isolation.

Each user gets their own isolated memory space.
User A cannot see User B's memories, and vice versa.
"""

from lore_memory import Memory

# Create two separate users
alice = Memory(user_id="alice")
bob = Memory(user_id="bob")

# Store different facts for each user
alice.store("I work at Google")
alice.store("My favorite language is Rust")

bob.store("I work at Microsoft")
bob.store("My favorite language is Go")

# Query — each user only sees their own memories
print("Alice asks 'where do I work?':")
for r in alice.query("where do I work?"):
    print(f"  {r.text}")

print("\nBob asks 'where do I work?':")
for r in bob.query("where do I work?"):
    print(f"  {r.text}")

# Stats are separate too
print(f"\nAlice stats: {alice.stats()}")
print(f"Bob stats:   {bob.stats()}")

# Clean up
alice.close()
bob.close()
