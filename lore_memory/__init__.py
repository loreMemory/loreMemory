"""
Lore Memory — persistent AI memory that learns from every interaction.

Usage:
    from lore_memory import Memory

    m = Memory()
    m.store("I work at Google and love Python")
    results = m.query("where do I work?")
    print(results[0]["text"])  # "I work at Google and love Python"
"""

__version__ = "1.0.6"

from lore_memory.memory import Memory

__all__ = ["Memory", "__version__"]
