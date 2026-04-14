"""
Lore Memory REST API — FastAPI server for persistent AI memory.

Start with:
    lore serve
    # or
    uvicorn api.server:app --port 8000

Requires: pip install lore-memory[api]
"""

from __future__ import annotations

import sys

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except ImportError:
    print(
        "Error: FastAPI is not installed.\n"
        "Install it with: pip install lore-memory[api]\n"
        "Or directly: pip install fastapi uvicorn",
        file=sys.stderr,
    )
    sys.exit(1)

from lore_memory.memory import Memory

app = FastAPI(
    title="Lore Memory API",
    description="Persistent AI memory that learns from every interaction",
    version="1.0.0",
)

# Instance cache keyed by user_id
_instances: dict[str, Memory] = {}


def _get_memory(user_id: str = "default") -> Memory:
    if user_id not in _instances:
        _instances[user_id] = Memory(user_id=user_id)
    return _instances[user_id]


# --- Request / Response models ---

class StoreRequest(BaseModel):
    text: str
    user_id: str = "default"
    scope: str = "private"


class QueryRequest(BaseModel):
    query: str
    user_id: str = "default"
    limit: int = 10


# --- Routes ---

@app.post("/memory")
def store_memory(req: StoreRequest):
    """Store a memory from natural language text."""
    m = _get_memory(req.user_id)
    result = m.store(req.text, scope=req.scope)
    return result


@app.get("/memory")
def query_memory(query: str, user_id: str = "default", limit: int = 10):
    """Query stored memories using natural language."""
    m = _get_memory(user_id)
    results = m.query(query, limit=limit)
    return [r.to_dict() for r in results]


@app.delete("/memory/{memory_id}")
def delete_memory(memory_id: str, user_id: str = "default"):
    """Delete a specific memory by ID."""
    m = _get_memory(user_id)
    ok = m.forget(memory_id=memory_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"deleted": True}


@app.get("/memory/list")
def list_memories(user_id: str = "default"):
    """List all stored memories organized by category."""
    m = _get_memory(user_id)
    return m.profile()


@app.get("/memory/stats")
def memory_stats(user_id: str = "default"):
    """Get memory statistics."""
    m = _get_memory(user_id)
    return m.stats()


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "lore-memory", "version": "1.0.0"}
