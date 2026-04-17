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
    from pydantic import BaseModel, Field
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

class Fact(BaseModel):
    """LLM-extracted subject-predicate-object triple.

    Subject "user" maps to the caller's user_id. Predicate is normalized
    and canonicalized via the active Schema. Confidence defaults to 0.9.
    """
    subject: str
    predicate: str
    object: str
    confidence: float = 0.9
    is_negation: bool = False


class StoreRequest(BaseModel):
    text: str
    user_id: str = "default"
    scope: str = "private"
    facts: list[Fact] | None = Field(
        default=None,
        description=(
            "Optional LLM-extracted S-P-O triples. When provided, the "
            "local grammar parser is skipped and these triples are written "
            "directly. The raw text is always saved for FTS keyword recall. "
            "Use canonical predicates: lives_in, works_at, job_title, pet, "
            "hobby, partner, sister, manager, likes, allergic_to, etc."
        ),
    )


class QueryRequest(BaseModel):
    query: str
    user_id: str = "default"
    limit: int = 10
    predicate_hint: str | list[str] | None = Field(
        default=None,
        description=(
            "Canonical predicate(s) the answer likely uses. Boost only — "
            "wrong hints don't hide correct answers."
        ),
    )
    subject_hint: str | None = Field(
        default=None,
        description='Subject to favor. Pass "user" for first-person questions.',
    )


# --- Routes ---

@app.post("/memory")
def store_memory(req: StoreRequest):
    """Store a memory.

    Two modes:
      1. text only — local grammar parser extracts facts (works for any
         caller).
      2. text + facts — caller (typically an LLM) supplies S-P-O triples.
         Grammar parser is skipped; raw text is still saved for FTS.
    """
    m = _get_memory(req.user_id)
    facts_payload = [f.model_dump() for f in req.facts] if req.facts else None
    result = m.store(req.text, scope=req.scope, facts=facts_payload)
    return result


@app.post("/memory/query")
def query_memory_post(req: QueryRequest):
    """Query memory with optional LLM hints (preferred for LLM callers).

    Hints are boosts, not filters: a wrong hint never hides a correct
    answer; a right hint surfaces it instantly.
    """
    m = _get_memory(req.user_id)
    results = m.query(
        req.query, limit=req.limit,
        predicate_hint=req.predicate_hint, subject_hint=req.subject_hint)
    return [r.to_dict() for r in results]


@app.get("/memory")
def query_memory(query: str, user_id: str = "default", limit: int = 10,
                 predicate_hint: str | None = None,
                 subject_hint: str | None = None):
    """Query memory via GET. Hints accepted as query-string params."""
    m = _get_memory(user_id)
    results = m.query(query, limit=limit,
                      predicate_hint=predicate_hint,
                      subject_hint=subject_hint)
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
