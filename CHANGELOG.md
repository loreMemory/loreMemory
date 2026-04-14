# Changelog

## v1.0.0 — 2026-04-14

Initial public release.

### Features

- **Zero-dependency core** — works out of the box with Python 3.9+, no external packages required.
- **3-line quickstart** — `from lore_memory import Memory; m = Memory(); m.store("...")`
- **Multi-channel retrieval** — semantic, keyword, temporal, belief, frequency, graph, and resonance channels fused into a single ranked result list.
- **Automatic fact extraction** — grammar-based parsing extracts structured subject-predicate-object triples from natural text.
- **Contradiction detection** — new facts that conflict with existing ones are flagged and resolved via Bayesian belief updates.
- **Deduplication** — duplicate or near-duplicate memories are merged automatically.
- **Multi-user isolation** — each user gets a fully isolated memory space.
- **Organization scopes** — shared memories visible across an org, private memories visible only to the owner.
- **Temporal consolidation** — memory decay, access replay, and archival for long-running systems.
- **Feedback-driven learning** — mark results as helpful/unhelpful to tune retrieval weights.
- **CLI** — `lore store`, `lore query`, `lore list`, `lore forget`, `lore stats`, `lore serve`, `lore mcp`.
- **REST API** — FastAPI server with full CRUD endpoints (optional dependency).
- **MCP server** — Claude Desktop integration via Model Context Protocol (optional dependency).
- **Custom embeddings** — plug in any embedding model (e.g., sentence-transformers) for higher accuracy.
