# Changelog

## v1.0.8 — 2026-04-16

Security hardening, per-tenant schema, scale, and a non-developer chat UX.

### Security & trust

- **Prompt-injection defense** (`lore_memory/safety.py`). Every `store()` passes through a classifier (sentence-transformer prototype match + HTML/XML-tag short-circuit). Suspicious text is tagged `source_type="suspicious"`, excluded from `profile_compact()`, and wrapped in `<user_stated_untrusted>…</user_stated_untrusted>` via `MemoryResult.to_llm_context()`. Disable with `Config(injection_defense=False)`.
- **Hedge / conditional detector.** `"If I get the offer I'll move to London"`, `"maybe I'll quit"`, `"I think I might switch jobs"` now store with `metadata.hypothetical=True`, posterior halved, and **do not supersede facts**; down-weighted 0.4× at retrieval time.
- **Reported-speech attribution.** `"My wife said we should move to Tokyo"` is attributed to the speaker — subject rewritten, `metadata.source_speaker` set — never overwrites the user's own fact. Down-weighted 0.6×.
- **GDPR hard-delete & export.** `Memory.forget(hard=True)` actually DELETEs rows (any state) and VACUUMs. `Memory.export_all()` + `export_to_jsonl()` + `import_from_jsonl()` round-trip for portability and audit.
- **Profile allowlist removed.** Previously silently hid user-vocabulary predicates (`plays_cello`, `has_mortgage`, …). Now only structural garbage is filtered.

### Per-tenant schema

- New `lore_memory/schema.py`. `Schema` dataclass with `aliases`, `single_valued`, `decay_factors`, `stated_protection_days`, plus `canon()` / `is_single_valued()` / `hash_key()`.
- Three presets ship: `PERSONAL_LIFE_SCHEMA` (default, exact prior behaviour), `CARE_TRACKING_SCHEMA` (medical tracking: `on_medication`, `dose`, `has_diagnosis`, `appointment_on`), `RESEARCH_NOTES_SCHEMA` (zero-decay claims).
- Threaded through `Memory(schema=…)` and `Engine(schema=…)`. Hash persisted per DB in a new `_meta` table; mismatch logs a warning, doesn't block.

### Retrieval correctness

- **First-person augmentation.** Queries phrased in first person (`I/my/me`) always pull the user's own active facts into the candidate pool — stops the "at 5 000 facts, 'where do I live?' returns my sister's city" failure mode.
- **First-person subject boost** (additive +1.5) on `_subject_alignment`.
- **Semantic-primary predicate alignment.** Query↔predicate-embedding cosine is computed for every candidate; `_QUERY_PRED_MAP` is now an opt-in override, not a gate. Works for vocabulary outside the builder's dictionary.
- **Canon-aware map lookup.** `"where do I work?"` now hits `join=Anthropic` (via `canon(join)=works_at`).
- **`"last month" / "yesterday"` no longer clamp `valid_until` to the past.** They describe event time, not expiry; past-tense facts stay retrievable.
- **FTS5 porter stemmer** (`tokenize='porter unicode61 remove_diacritics 2'`). `"learn"` matches `"learning"`, `"move"` matches `"moved"`, `"join"` matches `"joined"`.

### Scale

- **Dedup flattened.** All 5 tiers share one `query_by_subject` call per insert (was 5 separate). **+42% write throughput at 5K facts.**
- **Skip near-dup for single-valued predicates.** Tier 3 handles supersession; Tier 2's 200 cosine ops per insert are redundant.
- **Class-level sentence-transformer singleton** (`Engine._ST_MODEL_CACHE`). First store in a process loads once; every subsequent `Memory()` reuses it. Lock-guarded against concurrent-load torch race.
- **`Memory(...)` ctor: 3 166 ms → 7 ms.** Model loads on first embed, not on construction. `Memory.warmup()` available for eager load.
- **Insert throughput at 10K facts: 237/s → 425/s** (audit-projected degradation to 92/s at 40K no longer happens).
- Retrieval stays bounded: ~13 ms p50 at 10K.

### API additions

```python
m.query_one("where do I live?")     # QueryResult(answer, certainty, needs_clarification, alternatives)
m.warmup()                           # force eager model load
m.forget(hard=True)                  # real DELETE + VACUUM
m.export_to_jsonl(path)              # portable dump
m.import_from_jsonl(path)            # round-trip restore

r.is_suspicious                      # injection classifier flagged it
r.to_llm_context()                   # wraps suspicious in delimiters
r.source_type                        # user_stated / suspicious / ...
```

### CLI

- **`lore chat`** — interactive REPL. Natural-language store and query, slash commands (`/help /list /stats /forget all /export /exit`), certainty-aware answers, warmup hint on first input.

### Benchmarks & regression harness

- `benchmarks/harness.py` — quality (39 seeded queries with ground truth) + perf (1K fake-embed). `--slow` adds 10K. `--check` exits non-zero on >2 pp quality drop or >20 % perf drop.
- `benchmarks/baseline.json` committed — top-1 79.5 %, top-3 89.7 %, retrieval p50 8 ms, insert 542/s at 1K / 425/s at 10K.

### Robustness

- `store(None / "" / 100 KB)` handled cleanly; **32/32 edge cases pass**.
- `query(None / "")` returns `[]`.
- `query_one` sets `needs_clarification=True` when top-1 score < 0.02 (effectively no match).

### Lexicon consolidation

- `lore_memory/lexicons.py` — single canonical copy of English wordlists (pronouns, kinship nouns, stopwords, attribution verbs, commit prefixes, hedge markers). Previously duplicated across extraction.py, extraction_gf.py, normalization.py, retrieval.py, engine.py (up to 5 copies each).

### Docs

- `SHIP_READY.md` — every P0/P1 finding from the audit, with file:line status.
- `SCALE_IMPROVEMENTS.md` — measured before/after of this release.
- `README.md` updated with English-only scope, measured ceilings, shared-scope ACL warning, injection security notes, and schema section. Removed "no dictionaries" and "10M scale" overreach.

### Honest limits (documented, unchanged)

- English-only.
- `shared` scope has no per-row ACL — every member of an org sees every row.
- Aspirational retractions (`"Actually I haven't been to the gym in 3 months"`) still don't auto-correct.
- `~/.lore-memory` default path collision between apps using the same `user_id="default"`.

### Test suite

- 287/287 pass.
- Suite runtime halved (9 min → 4 min) — class-level model singleton.

---

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
