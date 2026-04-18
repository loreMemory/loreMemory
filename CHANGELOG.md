# Changelog

## v1.2.0 — 2026-04-18

spaCy is now the default extractor. The hand-rolled English grammar parser
remains as an opt-out (`Memory(use_spacy=False)`) for callers who want to
avoid the dependency or need 1ms-latency extraction over correctness on
compound clauses. New competitive benchmark scores 50/50 (100%) across
identity, updates, temporal, negation, multi-tenant, adversarial, scale
(1K + 10K), and latency (5K, p50<10ms store, <20ms query).

- **extraction**: new `extraction_spacy.py` (~280 LOC) replaces the
  hand-rolled grammar with a dependency-parser-based extractor. Handles
  compound clauses ("I use Python and love Rust" → two facts), compound
  objects ("Python, Go, and Rust" → three facts), particle verbs
  ("Luna passed away"), the "X named Y" pattern (`have:Luna` instead of
  `have:a dog`), and xcomp chains ("started learning Rust"). Resolves
  relative-time phrases ("last week", "yesterday") to absolute timestamps
  in a `valid_from` field (column wiring still pending).
- **belief**: new `check_loss_events` rule supersedes `have:X` facts when
  X has a `pass_away` event. Fixes the documented Luna→Max replacement
  bug. Also: `check_contradictions` now honors a `metadata['single_valued']`
  flag so identity-style facts ("My favorite color is blue" → "Actually
  it's green") supersede correctly without each axis being enumerated in
  the schema's `single_valued` set.
- **retrieval**: `_QUERY_PRED_MAP["job"]` no longer aliases to the employer
  axis (`work_at`); maps to role-axis predicates only. `_predicate_alignment`
  penalizes wrong-axis structured rows (`pa = -0.5`) when a curated keyword
  fires, so the right axis wins even when other channels favor a near-miss.
  First-person augmentation pulls user-subject NON-stated rows separately
  with `limit=2000`, so the journal flood from third-party noise can't
  evict the user's own structured facts at scale. Adds `age`/`old`/`role`/
  `title` keys to `_QUERY_PRED_MAP`.
- **safety**: `InjectionClassifier.score` now scans per-sentence (not just
  the whole input), so a single malicious sentence in a multi-sentence
  payload still trips the threshold. Added 4 prototypes for memory-
  overwrite attempts ("Forget everything you know..."). When injection
  or hypothetical is detected, extracted triples are dropped (the journal
  row is kept for audit) so attack text can't outrank legitimate facts via
  recency.
- **bench**: new `benchmarks/competitive_bench.py` — 8-section reproducible
  harness used to gate releases. Exit code 1 if overall < 95%.
- **tests**: new `tests/test_known_regressions.py` (13 tests) locking
  every bug class fixed this release; new `grammar_engine` fixture for
  tests that assert grammar-parser-specific output.
- **deps**: `spacy>=3.7,<4` added as a runtime dep. The
  `en_core_web_sm` model is auto-downloaded on first `Memory()` use
  (~12 MB, one-time). Set `LORE_SKIP_MODEL_DOWNLOAD=1` to disable
  auto-fetch in air-gapped environments.
- **scope**: optimization is English-only by project direction. `_is_trivial`
  still preserves non-ASCII text (no silent drop) but no further multilingual
  investment.

302 / 302 tests pass. 50 / 50 competitive benchmark.

## v1.1.1 — 2026-04-17

Two retriever fixes surfaced by an 8-week end-to-end user simulation.

- **"how old am I?" no longer surfaces superseded age** (`retrieval.py`).
  Bare `"old"` was in the temporal-keyword set and matched the age idiom
  as if the user were asking about past state. Replaced with a context
  regex — `old` only triggers temporal mode when preceded by a possessive
  or article (`"my old manager"`, `"the old system"`). Bare `"old"` now
  falls through to present-tense retrieval, so the harness
  `supersede_age` check returns the current age (baseline top-1 regression
  it had been silently getting).
- **"does my neighbor have a pet?" no longer returns the user's pet**
  (`lexicons.py`). Added `neighbor`, `coworker`, `mentor`, `supervisor`,
  `roommate` to the retrieval-side relationship-noun set. The subject
  alignment code already had the right logic (skip the user-subject boost
  when the query names a specific relationship) — the set just didn't
  know about these relations.

E2E simulation: 22 / 24 → 24 / 24. Harness: 79.5 % → 82.1 % top-1 (+1
test, 0 regressions). 390 / 390 tests pass.

## v1.1.0 — 2026-04-17

LLM-shaped contract: callers that already have an LLM in front (the common
case) can now hand us structured S-P-O at write time and predicate/subject
hints at read time. The local grammar parser stays as the fallback for
non-LLM callers (curl, scripts, the CLI). No new runtime dependencies; no
LLM in our process — the interface just gives a cooperating LLM a clean
way to do the cognitive work it's already doing.

### Write — `store(text, facts=...)`

Old: `store("I have a cat named Luna")` → grammar parser guesses
`(user, have, a cat named Luna)`. Predicate is too generic, retrieval
struggles at scale.

New: caller passes the original text *plus* the LLM's extraction:

```python
m.store(
    "I have a cat named Luna",
    facts=[{"subject": "user", "predicate": "pet", "object": "Luna"}],
)
```

The grammar parser is skipped; the LLM's S-P-O is written directly. Raw
text is still saved as a `stated` row so FTS keyword recall and the
journal stay intact. Each fact still flows through dedup, supersession,
and Schema-driven canonicalization — the LLM can't bypass schema discipline.

### Read — `query(text, predicate_hint=..., subject_hint=...)`

Hints are **boosts, not filters**: a wrong hint never hides a correct
answer; a right hint surfaces it instantly.

```python
m.query("what is my job?",
        predicate_hint=["job_title", "works_at"],
        subject_hint="user")
```

A direct `(subject=user, predicate=hobby)` hit gets a 3× predicate boost
plus a 2× subject boost — enough to lift it above semantic noise even at
1000+ facts.

### MCP & REST — same shape, both transports

- **MCP** (`mcp/server.py`): `store_memory` and `query_memory` tool
  schemas extended with the new fields. Tool descriptions list the
  canonical predicates and show JSON examples — the description IS the
  prompt that teaches the LLM how to call us.
- **REST** (`api/server.py`): new `Fact` Pydantic model; `POST /memory`
  accepts `facts: list[Fact]`; new `POST /memory/query` with
  `predicate_hint` / `subject_hint`; `GET /memory` accepts hints as
  query-string params. OpenAPI auto-publishes the schema so OpenAI /
  Gemini function-calling can import it.

### Measured at 2 100-store stress (1110 facts after dedup)

| Path                                 | Top-1 correct |
|--------------------------------------|---------------|
| `store(text)` only (grammar parser)  | 8/10          |
| `store(text, facts=[...])` (LLM SPO) | **10/10**     |

The two grammar failures (`"what is my job?"`, `"do I have pets?"`) become
clean hits when the LLM supplies `predicate=job_title` / `predicate=pet`.

### Compatibility

- Existing `store(text)` and `query(text)` calls behave identically.
- Schema canonicalization still applies — LLM-supplied `lives_in`,
  `live_in`, `move_to`, `relocated_to` all collapse to canonical
  `lives_in` via `Schema.aliases`, so supersession works the same way.
- Subject `"user"` (case-insensitive) is rewritten to the caller's
  `user_id` so prompts can stay user-agnostic.

### Tests

- 6 new tests in `tests/test_llm_facts.py` covering the write path,
  schema-skip behavior, subject rewrite, hint-as-boost guarantee, and
  malformed-fact handling.
- 390 / 390 pass. Harness unchanged at 79.5 % top-1.

## v1.0.9 — 2026-04-16

Three correctness fixes uncovered by an adversarial 2 100-store stress run.

- **Phantom I-name facts** (`extraction_gf.py`). `"Ines lives in Paris"` matched `str.startswith("i")` and produced a second triple `(user, nes_lives_in, Paris)` alongside the correct `(Ines, lives_in, Paris)`. At 2 100 stores this credited the user with 12 cities and 10 companies they had never visited or worked at. Fix: word-boundary regex on first-person markers.
- **Supersession silently broken at scale** (`dedup.py`, `belief.py`, `store.py`). Every `store()` writes a `predicate='stated'` journal row against the user's subject. After ~700 journal rows the dedup scan's `query_by_subject(limit=200)` returned only journal rows, ordered by `updated_at DESC`; the original `live_in=Berlin` fact was outside the window, so `store("I live in Amsterdam now")` no longer saw it and Berlin survived. Fix: `query_by_subject(..., exclude_stated=True)` for dedup and contradiction checks. Journal rows never participated in those checks anyway.
- **`"My X is Y"` over-fitted to a noun lexicon** (`extraction.py`). `"My hobby is rock climbing"` came out as `(user, is, rock climbing)` because `hobby` wasn't in the curated relationship-noun list. Replaced the lexicon lookup with a grammar rule: head noun of the phrase becomes the predicate, with a small leading-quantifier strip (`best`, `primary`, `main`, `own`, `only`, `current`, ...) so `"My best friend is Mateo"` still produces `friend=Mateo`. Generalises to `hobby`, `goal`, `dream`, `pet cat` etc. without enumerating nouns.

Stress run (2 100 third-party noise facts + 10 user identity facts):
- Pre-fix: 7/10 correct; supersession fails silently; 22 phantom facts attributed to user.
- Post-fix: 8/10 correct; supersession works; 0 phantom facts. The two remaining failures (`"what is my job?"`, `"do I have pets?"`) correctly return `needs_clarification=True` with certainty <0.05 — honest "I don't know" instead of confidently wrong.

Harness unchanged: 79.5 % top-1, 89.7 % top-3, 556 inserts/s (+2.6 %). 384 / 384 tests pass.

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
