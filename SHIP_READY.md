# Ship-readiness — loremem-ai

Status of the product against the audit we ran earlier (see `/tmp/lore_test/audit/CRITIQUE.md`). Every claim here is either measured (file:line, benchmark number) or called out as out-of-scope. No hedging.

Audit date: 2026-04-16. This document: same session, post-Phase-1-and-2 work.

---

## Headline

| Axis | Before audit | After Phase 1+2 |
|---|---|---|
| Test suite | 287 passing | **287 passing** |
| Harness top-1 | 69.8% (different test set) | **76.9%** *(new harness with per-check ground truth)* |
| Harness top-3 | 84.1% | 82.1% *(new harness, different queries)* |
| Harness retrieval p50 | 9.9 ms | **10.1 ms** |
| Insert throughput (1K, fake embed) | 887/s | **504/s** *(28% feature cost from injection defense + 21% from hypothetical detection; still 5× above ship floor)* |
| Injection probe | 0/12 rejected — every injection leaked | **12/12 handled** (11 flagged, 1 not-stored, 0 leaked) |
| Profile allowlist | Silently hid user's own predicates | **Removed** — all predicates surface |
| GDPR hard delete | Not available | **Implemented** (`forget(hard=True)`, `forget_all(hard=True)`, `export_all()`, `export_to_jsonl()`) |
| Per-tenant schema | Module globals — one schema for everyone | **`Schema` dataclass** + 3 presets, threaded through Memory |
| Hedges / conditionals stored as facts | "If I get the offer I'll move to London" → `move_to=London` as fact | **Flagged `hypothetical=True`**, posterior halved, skipped from supersession, retrieval × 0.4 |
| Reported speech | "My wife said we should move to Tokyo" → user's move fact | **Attributed**: subject rewritten to "My wife", `source_speaker` tagged, retrieval × 0.6 |
| Retrieval trust signal | Overlapping score distributions | **Margin-based certainty + `needs_clarification` hint** via `query_one()` |

Bottom line: every P0 and P1 finding from CRITIQUE.md is addressed or explicitly scoped out. Scale findings (P3 in the roadmap) are deliberately deferred until there are users for that work.

---

## Audit finding status — all 15 items

### P0 — shipped / scoped

**1. `_QUERY_PRED_MAP` is a builder-vocabulary dictionary doing the semantic-bridge work.** *Fixed — semantic-primary.* `retrieval.py:593` `_predicate_alignment` now runs predicate-embedding cosine for every candidate as the **base** signal; the curated map is kept as an opt-in precision override for canned patterns. Out-of-map vocabulary now gets a real score instead of falling through. *Harness quality unchanged on in-map queries (76.9%); out-of-map improvement is not measured by the current harness (its vocabulary is in-map). Follow-up: add an out-of-map harness in Phase 3.*

**2. Identity schema is a module global.** *Fixed.* New `lore_memory/schema.py`. `belief.py` reads from `PERSONAL_LIFE_SCHEMA` (default) or a caller-supplied `Schema`. Threaded through `Engine(schema=…)` / `Memory(schema=…)`. Three shipped presets: `PERSONAL_LIFE_SCHEMA` (default, exact prior behavior), `CARE_TRACKING_SCHEMA` (adds `on_medication`, `dose`, `has_diagnosis`, `appointment_on` as single-valued), `RESEARCH_NOTES_SCHEMA` (zero-decay, multi-valued claims). Schema hash persisted per DB; mismatch logs a warning. *Confirmed: caregiver preset makes `on_medication` supersede; personal preset keeps both values.*

**3. Write throughput degrades super-linearly with corpus size.** *Scoped out.* Dedup flattening (the free 4× win) is Phase 3 Step 11. Roadmap explicitly deferred it until scale demand is real. Current per-insert cost is acceptable at the documented ~1M ceiling.

**4. Prompt injection via stored memory was the default.** *Fixed.* New `lore_memory/safety.py` + `InjectionClassifier`. Three layers:
- Write-time: `source_type="suspicious"` tagging on semantic+structural match.
- Output layer: `profile_compact()` skips suspicious rows.
- Read layer: `MemoryResult.is_suspicious` + `.to_llm_context()` wraps suspicious text in `<user_stated_untrusted>…</user_stated_untrusted>`.
- Opt-out via `Config(injection_defense=False)` for trusted batch pipelines.
- *Verified: 11/12 injection strings flagged at write, 1/12 produced zero rows (couldn't leak), 0 leaked, 0 false positives on 10 legit statements.*

**5. SQLite-file-per-tenant breaks at 10K tenants.** *Scoped out.* Tenant-resolver interface is Phase 3 Step 14. The current LRU=100 connection cache is fine for under ~1K tenants. Anyone crossing that scale needs a sharded layout — documented in `README.md:Scope & Limits`.

**6. Profile allowlist silently hides user's own facts.** *Fixed.* `engine.py:_is_profile_worthy` keeps only structural garbage filters (length, vowels, commit-regex, object sanity). Vocabulary allowlist deleted. *Verified: `plays_cello`, `has_mortgage`, `sees_therapist`, `runs_weekly`, `owns_domain` all appear in `profile()` and `profile_compact()` without code changes.*

**7. Supersession grammar is verb-pattern narrow.** *Partially fixed.* In-line grammar fixes for `move_from/to`, `left/joined`, age (`extraction.py` Phase-1-Fix-1/2/4). Broader coverage is bounded by the parser architecture; the remaining gap is a known limitation. More importantly, the grammar's narrowness is no longer a trust issue because **hedges and attributions no longer supersede at all** (Phase 2 Step 8).

**8. `confidence=0.9` is a default, not a measurement.** *Fixed (read-side).* New `Memory.query_one()` returns a `QueryResult` with `.certainty` (margin of top-1 over top-2, in `[0,1]`) and `.needs_clarification` (true when certainty < 0.15). A downstream LLM can now threshold on certainty. Write-side `confidence` defaults are tuned per extractor (0.75 grammar, 0.6–0.7 grammar-free, 0.95 system) — not touched.

### P1 — shipped

**9. Aspirational / contradictory user statements don't self-correct.** *Open.* Unchanged. `"I go to the gym every day"` followed by `"Actually I haven't been in 3 months"` still returns the first as top-1. The `Actually` prefix is stripped at the parser level but the resulting fact has a different predicate (`have/not` vs `go_to`), so supersession doesn't fire. Honest status: this needs a specific "retraction pattern" detector that I did not build. Known gap, documented.

**10. Conditionals and hedges stored as facts.** *Fixed — this is the big one.* `HypotheticalClassifier` in `safety.py` (hybrid prefix regex + sentence-transformer prototype cosine, threshold 0.45). Flagged memories: `metadata["hypothetical"]=True`, confidence halved, supersession skipped, retrieval × 0.4. Attribution via `detect_speaker(subject)`: `"My wife said we should move to Tokyo"` → `subject="My wife"`, `source_speaker="My wife"`, retrieval × 0.6. *Verified: 11/11 hedges caught, 0 false positives on hard cases including `"I'm learning Italian"`. End-to-end: "where do I live?" returns "Amsterdam" (0.226) vs "London conditional" (0.043) — 5× margin.*

**11. Graph cold-cache at scale — untested.** *Open — scoped out.* Phase 3 Step 13 covers binary graph persistence. At current scales the warm-cache path is fine (79 ms restart holds). Cold-path under corruption not tested at 10M+.

**12. Retrieval per-query materializes up to 100 memories.** *Open — scoped out.* Not the bottleneck below 10K facts. Phase 3 concern.

**13. Dedup is correctness-critical and blocks writes.** *Scoped out.* See #3. Phase 3 Step 11.

**14. `forget` by predicate is potentially silent at large N.** *Fixed.* `delete_hard_by_subject(subject, predicate)` erases every row regardless of state (including soft-deleted), then VACUUMs. `Memory.forget(subject=…, hard=True)` wires to this. *Verified end-to-end: 0 rows remain after GDPR erasure.*

**15. Single shared `~/.lore-memory` default causes cross-app leakage.** *Open — documentation.* The default path is unchanged (changing it is a breaking change). README now documents the risk explicitly under "Scope & Limits": two apps with the same `user_id="default"` and no `data_dir` share a store. Users expected to pass `data_dir` per app.

---

## What ships behind each flag

| Feature | Constructor | Opt-out |
|---|---|---|
| Injection defense | `Config.injection_defense = True` (default) | Set to `False` for trusted ingest |
| Hypothetical detection | `Config.hypothetical_detection = True` (default) | Set to `False` for trusted ingest |
| Schema | `Memory(schema=...)` | Default = `PERSONAL_LIFE_SCHEMA` |

---

## Known limitations (shipped honest)

The `README.md:Scope & Limits` section calls these out for external readers; duplicated here for completeness.

- **Language.** English only. Mandarin / Japanese inputs return `None` from the grammar; Arabic / Hindi fall back to raw-text FTS only. Non-English is out of scope by product choice.
- **Scale ceiling.** ~1M facts per tenant. Write throughput degrades super-linearly above ~10K (~500/s → ~20/s projected at 1M). Retrieval stays bounded.
- **Shared-scope isolation.** Per-row ACL is **not** implemented on `shared` scope. Every member of an org sees every row. This is a P0 blocker for multi-user team products. Documented.
- **Aspirational retractions.** "Actually I haven't been to the gym in 3 months" does not supersede "I go to the gym every day". Retraction-pattern detection is not built.
- **`~/.lore-memory` default path.** Two apps sharing the path + default `user_id` will see each other's facts. Always pass explicit `data_dir` and `user_id`.
- **Every injection string is stored.** Flagged, not refused. The audit log is intact; the LLM-context surface is clean. This is a deliberate tradeoff — a deletion policy could be added later.
- **Conditional hedges can still produce low-ranked noise.** The classifier flags them so they don't dominate; they remain retrievable at reduced weight. A user who searches for "what was my conditional plan?" can still get it. If that's wrong for your use case, set `Config(hypothetical_detection=False)` and build your own filter.

---

## What I didn't do, by design

- **No Phase 3 (scale, perf harness, ANN, tenant resolver, ACL).** Roadmap explicitly marked these "only once you have users who need it." Doing them now is speculation without ground truth.
- **No changes to stored `confidence` defaults.** The extractors already emit 0.6–0.95 tuned per source type; re-tuning without evidence would be a regression risk.
- **No new multi-language support.** English-only is the product scope you clarified; broadening it now would undercut the other fixes.
- **No changes to `~/.lore-memory` default path.** Breaking change. Documented instead.

---

## Three readiness tests (roadmap gates)

**1. Adversarial test.** Re-run `/tmp/lore_test/adversarial.py` — **passes**. Injection 12/12 safe, isolation holds, GDPR erasure complete, profile output clean. The aspirational-retraction gap is explicitly documented. Conditionals and attribution no longer store as factual top-1.

**2. Trust test.** Not run. Would require a non-developer friend over a week. Prerequisite for genuine product-market signal.

**3. Scale test.** Not re-run. Phase 3 gate. Current numbers are the measured 1K/10K/40K data already in `SCALE_AUDIT.md`.

---

## How to regress-test any future change

```bash
# Quality + perf in one pass; exits non-zero on regression > 2pp / 20%.
python3 benchmarks/harness.py --check

# Update baseline after an intentional change:
python3 benchmarks/harness.py --save-baseline

# Full test suite (8-9 minutes):
python3 -m pytest tests/ -q -k "not slow"
```

Baseline as of this session: **top-1 76.9%, top-3 82.1%, retrieval p50 10 ms, insert 504/s** (1K, fake-embed). All deltas above ±20% on insert or ±2pp on quality are regressions unless the baseline is explicitly updated.
