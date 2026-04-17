<div align="center">

<br>

<img alt="LoreMem — Persistent Memory for AI Agents" src="https://raw.githubusercontent.com/loreMemory/loreMemory/main/.github/assets/hero.svg" width="100%">

<br>

[![PyPI](https://img.shields.io/pypi/v/loremem-ai?style=flat-square&color=5A5BFF&logo=pypi&logoColor=white)](https://pypi.org/project/loremem-ai/)
&nbsp;&nbsp;
[![Downloads](https://img.shields.io/pypi/dm/loremem-ai?style=flat-square&color=29CB5B&logo=python&logoColor=white)](https://pypi.org/project/loremem-ai/)
&nbsp;&nbsp;
[![Python](https://img.shields.io/pypi/pyversions/loremem-ai?style=flat-square&logo=python&logoColor=white&color=3776AB)](https://pypi.org/project/loremem-ai/)
&nbsp;&nbsp;
[![CI](https://img.shields.io/github/actions/workflow/status/loreMemory/loreMemory/ci.yml?style=flat-square&label=tests&logo=githubactions&logoColor=white)](https://github.com/loreMemory/loreMemory/actions/workflows/ci.yml)
&nbsp;&nbsp;
[![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square&logo=opensourceinitiative&logoColor=white)](LICENSE)

</div>

<br>

<div align="center">
<img src="https://raw.githubusercontent.com/loreMemory/loreMemory/main/.github/assets/divider.svg" width="100%">
</div>

<br>

## 1. Install

```bash
pip install loremem-ai
```

That's it. Python 3.9+. Includes `sentence-transformers` for semantic search.

<br>

## 2. Use

```python
from lore_memory import Memory

m = Memory()
m.store("I live in Amsterdam and work at Google")
m.store("I love Python and hate Java")

m.query("where do I work?")  #> Google (conf=0.867)

m.store("I moved to Berlin")
m.query("where do I live?")  #> Berlin — Amsterdam auto-superseded
```

<br>

## 3. Connect to your AI tool

One config. Works with Claude, Cursor, Windsurf, or any MCP client.

```json
{
  "mcpServers": {
    "lore-memory": {
      "command": "python3",
      "args": ["/path/to/lore-memory/mcp/server.py"]
    }
  }
}
```

| Tool | Where to put it |
|:-----|:----------------|
| **Claude Desktop** | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| **Claude Code** | `.mcp.json` in project root |
| **Cursor** | `.cursor/mcp.json` in project root |
| **Windsurf** | `~/.codeium/windsurf/mcp_config.json` |

Your AI now remembers everything across conversations.

<br>

<div align="center">
<img src="https://raw.githubusercontent.com/loreMemory/loreMemory/main/.github/assets/divider.svg" width="100%">
</div>

<br>

## Why LoreMem

<table>
<tr>
<td align="center" width="33%">
<br>
<img src="https://img.shields.io/badge/local-first-5A5BFF?style=for-the-badge" alt="local-first">
<br><br>
<b>Local-First</b><br>
<sub>SQLite + sentence-transformers.<br>No API keys. No cloud. No cost.</sub>
<br><br>
</td>
<td align="center" width="33%">
<br>
<img src="https://img.shields.io/badge/no-LLM_at_write-E8425F?style=for-the-badge" alt="no LLM at write">
<br><br>
<b>English Grammar Extraction</b><br>
<sub>Positional parser for English.<br>No LLM required at write time.<br>English-only scope.</sub>
<br><br>
</td>
<td align="center" width="33%">
<br>
<img src="https://img.shields.io/badge/7-channels-FF6B35?style=for-the-badge" alt="7 channels">
<br><br>
<b>Self-Learning</b><br>
<sub>7 retrieval channels adapt via<br>feedback and Hebbian learning.</sub>
<br><br>
</td>
</tr>
<tr>
<td align="center" width="33%">
<br>
<img src="https://img.shields.io/badge/%3C50ms-retrieval-29CB5B?style=for-the-badge" alt="<50ms">
<br><br>
<b>Fast</b><br>
<sub>~10–20ms retrieval to 10K facts.<br>In-process; no network.</sub>
<br><br>
</td>
<td align="center" width="33%">
<br>
<img src="https://img.shields.io/badge/filesystem-isolation-8B5CF6?style=for-the-badge" alt="isolation">
<br><br>
<b>Private-Scope Isolation</b><br>
<sub>One SQLite file per user.<br>Shared scope has no row-level ACL.</sub>
<br><br>
</td>
<td align="center" width="33%">
<br>
<img src="https://img.shields.io/badge/fully-offline-6B7280?style=for-the-badge" alt="offline">
<br><br>
<b>Offline</b><br>
<sub>Everything local. No telemetry.<br>Your data never leaves.</sub>
<br><br>
</td>
</tr>
</table>

<br>

<div align="center">

|  | **LoreMem** | Cloud alternatives |
|:--|:----:|:----:|
| Requires LLM at write | **No** | Yes |
| Cost | **Free** | $19–249/mo |
| Works offline | **Yes** | No |
| Extraction | English positional grammar | LLM-dependent |
| Language scope | English only | Multilingual |
| Private isolation | Filesystem (file-per-user) | API-level |

</div>

<br>

<div align="center">
<img src="https://raw.githubusercontent.com/loreMemory/loreMemory/main/.github/assets/divider.svg" width="100%">
</div>

<br>

## How It Works

<div align="center">
<table>
<tr>
<td align="center" width="33%">
<img src="https://img.shields.io/badge/1-Store-5A5BFF?style=for-the-badge&labelColor=1a1a2e" alt="Store">
<br><sub>Grammar extraction</sub>
</td>
<td align="center" width="33%">
<img src="https://img.shields.io/badge/2-Recall-29CB5B?style=for-the-badge&labelColor=1a1a2e" alt="Recall">
<br><sub>7-channel retrieval</sub>
</td>
<td align="center" width="33%">
<img src="https://img.shields.io/badge/3-Learn-FF6B35?style=for-the-badge&labelColor=1a1a2e" alt="Learn">
<br><sub>Adaptive improvement</sub>
</td>
</tr>
</table>
</div>

<br>

<details>
<summary><b>Store</b> — text in, structured facts out</summary>
<br>

```
  "I live in Amsterdam and work at Google"
           │                        │
           ▼                        ▼
   (user, live_in, Amsterdam)    (user, work_at, Google)
```

Parses **English** by grammar position — pronouns, copulas, prepositions, and a few irregular verbs are recognised; the rest is structural. Raw text is always FTS5-indexed as a fallback. A `sentence-transformer` is loaded for retrieval and write-time safety classification, not for extraction.

</details>

<details>
<summary><b>Recall</b> — 7 scoring channels, fused into one ranked result</summary>
<br>

| Channel | What it does |
|:--------|:------------|
| Semantic | Cosine similarity (embeddings) |
| Keyword | BM25-style term overlap (FTS5) |
| Temporal | Exponential recency decay |
| Belief | Bayesian posterior (evidence + contradictions) |
| Frequency | Log-scaled access count |
| Graph | Spreading activation, 3-hop |
| Resonance | Co-activation frequency |

Weights adapt automatically through feedback.

</details>

<details>
<summary><b>Learn</b> — gets better the more you use it</summary>
<br>

```python
m.feedback(results[0].id, helpful=True)   # adapt channel weights
m.consolidate()                           # decay + replay + archive
```

| Mechanism | Effect |
|:----------|:-------|
| Adaptive weights | Channels shift toward what works |
| Hebbian synapses | Co-retrieved facts strengthen links |
| Memory replay | Active memories resist decay |
| Ebbinghaus forgetting | Unused facts fade over time |
| Contradiction resolution | New facts supersede old ones |

</details>

<br>

<div align="center">
<img src="https://raw.githubusercontent.com/loreMemory/loreMemory/main/.github/assets/divider.svg" width="100%">
</div>

<br>

## Benchmarks

<sub>Actual runs on Apple M-series, Python 3.9. Reproduce: `python benchmarks/lore_bench.py`</sub>

<table>
<tr>
<td width="50%" valign="top">

**Test Suite** — 138 tests

| Capability | Pass |
|:-----------|-----:|
| Correction chains | 10/10 |
| Negation & retraction | 5/8 |
| Memory decay | 5/5 |
| Self-learning | 2/2 |
| User isolation | 100/100 |
| Grammar extraction | 10/10 |
| Scale & latency | 3/3 |
| **Overall** | **135/138** |

</td>
<td width="50%" valign="top">

**Latency** — per operation

| Facts stored | Recall p50 | Write |
|------:|:----------:|:-----:|
| 100 | 14ms | 8.4ms |
| 1,000 | 21ms | 7.9ms |
| 5,000 | 33ms | 7.7ms |
| 10,000 | 50ms | 7.6ms |

<sub>Hash embeddings. Real embeddings add ~7ms/write.</sub>

</td>
</tr>
</table>

> [!NOTE]
> Negation detection (62%) is a known limitation. Phrases like *"I can't stand X"* and *"I stopped doing X"* are not yet reliably parsed.

<br>

<div align="center">
<img src="https://raw.githubusercontent.com/loreMemory/loreMemory/main/.github/assets/divider.svg" width="100%">
</div>

<br>

## Scope & Limits

**Language.** English only. The grammar is a positional parser for English; other languages either fail to parse or fall back to FTS5 raw-text indexing.

**Scale tested.** ~1M facts per tenant on a laptop SSD. Write throughput degrades super-linearly with corpus size (measured: ~650/s at 1K, ~90/s at 40K). Retrieval stays bounded (~10ms p50 at 10K). See `benchmarks/` for reproducibility. Larger scales are not yet validated.

**Multi-tenant.** `private` scope is isolated at the filesystem layer: one SQLite file per user. `shared` (org) scope does **not** have per-row ACL today — every member of an org sees every row. Not safe for cross-employee segmentation without an additional check at the application layer.

**Security.** Every `store()` passes through a prompt-injection classifier. Suspicious text is flagged `source_type="suspicious"`, excluded from `profile_compact()`, and wrapped in `<user_stated_untrusted>…</user_stated_untrusted>` delimiters via `MemoryResult.to_llm_context()`. Your LLM system prompt is expected to recognise the delimiters as data, not instructions. The classifier is a sentence-transformer prototype match plus an HTML/XML-tag short-circuit — tune the threshold or disable via `Config(injection_defense=False)` for trusted batch-ingest pipelines.

**Identity schema.** Which predicates can be superseded, how they alias, and how fast they decay is per-tenant. The default `PERSONAL_LIFE_SCHEMA` matches prior behavior. `CARE_TRACKING_SCHEMA` adds `on_medication`, `dose`, `has_diagnosis`, `appointment_on` as single-valued. `RESEARCH_NOTES_SCHEMA` keeps claims multi-valued and zero-decay. Pass via `Memory(..., schema=CARE_TRACKING_SCHEMA)`. The schema hash is persisted per DB; opening with a different schema logs a warning but is not blocked.

**Hypothetical and reported speech.** Conditional / hedged inputs (*"If I get the offer, I'll move to London"*, *"maybe I'll quit next year"*) are flagged `hypothetical=True` in metadata, stored at lower confidence, and **do not supersede factual memories**. Reported speech (*"My wife said we should move to Tokyo"*) is attributed: the parsed speaker becomes the fact's subject and `source_speaker` is recorded in metadata — the fact never overwrites the user's own. Both kinds remain retrievable at reduced retrieval weight. Disable via `Config(hypothetical_detection=False)` for trusted-input pipelines.

**No cloud, no telemetry.** Data never leaves the process. Backups are your responsibility.

<br>

<div align="center">
<img src="https://raw.githubusercontent.com/loreMemory/loreMemory/main/.github/assets/divider.svg" width="100%">
</div>

<br>

## API Reference

<details>
<summary><b>Core API</b></summary>
<br>

```python
m = Memory(user_id="alice", org_id="acme", data_dir="~/.lore-memory")

m.store(text, scope="private")             # Store from natural language
m.query(query, limit=10)                   # 7-channel retrieval
m.forget(memory_id=...)                    # Soft-delete by ID
m.forget(subject="alice", hard=True)       # GDPR-style hard erase + VACUUM
m.forget_all(hard=True)                    # Remove entire user DB file
m.export_all()                             # Every row (incl. deleted) as dicts
m.export_to_jsonl("alice.jsonl")           # Portable audit dump
m.close()                                  # Persist and close

# Retrieval results that flowed through the injection classifier:
for r in m.query("who am I?"):
    r.is_suspicious          # True if write-time classifier flagged it
    r.to_llm_context()       # wraps suspicious results in untrusted delimiters

# Single-answer query with a margin-based certainty signal:
r = m.query_one("where do I live?")
if r.needs_clarification:
    ask_user(r.alternatives)   # top-1 is within 15% of top-2
else:
    use(r.answer.text)         # certainty: fraction top-1 leads top-2
```

</details>

<details>
<summary><b>Advanced API</b></summary>
<br>

```python
m.store_triple("alice", "works_at", "Google", confidence=0.9)
m.profile()                            # All facts by predicate
m.profile_compact(max_tokens=200)      # Token-budgeted LLM context
m.feedback(memory_id, helpful=True)    # Drive adaptive learning
m.consolidate()                        # Decay + replay + archive
m.stats()                              # Memory counts by scope
```

</details>

<details>
<summary><b>Context manager</b></summary>
<br>

```python
with Memory(user_id="alice") as m:
    m.store("I live in Amsterdam")
    results = m.query("where do I live?")
```

</details>

<details>
<summary><b>Custom embeddings</b></summary>
<br>

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")

m = Memory(user_id="alice", embedding_dims=384, embed_fn=model.encode)
```

</details>

<details>
<summary><b>Multi-user isolation</b></summary>
<br>

```python
alice = Memory(user_id="alice")
bob   = Memory(user_id="bob")

alice.store("I work at Google")
bob.query("where does alice work?")  #> [] — fully isolated
```

Shared org memories:

```python
alice = Memory(user_id="alice", org_id="acme")
alice.store("Our mission is to democratize AI", scope="shared")

bob = Memory(user_id="bob", org_id="acme")
bob.query("what is our mission?")  #> Returns shared memory
```

</details>

<details>
<summary><b>CLI</b></summary>
<br>

```bash
lore store "I work at Google"
lore query "where do I work?"
lore list
lore stats
lore forget --id <id>
lore serve --port 8420     # REST API
lore mcp                   # MCP server
```

</details>

<details>
<summary><b>REST API</b></summary>
<br>

```bash
pip install loremem-ai[api]
lore serve --port 8420

# Store
curl -X POST localhost:8420/memory \
  -H "Content-Type: application/json" \
  -d '{"user_id":"alice","text":"I prefer dark mode"}'

# Query
curl "localhost:8420/memory?user_id=alice&query=preferences"
```

</details>

<details>
<summary><b>Docker</b></summary>
<br>

```bash
docker build -t loremem -f docker/Dockerfile .
docker run -p 8420:8000 -v lore_data:/data loremem
```

</details>

<br>

<div align="center">
<img src="https://raw.githubusercontent.com/loreMemory/loreMemory/main/.github/assets/divider.svg" width="100%">
</div>

<br>

<div align="center">

## Contributing

Contributions welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

```
git clone https://github.com/loreMemory/loreMemory.git && cd loreMemory
pip install -e ".[dev]" && pytest tests/ -v
```

<br>

[Security](SECURITY.md) &nbsp;·&nbsp; [Changelog](CHANGELOG.md) &nbsp;·&nbsp; [License](LICENSE)

**MIT** — free for personal and commercial use.

<br>

</div>
