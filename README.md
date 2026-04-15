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

That's it. No dependencies. Python 3.9+.

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
<img src="https://img.shields.io/badge/0-dependencies-5A5BFF?style=for-the-badge" alt="0 deps">
<br><br>
<b>Zero Dependencies</b><br>
<sub>Python stdlib + SQLite. No API keys.<br>No cloud. No external services.</sub>
<br><br>
</td>
<td align="center" width="33%">
<br>
<img src="https://img.shields.io/badge/no-LLM_needed-E8425F?style=for-the-badge" alt="no LLM">
<br><br>
<b>Grammar Extraction</b><br>
<sub>Parses by sentence structure.<br>No regex. No dictionaries. No LLM.</sub>
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
<sub>Sub-50ms at 10K facts.<br>~20ms at 1K. No network calls.</sub>
<br><br>
</td>
<td align="center" width="33%">
<br>
<img src="https://img.shields.io/badge/100%25-isolation-8B5CF6?style=for-the-badge" alt="isolation">
<br><br>
<b>User Isolation</b><br>
<sub>Separate database per user.<br>Zero data leakage.</sub>
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
| Requires LLM | **No** | Yes |
| Cost | **Free** | $19–249/mo |
| Works offline | **Yes** | No |
| Extraction | Grammar-based | LLM-dependent |
| Self-learning | 7 mechanisms | Limited |
| User isolation | Physical (file-per-user) | API-level |

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

Parses English by **grammar position**. No verb dictionaries, no regex, no LLM. Raw text is always FTS5-indexed as a fallback.

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

## API Reference

<details>
<summary><b>Core API</b></summary>
<br>

```python
m = Memory(user_id="alice", org_id="acme", data_dir="~/.lore-memory")

m.store(text, scope="private")         # Store from natural language
m.query(query, limit=10)               # 7-channel retrieval
m.forget(memory_id=...)                # Delete by ID
m.forget(subject="alice")              # Delete by subject
m.forget_all()                         # Purge all user data
m.close()                              # Persist and close
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
