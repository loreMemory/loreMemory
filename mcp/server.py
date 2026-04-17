"""
Lore Memory MCP Server — stdio-based for Claude Code / VS Code / Claude Desktop.

This server communicates via JSON-RPC over stdin/stdout.
No external MCP package needed — implements the protocol directly.

Add to Claude Code settings:
    "mcpServers": {
        "lore-memory": {
            "command": "python",
            "args": ["/path/to/lore-memory/mcp/server.py"]
        }
    }
"""

from __future__ import annotations

import json
import sys
import os

# Add parent directory so lore_memory is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lore_memory.memory import Memory

# Global memory instance
_memory: Memory | None = None


def _get_memory() -> Memory:
    global _memory
    if _memory is None:
        _memory = Memory()
    return _memory


# --- Tool definitions ---

TOOLS = [
    {
        "name": "store_memory",
        "description": (
            "Save a memory. ALWAYS pass the user's original text in `text`. "
            "If you can identify clear subject-predicate-object triples in "
            "the text, ALSO pass them in `facts` — this gives much better "
            "recall later than letting the local grammar parser guess.\n\n"
            "Use canonical predicates whenever possible:\n"
            "  Identity: name, age, birthday, email, phone, nationality, "
            "native_language\n"
            "  Location: lives_in, born_in, from\n"
            "  Work: works_at, job_title, role, manages, reports_to\n"
            "  Relations: partner, spouse, sister, brother, mother, father, "
            "child, friend, manager, colleague, neighbor, pet\n"
            "  Preferences: likes, loves, dislikes, hates, prefers, favorite_*\n"
            "  Activities: hobby, plays, drives, owns, learning, studied\n"
            "  Health: allergic_to, on_medication, has_diagnosis\n\n"
            "Subject is usually \"user\" (the person speaking). For facts "
            "about other people, use their name (\"Sarah\", \"my neighbor\").\n\n"
            "Example: text=\"I have a cat named Luna and I love climbing\", "
            "facts=[{\"subject\":\"user\",\"predicate\":\"pet\",\"object\":\"Luna\"},"
            "{\"subject\":\"user\",\"predicate\":\"hobby\",\"object\":\"climbing\"}]"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Original user utterance — always required"},
                "facts": {
                    "type": "array",
                    "description": "Optional LLM-extracted S-P-O triples. Skip the local grammar parser when supplied.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "subject":   {"type": "string", "description": "\"user\" or a named entity"},
                            "predicate": {"type": "string", "description": "Canonical relation (lives_in, pet, hobby, ...)"},
                            "object":    {"type": "string", "description": "The value"},
                            "confidence":{"type": "number", "description": "0.0–1.0 (default 0.9)"},
                            "is_negation":{"type": "boolean", "description": "True for \"I do NOT like X\""},
                        },
                        "required": ["subject", "predicate", "object"],
                    },
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": "query_memory",
        "description": (
            "Search the user's memory. ALWAYS pass the natural-language "
            "question as `query`. If the question targets a specific "
            "attribute, ALSO pass `predicate_hint` (string or list) using "
            "the canonical predicates listed in store_memory. If asking "
            "about the user themselves, pass `subject_hint=\"user\"`. "
            "Hints are boosts, not filters — wrong hints don't hide "
            "results, but right hints surface the answer instantly.\n\n"
            "Examples:\n"
            "  query=\"what is my job?\", predicate_hint=[\"job_title\",\"works_at\"], subject_hint=\"user\"\n"
            "  query=\"where does my sister live?\", predicate_hint=\"lives_in\"\n"
            "  query=\"do I have any pets?\", predicate_hint=\"pet\", subject_hint=\"user\""
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language question"},
                "limit": {"type": "integer", "description": "Max results", "default": 5},
                "predicate_hint": {
                    "description": "Canonical predicate(s) the answer likely uses. Boost only.",
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                    ],
                },
                "subject_hint": {"type": "string", "description": "Subject to favor. Pass \"user\" for first-person questions."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "forget_memory",
        "description": "Delete a specific memory by ID.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "Memory ID to delete"},
            },
            "required": ["memory_id"],
        },
    },
    {
        "name": "list_memories",
        "description": "List all stored memories organized by category.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "memory_stats",
        "description": "Get memory statistics — total counts.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def handle_tool_call(name: str, arguments: dict) -> str:
    """Execute a tool and return the result as a string."""
    m = _get_memory()

    if name == "store_memory":
        result = m.store(arguments["text"], facts=arguments.get("facts"))
        return json.dumps(result)

    elif name == "query_memory":
        limit = arguments.get("limit", 5)
        results = m.query(
            arguments["query"], limit=limit,
            predicate_hint=arguments.get("predicate_hint"),
            subject_hint=arguments.get("subject_hint"))
        return json.dumps([r.to_dict() for r in results], indent=2)

    elif name == "forget_memory":
        ok = m.forget(memory_id=arguments["memory_id"])
        return json.dumps({"deleted": ok})

    elif name == "list_memories":
        return json.dumps(m.profile(), indent=2, default=str)

    elif name == "memory_stats":
        return json.dumps(m.stats(), indent=2)

    else:
        return json.dumps({"error": f"Unknown tool: {name}"})


# --- JSON-RPC stdio transport ---

def send_response(id, result):
    """Send a JSON-RPC response to stdout."""
    msg = json.dumps({"jsonrpc": "2.0", "id": id, "result": result})
    # MCP uses Content-Length header framing
    header = f"Content-Length: {len(msg)}\r\n\r\n"
    sys.stdout.write(header + msg)
    sys.stdout.flush()


def send_error(id, code, message):
    msg = json.dumps({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}})
    header = f"Content-Length: {len(msg)}\r\n\r\n"
    sys.stdout.write(header + msg)
    sys.stdout.flush()


def read_message():
    """Read a JSON-RPC message from stdin (Content-Length framed)."""
    # Read headers
    content_length = 0
    while True:
        line = sys.stdin.readline()
        if not line:
            return None  # EOF
        line = line.strip()
        if not line:
            break  # Empty line = end of headers
        if line.lower().startswith("content-length:"):
            content_length = int(line.split(":")[1].strip())

    if content_length == 0:
        return None

    # Read body
    body = sys.stdin.read(content_length)
    return json.loads(body)


def main():
    """Run the MCP server loop."""
    sys.stderr.write("Lore Memory MCP server starting...\n")

    while True:
        try:
            msg = read_message()
            if msg is None:
                break

            method = msg.get("method", "")
            id = msg.get("id")
            params = msg.get("params", {})

            if method == "initialize":
                send_response(id, {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "lore-memory", "version": "1.0.0"},
                })

            elif method == "notifications/initialized":
                pass  # No response needed

            elif method == "tools/list":
                send_response(id, {"tools": TOOLS})

            elif method == "tools/call":
                tool_name = params.get("name", "")
                arguments = params.get("arguments", {})
                try:
                    result_text = handle_tool_call(tool_name, arguments)
                    send_response(id, {
                        "content": [{"type": "text", "text": result_text}],
                    })
                except Exception as e:
                    send_response(id, {
                        "content": [{"type": "text", "text": f"Error: {e}"}],
                        "isError": True,
                    })

            elif method == "ping":
                send_response(id, {})

            else:
                if id is not None:
                    send_error(id, -32601, f"Method not found: {method}")

        except json.JSONDecodeError:
            continue
        except Exception as e:
            sys.stderr.write(f"Error: {e}\n")
            continue

    # Cleanup
    if _memory:
        _memory.close()


if __name__ == "__main__":
    main()
