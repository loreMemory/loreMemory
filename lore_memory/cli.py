"""
Lore Memory CLI — command-line interface for persistent AI memory.

Usage:
    lore chat                              # interactive REPL (recommended)
    lore store "I work at Google and love Python"
    lore query "where do I work?"
    lore list
    lore forget <memory_id>
    lore stats
    lore serve [--port 8000]
    lore mcp
"""

from __future__ import annotations

import argparse
import json
import sys
import re


def _get_memory(args):
    """Create a Memory instance from CLI args."""
    from lore_memory.memory import Memory
    return Memory(
        user_id=getattr(args, "user", "default"),
        data_dir=getattr(args, "data_dir", None),
    )


def cmd_store(args):
    """Store a memory from text."""
    m = _get_memory(args)
    try:
        result = m.store(args.text, scope=args.scope)
        print(f"Stored: {result['created']} new, "
              f"{result['deduplicated']} deduplicated, "
              f"{result['contradictions']} contradictions resolved")
    except Exception as e:
        print(f"Error storing memory: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        m.close()


def cmd_query(args):
    """Query memories."""
    m = _get_memory(args)
    try:
        results = m.query(args.query, limit=args.limit)
        if not results:
            print("No memories found.")
            return
        for i, r in enumerate(results, 1):
            d = r.to_dict()
            if args.json:
                print(json.dumps(d))
            else:
                print(f"  [{i}] {d['text']}")
                print(f"      score={d['score']}  confidence={d['confidence']}")
    except Exception as e:
        print(f"Error querying memory: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        m.close()


def cmd_list(args):
    """List all memories via profile."""
    m = _get_memory(args)
    try:
        profile = m.profile()
        if not profile:
            print("No memories stored yet.")
            return
        for predicate, entries in profile.items():
            print(f"\n  {predicate}:")
            for entry in entries:
                if isinstance(entry, dict):
                    val = entry.get("value", entry.get("object", str(entry)))
                    conf = entry.get("confidence", "")
                    conf_str = f"  (confidence: {conf})" if conf else ""
                    print(f"    - {val}{conf_str}")
                else:
                    print(f"    - {entry}")
    except Exception as e:
        print(f"Error listing memories: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        m.close()


def cmd_forget(args):
    """Forget a specific memory."""
    m = _get_memory(args)
    try:
        if args.all:
            ok = m.forget_all()
        else:
            ok = m.forget(memory_id=args.id)
        if ok:
            print("Memory deleted.")
        else:
            print("Memory not found or already deleted.")
    except Exception as e:
        print(f"Error deleting memory: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        m.close()


def cmd_stats(args):
    """Show memory statistics."""
    m = _get_memory(args)
    try:
        s = m.stats()
        print("Memory Statistics:")
        for key, value in sorted(s.items()):
            label = key.replace("_", " ").title()
            print(f"  {label}: {value}")
    except Exception as e:
        print(f"Error getting stats: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        m.close()


def cmd_serve(args):
    """Start the REST API server."""
    try:
        import uvicorn
    except ImportError:
        print("The API server requires FastAPI and uvicorn.\n"
              "Install with: pip install lore-memory[api]",
              file=sys.stderr)
        sys.exit(1)
    print(f"Starting Lore Memory API on port {args.port}...")
    uvicorn.run("api.server:app", host=args.host, port=args.port, reload=False)


def cmd_mcp(args):
    """Start the MCP server."""
    try:
        import importlib
        mod = importlib.import_module("mcp.server")
    except ImportError:
        print("The MCP server requires the mcp package.\n"
              "Install with: pip install lore-memory[mcp]",
              file=sys.stderr)
        sys.exit(1)
    # The MCP server module handles its own startup
    if hasattr(mod, "main"):
        mod.main()
    else:
        print("MCP server module loaded but no main() found.", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
#  Interactive REPL
# ---------------------------------------------------------------------------

# Sentence-start tokens that mark a question in English. Combined with a
# trailing "?" check, these catch the vast majority of questions without
# forcing the user to remember a special syntax. English-only, matches the
# product scope.
_QUESTION_STARTERS = (
    "what", "where", "when", "who", "whom", "why", "how", "which",
    "is", "are", "am", "do", "does", "did", "can", "could", "should",
    "would", "was", "were", "will", "have", "has", "had",
    "tell me", "show me", "list",
)


def _looks_like_question(text: str) -> bool:
    """Distinguish a question from a statement for the REPL.

    Signals: trailing '?', or sentence-starts with a question word. Lightly
    tuned for English natural chat; when in doubt, we treat as a statement
    so the user's facts are stored rather than quietly queried.
    """
    t = text.strip()
    if not t:
        return False
    if t.endswith("?"):
        return True
    low = t.lower()
    for start in _QUESTION_STARTERS:
        if low == start or low.startswith(start + " "):
            return True
    return False


def _render_answer(r) -> str:
    """Format a QueryResult for conversational display."""
    if r.answer is None:
        return "I don't have anything about that yet."
    text = r.answer.text.strip()
    if r.answer.is_suspicious:
        text = f"(flagged as suspicious) {text}"
    if not r.needs_clarification:
        return f"→ {text}"
    # Ambiguous: show top options so the user can disambiguate.
    lines = [
        f"I'm not sure — I see several things that could match "
        f"(certainty {r.certainty:.2f}).",
        f"  • {text}",
    ]
    for alt in r.alternatives[1:3]:
        lines.append(f"  • {alt.text.strip()}")
    return "\n".join(lines)


def _chat_help() -> str:
    return (
        "Commands:\n"
        "  /help            show this help\n"
        "  /list            show everything I know about you\n"
        "  /stats           fact counts\n"
        "  /forget all      delete every memory for this user\n"
        "  /export <path>   write all memories to a JSONL file\n"
        "  /exit  or  /quit leave\n"
        "\n"
        "Anything else: if it ends with '?' or starts with 'what/where/who/...'\n"
        "I treat it as a question. Otherwise I remember it."
    )


def cmd_chat(args):
    """Interactive REPL — natural-language store and ask.

    A non-developer friendly mode: type a fact to remember it, type a
    question to recall. Slash commands handle the rest.
    """
    m = _get_memory(args)
    user = getattr(args, "user", "default") or "default"
    _first_input = [True]

    def _warmup_hint():
        # Emit exactly once, right before the first real request runs.
        if _first_input[0]:
            _first_input[0] = False
            from lore_memory.engine import Engine as _Eng
            if not _Eng._ST_PROBED:
                print("  (loading embedding model — a few seconds only the first time...)",
                      flush=True)

    try:
        print(f"Lore memory — chatting as '{user}'. Type /help for commands, /exit to leave.")
        while True:
            try:
                line = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            # --- slash commands ---
            if line.startswith("/"):
                cmd, _, rest = line[1:].partition(" ")
                cmd = cmd.lower()
                rest = rest.strip()
                if cmd in ("exit", "quit", "q"):
                    break
                elif cmd in ("help", "h", "?"):
                    print(_chat_help())
                elif cmd == "list":
                    prof = m.profile()
                    if not prof:
                        print("(nothing stored yet)")
                    else:
                        for pred in sorted(prof):
                            for e in prof[pred]:
                                val = e.get("value", str(e))
                                neg = "not " if e.get("negation") else ""
                                print(f"  {pred}: {neg}{val}")
                elif cmd == "stats":
                    for k, v in sorted(m.stats().items()):
                        print(f"  {k.replace('_',' ')}: {v}")
                elif cmd == "forget":
                    if rest == "all":
                        confirm = input("  Delete ALL your memories? type YES to confirm: ").strip()
                        if confirm == "YES":
                            m.forget_all(hard=True)
                            print("  Done — all memories deleted.")
                        else:
                            print("  Cancelled.")
                    else:
                        print("  Only '/forget all' is supported in chat. For per-fact deletion use 'lore forget <id>'.")
                elif cmd == "export":
                    if not rest:
                        print("  Usage: /export <path>.jsonl")
                        continue
                    n = m.export_to_jsonl(rest)
                    print(f"  Wrote {n} rows to {rest}.")
                else:
                    print(f"  Unknown command: /{cmd}. Try /help.")
                continue

            # --- plain text: question or fact ---
            _warmup_hint()
            if _looks_like_question(line):
                r = m.query_one(line)
                print(_render_answer(r))
            else:
                res = m.store(line)
                bits = []
                if res.get("created"):
                    bits.append(f"remembered ({res['created']})")
                if res.get("contradictions"):
                    bits.append(f"superseded {res['contradictions']}")
                if res.get("deduplicated"):
                    bits.append(f"deduplicated {res['deduplicated']}")
                print("  " + (", ".join(bits) if bits else "got it."))
    finally:
        m.close()
    print("bye.")


def main():
    parser = argparse.ArgumentParser(
        prog="lore",
        description="Lore Memory — persistent AI memory from the command line",
    )
    parser.add_argument("--user", default="default",
                        help="User ID (default: 'default')")
    parser.add_argument("--data-dir", dest="data_dir", default=None,
                        help="Data directory (default: ~/.lore-memory)")

    sub = parser.add_subparsers(dest="command")

    # chat (interactive REPL — recommended entry point)
    p_chat = sub.add_parser("chat",
                             help="Interactive REPL (recommended for new users)")
    p_chat.set_defaults(func=cmd_chat)

    # store
    p_store = sub.add_parser("store", help="Store a memory")
    p_store.add_argument("text", help="Text to remember")
    p_store.add_argument("--scope", choices=["private", "shared"],
                         default="private", help="Memory scope")
    p_store.set_defaults(func=cmd_store)

    # query
    p_query = sub.add_parser("query", help="Query memories")
    p_query.add_argument("query", help="Search query")
    p_query.add_argument("--limit", type=int, default=10,
                         help="Max results (default: 10)")
    p_query.add_argument("--json", action="store_true",
                         help="Output as JSON lines")
    p_query.set_defaults(func=cmd_query)

    # list
    p_list = sub.add_parser("list", help="List all memories")
    p_list.set_defaults(func=cmd_list)

    # forget
    p_forget = sub.add_parser("forget", help="Forget a memory")
    p_forget.add_argument("id", nargs="?", default=None,
                          help="Memory ID to delete")
    p_forget.add_argument("--all", action="store_true",
                          help="Delete ALL memories (irreversible)")
    p_forget.set_defaults(func=cmd_forget)

    # stats
    p_stats = sub.add_parser("stats", help="Show memory statistics")
    p_stats.set_defaults(func=cmd_stats)

    # serve
    p_serve = sub.add_parser("serve", help="Start REST API server")
    p_serve.add_argument("--host", default="127.0.0.1",
                         help="Bind address (default: 127.0.0.1)")
    p_serve.add_argument("--port", type=int, default=8000,
                         help="Port (default: 8000)")
    p_serve.set_defaults(func=cmd_serve)

    # mcp
    p_mcp = sub.add_parser("mcp", help="Start MCP server")
    p_mcp.set_defaults(func=cmd_mcp)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
