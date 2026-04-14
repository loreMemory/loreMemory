"""
Lore Memory CLI — command-line interface for persistent AI memory.

Usage:
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
