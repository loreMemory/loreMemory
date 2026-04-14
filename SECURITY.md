# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 1.0.x   | Yes                |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly.

**Do not open a public GitHub issue for security vulnerabilities.**

Instead, open a **private security advisory** on GitHub (Settings > Security > Advisories > New) with:

- A description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

You will receive a response within 48 hours. We will work with you to understand the issue and coordinate a fix before any public disclosure.

## Security Design

Lore Memory is designed with security as a core principle:

- **Local-first**: All data stays on your machine. No cloud, no telemetry, no external API calls.
- **User isolation**: Each user gets a physically separate SQLite database. No shared tables, no row-level filtering.
- **Input sanitization**: All inputs are parameterized through SQLite prepared statements. No raw SQL concatenation.
- **No code execution**: The grammar parser extracts facts from text structure only. No `eval()`, no dynamic code generation.
- **Dependency-free core**: Zero external dependencies in the core package reduces supply chain risk.

## Scope

The following are in scope for security reports:

- SQL injection in store/query/forget operations
- Cross-user data leakage
- Memory corruption or data loss
- Denial of service through crafted inputs
- Path traversal in data directory configuration

The following are out of scope:

- Issues in optional dependencies (sentence-transformers, FastAPI)
- Physical access to the SQLite database files
- Social engineering attacks
