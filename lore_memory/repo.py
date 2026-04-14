"""
Repo memory — git-based code understanding.
Carried forward from v2, same approach: commits + file tree.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from lore_memory.store import Memory


@dataclass
class Commit:
    sha: str; author: str; date: float; message: str
    files: list[str] = field(default_factory=list)
    insertions: int = 0; deletions: int = 0


def repo_id(path: str | Path) -> str:
    return hashlib.sha256(str(Path(path).resolve()).encode()).hexdigest()[:16]


def recent_commits(path: str | Path, limit: int = 50) -> list[Commit]:
    try:
        r = subprocess.run(
            ["git", "log", f"--max-count={limit}", "--format=%H%n%an%n%at%n%s%n---END---", "--stat"],
            cwd=str(path), capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            return []
        commits = []
        for block in r.stdout.split("---END---"):
            block = block.strip()
            if not block:
                continue
            lines = block.split("\n")
            if len(lines) < 4:
                continue
            try:
                date = float(lines[2].strip())
            except ValueError:
                date = time.time()
            files = [m.group(1) for line in lines[4:] if (m := re.match(r"^\s*([\w/.\-]+)\s+\|", line))]
            commits.append(Commit(sha=lines[0].strip(), author=lines[1].strip(), date=date,
                                  message=lines[3].strip(), files=files))
        return commits
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return []


def file_tree(path: str | Path, max_files: int = 500) -> list[str]:
    try:
        r = subprocess.run(["git", "ls-files"], cwd=str(path), capture_output=True, text=True, timeout=10)
        return [f for f in r.stdout.strip().split("\n")[:max_files] if f] if r.returncode == 0 else []
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return []


def commits_to_memories(commits: list[Commit], rid: str) -> list[Memory]:
    mems = []
    now = time.time()
    for c in commits:
        mems.append(Memory(
            scope="repo", context="repo", repo_id=rid,
            subject=c.author, predicate="committed", object_value=c.message[:200],
            source_text=f"sha:{c.sha} files:{','.join(c.files[:5])} +{c.insertions}-{c.deletions}",
            source_type="system", confidence=0.95,
            created_at=c.date, updated_at=c.date, last_accessed=now,
            metadata={"sha": c.sha, "files": c.files[:20]},
        ))
        for fp in c.files[:10]:
            mems.append(Memory(
                scope="repo", context="repo", repo_id=rid,
                subject=fp, predicate="changed_by", object_value=c.author,
                source_text=c.message[:200], source_type="system", confidence=0.95,
                created_at=c.date, updated_at=c.date, last_accessed=now,
                metadata={"sha": c.sha},
            ))
    return mems


def tree_to_memories(files: list[str], rid: str) -> list[Memory]:
    mems = []
    now = time.time()
    dirs: dict[str, int] = {}
    exts: dict[str, int] = {}
    for f in files:
        parts = f.split("/")
        if len(parts) > 1:
            dirs[parts[0]] = dirs.get(parts[0], 0) + 1
        ext = os.path.splitext(f)[1]
        if ext:
            exts[ext] = exts.get(ext, 0) + 1
    for d, cnt in sorted(dirs.items(), key=lambda x: -x[1])[:20]:
        mems.append(Memory(
            scope="repo", context="repo", repo_id=rid,
            subject="repo", predicate="has_directory", object_value=f"{d}/ ({cnt} files)",
            source_type="system", confidence=0.99,
            created_at=now, updated_at=now, last_accessed=now,
        ))
    for ext, cnt in sorted(exts.items(), key=lambda x: -x[1])[:10]:
        mems.append(Memory(
            scope="repo", context="repo", repo_id=rid,
            subject="repo", predicate="uses_language", object_value=f"{ext} ({cnt} files)",
            source_type="system", confidence=0.99,
            created_at=now, updated_at=now, last_accessed=now,
        ))
    return mems
