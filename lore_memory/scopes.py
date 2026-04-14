"""
Memory scopes — hard isolation layer. Carried forward from v2 unchanged.
File-per-scope physical isolation for PRIVATE and REPO.
Logical isolation (org-level file) for SHARED.
"""

from __future__ import annotations

import hashlib
from enum import Enum
from pathlib import Path


class Scope(str, Enum):
    PRIVATE = "private"
    SHARED = "shared"
    REPO = "repo"


class Context(str, Enum):
    PERSONAL = "personal"
    CHAT = "chat"
    REPO = "repo"
    COMPANY = "company"


def scope_db_path(data_dir: Path, scope: Scope, *, user_id: str = "", org_id: str = "", repo_id: str = "") -> Path:
    """Resolve scope to a database file path."""
    scope_dir = data_dir / scope.value
    scope_dir.mkdir(parents=True, exist_ok=True)
    if scope == Scope.PRIVATE:
        return scope_dir / f"user_{_safe(user_id)}.db"
    elif scope == Scope.SHARED:
        return scope_dir / f"shared_{_safe(org_id)}.db"
    elif scope == Scope.REPO:
        return scope_dir / f"repo_{_safe(repo_id)}.db"
    raise ValueError(f"Unknown scope: {scope}")


def can_access(accessor_user_id: str, scope: Scope, owner_user_id: str = "") -> bool:
    """Check if accessor can read from a scope."""
    if scope == Scope.PRIVATE:
        return accessor_user_id == owner_user_id
    return True  # SHARED/REPO: app-layer authorization


def _safe(raw: str) -> str:
    if not raw:
        raise ValueError("Scope identifier cannot be empty")
    if raw.replace("_", "").replace("-", "").isalnum() and len(raw) < 64:
        return raw
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
