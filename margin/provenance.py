"""
Provenance tracking for uncertain values.
Detects correlation between derived values via shared ancestry.
"""

import uuid


def new_id() -> str:
    """Short unique ID for provenance tracking."""
    return uuid.uuid4().hex[:8]


def are_correlated(ids_a: list[str], ids_b: list[str]) -> bool:
    """True if two provenance chains share any common ancestor."""
    return bool(set(ids_a) & set(ids_b))


def merge(ids_a: list[str], ids_b: list[str]) -> list[str]:
    """Merge two provenance chains, adding a new derived ID."""
    return list(set(ids_a) | set(ids_b) | {new_id()})
