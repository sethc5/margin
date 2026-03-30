"""
Provenance tracking for uncertain values.
Detects correlation between derived values via shared ancestry.
Optionally tracks full lineage via ProvenanceGraph.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional


def new_id() -> str:
    """Short unique ID for provenance tracking."""
    return uuid.uuid4().hex[:8]

# margin-poc compat alias
_new_id = new_id


def are_correlated(ids_a: list[str], ids_b: list[str]) -> bool:
    """True if two provenance chains share any common ancestor."""
    return bool(set(ids_a) & set(ids_b))


def merge(ids_a: list[str], ids_b: list[str]) -> list[str]:
    """Merge two provenance chains, adding a new derived ID."""
    return list(set(ids_a) | set(ids_b) | {new_id()})

# margin-poc compat alias
merge_provenance = merge


# -----------------------------------------------------------------------
# Provenance graph — full lineage tracking
# -----------------------------------------------------------------------

@dataclass
class ProvenanceNode:
    """
    A node in the provenance graph representing a single derived value.

    id:         unique identifier
    operation:  what created this value (e.g. "add", "measure_gap_IOI")
    source_ids: IDs of input values this was derived from
    """
    id: str
    operation: str
    source_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"id": self.id, "operation": self.operation, "source_ids": self.source_ids}

    @classmethod
    def from_dict(cls, d: dict) -> ProvenanceNode:
        return cls(id=d["id"], operation=d["operation"], source_ids=d.get("source_ids", []))


class ProvenanceGraph:
    """
    Tracks the full lineage of uncertain values through computations.

    Used to detect correlations (shared provenance affects how
    uncertainties combine) and to explain where a value came from.
    """

    def __init__(self):
        self.nodes: dict[str, ProvenanceNode] = {}

    def add_node(self, node: ProvenanceNode) -> None:
        self.nodes[node.id] = node

    def get_node(self, node_id: str) -> Optional[ProvenanceNode]:
        return self.nodes.get(node_id)

    def find_common_ancestors(self, ids_a: set[str], ids_b: set[str]) -> set[str]:
        """Find common ancestor IDs between two provenance sets."""
        return ids_a & ids_b

    def trace_lineage(self, node_id: str, depth: int = 10) -> list[ProvenanceNode]:
        """Trace the lineage of a node back through the graph (oldest first)."""
        lineage = []
        to_visit = [node_id]
        visited: set[str] = set()

        while to_visit and len(lineage) < depth:
            current_id = to_visit.pop(0)
            if current_id in visited:
                continue
            visited.add(current_id)

            node = self.get_node(current_id)
            if node is None:
                continue

            lineage.append(node)
            to_visit.extend(node.source_ids)

        return list(reversed(lineage))

    def create_root(self, operation: str = "root") -> str:
        """Create a root provenance node and return its ID."""
        node_id = new_id()
        self.add_node(ProvenanceNode(id=node_id, operation=operation))
        return node_id

    def derive(self, operation: str, source_ids: list[str]) -> str:
        """Create a derived provenance node and return its ID."""
        node_id = new_id()
        self.add_node(ProvenanceNode(id=node_id, operation=operation, source_ids=source_ids))
        return node_id

    def compress(self, max_nodes: int = 500) -> 'ProvenanceGraph':
        """
        Prune the oldest nodes to stay within max_nodes.

        Nodes are removed in insertion order (oldest first). Any
        ``source_ids`` in surviving nodes that reference a pruned node
        are removed, so the graph stays self-consistent — survivors
        whose parents were pruned become new roots.

        Returns self for chaining (e.g. before save_monitor).

            monitor.provenance_graph.compress(500)
            save_monitor(monitor, "state.json")
        """
        if len(self.nodes) <= max_nodes:
            return self
        all_ids = list(self.nodes.keys())
        pruned = set(all_ids[:-max_nodes])
        for node_id in pruned:
            del self.nodes[node_id]
        for node in self.nodes.values():
            node.source_ids = [sid for sid in node.source_ids if sid not in pruned]
        return self

    def to_dict(self) -> dict:
        return {"nodes": {k: v.to_dict() for k, v in self.nodes.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> ProvenanceGraph:
        g = cls()
        for k, v in d.get("nodes", {}).items():
            g.add_node(ProvenanceNode.from_dict(v))
        return g

    def __repr__(self) -> str:
        return f"ProvenanceGraph({len(self.nodes)} nodes)"


def create_root_provenance(operation: str = "root") -> list[str]:
    """Create a fresh provenance ID for a root (measured) value."""
    return [new_id()]
