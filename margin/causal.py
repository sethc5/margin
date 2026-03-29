"""
Causal language: typed dependency graphs between components.

Expresses WHY a component is in its current state by linking observations
through typed causal relationships. A CausalGraph is a DAG where nodes
are component observations and edges are typed causal links.

The causal language does not infer causation — it provides the vocabulary
for declaring and querying causal structure. Domain-specific layers
(transformer circuits, infrastructure dependencies, clinical pathways)
populate the graph; the causal language provides the typed query interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .health import Health
from .observation import Expression


# -----------------------------------------------------------------------
# Causal link types
# -----------------------------------------------------------------------

class CauseType(Enum):
    """How one component's state causes another's."""
    DEGRADES = "DEGRADES"       # A degrading causes B to degrade
    BLOCKS = "BLOCKS"           # A failing prevents B from recovering
    TRIGGERS = "TRIGGERS"       # A's state change triggers B's state change
    CORRELATES = "CORRELATES"   # co-occurring, direction uncertain
    MITIGATES = "MITIGATES"     # A's health improvement helps B recover


@dataclass
class CausalLink:
    """
    A typed causal relationship between two components.

    source:     the causing component
    target:     the affected component
    cause_type: how source affects target
    strength:   0.0-1.0, how strong the relationship is (1.0 = deterministic)
    condition:  optional — this link only applies when source is in this health state
    evidence:   free text — why we believe this link exists
    """
    source: str
    target: str
    cause_type: CauseType
    strength: float = 1.0
    condition: Optional[Health] = None
    evidence: str = ""

    def to_dict(self) -> dict:
        d = {
            "source": self.source, "target": self.target,
            "cause_type": self.cause_type.value,
            "strength": round(self.strength, 4),
            "evidence": self.evidence,
        }
        if self.condition:
            d["condition"] = self.condition.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> CausalLink:
        return cls(
            source=d["source"], target=d["target"],
            cause_type=CauseType(d["cause_type"]),
            strength=d.get("strength", 1.0),
            condition=Health(d["condition"]) if "condition" in d else None,
            evidence=d.get("evidence", ""),
        )

    def __repr__(self) -> str:
        cond = f" when {self.condition.value}" if self.condition else ""
        return f"{self.source} --{self.cause_type.value}--> {self.target}{cond}"


# -----------------------------------------------------------------------
# CausalGraph — the dependency structure
# -----------------------------------------------------------------------

@dataclass
class CausalGraph:
    """
    A directed acyclic graph of causal relationships between components.

    Nodes are component names. Edges are CausalLinks. The graph is
    populated by the domain layer and queried by the reasoning layer.
    """
    links: list[CausalLink] = field(default_factory=list)

    def add(self, link: CausalLink) -> None:
        self.links.append(link)

    def add_degrades(self, source: str, target: str, strength: float = 1.0,
                     condition: Optional[Health] = None, evidence: str = "") -> None:
        self.add(CausalLink(source, target, CauseType.DEGRADES, strength, condition, evidence))

    def add_blocks(self, source: str, target: str, strength: float = 1.0,
                   condition: Optional[Health] = None, evidence: str = "") -> None:
        self.add(CausalLink(source, target, CauseType.BLOCKS, strength, condition, evidence))

    def add_triggers(self, source: str, target: str, strength: float = 1.0,
                     condition: Optional[Health] = None, evidence: str = "") -> None:
        self.add(CausalLink(source, target, CauseType.TRIGGERS, strength, condition, evidence))

    def add_mitigates(self, source: str, target: str, strength: float = 1.0,
                      condition: Optional[Health] = None, evidence: str = "") -> None:
        self.add(CausalLink(source, target, CauseType.MITIGATES, strength, condition, evidence))

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def causes_of(self, component: str) -> list[CausalLink]:
        """All links where `component` is the target (what causes it)."""
        return [l for l in self.links if l.target == component]

    def effects_of(self, component: str) -> list[CausalLink]:
        """All links where `component` is the source (what it affects)."""
        return [l for l in self.links if l.source == component]

    def upstream(self, component: str, depth: int = 10) -> list[str]:
        """All components upstream of `component` (transitive causes)."""
        visited = set()
        queue = [component]
        result = []
        while queue and len(result) < depth:
            current = queue.pop(0)
            for link in self.causes_of(current):
                if link.source not in visited:
                    visited.add(link.source)
                    result.append(link.source)
                    queue.append(link.source)
        return result

    def downstream(self, component: str, depth: int = 10) -> list[str]:
        """All components downstream of `component` (transitive effects)."""
        visited = set()
        queue = [component]
        result = []
        while queue and len(result) < depth:
            current = queue.pop(0)
            for link in self.effects_of(current):
                if link.target not in visited:
                    visited.add(link.target)
                    result.append(link.target)
                    queue.append(link.target)
        return result

    def roots(self) -> list[str]:
        """Components with no incoming causal links (root causes)."""
        targets = {l.target for l in self.links}
        sources = {l.source for l in self.links}
        return sorted(sources - targets)

    def leaves(self) -> list[str]:
        """Components with no outgoing causal links (end effects)."""
        targets = {l.target for l in self.links}
        sources = {l.source for l in self.links}
        return sorted(targets - sources)

    def components(self) -> list[str]:
        """All components mentioned in any link."""
        names = set()
        for l in self.links:
            names.add(l.source)
            names.add(l.target)
        return sorted(names)

    # ------------------------------------------------------------------
    # Explanation
    # ------------------------------------------------------------------

    def explain(self, component: str, expr: Optional[Expression] = None) -> Explanation:
        """
        Build a typed explanation of why a component is in its current state.
        Traces upstream through the causal graph, optionally annotating
        with current health from an Expression.
        """
        causes = self.causes_of(component)
        health = expr.health_of(component) if expr else None

        upstream_explanations = []
        for link in causes:
            # Check if the link's condition is met
            if link.condition and expr:
                source_health = expr.health_of(link.source)
                if source_health != link.condition:
                    continue  # link not active

            source_health = expr.health_of(link.source) if expr else None
            upstream_explanations.append(CauseExplanation(
                source=link.source,
                source_health=source_health,
                cause_type=link.cause_type,
                strength=link.strength,
                evidence=link.evidence,
            ))

        return Explanation(
            component=component,
            health=health,
            causes=upstream_explanations,
        )

    def explain_all(self, expr: Expression) -> dict[str, Explanation]:
        """Explain every component in the expression."""
        return {o.name: self.explain(o.name, expr) for o in expr.observations}

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {"links": [l.to_dict() for l in self.links]}

    @classmethod
    def from_dict(cls, d: dict) -> CausalGraph:
        return cls(links=[CausalLink.from_dict(l) for l in d.get("links", [])])

    def to_string(self) -> str:
        if not self.links:
            return "(no causal links)"
        return "\n".join(repr(l) for l in self.links)

    def __repr__(self) -> str:
        return f"CausalGraph({len(self.links)} links, {len(self.components())} components)"


# -----------------------------------------------------------------------
# Explanation — the output of causal reasoning
# -----------------------------------------------------------------------

@dataclass
class CauseExplanation:
    """One upstream cause in an explanation."""
    source: str
    source_health: Optional[Health]
    cause_type: CauseType
    strength: float
    evidence: str = ""

    def to_string(self) -> str:
        h = self.source_health.value if self.source_health else "?"
        return f"{self.source}:{h} --{self.cause_type.value}({self.strength:.0%})--> "

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "source_health": self.source_health.value if self.source_health else None,
            "cause_type": self.cause_type.value,
            "strength": round(self.strength, 4),
            "evidence": self.evidence,
        }


@dataclass
class Explanation:
    """
    Typed explanation of why a component is in its current state.

    component:  the component being explained
    health:     its current health (if known)
    causes:     upstream causal links that are currently active
    """
    component: str
    health: Optional[Health] = None
    causes: list[CauseExplanation] = field(default_factory=list)

    @property
    def has_known_cause(self) -> bool:
        return len(self.causes) > 0

    @property
    def root_cause(self) -> Optional[CauseExplanation]:
        """Strongest upstream cause, or None."""
        if not self.causes:
            return None
        return max(self.causes, key=lambda c: c.strength)

    def to_string(self) -> str:
        h = self.health.value if self.health else "?"
        if not self.causes:
            return f"{self.component}:{h} (no known causes)"
        cause_strs = [c.to_string() for c in self.causes]
        return f"{self.component}:{h} ← " + ", ".join(cause_strs)

    def to_dict(self) -> dict:
        return {
            "component": self.component,
            "health": self.health.value if self.health else None,
            "has_known_cause": self.has_known_cause,
            "root_cause": self.root_cause.to_dict() if self.root_cause else None,
            "causes": [c.to_dict() for c in self.causes],
        }

    def __repr__(self) -> str:
        return f"Explanation({self.to_string()})"
