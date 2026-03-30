"""
Streaming monitor for transformer circuit health.

Tracks circuit gap drift, anomalies, and correlations across inference
steps. Wraps margin's Monitor with transformer-specific convenience.

Usage:
    from adapters.transformer import make_pythia_parser
    from adapters.transformer.monitor import CircuitMonitor

    monitor = CircuitMonitor(make_pythia_parser())

    for step in inference_steps:
        gaps = {"IOI": ioi_gap, "NM": nm_gap, "IH": ih_gap, "SH": sh_gap}
        monitor.update(gaps)

    monitor.drift("IOI")       # DriftClassification
    monitor.anomaly("NM")      # AnomalyClassification
    monitor.correlations       # which circuits degrade together
    monitor.causal_graph()     # auto-discovered + known dependencies
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from margin.observation import Parser, Expression
from margin.streaming import Monitor
from margin.drift import DriftClassification
from margin.anomaly import AnomalyClassification
from margin.correlate import CorrelationMatrix, auto_causal_graph
from margin.causal import CausalGraph
from .causal_templates import IOI_GRAPH


class CircuitMonitor:
    """
    Streaming health monitor for transformer circuits.

    Wraps margin.Monitor with known causal structure and
    transformer-specific convenience methods.
    """

    def __init__(
        self,
        parser: Parser,
        window: int = 200,
        known_graph: Optional[CausalGraph] = None,
    ):
        self.parser = parser
        self.monitor = Monitor(parser, window=window)
        self.known_graph = known_graph or CausalGraph()

    def update(
        self,
        gaps: dict[str, float],
        now: Optional[datetime] = None,
    ) -> Expression:
        """Update with new circuit gap measurements."""
        return self.monitor.update(gaps, now=now)

    @property
    def expression(self) -> Optional[Expression]:
        return self.monitor.expression

    @property
    def step(self) -> int:
        return self.monitor.step

    def drift(self, circuit: str) -> Optional[DriftClassification]:
        return self.monitor.drift(circuit)

    def anomaly(self, circuit: str) -> Optional[AnomalyClassification]:
        return self.monitor.anomaly(circuit)

    @property
    def correlations(self) -> Optional[CorrelationMatrix]:
        return self.monitor.correlations

    def causal_graph(self, min_correlation: float = 0.7) -> CausalGraph:
        """
        Build a causal graph combining known circuit dependencies
        with auto-discovered correlations from the monitoring history.
        """
        # Start with known structure
        graph = CausalGraph(links=list(self.known_graph.links))

        # Add auto-discovered correlations
        if self.monitor.correlations:
            for corr in self.monitor.correlations.correlations:
                if corr.strength >= min_correlation:
                    graph.add(corr.to_causal_link())

        return graph

    def status(self) -> dict:
        """Full snapshot with circuit-specific annotations."""
        s = self.monitor.status()
        s["known_causal_links"] = len(self.known_graph.links)
        return s

    def __repr__(self) -> str:
        return f"CircuitMonitor(step={self.step}, {len(self.parser.baselines)} circuits)"
