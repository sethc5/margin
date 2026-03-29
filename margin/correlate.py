"""
Auto-correlation: discover which components move together.

CausalGraph requires manual edge definition. This module discovers
correlations from observation data and feeds them into the causal layer.

Given a Ledger or observation history, computes pairwise Pearson
correlations (optionally with lag) and returns typed Correlation objects
that can be converted to CausalLinks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

from .confidence import Confidence
from .causal import CausalGraph, CausalLink, CauseType
from .observation import Expression, Observation


@dataclass
class Correlation:
    """
    Discovered correlation between two components.

    component_a:  first component
    component_b:  second component
    coefficient:  Pearson r (-1 to +1)
    lag:          a leads b by this many steps (0 = simultaneous)
    n_samples:    overlapping samples used
    confidence:   based on sample size and strength
    """
    component_a: str
    component_b: str
    coefficient: float
    lag: int
    n_samples: int
    confidence: Confidence

    @property
    def strength(self) -> float:
        """Absolute correlation strength [0, 1]."""
        return abs(self.coefficient)

    @property
    def positive(self) -> bool:
        """True if components move in the same direction."""
        return self.coefficient > 0

    @property
    def negative(self) -> bool:
        """True if components move in opposite directions."""
        return self.coefficient < 0

    @property
    def a_leads(self) -> bool:
        """True if component_a leads component_b (lag > 0)."""
        return self.lag > 0

    @property
    def simultaneous(self) -> bool:
        """True if no lag detected."""
        return self.lag == 0

    def to_causal_link(self) -> CausalLink:
        """Convert to a CausalLink for use in a CausalGraph."""
        if self.lag > 0:
            # A leads B → A likely causes B
            source, target = self.component_a, self.component_b
            cause_type = CauseType.DEGRADES if self.negative else CauseType.CORRELATES
        elif self.lag < 0:
            # B leads A → B likely causes A
            source, target = self.component_b, self.component_a
            cause_type = CauseType.DEGRADES if self.negative else CauseType.CORRELATES
        else:
            # Simultaneous — direction unknown
            source, target = self.component_a, self.component_b
            cause_type = CauseType.CORRELATES

        return CausalLink(
            source=source,
            target=target,
            cause_type=cause_type,
            strength=self.strength,
            evidence=f"auto-correlated: r={self.coefficient:.3f}, lag={self.lag}, n={self.n_samples}",
        )

    def to_dict(self) -> dict:
        return {
            "component_a": self.component_a,
            "component_b": self.component_b,
            "coefficient": round(self.coefficient, 4),
            "lag": self.lag,
            "n_samples": self.n_samples,
            "confidence": self.confidence.value,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Correlation:
        return cls(
            component_a=d["component_a"],
            component_b=d["component_b"],
            coefficient=d["coefficient"],
            lag=d["lag"],
            n_samples=d["n_samples"],
            confidence=Confidence(d["confidence"]),
        )

    def __repr__(self) -> str:
        sign = "+" if self.coefficient >= 0 else ""
        lag_str = f", lag={self.lag}" if self.lag != 0 else ""
        return f"Correlation({self.component_a} ~ {self.component_b}: r={sign}{self.coefficient:.3f}{lag_str})"


# -----------------------------------------------------------------------
# Correlation matrix (all pairwise)
# -----------------------------------------------------------------------

@dataclass
class CorrelationMatrix:
    """
    All pairwise correlations for a set of components.

    components:   ordered list of component names
    correlations: list of significant Correlation objects
    matrix:       dict of (a, b) → coefficient for all pairs (including weak ones)
    """
    components: list[str]
    correlations: list[Correlation]
    matrix: dict[tuple[str, str], float] = field(default_factory=dict)

    def strongest(self, n: int = 5) -> list[Correlation]:
        """Top N correlations by absolute strength."""
        return sorted(self.correlations, key=lambda c: c.strength, reverse=True)[:n]

    def for_component(self, name: str) -> list[Correlation]:
        """All significant correlations involving this component."""
        return [c for c in self.correlations
                if c.component_a == name or c.component_b == name]

    def coefficient(self, a: str, b: str) -> Optional[float]:
        """Get the correlation coefficient between two components."""
        return self.matrix.get((a, b)) or self.matrix.get((b, a))

    def to_causal_graph(self, existing: Optional[CausalGraph] = None) -> CausalGraph:
        """Convert all discovered correlations to a CausalGraph."""
        graph = existing or CausalGraph()
        for corr in self.correlations:
            graph.add(corr.to_causal_link())
        return graph

    def to_dict(self) -> dict:
        return {
            "components": self.components,
            "correlations": [c.to_dict() for c in self.correlations],
        }

    def __repr__(self) -> str:
        return (f"CorrelationMatrix({len(self.components)} components, "
                f"{len(self.correlations)} significant pairs)")


# -----------------------------------------------------------------------
# Pearson correlation
# -----------------------------------------------------------------------

def _pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation coefficient between two aligned series."""
    n = len(xs)
    if n < 2:
        return 0.0

    mx = sum(xs) / n
    my = sum(ys) / n
    sx = sum((x - mx) ** 2 for x in xs)
    sy = sum((y - my) ** 2 for y in ys)

    if sx == 0 or sy == 0:
        return 0.0

    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / math.sqrt(sx * sy)


def _confidence_from_correlation(r: float, n: int) -> Confidence:
    """Confidence based on sample size and correlation strength."""
    abs_r = abs(r)
    if n >= 30 and abs_r > 0.7:
        return Confidence.HIGH
    if n >= 15 and abs_r > 0.5:
        return Confidence.MODERATE
    if n >= 5 and abs_r > 0.3:
        return Confidence.LOW
    return Confidence.INDETERMINATE


# -----------------------------------------------------------------------
# Core correlation functions
# -----------------------------------------------------------------------

def correlate_pair(
    xs: list[float],
    ys: list[float],
    name_a: str = "a",
    name_b: str = "b",
    max_lag: int = 0,
) -> Optional[Correlation]:
    """
    Compute correlation between two value series.

    If max_lag > 0, tests lags from -max_lag to +max_lag and returns
    the lag with the strongest correlation. Positive lag means A leads B.

    Returns None if insufficient overlapping data.
    """
    if max_lag == 0:
        n = min(len(xs), len(ys))
        if n < 3:
            return None
        r = _pearson(xs[:n], ys[:n])
        return Correlation(
            component_a=name_a,
            component_b=name_b,
            coefficient=r,
            lag=0,
            n_samples=n,
            confidence=_confidence_from_correlation(r, n),
        )

    # Test multiple lags
    best_r = 0.0
    best_lag = 0
    best_n = 0

    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            # A leads: align A[:-lag] with B[lag:] (or full if lag=0)
            a_slice = xs[:len(xs) - lag] if lag > 0 else xs
            b_slice = ys[lag:] if lag > 0 else ys
        else:
            # B leads: align A[-lag:] with B[:len(B)+lag]
            a_slice = xs[-lag:]
            b_slice = ys[:len(ys) + lag]

        n = min(len(a_slice), len(b_slice))
        if n < 3:
            continue

        r = _pearson(a_slice[:n], b_slice[:n])
        # Prefer stronger correlation; tie-break by smaller absolute lag
        if abs(r) > abs(best_r) or (abs(r) == abs(best_r) and abs(lag) < abs(best_lag)):
            best_r = r
            best_lag = lag
            best_n = n

    if best_n < 3:
        return None

    return Correlation(
        component_a=name_a,
        component_b=name_b,
        coefficient=best_r,
        lag=best_lag,
        n_samples=best_n,
        confidence=_confidence_from_correlation(best_r, best_n),
    )


def correlate(
    values_by_component: dict[str, list[float]],
    min_correlation: float = 0.7,
    max_lag: int = 0,
    min_samples: int = 5,
) -> CorrelationMatrix:
    """
    Discover correlations between all component pairs.

    Args:
        values_by_component: {name: [value, value, ...]} aligned time series
        min_correlation:     minimum |r| to include in results (default 0.7)
        max_lag:             max lag to test (0 = simultaneous only)
        min_samples:         minimum overlapping samples required

    Returns a CorrelationMatrix with all significant correlations.
    """
    names = sorted(values_by_component.keys())
    correlations = []
    matrix: dict[tuple[str, str], float] = {}

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            xs = values_by_component[a]
            ys = values_by_component[b]

            corr = correlate_pair(xs, ys, name_a=a, name_b=b, max_lag=max_lag)
            if corr is None or corr.n_samples < min_samples:
                continue

            matrix[(a, b)] = corr.coefficient

            if corr.strength >= min_correlation:
                correlations.append(corr)

    return CorrelationMatrix(
        components=names,
        correlations=correlations,
        matrix=matrix,
    )


# -----------------------------------------------------------------------
# Ledger integration
# -----------------------------------------------------------------------

def _extract_aligned_series(ledger, components: Optional[list[str]] = None) -> dict[str, list[float]]:
    """Extract value series from a Ledger, aligned by step.

    Groups observations by step, then for each step extracts the value
    for each component. Only includes steps where ALL requested components
    have observations.
    """
    # Collect all observations by step
    by_step: dict[int, dict[str, float]] = {}
    all_names: set[str] = set()

    for rec in ledger.records:
        step = rec.step
        if step not in by_step:
            by_step[step] = {}
        obs = rec.after if rec.after else rec.before
        by_step[step][obs.name] = obs.value
        all_names.add(obs.name)

    if components is None:
        components = sorted(all_names)

    # Build aligned series (only steps with all components present)
    series: dict[str, list[float]] = {c: [] for c in components}
    for step in sorted(by_step.keys()):
        step_data = by_step[step]
        if all(c in step_data for c in components):
            for c in components:
                series[c].append(step_data[c])

    return series


def correlate_from_ledger(
    ledger,
    components: Optional[list[str]] = None,
    min_correlation: float = 0.7,
    max_lag: int = 0,
    min_samples: int = 5,
) -> CorrelationMatrix:
    """
    Discover correlations from a Ledger's observation history.

    Args:
        ledger:           correction Ledger
        components:       component names to include (None = all)
        min_correlation:  minimum |r| to report (default 0.7)
        max_lag:          max lag to test (0 = simultaneous only)
        min_samples:      minimum overlapping samples required
    """
    series = _extract_aligned_series(ledger, components)
    return correlate(series, min_correlation, max_lag, min_samples)


def auto_causal_graph(
    ledger,
    existing: Optional[CausalGraph] = None,
    min_correlation: float = 0.7,
    max_lag: int = 3,
    min_samples: int = 5,
) -> CausalGraph:
    """
    Build a CausalGraph from auto-discovered correlations in a Ledger.

    Convenience function: extracts series, computes correlations with lag,
    converts to causal links, and merges with an existing graph.
    """
    matrix = correlate_from_ledger(
        ledger,
        min_correlation=min_correlation,
        max_lag=max_lag,
        min_samples=min_samples,
    )
    return matrix.to_causal_graph(existing)


# -----------------------------------------------------------------------
# Predicates
# -----------------------------------------------------------------------

PredicateFn = Callable[[Expression], bool]


def correlated_with(
    component_a: str,
    component_b: str,
    ledger,
    min_correlation: float = 0.7,
    **kwargs,
) -> PredicateFn:
    """True if two components are correlated above the threshold."""
    def check(expr: Expression) -> bool:
        matrix = correlate_from_ledger(ledger, [component_a, component_b], min_correlation, **kwargs)
        return any(
            (c.component_a == component_a and c.component_b == component_b) or
            (c.component_a == component_b and c.component_b == component_a)
            for c in matrix.correlations
        )
    return check


def any_new_correlation(
    ledger,
    baseline_graph: CausalGraph,
    min_correlation: float = 0.7,
    **kwargs,
) -> PredicateFn:
    """True if the ledger contains correlations not in the baseline graph."""
    def check(expr: Expression) -> bool:
        matrix = correlate_from_ledger(ledger, min_correlation=min_correlation, **kwargs)
        known_pairs = {(l.source, l.target) for l in baseline_graph.links}
        known_pairs |= {(l.target, l.source) for l in baseline_graph.links}
        for corr in matrix.correlations:
            pair = (corr.component_a, corr.component_b)
            if pair not in known_pairs and (pair[1], pair[0]) not in known_pairs:
                return True
        return False
    return check
