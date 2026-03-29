"""Tests for margin.correlate — auto-correlation discovery."""

import math
from datetime import datetime, timedelta

from margin import (
    Observation, Health, Confidence, Op,
    Ledger, Record, Expression,
    CausalGraph, CausalLink, CauseType,
    Correlation, CorrelationMatrix,
    correlate_pair, correlate, correlate_from_ledger,
    auto_causal_graph,
    correlated_with, any_new_correlation,
)


def _obs(name, value, baseline=100.0, t=None):
    return Observation(
        name=name, health=Health.INTACT, value=value, baseline=baseline,
        confidence=Confidence.HIGH, measured_at=t,
    )


def _ledger_multi(components_values, n_steps=20):
    """Build a Ledger with multiple components, one record per component per step."""
    t0 = datetime(2026, 1, 1)
    ledger = Ledger(label="test")
    step = 0
    for i in range(n_steps):
        t = t0 + timedelta(seconds=i * 60)
        for name, values in components_values.items():
            if i < len(values):
                o = _obs(name, values[i], t=t)
                ledger.append(Record(step=i, tag=f"{name}-{i}", before=o, after=o, timestamp=t))
        step += 1
    return ledger


# -----------------------------------------------------------------------
# Pearson correlation (correlate_pair)
# -----------------------------------------------------------------------

class TestCorrelatePair:
    def test_perfect_positive(self):
        xs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        ys = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        c = correlate_pair(xs, ys, "a", "b")
        assert c is not None
        assert abs(c.coefficient - 1.0) < 1e-10
        assert c.positive is True
        assert c.negative is False

    def test_perfect_negative(self):
        xs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        ys = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
        c = correlate_pair(xs, ys, "a", "b")
        assert abs(c.coefficient - (-1.0)) < 1e-10
        assert c.negative is True

    def test_no_correlation(self):
        xs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        ys = [5, 3, 8, 1, 9, 2, 7, 4, 6, 10]  # ~random
        c = correlate_pair(xs, ys, "a", "b")
        assert c.strength < 0.7

    def test_insufficient_data(self):
        c = correlate_pair([1, 2], [3, 4], "a", "b")
        assert c is None

    def test_constant_series(self):
        xs = [5, 5, 5, 5, 5]
        ys = [1, 2, 3, 4, 5]
        c = correlate_pair(xs, ys, "a", "b")
        assert c.coefficient == 0.0

    def test_lag_detection(self):
        # A leads B by 2 steps
        a = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
        b = [0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
        c = correlate_pair(a, b, "a", "b", max_lag=4)
        assert c is not None
        assert c.lag > 0  # A leads B
        assert c.a_leads is True

    def test_lag_zero_when_simultaneous(self):
        xs = list(range(10))
        ys = [x * 2 for x in xs]
        c = correlate_pair(xs, ys, "a", "b", max_lag=3)
        assert c.lag == 0
        assert c.simultaneous is True

    def test_serialization_roundtrip(self):
        xs = list(range(10))
        ys = [x * 3 + 1 for x in xs]
        c = correlate_pair(xs, ys, "cpu", "mem")
        d = c.to_dict()
        c2 = Correlation.from_dict(d)
        assert c2.component_a == c.component_a
        assert c2.component_b == c.component_b
        assert c2.coefficient == c.coefficient
        assert c2.lag == c.lag

    def test_repr(self):
        c = correlate_pair(list(range(10)), list(range(10)), "a", "b")
        assert "a ~ b" in repr(c)


# -----------------------------------------------------------------------
# Correlation matrix (correlate)
# -----------------------------------------------------------------------

class TestCorrelate:
    def test_three_components(self):
        n = 20
        a = list(range(n))
        b = [x * 2 for x in a]  # perfectly correlated with a
        c = [10 - x * 0.1 for x in a]  # weakly anti-correlated

        matrix = correlate({"a": a, "b": b, "c": c}, min_correlation=0.5)
        assert len(matrix.components) == 3
        assert len(matrix.correlations) >= 1  # a-b should be found

        # a and b should be strongly correlated
        ab = [corr for corr in matrix.correlations
              if {corr.component_a, corr.component_b} == {"a", "b"}]
        assert len(ab) == 1
        assert ab[0].strength > 0.99

    def test_min_correlation_filter(self):
        n = 20
        a = list(range(n))
        b = [x * 2 for x in a]
        c = [5] * n  # constant, r=0 with everything

        matrix = correlate({"a": a, "b": b, "c": c}, min_correlation=0.9)
        # Only a-b should survive the 0.9 threshold
        assert all(
            {c.component_a, c.component_b} == {"a", "b"}
            for c in matrix.correlations
        )

    def test_strongest(self):
        n = 20
        a = list(range(n))
        b = [x * 2 for x in a]
        c = [x * -1 for x in a]

        matrix = correlate({"a": a, "b": b, "c": c}, min_correlation=0.5)
        top = matrix.strongest(1)
        assert len(top) == 1
        assert top[0].strength > 0.99

    def test_for_component(self):
        n = 20
        a = list(range(n))
        b = [x * 2 for x in a]
        c = [x * -1 for x in a]

        matrix = correlate({"a": a, "b": b, "c": c}, min_correlation=0.5)
        a_corrs = matrix.for_component("a")
        assert all("a" in {c.component_a, c.component_b} for c in a_corrs)

    def test_coefficient_lookup(self):
        n = 20
        a = list(range(n))
        b = [x * 2 for x in a]

        matrix = correlate({"a": a, "b": b}, min_correlation=0.5)
        r = matrix.coefficient("a", "b")
        assert r is not None
        assert abs(r - 1.0) < 1e-10
        # Reverse order should also work
        r2 = matrix.coefficient("b", "a")
        assert r2 is not None

    def test_empty_when_no_strong_correlations(self):
        import random
        random.seed(99)
        matrix = correlate({
            "a": [random.random() for _ in range(20)],
            "b": [random.random() for _ in range(20)],
        }, min_correlation=0.9)
        assert len(matrix.correlations) == 0

    def test_to_dict(self):
        n = 20
        matrix = correlate({"a": list(range(n)), "b": list(range(n))}, min_correlation=0.5)
        d = matrix.to_dict()
        assert "components" in d
        assert "correlations" in d


# -----------------------------------------------------------------------
# Causal graph integration
# -----------------------------------------------------------------------

class TestCausalIntegration:
    def test_to_causal_link_simultaneous(self):
        c = Correlation("cpu", "mem", 0.85, lag=0, n_samples=20, confidence=Confidence.HIGH)
        link = c.to_causal_link()
        assert link.cause_type == CauseType.CORRELATES
        assert link.strength == 0.85
        assert "auto-correlated" in link.evidence

    def test_to_causal_link_with_lag(self):
        c = Correlation("cpu", "mem", 0.9, lag=2, n_samples=20, confidence=Confidence.HIGH)
        link = c.to_causal_link()
        assert link.source == "cpu"  # cpu leads
        assert link.target == "mem"

    def test_to_causal_link_negative_lag(self):
        c = Correlation("cpu", "mem", 0.9, lag=-2, n_samples=20, confidence=Confidence.HIGH)
        link = c.to_causal_link()
        assert link.source == "mem"  # mem leads (negative lag)
        assert link.target == "cpu"

    def test_to_causal_link_negative_correlation(self):
        c = Correlation("cpu", "mem", -0.9, lag=1, n_samples=20, confidence=Confidence.HIGH)
        link = c.to_causal_link()
        assert link.cause_type == CauseType.DEGRADES  # inverse relationship with lead

    def test_to_causal_graph(self):
        n = 20
        a = list(range(n))
        b = [x * 2 for x in a]

        matrix = correlate({"a": a, "b": b}, min_correlation=0.5)
        graph = matrix.to_causal_graph()
        assert len(graph.links) >= 1
        assert graph.links[0].cause_type == CauseType.CORRELATES

    def test_merge_with_existing_graph(self):
        existing = CausalGraph()
        existing.add_degrades("db", "api", 0.8, evidence="manual")

        n = 20
        matrix = correlate({"a": list(range(n)), "b": list(range(n))}, min_correlation=0.5)
        graph = matrix.to_causal_graph(existing)
        assert len(graph.links) >= 2  # manual + auto


# -----------------------------------------------------------------------
# Ledger integration
# -----------------------------------------------------------------------

class TestLedgerIntegration:
    def test_correlate_from_ledger(self):
        n = 20
        ledger = _ledger_multi({
            "cpu": list(range(n)),
            "mem": [x * 2 for x in range(n)],
        }, n_steps=n)

        matrix = correlate_from_ledger(ledger, min_correlation=0.5)
        assert len(matrix.correlations) >= 1

    def test_auto_causal_graph(self):
        n = 20
        ledger = _ledger_multi({
            "cpu": list(range(n)),
            "mem": [x * 2 for x in range(n)],
            "disk": [50] * n,  # constant, no correlation
        }, n_steps=n)

        graph = auto_causal_graph(ledger, min_correlation=0.7)
        assert len(graph.links) >= 1
        # cpu-mem should be linked, disk should not
        linked = {(l.source, l.target) for l in graph.links}
        linked_flat = {name for pair in linked for name in pair}
        assert "cpu" in linked_flat or "mem" in linked_flat

    def test_auto_causal_graph_merges(self):
        existing = CausalGraph()
        existing.add_degrades("external", "cpu", 0.5)

        n = 20
        ledger = _ledger_multi({
            "cpu": list(range(n)),
            "mem": [x * 2 for x in range(n)],
        }, n_steps=n)

        graph = auto_causal_graph(ledger, existing=existing, min_correlation=0.5)
        assert len(graph.links) >= 2  # manual + discovered

    def test_insufficient_data(self):
        ledger = _ledger_multi({"cpu": [1, 2], "mem": [3, 4]}, n_steps=2)
        matrix = correlate_from_ledger(ledger, min_correlation=0.5)
        assert len(matrix.correlations) == 0


# -----------------------------------------------------------------------
# Predicates
# -----------------------------------------------------------------------

class TestPredicates:
    def _expr(self):
        return Expression()

    def test_correlated_with(self):
        n = 20
        ledger = _ledger_multi({
            "cpu": list(range(n)),
            "mem": [x * 2 for x in range(n)],
        }, n_steps=n)

        pred = correlated_with("cpu", "mem", ledger, min_correlation=0.5)
        assert pred(self._expr()) is True

    def test_correlated_with_negative(self):
        n = 20
        ledger = _ledger_multi({
            "cpu": list(range(n)),
            "disk": [50] * n,
        }, n_steps=n)

        pred = correlated_with("cpu", "disk", ledger, min_correlation=0.5)
        assert pred(self._expr()) is False

    def test_any_new_correlation(self):
        n = 20
        ledger = _ledger_multi({
            "cpu": list(range(n)),
            "mem": [x * 2 for x in range(n)],
        }, n_steps=n)

        # Empty baseline — everything is new
        empty_graph = CausalGraph()
        pred = any_new_correlation(ledger, empty_graph, min_correlation=0.5)
        assert pred(self._expr()) is True

    def test_any_new_correlation_all_known(self):
        n = 20
        ledger = _ledger_multi({
            "cpu": list(range(n)),
            "mem": [x * 2 for x in range(n)],
        }, n_steps=n)

        # Baseline already has cpu-mem
        known = CausalGraph()
        known.add(CausalLink("cpu", "mem", CauseType.CORRELATES))
        pred = any_new_correlation(ledger, known, min_correlation=0.5)
        assert pred(self._expr()) is False


# -----------------------------------------------------------------------
# Confidence
# -----------------------------------------------------------------------

class TestConfidence:
    def test_high_confidence(self):
        xs = list(range(30))
        ys = [x * 2 for x in xs]
        c = correlate_pair(xs, ys, "a", "b")
        assert c.confidence >= Confidence.MODERATE

    def test_low_confidence_few_samples(self):
        c = correlate_pair([1, 2, 3, 4, 5], [2, 4, 6, 8, 10], "a", "b")
        assert c.confidence <= Confidence.MODERATE
