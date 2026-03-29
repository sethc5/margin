import pytest
from margin.composite import CompositeObservation, AggregateStrategy
from margin.observation import Observation
from margin.health import Health
from margin.confidence import Confidence


def _obs(name, health, value, baseline=100.0, conf=Confidence.HIGH):
    return Observation(name, health, value, baseline, conf)


class TestCompositeObservation:
    def test_worst_strategy(self):
        c = CompositeObservation(
            name="latency",
            sub_observations=[
                _obs("p50", Health.INTACT, 90.0),
                _obs("p95", Health.DEGRADED, 50.0),
                _obs("p99", Health.ABLATED, 10.0),
            ],
            strategy=AggregateStrategy.WORST,
        )
        assert c.health == Health.ABLATED

    def test_best_strategy(self):
        c = CompositeObservation(
            name="latency",
            sub_observations=[
                _obs("p50", Health.INTACT, 90.0),
                _obs("p95", Health.DEGRADED, 50.0),
                _obs("p99", Health.ABLATED, 10.0),
            ],
            strategy=AggregateStrategy.BEST,
        )
        assert c.health == Health.INTACT

    def test_majority_strategy(self):
        c = CompositeObservation(
            name="latency",
            sub_observations=[
                _obs("p50", Health.INTACT, 90.0),
                _obs("p95", Health.DEGRADED, 50.0),
                _obs("p99", Health.DEGRADED, 40.0),
            ],
            strategy=AggregateStrategy.MAJORITY,
        )
        assert c.health == Health.DEGRADED

    def test_empty_is_ood(self):
        c = CompositeObservation(name="empty")
        assert c.health == Health.OOD
        assert c.confidence == Confidence.INDETERMINATE

    def test_confidence_is_weakest(self):
        c = CompositeObservation(
            name="x",
            sub_observations=[
                _obs("a", Health.INTACT, 90.0, conf=Confidence.HIGH),
                _obs("b", Health.INTACT, 85.0, conf=Confidence.LOW),
            ],
        )
        assert c.confidence == Confidence.LOW

    def test_worst_property(self):
        c = CompositeObservation(
            name="x",
            sub_observations=[
                _obs("a", Health.INTACT, 90.0),
                _obs("b", Health.ABLATED, 10.0),
            ],
        )
        assert c.worst.name == "b"

    def test_best_property(self):
        c = CompositeObservation(
            name="x",
            sub_observations=[
                _obs("a", Health.INTACT, 90.0),
                _obs("b", Health.ABLATED, 10.0),
            ],
        )
        assert c.best.name == "a"

    def test_as_observation(self):
        c = CompositeObservation(
            name="latency",
            sub_observations=[
                _obs("p50", Health.INTACT, 90.0),
                _obs("p99", Health.DEGRADED, 50.0),
            ],
        )
        flat = c.as_observation()
        assert flat.name == "latency"
        assert flat.health == Health.DEGRADED  # worst
        assert flat.value == 50.0  # worst sub-obs value

    def test_as_observation_empty(self):
        c = CompositeObservation(name="empty")
        flat = c.as_observation()
        assert flat.health == Health.OOD

    def test_to_atom(self):
        c = CompositeObservation(
            name="latency",
            sub_observations=[_obs("p50", Health.INTACT, 90.0)],
        )
        atom = c.to_atom()
        assert "latency" in atom
        assert "INTACT" in atom
        assert "1 sub" in atom

    def test_roundtrip(self):
        c = CompositeObservation(
            name="latency",
            sub_observations=[
                _obs("p50", Health.INTACT, 90.0),
                _obs("p99", Health.DEGRADED, 50.0),
            ],
            strategy=AggregateStrategy.WORST,
        )
        c2 = CompositeObservation.from_dict(c.to_dict())
        assert c2.name == "latency"
        assert c2.health == Health.DEGRADED
        assert len(c2.sub_observations) == 2
        assert c2.strategy == AggregateStrategy.WORST
