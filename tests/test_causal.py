import pytest
from margin.causal import (
    CauseType, CausalLink, CausalGraph,
    CauseExplanation, Explanation,
)
from margin.observation import Observation, Expression
from margin.health import Health
from margin.confidence import Confidence


def _obs(name, health, value=50.0):
    return Observation(name, health, value, 100.0, Confidence.HIGH)


def _expr(*observations):
    return Expression(observations=list(observations), confidence=Confidence.HIGH)


class TestCausalLink:
    def test_roundtrip(self):
        link = CausalLink("db", "api", CauseType.DEGRADES, 0.9,
                          condition=Health.ABLATED, evidence="db outage kills api")
        d = link.to_dict()
        link2 = CausalLink.from_dict(d)
        assert link2.source == "db"
        assert link2.target == "api"
        assert link2.cause_type == CauseType.DEGRADES
        assert link2.strength == 0.9
        assert link2.condition == Health.ABLATED

    def test_repr(self):
        link = CausalLink("db", "api", CauseType.DEGRADES)
        assert "DEGRADES" in repr(link)
        assert "db" in repr(link)

    def test_conditional_repr(self):
        link = CausalLink("db", "api", CauseType.DEGRADES, condition=Health.ABLATED)
        assert "ABLATED" in repr(link)


class TestCausalGraph:
    def _make_graph(self):
        g = CausalGraph()
        g.add_degrades("db", "api", 0.9, evidence="db outage kills api")
        g.add_degrades("api", "frontend", 0.7)
        g.add_blocks("cache", "api", 0.5, condition=Health.ABLATED)
        g.add_mitigates("cdn", "frontend", 0.3)
        return g

    def test_causes_of(self):
        g = self._make_graph()
        causes = g.causes_of("api")
        assert len(causes) == 2
        sources = {c.source for c in causes}
        assert sources == {"db", "cache"}

    def test_effects_of(self):
        g = self._make_graph()
        effects = g.effects_of("api")
        assert len(effects) == 1
        assert effects[0].target == "frontend"

    def test_upstream(self):
        g = self._make_graph()
        up = g.upstream("frontend")
        assert "api" in up
        assert "db" in up

    def test_downstream(self):
        g = self._make_graph()
        down = g.downstream("db")
        assert "api" in down
        assert "frontend" in down

    def test_roots(self):
        g = self._make_graph()
        assert "db" in g.roots()
        assert "cache" in g.roots()
        assert "cdn" in g.roots()

    def test_leaves(self):
        g = self._make_graph()
        assert "frontend" in g.leaves()

    def test_components(self):
        g = self._make_graph()
        assert set(g.components()) == {"db", "api", "frontend", "cache", "cdn"}

    def test_explain_with_expression(self):
        g = self._make_graph()
        expr = _expr(
            _obs("db", Health.ABLATED, 10.0),
            _obs("api", Health.DEGRADED),
            _obs("cache", Health.INTACT, 90.0),
            _obs("frontend", Health.DEGRADED),
        )
        expl = g.explain("api", expr)
        assert expl.component == "api"
        assert expl.health == Health.DEGRADED
        assert expl.has_known_cause
        # db DEGRADES api (no condition, always active)
        # cache BLOCKS api (condition=ABLATED, but cache is INTACT → not active)
        assert len(expl.causes) == 1
        assert expl.causes[0].source == "db"

    def test_explain_without_expression(self):
        g = CausalGraph()
        g.add_degrades("a", "b")
        expl = g.explain("b")
        assert expl.has_known_cause
        assert expl.health is None

    def test_explain_no_causes(self):
        g = CausalGraph()
        g.add_degrades("a", "b")
        expl = g.explain("a", _expr(_obs("a", Health.DEGRADED)))
        assert not expl.has_known_cause
        assert expl.root_cause is None

    def test_explain_all(self):
        g = CausalGraph()
        g.add_degrades("a", "b")
        g.add_degrades("b", "c")
        expr = _expr(_obs("a", Health.ABLATED), _obs("b", Health.DEGRADED),
                     _obs("c", Health.DEGRADED))
        explanations = g.explain_all(expr)
        assert set(explanations.keys()) == {"a", "b", "c"}
        assert explanations["b"].has_known_cause
        assert explanations["a"].has_known_cause is False

    def test_root_cause_picks_strongest(self):
        g = CausalGraph()
        g.add_degrades("weak", "target", 0.3)
        g.add_degrades("strong", "target", 0.9)
        expl = g.explain("target", _expr(
            _obs("weak", Health.DEGRADED),
            _obs("strong", Health.DEGRADED),
            _obs("target", Health.ABLATED),
        ))
        assert expl.root_cause.source == "strong"

    def test_conditional_link_not_active(self):
        g = CausalGraph()
        g.add_blocks("cache", "api", condition=Health.ABLATED)
        expr = _expr(_obs("cache", Health.INTACT, 90.0), _obs("api", Health.DEGRADED))
        expl = g.explain("api", expr)
        assert not expl.has_known_cause  # cache is INTACT, condition requires ABLATED

    def test_conditional_link_active(self):
        g = CausalGraph()
        g.add_blocks("cache", "api", condition=Health.ABLATED)
        expr = _expr(_obs("cache", Health.ABLATED, 10.0), _obs("api", Health.DEGRADED))
        expl = g.explain("api", expr)
        assert expl.has_known_cause
        assert expl.causes[0].source == "cache"

    def test_roundtrip(self):
        g = CausalGraph()
        g.add_degrades("a", "b", 0.8, evidence="test")
        g.add_blocks("c", "b", condition=Health.ABLATED)
        d = g.to_dict()
        g2 = CausalGraph.from_dict(d)
        assert len(g2.links) == 2
        assert g2.links[0].evidence == "test"
        assert g2.links[1].condition == Health.ABLATED

    def test_to_string(self):
        g = CausalGraph()
        g.add_degrades("a", "b")
        s = g.to_string()
        assert "DEGRADES" in s

    def test_empty_graph(self):
        g = CausalGraph()
        assert g.components() == []
        assert g.roots() == []
        assert g.to_string() == "(no causal links)"

    def test_repr(self):
        g = CausalGraph()
        g.add_degrades("a", "b")
        assert "1 links" in repr(g)


class TestExplanation:
    def test_to_string(self):
        expl = Explanation(
            component="api",
            health=Health.DEGRADED,
            causes=[CauseExplanation("db", Health.ABLATED, CauseType.DEGRADES, 0.9)],
        )
        s = expl.to_string()
        assert "api" in s
        assert "db" in s
        assert "DEGRADES" in s

    def test_to_dict(self):
        expl = Explanation(
            component="api",
            health=Health.DEGRADED,
            causes=[CauseExplanation("db", Health.ABLATED, CauseType.DEGRADES, 0.9, "outage")],
        )
        d = expl.to_dict()
        assert d["component"] == "api"
        assert d["has_known_cause"] is True
        assert d["root_cause"]["source"] == "db"
