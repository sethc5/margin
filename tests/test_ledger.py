import pytest
from margin.observation import Op, Observation, Correction
from margin.health import Health
from margin.confidence import Confidence
from margin.ledger import Record, Ledger


def _obs(name, health, value, baseline, hib=True):
    return Observation(name, health, value, baseline, Confidence.HIGH, higher_is_better=hib)


class TestRecordImprovement:
    def test_positive_for_higher_is_better(self):
        r = Record(
            step=0, tag="t",
            before=_obs("x", Health.DEGRADED, 50.0, 100.0),
            after=_obs("x", Health.INTACT, 90.0, 100.0),
            fired=True, op=Op.RESTORE,
        )
        assert r.improvement == 40.0

    def test_positive_for_lower_is_better(self):
        r = Record(
            step=0, tag="t",
            before=_obs("err", Health.DEGRADED, 0.08, 0.01, hib=False),
            after=_obs("err", Health.RECOVERING, 0.03, 0.01, hib=False),
            fired=True, op=Op.RESTORE,
        )
        assert r.improvement == pytest.approx(0.05)

    def test_zero_when_not_fired(self):
        r = Record(step=0, tag="t", before=_obs("x", Health.INTACT, 90.0, 100.0))
        assert r.improvement == 0.0

    def test_negative_for_harmful_correction(self):
        r = Record(
            step=0, tag="t",
            before=_obs("x", Health.DEGRADED, 50.0, 100.0),
            after=_obs("x", Health.ABLATED, 20.0, 100.0),
            fired=True, op=Op.RESTORE,
        )
        assert r.improvement == -30.0
        assert r.was_beneficial() is False


class TestRecordRecoveryRatio:
    def test_fully_restored_higher_is_better(self):
        r = Record(
            step=0, tag="t",
            before=_obs("x", Health.DEGRADED, 50.0, 100.0),
            after=_obs("x", Health.INTACT, 100.0, 100.0),
            fired=True, op=Op.RESTORE,
        )
        assert r.recovery_ratio == pytest.approx(1.0)

    def test_fully_restored_lower_is_better(self):
        r = Record(
            step=0, tag="t",
            before=_obs("err", Health.DEGRADED, 0.08, 0.01, hib=False),
            after=_obs("err", Health.INTACT, 0.01, 0.01, hib=False),
            fired=True, op=Op.RESTORE,
        )
        assert r.recovery_ratio == pytest.approx(1.0)

    def test_partial_recovery_lower_is_better(self):
        r = Record(
            step=0, tag="t",
            before=_obs("err", Health.DEGRADED, 0.08, 0.01, hib=False),
            after=_obs("err", Health.RECOVERING, 0.04, 0.01, hib=False),
            fired=True, op=Op.RESTORE,
        )
        # baseline/after = 0.01/0.04 = 0.25
        assert r.recovery_ratio == pytest.approx(0.25)

    def test_zero_baseline(self):
        r = Record(
            step=0, tag="t",
            before=_obs("x", Health.DEGRADED, 50.0, 0.0),
            after=_obs("x", Health.INTACT, 90.0, 0.0),
            fired=True, op=Op.RESTORE,
        )
        assert r.recovery_ratio == 0.0


class TestRecordRoundtrip:
    def test_roundtrip(self):
        r = Record(
            step=3, tag="spike",
            before=_obs("err", Health.DEGRADED, 0.08, 0.01, hib=False),
            after=_obs("err", Health.RECOVERING, 0.03, 0.01, hib=False),
            fired=True, op=Op.RESTORE, alpha=0.5, magnitude=1.0,
        )
        rr = Record.from_dict(r.to_dict())
        assert rr.step == 3
        assert rr.tag == "spike"
        assert rr.fired is True
        assert rr.op == Op.RESTORE
        assert rr.before.higher_is_better is False
        assert rr.after.health == Health.RECOVERING
        assert abs(rr.improvement - r.improvement) < 0.001

    def test_roundtrip_no_after(self):
        r = Record(step=0, tag="skip", before=_obs("x", Health.INTACT, 90.0, 100.0))
        rr = Record.from_dict(r.to_dict())
        assert rr.after is None
        assert rr.fired is False


class TestLedger:
    def _make_ledger(self):
        ledger = Ledger(label="test")
        ledger.append(Record(
            step=0, tag="skip",
            before=_obs("x", Health.INTACT, 90.0, 100.0),
        ))
        ledger.append(Record(
            step=1, tag="fire",
            before=_obs("x", Health.DEGRADED, 50.0, 100.0),
            after=_obs("x", Health.RECOVERING, 75.0, 100.0),
            fired=True, op=Op.RESTORE, alpha=0.5, magnitude=1.0,
        ))
        ledger.append(Record(
            step=2, tag="fire",
            before=_obs("x", Health.RECOVERING, 75.0, 100.0),
            after=_obs("x", Health.INTACT, 90.0, 100.0),
            fired=True, op=Op.RESTORE, alpha=0.3, magnitude=0.5,
        ))
        return ledger

    def test_len(self):
        assert len(self._make_ledger()) == 3

    def test_n_fired(self):
        assert self._make_ledger().n_fired == 2

    def test_fire_rate(self):
        assert self._make_ledger().fire_rate == pytest.approx(2 / 3)

    def test_mean_improvement(self):
        ledger = self._make_ledger()
        # step 1: 75-50=25, step 2: 90-75=15, mean=20
        assert ledger.mean_improvement == pytest.approx(20.0)

    def test_harmful_empty_for_good_corrections(self):
        assert len(self._make_ledger().harmful()) == 0

    def test_harmful_detects_bad(self):
        ledger = Ledger()
        ledger.append(Record(
            step=0, tag="bad",
            before=_obs("x", Health.DEGRADED, 50.0, 100.0),
            after=_obs("x", Health.ABLATED, 20.0, 100.0),
            fired=True, op=Op.RESTORE,
        ))
        assert len(ledger.harmful()) == 1

    def test_empty_ledger_safe(self):
        ledger = Ledger()
        assert ledger.fire_rate == 0.0
        assert ledger.mean_improvement == 0.0
        assert ledger.mean_recovery == 0.0
        assert len(ledger.harmful()) == 0

    def test_render_produces_lines(self):
        ledger = self._make_ledger()
        lines = ledger.render().split("\n")
        assert len(lines) == 3
        assert "step   0" in lines[0]
        assert "step   2" in lines[2]

    def test_summary_keys(self):
        s = self._make_ledger().summary()
        assert set(s.keys()) == {
            "label", "n_steps", "n_fired", "fire_rate",
            "mean_improvement", "mean_recovery", "n_harmful",
        }


class TestLedgerRoundtrip:
    def test_json_roundtrip(self):
        ledger = Ledger(label="incident")
        ledger.append(Record(
            step=0, tag="a",
            before=_obs("x", Health.DEGRADED, 50.0, 100.0),
            after=_obs("x", Health.INTACT, 90.0, 100.0),
            fired=True, op=Op.RESTORE, alpha=0.5, magnitude=1.0,
        ))
        ledger.append(Record(
            step=1, tag="b",
            before=_obs("x", Health.INTACT, 90.0, 100.0),
        ))

        lr = Ledger.from_json(ledger.to_json())
        assert lr.label == "incident"
        assert len(lr) == 2
        assert lr.n_fired == 1
        assert lr.records[0].op == Op.RESTORE
        assert lr.records[1].after is None
        assert lr.render() == ledger.render()
