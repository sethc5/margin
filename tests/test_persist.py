"""Tests for margin.persist — save/load and replay."""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from margin import (
    Parser, Thresholds, Monitor,
    save_monitor, load_monitor, replay, replay_csv,
    DriftState, AnomalyState,
)


def _parser():
    return Parser(
        baselines={"cpu": 50.0, "mem": 70.0},
        thresholds=Thresholds(intact=40.0, ablated=10.0),
    )


t0 = datetime(2026, 1, 1)


class TestSaveLoad:
    def test_roundtrip(self):
        p = _parser()
        m = Monitor(p, window=50)
        for i in range(15):
            m.update({"cpu": 50.0 - i, "mem": 70.0}, now=t0 + timedelta(seconds=i * 60))

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            save_monitor(m, f.name)
            m2 = load_monitor(f.name, p, window=50)
            Path(f.name).unlink()

        assert m2.step == m.step
        assert m2._drift_trackers["cpu"].n_observations == m._drift_trackers["cpu"].n_observations
        assert m2._anomaly_trackers["cpu"].n_values == m._anomaly_trackers["cpu"].n_values

    def test_drift_restored(self):
        p = _parser()
        m = Monitor(p, window=50)
        for i in range(15):
            m.update({"cpu": 50.0 - i * 2, "mem": 70.0}, now=t0 + timedelta(seconds=i * 60))

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            save_monitor(m, f.name)
            m2 = load_monitor(f.name, p, window=50)
            Path(f.name).unlink()

        # Drift should be reclassified from restored observations
        dc = m2._drift_trackers["cpu"].classification
        assert dc is not None
        assert dc.direction == m._drift_trackers["cpu"].classification.direction

    def test_anomaly_values_restored(self):
        p = _parser()
        m = Monitor(p, window=50)
        for i in range(20):
            m.update({"cpu": 50.0, "mem": 70.0}, now=t0 + timedelta(seconds=i * 60))

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            save_monitor(m, f.name)
            m2 = load_monitor(f.name, p, window=50)
            Path(f.name).unlink()

        assert m2._anomaly_trackers["cpu"].n_values == 20

    def test_correlation_restored(self):
        p = _parser()
        m = Monitor(p, window=50)
        for i in range(15):
            m.update({"cpu": float(i), "mem": float(i * 2)}, now=t0 + timedelta(seconds=i * 60))

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            save_monitor(m, f.name)
            m2 = load_monitor(f.name, p, window=50)
            Path(f.name).unlink()

        assert m2._correlation_tracker.n_updates == m._correlation_tracker.n_updates

    def test_empty_monitor(self):
        p = _parser()
        m = Monitor(p)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            save_monitor(m, f.name)
            m2 = load_monitor(f.name, p)
            Path(f.name).unlink()

        assert m2.step == 0

    def test_file_is_valid_json(self):
        p = _parser()
        m = Monitor(p)
        m.update({"cpu": 42.0, "mem": 60.0})

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            save_monitor(m, f.name)
            data = json.loads(Path(f.name).read_text())
            Path(f.name).unlink()

        assert "step" in data
        assert "drift" in data
        assert "anomaly" in data
        assert "correlation" in data


class TestReplay:
    def test_basic_replay(self):
        p = _parser()
        data = [{"cpu": 50.0 - i, "mem": 70.0} for i in range(20)]
        monitor, snapshots = replay(p, data)

        assert len(snapshots) == 20
        assert monitor.step == 20
        assert all("step" in s for s in snapshots)

    def test_replay_with_timestamps(self):
        p = _parser()
        data = [{"cpu": 50.0, "mem": 70.0}] * 10
        timestamps = [t0 + timedelta(seconds=i * 60) for i in range(10)]
        monitor, snapshots = replay(p, data, timestamps=timestamps)
        assert monitor.step == 10

    def test_replay_drift_detection(self):
        p = _parser()
        data = [{"cpu": 50.0 - i * 2, "mem": 70.0} for i in range(20)]
        monitor, _ = replay(p, data)
        dc = monitor.drift("cpu")
        assert dc is not None
        assert dc.direction.value == "WORSENING"

    def test_replay_anomaly_detection(self):
        p = _parser()
        data = [{"cpu": 50.0, "mem": 70.0}] * 20 + [{"cpu": 200.0, "mem": 70.0}]
        monitor, _ = replay(p, data)
        ac = monitor.anomaly("cpu")
        assert ac is not None
        assert ac.state in (AnomalyState.ANOMALOUS, AnomalyState.NOVEL)

    def test_replay_empty(self):
        p = _parser()
        monitor, snapshots = replay(p, [])
        assert len(snapshots) == 0
        assert monitor.step == 0


class TestReplayCsv:
    def test_csv_replay(self):
        p = _parser()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("cpu,mem\n")
            for i in range(20):
                f.write(f"{50.0 - i},{70.0}\n")
            f.flush()
            monitor, snapshots = replay_csv(p, f.name)
            Path(f.name).unlink()

        assert len(snapshots) == 20
        assert monitor.step == 20

    def test_csv_with_timestamps(self):
        p = _parser()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("timestamp,cpu,mem\n")
            for i in range(10):
                t = (t0 + timedelta(seconds=i * 60)).isoformat()
                f.write(f"{t},{50.0},{70.0}\n")
            f.flush()
            monitor, snapshots = replay_csv(p, f.name, timestamp_column="timestamp")
            Path(f.name).unlink()

        assert len(snapshots) == 10

    def test_csv_skips_bad_values(self):
        p = _parser()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("cpu,mem\n")
            f.write("50.0,70.0\n")
            f.write("bad,70.0\n")  # bad cpu value
            f.write("45.0,70.0\n")
            f.flush()
            monitor, snapshots = replay_csv(p, f.name)
            Path(f.name).unlink()

        # Should have at least 2 valid rows (bad row has only mem)
        assert len(snapshots) >= 2


class TestCLI:
    def test_main_help(self):
        """Just verify the CLI module loads without error."""
        from margin.__main__ import main
        assert callable(main)

    def test_status_command(self):
        """Test one-shot status via config."""
        import subprocess
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({
                "components": {
                    "cpu": {"baseline": 50.0, "intact": 80.0, "ablated": 30.0},
                },
            }, f)
            f.flush()
            result = subprocess.run(
                ["python", "-m", "margin", "status", "--config", f.name, "cpu=48"],
                capture_output=True, text=True, timeout=10,
            )
            Path(f.name).unlink()

        assert result.returncode == 0
        assert "cpu" in result.stdout
        assert "INTACT" in result.stdout or "DEGRADED" in result.stdout
