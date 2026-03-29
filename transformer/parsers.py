"""
Pre-calibrated parsers for known transformer models.

These are the domain-specific factory functions that pin calibration
data from experimental runs. Each returns a generic.Parser ready to use.
"""

from __future__ import annotations

from typing import Optional

from margin.health import Thresholds
from margin.observation import Parser


def make_pythia_parser() -> Parser:
    """
    Pre-configured Parser for pythia-6.9b IOI + NM + IH + SH circuits.

    Baselines and layers from v13 training (A100x80GB, March 2026):
      IOI: logit-gap baseline ~5.0, measured at L26
      NM:  name-mover contribution ~4.2, measured at L22
      IH:  induction-head baseline ~2.8, measured at L1
      SH:  suppressor-head baseline ~0.25, measured at L14

    SH calibration (from v18 run, 2026-03-27):
      SH gap = s_logprob(SH_ablated) - s_logprob(intact).
      Positive = SH actively suppressing S token (healthy).
      Step-0 only: SH gap meaningful only at step 0.
      Baseline 0.25 (training mean). INTACT >= 0.40. ABLATED < 0.05.

    Default thresholds: INTACT=3.5, ABLATED=1.5 (pythia-6.9b IOI).
    All circuits are higher-is-better (logit gaps).
    """
    return Parser(
        baselines={"IOI": 5.0, "NM": 4.2, "IH": 2.8, "SH": 0.25},
        thresholds=Thresholds(intact=3.5, ablated=1.5),
        component_thresholds={
            "SH": Thresholds(intact=0.40, ablated=0.05),
        },
    )


def make_from_sweep(
    sweep_results: dict[str, dict],
    thresholds: Optional[Thresholds] = None,
) -> Parser:
    """
    Build a calibrated Parser from a verification sweep output.

    Model-agnostic entry point. Run a circuit sweep on any
    HookedTransformer-compatible model, collect the mean intact logit gap
    per circuit type, then pass those results here.

    Args:
        sweep_results: {circuit_name: {"baseline_gap": float, "layer": int}}
            e.g. {"IOI": {"baseline_gap": 3.1, "layer": 19},
                  "NM":  {"baseline_gap": 2.8, "layer": 17}}
        thresholds: Override thresholds. If None, derives from sweep data:
            INTACT  = 0.70 * min(baseline_gap)
            ABLATED = 0.30 * min(baseline_gap)
    """
    baselines = {name: info["baseline_gap"] for name, info in sweep_results.items()}

    if thresholds is None and baselines:
        min_bl = min(baselines.values())
        thresholds = Thresholds(
            intact=round(0.70 * min_bl, 2),
            ablated=round(0.30 * min_bl, 2),
        )

    return Parser(
        baselines=baselines,
        thresholds=thresholds or Thresholds(intact=3.5, ablated=1.5),
    )
