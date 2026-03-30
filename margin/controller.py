"""
Controller: adaptive gain / alpha adjustment for Layer 2/3 feedback loops.

A Controller computes the next value of a scalar parameter (``alpha``) based
on the current system observations.  The canonical use-case is adjusting a
session confidence multiplier or circuit-breaker gain after each step.

Usage::

    ctrl = Controller(strategy="proportional_asymmetric", kp=0.3, target=0.5)
    alpha_next, reason = ctrl.step(alpha, expr.observations)

Warm-start from a Monitor fingerprint (condenses ~30 lines of boilerplate)::

    fp = monitor.fingerprint()
    ctrl = Controller.from_fingerprint(fp, "recovery_ratio", kp=0.3, cold_target=0.5)
    alpha_next, reason = ctrl.step(alpha, expr.observations)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .observation import Observation


# Minimum observations before from_fingerprint trusts the fingerprint data.
_WARM_MIN_N = 10


class Controller:
    """
    Adaptive scalar controller for incremental feedback loops.

    Strategies
    ----------
    ``"proportional_asymmetric"`` (default)
        Treats the mean sigma of ``observations`` as the performance metric.

        * metric ≥ 0 → ``alpha += kp * metric``  (proportional nudge upward)
        * metric < 0 → ``alpha *= 0.90``          (hard multiplicative backoff)

        The asymmetry reflects a common control preference: slow ramp-up,
        fast backoff when the system goes below baseline.

    Parameters
    ----------
    strategy:   name of the update rule (currently only ``"proportional_asymmetric"``)
    kp:         proportional gain (how aggressively to increase alpha on good signals)
    target:     initial / warm-start value for alpha — used by callers as the
                starting point before the first ``step()`` call
    backoff:    multiplicative factor applied on negative metric (default 0.90)
    """

    def __init__(
        self,
        strategy: str = "proportional_asymmetric",
        kp: float = 0.3,
        target: float = 0.5,
        backoff: float = 0.90,
    ):
        if strategy != "proportional_asymmetric":
            raise ValueError(
                f"Unknown strategy {strategy!r}. "
                "Currently supported: 'proportional_asymmetric'"
            )
        self.strategy = strategy
        self.kp = kp
        self.target = target
        self.backoff = backoff

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_fingerprint(
        cls,
        fp,
        component: str,
        kp: float,
        cold_target: float,
        strategy: str = "proportional_asymmetric",
        backoff: float = 0.90,
        min_n: int = _WARM_MIN_N,
    ) -> "Controller":
        """
        Build a Controller warm-started from a Monitor fingerprint.

        If ``fp`` has >= ``min_n`` observations for ``component``, the
        ``target`` is set to ``fp.robust_target(component)`` (median by
        default — more noise-resistant than mean when std is high).

        Falls back to ``cold_target`` when the fingerprint has insufficient
        data (new session, component not yet warm, etc.).

        This condenses ~30 lines of per-session calibration boilerplate::

            ctrl = Controller.from_fingerprint(
                fp, "recovery_ratio", kp=0.3, cold_target=0.5
            )
            # instead of:
            #   n = fp.get("recovery_ratio", {}).get("n", 0)
            #   if n >= 10:
            #       vals = [...]  # re-extract from tracker
            #       target = sorted(vals)[len(vals)//2]  # median
            #   else:
            #       target = 0.5
            #   alpha_min, alpha_max = ...
            #   ctrl = MyPController(kp=0.3, target=target)

        Parameters
        ----------
        fp:           Fingerprint (or plain dict) from Monitor.fingerprint()
        component:    which component's statistics to use for calibration
        kp:           proportional gain
        cold_target:  fallback target when data is insufficient
        strategy:     update strategy (default "proportional_asymmetric")
        backoff:      multiplicative factor on negative metric (default 0.90)
        min_n:        minimum observations required to use warm target (default 10)
        """
        target = cold_target

        stats = fp.get(component) if hasattr(fp, "get") else None
        if stats is not None:
            n = stats.get("n", 0) if isinstance(stats, dict) else 0
            if n >= min_n:
                # Use robust_target if available (Fingerprint), else mean
                if hasattr(fp, "robust_target"):
                    target = fp.robust_target(component)
                else:
                    target = stats.get("mean", cold_target)

        return cls(strategy=strategy, kp=kp, target=target, backoff=backoff)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self,
        alpha: float,
        observations,
        alpha_min: float = 0.0,
        alpha_max: float = 1.0,
    ) -> tuple[float, str]:
        """
        Compute the next alpha given the current system observations.

        Parameters
        ----------
        alpha:        current alpha value
        observations: list of Observation objects  → metric = mean(sigma)
                      OR a scalar float            → used directly as metric
        alpha_min:    lower clamp bound (default 0.0)
        alpha_max:    upper clamp bound (default 1.0)

        Returns
        -------
        (alpha_next, reason)
            ``alpha_next`` is clamped to [alpha_min, alpha_max].
            ``reason`` is a short human-readable string explaining the update.
        """
        metric = self._metric(observations)

        if metric >= 0:
            alpha_next = alpha + self.kp * metric
            reason = f"P({metric:+.3f}): {alpha:.3f}→{min(alpha_next, alpha_max):.3f}"
        else:
            alpha_next = alpha * self.backoff
            reason = f"backoff({metric:.3f}): {alpha:.3f}→{max(alpha_next, alpha_min):.3f}"

        alpha_next = max(alpha_min, min(alpha_max, alpha_next))
        return alpha_next, reason

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _metric(self, observations) -> float:
        """Extract a scalar metric from observations or pass through a scalar."""
        if isinstance(observations, (int, float)):
            return float(observations)
        # list of Observation objects — use mean sigma
        sigmas = []
        for obs in observations:
            s = getattr(obs, "sigma", None)
            if s is not None:
                sigmas.append(s)
        if not sigmas:
            return 0.0
        return sum(sigmas) / len(sigmas)

    def __repr__(self) -> str:
        return (
            f"Controller(strategy={self.strategy!r}, kp={self.kp}, "
            f"target={self.target:.3f}, backoff={self.backoff})"
        )
