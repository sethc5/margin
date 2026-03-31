"""
Controller: adaptive gain / alpha adjustment for Layer 2/3 feedback loops.

A Controller computes the next value of a scalar parameter (``alpha``) based
on the current system observations.  The canonical use-case is adjusting a
session confidence multiplier or circuit-breaker gain after each step.

Two strategies
--------------
``"proportional_setpoint"`` (recommended when warm/cold distinction matters)
    Standard P controller — drives alpha toward ``target``::

        alpha_next = alpha + kp * (target - metric)

    ``target`` is the setpoint: the metric value you want to track.
    Warm and cold controllers differ because they have different targets.

``"proportional_asymmetric"``
    Asymmetric P controller — ramp up on good signals, hard backoff on bad::

        metric >= 0:  alpha_next = alpha + kp * metric
        metric <  0:  alpha_next = alpha * backoff

    ``target`` is the *initial alpha value* for the caller to seed with
    ``alpha = ctrl.target`` before the loop — it does not enter step() math.

Normalization
-------------
When passing raw scalar measurements from adapters/proprioception, normalize
against the fingerprint before stepping — otherwise the fingerprint baseline
is never used and warm/cold controllers behave identically::

    # Mean-based normalization
    metric = fp.sigma("recovery_ratio", cq.recovery_ratio)
    alpha, reason = ctrl.step(alpha, metric)

    # Robust normalization (median/IQR — preferred for high-std components)
    metric = fp.robust_sigma("recovery_ratio", cq.recovery_ratio)
    alpha, reason = ctrl.step(alpha, metric)

    # Convenience wrapper (same as above in one call)
    alpha, reason = ctrl.step_normalized(alpha, "recovery_ratio", cq.recovery_ratio, fp)
    alpha, reason = ctrl.step_normalized(alpha, "recovery_ratio", cq.recovery_ratio, fp,
                                         robust=True)

Usage::

    fp = monitor.fingerprint()
    ctrl = Controller.from_fingerprint(fp, "recovery_ratio", kp=0.3, cold_target=0.5,
                                       strategy="proportional_setpoint",
                                       alpha_min=1.0, alpha_max=4.0)
    alpha_next, reason = ctrl.step_normalized(alpha, "recovery_ratio", value, fp)
"""

from __future__ import annotations

import warnings
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .observation import Observation


# Minimum observations before from_fingerprint trusts the fingerprint data.
_WARM_MIN_N = 10


class Controller:
    """
    Adaptive scalar controller for incremental feedback loops.

    Strategies
    ----------
    ``"proportional_setpoint"``
        Standard P controller — drives alpha toward ``target``::

            alpha_next = alpha + kp * (target - metric)

        ``target`` is the setpoint; the controller actively works to make
        the metric equal to target.  Warm and cold controllers (different
        targets) produce different alpha trajectories from the first step.

    ``"proportional_asymmetric"`` (default, original behaviour)
        Asymmetric P controller — ramp up on good signals, hard backoff on bad::

            metric >= 0:  alpha_next = alpha + kp * metric
            metric <  0:  alpha_next = alpha * backoff

        ``target`` here is the *initial alpha value* the caller should use to
        seed the loop: ``alpha = ctrl.target``.  It does not enter step() math.

    Parameters
    ----------
    strategy:   ``"proportional_setpoint"`` or ``"proportional_asymmetric"``
    kp:         proportional gain
    target:     setpoint (``proportional_setpoint``) or initial alpha seed
                (``proportional_asymmetric``)
    backoff:    multiplicative factor on negative metric (``proportional_asymmetric``
                only; ignored by ``proportional_setpoint``)
    alpha_min:  lower clamp bound stored on the controller (default 0.0)
    alpha_max:  upper clamp bound stored on the controller (default 1.0)
    """

    _STRATEGIES = {"proportional_asymmetric", "proportional_setpoint"}

    def __init__(
        self,
        strategy: str = "proportional_asymmetric",
        kp: float = 0.3,
        target: float = 0.5,
        backoff: float = 0.90,
        alpha_min: float = 0.0,
        alpha_max: float = 1.0,
    ):
        if strategy not in self._STRATEGIES:
            raise ValueError(
                f"Unknown strategy {strategy!r}. "
                f"Supported: {sorted(self._STRATEGIES)}"
            )
        self.strategy = strategy
        self.kp = kp
        self.target = target
        self.backoff = backoff
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self._from_fingerprint: bool = False
        self.feature_weights: Optional[list[float]] = None

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
        alpha_min: float = 0.0,
        alpha_max: float = 1.0,
        min_n: int = _WARM_MIN_N,
    ) -> "Controller":
        """
        Build a Controller warm-started from a Monitor fingerprint.

        If ``fp`` has >= ``min_n`` observations for ``component``, the
        ``target`` is set to ``fp.robust_target(component)`` (median by
        default — more noise-resistant than mean when std is high).

        Falls back to ``cold_target`` when the fingerprint has insufficient
        data (new session, component not yet warm, etc.).

        With ``strategy="proportional_setpoint"``, the warm target becomes the
        control setpoint — warm and cold controllers actively drive alpha toward
        different values from the first step.

        With ``strategy="proportional_asymmetric"``, the warm target is the
        starting alpha seed (use ``alpha = ctrl.target`` before the loop).

        **Important**: when passing raw scalar measurements to ``step()``, the
        fingerprint baseline is not applied automatically.  Use
        ``fp.sigma(component, value)`` or ``ctrl.step_normalized()`` to normalize
        raw values against the session distribution before stepping::

            ctrl = Controller.from_fingerprint(
                fp, "recovery_ratio", kp=0.3, cold_target=0.5,
                strategy="proportional_setpoint",
                alpha_min=1.0, alpha_max=4.0,
            )
            # Correct — normalization closes the calibration loop:
            alpha, reason = ctrl.step_normalized(alpha, "recovery_ratio", value, fp)

            # Also correct:
            metric = fp.robust_sigma("recovery_ratio", value)
            alpha, reason = ctrl.step(alpha, metric)

        Parameters
        ----------
        fp:           Fingerprint (or plain dict) from Monitor.fingerprint()
        component:    which component's statistics to use for calibration
        kp:           proportional gain
        cold_target:  fallback target when data is insufficient
        strategy:     update strategy (default "proportional_asymmetric")
        backoff:      multiplicative factor on negative metric (default 0.90)
        alpha_min:    lower bound stored on the controller (default 0.0)
        alpha_max:    upper bound stored on the controller (default 1.0)
        min_n:        minimum observations required to use warm target (default 10)
        """
        target = cold_target

        stats = fp.get(component) if hasattr(fp, "get") else None
        if stats is not None:
            n = stats.get("n", 0) if isinstance(stats, dict) else 0
            if n >= min_n:
                if hasattr(fp, "robust_target"):
                    target = fp.robust_target(component)
                else:
                    target = stats.get("mean", cold_target)

        ctrl = cls(
            strategy=strategy, kp=kp, target=target, backoff=backoff,
            alpha_min=alpha_min, alpha_max=alpha_max,
        )
        ctrl._from_fingerprint = True
        return ctrl

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self,
        alpha: float,
        observations,
        alpha_min: Optional[float] = None,
        alpha_max: Optional[float] = None,
    ) -> tuple[float, str]:
        """
        Compute the next alpha given the current system observations.

        Parameters
        ----------
        alpha:        current alpha value
        observations: list of Observation objects  → metric = mean(sigma)
                      OR a scalar float            → used directly as metric
        alpha_min:    lower clamp bound; overrides stored ``alpha_min`` for
                      this call only (default: use stored)
        alpha_max:    upper clamp bound; overrides stored ``alpha_max`` for
                      this call only (default: use stored)

        Returns
        -------
        (alpha_next, reason)
            ``alpha_next`` is clamped to [alpha_min, alpha_max].
            ``reason`` is a short human-readable string explaining the update.

        Note
        ----
        When this controller was built with :meth:`from_fingerprint` and a
        raw scalar is passed, a ``UserWarning`` is emitted because fingerprint
        normalization is inactive.  Use :meth:`step_normalized` or normalize
        first with ``fp.sigma(component, value)``.
        """
        a_min = alpha_min if alpha_min is not None else self.alpha_min
        a_max = alpha_max if alpha_max is not None else self.alpha_max

        if self._from_fingerprint and isinstance(observations, (int, float)):
            warnings.warn(
                "Controller built from fingerprint but step() received a scalar. "
                "Fingerprint baseline normalization is inactive — warm and cold "
                "controllers will behave identically. "
                "Use ctrl.step_normalized(alpha, component, value, fp) or "
                "normalize first: metric = fp.sigma(component, value).",
                UserWarning,
                stacklevel=2,
            )

        metric = self._metric(observations)
        return self._apply(alpha, metric, a_min, a_max)

    def step_from_features(
        self,
        alpha: float,
        feature_vector: list[float],
        weights: Optional[list[float]] = None,
        alpha_min: Optional[float] = None,
        alpha_max: Optional[float] = None,
    ) -> tuple[float, str]:
        """
        Compute the next alpha from a raw feature vector.

        Converts ``feature_vector`` to a scalar metric via weighted dot product,
        then runs the normal control law.  This enables a soft, continuous
        priority score (e.g. ``[t025_score, community_similarity]``) instead
        of a hard binary threshold gate.

        ``metric = dot(weights, feature_vector)``

        If ``weights`` is None, falls back to ``self.feature_weights`` (set via
        :meth:`with_feature_weights`), then to equal weights (mean of features).

        Parameters
        ----------
        alpha:          current alpha value
        feature_vector: list of floats [f0, f1, ..., fN]
        weights:        per-call weight override; None = use stored weights
        alpha_min:      per-call lower clamp override
        alpha_max:      per-call upper clamp override

        Example::

            ctrl = Controller(strategy="proportional_setpoint", kp=0.3, target=0.5)
            ctrl = ctrl.with_feature_weights([0.6, 0.4])
            alpha, reason = ctrl.step_from_features(
                alpha, [t025_score, community_sim]
            )
        """
        a_min = alpha_min if alpha_min is not None else self.alpha_min
        a_max = alpha_max if alpha_max is not None else self.alpha_max

        _weights = weights or self.feature_weights
        if _weights is not None:
            n = min(len(_weights), len(feature_vector))
            metric = sum(_weights[i] * feature_vector[i] for i in range(n))
        else:
            metric = sum(feature_vector) / len(feature_vector) if feature_vector else 0.0

        reason_prefix = f"features({len(feature_vector)}): metric={metric:+.3f} "
        alpha_next, inner_reason = self._apply(alpha, metric, a_min, a_max)
        return alpha_next, reason_prefix + inner_reason

    def with_feature_weights(self, weights: list[float]) -> "Controller":
        """
        Return a copy of this controller with ``feature_weights`` set.

        Does not mutate the original::

            ctrl = Controller(strategy="proportional_setpoint", kp=0.3, target=0.5,
                              alpha_min=0.0, alpha_max=1.0)
            ctrl = ctrl.with_feature_weights([0.6, 0.4])
        """
        import copy
        clone = copy.copy(self)
        clone.feature_weights = list(weights)
        return clone

    def step_normalized(
        self,
        alpha: float,
        component: str,
        value: float,
        fingerprint,
        robust: bool = False,
        alpha_min: Optional[float] = None,
        alpha_max: Optional[float] = None,
    ) -> tuple[float, str]:
        """
        Normalize ``value`` against the fingerprint, then step.

        Equivalent to::

            metric = fp.sigma(component, value)        # or robust_sigma
            return ctrl.step(alpha, metric, ...)

        Does not trigger the fingerprint normalization warning.

        Parameters
        ----------
        alpha:       current alpha value
        component:   component name in the fingerprint
        value:       raw domain measurement (e.g. ``cq.recovery_ratio``)
        fingerprint: Fingerprint from ``Monitor.fingerprint()``
        robust:      if True, use ``fp.robust_sigma()`` (median/IQR) instead of
                     ``fp.sigma()`` (mean); recommended for high-std components
        alpha_min:   per-call lower clamp override
        alpha_max:   per-call upper clamp override
        """
        a_min = alpha_min if alpha_min is not None else self.alpha_min
        a_max = alpha_max if alpha_max is not None else self.alpha_max
        if robust:
            metric = fingerprint.robust_sigma(component, value)
        else:
            metric = fingerprint.sigma(component, value)
        return self._apply(alpha, metric, a_min, a_max)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _apply(
        self, alpha: float, metric: float, a_min: float, a_max: float
    ) -> tuple[float, str]:
        """Apply the control law to a pre-computed metric."""
        if self.strategy == "proportional_setpoint":
            error = self.target - metric
            alpha_next = alpha + self.kp * error
            reason = (
                f"SP(err={error:+.3f}, tgt={self.target:.3f}): "
                f"{alpha:.3f}→{max(a_min, min(a_max, alpha_next)):.3f}"
            )
        else:  # proportional_asymmetric
            if metric >= 0:
                alpha_next = alpha + self.kp * metric
                reason = f"P({metric:+.3f}): {alpha:.3f}→{min(alpha_next, a_max):.3f}"
            else:
                alpha_next = alpha * self.backoff
                reason = f"backoff({metric:.3f}): {alpha:.3f}→{max(alpha_next, a_min):.3f}"

        alpha_next = max(a_min, min(a_max, alpha_next))
        return alpha_next, reason

    def _metric(self, observations) -> float:
        """Extract a scalar metric from observations or pass through a scalar."""
        if isinstance(observations, (int, float)):
            return float(observations)
        sigmas = []
        for obs in observations:
            s = getattr(obs, "sigma", None)
            if s is not None:
                sigmas.append(s)
        if not sigmas:
            return 0.0
        return sum(sigmas) / len(sigmas)

    def __repr__(self) -> str:
        warm = ", warm=True" if self._from_fingerprint else ""
        return (
            f"Controller(strategy={self.strategy!r}, kp={self.kp}, "
            f"target={self.target:.3f}, backoff={self.backoff}, "
            f"alpha=[{self.alpha_min}, {self.alpha_max}]{warm})"
        )
