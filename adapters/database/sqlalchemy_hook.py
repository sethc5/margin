"""
SQLAlchemy database health hook for margin.

Pass an engine, get typed health back.

    from adapters.database.sqlalchemy_hook import db_health
    expr = db_health(engine)
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from margin.observation import Expression
from margin.confidence import Confidence


def db_health(
    engine,
    label: str = "",
    confidence: Confidence = Confidence.MODERATE,
) -> Expression:
    """
    Compute database health metrics from a SQLAlchemy engine's pool
    and return a typed margin Expression.

    Reads pool statistics (no queries executed). For query-level metrics,
    use parse_db() with your own instrumentation data.
    """
    from .health import parse_db

    pool = engine.pool
    metrics: dict[str, float] = {}

    try:
        pool_size = pool.size()
        checked_out = pool.checkedout()
        overflow = pool.overflow()
        checked_in = pool.checkedin()

        total_capacity = pool_size + (pool._max_overflow if hasattr(pool, '_max_overflow') else 0)

        if total_capacity > 0:
            metrics["pool_usage"] = checked_out / total_capacity

        metrics["pool_available"] = float(checked_in)

    except (AttributeError, TypeError):
        # Not a QueuePool or pool doesn't expose these
        pass

    if not metrics:
        return Expression(label=label, confidence=Confidence.INDETERMINATE)

    obs = parse_db(metrics, confidence=confidence, measured_at=datetime.now())
    return Expression(
        observations=list(obs.values()),
        confidence=min((o.confidence for o in obs.values()), default=Confidence.INDETERMINATE),
        label=label,
    )
