"""
Celery task queue health hook for margin.

Pass a Celery app, get typed health back.

    from adapters.celery.hook import celery_health
    expr = celery_health(app)
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from margin.observation import Expression
from margin.confidence import Confidence


def celery_health(
    app,
    label: str = "",
    confidence: Confidence = Confidence.MODERATE,
    timeout: float = 5.0,
) -> Expression:
    """
    Compute task queue health from a Celery app's inspect API.

    Queries active/reserved/scheduled tasks and worker stats.
    Times out gracefully if workers are unreachable.
    """
    from .tasks import parse_queue

    metrics: dict[str, float] = {}

    try:
        inspect = app.control.inspect(timeout=timeout)

        # Active tasks across all workers
        active = inspect.active() or {}
        total_active = sum(len(tasks) for tasks in active.values())

        # Reserved (prefetched) tasks
        reserved = inspect.reserved() or {}
        total_reserved = sum(len(tasks) for tasks in reserved.values())

        # Scheduled (eta/countdown) tasks
        scheduled = inspect.scheduled() or {}
        total_scheduled = sum(len(tasks) for tasks in scheduled.values())

        metrics["queue_depth"] = float(total_reserved + total_scheduled)

        # Worker count
        n_workers = len(active)
        if n_workers > 0:
            metrics["worker_utilization"] = total_active / max(n_workers, 1)

        # Worker stats if available
        stats = inspect.stats() or {}
        total_succeeded = 0
        total_failed = 0
        total_retried = 0
        for worker_stats in stats.values():
            totals = worker_stats.get("total", {})
            for task_name, count in totals.items():
                total_succeeded += count
            total_failed += worker_stats.get("total_failed", 0)
            total_retried += worker_stats.get("total_retried", 0)

        total_tasks = total_succeeded + total_failed
        if total_tasks > 0:
            metrics["failure_rate"] = total_failed / total_tasks
        if total_succeeded > 0:
            metrics["retry_rate"] = total_retried / total_succeeded

    except Exception:
        return Expression(label=label, confidence=Confidence.INDETERMINATE)

    if not metrics:
        return Expression(label=label, confidence=Confidence.INDETERMINATE)

    obs = parse_queue(metrics, confidence=confidence, measured_at=datetime.now())
    return Expression(
        observations=list(obs.values()),
        confidence=min((o.confidence for o in obs.values()), default=Confidence.INDETERMINATE),
        label=label,
    )
