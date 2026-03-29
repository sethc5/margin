"""
Pandas DataFrame quality hook for margin.

Pass a DataFrame, get typed health back. No configuration required —
computes completeness, null rate, duplicates, and row count automatically.

    from adapters.dataframe.pandas_hook import dataframe_health
    expr = dataframe_health(df, expected_rows=10000)
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from margin.observation import Expression
from margin.confidence import Confidence


def dataframe_health(
    df,
    expected_rows: Optional[int] = None,
    expected_columns: Optional[list[str]] = None,
    reference_df=None,
    label: str = "",
    confidence: Confidence = Confidence.MODERATE,
) -> Expression:
    """
    Compute data quality metrics from a pandas DataFrame and return
    a typed margin Expression.

    Args:
        df:               the DataFrame to evaluate
        expected_rows:    expected row count (for row_count_ratio)
        expected_columns: expected column names (for schema_match)
        reference_df:     reference DataFrame for drift detection
        label:            expression label (pipeline name, table name)
        confidence:       measurement confidence
    """
    from .quality import parse_quality, pipeline_expression

    metrics: dict[str, float] = {}

    n_rows, n_cols = df.shape

    # Completeness / null rate
    total_cells = n_rows * n_cols
    if total_cells > 0:
        null_count = int(df.isnull().sum().sum())
        metrics["completeness"] = 1.0 - (null_count / total_cells)
        metrics["null_rate"] = null_count / total_cells

    # Duplicate rate
    if n_rows > 0:
        n_dupes = int(df.duplicated().sum())
        metrics["duplicate_rate"] = n_dupes / n_rows

    # Row count ratio
    if expected_rows is not None and expected_rows > 0:
        metrics["row_count_ratio"] = n_rows / expected_rows

    # Schema match
    if expected_columns is not None and len(expected_columns) > 0:
        actual = set(df.columns)
        expected = set(expected_columns)
        if expected:
            metrics["schema_match"] = len(actual & expected) / len(expected)

    # Value drift (simple: compare column means if reference provided)
    if reference_df is not None:
        try:
            numeric_cols = df.select_dtypes(include="number").columns
            ref_numeric = reference_df.select_dtypes(include="number").columns
            shared = list(set(numeric_cols) & set(ref_numeric))
            if shared:
                drifts = []
                for col in shared:
                    ref_mean = float(reference_df[col].mean())
                    cur_mean = float(df[col].mean())
                    if abs(ref_mean) > 0.001:
                        drifts.append(abs(cur_mean - ref_mean) / abs(ref_mean))
                if drifts:
                    metrics["value_drift"] = sum(drifts) / len(drifts)
        except Exception:
            pass  # drift is best-effort

    # Outlier rate (simple: beyond 3 std from mean)
    try:
        numeric = df.select_dtypes(include="number")
        if not numeric.empty and n_rows > 0:
            means = numeric.mean()
            stds = numeric.std()
            outlier_mask = ((numeric - means).abs() > 3 * stds).any(axis=1)
            metrics["outlier_rate"] = float(outlier_mask.sum()) / n_rows
    except Exception:
        pass

    return pipeline_expression(metrics, pipeline_id=label, confidence=confidence,
                               measured_at=datetime.now())
