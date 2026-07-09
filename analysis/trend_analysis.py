"""Trend analysis: detect deteriorating or improving metric trends over time.

Compares the most recent trend_window_days average against the
baseline_window_days average to detect statistically meaningful shifts.
No LLM calls — pure statistics on DB data.
"""

from datetime import datetime, timezone
from typing import Optional

from database.db_client import (
    get_connection,
    get_dimension_averages,
    get_failure_distribution,
)
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# Keys returned by get_dimension_averages → display name mapping
_DIMENSIONS = {
    "avg_retrieval": "retrieval_relevance",
    "avg_utilization": "context_utilization",
    "avg_faithfulness": "faithfulness",
    "avg_factuality": "factuality",
    "avg_refusal": "refusal_calibration",
}

# Dimension → measurements table column (for per-day series / t-tests)
_DIMENSION_COLUMNS = {
    "retrieval_relevance": "retrieval_relevance_score",
    "context_utilization": "context_utilization_score",
    "faithfulness": "faithfulness_score",
    "factuality": "factuality_score",
    "refusal_calibration": "refusal_calibration_score",
}

# Higher = better for all dimensions (retrieval is on 0-3 scale, others 0-1 or 0-100)
_SCALE = {
    "retrieval_relevance": 3.0,
    "context_utilization": 100.0,
    "faithfulness": 1.0,
    "factuality": 1.0,
    "refusal_calibration": 1.0,
}


def _pct_change(baseline: float, recent: float, scale: float) -> float:
    """Compute percent change relative to scale (not relative to baseline)."""
    if scale <= 0:
        return 0.0
    return ((recent - baseline) / scale) * 100.0


# Minimum samples per series for a t-test verdict to be reported at all.
# Below this, p-values are dominated by noise and would be misleading.
_MIN_TTEST_SAMPLES = 5


def run_statistical_test(series_a: list[float], series_b: list[float]) -> dict:
    """Independent two-sample t-test (Welch) between two score series.

    Returns dict with:
        status: "ok" | "insufficient_data"
        n_a, n_b: sample sizes
        t_statistic: float | None
        p_value: float | None
        is_significant: bool — only True when status == "ok" and p < 0.05

    With fewer than _MIN_TTEST_SAMPLES points in either series the test
    reports insufficient_data instead of a potentially spurious verdict.
    """
    n_a, n_b = len(series_a), len(series_b)
    if n_a < _MIN_TTEST_SAMPLES or n_b < _MIN_TTEST_SAMPLES:
        logger.info(
            "t-test skipped — insufficient samples",
            extra={"n_a": n_a, "n_b": n_b, "required": _MIN_TTEST_SAMPLES},
        )
        return {
            "status": "insufficient_data",
            "n_a": n_a,
            "n_b": n_b,
            "t_statistic": None,
            "p_value": None,
            "is_significant": False,
        }

    from scipy import stats

    # Welch's t-test: does not assume equal variances between windows.
    t_stat, p_value = stats.ttest_ind(series_a, series_b, equal_var=False)
    return {
        "status": "ok",
        "n_a": n_a,
        "n_b": n_b,
        "t_statistic": round(float(t_stat), 4),
        "p_value": round(float(p_value), 6),
        "is_significant": bool(p_value < 0.05),
    }


def _daily_series(
    column: str, days: int, exclude_recent_days: int = 0
) -> list[float]:
    """Daily averages of one measurement column over the last N days.

    exclude_recent_days trims the most recent days off the window so a
    baseline series does not overlap the recent series it is compared to.
    """
    try:
        with get_connection() as conn:
            rows = conn.execute(
                f"""
                SELECT date(timestamp) AS day, AVG({column}) AS avg_val
                FROM measurements
                WHERE timestamp >= datetime('now', :start)
                  AND timestamp < datetime('now', :end)
                GROUP BY date(timestamp)
                ORDER BY day
                """,
                {
                    "start": f"-{days} days",
                    "end": f"-{exclude_recent_days} days" if exclude_recent_days else "+0 days",
                },
            ).fetchall()
        return [r["avg_val"] for r in rows if r["avg_val"] is not None]
    except Exception as exc:
        logger.warning("daily series query failed", extra={"error": str(exc)})
        return []


def analyze_trends() -> dict:
    """Compare recent window vs baseline window for all 5 dimensions.

    Returns dict with:
        trend_window_days: int
        baseline_window_days: int
        computed_at: str
        dimensions: dict[str, dict] — per-dimension trend data
        alerts: list[str] — dimensions with > threshold% degradation
        overall_direction: "improving" | "degrading" | "stable"
    """
    trend_days = settings.trend_window_days
    baseline_days = settings.baseline_window_days
    alert_pct = settings.trend_alert_percent

    recent = get_dimension_averages(days=trend_days)
    baseline = get_dimension_averages(days=baseline_days)

    dimensions: dict[str, dict] = {}
    alerts: list[str] = []
    changes: list[float] = []

    for db_key, dim in _DIMENSIONS.items():
        scale = _SCALE.get(dim, 1.0)
        recent_val = recent.get(db_key, 0.0) or 0.0
        baseline_val = baseline.get(db_key, 0.0) or 0.0
        pct = _pct_change(baseline_val, recent_val, scale)
        changes.append(pct)

        trend = "stable"
        if pct < -alert_pct:
            trend = "degrading"
            alerts.append(dim)
        elif pct > alert_pct:
            trend = "improving"

        # Statistical significance: daily averages, recent window vs the
        # part of the baseline window that precedes it. Reports
        # insufficient_data honestly when there are too few days.
        column = _DIMENSION_COLUMNS.get(dim)
        significance = {"status": "insufficient_data", "is_significant": False}
        if column:
            recent_series = _daily_series(column, days=trend_days)
            baseline_series = _daily_series(
                column, days=baseline_days, exclude_recent_days=trend_days
            )
            significance = run_statistical_test(recent_series, baseline_series)

        dimensions[dim] = {
            "recent_avg": round(recent_val, 4),
            "baseline_avg": round(baseline_val, 4),
            "pct_change": round(pct, 2),
            "trend": trend,
            "alert": pct < -alert_pct,
            "significance": significance,
        }

    # Overall direction: average of all dimension changes
    avg_change = sum(changes) / len(changes) if changes else 0.0
    if avg_change < -alert_pct / 2:
        overall_direction = "degrading"
    elif avg_change > alert_pct / 2:
        overall_direction = "improving"
    else:
        overall_direction = "stable"

    result = {
        "trend_window_days": trend_days,
        "baseline_window_days": baseline_days,
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "dimensions": dimensions,
        "alerts": alerts,
        "overall_direction": overall_direction,
        "avg_pct_change": round(avg_change, 2),
    }

    logger.info(
        "Trend analysis complete",
        extra={"alerts": alerts, "overall": overall_direction},
    )
    return result


def get_recent_failure_trend(days: int = 7) -> dict:
    """Return failure category counts for the recent window."""
    dist = get_failure_distribution(days=days)
    total = sum(dist.values())
    pass_count = dist.get("PASS", 0)
    failure_rate = ((total - pass_count) / total * 100) if total > 0 else 0.0
    return {
        "total_probes": total,
        "pass_count": pass_count,
        "fail_count": total - pass_count,
        "failure_rate_pct": round(failure_rate, 1),
        "distribution": dist,
    }
