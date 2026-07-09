"""Pattern detection: identify systemic patterns in failure data.

Runs 9 analyses:
  1-6: Standard SPEC analyses (failure clustering, source, difficulty, time, correlation, drift)
  7: Reflexion effectiveness — do probes with session_context fail less?
  8: Self-RAG effectiveness — do probes that passed Self-RAG score better?
  9: Loop engineering effectiveness — correlation of loop_retries with final quality

All analyses operate on probe + measurement data from the SQLite DB. No LLM calls.
"""

import json
from datetime import datetime, timezone
from typing import Any

from database.db_client import get_dimension_averages, get_failure_distribution, get_recent_probes
from utils.logger import get_logger

logger = get_logger(__name__)


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _get_measurements_with_details(days: int = 30) -> list[dict]:
    """Return recent measurements with parsed measurement_details JSON."""
    from database.db_client import get_connection
    from config.settings import settings
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT m.*, p.query_id, p.category, p.difficulty, p.latency_total_ms
            FROM measurements m
            JOIN probe_results p ON m.probe_id = p.probe_id
            WHERE m.timestamp >= datetime('now', :offset)
            ORDER BY m.timestamp DESC
            """,
            {"offset": f"-{days} days"},
        ).fetchall()
    results = []
    for row in rows:
        d = dict(row)
        if d.get("measurement_details"):
            try:
                d["details"] = json.loads(d["measurement_details"])
            except json.JSONDecodeError:
                d["details"] = {}
        else:
            d["details"] = {}
        results.append(d)
    return results


# ── Analysis 1: Failure clustering by category ─────────────────────────────────

def analysis_1_failure_clustering(days: int = 7) -> dict:
    """Which failure categories dominate? Are any categories recurring?"""
    dist = get_failure_distribution(days=days)
    total = sum(dist.values())
    if total == 0:
        return {"status": "no_data", "distribution": {}}
    rates = {cat: round(cnt / total * 100, 1) for cat, cnt in dist.items()}
    dominant = max(dist, key=dist.get) if dist else "PASS"
    return {
        "total_probes": total,
        "distribution": dist,
        "rates_pct": rates,
        "dominant_failure": dominant,
        "pass_rate_pct": rates.get("PASS", 0.0),
    }


# ── Analysis 2: Source-based breakdown ─────────────────────────────────────────

def analysis_2_source_breakdown(days: int = 7) -> dict:
    """Do failures cluster around specific document sources?"""
    rows = _get_measurements_with_details(days=days)
    by_source: dict[str, list[str]] = {}
    for row in rows:
        src = row["details"].get("top_chunk_source", "unknown")
        # Map filename to source category
        if "wikipedia" in src.lower() or any(w in src for w in ["_learning_", "Model", "model", "_AI", "BERT", "GPT"]):
            source_cat = "wikipedia"
        elif "__" in src:
            source_cat = "huggingface"
        elif "benchmark" in src or "task_" in src:
            source_cat = "paperswithcode"
        else:
            source_cat = "unknown"
        by_source.setdefault(source_cat, []).append(row["failure_category"])

    source_stats = {}
    for src, cats in by_source.items():
        failures = [c for c in cats if c != "PASS"]
        source_stats[src] = {
            "total": len(cats),
            "failures": len(failures),
            "failure_rate_pct": round(len(failures) / len(cats) * 100, 1) if cats else 0.0,
            "most_common_failure": max(set(failures), key=failures.count) if failures else "PASS",
        }
    return {"by_source": source_stats, "days": days}


# ── Analysis 3: Difficulty-based breakdown ─────────────────────────────────────

def analysis_3_difficulty_breakdown(days: int = 7) -> dict:
    """Do harder queries fail more often?"""
    rows = _get_measurements_with_details(days=days)
    by_diff: dict[str, list[str]] = {}
    for row in rows:
        diff = row.get("difficulty", "medium")
        by_diff.setdefault(diff, []).append(row["failure_category"])

    stats = {}
    for diff, cats in by_diff.items():
        failures = [c for c in cats if c != "PASS"]
        stats[diff] = {
            "total": len(cats),
            "failure_rate_pct": round(len(failures) / len(cats) * 100, 1) if cats else 0.0,
        }
    return {"by_difficulty": stats, "days": days}


# ── Analysis 4: Time-based drift detection ─────────────────────────────────────

def analysis_4_time_drift(days: int = 7) -> dict:
    """Is system quality improving or degrading over the analysis window?"""
    from analysis.trend_analysis import analyze_trends
    return analyze_trends()


# ── Analysis 5: Dimension correlation matrix ──────────────────────────────────

def analysis_5_dimension_correlation(days: int = 7) -> dict:
    """Are low-retrieval probes also low-faithfulness? Find correlated dimensions."""
    rows = _get_measurements_with_details(days=days)
    if not rows:
        return {"status": "no_data"}

    retrieval = [r.get("retrieval_relevance_score", 0) or 0 for r in rows]
    utilization = [r.get("context_utilization_score", 0) or 0 for r in rows]
    faithfulness = [r.get("faithfulness_score", 0) or 0 for r in rows]
    factuality = [r.get("factuality_score", 0) or 0 for r in rows]

    def corr(a: list, b: list) -> float:
        n = len(a)
        if n < 2:
            return 0.0
        ma, mb = _safe_mean(a), _safe_mean(b)
        num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        da = sum((x - ma) ** 2 for x in a) ** 0.5
        db = sum((y - mb) ** 2 for y in b) ** 0.5
        return round(num / (da * db), 3) if da * db > 0 else 0.0

    return {
        "retrieval_utilization_corr": corr(retrieval, utilization),
        "retrieval_faithfulness_corr": corr(retrieval, faithfulness),
        "faithfulness_factuality_corr": corr(faithfulness, factuality),
        "n_samples": len(rows),
    }


# ── Analysis 6: Latency spike detection ───────────────────────────────────────

def analysis_6_latency_spikes(days: int = 7) -> dict:
    """Identify probes with unusually high latency."""
    from database.db_client import get_connection
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT latency_total_ms, query_id, category
            FROM probe_results
            WHERE timestamp >= datetime('now', :offset)
              AND latency_total_ms IS NOT NULL
            ORDER BY latency_total_ms DESC
            """,
            {"offset": f"-{days} days"},
        ).fetchall()
    if not rows:
        return {"status": "no_data"}
    latencies = [r["latency_total_ms"] for r in rows]
    avg = _safe_mean(latencies)
    p95 = sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) >= 10 else max(latencies)
    spikes = [dict(r) for r in rows if r["latency_total_ms"] > avg * 3]
    return {
        "avg_latency_ms": round(avg),
        "p95_latency_ms": round(p95),
        "spike_threshold_ms": round(avg * 3),
        "spike_count": len(spikes),
        "total_probes": len(rows),
    }


# ── Analysis 7: Reflexion effectiveness ───────────────────────────────────────

def analysis_7_reflexion_effectiveness(days: int = 30) -> dict:
    """Do probes that used Reflexion lessons fail less than those without?

    Compares failure_rate for probes where reflexion_lessons_applied=True vs False.
    """
    rows = _get_measurements_with_details(days=days)
    if not rows:
        return {"status": "no_data"}

    with_reflexion: list[str] = []
    without_reflexion: list[str] = []

    for row in rows:
        applied = row["details"].get("reflexion_lessons_applied", False)
        cat = row["failure_category"]
        if applied:
            with_reflexion.append(cat)
        else:
            without_reflexion.append(cat)

    def failure_rate(cats: list[str]) -> float:
        if not cats:
            return 0.0
        fails = sum(1 for c in cats if c != "PASS")
        return round(fails / len(cats) * 100, 1)

    with_rate = failure_rate(with_reflexion)
    without_rate = failure_rate(without_reflexion)
    improvement = round(without_rate - with_rate, 1)

    return {
        "with_reflexion_count": len(with_reflexion),
        "without_reflexion_count": len(without_reflexion),
        "with_reflexion_failure_rate_pct": with_rate,
        "without_reflexion_failure_rate_pct": without_rate,
        "reflexion_improvement_pct": improvement,
        "effective": improvement > 0,
    }


# ── Analysis 8: Self-RAG effectiveness ────────────────────────────────────────

def analysis_8_self_rag_effectiveness(days: int = 30) -> dict:
    """Do probes where Self-RAG passed score better on faithfulness?

    Compares avg_faithfulness for self_rag_passed=True vs False.
    """
    rows = _get_measurements_with_details(days=days)
    if not rows:
        return {"status": "no_data"}

    passed_faithfulness: list[float] = []
    failed_faithfulness: list[float] = []
    passed_count = 0
    failed_count = 0

    for row in rows:
        sr_passed = row["details"].get("self_rag_checks", {}).get("retrieval_adequate") and \
                    row["details"].get("self_rag_checks", {}).get("answer_grounded") and \
                    row["details"].get("self_rag_checks", {}).get("answer_complete")
        faith = row.get("faithfulness_score", 0) or 0
        if sr_passed:
            passed_faithfulness.append(faith)
            passed_count += 1
        else:
            failed_faithfulness.append(faith)
            failed_count += 1

    avg_passed = round(_safe_mean(passed_faithfulness), 3)
    avg_failed = round(_safe_mean(failed_faithfulness), 3)
    improvement = round(avg_passed - avg_failed, 3)

    return {
        "self_rag_passed_count": passed_count,
        "self_rag_failed_count": failed_count,
        "avg_faithfulness_when_passed": avg_passed,
        "avg_faithfulness_when_failed": avg_failed,
        "faithfulness_improvement": improvement,
        "effective": improvement > 0,
    }


# ── Analysis 9: Loop engineering effectiveness ─────────────────────────────────

def analysis_9_loop_effectiveness(days: int = 30) -> dict:
    """Does more retrying (loop_retries > 0) improve final answer quality?

    Compares factuality and faithfulness between probes with/without pipeline retries.
    """
    rows = _get_measurements_with_details(days=days)
    if not rows:
        return {"status": "no_data"}

    no_retry_fact: list[float] = []
    with_retry_fact: list[float] = []
    no_retry_faith: list[float] = []
    with_retry_faith: list[float] = []

    for row in rows:
        loop_retries = row["details"].get("loop_retries", 0)
        fact = row.get("factuality_score", 0) or 0
        faith = row.get("faithfulness_score", 0) or 0
        if loop_retries == 0:
            no_retry_fact.append(fact)
            no_retry_faith.append(faith)
        else:
            with_retry_fact.append(fact)
            with_retry_faith.append(faith)

    return {
        "no_retry_count": len(no_retry_fact),
        "with_retry_count": len(with_retry_fact),
        "avg_factuality_no_retry": round(_safe_mean(no_retry_fact), 3),
        "avg_factuality_with_retry": round(_safe_mean(with_retry_fact), 3),
        "avg_faithfulness_no_retry": round(_safe_mean(no_retry_faith), 3),
        "avg_faithfulness_with_retry": round(_safe_mean(with_retry_faith), 3),
        "factuality_improvement": round(
            _safe_mean(with_retry_fact) - _safe_mean(no_retry_fact), 3
        ),
    }


# ── Main orchestrator ──────────────────────────────────────────────────────────

def run_all_analyses(days: int = 7) -> dict:
    """Run all 9 analyses and return combined results."""
    computed_at = datetime.now(timezone.utc).isoformat()

    analyses = {
        "1_failure_clustering": analysis_1_failure_clustering(days),
        "2_source_breakdown": analysis_2_source_breakdown(days),
        "3_difficulty_breakdown": analysis_3_difficulty_breakdown(days),
        "4_time_drift": analysis_4_time_drift(days),
        "5_dimension_correlation": analysis_5_dimension_correlation(days),
        "6_latency_spikes": analysis_6_latency_spikes(days),
        "7_reflexion_effectiveness": analysis_7_reflexion_effectiveness(days=30),
        "8_self_rag_effectiveness": analysis_8_self_rag_effectiveness(days=30),
        "9_loop_effectiveness": analysis_9_loop_effectiveness(days=30),
    }

    logger.info(
        "Pattern analyses complete",
        extra={"days": days, "analyses": list(analyses.keys())},
    )
    return {
        "computed_at": computed_at,
        "analysis_window_days": days,
        "analyses": analyses,
    }
