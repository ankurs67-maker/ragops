"""Daily report generator: produces structured JSON + plain text reports.

Aggregates all analyses and metrics into a daily summary stored in the DB.
Plain-text report is suitable for emailing or printing. No LLM calls.
"""

import json
import uuid
from datetime import datetime, timezone

from analysis.pattern_detector import run_all_analyses
from analysis.remediation_proposer import propose_remediations
from analysis.trend_analysis import analyze_trends, get_recent_failure_trend
from database.db_client import (
    get_dimension_averages,
    get_failure_distribution,
    get_system_health_score,
    insert_daily_report,
)
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


def _format_score_bar(score: float, max_val: float = 1.0, width: int = 20) -> str:
    """ASCII bar chart for a score."""
    filled = int((score / max_val) * width)
    return f"[{'#' * filled}{'-' * (width - filled)}] {score:.3f}"


def generate_daily_report() -> dict:
    """Generate a complete daily report and persist it to the database.

    Returns:
        report_json: dict with all metrics and analysis results
        report_text: human-readable plain text string
    """
    now = datetime.now(timezone.utc)
    report_date = now.date().isoformat()
    timestamp = now.isoformat()

    health_score = get_system_health_score()
    avgs = get_dimension_averages(days=1)
    failure_trend = get_recent_failure_trend(days=7)
    trend = analyze_trends()
    remediations = propose_remediations(days=1)
    analyses = run_all_analyses(days=7)

    # Build report JSON
    report_json = {
        "date": report_date,
        "generated_at": timestamp,
        "system_health_score": health_score,
        "dimension_averages_24h": {
            "retrieval_relevance": round((avgs.get("avg_retrieval") or 0.0), 3),
            "context_utilization": round((avgs.get("avg_utilization") or 0.0), 1),
            "faithfulness": round((avgs.get("avg_faithfulness") or 0.0), 3),
            "factuality": round((avgs.get("avg_factuality") or 0.0), 3),
            "refusal_calibration": round((avgs.get("avg_refusal") or 0.0), 3),
        },
        "failure_trend_7d": failure_trend,
        "trend_analysis": trend,
        "new_remediations": len(remediations),
        "analyses": analyses,
    }

    # Build plain-text report
    lines = [
        "=" * 70,
        f"RAGOps DAILY REPORT - {report_date}",
        "=" * 70,
        "",
        f"System Health Score: {health_score:.1f}/100",
        "",
        "-- DIMENSION AVERAGES (last 24h) " + "-" * 37,
    ]

    ret_avg = avgs.get("avg_retrieval") or 0.0
    util_avg = avgs.get("avg_utilization") or 0.0
    faith_avg = avgs.get("avg_faithfulness") or 0.0
    fact_avg = avgs.get("avg_factuality") or 0.0
    ref_avg = avgs.get("avg_refusal") or 0.0

    lines += [
        f"  Retrieval Relevance  {_format_score_bar(ret_avg, 3.0)}",
        f"  Context Utilization  {_format_score_bar(util_avg, 100.0)} %",
        f"  Faithfulness         {_format_score_bar(faith_avg, 1.0)}",
        f"  Factuality           {_format_score_bar(fact_avg, 1.0)}",
        f"  Refusal Calibration  {_format_score_bar(ref_avg, 1.0)}",
        "",
        "-- FAILURE TREND (7 days) " + "-" * 44,
        f"  Total probes:   {failure_trend['total_probes']}",
        f"  Pass count:     {failure_trend['pass_count']}",
        f"  Failure count:  {failure_trend['fail_count']}",
        f"  Failure rate:   {failure_trend['failure_rate_pct']}%",
        "",
    ]

    if failure_trend["distribution"]:
        lines.append("  Breakdown:")
        for cat, cnt in failure_trend["distribution"].items():
            pct = round(cnt / max(failure_trend["total_probes"], 1) * 100, 1)
            lines.append(f"    {cat:<30} {cnt:>4} probes  ({pct}%)")
        lines.append("")

    # Trend alerts
    trend_alerts = trend.get("alerts", [])
    lines.append("-- TREND ALERTS " + "-" * 54)
    if trend_alerts:
        for dim in trend_alerts:
            d = trend["dimensions"].get(dim, {})
            lines.append(
                f"  [!] {dim}: {d.get('pct_change', 0):+.1f}% "
                f"({d.get('baseline_avg', 0):.3f} -> {d.get('recent_avg', 0):.3f})"
            )
    else:
        lines.append("  No trend alerts — system stable.")
    lines.append("")

    # Advanced technique effectiveness
    a7 = analyses["analyses"].get("7_reflexion_effectiveness", {})
    a8 = analyses["analyses"].get("8_self_rag_effectiveness", {})
    a9 = analyses["analyses"].get("9_loop_effectiveness", {})

    lines.append("-- ADVANCED TECHNIQUE EFFECTIVENESS " + "-" * 34)
    if a7.get("status") != "no_data" and a7.get("with_reflexion_count", 0) > 0:
        imp = a7.get("reflexion_improvement_pct", 0)
        lines.append(f"  Reflexion:  {'[OK]' if a7.get('effective') else '[NO]'} {imp:+.1f}% failure rate reduction")
    else:
        lines.append("  Reflexion:  Not enough data yet (run more probe cycles)")

    if a8.get("status") != "no_data" and a8.get("self_rag_passed_count", 0) > 0:
        imp8 = a8.get("faithfulness_improvement", 0)
        lines.append(f"  Self-RAG:   {'[OK]' if a8.get('effective') else '[NO]'} {imp8:+.3f} faithfulness improvement")
    else:
        lines.append("  Self-RAG:   Not enough data yet")

    if a9.get("with_retry_count", 0) > 0:
        imp9 = a9.get("factuality_improvement", 0)
        lines.append(f"  Loop Eng.:  {imp9:+.3f} factuality improvement when retried")
    else:
        lines.append("  Loop Eng.:  No retries recorded yet")
    lines.append("")

    # Remediations
    lines.append("-- NEW REMEDIATIONS " + "-" * 50)
    if remediations:
        for r in remediations:
            lines.append(f"  [{r['priority'].upper()}] {r['alert_type']}: {r['remediation_text'][:80]}...")
    else:
        lines.append("  No new remediations needed.")
    lines.append("")
    lines.append("=" * 70)
    lines.append(f"Generated by RAGOps at {timestamp}")
    lines.append("=" * 70)

    report_text = "\n".join(lines)

    # Persist to DB
    insert_daily_report({
        "report_id": str(uuid.uuid4()),
        "date": report_date,
        "report_text": report_text,
        "report_json": report_json,
        "system_health_score": health_score,
    })

    # Also save to reports/ directory
    reports_dir = settings.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = reports_dir / f"daily_report_{report_date}.txt"
    report_file.write_text(report_text, encoding="utf-8")
    report_json_file = reports_dir / f"daily_report_{report_date}.json"
    report_json_file.write_text(json.dumps(report_json, indent=2), encoding="utf-8")

    logger.info(
        "Daily report generated",
        extra={"date": report_date, "health_score": health_score, "report_file": str(report_file)},
    )
    return {"report_json": report_json, "report_text": report_text, "report_file": str(report_file)}


if __name__ == "__main__":
    result = generate_daily_report()
    print(result["report_text"].encode("utf-8", errors="replace").decode("utf-8"))
