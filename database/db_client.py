"""SQLite database client for RAGOps.

Provides all database operations: initialisation, inserts, queries.
All queries use parameterised named placeholders.
WAL mode enabled for concurrent read performance.
"""

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


@contextmanager
def get_connection() -> Generator[sqlite3.Connection, None, None]:
    """Open a WAL-mode SQLite connection with row_factory and foreign keys."""
    conn = sqlite3.connect(str(settings.db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_database() -> None:
    """Run schema.sql and insert schema_version row if not present."""
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    schema_sql = settings.schema_path.read_text(encoding="utf-8")
    with get_connection() as conn:
        conn.executescript(schema_sql)
        row = conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()
        if row[0] == 0:
            conn.execute(
                "INSERT INTO schema_version (version) VALUES (:version)",
                {"version": 1},
            )
    logger.info(
        "Database initialised",
        extra={"db_path": str(settings.db_path)},
    )


def insert_probe_result(probe: dict) -> None:
    """Insert a single probe result row."""
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO probe_results (
                probe_id, run_id, timestamp, query_id, query_text,
                category, difficulty, retrieved_chunks, generated_answer,
                correct_answer, answer_correct, refused_when_should,
                latency_retrieval_ms, latency_generation_ms, latency_total_ms
            ) VALUES (
                :probe_id, :run_id, :timestamp, :query_id, :query_text,
                :category, :difficulty, :retrieved_chunks, :generated_answer,
                :correct_answer, :answer_correct, :refused_when_should,
                :latency_retrieval_ms, :latency_generation_ms, :latency_total_ms
            )
            """,
            {
                "probe_id": probe["probe_id"],
                "run_id": probe["run_id"],
                "timestamp": probe["timestamp"],
                "query_id": probe["query_id"],
                "query_text": probe["query_text"],
                "category": probe["category"],
                "difficulty": probe.get("difficulty"),
                "retrieved_chunks": (
                    json.dumps(probe["retrieved_chunks"])
                    if isinstance(probe.get("retrieved_chunks"), (list, dict))
                    else probe.get("retrieved_chunks")
                ),
                "generated_answer": probe.get("generated_answer"),
                "correct_answer": probe.get("correct_answer"),
                "answer_correct": probe.get("answer_correct"),
                "refused_when_should": probe.get("refused_when_should"),
                "latency_retrieval_ms": probe.get("latency_retrieval_ms"),
                "latency_generation_ms": probe.get("latency_generation_ms"),
                "latency_total_ms": probe.get("latency_total_ms"),
            },
        )
    logger.debug(
        "Probe result inserted",
        extra={"probe_id": probe["probe_id"]},
    )


def insert_measurement(m: dict) -> None:
    """Insert a single measurement row."""
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO measurements (
                measurement_id, probe_id, run_id, timestamp,
                retrieval_relevance_score, context_utilization_score,
                faithfulness_score, factuality_score, refusal_calibration_score,
                judge_model_version, judge_confidence, failure_category,
                measurement_details
            ) VALUES (
                :measurement_id, :probe_id, :run_id, :timestamp,
                :retrieval_relevance_score, :context_utilization_score,
                :faithfulness_score, :factuality_score, :refusal_calibration_score,
                :judge_model_version, :judge_confidence, :failure_category,
                :measurement_details
            )
            """,
            {
                "measurement_id": m["measurement_id"],
                "probe_id": m["probe_id"],
                "run_id": m["run_id"],
                "timestamp": m["timestamp"],
                "retrieval_relevance_score": m.get("retrieval_relevance_score"),
                "context_utilization_score": m.get("context_utilization_score"),
                "faithfulness_score": m.get("faithfulness_score"),
                "factuality_score": m.get("factuality_score"),
                "refusal_calibration_score": m.get("refusal_calibration_score"),
                "judge_model_version": m.get("judge_model_version"),
                "judge_confidence": m.get("judge_confidence"),
                "failure_category": m.get("failure_category"),
                "measurement_details": (
                    json.dumps(m["measurement_details"])
                    if isinstance(m.get("measurement_details"), (list, dict))
                    else m.get("measurement_details")
                ),
            },
        )
    logger.debug(
        "Measurement inserted",
        extra={"measurement_id": m["measurement_id"]},
    )


def insert_pattern_report(report: dict) -> None:
    """Insert a pattern analysis report."""
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO pattern_reports (
                report_id, date, timestamp, overall_health_score,
                alerts_triggered, dimension_scores, failure_distribution,
                category_breakdown, source_breakdown, top_finding, raw_analysis
            ) VALUES (
                :report_id, :date, :timestamp, :overall_health_score,
                :alerts_triggered, :dimension_scores, :failure_distribution,
                :category_breakdown, :source_breakdown, :top_finding, :raw_analysis
            )
            """,
            {
                "report_id": report["report_id"],
                "date": report["date"],
                "timestamp": report["timestamp"],
                "overall_health_score": report.get("overall_health_score"),
                "alerts_triggered": (
                    json.dumps(report["alerts_triggered"])
                    if isinstance(report.get("alerts_triggered"), (list, dict))
                    else report.get("alerts_triggered")
                ),
                "dimension_scores": (
                    json.dumps(report["dimension_scores"])
                    if isinstance(report.get("dimension_scores"), (list, dict))
                    else report.get("dimension_scores")
                ),
                "failure_distribution": (
                    json.dumps(report["failure_distribution"])
                    if isinstance(report.get("failure_distribution"), (list, dict))
                    else report.get("failure_distribution")
                ),
                "category_breakdown": (
                    json.dumps(report["category_breakdown"])
                    if isinstance(report.get("category_breakdown"), (list, dict))
                    else report.get("category_breakdown")
                ),
                "source_breakdown": (
                    json.dumps(report["source_breakdown"])
                    if isinstance(report.get("source_breakdown"), (list, dict))
                    else report.get("source_breakdown")
                ),
                "top_finding": report.get("top_finding"),
                "raw_analysis": report.get("raw_analysis"),
            },
        )
    logger.info(
        "Pattern report inserted",
        extra={"report_id": report["report_id"]},
    )


def insert_remediation(r: dict) -> None:
    """Insert a remediation proposal."""
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO remediations (
                remediation_id, triggered_by, timestamp, alert_type,
                root_cause, confidence, remediation_text, specific_steps,
                priority, status, outcome
            ) VALUES (
                :remediation_id, :triggered_by, :timestamp, :alert_type,
                :root_cause, :confidence, :remediation_text, :specific_steps,
                :priority, :status, :outcome
            )
            """,
            {
                "remediation_id": r["remediation_id"],
                "triggered_by": r.get("triggered_by"),
                "timestamp": r["timestamp"],
                "alert_type": r.get("alert_type"),
                "root_cause": r.get("root_cause"),
                "confidence": r.get("confidence"),
                "remediation_text": r.get("remediation_text"),
                "specific_steps": (
                    json.dumps(r["specific_steps"])
                    if isinstance(r.get("specific_steps"), (list, dict))
                    else r.get("specific_steps")
                ),
                "priority": r.get("priority"),
                "status": r.get("status", "pending"),
                "outcome": r.get("outcome"),
            },
        )
    logger.info(
        "Remediation inserted",
        extra={"remediation_id": r["remediation_id"]},
    )


def insert_daily_report(report: dict) -> None:
    """Insert a daily report entry."""
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO daily_reports (
                report_id, date, report_text, report_json, system_health_score
            ) VALUES (
                :report_id, :date, :report_text, :report_json, :system_health_score
            )
            """,
            {
                "report_id": report["report_id"],
                "date": report["date"],
                "report_text": report.get("report_text"),
                "report_json": (
                    json.dumps(report["report_json"])
                    if isinstance(report.get("report_json"), (list, dict))
                    else report.get("report_json")
                ),
                "system_health_score": report.get("system_health_score"),
            },
        )
    logger.info(
        "Daily report inserted",
        extra={"report_id": report["report_id"]},
    )


def get_recent_probes(hours: int = 24) -> list[dict]:
    """Return probe results from the last N hours."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM probe_results
            WHERE timestamp >= datetime('now', :offset)
            ORDER BY timestamp DESC
            """,
            {"offset": f"-{hours} hours"},
        ).fetchall()
    return [dict(row) for row in rows]


def get_dimension_averages(days: int = 1) -> dict:
    """Return average scores per dimension over the last N days."""
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT
                AVG(retrieval_relevance_score)   AS avg_retrieval,
                AVG(context_utilization_score)   AS avg_utilization,
                AVG(faithfulness_score)          AS avg_faithfulness,
                AVG(factuality_score)            AS avg_factuality,
                AVG(refusal_calibration_score)   AS avg_refusal
            FROM measurements
            WHERE timestamp >= datetime('now', :offset)
            """,
            {"offset": f"-{days} days"},
        ).fetchone()
    if row is None:
        return {
            "avg_retrieval": None,
            "avg_utilization": None,
            "avg_faithfulness": None,
            "avg_factuality": None,
            "avg_refusal": None,
        }
    return dict(row)


def get_failure_distribution(days: int = 7) -> dict:
    """Return count of each failure category over the last N days."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT failure_category, COUNT(*) AS cnt
            FROM measurements
            WHERE timestamp >= datetime('now', :offset)
                AND failure_category IS NOT NULL
            GROUP BY failure_category
            ORDER BY cnt DESC
            """,
            {"offset": f"-{days} days"},
        ).fetchall()
    return {row["failure_category"]: row["cnt"] for row in rows}


def get_pending_remediations() -> list[dict]:
    """Return all remediations with status='pending', newest first."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM remediations
            WHERE status = 'pending'
            ORDER BY timestamp DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def get_system_health_score() -> float:
    """Compute composite health score from 5 dimension averages."""
    avgs = get_dimension_averages(days=1)
    avg_retrieval = avgs.get("avg_retrieval")
    avg_utilization = avgs.get("avg_utilization")
    avg_faithfulness = avgs.get("avg_faithfulness")
    avg_factuality = avgs.get("avg_factuality")
    avg_refusal = avgs.get("avg_refusal")

    # Return 100 if no data yet
    if all(v is None for v in [avg_retrieval, avg_utilization, avg_faithfulness,
                                avg_factuality, avg_refusal]):
        return 100.0

    retrieval_norm = (avg_retrieval or 0.0) / 3.0
    utilization_norm = (avg_utilization or 0.0) / 100.0
    faithfulness_norm = avg_faithfulness or 0.0
    factuality_norm = avg_factuality or 0.0
    refusal_norm = avg_refusal or 0.0

    health = (
        (retrieval_norm + utilization_norm + faithfulness_norm
         + factuality_norm + refusal_norm) / 5.0
    ) * 100.0
    return round(health, 2)


def update_remediation_status(
    remediation_id: str, status: str, outcome: str
) -> None:
    """Update the status and outcome of a remediation row."""
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE remediations
            SET status = :status, outcome = :outcome
            WHERE remediation_id = :remediation_id
            """,
            {
                "remediation_id": remediation_id,
                "status": status,
                "outcome": outcome,
            },
        )
    logger.info(
        "Remediation status updated",
        extra={"remediation_id": remediation_id, "status": status},
    )


def get_probe_by_id(probe_id: str) -> dict:
    """Return a single probe result by probe_id."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM probe_results WHERE probe_id = :probe_id",
            {"probe_id": probe_id},
        ).fetchone()
    if row is None:
        return {}
    return dict(row)


if __name__ == "__main__":
    init_database()
    with get_connection() as conn:
        table_count = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
        ).fetchone()[0]
    print(f"Database initialised at: {settings.db_path}")
    print(f"Tables created: {table_count}")
