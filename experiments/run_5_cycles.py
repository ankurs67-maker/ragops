"""Run 5 consecutive probe cycles at full ground-truth scale and record
per-cycle summaries plus failed query_ids for Reflexion-effect analysis.

Output: reports/cycle_comparison_<date>.json
"""

import json
import sys
from datetime import datetime, timezone

from config.settings import settings
from database.db_client import get_connection, get_system_health_score
from monitoring.run_probe_cycle import run_probe_cycle
from utils.logger import get_logger

logger = get_logger(__name__)

N_CYCLES = 5


def _failures_since(start_iso: str) -> dict[str, str]:
    """query_id -> failure_category for all failed probes since start_iso."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT p.query_id, m.failure_category
            FROM measurements m
            JOIN probe_results p ON p.probe_id = m.probe_id
            WHERE m.timestamp >= :start AND m.failure_category != 'PASS'
            """,
            {"start": start_iso},
        ).fetchall()
    return {r["query_id"]: r["failure_category"] for r in rows}


def main() -> None:
    n_cycles = int(sys.argv[1]) if len(sys.argv) > 1 else N_CYCLES
    label = sys.argv[2] if len(sys.argv) > 2 else "cycles"
    all_cycles = []
    for i in range(1, n_cycles + 1):
        cycle_start = datetime.now(timezone.utc).isoformat()
        print(f"\n{'=' * 70}\nCYCLE {i}/{n_cycles} starting at {cycle_start}\n{'=' * 70}",
              flush=True)
        try:
            summary = run_probe_cycle()
        except Exception as exc:
            print(f"CYCLE {i} FAILED: {exc}", flush=True)
            break

        summary["cycle_number"] = i
        summary["health_score"] = get_system_health_score()
        summary["failed_query_ids"] = _failures_since(cycle_start)
        all_cycles.append(summary)

        print(f"\n--- CYCLE {i} SUMMARY ---", flush=True)
        for k, v in summary.items():
            if k != "failed_query_ids":
                print(f"  {k}: {v}", flush=True)
        print(f"  failed queries: {sorted(summary['failed_query_ids'])}", flush=True)

    out_path = settings.reports_dir / (
        f"cycle_comparison_{label}_{datetime.now(timezone.utc).date()}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_cycles, indent=2), encoding="utf-8")
    print(f"\nSaved {len(all_cycles)} cycle summaries to {out_path}", flush=True)


if __name__ == "__main__":
    main()
