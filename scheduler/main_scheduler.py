"""APScheduler-based probe scheduler for RAGOps.

Runs 3 scheduled jobs:
  - Probe cycles: at each hour in probe_schedule_hours (default [0, 12] UTC)
  - Pattern analysis: daily at pattern_schedule_hour (default 23:00 UTC)
  - Daily report: daily at report_schedule_hour (default 07:00 UTC)

Run with: python scheduler/main_scheduler.py
Or via:   make schedule
"""

import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from config.settings import settings
from database.db_client import init_database
from utils.logger import get_logger

logger = get_logger(__name__)


def job_probe_cycle() -> None:
    """Scheduled job: run a complete probe cycle."""
    logger.info("Scheduled probe cycle starting")
    try:
        from monitoring.run_probe_cycle import run_probe_cycle
        summary = run_probe_cycle()
        logger.info(
            "Scheduled probe cycle complete",
            extra={
                "total": summary["total_probes"],
                "pass": summary["pass_count"],
                "fail": summary["fail_count"],
            },
        )
    except Exception as exc:
        logger.error("Probe cycle failed", extra={"error": str(exc)})


def job_pattern_analysis() -> None:
    """Scheduled job: run all 9 pattern analyses and generate remediations."""
    logger.info("Scheduled pattern analysis starting")
    try:
        from analysis.pattern_detector import run_all_analyses
        from analysis.remediation_proposer import propose_remediations
        results = run_all_analyses(days=7)
        rems = propose_remediations(days=1)
        logger.info(
            "Pattern analysis complete",
            extra={"remediations": len(rems)},
        )
    except Exception as exc:
        logger.error("Pattern analysis failed", extra={"error": str(exc)})


def job_daily_report() -> None:
    """Scheduled job: generate and store daily report."""
    logger.info("Scheduled daily report starting")
    try:
        from analysis.reporter import generate_daily_report
        result = generate_daily_report()
        logger.info(
            "Daily report complete",
            extra={"file": result.get("report_file", "")},
        )
    except Exception as exc:
        logger.error("Daily report failed", extra={"error": str(exc)})


def start_scheduler() -> None:
    """Configure and start the blocking APScheduler."""
    # Ensure database is initialised
    init_database()

    scheduler = BlockingScheduler(timezone="UTC")

    # Probe cycles: one job per scheduled hour
    probe_hours = settings.probe_schedule_hours
    probe_cron_hours = ",".join(str(h) for h in probe_hours)
    scheduler.add_job(
        job_probe_cycle,
        CronTrigger(hour=probe_cron_hours, minute=0),
        id="probe_cycle",
        name="RAGOps probe cycle",
        max_instances=1,
        misfire_grace_time=300,  # 5 minutes grace
    )
    logger.info(
        "Probe cycle job scheduled",
        extra={"hours_utc": probe_hours},
    )

    # Pattern analysis: daily at 23:00 UTC
    scheduler.add_job(
        job_pattern_analysis,
        CronTrigger(hour=settings.pattern_schedule_hour, minute=0),
        id="pattern_analysis",
        name="RAGOps pattern analysis",
        max_instances=1,
        misfire_grace_time=600,
    )

    # Daily report: daily at 07:00 UTC
    scheduler.add_job(
        job_daily_report,
        CronTrigger(hour=settings.report_schedule_hour, minute=0),
        id="daily_report",
        name="RAGOps daily report",
        max_instances=1,
        misfire_grace_time=600,
    )

    logger.info(
        "Scheduler starting",
        extra={
            "probe_hours": probe_hours,
            "pattern_hour": settings.pattern_schedule_hour,
            "report_hour": settings.report_schedule_hour,
        },
    )
    print("RAGOps scheduler started. Press Ctrl+C to stop.")
    print(f"  Probe cycles:    {probe_hours} UTC")
    print(f"  Pattern analysis: {settings.pattern_schedule_hour}:00 UTC")
    print(f"  Daily report:    {settings.report_schedule_hour}:00 UTC")

    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")
        scheduler.shutdown()


if __name__ == "__main__":
    start_scheduler()
