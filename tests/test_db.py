"""Tests for database/db_client.py."""

import sqlite3
from pathlib import Path

import pytest


@pytest.fixture
def test_db(monkeypatch, tmp_path):
    """Override db_path property on the Settings class to use a temp database."""
    test_db_path = tmp_path / "test_ragops.db"

    from config.settings import Settings

    def fake_db_path(self):
        return test_db_path

    monkeypatch.setattr(Settings, "db_path", property(fake_db_path))

    from database.db_client import init_database
    init_database()
    return test_db_path


def test_init_database_creates_tables(test_db):
    """init_database should create all required tables."""
    conn = sqlite3.connect(str(test_db))
    tables = {
        row[0] for row in
        conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    conn.close()
    assert "probe_results" in tables
    assert "measurements" in tables
    assert "pattern_reports" in tables
    assert "remediations" in tables
    assert "daily_reports" in tables
    assert "schema_version" in tables


def test_get_system_health_score_no_data(test_db):
    """With no data, health score should return 100.0."""
    from database.db_client import get_system_health_score
    score = get_system_health_score()
    assert score == 100.0


def test_get_failure_distribution_empty(test_db):
    """With no data, failure distribution should be empty dict."""
    from database.db_client import get_failure_distribution
    dist = get_failure_distribution(days=7)
    assert dist == {}


def test_insert_and_retrieve_remediation(test_db):
    """insert_remediation should persist and be retrievable."""
    from database.db_client import insert_remediation, get_pending_remediations
    rem = {
        "remediation_id": "test-rem-001",
        "triggered_by": "test",
        "timestamp": "2026-06-29T12:00:00+00:00",
        "alert_type": "CONTEXT_BYPASS",
        "root_cause": "Model ignoring context",
        "confidence": 0.8,
        "remediation_text": "Fix system prompt.",
        "specific_steps": ["Step 1", "Step 2"],
        "priority": "high",
        "status": "pending",
        "outcome": None,
    }
    insert_remediation(rem)
    pending = get_pending_remediations()
    assert len(pending) == 1
    assert pending[0]["alert_type"] == "CONTEXT_BYPASS"
