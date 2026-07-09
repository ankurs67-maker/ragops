CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL,
    applied_at DATETIME NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS probe_results (
    probe_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    timestamp DATETIME NOT NULL,
    query_id TEXT NOT NULL,
    query_text TEXT NOT NULL,
    category TEXT NOT NULL,
    difficulty TEXT,
    retrieved_chunks TEXT,
    generated_answer TEXT,
    correct_answer TEXT,
    answer_correct TEXT,
    refused_when_should INTEGER,
    latency_retrieval_ms INTEGER,
    latency_generation_ms INTEGER,
    latency_total_ms INTEGER
);

CREATE TABLE IF NOT EXISTS measurements (
    measurement_id TEXT PRIMARY KEY,
    probe_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    timestamp DATETIME NOT NULL,
    retrieval_relevance_score REAL,
    context_utilization_score REAL,
    faithfulness_score REAL,
    factuality_score REAL,
    refusal_calibration_score REAL,
    judge_model_version TEXT,
    judge_confidence REAL,
    failure_category TEXT,
    measurement_details TEXT,
    FOREIGN KEY (probe_id) REFERENCES probe_results(probe_id)
);

CREATE TABLE IF NOT EXISTS pattern_reports (
    report_id TEXT PRIMARY KEY,
    date DATE NOT NULL,
    timestamp DATETIME NOT NULL,
    overall_health_score REAL,
    alerts_triggered TEXT,
    dimension_scores TEXT,
    failure_distribution TEXT,
    category_breakdown TEXT,
    source_breakdown TEXT,
    top_finding TEXT,
    raw_analysis TEXT
);

CREATE TABLE IF NOT EXISTS remediations (
    remediation_id TEXT PRIMARY KEY,
    triggered_by TEXT,
    timestamp DATETIME NOT NULL,
    alert_type TEXT,
    root_cause TEXT,
    confidence REAL,
    remediation_text TEXT,
    specific_steps TEXT,
    priority TEXT,
    status TEXT DEFAULT 'pending',
    outcome TEXT
);

CREATE TABLE IF NOT EXISTS daily_reports (
    report_id TEXT PRIMARY KEY,
    date DATE NOT NULL,
    report_text TEXT,
    report_json TEXT,
    system_health_score REAL
);

CREATE INDEX IF NOT EXISTS idx_probe_run_id ON probe_results(run_id);
CREATE INDEX IF NOT EXISTS idx_probe_timestamp ON probe_results(timestamp);
CREATE INDEX IF NOT EXISTS idx_probe_category ON probe_results(category);
CREATE INDEX IF NOT EXISTS idx_probe_answer_correct ON probe_results(answer_correct);
CREATE INDEX IF NOT EXISTS idx_measure_probe_id ON measurements(probe_id);
CREATE INDEX IF NOT EXISTS idx_measure_timestamp ON measurements(timestamp);
CREATE INDEX IF NOT EXISTS idx_measure_failure ON measurements(failure_category);
CREATE INDEX IF NOT EXISTS idx_pattern_date ON pattern_reports(date);
CREATE INDEX IF NOT EXISTS idx_remediation_status ON remediations(status);
CREATE INDEX IF NOT EXISTS idx_remediation_priority ON remediations(priority);
CREATE INDEX IF NOT EXISTS idx_report_date ON daily_reports(date);
