"""Sample 10 probes from the most recent probe cycle for a manual
inter-rater reliability check: a human (or reviewing agent) reads each
generated answer and states whether they agree with the automated
failure_category. Standard research validity check.

Usage: python experiments/interrater_sample.py
"""

import json
import random

from database.db_client import get_connection


def main() -> None:
    with get_connection() as conn:
        run_id = conn.execute(
            "SELECT run_id FROM measurements ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        if not run_id:
            print("No measurements found.")
            return
        run_id = run_id["run_id"]

        rows = conn.execute(
            """
            SELECT p.query_id, p.query_text, p.generated_answer, p.correct_answer,
                   m.failure_category, m.retrieval_relevance_score,
                   m.faithfulness_score, m.factuality_score,
                   m.refusal_calibration_score
            FROM measurements m
            JOIN probe_results p ON p.probe_id = m.probe_id
            WHERE m.run_id = :rid
            """,
            {"rid": run_id},
        ).fetchall()

    rows = [dict(r) for r in rows]
    random.seed(7)
    sample = random.sample(rows, min(10, len(rows)))

    print(f"Run: {run_id} — {len(rows)} probes, sampling {len(sample)}\n")
    for i, r in enumerate(sample, 1):
        print(f"--- [{i}] {r['query_id']} | automated: {r['failure_category']} ---")
        print(f"Q: {r['query_text']}")
        print(f"A: {(r['generated_answer'] or '')[:400]}")
        print(f"Expected: {r['correct_answer']}")
        print(f"Scores: retr={r['retrieval_relevance_score']}, "
              f"faith={r['faithfulness_score']}, fact={r['factuality_score']}, "
              f"refusal={r['refusal_calibration_score']}")
        print()


if __name__ == "__main__":
    main()
