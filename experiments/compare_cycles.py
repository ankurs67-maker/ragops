"""Print the cross-cycle comparison table and Reflexion repeat-failure analysis
from the JSON written by run_5_cycles.py.

Usage: python experiments/compare_cycles.py [path-to-cycle_comparison.json]
"""

import json
import sys
from pathlib import Path

from config.settings import settings


def main() -> None:
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        candidates = sorted(settings.reports_dir.glob("cycle_comparison_*.json"))
        if not candidates:
            print("No cycle_comparison_*.json found in reports/")
            return
        path = candidates[-1]

    cycles = json.loads(path.read_text(encoding="utf-8"))
    if not cycles:
        print("No cycles recorded.")
        return

    # ── Comparison table ──
    cols = [
        ("cycle", "cycle_number"),
        ("pass", "pass_count"),
        ("fail", "fail_count"),
        ("retr", "avg_retrieval_relevance"),
        ("util", "avg_context_utilization"),
        ("faith", "avg_faithfulness"),
        ("fact", "avg_factuality"),
        ("refus", "avg_refusal_calibration"),
        ("health", "health_score"),
        ("secs", "cycle_elapsed_seconds"),
    ]
    header = " | ".join(f"{name:>7}" for name, _ in cols)
    print("\n=== CYCLE COMPARISON ===")
    print(header)
    print("-" * len(header))
    for c in cycles:
        row = []
        for _, key in cols:
            v = c.get(key, "—")
            if isinstance(v, float):
                v = f"{v:.3f}" if v <= 3 else f"{v:.1f}"
            row.append(f"{v:>7}")
        print(" | ".join(row))

    print("\n=== FAILURE DISTRIBUTION PER CYCLE ===")
    for c in cycles:
        print(f"  cycle {c['cycle_number']}: {c.get('failure_distribution', {})}")

    # ── Reflexion repeat-failure analysis ──
    print("\n=== REFLEXION REPEAT-FAILURE ANALYSIS ===")
    fail_sets = {c["cycle_number"]: set(c.get("failed_query_ids", {})) for c in cycles}
    first = fail_sets.get(1, set())
    last_cycle = max(fail_sets)
    ever_failed = set().union(*fail_sets.values())

    recovered, persistent_ids = [], []
    for qid in sorted(first):
        failed_in = [n for n, s in fail_sets.items() if qid in s]
        if last_cycle not in fail_sets or qid not in fail_sets[last_cycle]:
            recovered.append((qid, failed_in))
        else:
            persistent_ids.append((qid, failed_in))

    for qid, failed_in in recovered:
        passed_in = [n for n in fail_sets if n not in failed_in]
        print(f"  Query {qid} failed in cycle(s) {failed_in} but passed in "
              f"cycle(s) {sorted(passed_in)} after Reflexion lessons were applied")
    for qid, failed_in in persistent_ids:
        print(f"  Query {qid} failed persistently (cycles {failed_in}) — "
              f"no Reflexion recovery")

    new_failures_late = sorted(
        qid for qid in ever_failed
        if qid not in first and qid in fail_sets.get(last_cycle, set())
    )
    if new_failures_late:
        print(f"  Queries failing at end but NOT in cycle 1 (regressions/noise): "
              f"{new_failures_late}")

    if recovered and len(recovered) > len(persistent_ids):
        print(f"\n  Verdict: Reflexion effect plausible — {len(recovered)} of "
              f"{len(first)} cycle-1 failures recovered by cycle {last_cycle}.")
    elif not recovered:
        print("\n  Verdict: No measurable Reflexion effect observed.")
    else:
        print(f"\n  Verdict: Mixed — {len(recovered)} recovered, "
              f"{len(persistent_ids)} persistent. Recovery may be noise rather "
              f"than a Reflexion effect.")


if __name__ == "__main__":
    main()
