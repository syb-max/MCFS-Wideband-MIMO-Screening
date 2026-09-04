"""Run all available screening pipelines."""

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from baseline_heuristic_uq import run as run_baseline_heuristic_uq
from baseline_j_only import run as run_baseline_j_only
from baseline_random import run as run_baseline_random
from baseline_xgb_mcfs import run as run_baseline_xgb_mcfs
from proposed_mcfs import run as run_proposed_mcfs


def run_all():
    """Run all baseline and proposed pipelines."""
    runners = [
        ("proposed_mcfs", run_proposed_mcfs),
        ("baseline_random", run_baseline_random),
        ("baseline_j_only", run_baseline_j_only),
        ("baseline_heuristic_uq", run_baseline_heuristic_uq),
        ("baseline_xgb_mcfs", run_baseline_xgb_mcfs),
    ]

    results = []
    for name, runner in runners:
        try:
            results.append(runner())
        except Exception as exc:  # pragma: no cover
            results.append(f"{name}: ERROR: {exc}")
    return results


if __name__ == "__main__":
    print("\n".join(run_all()))
