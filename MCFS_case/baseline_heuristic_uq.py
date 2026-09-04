from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import settings as S
from mcfs_shared import confidence_summary, freeze_and_score, write_csv
from proposed_mcfs import make_predictions


def heuristic_uq_order(pool_mean: np.ndarray, pool_std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    risk = np.max(pool_mean[:, :4] + np.maximum(pool_std[:, :4], S.EPS_SCALE), axis=1)
    order = np.lexsort((np.arange(len(risk)), pool_mean[:, 4], risk)).astype(int)
    return order, risk


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Heuristic-UQ baseline. Full mode uses 100 replays and the replay-matched 5-member MCFS ensemble."
    )
    parser.add_argument("--data", type=Path, default=S.DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=S.DEFAULT_OUTPUT / "baseline_heuristic_uq")
    parser.add_argument("--mode", choices=["full", "smoke"], default="full")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--replays", type=int, default=None, help="full: locked to 100; smoke: default 1")
    args = parser.parse_args()
    n_replays = S.resolve_replay_count(args.mode, args.replays)
    rows = []
    for replay in range(n_replays):
        replay_id = replay
        prediction = make_predictions(args.data, args.mode, args.device, replay_id)
        order, risk = heuristic_uq_order(prediction["pool_mean"], prediction["pool_std"])
        row = freeze_and_score(
            "Heuristic-UQ",
            order,
            prediction["pool"],
            args.data,
            args.output / f"replay_{replay:03d}",
            replay,
            risk=risk,
        )
        row.update(
            {
                "member_seeds": str(prediction["seeds"]),
                "neural_max_epochs": S.neural_epochs(args.mode),
                "neural_patience": S.neural_patience(args.mode),
                "settings_sha256": S.settings_sha256(),
            }
        )
        rows.append(row)
    result = pd.DataFrame(rows)
    write_csv(args.output / "metrics.csv", result)
    write_csv(args.output / "metrics_mean_std.csv", confidence_summary(result))
    print(result.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
