from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import settings as S
from mcfs_shared import confidence_summary, freeze_and_score, write_csv
from proposed_mcfs import make_predictions


def j_only_order(pool_mean: np.ndarray) -> np.ndarray:
    return np.argsort(pool_mean[:, 4], kind="stable").astype(int)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="J-only screening baseline. Full mode uses 100 replays and the replay-matched 5-member MCFS ensemble."
    )
    parser.add_argument("--data", type=Path, default=S.DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=S.DEFAULT_OUTPUT / "baseline_j_only")
    parser.add_argument("--mode", choices=["full", "smoke"], default="full")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--replays", type=int, default=None, help="full: locked to 100; smoke: default 1")
    args = parser.parse_args()
    n_replays = S.resolve_replay_count(args.mode, args.replays)
    rows = []
    for replay in range(n_replays):
        replay_id = replay
        prediction = make_predictions(args.data, args.mode, args.device, replay_id)
        order = j_only_order(prediction["pool_mean"])
        row = freeze_and_score(
            "J-only", order, prediction["pool"], args.data, args.output / f"replay_{replay:03d}", replay
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
