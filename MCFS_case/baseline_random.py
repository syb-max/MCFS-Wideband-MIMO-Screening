from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import settings as S
from mcfs_shared import confidence_summary, freeze_and_score, load_prospective_features, write_csv


def random_order(n_candidates: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).permutation(n_candidates).astype(int)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Random screening baseline. Full mode evaluates 100 deterministic, distinct permutations."
    )
    parser.add_argument("--data", type=Path, default=S.DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=S.DEFAULT_OUTPUT / "baseline_random")
    parser.add_argument("--mode", choices=["full", "smoke"], default="full")
    parser.add_argument("--replays", type=int, default=None, help="full: locked to 100; smoke: default 1")
    args = parser.parse_args()
    n_replays = S.resolve_replay_count(args.mode, args.replays)
    pool = load_prospective_features(args.data)
    rows = []
    for replay in range(n_replays):
        seed = S.random_seed(replay)
        order = random_order(len(pool), seed)
        row = freeze_and_score("Random", order, pool, args.data, args.output / f"replay_{replay:03d}", replay)
        row.update({"permutation_seed": seed, "settings_sha256": S.settings_sha256()})
        rows.append(row)
    result = pd.DataFrame(rows)
    write_csv(args.output / "metrics.csv", result)
    write_csv(args.output / "metrics_mean_std.csv", confidence_summary(result))
    print(result.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
