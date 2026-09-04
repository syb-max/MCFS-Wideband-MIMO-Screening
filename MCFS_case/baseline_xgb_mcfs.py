from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import settings as S
from baseline_xgboost import predict_member, train_ensemble
from mcfs_shared import (
    calibrate,
    confidence_summary,
    ensemble_predict,
    freeze_and_score,
    load_archive,
    load_prospective_features,
    mcfs_ranking,
    write_csv,
)


def make_predictions(data_zip: str | Path, mode: str, replay_id: int | None = None) -> dict[str, object]:
    archive = load_archive(data_zip)
    train = archive[archive["split"] == "train"].reset_index(drop=True)
    calibration = archive[archive["split"] == "calibration"].reset_index(drop=True)
    pool = load_prospective_features(data_zip)
    seeds = S.ensemble_seeds(mode, replay_id)
    members = train_ensemble(train, seeds, mode)
    cal_mean, cal_std = ensemble_predict(members, calibration, predict_member)
    pool_mean, pool_std = ensemble_predict(members, pool, predict_member)
    return {
        "calibration": calibration,
        "pool": pool,
        "seeds": seeds,
        "cal_mean": cal_mean,
        "cal_std": cal_std,
        "pool_mean": pool_mean,
        "pool_std": pool_std,
    }


def xgb_mcfs_order(predictions: dict[str, object]):
    q, rank = calibrate(
        predictions["calibration"][S.MARGINS].to_numpy(dtype=float),
        predictions["cal_mean"][:, :4],
        predictions["cal_std"][:, :4],
    )
    order, certified, risk, bounds = mcfs_ranking(predictions["pool_mean"], predictions["pool_std"], q)
    return order, certified, risk, bounds, q, rank


def main() -> int:
    parser = argparse.ArgumentParser(
        description="XGB+MCFS screening baseline. Full mode runs 100 replays x 5 XGBoost members."
    )
    parser.add_argument("--data", type=Path, default=S.DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=S.DEFAULT_OUTPUT / "baseline_xgb_mcfs")
    parser.add_argument("--mode", choices=["full", "smoke"], default="full")
    parser.add_argument("--replays", type=int, default=None, help="full: locked to 100; smoke: default 1")
    args = parser.parse_args()
    n_replays = S.resolve_replay_count(args.mode, args.replays)
    rows = []
    for replay in range(n_replays):
        replay_id = replay
        prediction = make_predictions(args.data, args.mode, replay_id)
        order, certified, risk, bounds, q, rank = xgb_mcfs_order(prediction)
        row = freeze_and_score(
            "XGB+MCFS", order, prediction["pool"], args.data, args.output / f"replay_{replay:03d}", replay, certified, risk, bounds
        )
        row.update(
            {
                "q": q,
                "quantile_rank": rank,
                "member_seeds": str(prediction["seeds"]),
                "xgb_estimators": S.xgb_estimators(args.mode),
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
