from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import settings as S
from baseline_gp import build_model as build_gp
from baseline_heuristic_uq import heuristic_uq_order
from baseline_j_only import j_only_order
from baseline_mlp import build_model as build_mlp
from baseline_random import random_order
from baseline_transformer_reg import build_model as build_transformer
from baseline_xgb_mcfs import make_predictions as make_xgb_predictions
from baseline_xgb_mcfs import xgb_mcfs_order
from baseline_xgboost import build_model as build_xgboost
from mcfs_shared import (
    audit_dataset,
    confidence_summary,
    freeze_all_and_score,
    load_archive,
    load_prospective_features,
    run_neural_validation,
    run_sklearn_validation,
    sha256,
    write_csv,
    write_json,
)
from proposed_mcfs import build_model as build_proposed
from proposed_mcfs import make_predictions as make_proposed_predictions
from proposed_mcfs import proposed_order


def run_surrogate_models(data: Path, output: Path, mode: str, device: str) -> pd.DataFrame:
    frames = [
        run_neural_validation("MLP", build_mlp, data, output / "baseline_mlp", mode, device),
        run_sklearn_validation("Gaussian process", build_gp, data, output / "baseline_gp", mode),
        run_sklearn_validation("XGBoost", build_xgboost, data, output / "baseline_xgboost", mode),
        run_neural_validation(
            "Transformer-reg", build_transformer, data, output / "baseline_transformer_reg", mode, device
        ),
        run_neural_validation("Compact encoder", build_proposed, data, output / "proposed_compact", mode, device),
    ]
    result = pd.concat(frames, ignore_index=True)
    write_csv(output / "surrogate_all_models_per_seed.csv", result)
    rows = []
    metrics = ["J_MAE", "Spearman_rho", "NDCG@5", "NDCG@10", "Regret@5", *[f"{m}_MAE" for m in S.MARGINS]]
    for model, group in result.groupby("model", sort=False):
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "n_seeds": len(values),
                }
            )
    write_csv(output / "surrogate_all_models_mean_std.csv", pd.DataFrame(rows))
    return result


def run_screening_replays(
    data: Path,
    output: Path,
    mode: str,
    device: str,
    n_replays: int,
    resume: bool,
) -> pd.DataFrame:
    rows = []
    for replay in range(n_replays):
        replay_dir = output / f"replay_{replay:03d}"
        metrics_path = replay_dir / "metrics.json"
        if resume and metrics_path.exists():
            existing = json.loads(metrics_path.read_text(encoding="utf-8"))
            if existing.get("dataset_sha256") != sha256(data):
                raise RuntimeError(f"cannot resume replay {replay}: dataset hash changed")
            expected_formal = mode == "full" and n_replays == S.N_REPLAYS
            expected_seeds = str(S.ensemble_seeds(mode, replay))
            expected_settings = S.settings_sha256()
            if (
                existing.get("formal_run") != expected_formal
                or existing.get("neural_max_epochs") != S.neural_epochs(mode)
                or existing.get("compact_seeds") != expected_seeds
                or existing.get("xgboost_seeds") != expected_seeds
                or existing.get("settings_sha256") != expected_settings
            ):
                raise RuntimeError(
                    f"cannot resume replay {replay}: existing result used a different mode, epoch count, or seed list"
                )
            rows.append(existing)
            print(f"replay {replay + 1}/{n_replays}: resumed", flush=True)
            continue
        print(f"replay {replay + 1}/{n_replays}: training", flush=True)
        proposed_prediction = make_proposed_predictions(data, mode, device, replay)
        proposed_rank, proposed_cert, proposed_risk, proposed_bounds, proposed_q, proposed_qrank = proposed_order(
            proposed_prediction
        )
        j_rank = j_only_order(proposed_prediction["pool_mean"])
        heuristic_rank, heuristic_risk = heuristic_uq_order(
            proposed_prediction["pool_mean"], proposed_prediction["pool_std"]
        )
        xgb_prediction = make_xgb_predictions(data, mode, replay)
        xgb_rank, xgb_cert, xgb_risk, xgb_bounds, xgb_q, xgb_qrank = xgb_mcfs_order(xgb_prediction)
        pool = proposed_prediction["pool"]
        specs = [
            {
                "method": "Random",
                "order": random_order(len(pool), S.random_seed(replay)),
            },
            {"method": "J-only", "order": j_rank},
            {"method": "Heuristic-UQ", "order": heuristic_rank, "risk": heuristic_risk},
            {
                "method": "XGB+MCFS",
                "order": xgb_rank,
                "certified": xgb_cert,
                "risk": xgb_risk,
                "bounds": xgb_bounds,
                "q": xgb_q,
                "quantile_rank": xgb_qrank,
            },
            {
                "method": "MCFS",
                "order": proposed_rank,
                "certified": proposed_cert,
                "risk": proposed_risk,
                "bounds": proposed_bounds,
                "q": proposed_q,
                "quantile_rank": proposed_qrank,
            },
        ]
        wide, _ = freeze_all_and_score(specs, pool, data, replay_dir, replay)
        wide.update(
            {
                "dataset_sha256": sha256(data),
                "compact_seeds": str(proposed_prediction["seeds"]),
                "xgboost_seeds": str(xgb_prediction["seeds"]),
                "neural_max_epochs": S.neural_epochs(mode),
                "neural_patience": S.neural_patience(mode),
                "formal_run": mode == "full" and n_replays == S.N_REPLAYS,
                "settings_sha256": S.settings_sha256(),
            }
        )
        write_json(metrics_path, wide)
        rows.append(wide)
        write_csv(output / "screening_metrics.partial.csv", pd.DataFrame(rows))
    result = pd.DataFrame(rows).sort_values("replay").reset_index(drop=True)
    write_csv(output / "screening_metrics.csv", result)
    write_csv(output / "screening_metrics_mean_std_ci95.csv", confidence_summary(result))
    partial = output / "screening_metrics.partial.csv"
    if partial.exists():
        partial.unlink()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run all separate MCFS models and baselines. Full mode: 100 replays x 5 members; "
            "neural members use max_epochs=300 and patience=30."
        )
    )
    parser.add_argument("--data", type=Path, default=S.DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=S.DEFAULT_OUTPUT / "paper_reproduction")
    parser.add_argument("--mode", choices=["full", "smoke"], default="full")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--replays",
        type=int,
        default=None,
        help="full: locked to 100; smoke: default 1",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-surrogate-models", action="store_true")
    args = parser.parse_args()
    n_replays = S.resolve_replay_count(args.mode, args.replays)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    # Before ranking, only archive labels and prospective features are loaded.
    load_archive(args.data)
    load_prospective_features(args.data)
    if not args.skip_surrogate_models:
        run_surrogate_models(args.data, output / "surrogate_models", args.mode, args.device)
    run_screening_replays(args.data, output / "screening_replays", args.mode, args.device, n_replays, args.resume)
    audit = audit_dataset(args.data)
    write_json(output / "dataset_audit.json", audit)
    write_json(
        output / "run_settings.json",
        {
            "mode": args.mode,
            "formal_run": args.mode == "full" and n_replays == S.N_REPLAYS,
            "base_member_seeds": S.BASE_MEMBER_SEEDS,
            "seed_rule": "member_seed = base_member_seed + 100 * replay_id",
            "ensemble_members_per_replay": S.ENSEMBLE_MEMBERS,
            "replay_seed_stride": S.REPLAY_SEED_STRIDE,
            "random_seed_rule": "permutation_seed = 2024 + replay_id",
            "formal_screening_replays": S.N_REPLAYS,
            "screening_replays_used": n_replays,
            "formal_proposed_member_trainings": S.N_REPLAYS * S.ENSEMBLE_MEMBERS,
            "formal_neural_max_epochs_per_member": S.MAX_EPOCHS,
            "formal_neural_patience": S.PATIENCE,
            "neural_max_epochs_used": S.neural_epochs(args.mode),
            "neural_patience_used": S.neural_patience(args.mode),
            "batch_size": S.BATCH_SIZE,
            "learning_rate": S.LEARNING_RATE,
            "weight_decay": S.WEIGHT_DECAY,
            "xgboost_trees_per_member": S.XGB_N_ESTIMATORS,
            "gp_epochs": None,
            "dataset_sha256": sha256(args.data),
            "settings_sha256": S.settings_sha256(),
        },
    )
    print(f"completed -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
