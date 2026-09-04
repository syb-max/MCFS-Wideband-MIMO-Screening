from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import settings as S
from mcfs_shared import (
    calibrate,
    confidence_summary,
    ensemble_predict,
    freeze_and_score,
    load_archive,
    load_prospective_features,
    mcfs_ranking,
    predict_torch,
    regression_metrics,
    train_torch_ensemble,
    write_csv,
)


class CompactGeometryEncoder(nn.Module):
    """Proposed paper model: 8 scalar tokens -> 4 margins + J."""

    def __init__(self) -> None:
        super().__init__()
        self.scalar_projection = nn.Linear(1, S.D_MODEL)
        self.variable_identity = nn.Parameter(torch.empty(S.INPUT_DIM, S.D_MODEL))
        nn.init.normal_(self.variable_identity, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=S.D_MODEL,
            nhead=S.N_HEADS,
            dim_feedforward=S.FFN_DIM,
            dropout=S.DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=S.N_LAYERS, enable_nested_tensor=False)
        self.pool_norm = nn.LayerNorm(S.D_MODEL)
        self.shared_hidden = nn.Linear(S.D_MODEL, S.D_MODEL)
        self.dropout = nn.Dropout(S.DROPOUT)
        self.margin_head = nn.Linear(S.D_MODEL, 4)
        self.objective_head = nn.Linear(S.D_MODEL, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != S.INPUT_DIM:
            raise ValueError(f"expected [batch,{S.INPUT_DIM}], received {tuple(x.shape)}")
        tokens = self.scalar_projection(x.unsqueeze(-1)) + self.variable_identity.unsqueeze(0)
        encoded = self.encoder(tokens)
        hidden = self.pool_norm(encoded.mean(dim=1))
        hidden = self.dropout(torch.nn.functional.gelu(self.shared_hidden(hidden)))
        return torch.cat([self.margin_head(hidden), self.objective_head(hidden)], dim=1)


def build_model() -> CompactGeometryEncoder:
    model = CompactGeometryEncoder()
    count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if count != S.EXPECTED_PROPOSED_PARAMETERS:
        raise RuntimeError(f"Proposed model has {count} parameters, expected {S.EXPECTED_PROPOSED_PARAMETERS}")
    return model


def make_predictions(
    data_zip: str | Path,
    mode: str,
    device: str,
    replay_id: int | None = None,
    checkpoint_dir: str | Path | None = None,
) -> dict[str, object]:
    archive = load_archive(data_zip)
    train = archive[archive["split"] == "train"].reset_index(drop=True)
    validation = archive[archive["split"] == "validation"].reset_index(drop=True)
    calibration = archive[archive["split"] == "calibration"].reset_index(drop=True)
    pool = load_prospective_features(data_zip)
    seeds = S.ensemble_seeds(mode, replay_id)
    members = train_torch_ensemble(build_model, train, validation, seeds, mode, device, checkpoint_dir)
    cal_mean, cal_std = ensemble_predict(members, calibration, predict_torch)
    pool_mean, pool_std = ensemble_predict(members, pool, predict_torch)
    val_mean, _ = ensemble_predict(members, validation, predict_torch)
    return {
        "archive": archive,
        "train": train,
        "validation": validation,
        "calibration": calibration,
        "pool": pool,
        "seeds": seeds,
        "members": members,
        "cal_mean": cal_mean,
        "cal_std": cal_std,
        "pool_mean": pool_mean,
        "pool_std": pool_std,
        "validation_metrics": regression_metrics(validation[S.TARGETS].to_numpy(dtype=float), val_mean),
    }


def proposed_order(predictions: dict[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int]:
    calibration = predictions["calibration"]
    q, rank = calibrate(
        calibration[S.MARGINS].to_numpy(dtype=float), predictions["cal_mean"][:, :4], predictions["cal_std"][:, :4]
    )
    order, certified, risk, bounds = mcfs_ranking(predictions["pool_mean"], predictions["pool_std"], q)
    return order, certified, risk, bounds, q, rank


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Proposed MCFS. Full mode runs 100 replays x 5 members; "
            "each member uses max_epochs=300 and patience=30."
        )
    )
    parser.add_argument("--data", type=Path, default=S.DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=S.DEFAULT_OUTPUT / "proposed_mcfs")
    parser.add_argument("--mode", choices=["full", "smoke"], default="full")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--replays", type=int, default=None, help="full: locked to 100; smoke: default 1")
    args = parser.parse_args()
    n_replays = S.resolve_replay_count(args.mode, args.replays)
    rows = []
    for replay in range(n_replays):
        replay_seed_id = replay
        output = args.output / f"replay_{replay:03d}"
        predictions = make_predictions(args.data, args.mode, args.device, replay_seed_id, output / "checkpoints")
        order, certified, risk, bounds, q, rank = proposed_order(predictions)
        row = freeze_and_score("MCFS", order, predictions["pool"], args.data, output, replay, certified, risk, bounds)
        row.update(
            {
                "q": q,
                "quantile_rank": rank,
                "member_seeds": str(predictions["seeds"]),
                "neural_max_epochs": S.neural_epochs(args.mode),
                "neural_patience": S.neural_patience(args.mode),
                "settings_sha256": S.settings_sha256(),
            }
        )
        row.update({f"validation_{key}": value for key, value in predictions["validation_metrics"].items()})
        rows.append(row)
    results = pd.DataFrame(rows)
    write_csv(args.output / "metrics.csv", results)
    write_csv(args.output / "metrics_mean_std.csv", confidence_summary(results))
    print(results.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
