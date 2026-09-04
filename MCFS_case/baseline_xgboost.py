from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

import settings as S
from mcfs_shared import run_sklearn_validation


def build_model(seed: int, mode: str) -> MultiOutputRegressor:
    base = XGBRegressor(
        n_estimators=S.xgb_estimators(mode),
        max_depth=S.XGB_MAX_DEPTH,
        learning_rate=S.XGB_LEARNING_RATE,
        subsample=S.XGB_SUBSAMPLE,
        colsample_bytree=S.XGB_COLSAMPLE,
        reg_lambda=1.0,
        reg_alpha=0.0,
        objective="reg:squarederror",
        random_state=seed,
        n_jobs=1,
        verbosity=0,
    )
    return MultiOutputRegressor(base, n_jobs=1)


@dataclass
class XGBMember:
    model: MultiOutputRegressor
    scaler: StandardScaler
    seed: int
    train_seconds: float


def train_ensemble(train: pd.DataFrame, seeds: list[int], mode: str) -> list[XGBMember]:
    scaler = StandardScaler().fit(train[S.FEATURES].to_numpy(dtype=float))
    x = scaler.transform(train[S.FEATURES].to_numpy(dtype=float))
    y = train[S.TARGETS].to_numpy(dtype=float)
    members = []
    for seed in seeds:
        model = build_model(seed, mode)
        started = time.perf_counter()
        model.fit(x, y)
        members.append(XGBMember(model, scaler, seed, time.perf_counter() - started))
    return members


def predict_member(member: XGBMember, frame: pd.DataFrame) -> np.ndarray:
    features = frame[S.FEATURES].to_numpy(dtype=float)
    return np.asarray(member.model.predict(member.scaler.transform(features)), dtype=float)


def main() -> int:
    parser = argparse.ArgumentParser(description="XGBoost regression baseline: 5 seeded fits; 600 trees per fit in full mode.")
    parser.add_argument("--data", type=Path, default=S.DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=S.DEFAULT_OUTPUT / "baseline_xgboost")
    parser.add_argument("--mode", choices=["full", "smoke"], default="full")
    args = parser.parse_args()
    result = run_sklearn_validation("XGBoost", build_model, args.data, args.output, args.mode)
    print(result.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
