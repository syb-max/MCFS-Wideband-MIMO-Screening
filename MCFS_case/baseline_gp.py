from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.multioutput import MultiOutputRegressor

import settings as S
from mcfs_shared import run_sklearn_validation


def build_model(seed: int, mode: str) -> MultiOutputRegressor:
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=np.ones(S.INPUT_DIM), length_scale_bounds=(1e-3, 1e3), nu=S.GP_MATERN_NU
    ) + WhiteKernel(noise_level=S.GP_NOISE_LEVEL, noise_level_bounds=(1e-8, 1e-1))
    model = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        random_state=seed,
        n_restarts_optimizer=S.gp_restarts(mode),
    )
    return MultiOutputRegressor(model, n_jobs=1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Gaussian-process baseline: 5 seeded fits; GP has no epochs.")
    parser.add_argument("--data", type=Path, default=S.DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=S.DEFAULT_OUTPUT / "baseline_gp")
    parser.add_argument("--mode", choices=["full", "smoke"], default="full")
    args = parser.parse_args()
    result = run_sklearn_validation("Gaussian process", build_model, args.data, args.output, args.mode)
    print(result.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
