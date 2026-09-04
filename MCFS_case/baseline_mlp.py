from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

import settings as S
from mcfs_shared import run_neural_validation


class MLPBaseline(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(S.INPUT_DIM, S.MLP_WIDTH),
            nn.GELU(),
            nn.LayerNorm(S.MLP_WIDTH),
            nn.Dropout(S.DROPOUT),
            nn.Linear(S.MLP_WIDTH, S.MLP_WIDTH),
            nn.GELU(),
            nn.Dropout(S.DROPOUT),
            nn.Linear(S.MLP_WIDTH, len(S.TARGETS)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def build_model() -> MLPBaseline:
    return MLPBaseline()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MLP regression baseline: 5 seeds; max_epochs=300, patience=30, batch_size=64 in full mode."
    )
    parser.add_argument("--data", type=Path, default=S.DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=S.DEFAULT_OUTPUT / "baseline_mlp")
    parser.add_argument("--mode", choices=["full", "smoke"], default="full")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    result = run_neural_validation("MLP", build_model, args.data, args.output, args.mode, args.device)
    print(result.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
