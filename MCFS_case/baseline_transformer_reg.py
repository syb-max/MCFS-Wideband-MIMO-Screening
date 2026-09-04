from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

import settings as S
from mcfs_shared import run_neural_validation


class TransformerRegressionBaseline(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scalar_projection = nn.Linear(1, S.D_MODEL)
        self.position = nn.Parameter(torch.empty(S.INPUT_DIM, S.D_MODEL))
        nn.init.normal_(self.position, std=0.02)
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
        self.readout = nn.Linear(S.D_MODEL, len(S.TARGETS))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.scalar_projection(x.unsqueeze(-1)) + self.position.unsqueeze(0)
        return self.readout(self.encoder(tokens).mean(dim=1))


def build_model() -> TransformerRegressionBaseline:
    return TransformerRegressionBaseline()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transformer-reg baseline: 5 seeds; max_epochs=300, patience=30, batch_size=64 in full mode."
    )
    parser.add_argument("--data", type=Path, default=S.DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=S.DEFAULT_OUTPUT / "baseline_transformer_reg")
    parser.add_argument("--mode", choices=["full", "smoke"], default="full")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    result = run_neural_validation("Transformer-reg", build_model, args.data, args.output, args.mode, args.device)
    print(result.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
