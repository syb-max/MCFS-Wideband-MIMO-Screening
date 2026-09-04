from __future__ import annotations

import torch

import settings as S
from mcfs_shared import audit_dataset
from proposed_mcfs import build_model


def main() -> int:
    report = audit_dataset(S.DEFAULT_DATA)
    assert report["rows"] == {"all": 700, "archive": 500, "prospective": 200}
    assert S.BASE_MEMBER_SEEDS == [2024, 2025, 2026, 2027, 2028]
    assert S.ENSEMBLE_MEMBERS == 5
    assert S.N_REPLAYS == 100
    assert S.ensemble_seeds("full", 0) == [2024, 2025, 2026, 2027, 2028]
    assert S.ensemble_seeds("full", 1) == [2124, 2125, 2126, 2127, 2128]
    all_member_seeds = [seed for replay in range(S.N_REPLAYS) for seed in S.ensemble_seeds("full", replay)]
    assert len(all_member_seeds) == 500 and len(set(all_member_seeds)) == 500
    assert [S.random_seed(replay) for replay in range(3)] == [2024, 2025, 2026]
    assert S.MAX_EPOCHS == 300 and S.PATIENCE == 30 and S.BATCH_SIZE == 64
    model = build_model()
    count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    assert count == 105_669
    assert tuple(model(torch.zeros(2, 8)).shape) == (2, 5)
    print("PASS")
    print(report)
    print("base member seeds:", S.BASE_MEMBER_SEEDS)
    print("formal protocol: 100 replays x 5 independently seeded members")
    print("replay 0 seeds:", S.ensemble_seeds("full", 0))
    print("replay 1 seeds:", S.ensemble_seeds("full", 1))
    print("proposed parameters:", count)
    print("neural epochs/patience/batch:", S.MAX_EPOCHS, S.PATIENCE, S.BATCH_SIZE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
