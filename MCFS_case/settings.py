from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "MCFS_OPENEMS_LABELS_700_500_200_v1.1.zip"
DEFAULT_OUTPUT = ROOT / "outputs"

TABLE_ALL = "MCFS_OPENEMS_LABELS_ALL_700.csv"
TABLE_ARCHIVE = "MCFS_OPENEMS_LABELS_ARCHIVE_500.csv"
TABLE_PROSPECTIVE = "MCFS_OPENEMS_LABELS_PROSPECTIVE_200.csv"

FEATURES = ["Lp", "Wp", "Lsv", "Lsh", "Ws", "b", "c", "LT"]
MARGINS = ["g11", "g22", "g21", "g12"]
TARGETS = [*MARGINS, "J"]
TRUTH_COLUMNS = [*MARGINS, "G", "J", "feasible"]

ARCHIVE_N = 500
TRAIN_N = 350
VALIDATION_N = 75
CALIBRATION_N = 75
PROSPECTIVE_N = 200
PROSPECTIVE_FEASIBLE_N = 24

# The paper fixes five independently initialized members per replay but does
# not print their numeric seeds.  The user-frozen reproducibility roots are
# 2024--2028.  Replay r derives five distinct seeds by adding 100*r, so the
# formal 100 replays do not collapse into 100 identical reruns.
BASE_MEMBER_SEEDS = [2024, 2025, 2026, 2027, 2028]
ENSEMBLE_MEMBERS = 5
REPLAY_SEED_STRIDE = 100
RANDOM_SEED_ROOT = 2024

# Identical neural-training settings for Proposed, MLP, and Transformer-reg.
MAX_EPOCHS = 300
PATIENCE = 30
BATCH_SIZE = 64
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 1.0e-4
MIN_DELTA = 1.0e-6
HUBER_DELTA = 0.25
LAMBDA_J = 1.0
GRADIENT_CLIP = 5.0

# Paper compact encoder.
INPUT_DIM = 8
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 3
FFN_DIM = 128
DROPOUT = 0.1
EXPECTED_PROPOSED_PARAMETERS = 105_669

# Baseline settings absent from the paper table are explicit here.
MLP_WIDTH = 128
XGB_N_ESTIMATORS = 600
XGB_MAX_DEPTH = 5
XGB_LEARNING_RATE = 0.03
XGB_SUBSAMPLE = 0.85
XGB_COLSAMPLE = 0.90
GP_MATERN_NU = 2.5
GP_NOISE_LEVEL = 1.0e-4
GP_RESTARTS = 1

ALPHA = 0.05
EPS_SCALE = 1.0e-8
EXPECTED_CALIBRATION_RANK = 73
BOUNDARY_DB = 1.0
SUCCESS_K = [1, 3, 5, 10, 20]

# The paper fixes 100 stochastic retraining replays but does not print their
# numeric seed schedule.  The deterministic derivation above is an explicit
# reproducibility setting, not a claimed manuscript value.
N_REPLAYS = 100


def neural_epochs(mode: str) -> int:
    return 2 if mode == "smoke" else MAX_EPOCHS


def neural_patience(mode: str) -> int:
    return 2 if mode == "smoke" else PATIENCE


def model_seeds(mode: str) -> list[int]:
    return BASE_MEMBER_SEEDS[:1] if mode == "smoke" else list(BASE_MEMBER_SEEDS)


def ensemble_seeds(mode: str, replay_id: int | None = None) -> list[int]:
    if mode == "smoke":
        return BASE_MEMBER_SEEDS[:2]
    replay = 0 if replay_id is None else int(replay_id)
    if replay < 0 or replay >= N_REPLAYS:
        raise ValueError(f"replay_id must be in [0,{N_REPLAYS - 1}], received {replay}")
    return [seed + REPLAY_SEED_STRIDE * replay for seed in BASE_MEMBER_SEEDS]


def random_seed(replay_id: int) -> int:
    replay = int(replay_id)
    if replay < 0 or replay >= N_REPLAYS:
        raise ValueError(f"replay_id must be in [0,{N_REPLAYS - 1}], received {replay}")
    return RANDOM_SEED_ROOT + replay


def replay_count(mode: str) -> int:
    return 1 if mode == "smoke" else N_REPLAYS


def resolve_replay_count(mode: str, requested: int | None) -> int:
    count = replay_count(mode) if requested is None else int(requested)
    if count < 1:
        raise ValueError("replays must be positive")
    if mode == "full" and count != N_REPLAYS:
        raise ValueError(
            f"paper-aligned full mode is locked to {N_REPLAYS} replays; "
            "use --mode smoke for a short test"
        )
    return count


def xgb_estimators(mode: str) -> int:
    return 10 if mode == "smoke" else XGB_N_ESTIMATORS


def gp_restarts(mode: str) -> int:
    return 0 if mode == "smoke" else GP_RESTARTS


def reproducibility_settings() -> dict[str, object]:
    return {
        "base_member_seeds": BASE_MEMBER_SEEDS,
        "ensemble_members": ENSEMBLE_MEMBERS,
        "replay_seed_stride": REPLAY_SEED_STRIDE,
        "random_seed_root": RANDOM_SEED_ROOT,
        "replays": N_REPLAYS,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "alpha": ALPHA,
        "calibration_rank": EXPECTED_CALIBRATION_RANK,
        "proposed_parameters": EXPECTED_PROPOSED_PARAMETERS,
        "xgb_estimators": XGB_N_ESTIMATORS,
    }


def settings_sha256() -> str:
    payload = json.dumps(reproducibility_settings(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
