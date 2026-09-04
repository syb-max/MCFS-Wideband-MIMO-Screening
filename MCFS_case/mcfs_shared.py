from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

import settings as S


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(_json_safe(value), handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_csv(path: str | Path, frame: pd.DataFrame) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_zip_csv(data_zip: str | Path, filename: str, usecols: list[str] | None = None) -> pd.DataFrame:
    data_zip = Path(data_zip)
    if not zipfile.is_zipfile(data_zip):
        raise ValueError(f"not a ZIP dataset: {data_zip}")
    with zipfile.ZipFile(data_zip) as archive:
        matches = [name for name in archive.namelist() if Path(name).name == filename]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one {filename}, found {len(matches)}")
        return pd.read_csv(io.BytesIO(archive.read(matches[0])), usecols=usecols)


def load_archive(data_zip: str | Path) -> pd.DataFrame:
    frame = _read_zip_csv(data_zip, S.TABLE_ARCHIVE)
    required = {"geometry_id", "split", *S.FEATURES, *S.TRUTH_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"archive missing columns: {sorted(missing)}")
    expected = {"train": S.TRAIN_N, "validation": S.VALIDATION_N, "calibration": S.CALIBRATION_N}
    counts = frame["split"].value_counts().to_dict()
    if len(frame) != S.ARCHIVE_N or counts != expected:
        raise ValueError(f"archive split mismatch: rows={len(frame)}, splits={counts}")
    _check_numeric_and_labels(frame, "archive")
    return frame


def load_prospective_features(data_zip: str | Path) -> pd.DataFrame:
    columns = ["geometry_id", "split", *S.FEATURES, "data_origin", "value_role", "is_prediction"]
    frame = _read_zip_csv(data_zip, S.TABLE_PROSPECTIVE, usecols=columns)
    if len(frame) != S.PROSPECTIVE_N or set(frame["split"].astype(str)) != {"prospective"}:
        raise ValueError("prospective feature table is not the locked 200-row pool")
    if frame["geometry_id"].astype(str).duplicated().any():
        raise ValueError("duplicate prospective geometry_id")
    if not np.isfinite(frame[S.FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("prospective features contain NaN/Inf")
    return frame


def load_prospective_truth(data_zip: str | Path) -> pd.DataFrame:
    frame = _read_zip_csv(data_zip, S.TABLE_PROSPECTIVE, usecols=["geometry_id", *S.TRUTH_COLUMNS])
    _check_numeric_and_labels(frame, "prospective")
    if int(frame["feasible"].sum()) != S.PROSPECTIVE_FEASIBLE_N:
        raise ValueError("prospective feasible count is not 24")
    return frame


def audit_dataset(data_zip: str | Path) -> dict[str, Any]:
    archive = load_archive(data_zip)
    prospective = _read_zip_csv(data_zip, S.TABLE_PROSPECTIVE)
    all_rows = _read_zip_csv(data_zip, S.TABLE_ALL)
    _check_numeric_and_labels(prospective, "prospective")
    _check_numeric_and_labels(all_rows, "all")
    if len(prospective) != S.PROSPECTIVE_N or len(all_rows) != S.ARCHIVE_N + S.PROSPECTIVE_N:
        raise ValueError("dataset row counts are not 500/200/700")
    expected = pd.concat([archive, prospective], ignore_index=True).sort_values("geometry_id").reset_index(drop=True)
    observed = all_rows.sort_values("geometry_id").reset_index(drop=True)
    columns = sorted(set(expected.columns) & set(observed.columns))
    pd.testing.assert_frame_equal(expected[columns], observed[columns], check_dtype=False, atol=1e-12, rtol=0)
    if set(archive["geometry_id"].astype(str)) & set(prospective["geometry_id"].astype(str)):
        raise ValueError("archive and prospective IDs overlap")
    if int(prospective["feasible"].sum()) != S.PROSPECTIVE_FEASIBLE_N:
        raise ValueError("prospective feasible count is not 24")
    return {
        "status": "PASS",
        "rows": {"all": len(all_rows), "archive": len(archive), "prospective": len(prospective)},
        "splits": archive["split"].value_counts().to_dict(),
        "prospective_feasible": int(prospective["feasible"].sum()),
        "dataset_sha256": sha256(data_zip),
        "labels_are_used_as_targets": True,
        "labels_are_predictions": False,
    }


def _check_numeric_and_labels(frame: pd.DataFrame, name: str) -> None:
    numeric = [*S.FEATURES, *S.TRUTH_COLUMNS]
    available = [column for column in numeric if column in frame]
    if not np.isfinite(frame[available].to_numpy(dtype=float)).all():
        raise ValueError(f"{name} contains NaN/Inf")
    if set(S.MARGINS + ["G", "feasible"]).issubset(frame.columns):
        margins = frame[S.MARGINS].to_numpy(dtype=float)
        g = frame["G"].to_numpy(dtype=float)
        if not np.allclose(g, margins.max(axis=1), atol=1e-9, rtol=0):
            raise ValueError(f"{name}: G != max(margins)")
        if not np.array_equal(frame["feasible"].to_numpy(dtype=int), (g <= 0).astype(int)):
            raise ValueError(f"{name}: feasible != int(G<=0)")
    if "is_prediction" in frame and set(frame["is_prediction"].astype(int)) != {0}:
        raise ValueError(f"{name}: labels are marked as predictions")


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "Standardizer":
        values = np.asarray(values, dtype=float)
        return cls(values.mean(axis=0), np.maximum(values.std(axis=0, ddof=0), 1e-12))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=float) - self.mean) / self.scale

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float) * self.scale + self.mean


def seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def resolve_device(requested: str) -> Any:
    import torch

    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class TorchMember:
    model: Any
    seed: int
    x_scaler: Standardizer
    y_scaler: Standardizer
    best_epoch: int
    best_val_loss: float
    train_seconds: float
    parameter_count: int


def train_torch_member(
    builder: Callable[[], Any],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    seed: int,
    mode: str,
    device_name: str,
    checkpoint: str | Path | None = None,
) -> TorchMember:
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, TensorDataset

    seed_everything(seed)
    x_scaler = Standardizer.fit(train[S.FEATURES].to_numpy(dtype=float))
    y_scaler = Standardizer.fit(train[S.TARGETS].to_numpy(dtype=float))
    x_train = torch.tensor(x_scaler.transform(train[S.FEATURES]), dtype=torch.float32)
    y_train = torch.tensor(y_scaler.transform(train[S.TARGETS]), dtype=torch.float32)
    x_val = torch.tensor(x_scaler.transform(validation[S.FEATURES]), dtype=torch.float32)
    y_val = torch.tensor(y_scaler.transform(validation[S.TARGETS]), dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=S.BATCH_SIZE, shuffle=True, generator=generator
    )
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=256, shuffle=False)
    model = builder()
    device = resolve_device(device_name)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=S.LEARNING_RATE, weight_decay=S.WEIGHT_DECAY)

    def loss_fn(prediction: Any, target: Any) -> Any:
        margin = F.huber_loss(prediction[:, :4], target[:, :4], delta=S.HUBER_DELTA, reduction="mean")
        quality = F.huber_loss(prediction[:, 4], target[:, 4], delta=S.HUBER_DELTA, reduction="mean")
        return margin + S.LAMBDA_J * quality

    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    stale = 0
    started = time.perf_counter()
    for epoch in range(1, S.neural_epochs(mode) + 1):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x_batch), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), S.GRADIENT_CLIP)
            optimizer.step()
        model.eval()
        total = 0.0
        count = 0
        with torch.inference_mode():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                batch_loss = loss_fn(model(x_batch), y_batch)
                total += float(batch_loss.item()) * len(x_batch)
                count += len(x_batch)
        val_loss = total / count
        if val_loss < best_loss - S.MIN_DELTA:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= S.neural_patience(mode):
            break
    if best_state is None:
        raise RuntimeError("no neural checkpoint was produced")
    model.load_state_dict(best_state)
    model.cpu().eval()
    member = TorchMember(
        model=model,
        seed=seed,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        best_epoch=best_epoch,
        best_val_loss=best_loss,
        train_seconds=time.perf_counter() - started,
        parameter_count=sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    )
    if checkpoint is not None:
        target = Path(checkpoint)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "seed": seed,
                "state_dict": model.state_dict(),
                "x_mean": x_scaler.mean,
                "x_scale": x_scaler.scale,
                "y_mean": y_scaler.mean,
                "y_scale": y_scaler.scale,
                "best_epoch": best_epoch,
                "best_val_loss": best_loss,
            },
            target,
        )
    return member


def predict_torch(member: TorchMember, frame: pd.DataFrame) -> np.ndarray:
    import torch

    values = torch.tensor(member.x_scaler.transform(frame[S.FEATURES]), dtype=torch.float32)
    with torch.inference_mode():
        prediction = member.model(values).numpy()
    return member.y_scaler.inverse(prediction)


def train_torch_ensemble(
    builder: Callable[[], Any],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    seeds: list[int],
    mode: str,
    device: str,
    checkpoint_dir: str | Path | None = None,
) -> list[TorchMember]:
    members = []
    for index, seed in enumerate(seeds):
        checkpoint = Path(checkpoint_dir) / f"member_{index}_seed_{seed}.pt" if checkpoint_dir else None
        members.append(train_torch_member(builder, train, validation, seed, mode, device, checkpoint))
    return members


def ensemble_predict(members: list[Any], frame: pd.DataFrame, predictor: Callable[[Any, pd.DataFrame], np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if len(members) < 2:
        raise ValueError("an uncertainty ensemble needs at least two members")
    values = np.stack([predictor(member, frame) for member in members])
    return values.mean(axis=0), values.std(axis=0, ddof=1)


def ndcg_at_k(true_j: np.ndarray, predicted_j: np.ndarray, k: int) -> float:
    true_j = np.asarray(true_j, dtype=float)
    predicted_j = np.asarray(predicted_j, dtype=float)
    ranks = np.empty(len(true_j), dtype=int)
    ranks[np.argsort(true_j, kind="stable")] = np.arange(len(true_j))
    relevance = (len(true_j) - ranks).astype(float)
    k = min(k, len(true_j))
    discount = 1 / np.log2(np.arange(2, k + 2))
    predicted_order = np.argsort(predicted_j, kind="stable")[:k]
    ideal_order = np.argsort(true_j, kind="stable")[:k]
    gain = lambda order: np.sum((2 ** (relevance[order] / len(true_j)) - 1) * discount)
    ideal = gain(ideal_order)
    return float(gain(predicted_order) / ideal) if ideal else 0.0


def regression_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    selected = np.argsort(prediction[:, 4], kind="stable")[:5]
    result = {
        "J_MAE": float(mean_absolute_error(truth[:, 4], prediction[:, 4])),
        "J_RMSE": float(math.sqrt(mean_squared_error(truth[:, 4], prediction[:, 4]))),
        "Spearman_rho": float(spearmanr(truth[:, 4], prediction[:, 4]).statistic),
        "NDCG@5": ndcg_at_k(truth[:, 4], prediction[:, 4], 5),
        "NDCG@10": ndcg_at_k(truth[:, 4], prediction[:, 4], 10),
        "Regret@5": float(truth[selected, 4].min() - truth[:, 4].min()),
    }
    for index, name in enumerate(S.MARGINS):
        result[f"{name}_MAE"] = float(mean_absolute_error(truth[:, index], prediction[:, index]))
    return result


def run_neural_validation(
    name: str,
    builder: Callable[[], Any],
    data_zip: str | Path,
    output_dir: str | Path,
    mode: str,
    device: str,
) -> pd.DataFrame:
    archive = load_archive(data_zip)
    train = archive[archive["split"] == "train"].reset_index(drop=True)
    validation = archive[archive["split"] == "validation"].reset_index(drop=True)
    rows = []
    output = Path(output_dir)
    for seed in S.model_seeds(mode):
        member = train_torch_member(builder, train, validation, seed, mode, device, output / f"seed_{seed}.pt")
        prediction = predict_torch(member, validation)
        row = regression_metrics(validation[S.TARGETS].to_numpy(dtype=float), prediction)
        row.update(
            {
                "model": name,
                "seed": seed,
                "epochs_max": S.neural_epochs(mode),
                "best_epoch": member.best_epoch,
                "best_val_loss": member.best_val_loss,
                "parameter_count": member.parameter_count,
                "train_seconds": member.train_seconds,
                "settings_sha256": S.settings_sha256(),
            }
        )
        rows.append(row)
    results = pd.DataFrame(rows)
    write_csv(output / "metrics_per_seed.csv", results)
    write_csv(output / "metrics_mean_std.csv", mean_std_table(results, ["seed"]))
    return results


def run_sklearn_validation(
    name: str,
    factory: Callable[[int, str], Any],
    data_zip: str | Path,
    output_dir: str | Path,
    mode: str,
) -> pd.DataFrame:
    import joblib

    archive = load_archive(data_zip)
    train = archive[archive["split"] == "train"].reset_index(drop=True)
    validation = archive[archive["split"] == "validation"].reset_index(drop=True)
    scaler = StandardScaler().fit(train[S.FEATURES].to_numpy(dtype=float))
    x_train = scaler.transform(train[S.FEATURES].to_numpy(dtype=float))
    x_validation = scaler.transform(validation[S.FEATURES].to_numpy(dtype=float))
    y_train = train[S.TARGETS].to_numpy(dtype=float)
    output = Path(output_dir)
    rows = []
    for seed in S.model_seeds(mode):
        model = factory(seed, mode)
        started = time.perf_counter()
        model.fit(x_train, y_train)
        seconds = time.perf_counter() - started
        prediction = np.asarray(model.predict(x_validation), dtype=float)
        row = regression_metrics(validation[S.TARGETS].to_numpy(dtype=float), prediction)
        row.update({"model": name, "seed": seed, "train_seconds": seconds, "settings_sha256": S.settings_sha256()})
        if name == "XGBoost":
            row["boosting_trees"] = S.xgb_estimators(mode)
        if name == "Gaussian process":
            row["epochs_applicable"] = False
            row["optimizer_restarts"] = S.gp_restarts(mode)
        rows.append(row)
        output.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "x_scaler": scaler, "seed": seed}, output / f"seed_{seed}.joblib")
    results = pd.DataFrame(rows)
    write_csv(output / "metrics_per_seed.csv", results)
    write_csv(output / "metrics_mean_std.csv", mean_std_table(results, ["seed"]))
    return results


def mean_std_table(frame: pd.DataFrame, exclude: list[str] | None = None) -> pd.DataFrame:
    excluded = set(exclude or [])
    rows = []
    for column in frame.select_dtypes(include=[np.number]).columns:
        if column in excluded:
            continue
        values = frame[column].dropna().to_numpy(dtype=float)
        if len(values):
            rows.append(
                {
                    "metric": column,
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "n": len(values),
                }
            )
    return pd.DataFrame(rows)


def calibrate(true_margins: np.ndarray, mean: np.ndarray, std: np.ndarray) -> tuple[float, int]:
    scale = np.maximum(np.asarray(std, dtype=float), S.EPS_SCALE)
    score = np.max((np.asarray(true_margins, dtype=float) - np.asarray(mean, dtype=float)) / scale, axis=1)
    rank = int(math.ceil((len(score) + 1) * (1 - S.ALPHA)))
    if rank != S.EXPECTED_CALIBRATION_RANK:
        raise ValueError(f"calibration rank={rank}, expected {S.EXPECTED_CALIBRATION_RANK}")
    return float(np.sort(score)[rank - 1]), rank


def mcfs_ranking(mean: np.ndarray, std: np.ndarray, q: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bounds = mean[:, :4] + q * np.maximum(std[:, :4], S.EPS_SCALE)
    risk = bounds.max(axis=1)
    certified = risk <= 0
    tier = (~certified).astype(int)
    second = np.where(certified, mean[:, 4], risk)
    order = np.lexsort((np.arange(len(mean)), mean[:, 4], second, tier)).astype(int)
    return order, certified, risk, bounds


def first_feasible(order: np.ndarray, feasible: np.ndarray) -> int:
    hit = np.flatnonzero(np.asarray(feasible, dtype=bool)[np.asarray(order, dtype=int)])
    return int(hit[0] + 1) if len(hit) else len(order) + 1


def freeze_and_score(
    method: str,
    order: np.ndarray,
    pool_features: pd.DataFrame,
    data_zip: str | Path,
    output_dir: str | Path,
    replay_id: int,
    certified: np.ndarray | None = None,
    risk: np.ndarray | None = None,
    bounds: np.ndarray | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    order = np.asarray(order, dtype=int)
    if sorted(order.tolist()) != list(range(len(pool_features))):
        raise ValueError("ranking is not a full prospective permutation")
    ids = pool_features["geometry_id"].astype(str).to_numpy()
    frozen = pd.DataFrame(
        {
            "method": method,
            "rank": np.arange(1, len(order) + 1),
            "candidate_index": order,
            "geometry_id": ids[order],
        }
    )
    if certified is not None:
        frozen["certified"] = np.asarray(certified, dtype=int)[order]
    if risk is not None:
        frozen["risk"] = np.asarray(risk, dtype=float)[order]
    if bounds is not None:
        for index, margin in enumerate(S.MARGINS):
            frozen[f"U_{margin}"] = np.asarray(bounds, dtype=float)[order, index]
    frozen_path = output / "frozen_order.csv"
    write_csv(frozen_path, frozen)
    order_hash = sha256(frozen_path)
    # Truth is intentionally read only after the complete order is on disk.
    truth = load_prospective_truth(data_zip).set_index("geometry_id").loc[ids].reset_index()
    feasible = truth["feasible"].to_numpy(dtype=int)
    true_g = truth["G"].to_numpy(dtype=float)
    first = first_feasible(order, feasible)
    metrics: dict[str, Any] = {
        "method": method,
        "replay": replay_id,
        "N_first": first,
        "frozen_order_sha256": order_hash,
    }
    for k in S.SUCCESS_K:
        metrics[f"Success@{k}"] = 100.0 * int(first <= k)
    if certified is not None:
        cert = np.asarray(certified, dtype=bool)
        metrics["CertifiedFraction"] = 100.0 * float(cert.mean())
        metrics["UnsafePass"] = 100.0 * float(np.mean(true_g[cert] > 0)) if cert.any() else float("nan")
        boundary = cert & (np.abs(true_g) <= S.BOUNDARY_DB)
        metrics["BoundaryUnsafePass"] = (
            100.0 * float(np.mean(true_g[boundary] > 0)) if boundary.any() else float("nan")
        )
        if bounds is not None:
            true_margins = truth[S.MARGINS].to_numpy(dtype=float)
            metrics["SimultaneousCoverage"] = 100.0 * float(np.mean(np.all(true_margins <= bounds, axis=1)))
    verification = truth.iloc[order][["geometry_id", "feasible", "G", *S.MARGINS, "J"]].copy()
    verification.insert(0, "rank", np.arange(1, len(order) + 1))
    write_csv(output / "postfreeze_verification.csv", verification)
    write_json(output / "metrics.json", metrics)
    return metrics


def confidence_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in frame.select_dtypes(include=[np.number]).columns:
        if column == "replay":
            continue
        values = frame[column].dropna().to_numpy(dtype=float)
        if not len(values):
            continue
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        half = float(t.ppf(0.975, len(values) - 1) * std / math.sqrt(len(values))) if len(values) > 1 else 0.0
        rows.append({"metric": column, "mean": mean, "std": std, "ci95_low": mean - half, "ci95_high": mean + half, "n": len(values)})
    return pd.DataFrame(rows)


def freeze_all_and_score(
    specs: list[dict[str, Any]],
    pool_features: pd.DataFrame,
    data_zip: str | Path,
    output_dir: str | Path,
    replay_id: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Freeze every method before loading any prospective target column."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ids = pool_features["geometry_id"].astype(str).to_numpy()
    frozen_parts = []
    for spec in specs:
        order = np.asarray(spec["order"], dtype=int)
        if sorted(order.tolist()) != list(range(len(ids))):
            raise ValueError(f"{spec['method']} ranking is not a complete permutation")
        part = pd.DataFrame(
            {
                "method": spec["method"],
                "rank": np.arange(1, len(order) + 1),
                "candidate_index": order,
                "geometry_id": ids[order],
            }
        )
        if spec.get("certified") is not None:
            part["certified"] = np.asarray(spec["certified"], dtype=int)[order]
        if spec.get("risk") is not None:
            part["risk"] = np.asarray(spec["risk"], dtype=float)[order]
        if spec.get("bounds") is not None:
            for index, margin in enumerate(S.MARGINS):
                part[f"U_{margin}"] = np.asarray(spec["bounds"], dtype=float)[order, index]
        frozen_parts.append(part)
    frozen = pd.concat(frozen_parts, ignore_index=True)
    frozen_path = output / "frozen_orders.csv"
    write_csv(frozen_path, frozen)
    frozen_hash = sha256(frozen_path)

    # No prospective target is loaded above this line.
    truth = load_prospective_truth(data_zip).set_index("geometry_id").loc[ids].reset_index()
    feasible = truth["feasible"].to_numpy(dtype=int)
    true_g = truth["G"].to_numpy(dtype=float)
    true_margins = truth[S.MARGINS].to_numpy(dtype=float)
    wide: dict[str, Any] = {"replay": replay_id, "frozen_orders_sha256": frozen_hash}
    long_rows = []
    verification_parts = []
    for spec in specs:
        method = str(spec["method"])
        order = np.asarray(spec["order"], dtype=int)
        first = first_feasible(order, feasible)
        row: dict[str, Any] = {"method": method, "replay": replay_id, "N_first": first}
        for k in S.SUCCESS_K:
            row[f"Success@{k}"] = 100.0 * int(first <= k)
        certified = spec.get("certified")
        if certified is not None:
            cert = np.asarray(certified, dtype=bool)
            row["CertifiedFraction"] = 100.0 * float(cert.mean())
            row["UnsafePass"] = 100.0 * float(np.mean(true_g[cert] > 0)) if cert.any() else float("nan")
            boundary = cert & (np.abs(true_g) <= S.BOUNDARY_DB)
            row["BoundaryUnsafePass"] = (
                100.0 * float(np.mean(true_g[boundary] > 0)) if boundary.any() else float("nan")
            )
            bounds = spec.get("bounds")
            if bounds is not None:
                row["SimultaneousCoverage"] = 100.0 * float(np.mean(np.all(true_margins <= bounds, axis=1)))
        if spec.get("q") is not None:
            row["q"] = float(spec["q"])
            row["quantile_rank"] = int(spec["quantile_rank"])
        long_rows.append(row)
        for key, value in row.items():
            if key not in {"method", "replay"}:
                wide[f"{method}_{key}"] = value
        verification = truth.iloc[order][["geometry_id", "feasible", "G", *S.MARGINS, "J"]].copy()
        verification.insert(0, "rank", np.arange(1, len(order) + 1))
        verification.insert(0, "method", method)
        verification_parts.append(verification)
    long_frame = pd.DataFrame(long_rows)
    write_csv(output / "metrics_by_method.csv", long_frame)
    write_csv(output / "postfreeze_verification.csv", pd.concat(verification_parts, ignore_index=True))
    write_json(output / "metrics.json", wide)
    return wide, long_frame
