"""Toy surrogate comparison for archived Euler-1 angle evolution data.

This script is meant to produce a compact model-comparison table for the
reviewer question about surrogate choices. It uses the small HDF5 split in
``Archive (1)`` and compares the existing FNO architecture with simpler
baselines that have less spatial/global modeling capacity.

Example:
    cd example
    python compare_euler_surrogates.py --epochs 5 --models persistence,linear_pointwise,tiny_cnn,fno
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from timeit import default_timer

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model_euler import FNO2d  # noqa: E402


DATASET_NAMES = ("euler1_known", "euler1_predict", "strain_rate", "temperature", "pressure")


class Euler1ArchiveDataset(Dataset):
    """Read one Euler-1 HDF5 split from Archive (1)."""

    def __init__(self, h5_path: Path, preload: bool = True):
        self.h5_path = Path(h5_path)
        self.preload = bool(preload)
        self.arrays: dict[str, torch.Tensor] | None = None
        with h5py.File(self.h5_path, "r") as f:
            missing = [name for name in DATASET_NAMES if name not in f]
            if missing:
                raise KeyError(f"{self.h5_path} is missing datasets: {missing}")

            self.n_samples = int(f["euler1_known"].shape[0])
            self.grid_size = int(f["euler1_known"].shape[1])
            self.step_known = int(f["euler1_known"].shape[-1])
            self.step_predict = int(f["euler1_predict"].shape[-1])
            for name in DATASET_NAMES[1:]:
                if int(f[name].shape[0]) != self.n_samples:
                    raise ValueError(
                        f"Inconsistent sample count in {self.h5_path}: "
                        f"euler1_known has {self.n_samples}, {name} has {f[name].shape[0]}"
                    )

            if self.preload:
                self.arrays = {name: torch.from_numpy(f[name][...]).float() for name in DATASET_NAMES}

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        if self.arrays is not None:
            return tuple(self.arrays[name][idx] for name in DATASET_NAMES)

        with h5py.File(self.h5_path, "r") as f:
            euler_known = torch.from_numpy(f["euler1_known"][idx]).float()
            euler_predict = torch.from_numpy(f["euler1_predict"][idx]).float()
            strain_rate = torch.from_numpy(f["strain_rate"][idx]).float()
            temperature = torch.from_numpy(f["temperature"][idx]).float()
            pressure = torch.from_numpy(f["pressure"][idx]).float()

        return euler_known, euler_predict, strain_rate, temperature, pressure


def coordinate_grid(batch_size: int, size_x: int, size_y: int, device: torch.device) -> torch.Tensor:
    grid_x = torch.linspace(0.0, 1.0, size_x, device=device)
    grid_y = torch.linspace(0.0, 1.0, size_y, device=device)
    grid_x = grid_x.reshape(1, size_x, 1, 1).repeat(batch_size, 1, size_y, 1)
    grid_y = grid_y.reshape(1, 1, size_y, 1).repeat(batch_size, size_x, 1, 1)
    return torch.cat((grid_x, grid_y), dim=-1)


def nhwc_features(
    x: torch.Tensor,
    strain_rate: torch.Tensor,
    temperature: torch.Tensor,
    pressure: torch.Tensor,
) -> torch.Tensor:
    batch_size, size_x, size_y, _ = x.shape
    grid = coordinate_grid(batch_size, size_x, size_y, x.device)
    return torch.cat((x, grid, strain_rate, temperature, pressure), dim=-1)


class PersistenceSurrogate(nn.Module):
    """No-training baseline: predict the known Euler field unchanged."""

    def forward(
        self,
        x: torch.Tensor,
        strain_rate: torch.Tensor,
        temperature: torch.Tensor,
        pressure: torch.Tensor,
    ) -> torch.Tensor:
        del strain_rate, temperature, pressure
        return x[..., -1:]


class LinearPointwiseSurrogate(nn.Module):
    """Per-pixel linear model with no spatial context."""

    def __init__(self, step_known: int):
        super().__init__()
        self.fc = nn.Linear(4 * step_known + 2, 1)

    def forward(
        self,
        x: torch.Tensor,
        strain_rate: torch.Tensor,
        temperature: torch.Tensor,
        pressure: torch.Tensor,
    ) -> torch.Tensor:
        features = nhwc_features(x, strain_rate, temperature, pressure)
        return torch.tanh(self.fc(features))


class PointwiseMLPSurrogate(nn.Module):
    """Per-pixel MLP with no convolutional or spectral spatial coupling."""

    def __init__(self, step_known: int, hidden: int, layers: int, activation: str):
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be >= 1")

        activation_layer = build_activation(activation)
        input_dim = 4 * step_known + 2
        blocks: list[nn.Module] = []
        last_dim = input_dim
        for _ in range(max(0, layers - 1)):
            blocks.append(nn.Linear(last_dim, hidden))
            blocks.append(build_activation(activation))
            last_dim = hidden
        blocks.append(nn.Linear(last_dim, 1))
        self.net = nn.Sequential(*blocks)
        self.output_activation = activation_layer

    def forward(
        self,
        x: torch.Tensor,
        strain_rate: torch.Tensor,
        temperature: torch.Tensor,
        pressure: torch.Tensor,
    ) -> torch.Tensor:
        features = nhwc_features(x, strain_rate, temperature, pressure)
        return self.output_activation(self.net(features))


class TinyCNNSurrogate(nn.Module):
    """Small local CNN baseline without global Fourier modes."""

    def __init__(self, step_known: int, channels: int, layers: int, kernel_size: int, activation: str):
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be >= 1")
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd so the output grid size is unchanged")

        in_channels = 4 * step_known + 2
        padding = kernel_size // 2
        blocks: list[nn.Module] = []
        last_channels = in_channels
        for _ in range(max(0, layers - 1)):
            blocks.append(nn.Conv2d(last_channels, channels, kernel_size, padding=padding))
            blocks.append(build_activation(activation))
            last_channels = channels
        blocks.append(nn.Conv2d(last_channels, 1, kernel_size, padding=padding))
        self.net = nn.Sequential(*blocks)
        self.output_activation = build_activation(activation)

    def forward(
        self,
        x: torch.Tensor,
        strain_rate: torch.Tensor,
        temperature: torch.Tensor,
        pressure: torch.Tensor,
    ) -> torch.Tensor:
        features = nhwc_features(x, strain_rate, temperature, pressure)
        features = features.permute(0, 3, 1, 2)
        out = self.net(features).permute(0, 2, 3, 1)
        return self.output_activation(out)


def build_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    if name == "sig":
        return nn.Sigmoid()
    raise ValueError("activation must be one of: tanh, relu, sig")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(int(p.numel() * (2 if p.is_complex() else 1)) for p in model.parameters() if p.requires_grad)


def make_subset(dataset: Dataset, max_samples: int | None, seed: int) -> Dataset:
    if max_samples is None or max_samples <= 0 or max_samples >= len(dataset):
        return dataset

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:max_samples].tolist()
    return Subset(dataset, sorted(indices))


def batch_to_device(batch, device: torch.device):
    return [tensor.to(device, non_blocking=True) for tensor in batch]


@dataclass
class Metrics:
    mse: float
    rmse: float
    mae: float
    relative_l2: float
    circular_rmse_deg: float


def metric_accumulators() -> dict[str, float]:
    return {
        "sq_error": 0.0,
        "abs_error": 0.0,
        "target_sq": 0.0,
        "circular_sq_rad": 0.0,
        "num_values": 0.0,
    }


def update_metrics(acc: dict[str, float], pred: torch.Tensor, target: torch.Tensor) -> None:
    diff = pred - target
    circular_diff_rad = torch.atan2(torch.sin(math.pi * diff), torch.cos(math.pi * diff))

    acc["sq_error"] += float(torch.sum(diff * diff).detach().cpu())
    acc["abs_error"] += float(torch.sum(torch.abs(diff)).detach().cpu())
    acc["target_sq"] += float(torch.sum(target * target).detach().cpu())
    acc["circular_sq_rad"] += float(torch.sum(circular_diff_rad * circular_diff_rad).detach().cpu())
    acc["num_values"] += float(target.numel())


def finalize_metrics(acc: dict[str, float]) -> Metrics:
    if acc["num_values"] == 0:
        raise RuntimeError("No samples were evaluated")

    mse = acc["sq_error"] / acc["num_values"]
    circular_mse = acc["circular_sq_rad"] / acc["num_values"]
    relative_l2 = math.sqrt(acc["sq_error"] / max(acc["target_sq"], 1e-12))
    return Metrics(
        mse=mse,
        rmse=math.sqrt(mse),
        mae=acc["abs_error"] / acc["num_values"],
        relative_l2=relative_l2,
        circular_rmse_deg=math.sqrt(circular_mse) * 180.0 / math.pi,
    )


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Metrics:
    model.eval()
    acc = metric_accumulators()
    for batch in loader:
        x, target, strain_rate, temperature, pressure = batch_to_device(batch, device)
        pred = model(x, strain_rate, temperature, pressure)
        update_metrics(acc, pred, target)
    return finalize_metrics(acc)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
) -> list[dict[str, float]]:
    if trainable_parameter_count(model) == 0:
        return []

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_values = 0
        t0 = default_timer()

        for batch in train_loader:
            x, target, strain_rate, temperature, pressure = batch_to_device(batch, device)
            pred = model(x, strain_rate, temperature, pressure)
            loss = loss_fn(pred, target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.detach().cpu()) * target.numel()
            train_values += target.numel()

        valid_metrics = evaluate_model(model, valid_loader, device)
        history.append(
            {
                "epoch": float(epoch),
                "train_mse": train_loss_sum / max(train_values, 1),
                "valid_mse": valid_metrics.mse,
                "valid_rmse": valid_metrics.rmse,
                "epoch_seconds": default_timer() - t0,
            }
        )
        print(
            f"    epoch {epoch:03d}: train mse {history[-1]['train_mse']:.6g}, "
            f"valid rmse {valid_metrics.rmse:.6g}, {history[-1]['epoch_seconds']:.1f}s",
            flush=True,
        )

    return history


def build_model(model_name: str, step_known: int, args: argparse.Namespace) -> nn.Module:
    model_name = model_name.lower()
    if model_name == "persistence":
        return PersistenceSurrogate()
    if model_name == "linear_pointwise":
        return LinearPointwiseSurrogate(step_known)
    if model_name == "pointwise_mlp":
        return PointwiseMLPSurrogate(step_known, args.mlp_hidden, args.mlp_layers, args.activation)
    if model_name == "tiny_cnn":
        return TinyCNNSurrogate(step_known, args.cnn_channels, args.cnn_layers, args.cnn_kernel, args.activation)
    if model_name == "fno":
        return FNO2d(args.fno_modes, args.fno_modes, args.fno_width, step_known, args.activation, "mse")
    raise ValueError(
        f"Unknown model '{model_name}'. Choose from: "
        "persistence, linear_pointwise, pointwise_mlp, tiny_cnn, fno"
    )


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_history_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "epoch", "train_mse", "valid_mse", "valid_rmse", "epoch_seconds"])
        writer.writeheader()
        writer.writerows(rows)


def parse_models(models_text: str) -> list[str]:
    models = [item.strip().lower() for item in models_text.split(",") if item.strip()]
    if not models:
        raise ValueError("--models must include at least one model")
    return models


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)

    archive_dir = Path(args.archive_dir)
    if not archive_dir.is_absolute():
        archive_dir = (PROJECT_ROOT / archive_dir).resolve()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    preload = not args.no_preload
    train_dataset_full = Euler1ArchiveDataset(archive_dir / "euler1_train_data.h5", preload=preload)
    valid_dataset_full = Euler1ArchiveDataset(archive_dir / "euler1_valid_data.h5", preload=preload)
    test_dataset_full = Euler1ArchiveDataset(archive_dir / "euler1_test_data.h5", preload=preload)

    train_dataset = make_subset(train_dataset_full, args.max_train_samples, args.seed)
    valid_dataset = make_subset(valid_dataset_full, args.max_valid_samples, args.seed + 1)
    test_dataset = make_subset(test_dataset_full, args.max_test_samples, args.seed + 2)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(f"Using device: {device}", flush=True)
    print(f"Archive data: {archive_dir}", flush=True)
    print(
        f"Samples train/valid/test: {len(train_dataset)}/{len(valid_dataset)}/{len(test_dataset)} "
        f"(full split {len(train_dataset_full)}/{len(valid_dataset_full)}/{len(test_dataset_full)})",
        flush=True,
    )
    print(f"Grid size: {train_dataset_full.grid_size} x {train_dataset_full.grid_size}", flush=True)

    summary_rows: list[dict[str, object]] = []
    history_rows: list[dict[str, object]] = []
    json_payload: dict[str, object] = {
        "args": vars(args),
        "archive_dir": str(archive_dir),
        "output_dir": str(output_dir),
        "device": str(device),
        "preload": preload,
        "split_sizes": {
            "train": len(train_dataset),
            "valid": len(valid_dataset),
            "test": len(test_dataset),
            "train_full": len(train_dataset_full),
            "valid_full": len(valid_dataset_full),
            "test_full": len(test_dataset_full),
        },
        "results": [],
    }

    for model_name in parse_models(args.models):
        print(f"\nTraining/evaluating model: {model_name}", flush=True)
        model = build_model(model_name, train_dataset_full.step_known, args).to(device)
        params = trainable_parameter_count(model)
        t0 = default_timer()
        history = train_model(
            model,
            train_loader,
            valid_loader,
            args.epochs,
            args.learning_rate,
            args.weight_decay,
            device,
        )
        train_seconds = default_timer() - t0

        split_metrics = {
            "train": evaluate_model(model, train_loader, device),
            "valid": evaluate_model(model, valid_loader, device),
            "test": evaluate_model(model, test_loader, device),
        }

        for item in history:
            row = {"model": model_name}
            row.update(item)
            history_rows.append(row)

        result_entry = {
            "model": model_name,
            "parameters": params,
            "train_seconds": train_seconds,
            "metrics": {split: asdict(metrics) for split, metrics in split_metrics.items()},
        }
        json_payload["results"].append(result_entry)

        for split, metrics in split_metrics.items():
            summary_rows.append(
                {
                    "model": model_name,
                    "split": split,
                    "parameters": params,
                    "epochs": args.epochs if params > 0 else 0,
                    "train_seconds": f"{train_seconds:.3f}",
                    "mse": f"{metrics.mse:.8g}",
                    "rmse": f"{metrics.rmse:.8g}",
                    "mae": f"{metrics.mae:.8g}",
                    "relative_l2": f"{metrics.relative_l2:.8g}",
                    "circular_rmse_deg": f"{metrics.circular_rmse_deg:.8g}",
                }
            )

        if args.save_models and params > 0:
            torch.save(model.state_dict(), output_dir / f"{model_name}.pt")

        test_metrics = split_metrics["test"]
        print(
            f"  test rmse {test_metrics.rmse:.6g}, relative_l2 {test_metrics.relative_l2:.6g}, "
            f"circular_rmse_deg {test_metrics.circular_rmse_deg:.3f}, params {params}",
            flush=True,
        )

    write_summary_csv(output_dir / "summary.csv", summary_rows)
    write_history_csv(output_dir / "history.csv", history_rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2)

    test_rows = [row for row in summary_rows if row["split"] == "test"]
    test_rows.sort(key=lambda row: float(row["rmse"]))
    print("\nTest RMSE ranking:", flush=True)
    for rank, row in enumerate(test_rows, start=1):
        print(
            f"  {rank}. {row['model']}: rmse={row['rmse']}, "
            f"relative_l2={row['relative_l2']}, circular_rmse_deg={row['circular_rmse_deg']}",
            flush=True,
        )
    print(f"\nWrote results to: {output_dir}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare toy surrogate models on archived Euler-1 HDF5 data")
    parser.add_argument("--archive_dir", type=str, default="Archive (1)", help="Directory containing euler1_*_data.h5")
    parser.add_argument("--output_dir", type=str, default="results/euler1_surrogate_comparison")
    parser.add_argument(
        "--models",
        type=str,
        default="persistence,linear_pointwise,tiny_cnn,fno",
        help="Comma-separated list: persistence, linear_pointwise, pointwise_mlp, tiny_cnn, fno",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda:0, etc.")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--no_preload",
        action="store_true",
        help="Read HDF5 samples lazily instead of preloading the small archive split into memory",
    )
    parser.add_argument("--activation", type=str, default="tanh", choices=["tanh", "relu", "sig"])
    parser.add_argument("--max_train_samples", type=int, default=None, help="Optional toy subset for quick smoke runs")
    parser.add_argument("--max_valid_samples", type=int, default=None, help="Optional validation subset")
    parser.add_argument("--max_test_samples", type=int, default=None, help="Optional test subset")
    parser.add_argument("--mlp_hidden", type=int, default=64)
    parser.add_argument("--mlp_layers", type=int, default=3)
    parser.add_argument("--cnn_channels", type=int, default=16)
    parser.add_argument("--cnn_layers", type=int, default=3)
    parser.add_argument("--cnn_kernel", type=int, default=3)
    parser.add_argument("--fno_modes", type=int, default=12)
    parser.add_argument("--fno_width", type=int, default=16)
    parser.add_argument("--save_models", action="store_true", help="Save trained model weights into output_dir")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
