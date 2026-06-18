#!/usr/bin/env python
"""Compare persistence, ResNet CNN, and FNO on cleaned Euler-1 splits."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from copy import deepcopy
from pathlib import Path
from timeit import default_timer
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from euler1_data import Euler1H5Dataset, batch_to_model_tensors, delta_bounds_from_h5  # noqa: E402
from euler1_metrics import (  # noqa: E402
    evaluate_model,
    flatten_metrics,
    mean_std,
    rmse_improvement_percent,
    trainable_parameter_count,
)
from euler1_models import build_model  # noqa: E402
from plot_euler1_prediction_maps import save_prediction_map_plot  # noqa: E402


def parse_models(text: str) -> list[str]:
    models = [item.strip().lower() for item in text.split(",") if item.strip()]
    allowed = {"persistence", "resnet_cnn", "fno"}
    unknown = sorted(set(models) - allowed)
    if unknown:
        raise ValueError(f"Unknown models {unknown}. Choose from {sorted(allowed)}")
    if not models:
        raise ValueError("--models must contain at least one model")
    return models


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_loader(dataset: Euler1H5Dataset, batch_size: int, shuffle: bool, seed: int, num_workers: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator if shuffle else None,
        pin_memory=torch.cuda.is_available(),
    )


def train_one_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    device: torch.device,
    delta_min: float,
    delta_max: float,
    epochs: int,
    learning_rate: float,
    patience: int,
    grad_clip: float,
    selection_metric: str,
) -> tuple[nn.Module, list[dict[str, Any]], float, float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    best_state: dict[str, torch.Tensor] | None = None
    best_metric = float("inf")
    best_epoch = 0
    epochs_since_improvement = 0
    history: list[dict[str, Any]] = []
    start = default_timer()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_values = 0

        for batch in train_loader:
            features, target = batch_to_model_tensors(batch, device, delta_min, delta_max)
            pred = model(features)
            loss = loss_fn(pred, target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            train_loss_sum += float(loss.detach().cpu()) * int(target.numel())
            train_values += int(target.numel())

        valid_metrics = evaluate_model(model, valid_loader, device, delta_min, delta_max)
        metric_value = (
            valid_metrics.rmse if selection_metric == "valid_rmse" else valid_metrics.circular_rmse_deg
        )
        improved = metric_value < best_metric
        if improved:
            best_metric = metric_value
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        row = {
            "epoch": epoch,
            "train_mse": train_loss_sum / max(train_values, 1),
            "valid_rmse": valid_metrics.rmse,
            "valid_circular_rmse_deg": valid_metrics.circular_rmse_deg,
            "selection_metric": metric_value,
            "is_best": int(improved),
        }
        history.append(row)
        print(
            f"    epoch {epoch:03d}: train_mse={row['train_mse']:.6g}, "
            f"valid_rmse={valid_metrics.rmse:.6g}, "
            f"valid_circular_rmse={valid_metrics.circular_rmse_deg:.3f}, "
            f"best_epoch={best_epoch}",
            flush=True,
        )

        if epochs_since_improvement >= patience:
            print(f"    early stopping at epoch {epoch} after {patience} stale epochs", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, default_timer() - start, best_metric


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    delta_min: float,
    delta_max: float,
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    known_parts = []
    target_parts = []
    pred_parts = []
    seen = 0
    for batch in loader:
        features, target = batch_to_model_tensors(batch, device, delta_min, delta_max)
        pred = model(features)
        known = features[:, 0:1]
        known_parts.append(known.detach().cpu().numpy())
        target_parts.append(target.detach().cpu().numpy())
        pred_parts.append(pred.detach().cpu().numpy())
        seen += int(target.shape[0])
        if max_samples is not None and seen >= max_samples:
            break
    known_array = np.concatenate(known_parts, axis=0)
    target_array = np.concatenate(target_parts, axis=0)
    pred_array = np.concatenate(pred_parts, axis=0)
    if max_samples is not None:
        known_array = known_array[:max_samples]
        target_array = target_array[:max_samples]
        pred_array = pred_array[:max_samples]
    return known_array, target_array, pred_array


def save_loss_plot(history_rows: list[dict[str, Any]], out_path: Path) -> None:
    if not history_rows:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    keys = sorted({(row["seed"], row["model"]) for row in history_rows})
    for seed, model in keys:
        rows = [row for row in history_rows if row["seed"] == seed and row["model"] == model]
        epochs = [row["epoch"] for row in rows]
        valid = [row["valid_rmse"] for row in rows]
        ax.plot(epochs, valid, marker="o", linewidth=1.2, label=f"{model} seed {seed}")
    ax.set_title("Euler-1 reduced-data benchmark validation RMSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation RMSE")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_bar_plot(rows: list[dict[str, Any]], metric: str, ylabel: str, out_path: Path) -> None:
    models = [row["model"] for row in rows]
    means = [float(row[f"test_{metric}_mean"]) for row in rows]
    stds = [float(row[f"test_{metric}_std"]) for row in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(models, means, yerr=stds, capsize=4, color=["#777777", "#4C78A8", "#F58518"][: len(models)])
    ax.set_title("Euler-1 reduced-data benchmark")
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_scatter_plot(predictions: dict[str, np.ndarray], target: np.ndarray, out_path: Path) -> None:
    if not predictions:
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    target_flat = target.reshape(-1)
    stride = max(1, target_flat.size // 20000)
    target_sample = target_flat[::stride]
    for model, pred in predictions.items():
        ax.scatter(target_sample, pred.reshape(-1)[::stride], s=2, alpha=0.25, label=model)
    ax.plot([-1, 1], [-1, 1], color="black", linewidth=1)
    ax.set_title("Euler-1 reduced-data benchmark test predictions")
    ax.set_xlabel("True normalized Euler-1")
    ax.set_ylabel("Predicted normalized Euler-1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def make_mean_std_rows(summary_by_seed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    test_rows = [row for row in summary_by_seed if row["split"] == "test"]
    preferred_order = ["persistence", "resnet_cnn", "fno"]
    models = [model for model in preferred_order if any(row["model"] == model for row in test_rows)]
    models.extend(sorted({row["model"] for row in test_rows} - set(models)))
    metric_names = [
        "mse",
        "rmse",
        "mae",
        "relative_l2",
        "circular_rmse_deg",
        "circular_mae_deg",
        "rmse_improvement_percent",
        "inference_time_per_sample_sec",
    ]
    rows = []
    for model in models:
        model_rows = [row for row in test_rows if row["model"] == model]
        row: dict[str, Any] = {
            "label": "Euler-1 reduced-data benchmark",
            "model": model,
            "seeds": len(model_rows),
            "training_samples": model_rows[0]["training_samples"],
            "parameters": model_rows[0]["parameters"],
        }
        for metric in metric_names:
            values = [float(item[metric]) for item in model_rows]
            mean, std = mean_std(values)
            row[f"test_{metric}_mean"] = mean
            row[f"test_{metric}_std"] = std
            row[f"test_{metric}_mean_std"] = f"{mean:.6g} +/- {std:.6g}"
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Euler-1 reduced-data surrogate comparison")
    parser.add_argument("--train_h5", required=True)
    parser.add_argument("--valid_h5", required=True)
    parser.add_argument("--test_h5", required=True)
    parser.add_argument("--models", default="persistence,resnet_cnn,fno")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--seeds", nargs="+", type=int, default=[12345, 23456, 34567])
    parser.add_argument("--output_dir", default="results/euler1_comparison")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--grad_clip", type=float, default=10.0)
    parser.add_argument("--selection_metric", choices=["valid_rmse", "valid_circular_rmse"], default="valid_rmse")
    parser.add_argument("--preload", action="store_true")
    parser.add_argument("--cnn_width", type=int, default=32)
    parser.add_argument("--cnn_blocks", type=int, default=4)
    parser.add_argument("--fno_modes", type=int, default=12)
    parser.add_argument("--fno_width", type=int, default=16)
    args = parser.parse_args()

    model_names = parse_models(args.models)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    delta_min, delta_max = delta_bounds_from_h5(args.train_h5)

    train_dataset = Euler1H5Dataset(args.train_h5, preload=args.preload)
    valid_dataset = Euler1H5Dataset(args.valid_h5, preload=args.preload)
    test_dataset = Euler1H5Dataset(args.test_h5, preload=args.preload)

    print("Euler-1 reduced-data benchmark")
    print(f"  device: {device}")
    print(f"  train/valid/test samples: {len(train_dataset)}/{len(valid_dataset)}/{len(test_dataset)}")
    print(f"  delta_step train bounds: {delta_min} to {delta_max}")

    summary_by_seed: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    sampled_predictions: dict[str, np.ndarray] = {}
    sampled_target: np.ndarray | None = None
    sampled_known: np.ndarray | None = None
    scatter_predictions: dict[str, np.ndarray] = {}
    scatter_target: np.ndarray | None = None
    best_global: dict[str, tuple[float, dict[str, torch.Tensor]]] = {}

    for seed in args.seeds:
        set_seed(seed)
        train_loader = make_loader(train_dataset, args.batch_size, True, seed, args.num_workers)
        valid_loader = make_loader(valid_dataset, args.batch_size, False, seed, args.num_workers)
        test_loader = make_loader(test_dataset, args.batch_size, False, seed, args.num_workers)

        seed_models: dict[str, nn.Module] = {}
        seed_metrics: dict[str, dict[str, Any]] = {}

        print(f"\nSeed {seed}", flush=True)
        for model_name in model_names:
            print(f"  Model: {model_name}", flush=True)
            model = build_model(
                model_name,
                input_channels=7,
                cnn_width=args.cnn_width,
                cnn_blocks=args.cnn_blocks,
                fno_modes=args.fno_modes,
                fno_width=args.fno_width,
            ).to(device)
            parameters = trainable_parameter_count(model)
            train_seconds = 0.0
            best_metric = 0.0

            if parameters > 0:
                model, model_history, train_seconds, best_metric = train_one_model(
                    model,
                    train_loader,
                    valid_loader,
                    device,
                    delta_min,
                    delta_max,
                    args.epochs,
                    args.learning_rate,
                    args.patience,
                    args.grad_clip,
                    args.selection_metric,
                )
                for row in model_history:
                    history_row = {"seed": seed, "model": model_name}
                    history_row.update(row)
                    history_rows.append(history_row)
                checkpoint_path = output_dir / f"best_{model_name}_seed{seed}.pt"
                torch.save(model.state_dict(), checkpoint_path)
                if model_name not in best_global or best_metric < best_global[model_name][0]:
                    best_global[model_name] = (
                        best_metric,
                        {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                    )

            split_loaders = {"train": train_loader, "valid": valid_loader, "test": test_loader}
            split_metrics = {
                split: evaluate_model(model, loader, device, delta_min, delta_max)
                for split, loader in split_loaders.items()
            }
            seed_models[model_name] = model
            seed_metrics[model_name] = {split: metrics.to_dict() for split, metrics in split_metrics.items()}

            for split, metrics in split_metrics.items():
                row = {
                    "label": "Euler-1 reduced-data benchmark",
                    "seed": seed,
                    "model": model_name,
                    "split": split,
                    "training_samples": 0 if parameters == 0 else len(train_dataset),
                    "parameters": parameters,
                    "train_seconds": train_seconds,
                }
                row.update(metrics.to_dict())
                summary_by_seed.append(row)

        persistence_rmse = seed_metrics["persistence"]["test"]["rmse"] if "persistence" in seed_metrics else None
        if persistence_rmse is not None:
            for row in summary_by_seed:
                if row["seed"] == seed and row["split"] == "test":
                    row["rmse_improvement_percent"] = rmse_improvement_percent(
                        float(persistence_rmse),
                        float(row["rmse"]),
                    )
                elif row["seed"] == seed:
                    row["rmse_improvement_percent"] = ""

        if seed == args.seeds[0]:
            for model_name, model in seed_models.items():
                known, target, pred = collect_predictions(
                    model,
                    test_loader,
                    device,
                    delta_min,
                    delta_max,
                    max_samples=min(3, len(test_dataset)),
                )
                sampled_predictions[model_name] = pred
                sampled_known = known if sampled_known is None else sampled_known
                sampled_target = target if sampled_target is None else sampled_target

                _, full_target, full_pred = collect_predictions(model, test_loader, device, delta_min, delta_max)
                scatter_predictions[model_name] = full_pred
                scatter_target = full_target if scatter_target is None else scatter_target

    for model_name in ("resnet_cnn", "fno"):
        if model_name in best_global:
            torch.save(best_global[model_name][1], output_dir / f"best_{model_name}.pt")

    mean_std_rows = make_mean_std_rows(summary_by_seed)
    write_csv(output_dir / "summary_by_seed.csv", summary_by_seed)
    write_csv(output_dir / "summary_mean_std.csv", mean_std_rows)
    write_csv(output_dir / "history.csv", history_rows)

    metadata = {
        "label": "Euler-1 reduced-data benchmark",
        "args": vars(args),
        "delta_step_min_train": delta_min,
        "delta_step_max_train": delta_max,
        "train_samples": len(train_dataset),
        "valid_samples": len(valid_dataset),
        "test_samples": len(test_dataset),
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if sampled_target is not None and sampled_known is not None:
        npz_payload = {
            "known": sampled_known,
            "target": sampled_target,
        }
        for model_name, pred in sampled_predictions.items():
            npz_payload[f"{model_name}_prediction"] = pred
        np.savez(output_dir / "test_predictions_sampled.npz", **npz_payload)
        save_prediction_map_plot(
            sampled_predictions,
            sampled_known,
            sampled_target,
            output_dir / "test_prediction_maps.png",
        )

    if scatter_target is not None:
        save_scatter_plot(scatter_predictions, scatter_target, output_dir / "test_pred_vs_true.png")
    save_loss_plot(history_rows, output_dir / "loss_curves.png")
    save_bar_plot(mean_std_rows, "rmse", "Test RMSE", output_dir / "test_rmse_bar.png")
    save_bar_plot(mean_std_rows, "circular_rmse_deg", "Circular RMSE (degrees)", output_dir / "circular_rmse_bar.png")

    print("\nEuler-1 reduced-data benchmark complete")
    print(f"  summary_by_seed: {output_dir / 'summary_by_seed.csv'}")
    print(f"  summary_mean_std: {output_dir / 'summary_mean_std.csv'}")
    print("  Reminder: this is a reduced Euler-1 comparison, not a full microscale surrogate benchmark.")


if __name__ == "__main__":
    main()
