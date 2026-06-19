#!/usr/bin/env python3
"""Export full-test Euler-1 predictions from saved benchmark checkpoints.

This script is for post-processing an already completed comparison run. It
loads the cleaned test split and saved model checkpoints, then writes one NPZ
containing every test sample in dataset order.

Example:
    python3 scripts/export_euler1_test_predictions.py \
        --output results/euler1_comparison/test_predictions_full.npz
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from euler1_data import Euler1H5Dataset, batch_to_model_tensors, delta_bounds_from_h5  # noqa: E402
from euler1_models import build_model  # noqa: E402


def parse_models(text: str) -> list[str]:
    models = [item.strip().lower() for item in text.split(",") if item.strip()]
    if not models:
        raise ValueError("--models must contain at least one model")
    return models


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    delta_min: float,
    delta_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    known_parts = []
    target_parts = []
    pred_parts = []
    for batch in loader:
        features, target = batch_to_model_tensors(batch, device, delta_min, delta_max)
        pred = model(features)
        known_parts.append(features[:, 0:1].detach().cpu().numpy())
        target_parts.append(target.detach().cpu().numpy())
        pred_parts.append(pred.detach().cpu().numpy())
    return (
        np.concatenate(known_parts, axis=0),
        np.concatenate(target_parts, axis=0),
        np.concatenate(pred_parts, axis=0),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export full Euler-1 test predictions")
    parser.add_argument("--train_h5", default="data/euler1_clean/euler1_train_clean.h5")
    parser.add_argument("--test_h5", default="data/euler1_clean/euler1_test_clean.h5")
    parser.add_argument("--output", default="results/euler1_comparison/test_predictions_full.npz")
    parser.add_argument("--checkpoint_dir", default="results/euler1_comparison")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--models", default="persistence,resnet_cnn,fno")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cnn_width", type=int, default=32)
    parser.add_argument("--cnn_blocks", type=int, default=4)
    parser.add_argument("--fno_modes", type=int, default=12)
    parser.add_argument("--fno_width", type=int, default=16)
    parser.add_argument("--preload", action="store_true", default=True)
    args = parser.parse_args()

    device = resolve_device(args.device)
    model_names = parse_models(args.models)
    delta_min, delta_max = delta_bounds_from_h5(args.train_h5)
    dataset = Euler1H5Dataset(args.test_h5, preload=args.preload)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    payload: dict[str, np.ndarray | np.generic] = {}
    known_reference: np.ndarray | None = None
    target_reference: np.ndarray | None = None

    for model_name in model_names:
        model = build_model(
            model_name,
            input_channels=7,
            cnn_width=args.cnn_width,
            cnn_blocks=args.cnn_blocks,
            fno_modes=args.fno_modes,
            fno_width=args.fno_width,
        ).to(device)
        if model_name != "persistence":
            checkpoint_path = checkpoint_dir / f"best_{model_name}_seed{args.seed}.pt"
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Missing checkpoint for {model_name}: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state_dict)

        known, target, pred = collect_predictions(model, loader, device, delta_min, delta_max)
        if known_reference is None:
            known_reference = known
            target_reference = target
        payload[f"{model_name}_prediction"] = pred
        print(f"{model_name}: {pred.shape}")

    if known_reference is None or target_reference is None:
        raise RuntimeError("No predictions were exported")

    payload["known"] = known_reference
    payload["target"] = target_reference
    payload["seed"] = np.asarray(args.seed, dtype=np.int64)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **payload)
    print(output)


if __name__ == "__main__":
    main()
