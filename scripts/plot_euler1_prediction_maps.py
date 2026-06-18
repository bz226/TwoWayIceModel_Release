#!/usr/bin/env python
"""Plot reusable Euler-1 before / prediction / ground-truth comparison maps.

Example:
    python scripts/plot_euler1_prediction_maps.py \
        --predictions_npz results/euler1_comparison/test_predictions_sampled.npz \
        --output results/euler1_comparison/test_prediction_maps.png
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np


DEFAULT_MODEL_ORDER = ("persistence", "resnet_cnn", "fno")
DEFAULT_DISPLAY_NAMES = {
    "persistence": "Persistence",
    "resnet_cnn": "ResNet CNN",
    "fno": "FNO",
}


def parse_models(text: str) -> list[str]:
    return [item.strip().lower() for item in text.split(",") if item.strip()]


def sample_field(array: np.ndarray, sample_index: int) -> np.ndarray:
    """Return a 2D field from common sampled prediction array layouts."""

    values = np.asarray(array)
    if values.ndim == 2:
        return values
    if values.ndim == 3:
        return values[sample_index]
    if values.ndim == 4:
        if values.shape[1] == 1:
            return values[sample_index, 0]
        if values.shape[-1] == 1:
            return values[sample_index, :, :, 0]
    raise ValueError(
        "Expected array layout (H, W), (N, H, W), (N, 1, H, W), or (N, H, W, 1); "
        f"got {values.shape}"
    )


def load_prediction_npz(
    predictions_npz: str | Path,
    models: Sequence[str] = DEFAULT_MODEL_ORDER,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Load known input, target, and model predictions from a sampled NPZ file."""

    predictions_npz = Path(predictions_npz)
    with np.load(predictions_npz) as data:
        missing_base = [key for key in ("known", "target") if key not in data]
        if missing_base:
            raise KeyError(f"{predictions_npz} is missing required arrays: {missing_base}")

        known = data["known"]
        target = data["target"]
        predictions: dict[str, np.ndarray] = {}
        missing_predictions = []
        for model in models:
            key = f"{model}_prediction"
            if key in data:
                predictions[model] = data[key]
            else:
                missing_predictions.append(key)

    if missing_predictions:
        raise KeyError(f"{predictions_npz} is missing prediction arrays: {missing_predictions}")
    if not predictions:
        raise ValueError("No model predictions were loaded")
    return known, target, predictions


def save_prediction_map_plot(
    predictions: dict[str, np.ndarray],
    known: np.ndarray,
    target: np.ndarray,
    out_path: str | Path,
    sample_index: int = 0,
    model_order: Sequence[str] = DEFAULT_MODEL_ORDER,
    display_names: dict[str, str] | None = None,
    title: str | None = "Euler-1 reduced-data benchmark sample maps",
    cmap: str = "coolwarm",
    vmin: float = -1.0,
    vmax: float = 1.0,
    dpi: int = 200,
    show_colorbar: bool = True,
) -> None:
    """Save a 3-column qualitative figure: before input, prediction, ground truth."""

    if not predictions:
        raise ValueError("predictions must contain at least one model")

    display_names = dict(DEFAULT_DISPLAY_NAMES if display_names is None else display_names)
    ordered_models = [model for model in model_order if model in predictions]
    ordered_models.extend(model for model in predictions if model not in set(ordered_models))

    fig, axes = plt.subplots(
        len(ordered_models),
        3,
        figsize=(8.8, 2.8 * len(ordered_models)),
        squeeze=False,
        constrained_layout=True,
    )
    known_map = sample_field(known, sample_index)
    target_map = sample_field(target, sample_index)
    column_titles = ("Before (input)", "Prediction", "Ground truth")

    image_handle = None
    for row_idx, model in enumerate(ordered_models):
        pred_map = sample_field(predictions[model], sample_index)
        for col_idx, image in enumerate((known_map, pred_map, target_map)):
            image_handle = axes[row_idx, col_idx].imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(column_titles[col_idx])
            axes[row_idx, col_idx].axis("off")

        axes[row_idx, 0].text(
            -0.10,
            0.5,
            display_names.get(model, model),
            transform=axes[row_idx, 0].transAxes,
            fontsize=12,
            fontweight="bold",
            rotation=90,
            va="center",
            ha="center",
        )

    if show_colorbar and image_handle is not None:
        fig.colorbar(image_handle, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    if title:
        fig.suptitle(title, y=1.03)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Euler-1 sampled prediction maps")
    parser.add_argument("--predictions_npz", default="results/euler1_comparison/test_predictions_sampled.npz")
    parser.add_argument("--output", default="results/euler1_comparison/test_prediction_maps.png")
    parser.add_argument("--models", default="persistence,resnet_cnn,fno")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--title", default="Euler-1 reduced-data benchmark sample maps")
    parser.add_argument("--cmap", default="coolwarm")
    parser.add_argument("--vmin", type=float, default=-1.0)
    parser.add_argument("--vmax", type=float, default=1.0)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--no_colorbar", action="store_true")
    args = parser.parse_args()

    model_order = parse_models(args.models)
    known, target, predictions = load_prediction_npz(args.predictions_npz, model_order)
    save_prediction_map_plot(
        predictions=predictions,
        known=known,
        target=target,
        out_path=args.output,
        sample_index=args.sample_index,
        model_order=model_order,
        title=args.title if args.title else None,
        cmap=args.cmap,
        vmin=args.vmin,
        vmax=args.vmax,
        dpi=args.dpi,
        show_colorbar=not args.no_colorbar,
    )
    print(args.output)


if __name__ == "__main__":
    main()
