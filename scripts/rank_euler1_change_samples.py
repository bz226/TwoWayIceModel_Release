#!/usr/bin/env python3
"""Rank Euler-1 test samples by input-to-ground-truth change."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def per_sample_metrics(known: np.ndarray, target: np.ndarray) -> list[dict[str, float | int]]:
    diff = target - known
    circular_diff = np.arctan2(np.sin(np.pi * diff), np.cos(np.pi * diff))

    rmse = np.sqrt(np.mean(diff * diff, axis=(1, 2, 3)))
    mae = np.mean(np.abs(diff), axis=(1, 2, 3))
    max_abs = np.max(np.abs(diff), axis=(1, 2, 3))
    circular_rmse_deg = np.sqrt(np.mean(circular_diff * circular_diff, axis=(1, 2, 3))) * 180.0 / np.pi
    circular_mae_deg = np.mean(np.abs(circular_diff), axis=(1, 2, 3)) * 180.0 / np.pi
    changed_fraction_gt10deg = np.mean(np.abs(circular_diff) > (10.0 * np.pi / 180.0), axis=(1, 2, 3))

    rows: list[dict[str, float | int]] = []
    for sample_index in range(int(known.shape[0])):
        rows.append(
            {
                "sample_index": sample_index,
                "normalized_rmse": float(rmse[sample_index]),
                "normalized_mae": float(mae[sample_index]),
                "normalized_max_abs": float(max_abs[sample_index]),
                "circular_rmse_deg": float(circular_rmse_deg[sample_index]),
                "circular_mae_deg": float(circular_mae_deg[sample_index]),
                "changed_fraction_gt10deg": float(changed_fraction_gt10deg[sample_index]),
            }
        )
    return sorted(rows, key=lambda row: float(row["circular_rmse_deg"]), reverse=True)


def write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank Euler-1 samples by before-to-after change")
    parser.add_argument("--predictions_npz", default="results/euler1_comparison/test_predictions_full.npz")
    parser.add_argument("--output_csv", default="results/euler1_comparison/input_target_change_ranking.csv")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    with np.load(args.predictions_npz) as data:
        known = data["known"]
        target = data["target"]

    rows = per_sample_metrics(known, target)
    write_csv(Path(args.output_csv), rows)

    print(f"wrote {args.output_csv}")
    print("Top samples by circular RMSE:")
    for rank, row in enumerate(rows[: args.top], 1):
        print(
            f"{rank:2d}. sample_index={int(row['sample_index']):2d} "
            f"circular_rmse_deg={float(row['circular_rmse_deg']):.3f} "
            f"normalized_rmse={float(row['normalized_rmse']):.6f} "
            f"changed_fraction_gt10deg={float(row['changed_fraction_gt10deg']):.3f}"
        )


if __name__ == "__main__":
    main()
