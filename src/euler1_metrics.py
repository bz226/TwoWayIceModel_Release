"""Metrics for the Euler-1 reduced-data surrogate benchmark."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from timeit import default_timer
from typing import Any

import numpy as np
import torch

from euler1_data import batch_to_model_tensors


@dataclass
class MetricResult:
    mse: float
    rmse: float
    mae: float
    relative_l2: float
    circular_rmse_deg: float
    circular_mae_deg: float
    inference_time_per_sample_sec: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def trainable_parameter_count(model: torch.nn.Module) -> int:
    count = 0
    for param in model.parameters():
        if param.requires_grad:
            count += int(param.numel() * (2 if param.is_complex() else 1))
    return count


class MetricAccumulator:
    def __init__(self) -> None:
        self.sq_error = 0.0
        self.abs_error = 0.0
        self.target_sq = 0.0
        self.circular_sq_rad = 0.0
        self.circular_abs_rad = 0.0
        self.num_values = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        diff = pred - target
        circular_diff = torch.atan2(torch.sin(math.pi * diff), torch.cos(math.pi * diff))
        self.sq_error += float(torch.sum(diff * diff).detach().cpu())
        self.abs_error += float(torch.sum(torch.abs(diff)).detach().cpu())
        self.target_sq += float(torch.sum(target * target).detach().cpu())
        self.circular_sq_rad += float(torch.sum(circular_diff * circular_diff).detach().cpu())
        self.circular_abs_rad += float(torch.sum(torch.abs(circular_diff)).detach().cpu())
        self.num_values += int(target.numel())

    def finalize(self, inference_time_per_sample_sec: float = 0.0) -> MetricResult:
        if self.num_values <= 0:
            raise RuntimeError("No values were accumulated for metric computation")
        mse = self.sq_error / self.num_values
        circular_mse = self.circular_sq_rad / self.num_values
        return MetricResult(
            mse=mse,
            rmse=math.sqrt(mse),
            mae=self.abs_error / self.num_values,
            relative_l2=math.sqrt(self.sq_error / max(self.target_sq, 1e-12)),
            circular_rmse_deg=math.sqrt(circular_mse) * 180.0 / math.pi,
            circular_mae_deg=(self.circular_abs_rad / self.num_values) * 180.0 / math.pi,
            inference_time_per_sample_sec=float(inference_time_per_sample_sec),
        )


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    delta_min: float,
    delta_max: float,
) -> MetricResult:
    model.eval()
    accumulator = MetricAccumulator()
    sample_count = 0
    start = default_timer()
    for batch in loader:
        features, target = batch_to_model_tensors(batch, device, delta_min, delta_max)
        pred = model(features)
        accumulator.update(pred, target)
        sample_count += int(target.shape[0])
    elapsed = default_timer() - start
    return accumulator.finalize(elapsed / max(sample_count, 1))


def rmse_improvement_percent(persistence_rmse: float, model_rmse: float) -> float:
    if persistence_rmse == 0:
        return 0.0
    return 100.0 * (persistence_rmse - model_rmse) / persistence_rmse


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    array = np.asarray(values, dtype=np.float64)
    return float(np.mean(array)), float(np.std(array, ddof=1 if len(array) > 1 else 0))


def flatten_metrics(prefix: str, metrics: MetricResult) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in metrics.to_dict().items()}
