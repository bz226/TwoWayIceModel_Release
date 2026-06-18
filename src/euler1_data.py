"""Data utilities for the Euler-1 reduced-data surrogate benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


FIELD_NAMES = ("euler1_known", "euler1_predict", "strain_rate", "temperature", "pressure")
METADATA_NAMES = (
    "known_step_1based",
    "predict_step_1based",
    "source_file",
    "source_global_index",
)


def _decode_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _torch_field(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(array)).float()


class Euler1H5Dataset(Dataset):
    """HDF5 dataset for cleaned Euler-1 benchmark splits.

    Items return a dictionary with raw fields and metadata. The arrays remain in
    HDF5 layout ``(H, W, 1)``; use :func:`batch_to_model_tensors` to build the
    ``(N, C, H, W)`` model input.
    """

    def __init__(self, h5_path: str | Path, preload: bool = False, include_delta_step: bool = True):
        self.h5_path = Path(h5_path)
        self.preload = bool(preload)
        self.include_delta_step = bool(include_delta_step)
        self._cache: dict[str, np.ndarray] | None = None

        with h5py.File(self.h5_path, "r") as f:
            required = list(FIELD_NAMES) + list(METADATA_NAMES)
            missing = [name for name in required if name not in f]
            if missing:
                raise KeyError(f"{self.h5_path} is missing required datasets: {missing}")

            self.n_samples = int(f["euler1_known"].shape[0])
            self.grid_shape = tuple(int(v) for v in f["euler1_known"].shape[1:3])
            for name in FIELD_NAMES[1:] + METADATA_NAMES:
                if int(f[name].shape[0]) != self.n_samples:
                    raise ValueError(
                        f"Inconsistent sample count in {self.h5_path}: "
                        f"euler1_known has {self.n_samples}, {name} has {f[name].shape[0]}"
                    )

            self.attrs = {key: _decode_value(value) for key, value in f.attrs.items()}
            if self.preload:
                self._cache = {name: f[name][...] for name in required if name in f}

    def __len__(self) -> int:
        return self.n_samples

    def _get_value(self, name: str, idx: int) -> Any:
        if self._cache is not None:
            return self._cache[name][idx]
        with h5py.File(self.h5_path, "r") as f:
            return f[name][idx]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        known_step = int(_decode_value(self._get_value("known_step_1based", idx)))
        predict_step = int(_decode_value(self._get_value("predict_step_1based", idx)))
        item: dict[str, Any] = {
            "x": _torch_field(self._get_value("euler1_known", idx)),
            "y": _torch_field(self._get_value("euler1_predict", idx)),
            "strain_rate": _torch_field(self._get_value("strain_rate", idx)),
            "temperature": _torch_field(self._get_value("temperature", idx)),
            "pressure": _torch_field(self._get_value("pressure", idx)),
            "known_step": torch.tensor(known_step, dtype=torch.long),
            "predict_step": torch.tensor(predict_step, dtype=torch.long),
            "source_file": str(_decode_value(self._get_value("source_file", idx))),
            "source_global_index": torch.tensor(
                int(_decode_value(self._get_value("source_global_index", idx))),
                dtype=torch.long,
            ),
        }
        if self.include_delta_step:
            item["delta_step"] = torch.tensor(predict_step - known_step, dtype=torch.float32)
        return item


def delta_bounds_from_h5(h5_path: str | Path) -> tuple[float, float]:
    with h5py.File(h5_path, "r") as f:
        delta = f["predict_step_1based"][...] - f["known_step_1based"][...]
    return float(np.min(delta)), float(np.max(delta))


def normalize_delta_step(delta_step: torch.Tensor, delta_min: float, delta_max: float) -> torch.Tensor:
    delta_step = delta_step.float()
    if float(delta_max) == float(delta_min):
        return torch.zeros_like(delta_step)
    return 2.0 * (delta_step - float(delta_min)) / (float(delta_max) - float(delta_min)) - 1.0


def coordinate_channels(batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
    y = torch.linspace(-1.0, 1.0, height, device=device).view(1, 1, height, 1).expand(batch_size, 1, height, width)
    x = torch.linspace(-1.0, 1.0, width, device=device).view(1, 1, 1, width).expand(batch_size, 1, height, width)
    return torch.cat((x, y), dim=1)


def nhwc_to_nchw(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 4:
        raise ValueError(f"Expected a 4D tensor in NHWC layout, got shape {tuple(tensor.shape)}")
    return tensor.permute(0, 3, 1, 2).contiguous()


def batch_to_model_tensors(
    batch: dict[str, Any],
    device: torch.device,
    delta_min: float,
    delta_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a DataLoader batch to model input and target tensors.

    The model input channels are:
    Euler-1 known, strain rate, temperature, pressure, x grid, y grid,
    normalized delta step.
    """

    x = nhwc_to_nchw(batch["x"].to(device, non_blocking=True))
    y = nhwc_to_nchw(batch["y"].to(device, non_blocking=True))
    strain_rate = nhwc_to_nchw(batch["strain_rate"].to(device, non_blocking=True))
    temperature = nhwc_to_nchw(batch["temperature"].to(device, non_blocking=True))
    pressure = nhwc_to_nchw(batch["pressure"].to(device, non_blocking=True))

    batch_size, _, height, width = x.shape
    grid = coordinate_channels(batch_size, height, width, device)

    delta = batch["delta_step"].to(device, non_blocking=True)
    delta_norm = normalize_delta_step(delta, delta_min, delta_max).view(batch_size, 1, 1, 1)
    delta_channel = delta_norm.expand(batch_size, 1, height, width)

    features = torch.cat((x, strain_rate, temperature, pressure, grid, delta_channel), dim=1)
    return features, y
