#!/usr/bin/env python
"""Prepare source-file-disjoint clean Euler-1 benchmark splits."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from inspect_euler1_h5 import decode_string_array, inspect_files, json_safe, row_quality_checks


def read_dataset(dset: h5py.Dataset) -> np.ndarray:
    data = dset[...]
    if dset.dtype.kind in "OSU":
        return np.asarray(decode_string_array(data), dtype=object)
    return data


def load_and_concatenate(paths: list[str | Path]) -> dict[str, np.ndarray]:
    paths = [Path(path) for path in paths]
    with h5py.File(paths[0], "r") as f:
        dataset_names = sorted(f.keys())

    pieces: dict[str, list[np.ndarray]] = {name: [] for name in dataset_names}
    for path in paths:
        with h5py.File(path, "r") as f:
            missing = [name for name in dataset_names if name not in f]
            if missing:
                raise KeyError(f"{path} is missing datasets present in {paths[0]}: {missing}")
            for name in dataset_names:
                pieces[name].append(read_dataset(f[name]))

    return {name: np.concatenate(parts, axis=0) for name, parts in pieces.items()}


def filter_positive_offsets(data: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], int]:
    delta = data["predict_step_1based"] - data["known_step_1based"]
    keep = delta > 0
    removed = int(np.sum(~keep))
    filtered = {name: values[keep] for name, values in data.items()}
    return filtered, removed


def assert_no_contradictions(data: dict[str, np.ndarray]) -> None:
    checks = row_quality_checks(
        [str(value) for value in data["source_file"]],
        data["source_global_index"],
        data["known_step_1based"],
        data["predict_step_1based"],
        data["euler1_predict"],
    )
    if int(checks["contradictory_input_group_count"]) > 0:
        examples = checks["contradictory_input_examples"]
        raise RuntimeError(
            "Contradictory Euler-1 input-target pairs remain after removing "
            f"nonpositive offsets. Examples: {examples}"
        )


def source_groups(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, source_file in enumerate(data["source_file"]):
        groups[str(source_file)].append(idx)
    return {source_file: np.asarray(indices, dtype=np.int64) for source_file, indices in groups.items()}


def stratified_source_split(
    data: dict[str, np.ndarray],
    seed: int,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    num_bins: int = 5,
) -> dict[str, set[str]]:
    if not np.isclose(train_ratio + valid_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + valid_ratio + test_ratio must sum to 1")

    groups = source_groups(data)
    source_info = []
    for source_file, indices in groups.items():
        source_s = np.asarray(data["source_S_value"][indices], dtype=np.float64)
        source_info.append(
            {
                "source_file": source_file,
                "logS": float(np.log10(np.median(source_s))),
                "sample_count": int(len(indices)),
            }
        )

    rng = np.random.default_rng(seed)
    source_info.sort(key=lambda item: item["logS"])
    num_bins = max(1, min(int(num_bins), len(source_info)))
    binned: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for rank, item in enumerate(source_info):
        bin_id = min(num_bins - 1, rank * num_bins // len(source_info))
        binned[bin_id].append(item)

    splits = {"train": set(), "valid": set(), "test": set()}
    for bin_items in binned.values():
        shuffled = list(bin_items)
        rng.shuffle(shuffled)
        n_items = len(shuffled)
        n_train = int(round(train_ratio * n_items))
        n_valid = int(round(valid_ratio * n_items))
        if n_items >= 3:
            n_train = min(max(n_train, 1), n_items - 2)
            n_valid = min(max(n_valid, 1), n_items - n_train - 1)
        n_test = n_items - n_train - n_valid
        if n_test < 0:
            n_valid = max(0, n_valid + n_test)
            n_test = 0

        for item in shuffled[:n_train]:
            splits["train"].add(item["source_file"])
        for item in shuffled[n_train : n_train + n_valid]:
            splits["valid"].add(item["source_file"])
        for item in shuffled[n_train + n_valid : n_train + n_valid + n_test]:
            splits["test"].add(item["source_file"])

    return splits


def rows_for_sources(data: dict[str, np.ndarray], selected_sources: set[str]) -> np.ndarray:
    mask = np.asarray([str(source_file) in selected_sources for source_file in data["source_file"]], dtype=bool)
    return np.flatnonzero(mask)


def write_h5_split(
    data: dict[str, np.ndarray],
    row_indices: np.ndarray,
    out_path: Path,
    split_name: str,
    seed: int,
    removed_count: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    source_count = len(set(str(value) for value in data["source_file"][row_indices]))

    with h5py.File(out_path, "w") as f:
        for name, values in data.items():
            subset = values[row_indices]
            if subset.dtype.kind in "OSU":
                f.create_dataset(name, data=np.asarray(subset, dtype=object), dtype=string_dtype)
            else:
                f.create_dataset(name, data=subset)

        delta = data["predict_step_1based"][row_indices] - data["known_step_1based"][row_indices]
        f.attrs["split_name"] = split_name
        f.attrs["split_method"] = "source_file_grouped_logS_stratified"
        f.attrs["split_seed"] = int(seed)
        f.attrs["removed_zero_or_negative_offset_count"] = int(removed_count)
        f.attrs["source_file_count"] = int(source_count)
        f.attrs["sample_count"] = int(len(row_indices))
        f.attrs["delta_step_min"] = int(np.min(delta)) if len(delta) else -1
        f.attrs["delta_step_max"] = int(np.max(delta)) if len(delta) else -1


def assert_clean_report(report: dict[str, Any]) -> None:
    source_overlap = sum(item["count"] for item in report["overlap"]["source_file_overlap"].values())
    global_overlap = sum(item["count"] for item in report["overlap"]["source_global_index_overlap"].values())
    if source_overlap:
        raise RuntimeError(f"Clean split still has source_file overlap: {source_overlap}")
    if global_overlap:
        raise RuntimeError(f"Clean split still has source_global_index overlap: {global_overlap}")

    for split in report["splits"]:
        checks = split["checks"]
        if not checks["required_arrays_all_finite"]:
            raise RuntimeError(f"{split['split_name']} has nonfinite required arrays")
        if int(checks["zero_or_negative_offset_count"]) != 0:
            raise RuntimeError(f"{split['split_name']} has zero/negative offset rows")
        if int(checks["contradictory_input_group_count"]) != 0:
            raise RuntimeError(f"{split['split_name']} has contradictory input rows")
    combined = report["combined_checks"]
    if int(combined["zero_or_negative_offset_count"]) != 0:
        raise RuntimeError("Combined clean split has zero/negative offset rows")
    if int(combined["contradictory_input_group_count"]) != 0:
        raise RuntimeError("Combined clean split has contradictory input rows")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build clean Euler-1 source-file-grouped HDF5 splits")
    parser.add_argument("--input_h5", nargs="+", required=True, help="Raw input HDF5 files")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for clean HDF5 splits")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--stratify_by", type=str, default="logS", choices=["logS"])
    args = parser.parse_args()

    del args.stratify_by
    data = load_and_concatenate(args.input_h5)
    filtered, removed_count = filter_positive_offsets(data)
    assert_no_contradictions(filtered)

    splits = stratified_source_split(
        filtered,
        seed=args.seed,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        num_bins=5,
    )

    out_dir = Path(args.out_dir)
    out_paths = {
        "train": out_dir / "euler1_train_clean.h5",
        "valid": out_dir / "euler1_valid_clean.h5",
        "test": out_dir / "euler1_test_clean.h5",
    }
    for split_name, out_path in out_paths.items():
        row_indices = rows_for_sources(filtered, splits[split_name])
        write_h5_split(filtered, row_indices, out_path, split_name, args.seed, removed_count)

    clean_report = inspect_files(
        [out_paths["train"], out_paths["valid"], out_paths["test"]],
        split_names=["train", "valid", "test"],
    )
    assert_clean_report(clean_report)

    report_path = out_dir / "clean_split_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(clean_report), f, indent=2)

    print("Euler-1 reduced-data benchmark clean split: PASS")
    for split_name, out_path in out_paths.items():
        with h5py.File(out_path, "r") as f:
            print(
                f"  {split_name}: {f.attrs['sample_count']} samples, "
                f"{f.attrs['source_file_count']} source files -> {out_path}"
            )
    print(f"  removed nonpositive-offset rows: {removed_count}")
    print(f"  clean report: {report_path}")


if __name__ == "__main__":
    main()
