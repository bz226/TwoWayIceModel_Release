#!/usr/bin/env python
"""Inspect Euler-1 HDF5 files for reduced-data benchmark readiness."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np


REQUIRED_ARRAYS = ("euler1_known", "euler1_predict", "strain_rate", "temperature", "pressure")
REQUIRED_METADATA = ("source_file", "source_global_index", "known_step_1based", "predict_step_1based")


def decode_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return [decode_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, set):
        return sorted(json_safe(item) for item in value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, (bytes, np.bytes_)):
        return decode_value(value)
    return value


def decode_string_array(array: np.ndarray) -> list[str]:
    return [str(decode_value(value)) for value in array]


def infer_split_name(path: Path, fallback_index: int) -> str:
    name = path.name.lower()
    if "train" in name:
        return "train"
    if "valid" in name or "val" in name:
        return "valid"
    if "test" in name:
        return "test"
    return f"split_{fallback_index}"


def numeric_stats(array: np.ndarray) -> dict[str, Any]:
    data = np.asarray(array)
    finite = np.isfinite(data)
    finite_values = data[finite]
    stats: dict[str, Any] = {
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "finite_count": int(np.sum(finite)),
        "nan_count": int(np.sum(np.isnan(data))) if np.issubdtype(data.dtype, np.floating) else 0,
        "posinf_count": int(np.sum(np.isposinf(data))) if np.issubdtype(data.dtype, np.floating) else 0,
        "neginf_count": int(np.sum(np.isneginf(data))) if np.issubdtype(data.dtype, np.floating) else 0,
        "all_finite": bool(np.all(finite)),
    }
    if finite_values.size:
        stats.update(
            {
                "min": float(np.min(finite_values)),
                "max": float(np.max(finite_values)),
                "mean": float(np.mean(finite_values)),
                "std": float(np.std(finite_values)),
            }
        )
    else:
        stats.update({"min": None, "max": None, "mean": None, "std": None})
    return stats


def dataset_summary(dset: h5py.Dataset) -> dict[str, Any]:
    summary: dict[str, Any] = {"shape": list(dset.shape), "dtype": str(dset.dtype)}
    if dset.dtype.kind in "biufc":
        summary.update(numeric_stats(dset[...]))
    else:
        values = decode_string_array(dset[...])
        summary.update(
            {
                "unique_count": len(set(values)),
                "sample_values": values[:5],
            }
        )
    return summary


def target_hashes(target: np.ndarray) -> list[str]:
    return [hashlib.sha1(np.ascontiguousarray(row).view(np.uint8)).hexdigest() for row in target]


def row_quality_checks(
    source_files: list[str],
    source_global_index: np.ndarray,
    known_step: np.ndarray,
    predict_step: np.ndarray,
    target: np.ndarray,
) -> dict[str, Any]:
    delta = predict_step - known_step
    exact_counts: dict[tuple[str, int, int], int] = defaultdict(int)
    target_hash_by_input: dict[tuple[str, int], set[tuple[int, str]]] = defaultdict(set)
    hashes = target_hashes(target)

    for i, source_file in enumerate(source_files):
        known = int(known_step[i])
        predict = int(predict_step[i])
        exact_counts[(source_file, known, predict)] += 1
        target_hash_by_input[(source_file, known)].add((predict, hashes[i]))

    duplicate_exact_groups = {key: count for key, count in exact_counts.items() if count > 1}
    contradictory_groups = {
        key: values for key, values in target_hash_by_input.items() if len(values) > 1
    }
    contradictory_row_count = sum(
        1 for i, source_file in enumerate(source_files) if (source_file, int(known_step[i])) in contradictory_groups
    )

    unique_delta, delta_counts = np.unique(delta, return_counts=True)
    return {
        "sample_count": int(len(source_files)),
        "unique_source_file_count": int(len(set(source_files))),
        "unique_source_global_index_count": int(len(set(int(v) for v in source_global_index))),
        "known_step_range": [int(np.min(known_step)), int(np.max(known_step))] if len(known_step) else [None, None],
        "predict_step_range": [int(np.min(predict_step)), int(np.max(predict_step))] if len(predict_step) else [None, None],
        "offset_distribution": {str(int(k)): int(v) for k, v in zip(unique_delta, delta_counts)},
        "zero_offset_count": int(np.sum(delta == 0)),
        "negative_offset_count": int(np.sum(delta < 0)),
        "zero_or_negative_offset_count": int(np.sum(delta <= 0)),
        "duplicate_exact_group_count": int(len(duplicate_exact_groups)),
        "duplicate_exact_extra_row_count": int(sum(count - 1 for count in duplicate_exact_groups.values())),
        "contradictory_input_group_count": int(len(contradictory_groups)),
        "contradictory_input_row_count": int(contradictory_row_count),
        "contradictory_input_examples": [
            {
                "source_file": key[0],
                "known_step_1based": key[1],
                "target_count": len(values),
            }
            for key, values in list(contradictory_groups.items())[:10]
        ],
    }


def inspect_one(path: str | Path, split_name: str) -> dict[str, Any]:
    path = Path(path)
    with h5py.File(path, "r") as f:
        dataset_names = sorted(f.keys())
        missing_required_arrays = [name for name in REQUIRED_ARRAYS if name not in f]
        missing_required_metadata = [name for name in REQUIRED_METADATA if name not in f]
        datasets = {name: dataset_summary(f[name]) for name in dataset_names}

        checks: dict[str, Any] = {
            "required_arrays_exist": not missing_required_arrays,
            "required_metadata_exist": not missing_required_metadata,
            "missing_required_arrays": missing_required_arrays,
            "missing_required_metadata": missing_required_metadata,
        }

        if not missing_required_arrays and not missing_required_metadata:
            source_files = decode_string_array(f["source_file"][...])
            source_global_index = f["source_global_index"][...]
            known_step = f["known_step_1based"][...]
            predict_step = f["predict_step_1based"][...]
            target = f["euler1_predict"][...]
            checks.update(row_quality_checks(source_files, source_global_index, known_step, predict_step, target))
            checks["required_arrays_all_finite"] = all(
                bool(np.all(np.isfinite(f[name][...]))) for name in REQUIRED_ARRAYS
            )
        else:
            checks["required_arrays_all_finite"] = False

        attrs = {key: decode_value(value) for key, value in f.attrs.items()}

    return {
        "split_name": split_name,
        "path": str(path),
        "dataset_names": dataset_names,
        "datasets": datasets,
        "attributes": attrs,
        "checks": checks,
    }


def collect_rows(path: str | Path) -> dict[str, Any]:
    with h5py.File(path, "r") as f:
        return {
            "source_file": decode_string_array(f["source_file"][...]),
            "source_global_index": [int(v) for v in f["source_global_index"][...]],
            "known_step_1based": f["known_step_1based"][...],
            "predict_step_1based": f["predict_step_1based"][...],
            "euler1_predict": f["euler1_predict"][...],
        }


def overlap_report(split_reports: list[dict[str, Any]], paths: list[str | Path]) -> dict[str, Any]:
    per_split_source: dict[str, set[str]] = {}
    per_split_global: dict[str, set[int]] = {}
    for report, path in zip(split_reports, paths):
        rows = collect_rows(path)
        split_name = report["split_name"]
        per_split_source[split_name] = set(rows["source_file"])
        per_split_global[split_name] = set(rows["source_global_index"])

    source_overlaps = {}
    global_overlaps = {}
    names = list(per_split_source)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            key = f"{left}__{right}"
            source_intersection = sorted(per_split_source[left] & per_split_source[right])
            global_intersection = sorted(per_split_global[left] & per_split_global[right])
            source_overlaps[key] = {
                "count": len(source_intersection),
                "examples": source_intersection[:10],
            }
            global_overlaps[key] = {
                "count": len(global_intersection),
                "examples": global_intersection[:10],
            }
    return {
        "source_file_overlap": source_overlaps,
        "source_global_index_overlap": global_overlaps,
    }


def combined_quality(paths: list[str | Path]) -> dict[str, Any]:
    source_files: list[str] = []
    source_global_index: list[int] = []
    known_steps: list[np.ndarray] = []
    predict_steps: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for path in paths:
        rows = collect_rows(path)
        source_files.extend(rows["source_file"])
        source_global_index.extend(rows["source_global_index"])
        known_steps.append(rows["known_step_1based"])
        predict_steps.append(rows["predict_step_1based"])
        targets.append(rows["euler1_predict"])
    return row_quality_checks(
        source_files,
        np.asarray(source_global_index),
        np.concatenate(known_steps),
        np.concatenate(predict_steps),
        np.concatenate(targets, axis=0),
    )


def build_status(report: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, str]] = []
    any_fail = False
    any_warn = False

    def add(name: str, status: str, message: str) -> None:
        nonlocal any_fail, any_warn
        checks.append({"name": name, "status": status, "message": message})
        any_fail = any_fail or status == "FAIL"
        any_warn = any_warn or status == "WARN"

    missing_arrays = []
    missing_metadata = []
    all_finite = True
    zero_or_negative = 0
    contradictory = 0
    for split in report["splits"]:
        check = split["checks"]
        missing_arrays.extend(check.get("missing_required_arrays", []))
        missing_metadata.extend(check.get("missing_required_metadata", []))
        all_finite = all_finite and bool(check.get("required_arrays_all_finite", False))
        zero_or_negative += int(check.get("zero_or_negative_offset_count", 0))
        contradictory += int(check.get("contradictory_input_group_count", 0))

    add(
        "Required arrays exist",
        "PASS" if not missing_arrays and not missing_metadata else "FAIL",
        f"missing arrays={sorted(set(missing_arrays))}, missing metadata={sorted(set(missing_metadata))}",
    )
    add(
        "Required arrays are finite",
        "PASS" if all_finite else "FAIL",
        "all required numeric arrays are finite" if all_finite else "at least one required array has NaN or Inf",
    )

    source_overlap_count = sum(v["count"] for v in report["overlap"]["source_file_overlap"].values())
    global_overlap_count = sum(v["count"] for v in report["overlap"]["source_global_index_overlap"].values())
    add(
        "Source-file leakage across splits",
        "WARN" if source_overlap_count else "PASS",
        f"{source_overlap_count} pairwise source_file overlaps",
    )
    add(
        "Source-global-index leakage across splits",
        "FAIL" if global_overlap_count else "PASS",
        f"{global_overlap_count} pairwise source_global_index overlaps",
    )
    add(
        "Zero or negative offset rows",
        "FAIL" if zero_or_negative else "PASS",
        f"{zero_or_negative} rows have predict_step_1based <= known_step_1based",
    )
    add(
        "Contradictory labels",
        "FAIL" if contradictory else "PASS",
        f"{contradictory} input groups have more than one target",
    )

    for split in report["splits"]:
        if split["split_name"] == "test":
            sample_count = int(split["checks"].get("sample_count", 0))
            add(
                "Test set size",
                "WARN" if sample_count < 30 else "PASS",
                f"test split has {sample_count} samples",
            )

    overall = "FAIL" if any_fail else "WARN" if any_warn else "PASS"
    return {"overall_status": overall, "checks": checks}


def inspect_files(paths: list[str | Path], split_names: list[str] | None = None) -> dict[str, Any]:
    resolved_paths = [Path(path) for path in paths]
    if split_names is None:
        split_names = [infer_split_name(path, i) for i, path in enumerate(resolved_paths)]
    if len(split_names) != len(resolved_paths):
        raise ValueError("--split_names must have the same length as --h5")

    splits = [inspect_one(path, split_name) for path, split_name in zip(resolved_paths, split_names)]
    report = {
        "label": "Euler-1 reduced-data benchmark",
        "splits": splits,
        "overlap": overlap_report(splits, resolved_paths),
        "combined_checks": combined_quality(resolved_paths),
    }
    report["status"] = build_status(report)
    return report


def print_status(report: dict[str, Any]) -> None:
    print(f"Euler-1 reduced-data benchmark inspection: {report['status']['overall_status']}")
    for check in report["status"]["checks"]:
        print(f"  {check['status']}: {check['name']} - {check['message']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Euler-1 benchmark HDF5 files")
    parser.add_argument("--h5", nargs="+", required=True, help="HDF5 files to inspect")
    parser.add_argument("--out", type=str, default=None, help="Optional JSON report path")
    parser.add_argument("--split_names", nargs="*", default=None, help="Optional names matching --h5 order")
    args = parser.parse_args()

    report = inspect_files(args.h5, args.split_names)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(json_safe(report), f, indent=2)
    print_status(report)


if __name__ == "__main__":
    main()
