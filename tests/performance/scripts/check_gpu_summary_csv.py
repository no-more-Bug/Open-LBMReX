#!/usr/bin/env python3
"""Validate the fixed eight-row GPU throughput and breakdown summaries."""

import argparse
from performance_expectations import GPU_FIELDS, add_expected_arguments, expected_values
import csv
import math
from pathlib import Path
import re
from typing import Dict, List

from check_gpu_performance_csv import (
    CRITICAL_PER_STEP_FIELDS,
    IDENTITIES,
    INDEPENDENT_MAX_FIELDS,
    SHARE_FIELDS,
    ValidationError,
)
from gpu_summary_common import (
    FORMAL_AMREX_USE_GPU_AWARE_MPI,
    FORMAL_COMMUNICATION_MODE,
    GROUP_ORDER,
    IDENTITY_FIELDS,
    REPEAT_PROVENANCE_FIELDS,
    require,
)


SUMMARY_IDENTITY_FIELDS = [
    field for field in IDENTITY_FIELDS
    if field not in {"measurement_mode", "run_mode", "workload", "layout"}
]
SUMMARY_KEY_FIELDS = ["measurement_mode", "run_mode", "workload", "layout"]
PROVENANCE_COLUMNS = [
    "repeat{}_{}".format(repeat, field)
    for repeat in (1, 2, 3)
    for field in REPEAT_PROVENANCE_FIELDS
]
BASELINE_METRICS = [
    "repeat_count",
    "noise_status",
    "total_min_s_per_step",
    "total_median_s_per_step",
    "total_max_s_per_step",
    "total_cv",
    "mlups_min",
    "mlups_median",
    "mlups_max",
    "speedup_1_to_4",
    "efficiency_1_to_4",
]
STAGE_METRICS = [
    "repeat_count",
    "noise_status",
    "diagnostic_total_min_s_per_step",
    "diagnostic_total_median_s_per_step",
    "diagnostic_total_max_s_per_step",
    "diagnostic_total_cv",
    "diagnostic_mlups_median",
] + [
    "{}_median".format(field)
    for field in CRITICAL_PER_STEP_FIELDS + SHARE_FIELDS
] + [
    "{}_median_s_per_step".format(
        field[:-len("_window_s")]
        if field.endswith("_window_s") else field
    )
    for field in INDEPENDENT_MAX_FIELDS
]
BASELINE_COLUMNS = (
    ["schema_version"] + SUMMARY_IDENTITY_FIELDS + SUMMARY_KEY_FIELDS
    + PROVENANCE_COLUMNS + BASELINE_METRICS
)
STAGE_COLUMNS = (
    ["schema_version"] + SUMMARY_IDENTITY_FIELDS + SUMMARY_KEY_FIELDS
    + PROVENANCE_COLUMNS + STAGE_METRICS
)
CROSS_MODE_FIELDS = [
    "descriptor",
    "git_commit",
    "amrex_source_commit",
    "amrex_prefix",
    "amrex_config_sha256",
    "amrex_build_type",
    "build_type",
    "compiler",
    "nvcc",
    "cuda_toolkit",
    "cuda_runtime",
    "driver",
    "mpi",
    "mpi_cuda_build_version",
    "mpi_cuda_support_reported",
    "mpi_cuda_query_support",
    "partition",
    "qos",
    "grid_x",
    "grid_y",
    "grid_z",
    "global_valid_cells",
    "max_grid_size",
    "box_count",
    "mpi_ranks",
    "mig_instance_count",
    "parent_gpu_count",
    "boxes_per_rank",
    "omp_threads",
    "amrex_use_gpu_aware_mpi",
    "communication_mode",
    "binding_policy",
    "mig_profile",
    "num_devices_used",
    "num_device_partners",
]


def integer(row: Dict[str, str], field: str, path: Path) -> int:
    try:
        return int(row[field])
    except ValueError as exc:
        raise ValidationError(f"{path}: {field} is not an integer") from exc


def number(row: Dict[str, str], field: str, path: Path) -> float:
    try:
        value = float(row[field])
    except ValueError as exc:
        raise ValidationError(f"{path}: {field} is not numeric") from exc
    require(math.isfinite(value), f"{path}: {field} is not finite")
    return value


def close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=1.0e-11, abs_tol=1.0e-12)


def read_rows(path: Path, columns: List[str]) -> List[Dict[str, str]]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == columns, f"{path}: summary columns mismatch")
        rows = list(reader)
    require(len(rows) == 8, f"{path}: expected eight summary rows")
    require(
        all(
            None not in row
            and all(value not in {None, ""} for value in row.values())
            for row in rows
        ),
        f"{path}: malformed or empty summary field",
    )
    return rows


def validate_common(row: Dict[str, str], path: Path, key: tuple, expected: Dict[str, str]) -> None:
    workload, layout = key
    descriptor, grid, max_grid_size, box_count = IDENTITIES[workload]
    ranks = 1 if layout == "1mig" else 4
    require(integer(row, "schema_version", path) == 1, f"{path}: wrong schema")
    require(row["run_mode"] == "baseline", f"{path}: wrong run mode")
    require(row["workload"] == workload and row["layout"] == layout, f"{path}: row order/key mismatch")
    require(row["descriptor"] == descriptor, f"{path}: descriptor mismatch")
    require(
        tuple(integer(row, field, path) for field in ("grid_x", "grid_y", "grid_z")) == grid,
        f"{path}: grid mismatch",
    )
    require(
        integer(row, "global_valid_cells", path) == grid[0] * grid[1] * grid[2],
        f"{path}: cell count mismatch",
    )
    require(integer(row, "max_grid_size", path) == max_grid_size, f"{path}: max_grid_size mismatch")
    require(integer(row, "box_count", path) == box_count, f"{path}: box count mismatch")
    require(integer(row, "mpi_ranks", path) == ranks, f"{path}: rank count mismatch")
    require(integer(row, "mig_instance_count", path) == ranks, f"{path}: MIG count mismatch")
    require(integer(row, "parent_gpu_count", path) == 1, f"{path}: parent GPU count mismatch")
    require(integer(row, "boxes_per_rank", path) == box_count // ranks, f"{path}: boxes/rank mismatch")
    require(integer(row, "omp_threads", path) == 1, f"{path}: OpenMP identity mismatch")
    require(integer(row, "num_devices_used", path) == 1, f"{path}: AMReX device count mismatch")
    require(integer(row, "num_device_partners", path) == ranks, f"{path}: AMReX partner count mismatch")
    require(
        row["amrex_use_gpu_aware_mpi"] == FORMAL_AMREX_USE_GPU_AWARE_MPI
        and row["communication_mode"] == FORMAL_COMMUNICATION_MODE,
        f"{path}: formal communication identity is not pinned staging",
    )
    require(row["mpi_cuda_query_support"] == "true", f"{path}: MPIX capability identity mismatch")
    require(re.fullmatch(r"[0-9a-f]{40}", row["git_commit"]) is not None, f"{path}: invalid Git SHA")
    require(re.fullmatch(r"[0-9a-f]{40}", row["amrex_source_commit"]) is not None, f"{path}: invalid AMReX SHA")
    require(re.fullmatch(r"[0-9a-f]{64}", row["amrex_config_sha256"]) is not None, f"{path}: invalid AMReX config hash")
    require(Path(row["amrex_prefix"]).is_absolute(), f"{path}: AMReX prefix must be absolute")
    for field in GPU_FIELDS:
        require(row[field] not in {"", "unknown", "unset"}, f"{path}: missing {field}")
        require(row[field] == expected[field], f"{path}: {field} differs from explicit expectation")
    require(row["amrex_build_type"] == row["build_type"] == "Release", f"{path}: build type mismatch")
    require(row["mpi_cuda_support_reported"] == "true", f"{path}: MPI CUDA report mismatch")
    warmup = integer(row, "warmup_steps", path)
    sample_start = integer(row, "sample_start", path)
    sample_end = integer(row, "sample_end", path)
    nsteps = integer(row, "nsteps", path)
    require(warmup > 0 and nsteps > 0, f"{path}: invalid formal window")
    require(sample_start == warmup + 1 and sample_end - sample_start + 1 == nsteps, f"{path}: incomplete formal window")
    require(integer(row, "repeat_count", path) == 3, f"{path}: repeat count mismatch")
    for column in PROVENANCE_COLUMNS:
        require(row[column] not in {"unknown", "unset"}, f"{path}: missing {column}")


def validate_noise(row: Dict[str, str], path: Path, prefix: str) -> None:
    minimum = number(row, f"{prefix}_min_s_per_step", path)
    median = number(row, f"{prefix}_median_s_per_step", path)
    maximum = number(row, f"{prefix}_max_s_per_step", path)
    cv = number(row, f"{prefix}_cv", path)
    require(0.0 < minimum <= median <= maximum and cv >= 0.0, f"{path}: invalid timing aggregate")
    expected = "shared-node-noisy" if cv > 0.05 or maximum / minimum > 1.10 else "baseline"
    require(row["noise_status"] == expected, f"{path}: noise status mismatch")


def validate_baseline(rows: List[Dict[str, str]], path: Path, expected: Dict[str, str]) -> None:
    for row, key in zip(rows, GROUP_ORDER):
        validate_common(row, path, key, expected)
        require(row["measurement_mode"] == "throughput", f"{path}: wrong throughput mode")
        validate_noise(row, path, "total")
        minimum = number(row, "mlups_min", path)
        median = number(row, "mlups_median", path)
        maximum = number(row, "mlups_max", path)
        require(0.0 < minimum <= median <= maximum, f"{path}: invalid MLUPS aggregate")
        cells = integer(row, "global_valid_cells", path)
        require(close(median, cells / number(row, "total_median_s_per_step", path) / 1.0e6), f"{path}: median MLUPS mismatch")
    by_key = {(row["workload"], row["layout"]): row for row in rows}
    for workload in {key[0] for key in GROUP_ORDER}:
        one = by_key[(workload, "1mig")]
        four = by_key[(workload, "4mig")]
        speedup = number(four, "mlups_median", path) / number(one, "mlups_median", path)
        for row in (one, four):
            require(close(number(row, "speedup_1_to_4", path), speedup), f"{path}: speedup mismatch")
            require(close(number(row, "efficiency_1_to_4", path), speedup / 4.0), f"{path}: efficiency mismatch")


def validate_breakdown(rows: List[Dict[str, str]], path: Path, expected: Dict[str, str]) -> None:
    for row, key in zip(rows, GROUP_ORDER):
        validate_common(row, path, key, expected)
        require(row["measurement_mode"] == "synchronized_stage_breakdown", f"{path}: wrong breakdown mode")
        validate_noise(row, path, "diagnostic_total")
        require(number(row, "diagnostic_mlups_median", path) > 0.0, f"{path}: non-positive diagnostic MLUPS")
        for field in CRITICAL_PER_STEP_FIELDS:
            require(number(row, "{}_median".format(field), path) >= 0.0, f"{path}: negative critical timing")
        require(
            close(
                number(row, "critical_rank_total_s_per_step_median", path),
                number(row, "diagnostic_total_median_s_per_step", path),
            ),
            f"{path}: diagnostic total mismatch",
        )
        for field in SHARE_FIELDS:
            share = number(row, "{}_median".format(field), path)
            require(0.0 <= share <= 100.0, f"{path}: invalid stage share")
        for field in INDEPENDENT_MAX_FIELDS:
            stem = field[:-len("_window_s")] if field.endswith("_window_s") else field
            require(number(row, "{}_median_s_per_step".format(stem), path) >= 0.0, f"{path}: negative independent maximum")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("breakdown", type=Path)
    add_expected_arguments(parser, GPU_FIELDS)
    args = parser.parse_args()
    expected = expected_values(args, GPU_FIELDS)
    try:
        baseline = read_rows(args.baseline, BASELINE_COLUMNS)
        breakdown = read_rows(args.breakdown, STAGE_COLUMNS)
        validate_baseline(baseline, args.baseline, expected)
        validate_breakdown(breakdown, args.breakdown, expected)
        for baseline_row, breakdown_row in zip(baseline, breakdown):
            require(
                all(baseline_row[field] == breakdown_row[field] for field in CROSS_MODE_FIELDS),
                "throughput/breakdown fixed identity drift",
            )
    except ValidationError as exc:
        print(f"FAIL: {exc}")
        return 1
    except (OSError, csv.Error) as exc:
        print(f"ERROR: {exc}")
        return 2
    print("gpu_summary_csv passed baseline_rows=8 breakdown_rows=8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
