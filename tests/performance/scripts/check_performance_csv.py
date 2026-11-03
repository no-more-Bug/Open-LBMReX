#!/usr/bin/env python3
"""Validate one or more tests-owned CPU MPI performance records."""

import argparse
from performance_expectations import CPU_FIELDS, add_expected_arguments, expected_values
import csv
import math
from pathlib import Path
import re
from typing import Dict


EXPECTED_COLUMNS = [
    "schema_version",
    "workload",
    "mode",
    "descriptor",
    "repeat",
    "git_commit",
    "amrex_source_commit",
    "amrex_prefix",
    "amrex_build_type",
    "build_type",
    "compiler",
    "mpi",
    "ucx_version",
    "ucx_tls",
    "node",
    "partition",
    "qos",
    "grid_x",
    "grid_y",
    "grid_z",
    "global_valid_cells",
    "max_grid_size",
    "box_count",
    "mpi_ranks",
    "omp_threads",
    "cpuset",
    "binding_policy",
    "sample_start",
    "sample_end",
    "nsteps",
    "timing_reduction",
    "collision_window_s",
    "pre_stream_window_s",
    "pre_stream_fill_boundary_window_s",
    "pre_stream_setup_window_s",
    "pre_stream_physical_ghosts_window_s",
    "streaming_window_s",
    "domain_post_stream_window_s",
    "internal_solid_reconstruction_window_s",
    "post_stream_window_s",
    "diagnostics_window_s",
    "rank_max_local_total_window_s",
    "synchronized_window_s",
    "total_s_per_step",
    "mlups",
]

IDENTITIES = {
    "tgv_d2q9": ("D2Q9", (1024, 1024, 1), 256, 16),
    "tgv_d3q27": ("D3Q27", (512, 512, 4), 128, 16),
    "dfg_d2q9": ("D2Q9", (1761, 329, 1), 224, 16),
    "dfg_d3q27": ("D3Q27", (881, 165, 4), 112, 16),
}

CALIBRATION_SAMPLES = {
    "tgv_d2q9": (1201, 13200, 12000),
    "tgv_d3q27": (201, 2200, 2000),
    "dfg_d2q9": (2401, 26400, 24000),
    "dfg_d3q27": (301, 3300, 3000),
}
MIN_BASELINE_WINDOW_SECONDS = 600.0

INTEGER_FIELDS = {
    "schema_version",
    "repeat",
    "grid_x",
    "grid_y",
    "grid_z",
    "global_valid_cells",
    "max_grid_size",
    "box_count",
    "mpi_ranks",
    "omp_threads",
    "sample_start",
    "sample_end",
    "nsteps",
}

TIMING_FIELDS = [
    "collision_window_s",
    "pre_stream_window_s",
    "pre_stream_fill_boundary_window_s",
    "pre_stream_setup_window_s",
    "pre_stream_physical_ghosts_window_s",
    "streaming_window_s",
    "domain_post_stream_window_s",
    "internal_solid_reconstruction_window_s",
    "post_stream_window_s",
    "diagnostics_window_s",
    "rank_max_local_total_window_s",
    "synchronized_window_s",
    "total_s_per_step",
    "mlups",
]


class ValidationError(Exception):
    """The artifact is readable but violates the performance contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _product(values):
    result = 1
    for value in values:
        result *= value
    return result


def _read_row(path: Path) -> Dict[str, str]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        header = reader.fieldnames or []
        _require(header == EXPECTED_COLUMNS, f"{path}: header mismatch")
        rows = list(reader)
    _require(len(rows) == 1, f"{path}: expected one row, found {len(rows)}")
    row = rows[0]
    _require(
        None not in row and all(value is not None for value in row.values()),
        f"{path}: row width mismatch",
    )
    return row


def validate_path(path: Path, expected: Dict[str, str]) -> Dict[str, str]:
    row = _read_row(path)
    try:
        integers = {name: int(row[name]) for name in INTEGER_FIELDS}
        timings = {name: float(row[name]) for name in TIMING_FIELDS}
    except ValueError as exc:
        raise ValidationError(f"{path}: invalid numeric field") from exc

    _require(integers["schema_version"] == 2, f"{path}: unsupported schema")
    workload = row["workload"]
    _require(workload in IDENTITIES, f"{path}: unknown workload {workload!r}")
    descriptor, grid, max_grid_size, box_count = IDENTITIES[workload]
    _require(row["descriptor"] == descriptor, f"{path}: descriptor mismatch")
    actual_grid = tuple(integers[f"grid_{axis}"] for axis in "xyz")
    _require(actual_grid == grid, f"{path}: grid mismatch {actual_grid} != {grid}")
    _require(
        integers["global_valid_cells"] == _product(grid),
        f"{path}: global cell count mismatch",
    )
    _require(
        integers["max_grid_size"] == max_grid_size
        and integers["box_count"] == box_count,
        f"{path}: box identity mismatch",
    )

    mode = row["mode"]
    _require(mode in {"smoke", "calibration", "baseline"}, f"{path}: bad mode")
    _require(
        integers["mpi_ranks"] in {8, 16},
        f"{path}: mpi_ranks must be the frozen 16 or recorded 8-rank fallback",
    )
    _require(integers["omp_threads"] == 1, f"{path}: omp_threads must be one")
    _require(
        integers["sample_start"] > 0
        and integers["sample_end"] >= integers["sample_start"]
        and integers["nsteps"] > 0,
        f"{path}: invalid sample indices",
    )
    _require(
        integers["sample_end"] - integers["sample_start"] + 1
        == integers["nsteps"],
        f"{path}: non-contiguous sample window",
    )
    sample_identity = (
        integers["sample_start"],
        integers["sample_end"],
        integers["nsteps"],
    )
    if mode == "smoke":
        _require(
            integers["repeat"] == 1 and sample_identity == (1, 20, 20),
            f"{path}: smoke must use repeat 1 and window 1:20",
        )
    elif mode == "calibration":
        _require(
            integers["repeat"] == 1
            and sample_identity == CALIBRATION_SAMPLES[workload],
            f"{path}: wrong fixed calibration window for {workload}",
        )
    elif mode == "baseline":
        _require(
            integers["repeat"] in {1, 2, 3}
            and integers["sample_start"] > 1,
            f"{path}: baseline repeat or warmup is invalid",
        )

    _require(
        row["timing_reduction"]
        == "independent_rank_max_with_synchronized_window",
        f"{path}: timing reduction mismatch",
    )
    for name, value in timings.items():
        _require(math.isfinite(value), f"{path}: {name} is not finite")
        _require(value >= 0.0, f"{path}: {name} is negative")
    for name in (
        "rank_max_local_total_window_s",
        "synchronized_window_s",
        "total_s_per_step",
        "mlups",
    ):
        _require(timings[name] > 0.0, f"{path}: {name} must be positive")

    nsteps = integers["nsteps"]
    synchronized = timings["synchronized_window_s"]
    if mode == "baseline":
        _require(
            synchronized >= MIN_BASELINE_WINDOW_SECONDS,
            f"{path}: baseline synchronized window is shorter than 600 seconds",
        )
    expected_per_step = synchronized / nsteps
    expected_mlups = integers["global_valid_cells"] * nsteps / synchronized / 1.0e6
    _require(
        math.isclose(
            timings["total_s_per_step"], expected_per_step, rel_tol=1.0e-12
        ),
        f"{path}: total_s_per_step formula mismatch",
    )
    _require(
        math.isclose(timings["mlups"], expected_mlups, rel_tol=1.0e-12),
        f"{path}: MLUPS formula mismatch",
    )

    required_context = [
        "git_commit",
        "amrex_source_commit",
        "amrex_prefix",
        "amrex_build_type",
        "build_type",
        "compiler",
        "mpi",
        "ucx_version",
        "ucx_tls",
        "node",
        "partition",
        "qos",
        "cpuset",
        "binding_policy",
    ]
    for name in required_context:
        _require(
            row[name].strip() not in {"", "unknown"},
            f"{path}: missing context {name}",
        )
    _require(
        re.fullmatch(r"[0-9a-f]{40}", row["git_commit"]) is not None,
        f"{path}: git_commit must be a full SHA",
    )
    _require(
        re.fullmatch(r"[0-9a-f]{40}", row["amrex_source_commit"]) is not None,
        f"{path}: amrex_source_commit must be a full SHA",
    )
    _require(Path(row["amrex_prefix"]).is_absolute(), f"{path}: AMReX prefix must be absolute")
    for field in CPU_FIELDS:
        _require(row[field] not in {"", "unknown", "unset"}, f"{path}: missing {field}")
        _require(row[field] == expected[field], f"{path}: {field} differs from explicit expectation")
    _require(row["amrex_build_type"] == "Release", f"{path}: AMReX is not Release")
    _require(row["build_type"] == "Release", f"{path}: application is not Release")
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", nargs="+", type=Path)
    add_expected_arguments(parser, CPU_FIELDS)
    args = parser.parse_args()
    expected = expected_values(args, CPU_FIELDS)
    try:
        for path in args.records:
            validate_path(path, expected)
    except ValidationError as exc:
        print(f"FAIL: {exc}")
        return 1
    except (OSError, csv.Error) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(f"performance_csv passed records={len(args.records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
