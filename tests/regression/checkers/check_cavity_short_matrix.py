#!/usr/bin/env python3
"""Validate the Stage 4.9B cavity short replacement matrix."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

DIAGNOSTICS = {"step", "time_lb", "time_phys", "mass", "relative_mass_error"}
COMMON_SCHEMAS = {
    "diagnostics.csv": DIAGNOSTICS | {"l2_uz", "linf_uz"},
    "cavity_centerline_x0p5_ux.csv": {"y_over_L", "ux_lb", "ux_over_Ulid"},
    "cavity_centerline_y0p5_uy.csv": {"x_over_L", "uy_lb", "uy_over_Ulid"},
}
METRICS_SCHEMA = {"step", "relative_mass_error", "rho_min", "rho_max"}


def check_csv(path: Path, required: set[str]) -> None:
    if not path.is_file():
        raise SystemExit(f"missing artifact: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        header = reader.fieldnames or []
        if (
            not header
            or any(not name for name in header)
            or len(set(header)) != len(header)
        ):
            raise SystemExit(f"{path}: empty or duplicate column name")
        missing = required - set(header)
        if missing:
            raise SystemExit(f"{path}: missing columns {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise SystemExit(f"{path}: no data rows")
    for row_index, row in enumerate(rows):
        if None in row or any(value is None for value in row.values()):
            raise SystemExit(f"{path}: row {row_index} width mismatch")
        for name, value in row.items():
            try:
                number = float(value)
            except ValueError as exc:
                raise SystemExit(
                    f"{path}: row {row_index} {name} is not numeric: {value!r}"
                ) from exc
            if not math.isfinite(number):
                raise SystemExit(f"{path}: row {row_index} {name}={number} is not finite")


def read_metadata(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise SystemExit(f"missing cavity metadata: {path}")
    values: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text().splitlines(), start=1):
        if "=" not in raw:
            raise SystemExit(f"{path}:{line_number}: malformed metadata line")
        key, value = (part.strip() for part in raw.split("=", 1))
        if not key or key in values:
            raise SystemExit(f"{path}:{line_number}: empty or duplicate key {key!r}")
        values[key] = value
    return values


def expect_value(values: dict[str, str], key: str, expected: str, owner: Path) -> None:
    actual = values.get(key)
    if actual != expected:
        raise SystemExit(f"{owner}: {key}={actual!r}, expected {expected!r}")


def expect_real(
    values: dict[str, str], key: str, expected: float, owner: Path
) -> None:
    try:
        actual = float(values[key])
    except (KeyError, ValueError) as exc:
        raise SystemExit(f"{owner}: missing or nonnumeric {key}") from exc
    if not math.isfinite(actual) or not math.isclose(
        actual, expected, rel_tol=1.0e-13, abs_tol=1.0e-13
    ):
        raise SystemExit(f"{owner}: {key}={actual}, expected {expected}")


def check_run(
    run_dir: Path,
    *,
    valid_nodes: int,
    intervals: int,
    on_lattice: bool,
) -> None:
    if not run_dir.is_dir():
        raise SystemExit(f"missing cavity run directory: {run_dir}")
    for name, columns in COMMON_SCHEMAS.items():
        check_csv(run_dir / name, columns)

    metrics = run_dir / "cavity_validation_metrics.csv"
    if on_lattice:
        check_csv(metrics, METRICS_SCHEMA)
    elif metrics.exists():
        raise SystemExit(f"HBB run unexpectedly wrote Guo wall metrics: {metrics}")

    metadata_path = run_dir / "cavity_centerline_metadata.txt"
    metadata = read_metadata(metadata_path)
    expected_text = {
        "boundary_geometry": (
            "on_lattice_valid_node" if on_lattice else "domain_face_halfway"
        ),
        "sampling": "direct_node" if on_lattice else "interpolated_cell_center",
        "interpolation": "false" if on_lattice else "true",
        "endpoint_injection": "false" if on_lattice else "true",
        "valid_nodes_x": str(valid_nodes),
        "valid_nodes_y": str(valid_nodes),
        "lattice_intervals_x": str(intervals),
        "lattice_intervals_y": str(intervals),
        "z_policy": "average",
    }
    for key, expected in expected_text.items():
        expect_value(metadata, key, expected, metadata_path)

    half_cell = 0.5 / intervals if on_lattice else 0.0
    expect_real(metadata, "physical_lo_x", 0.0, metadata_path)
    expect_real(metadata, "physical_hi_x", 1.0, metadata_path)
    expect_real(metadata, "physical_lo_y", 0.0, metadata_path)
    expect_real(metadata, "physical_hi_y", 1.0, metadata_path)
    expect_real(metadata, "envelope_lo_x", -half_cell, metadata_path)
    expect_real(metadata, "envelope_hi_x", 1.0 + half_cell, metadata_path)
    expect_real(metadata, "envelope_lo_y", -half_cell, metadata_path)
    expect_real(metadata, "envelope_hi_y", 1.0 + half_cell, metadata_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d2-guo", type=Path, required=True)
    parser.add_argument("--d2-hbb", type=Path, required=True)
    parser.add_argument("--d3-guo", type=Path, required=True)
    parser.add_argument("--d3-hbb", type=Path, required=True)
    args = parser.parse_args()

    check_run(args.d2_guo, valid_nodes=17, intervals=16, on_lattice=True)
    check_run(args.d2_hbb, valid_nodes=129, intervals=129, on_lattice=False)
    check_run(args.d3_guo, valid_nodes=129, intervals=128, on_lattice=True)
    check_run(args.d3_hbb, valid_nodes=129, intervals=129, on_lattice=False)
    print("cavity_short_matrix passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
