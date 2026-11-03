#!/usr/bin/env python3
"""Independently validate the direct uniform-flow short artifact matrix."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


EXPECTED_COLUMNS = [
    "step",
    "flow_throughs",
    "E_2_u",
    "E_inf_u",
    "mean_ux",
    "std_ux_over_u0",
    "max_abs_uy_over_u0",
    "max_abs_uz_over_u0",
    "relative_mass_error",
    "rho_mean_minus_rho0",
    "rho_std_over_rho0",
    "raw_lattice_mass",
    "raw_lattice_reference_mass",
    "physical_weighted_mass",
    "physical_weighted_reference_mass",
    "physical_weighted_relative_mass_error",
]


def read_single_row(directory: Path) -> dict[str, float]:
    path = directory / "uniform_flow_metrics.csv"
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        header = reader.fieldnames or []
        if (
            not header
            or any(not name for name in header)
            or len(set(header)) != len(header)
        ):
            raise SystemExit(f"{path}: empty or duplicate column name")
        if header != EXPECTED_COLUMNS:
            raise SystemExit(
                f"{path}: header mismatch; expected {EXPECTED_COLUMNS}, found {header}"
            )
        rows = list(reader)
    if len(rows) != 1:
        raise SystemExit(f"{path}: expected one final row, found {len(rows)}")
    row = rows[0]
    if None in row or any(value is None for value in row.values()):
        raise SystemExit(f"{path}: row width mismatch")
    try:
        values = {key: float(value) for key, value in row.items()}
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"{path}: non-numeric value") from exc
    for name, value in values.items():
        if not math.isfinite(value):
            raise SystemExit(f"{path}: {name}={value} is not finite")
    return values


def require_close(
    failures: list[str],
    label: str,
    value: float,
    expected: float,
    tolerance: float = 1.0e-10,
) -> None:
    if not math.isclose(value, expected, rel_tol=0.0, abs_tol=tolerance):
        failures.append(f"{label}={value:.17g} != {expected:.17g}")


def check_case(
    label: str,
    directory: Path,
    expected_step: int,
    expected_flow: float,
    expected_raw_mass: float,
    expected_weighted_mass: float,
) -> list[str]:
    row = read_single_row(directory)
    failures: list[str] = []
    require_close(failures, f"{label}.step", row["step"], expected_step)
    require_close(
        failures, f"{label}.flow_throughs", row["flow_throughs"], expected_flow
    )
    require_close(
        failures,
        f"{label}.raw_lattice_mass",
        row["raw_lattice_mass"],
        expected_raw_mass,
    )
    require_close(
        failures,
        f"{label}.raw_lattice_reference_mass",
        row["raw_lattice_reference_mass"],
        expected_raw_mass,
    )
    require_close(
        failures,
        f"{label}.physical_weighted_mass",
        row["physical_weighted_mass"],
        expected_weighted_mass,
    )
    require_close(
        failures,
        f"{label}.physical_weighted_reference_mass",
        row["physical_weighted_reference_mass"],
        expected_weighted_mass,
    )
    for name, tolerance in (
        ("E_2_u", 1.0e-8),
        ("E_inf_u", 1.0e-7),
        ("std_ux_over_u0", 1.0e-8),
        ("max_abs_uy_over_u0", 1.0e-8),
        ("max_abs_uz_over_u0", 1.0e-8),
        ("relative_mass_error", 1.0e-8),
        ("physical_weighted_relative_mass_error", 1.0e-8),
    ):
        if row[name] > tolerance:
            failures.append(f"{label}.{name}={row[name]:.17g} > {tolerance:.17g}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hbb", required=True, type=Path)
    parser.add_argument("--guo-velocity-d2q9", required=True, type=Path)
    parser.add_argument("--guo-free-d2q9", required=True, type=Path)
    parser.add_argument("--guo-velocity-d3q27", required=True, type=Path)
    parser.add_argument("--guo-free-d3q27", required=True, type=Path)
    args = parser.parse_args()

    failures = []
    failures += check_case("hbb", args.hbb, 2, 0.02, 100.0, 100.0)
    failures += check_case(
        "guo_velocity_d2q9", args.guo_velocity_d2q9, 20, 0.2, 110.0, 100.0
    )
    failures += check_case(
        "guo_free_d2q9", args.guo_free_d2q9, 20, 0.2, 110.0, 100.0
    )
    failures += check_case(
        "guo_velocity_d3q27", args.guo_velocity_d3q27, 20, 0.2, 440.0, 400.0
    )
    failures += check_case(
        "guo_free_d3q27", args.guo_free_d3q27, 20, 0.2, 440.0, 400.0
    )
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print("uniform_short_matrix passed: HBB and D2/D3 Guo velocity/free")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
