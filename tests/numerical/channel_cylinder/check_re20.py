#!/usr/bin/env python3
"""Evaluate the four tests-owned DFG-derived Re20 artifacts."""

from __future__ import annotations

import csv
import hashlib
import math
import sys
from pathlib import Path

REFERENCE_SHA256 = "0a4cb839f008506e7541f76f0db23c7eaa5c44672f2347d09870118b46448043"
HEADER = [
    "reference_label", "descriptor", "reynolds", "diameter_intervals",
    "depth_cells", "sample_count", "final_population_step", "Cd_final",
    "Cl_final", "C_delta_p", "max_abs_Cz", "steady_reached",
    "steady_tolerance", "final_steady_residual",
]
IDENTITIES = {
    "d2_nd20": ("D2Q9", 20, 1, 60000),
    "d3_nd20": ("D3Q27", 20, 4, 60000),
    "d2_nd40": ("D2Q9", 40, 1, 120000),
    "d2_nd80": ("D2Q9", 80, 1, 240000),
}


def read_reference(path: Path) -> dict[str, tuple[float, str, float]]:
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != REFERENCE_SHA256:
        raise ValueError("Re20 reference is missing or its checksum changed")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if {row["metric"] for row in rows} != {
        "Cd_final", "Cl_final", "C_delta_p", "steady_residual",
        "Cd_resolution_difference", "Cl_resolution_difference",
    }:
        raise ValueError("Re20 reference rows changed")
    return {
        row["metric"]: (
            float(row["value"]), row["tolerance_kind"], float(row["tolerance"])
        )
        for row in rows
    }


def read_metrics(name: str, directory: Path) -> dict[str, float | str]:
    path = directory / "channel_cylinder_metrics.csv"
    if not path.is_file():
        raise ValueError(f"{name}: missing metrics")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    if reader.fieldnames != HEADER or len(rows) != 1:
        raise ValueError(f"{name}: invalid metrics schema or row count")
    row = rows[0]
    descriptor, diameter, depth, steps = IDENTITIES[name]
    expected = {
        "reference_label": "dfg_2d_1_re20", "descriptor": descriptor,
        "reynolds": "20", "diameter_intervals": str(diameter),
        "depth_cells": str(depth), "sample_count": str(steps),
        "final_population_step": str(steps), "steady_reached": "true",
    }
    if any(row.get(key) != value for key, value in expected.items()):
        raise ValueError(f"{name}: scientific identity or completion changed")
    numeric = {}
    for key in (
        "Cd_final", "Cl_final", "C_delta_p", "max_abs_Cz",
        "steady_tolerance", "final_steady_residual",
    ):
        value = float(row[key])
        if not math.isfinite(value):
            raise ValueError(f"{name}: {key} is not finite")
        numeric[key] = value
    return {**row, **numeric}


def error(actual: float, expected: float, kind: str) -> float:
    return abs(actual - expected) / abs(expected) if kind == "relative" else abs(actual - expected)


def main() -> int:
    if len(sys.argv) != 6:
        return 2
    try:
        reference = read_reference(Path(sys.argv[1]))
        values = {
            name: read_metrics(name, Path(path))
            for name, path in zip(IDENTITIES, sys.argv[2:])
        }
    except (OSError, KeyError, TypeError, ValueError) as exception:
        print(f"invalid Re20 result: {exception}", file=sys.stderr)
        return 2

    failures = []
    steady_limit = reference["steady_residual"][2]
    for name, row in values.items():
        if not math.isclose(float(row["steady_tolerance"]), steady_limit):
            failures.append(f"{name}: steady tolerance changed")

    finest = values["d2_nd80"]
    for metric in ("Cd_final", "Cl_final", "C_delta_p"):
        expected, kind, tolerance = reference[metric]
        observed = error(float(finest[metric]), expected, kind)
        if observed > tolerance:
            failures.append(f"N_D=80 {metric}={observed:.12g}>{tolerance:.12g}")

    medium = values["d2_nd40"]
    for metric, gate in (
        ("Cd_final", "Cd_resolution_difference"),
        ("Cl_final", "Cl_resolution_difference"),
    ):
        _, kind, tolerance = reference[gate]
        observed = error(float(medium[metric]), float(finest[metric]), kind)
        if observed > tolerance:
            failures.append(f"N_D=40/80 {metric}={observed:.12g}>{tolerance:.12g}")

    d2 = values["d2_nd20"]
    d3 = values["d3_nd20"]
    for metric in ("Cd_final", "Cl_final", "C_delta_p"):
        _, kind, tolerance = reference[metric]
        observed = error(float(d3[metric]), float(d2[metric]), kind)
        if observed > tolerance:
            failures.append(f"D3/D2 {metric}={observed:.12g}>{tolerance:.12g}")
    if float(d3["max_abs_Cz"]) > 1.0e-10:
        failures.append(f"D3 max_abs_Cz={float(d3['max_abs_Cz']):.12g}>1e-10")

    print(
        "Re20 metrics: "
        f"ND20=({float(d2['Cd_final']):.12g},{float(d2['Cl_final']):.12g},"
        f"{float(d2['C_delta_p']):.12g}) "
        f"ND40=({float(medium['Cd_final']):.12g},{float(medium['Cl_final']):.12g},"
        f"{float(medium['C_delta_p']):.12g}) "
        f"ND80=({float(finest['Cd_final']):.12g},{float(finest['Cl_final']):.12g},"
        f"{float(finest['C_delta_p']):.12g})"
    )
    if failures:
        print("Re20 numerical-fail: " + "; ".join(failures))
        return 1
    print("Re20 pass: reference, refinement, steady and quasi-D3 gates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
