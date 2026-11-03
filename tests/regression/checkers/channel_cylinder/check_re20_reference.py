#!/usr/bin/env python3
"""Validate the frozen Re20 reference and short production artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path


REFERENCE_SHA256 = (
    "0a4cb839f008506e7541f76f0db23c7eaa5c44672f2347d09870118b46448043"
)
REFERENCE_HEADER = ["metric", "value", "tolerance_kind", "tolerance", "scope"]
REFERENCE_ROWS = [
    ["Cd_final", "5.57953523384", "relative", "0.01", "N_D=80"],
    ["Cl_final", "0.010618948146", "absolute", "0.002", "N_D=80"],
    ["C_delta_p", "2.93800417425", "relative", "0.03", "N_D=80"],
    ["steady_residual", "0", "upper", "5e-7", "all_runs"],
    ["Cd_resolution_difference", "0", "relative", "0.01", "N_D=40_vs_80"],
    ["Cl_resolution_difference", "0", "absolute", "0.002", "N_D=40_vs_80"],
]
DIAGNOSTICS_HEADER = [
    "step", "time_lb", "time_phys", "mass", "relative_mass_error",
    "l2_ux", "l2_uy", "l2_uz", "l2_u", "l2_u_exact",
    "relative_l2_u", "linf_u", "linf_uz", "max_u", "max_rho_error",
]
HISTORY_HEADER = [
    "population_step", "force_time_lb", "t_star", "reference_label",
    "object_name", "object_id", "Fx_lb", "Fy_lb", "Fz_lb", "Cd", "Cl",
]
METRICS_HEADER = [
    "reference_label", "object_name", "object_id", "sample_count",
    "final_population_step", "Cd_final", "Cl_final", "Cd_min", "Cd_max",
    "Cd_midrange", "Cd_peak_to_peak", "Cl_min", "Cl_max", "Cl_midrange",
    "Cl_peak_to_peak", "rho_front_lb", "rho_rear_lb", "delta_p_lb",
    "C_delta_p", "gate_status",
]


def read_rows(path: Path, expected_header: list[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise SystemExit(f"missing file: {path}")
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != expected_header:
            raise SystemExit(
                f"{path}: schema mismatch {reader.fieldnames!r} != {expected_header!r}"
            )
        rows = list(reader)
    if not rows:
        raise SystemExit(f"{path}: no data rows")
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise SystemExit(f"{path}: row width mismatch")
    return rows


def finite(value: str, label: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise SystemExit(f"{label}: expected a numeric value") from exc
    if not math.isfinite(result):
        raise SystemExit(f"{label}: expected a finite value")
    return result


def integer(value: str, label: str) -> int:
    result = finite(value, label)
    if not result.is_integer():
        raise SystemExit(f"{label}: expected an integer, got {result}")
    return int(result)


def validate_reference(path: Path) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != REFERENCE_SHA256:
        raise SystemExit(
            f"{path}: checksum changed {digest} != {REFERENCE_SHA256}"
        )
    with path.open(newline="") as stream:
        rows = list(csv.reader(stream))
    if rows != [REFERENCE_HEADER, *REFERENCE_ROWS]:
        raise SystemExit(f"{path}: frozen Re20 rows or ordering changed")
    for row in REFERENCE_ROWS:
        finite(row[1], f"reference {row[0]} value")
        tolerance = finite(row[3], f"reference {row[0]} tolerance")
        if tolerance <= 0.0:
            raise SystemExit(f"reference {row[0]} tolerance must be positive")


def validate_short_artifacts(directory: Path, expect_d3: bool) -> None:
    diagnostics = read_rows(directory / "diagnostics.csv", DIAGNOSTICS_HEADER)
    diagnostic_steps = [integer(row["step"], "diagnostic step") for row in diagnostics]
    if diagnostic_steps != [0, 12]:
        raise SystemExit(f"{directory}: expected diagnostic steps [0, 12]")
    for row in diagnostics:
        for column in DIAGNOSTICS_HEADER:
            finite(row[column], f"{directory} diagnostics {column}")

    history = read_rows(
        directory / "channel_cylinder_force_history.csv", HISTORY_HEADER
    )
    if len(history) != 12:
        raise SystemExit(f"{directory}: expected 12 force-history rows")
    for expected_step, row in enumerate(history, start=1):
        if (
            integer(row["population_step"], "history population_step") != expected_step
            or row["reference_label"] != "dfg_2d_1_re20"
            or row["object_name"] != "cylinder"
            or integer(row["object_id"], "history object_id") != 0
        ):
            raise SystemExit(f"{directory}: history identity/final drain changed")
        for column in ("force_time_lb", "t_star", "Fx_lb", "Fy_lb", "Fz_lb", "Cd", "Cl"):
            finite(row[column], f"{directory} history {column}")
    if expect_d3 and max(abs(float(row["Fz_lb"])) for row in history) > 1.0e-12:
        raise SystemExit(f"{directory}: D3 quasi-2D Fz symmetry changed")

    metrics = read_rows(
        directory / "channel_cylinder_metrics.csv", METRICS_HEADER
    )
    if len(metrics) != 1:
        raise SystemExit(f"{directory}: expected one metrics row")
    row = metrics[0]
    if (
        row["reference_label"] != "dfg_2d_1_re20"
        or row["object_name"] != "cylinder"
        or integer(row["object_id"], "metrics object_id") != 0
        or integer(row["sample_count"], "metrics sample_count") != 12
        or integer(row["final_population_step"], "metrics final step") != 12
        or row["gate_status"] != "assert_disabled"
    ):
        raise SystemExit(f"{directory}: metrics identity/final drain changed")
    for column in METRICS_HEADER:
        if column not in {"reference_label", "object_name", "gate_status"}:
            finite(row[column], f"{directory} metrics {column}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--d2", type=Path, required=True)
    parser.add_argument("--d3", type=Path, required=True)
    args = parser.parse_args()

    validate_reference(args.reference)
    validate_short_artifacts(args.d2, expect_d3=False)
    validate_short_artifacts(args.d3, expect_d3=True)
    print(
        "channel_re20_reference passed "
        f"sha256={REFERENCE_SHA256} short_steps=12 long_acceptance=NOT_RUN"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
