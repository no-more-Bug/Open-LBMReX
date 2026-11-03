#!/usr/bin/env python3
"""Validate a real Re100 short artifact and its insufficient-cycle result."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


REFERENCE_LABEL = "dfg_2d_2_re100"
EXPECTED_STEPS = 40
EXPECTED_U_MEAN_LB = 0.035
EXPECTED_DIAMETER_LB = 40
LONG_ANALYSIS_WARMUP_T_STAR = 250.0
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
        raise SystemExit(f"missing artifact: {path}")
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
    except ValueError as error:
        raise SystemExit(f"{label}: expected numeric value") from error
    if not math.isfinite(result):
        raise SystemExit(f"{label}: expected finite value")
    return result


def integer(value: str, label: str) -> int:
    result = finite(value, label)
    if not result.is_integer():
        raise SystemExit(f"{label}: expected integer, got {result}")
    return int(result)


def validate_history(path: Path) -> float:
    rows = read_rows(path, HISTORY_HEADER)
    if len(rows) != EXPECTED_STEPS:
        raise SystemExit(f"{path}: expected {EXPECTED_STEPS} rows, got {len(rows)}")
    previous_t_star = -math.inf
    for expected_step, row in enumerate(rows, start=1):
        if (
            integer(row["population_step"], "history population_step")
            != expected_step
            or row["reference_label"] != REFERENCE_LABEL
            or row["object_name"] != "cylinder"
            or integer(row["object_id"], "history object_id") != 0
            or not math.isclose(
                finite(row["force_time_lb"], "history force_time_lb"),
                expected_step - 0.5,
                rel_tol=0.0,
                abs_tol=1.0e-14,
            )
        ):
            raise SystemExit(f"{path}: identity, cadence or final drain changed")
        t_star = finite(row["t_star"], "history t_star")
        expected_t_star = (
            (expected_step - 0.5) * EXPECTED_U_MEAN_LB / EXPECTED_DIAMETER_LB
        )
        if not math.isclose(
            t_star, expected_t_star, rel_tol=0.0, abs_tol=1.0e-14
        ):
            raise SystemExit(f"{path}: t_star physical scaling changed")
        if t_star <= previous_t_star:
            raise SystemExit(f"{path}: t_star must increase")
        previous_t_star = t_star
        for column in ("Fx_lb", "Fy_lb", "Fz_lb", "Cd", "Cl"):
            finite(row[column], f"history {column}")
    return previous_t_star


def validate_metrics(path: Path) -> None:
    rows = read_rows(path, METRICS_HEADER)
    if len(rows) != 1:
        raise SystemExit(f"{path}: expected one metrics row")
    row = rows[0]
    if (
        row["reference_label"] != REFERENCE_LABEL
        or row["object_name"] != "cylinder"
        or integer(row["object_id"], "metrics object_id") != 0
        or integer(row["sample_count"], "metrics sample_count") != EXPECTED_STEPS
        or integer(row["final_population_step"], "metrics final step")
        != EXPECTED_STEPS
        or row["gate_status"] != "assert_disabled"
    ):
        raise SystemExit(f"{path}: identity, sample count or final drain changed")
    for column in METRICS_HEADER:
        if column not in {"reference_label", "object_name", "gate_status"}:
            finite(row[column], f"metrics {column}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args()

    history = args.artifacts / "channel_cylinder_force_history.csv"
    metrics = args.artifacts / "channel_cylinder_metrics.csv"
    final_t_star = validate_history(history)
    validate_metrics(metrics)
    if final_t_star >= LONG_ANALYSIS_WARMUP_T_STAR:
        raise SystemExit("short history unexpectedly reaches long-analysis warmup")
    print(
        "channel_re100_short_artifact passed steps=40 final_drain=40 "
        f"final_t_star={final_t_star:.12g} long_acceptance=NOT_RUN"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
