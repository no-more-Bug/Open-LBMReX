#!/usr/bin/env python3
"""Validate direct single-cylinder artifacts and enabled CPU parity."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

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
TEXT_COLUMNS = {"reference_label", "object_name", "gate_status"}


def read_csv(path: Path, expected_header: list[str] | None = None) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise SystemExit(f"missing artifact: {path}")
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        header = reader.fieldnames or []
        if (
            not header
            or any(not column for column in header)
            or len(set(header)) != len(header)
        ):
            raise SystemExit(f"{path}: empty or duplicate column name")
        if expected_header is not None and header != expected_header:
            raise SystemExit(
                f"{path}: schema mismatch {header!r} != {expected_header!r}"
            )
        rows = list(reader)
    if not rows:
        raise SystemExit(f"{path}: no data rows")
    for row_index, row in enumerate(rows):
        if None in row or any(value is None for value in row.values()):
            raise SystemExit(f"{path}: row {row_index} width mismatch")
        for column, value in row.items():
            if column in TEXT_COLUMNS:
                if not value:
                    raise SystemExit(
                        f"{path}: row {row_index} empty {column}"
                    )
            else:
                try:
                    number = float(value)
                except ValueError as exc:
                    raise SystemExit(
                        f"{path}: row {row_index} non-numeric {column}"
                    ) from exc
                if not math.isfinite(number):
                    raise SystemExit(
                        f"{path}: row {row_index} non-finite {column}"
                    )
    return header, rows


def integer(row: dict[str, str], column: str) -> int:
    value = float(row[column])
    if not value.is_integer():
        raise SystemExit(f"{column}: expected integer, got {value}")
    return int(value)


def validate_run(
    directory: Path,
    expected_steps: int,
    expect_history: bool,
) -> tuple[
    tuple[list[str], list[dict[str, str]]],
    tuple[list[str], list[dict[str, str]]] | None,
    tuple[list[str], list[dict[str, str]]],
]:
    diagnostics = read_csv(directory / "diagnostics.csv")
    history_path = directory / "channel_cylinder_force_history.csv"
    history = None
    if expect_history:
        history = read_csv(history_path, HISTORY_HEADER)
        history_rows = history[1]
        if len(history_rows) != expected_steps:
            raise SystemExit(
                f"{history_path}: expected {expected_steps} rows, "
                f"got {len(history_rows)}"
            )
        steps = [integer(row, "population_step") for row in history_rows]
        if steps != list(range(1, expected_steps + 1)):
            raise SystemExit(f"{history_path}: force/final-drain steps changed")
        for step, row in zip(steps, history_rows):
            if (
                row["reference_label"] != "dfg_2d_1_re20"
                or row["object_name"] != "cylinder"
                or integer(row, "object_id") != 0
                or not math.isclose(
                    float(row["force_time_lb"]),
                    step - 0.5,
                    rel_tol=0.0,
                    abs_tol=1.0e-14,
                )
            ):
                raise SystemExit(f"{history_path}: row identity changed")
    elif history_path.exists():
        raise SystemExit(f"{history_path}: history must be absent")

    metrics = read_csv(
        directory / "channel_cylinder_metrics.csv", METRICS_HEADER
    )
    metrics_rows = metrics[1]
    if len(metrics_rows) != 1:
        raise SystemExit(f"{directory}: expected one metrics row")
    row = metrics_rows[0]
    if (
        row["reference_label"] != "dfg_2d_1_re20"
        or row["object_name"] != "cylinder"
        or integer(row, "object_id") != 0
        or integer(row, "sample_count") != expected_steps
        or integer(row, "final_population_step") != expected_steps
        or row["gate_status"] != "assert_disabled"
    ):
        raise SystemExit(f"{directory}: metrics identity/final drain changed")
    return diagnostics, history, metrics


def compare_csv(
    reference: tuple[list[str], list[dict[str, str]]],
    actual: tuple[list[str], list[dict[str, str]]],
    label: str,
) -> float:
    reference_header, reference_rows = reference
    header, rows = actual
    if header != reference_header or len(rows) != len(reference_rows):
        raise SystemExit(f"{label}: schema/row-count parity mismatch")
    max_difference = 0.0
    for row_index, (expected, observed) in enumerate(
        zip(reference_rows, rows)
    ):
        for column in reference_header:
            if column in TEXT_COLUMNS:
                if observed[column] != expected[column]:
                    raise SystemExit(
                        f"{label}: row {row_index} {column} text mismatch"
                    )
                continue
            lhs = float(expected[column])
            rhs = float(observed[column])
            max_difference = max(max_difference, abs(rhs - lhs))
            if not math.isclose(
                rhs, lhs, rel_tol=1.0e-12, abs_tol=1.0e-12
            ):
                raise SystemExit(
                    f"{label}: row {row_index} {column} "
                    f"{rhs:.17g} != {lhs:.17g}"
                )
    return max_difference


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yms-d2", type=Path, required=True)
    parser.add_argument("--bouzidi-d2", type=Path, required=True)
    parser.add_argument("--yms-d3", type=Path, required=True)
    parser.add_argument("--parallel", type=Path, nargs="*", default=[])
    args = parser.parse_args()

    yms_d2 = validate_run(args.yms_d2, 12, True)
    validate_run(args.bouzidi_d2, 2, False)
    validate_run(args.yms_d3, 12, True)

    max_difference = 0.0
    for directory in args.parallel:
        parallel = validate_run(directory, 12, True)
        max_difference = max(
            max_difference,
            compare_csv(yms_d2[0], parallel[0], f"{directory} diagnostics"),
            compare_csv(yms_d2[1], parallel[1], f"{directory} history"),
            compare_csv(yms_d2[2], parallel[2], f"{directory} metrics"),
        )

    print(
        "channel_single_short_matrix passed "
        f"parallel_runs={len(args.parallel)} "
        f"max_abs_difference={max_difference:.12g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
