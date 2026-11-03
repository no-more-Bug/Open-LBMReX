#!/usr/bin/env python3
"""Check distinct/shared ObjectId outputs from a public short regression."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path


HISTORY_FIELDS = [
    "population_step", "force_time_lb", "t_star", "reference_label",
    "object_name", "object_id", "Fx_lb", "Fy_lb", "Fz_lb", "Cd", "Cl",
]
METRICS_FIELDS = [
    "reference_label", "object_name", "object_id", "sample_count",
    "final_population_step", "Cd_final", "Cl_final", "Cd_min", "Cd_max",
    "Cd_midrange", "Cd_peak_to_peak", "Cl_min", "Cl_max", "Cl_midrange",
    "Cl_peak_to_peak", "rho_front_lb", "rho_rear_lb", "delta_p_lb",
    "C_delta_p", "gate_status",
]
SUM_FIELDS = ["Fx_lb", "Fy_lb", "Fz_lb", "Cd", "Cl"]
TEXT_FIELDS = {"reference_label", "object_name", "gate_status"}
ABS_TOL = 2.0e-12
REL_TOL = 2.0e-11
SIGNAL_TOL = 1.0e-12


def read_csv(
    path: Path, expected_fields: list[str] | None = None
) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if not fields or len(set(fields)) != len(fields):
            raise ValueError(f"{path}: empty or duplicate CSV column")
        if expected_fields is not None and fields != expected_fields:
            raise ValueError(f"{path}: unexpected CSV header")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: contains no data rows")
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"{path}: CSV row width mismatch")
    return rows


def finite(row: dict[str, str], field: str) -> float:
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {field} in row {row}")
    return value


def integer(row: dict[str, str], field: str) -> int:
    return int(row[field])


def close(lhs: float, rhs: float) -> bool:
    return abs(lhs - rhs) <= ABS_TOL + REL_TOL * max(abs(lhs), abs(rhs))


def compare_csv(
    reference: list[dict[str, str]],
    actual: list[dict[str, str]],
    label: str,
) -> float:
    if len(actual) != len(reference):
        raise ValueError(f"{label}: CSV row-count mismatch")
    expected_fields = list(reference[0])
    if list(actual[0]) != expected_fields:
        raise ValueError(f"{label}: CSV schema mismatch")
    max_difference = 0.0
    for row_index, (expected, observed) in enumerate(zip(reference, actual)):
        if list(observed) != expected_fields:
            raise ValueError(f"{label}: row {row_index} schema mismatch")
        for field in expected_fields:
            if field in TEXT_FIELDS:
                if observed[field] != expected[field]:
                    raise ValueError(
                        f"{label}: row {row_index} {field} text mismatch")
                continue
            lhs = finite(expected, field)
            rhs = finite(observed, field)
            max_difference = max(max_difference, abs(rhs - lhs))
            if not close(lhs, rhs):
                raise ValueError(
                    f"{label}: row {row_index} {field} parity mismatch")
    return max_difference


def check_identity(
    row: dict[str, str], name: str, object_id: int, label: str
) -> None:
    if (row["reference_label"] != "tandem_two_cylinder" or
            row["object_name"] != name or integer(row, "object_id") != object_id):
        raise ValueError(f"{label}: unexpected object identity {row}")


def check_history(
    distinct: list[dict[str, str]],
    shared: list[dict[str, str]],
    expected_final_step: int,
) -> tuple[float, float]:
    if len(distinct) != 2 * len(shared):
        raise ValueError("distinct/shared history row counts must have ratio 2:1")
    steps: list[int] = []
    max_abs = 0.0
    max_rel = 0.0
    max_signal = 0.0
    for sample, pair in enumerate(shared):
        front = distinct[2 * sample]
        rear = distinct[2 * sample + 1]
        check_identity(front, "front_cylinder", 0, "distinct front")
        check_identity(rear, "rear_cylinder", 1, "distinct rear")
        check_identity(pair, "cylinder_pair", 0, "shared pair")
        step = integer(pair, "population_step")
        if step != integer(front, "population_step") or step != integer(rear, "population_step"):
            raise ValueError("distinct/shared population steps are not aligned")
        steps.append(step)
        for row in (front, rear, pair):
            if not close(finite(row, "force_time_lb"), step - 0.5):
                raise ValueError(f"step {step}: incorrect force time")
            for field in ("t_star", *SUM_FIELDS):
                finite(row, field)
            max_signal = max(
                max_signal,
                abs(finite(row, "Fx_lb")),
                abs(finite(row, "Cd")),
            )
        if not (close(finite(front, "t_star"), finite(pair, "t_star")) and
                close(finite(rear, "t_star"), finite(pair, "t_star"))):
            raise ValueError(f"step {step}: distinct/shared t_star mismatch")
        for field in SUM_FIELDS:
            expected = finite(front, field) + finite(rear, field)
            actual = finite(pair, field)
            residual = abs(actual - expected)
            scale = max(abs(actual), abs(expected))
            max_abs = max(max_abs, residual)
            if scale > ABS_TOL:
                max_rel = max(max_rel, residual / scale)
            if not close(actual, expected):
                raise ValueError(
                    f"step {step}: {field} aggregation residual={residual:.17g}")
    if (
        steps != list(range(1, len(steps) + 1))
        or steps[-1] != expected_final_step
    ):
        raise ValueError(
            "history must contain every force sample from step 1 through "
            f"{expected_final_step}"
        )
    if max_signal <= SIGNAL_TOL:
        raise ValueError("history has no nonzero drag/streamwise-force signal")
    return max_abs, max_rel


def check_metrics(
    rows: list[dict[str, str]],
    expected: list[tuple[str, int]],
    history: list[dict[str, str]],
    expected_final_step: int,
) -> None:
    if len(rows) != len(expected):
        raise ValueError("unexpected metrics row count")
    for row, (name, object_id) in zip(rows, expected):
        check_identity(row, name, object_id, "metrics")
        selected = [entry for entry in history
                    if entry["object_name"] == name and
                    integer(entry, "object_id") == object_id]
        if (integer(row, "sample_count") != len(selected) or
                integer(row, "final_population_step") != expected_final_step or
                row["gate_status"] != "assert_disabled"):
            raise ValueError(f"{name}: inconsistent metrics metadata")
        cd = [finite(entry, "Cd") for entry in selected]
        cl = [finite(entry, "Cl") for entry in selected]
        expected_values = {
            "Cd_final": cd[-1], "Cl_final": cl[-1],
            "Cd_min": min(cd), "Cd_max": max(cd),
            "Cd_midrange": 0.5 * (min(cd) + max(cd)),
            "Cd_peak_to_peak": max(cd) - min(cd),
            "Cl_min": min(cl), "Cl_max": max(cl),
            "Cl_midrange": 0.5 * (min(cl) + max(cl)),
            "Cl_peak_to_peak": max(cl) - min(cl),
        }
        for field, expected_value in expected_values.items():
            if not close(finite(row, field), expected_value):
                raise ValueError(f"{name}: metrics field {field} disagrees with history")
        for field in ("rho_front_lb", "rho_rear_lb", "delta_p_lb", "C_delta_p"):
            finite(row, field)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distinct-history", required=True, type=Path)
    parser.add_argument("--distinct-metrics", required=True, type=Path)
    parser.add_argument("--distinct-diagnostics", type=Path)
    parser.add_argument("--shared-history", required=True, type=Path)
    parser.add_argument("--shared-metrics", required=True, type=Path)
    parser.add_argument("--shared-diagnostics", type=Path)
    parser.add_argument("--distinct-parity", action="append", type=Path, default=[])
    parser.add_argument("--shared-parity", action="append", type=Path, default=[])
    parser.add_argument("--expected-final-step", required=True, type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    distinct_history = read_csv(args.distinct_history, HISTORY_FIELDS)
    shared_history = read_csv(args.shared_history, HISTORY_FIELDS)
    distinct_metrics = read_csv(args.distinct_metrics, METRICS_FIELDS)
    shared_metrics = read_csv(args.shared_metrics, METRICS_FIELDS)
    if args.expected_final_step <= 0:
        raise ValueError("--expected-final-step must be positive")
    if (args.distinct_diagnostics is None) != (args.shared_diagnostics is None):
        raise ValueError(
            "distinct/shared diagnostics must be provided together")
    if (args.distinct_parity or args.shared_parity) and args.distinct_diagnostics is None:
        raise ValueError("parity runs require primary diagnostics")
    max_abs, max_rel = check_history(
        distinct_history, shared_history, args.expected_final_step)
    check_metrics(
        distinct_metrics,
        [("front_cylinder", 0), ("rear_cylinder", 1)],
        distinct_history,
        args.expected_final_step)
    check_metrics(
        shared_metrics,
        [("cylinder_pair", 0)],
        shared_history,
        args.expected_final_step)
    metric_rows = (*distinct_metrics, *shared_metrics)
    if any(
        finite(row, "rho_front_lb") <= 0.0
        or finite(row, "rho_rear_lb") <= 0.0
        for row in metric_rows
    ):
        raise ValueError("pressure probe densities must be positive")
    if max(abs(finite(row, "delta_p_lb")) for row in metric_rows) <= SIGNAL_TOL:
        raise ValueError("metrics have no nonzero pressure-difference signal")
    pressure = [
        finite(row, field)
        for row in metric_rows
        for field in ("rho_front_lb", "rho_rear_lb", "delta_p_lb", "C_delta_p")
    ]
    for field_index in range(4):
        values = pressure[field_index::4]
        if not all(close(values[0], value) for value in values[1:]):
            raise ValueError("run-level pressure metrics differ between object rows or runs")

    max_parity_difference = 0.0
    if args.distinct_diagnostics is not None:
        distinct_diagnostics = read_csv(args.distinct_diagnostics)
        shared_diagnostics = read_csv(args.shared_diagnostics)
        max_parity_difference = compare_csv(
            distinct_diagnostics,
            shared_diagnostics,
            "distinct/shared diagnostics",
        )
        for variant, directories, diagnostics, history, metrics in (
            ("distinct", args.distinct_parity, distinct_diagnostics,
             distinct_history, distinct_metrics),
            ("shared", args.shared_parity, shared_diagnostics,
             shared_history, shared_metrics),
        ):
            for directory in directories:
                max_parity_difference = max(
                    max_parity_difference,
                    compare_csv(
                        diagnostics, read_csv(directory / "diagnostics.csv"),
                        f"{variant} diagnostics {directory}"),
                    compare_csv(
                        history, read_csv(
                            directory / "channel_cylinder_force_history.csv",
                            HISTORY_FIELDS),
                        f"{variant} history {directory}"),
                    compare_csv(
                        metrics, read_csv(
                            directory / "channel_cylinder_metrics.csv",
                            METRICS_FIELDS),
                        f"{variant} metrics {directory}"),
                )
    print(
        f"status=pass samples={args.expected_final_step} "
        f"distinct_history_rows={len(distinct_history)} "
        f"shared_history_rows={len(shared_history)} "
        f"max_abs_residual={max_abs:.17g} "
        f"max_rel_residual={max_rel:.17g} "
        f"parity_runs={len(args.distinct_parity) + len(args.shared_parity)} "
        f"max_parity_difference={max_parity_difference:.17g}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"status=fail reason={error}", file=sys.stderr)
        raise SystemExit(1)
