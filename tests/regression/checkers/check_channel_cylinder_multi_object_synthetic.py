#!/usr/bin/env python3
"""Prove that invalid signal or aggregation cannot satisfy the checker."""

from __future__ import annotations

import csv
from pathlib import Path
import subprocess
import sys
import tempfile


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


def write_csv(path: Path, fields: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)


def history_row(name: str, object_id: int, fx: float, cd: float) -> list[object]:
    return [
        1, 0.5, 0.1, "tandem_two_cylinder", name, object_id,
        fx, 0.0, 0.0, cd, 0.0,
    ]


def metrics_row(
    name: str,
    object_id: int,
    cd: float,
    rho_front: float,
    rho_rear: float,
    delta_p: float,
) -> list[object]:
    return [
        "tandem_two_cylinder", name, object_id, 1, 1,
        cd, 0.0, cd, cd, cd, 0.0,
        0.0, 0.0, 0.0, 0.0,
        rho_front, rho_rear, delta_p, delta_p, "assert_disabled",
    ]


def run_failure_case(
    checker: Path,
    root: Path,
    name: str,
    force_signal: bool,
    positive_density: bool,
    pressure_signal: bool,
    expected_message: str,
    wrong_aggregation: bool = False,
) -> None:
    case = root / name
    case.mkdir()
    front_cd, rear_cd = ((2.0, 1.0) if force_signal else (0.0, 0.0))
    front_fx, rear_fx = ((1.0, 0.5) if force_signal else (0.0, 0.0))
    rho_front, rho_rear = ((1.1, 0.9) if positive_density else (0.0, 0.0))
    delta_p = 0.1 if pressure_signal else 0.0
    shared_fx = front_fx + rear_fx
    shared_cd = front_cd + rear_cd
    if wrong_aggregation:
        shared_fx += 0.25
        shared_cd += 0.25

    distinct_history = case / "distinct_history.csv"
    shared_history = case / "shared_history.csv"
    distinct_metrics = case / "distinct_metrics.csv"
    shared_metrics = case / "shared_metrics.csv"
    write_csv(
        distinct_history,
        HISTORY_FIELDS,
        [
            history_row("front_cylinder", 0, front_fx, front_cd),
            history_row("rear_cylinder", 1, rear_fx, rear_cd),
        ],
    )
    write_csv(
        shared_history,
        HISTORY_FIELDS,
        [history_row("cylinder_pair", 0, shared_fx, shared_cd)],
    )
    write_csv(
        distinct_metrics,
        METRICS_FIELDS,
        [
            metrics_row("front_cylinder", 0, front_cd, rho_front, rho_rear, delta_p),
            metrics_row("rear_cylinder", 1, rear_cd, rho_front, rho_rear, delta_p),
        ],
    )
    write_csv(
        shared_metrics,
        METRICS_FIELDS,
        [metrics_row(
            "cylinder_pair", 0, shared_cd, rho_front, rho_rear, delta_p,
        )],
    )
    completed = subprocess.run(
        (
            sys.executable, str(checker),
            "--distinct-history", str(distinct_history),
            "--distinct-metrics", str(distinct_metrics),
            "--shared-history", str(shared_history),
            "--shared-metrics", str(shared_metrics),
            "--expected-final-step", "1",
        ),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 1 or expected_message not in completed.stdout:
        raise RuntimeError(
            f"{name}: tandem checker accepted an invalid smoke signal:\n"
            f"{completed.stdout}"
        )


def main() -> int:
    checker = Path(__file__).with_name("check_channel_cylinder_multi_object.py")
    with tempfile.TemporaryDirectory(prefix="tandem_checker_failures_") as temporary:
        root = Path(temporary)
        run_failure_case(
            checker, root, "zero_force", False, True, True,
            "no nonzero drag/streamwise-force signal",
        )
        run_failure_case(
            checker, root, "zero_density", True, False, True,
            "pressure probe densities must be positive",
        )
        run_failure_case(
            checker, root, "zero_pressure", True, True, False,
            "no nonzero pressure-difference signal",
        )
        run_failure_case(
            checker, root, "wrong_aggregation", True, True, True,
            "aggregation residual", wrong_aggregation=True,
        )
    print(
        "tandem checker rejects zero force/density/pressure and wrong aggregation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
