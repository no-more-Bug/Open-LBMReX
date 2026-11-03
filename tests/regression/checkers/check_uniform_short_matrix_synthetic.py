#!/usr/bin/env python3
"""Prove that malformed uniform-flow schemas cannot satisfy the formal checker."""

from __future__ import annotations

import csv
from pathlib import Path
import subprocess
import sys
import tempfile


HEADER = [
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


def row(step: int, flow: float, raw_mass: float, weighted_mass: float) -> list[float]:
    return [
        step,
        flow,
        0.0,
        0.0,
        0.1,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        raw_mass,
        raw_mass,
        weighted_mass,
        weighted_mass,
        0.0,
    ]


def write_metrics(directory: Path, header: list[str], values: list[float]) -> None:
    directory.mkdir()
    with (directory / "uniform_flow_metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerow(values)


def run_invalid_case(
    checker: Path,
    root: Path,
    name: str,
    invalid_header: list[str],
    invalid_row: list[float],
    expected_message: str,
) -> None:
    case = root / name
    case.mkdir()
    write_metrics(case / "hbb", invalid_header, invalid_row)
    write_metrics(case / "velocity_d2", HEADER, row(20, 0.2, 110.0, 100.0))
    write_metrics(case / "free_d2", HEADER, row(20, 0.2, 110.0, 100.0))
    write_metrics(case / "velocity_d3", HEADER, row(20, 0.2, 440.0, 400.0))
    write_metrics(case / "free_d3", HEADER, row(20, 0.2, 440.0, 400.0))
    completed = subprocess.run(
        (
            sys.executable,
            str(checker),
            "--hbb",
            str(case / "hbb"),
            "--guo-velocity-d2q9",
            str(case / "velocity_d2"),
            "--guo-free-d2q9",
            str(case / "free_d2"),
            "--guo-velocity-d3q27",
            str(case / "velocity_d3"),
            "--guo-free-d3q27",
            str(case / "free_d3"),
        ),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 1 or expected_message not in completed.stdout:
        raise RuntimeError(
            f"{name}: checker did not reject malformed schema as expected:\n"
            f"{completed.stdout}"
        )


def main() -> int:
    checker = Path(__file__).with_name("check_uniform_short_matrix.py")
    hbb = row(2, 0.02, 100.0, 100.0)
    with tempfile.TemporaryDirectory(prefix="uniform_checker_failures_") as temporary:
        root = Path(temporary)
        run_invalid_case(
            checker,
            root,
            "duplicate_header",
            HEADER + ["mean_ux"],
            hbb + [0.1],
            "empty or duplicate column name",
        )
        missing_index = HEADER.index("rho_std_over_rho0")
        run_invalid_case(
            checker,
            root,
            "missing_header",
            HEADER[:missing_index] + HEADER[missing_index + 1 :],
            hbb[:missing_index] + hbb[missing_index + 1 :],
            "header mismatch",
        )
        run_invalid_case(
            checker,
            root,
            "extra_header",
            HEADER + ["unexpected_metric"],
            hbb + [0.0],
            "header mismatch",
        )
        run_invalid_case(
            checker,
            root,
            "row_width",
            HEADER,
            hbb + [0.0],
            "row width mismatch",
        )
    print(
        "uniform checker rejects duplicate, missing, extra, and width-mismatched schemas"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
