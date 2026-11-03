#!/usr/bin/env python3
"""Prove malformed Poiseuille schemas cannot satisfy the formal checker."""

from __future__ import annotations

import csv
from pathlib import Path
import subprocess
import sys
import tempfile

import check_poiseuille_short_matrix as checker


def write_rows(path: Path, header: list[str], rows: list[list[float]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def metric_row(on_lattice: int, pressure: int, prescribed: int, nz: int) -> list[float]:
    values = {name: 0.0 for name in checker.METRICS_COLUMNS}
    values.update(
        step=2.0,
        nx=float(41 if on_lattice else 40),
        ny=float(11 if on_lattice else 10),
        nz=float(nz),
        n_intervals_y=10.0,
        on_lattice_boundary_nodes=float(on_lattice),
        u_max_lb=0.1,
        pressure_driven=float(pressure),
        prescribed_state=float(prescribed),
        pressure_scaling_code=float(2 if pressure or prescribed else 0),
        re_effective=12.0,
        rho_in_target_lb=1.04 if pressure or prescribed else 0.0,
        rho_out_target_lb=0.96 if pressure or prescribed else 0.0,
        density_delta_target_lb=0.08 if pressure or prescribed else 0.0,
        density_delta_expected_lb=0.08 if pressure or prescribed else 0.0,
    )
    return [values[name] for name in checker.METRICS_COLUMNS]


def make_run(
    directory: Path,
    on_lattice: int,
    pressure: int,
    prescribed: int,
    nz: int,
) -> None:
    directory.mkdir(parents=True)
    diagnostics = [[0.0] * len(checker.DIAGNOSTICS_COLUMNS)]
    diagnostics[0][0] = 2.0
    write_rows(
        directory / "diagnostics.csv",
        checker.DIAGNOSTICS_COLUMNS,
        diagnostics,
    )
    ny = 11 if on_lattice else 10
    profile = []
    for j in range(ny):
        row = [0.0] * len(checker.PROFILE_COLUMNS)
        row[0] = float(j)
        row[2] = float(j) / float(ny - 1) if on_lattice else (j + 0.5) / ny
        row[-1] = float(on_lattice and j in (0, ny - 1))
        profile.append(row)
    write_rows(
        directory / "poiseuille_profile.csv",
        checker.PROFILE_COLUMNS,
        profile,
    )
    nx = 41 if on_lattice else 40
    density = []
    for i in range(nx):
        row = [0.0] * len(checker.DENSITY_COLUMNS)
        row[0] = float(i)
        row[2] = float(i) / float(nx - 1) if on_lattice else (i + 0.5) / nx
        density.append(row)
    write_rows(
        directory / "poiseuille_density_pressure.csv",
        checker.DENSITY_COLUMNS,
        density,
    )
    write_rows(
        directory / "poiseuille_profile_metrics.csv",
        checker.METRICS_COLUMNS,
        [metric_row(on_lattice, pressure, prescribed, nz)],
    )


def run_invalid(
    root: Path,
    label: str,
    header: list[str],
    row: list[float],
    expected: str,
) -> None:
    case = root / label
    make_run(case / "hbb", 0, 0, 0, 1)
    make_run(case / "velocity", 1, 0, 0, 1)
    make_run(case / "pressure", 1, 1, 0, 1)
    make_run(case / "prescribed", 1, 0, 1, 1)
    make_run(case / "d3", 1, 1, 0, 4)
    write_rows(
        case / "hbb" / "poiseuille_profile_metrics.csv",
        header,
        [row],
    )
    completed = subprocess.run(
        (
            sys.executable,
            str(Path(checker.__file__)),
            "--hbb-d2q9", str(case / "hbb"),
            "--guo-velocity-d2q9", str(case / "velocity"),
            "--guo-pressure-d2q9", str(case / "pressure"),
            "--guo-prescribed-d2q9", str(case / "prescribed"),
            "--representative-d3q27", str(case / "d3"),
        ),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 1 or expected not in completed.stdout:
        raise RuntimeError(
            f"{label}: checker did not reject malformed schema:\n"
            f"{completed.stdout}"
        )


def main() -> int:
    row = metric_row(0, 0, 0, 1)
    with tempfile.TemporaryDirectory(prefix="poiseuille_checker_") as temp:
        root = Path(temp)
        run_invalid(
            root,
            "duplicate",
            checker.METRICS_COLUMNS + ["step"],
            row + [2.0],
            "empty or duplicate column name",
        )
        index = checker.METRICS_COLUMNS.index("rho_std_over_rho0")
        run_invalid(
            root,
            "missing",
            checker.METRICS_COLUMNS[:index] + checker.METRICS_COLUMNS[index + 1 :],
            row[:index] + row[index + 1 :],
            "header mismatch",
        )
        run_invalid(
            root,
            "extra",
            checker.METRICS_COLUMNS + ["unexpected_metric"],
            row + [0.0],
            "header mismatch",
        )
        run_invalid(
            root,
            "width",
            checker.METRICS_COLUMNS,
            row + [0.0],
            "row 0 width mismatch",
        )
    print("Poiseuille checker rejects duplicate, missing, extra and width mismatch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
