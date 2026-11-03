#!/usr/bin/env python3
"""Synthetic positive and negative fixtures for the uniform-flow qualifier."""

from __future__ import annotations

import csv
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import check_uniform_flow as qualifier

CHECKER = Path(__file__).with_name("check_uniform_flow.py").resolve()


def _format_tuple(values: tuple[float, ...]) -> str:
    return "(" + ", ".join(f"{value:.12g}" for value in values) + ")"


def _direct_command(
    identity_name: str,
    identity: dict[str, object],
    *,
    ranks: int,
    threads: int,
    output_directory: str,
    log_directory: str,
) -> list[str]:
    command = [
        "env",
        "TMPDIR=/tmp",
        "TEMP=/tmp",
        "TMP=/tmp",
        "OMPI_MCA_orte_tmpdir_base=/tmp",
        f"OMP_NUM_THREADS={threads}",
        "OMP_DYNAMIC=FALSE",
        "OMP_PROC_BIND=spread",
        "OMP_PLACES=cores",
    ]
    if ranks > 1:
        mapping = "core" if threads == 1 else f"ppr:{ranks}:node:PE={threads}"
        command.extend(
            (
                "mpirun",
                "-np",
                str(ranks),
                "--map-by",
                mapping,
                "--bind-to",
                "core",
                "--rank-by",
                "core",
                "--report-bindings",
            )
        )
    command.extend(
        (
            str(identity["target"]),
            str(identity["input"]),
            f"uniform_flow.depth_cells={identity['depth']}",
            "run.max_step=10000",
            "diagnostics.interval=10000",
            "uniform_validation.write_metrics=1",
            "uniform_validation.assert=0",
            "uniform_validation.min_flow_throughs=100",
            "run.kind=long-run",
            f"run.variant={identity_name}",
            "output.plot_interval=0",
            "output.write_final_plotfile=0",
            f"output.directory={output_directory}",
            f"output.log_directory={log_directory}",
            f"amrex.omp_threads={threads}",
        )
    )
    return command


def write_artifact(
    path: Path,
    identity: dict[str, object],
    *,
    updates: dict[str, object] | None = None,
    columns: tuple[str, ...] | None = None,
    rows: int = 1,
    extra_value: bool = False,
) -> None:
    raw_reference = float(identity["raw_mass"])
    weighted_reference = float(identity["weighted_mass"])
    values: dict[str, object] = {
        "step": 10000,
        "flow_throughs": 100.0,
        "E_2_u": 2.0e-12,
        "E_inf_u": 3.0e-12,
        "mean_ux": 0.1,
        "std_ux_over_u0": 2.0e-12,
        "max_abs_uy_over_u0": 3.0e-12,
        "max_abs_uz_over_u0": 4.0e-12,
        "relative_mass_error": 5.0e-12,
        "rho_mean_minus_rho0": 6.0e-13,
        "rho_std_over_rho0": 7.0e-13,
        # Deliberately not bitwise-equal to the reference masses.
        "raw_lattice_mass": raw_reference * (1.0 + 5.0e-12),
        "raw_lattice_reference_mass": raw_reference,
        "physical_weighted_mass": weighted_reference * (1.0 + 5.0e-12),
        "physical_weighted_reference_mass": weighted_reference,
        "physical_weighted_relative_mass_error": 5.0e-12,
    }
    if updates:
        values.update(updates)
    header = columns or qualifier.EXPECTED_COLUMNS
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        row = [values.get(field, 0.0) for field in header]
        for _ in range(rows):
            writer.writerow((*row, "unexpected") if extra_value else row)


def write_log(
    path: Path,
    *,
    identity_name: str,
    workdir: Path,
    output_directory: str,
    ranks: int,
    threads: int,
) -> None:
    identity = qualifier.IDENTITIES[identity_name]
    log_directory = str(path.parent.relative_to(workdir))
    metrics = f"{output_directory}/uniform_flow_metrics.csv"
    command = _direct_command(
        identity_name,
        identity,
        ranks=ranks,
        threads=threads,
        output_directory=output_directory,
        log_directory=log_directory,
    )
    lines = (
        (
            f"runtime_context: mpi_ranks={ranks} OMP_NUM_THREADS={threads} "
            "OMP_DYNAMIC=FALSE OMP_PROC_BIND=spread OMP_PLACES=cores "
            f"amrex.omp_threads={threads}"
        ),
        f"working_directory = {workdir.resolve()}",
        f"command: {shlex.join(command)}",
        f"descriptor = {identity['descriptor']}",
        "case_name = uniform_flow",
        f"case_variant = {identity_name}",
        f"condition = {identity['condition']}",
        f"scheme = {identity['scheme']}",
        "profile = uniform",
        "location_source = xlo/xhi case composition",
        f"valid_nodes = {_format_tuple(identity['valid_nodes'])}",
        f"AMReX envelope prob_lo = {_format_tuple(identity['prob_lo'])}",
        f"AMReX envelope prob_hi = {_format_tuple(identity['prob_hi'])}",
        "physical_intervals_x = 10",
        "rho_ref_lb = 1",
        "u_ref_lb = 0.1",
        "nu_lb = 0.083333333333333333",
        "tau = 0.75",
        "omega = 1.3333333333333333",
        "max_step = 10000",
        "diagnostics_interval = 10000",
        "plot_interval = 0",
        f"output_directory = {output_directory}",
        f"terminal_log_directory = {log_directory}",
        (
            "uniform_flow diagnostics: write_uniform_flow_metrics = 1, "
            "uniform_flow_assert = 0, min_flow_throughs = 100, "
            "E_2_u_tol = 1e-08, E_inf_u_tol = 1e-07, "
            "std_ux/u0_tol = 1e-08, transverse_u/u0_tol = 1e-08, "
            "relative_mass_tol(raw_lattice) = 1e-08, "
            f"metrics_csv = {metrics}"
        ),
        f"uniform_flow_metrics_csv = {metrics}",
        "final_diagnostic_step = 10000",
        "convergence_row_written = false",
        (
            "LBM run complete: steps=10000, "
            f"descriptor={identity['descriptor']}, omega=1.33333333333"
        ),
        "",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def make_fixture(
    root: Path,
    identity_name: str,
    *,
    ranks: int = 1,
    threads: int = 1,
    artifact_updates: dict[str, object] | None = None,
    columns: tuple[str, ...] | None = None,
    rows: int = 1,
    extra_value: bool = False,
) -> tuple[Path, Path, Path]:
    identity = qualifier.IDENTITIES[identity_name]
    workdir = root / "case"
    target = workdir / str(identity["target"])
    control = workdir / str(identity["input"])
    target.parent.mkdir(parents=True, exist_ok=True)
    control.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("synthetic direct executable\n", encoding="utf-8")
    control.write_text("synthetic case-local control\n", encoding="utf-8")

    output_directory = f"outputs/baseline/{identity_name}/attempt-synthetic"
    artifact = workdir / output_directory / "uniform_flow_metrics.csv"
    log = (
        workdir
        / "log"
        / "baseline"
        / identity_name
        / "attempt-synthetic"
        / "console.log"
    )
    write_artifact(
        artifact,
        identity,
        updates=artifact_updates,
        columns=columns,
        rows=rows,
        extra_value=extra_value,
    )
    write_log(
        log,
        identity_name=identity_name,
        workdir=workdir,
        output_directory=output_directory,
        ranks=ranks,
        threads=threads,
    )
    return workdir, artifact, log


def _replace(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise AssertionError(f"fixture mutation did not match {old!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def expect_exit(
    root: Path,
    label: str,
    identity_name: str,
    expected: int,
    *,
    ranks: int = 1,
    threads: int = 1,
    artifact_updates: dict[str, object] | None = None,
    columns: tuple[str, ...] | None = None,
    rows: int = 1,
    extra_value: bool = False,
    log_replacements: tuple[tuple[str, str], ...] = (),
) -> None:
    workdir, artifact, log = make_fixture(
        root / label,
        identity_name,
        ranks=ranks,
        threads=threads,
        artifact_updates=artifact_updates,
        columns=columns,
        rows=rows,
        extra_value=extra_value,
    )
    for old, new in log_replacements:
        _replace(log, old, new)
    result = subprocess.run(
        (
            sys.executable,
            "-I",
            "-B",
            str(CHECKER),
            "--identity",
            identity_name,
            "--artifact",
            str(artifact),
            "--log",
            str(log),
            "--workdir",
            str(workdir),
        ),
        cwd=workdir,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != expected:
        raise AssertionError(
            f"{label}: expected exit {expected}, got {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="uniform-qualifier-") as tmp:
        root = Path(tmp)
        for identity_name in qualifier.IDENTITIES:
            expect_exit(
                root,
                "pass-" + identity_name,
                identity_name,
                0,
                ranks=2 if identity_name.endswith("d3q27") else 1,
            )

        base = "uniform.hbb-velocity-io.d2q9"
        d3 = "uniform.guo-free-outflow.d3q27"
        expect_exit(root, "scientific-e2", base, 1, artifact_updates={"E_2_u": 2e-8})
        expect_exit(
            root,
            "scientific-mass",
            base,
            1,
            artifact_updates={"relative_mass_error": 2e-8},
        )
        expect_exit(
            root,
            "scientific-d3-transverse",
            d3,
            1,
            ranks=2,
            artifact_updates={"max_abs_uz_over_u0": 2e-8},
        )

        expect_exit(root, "short-step", base, 2, artifact_updates={"step": 20})
        expect_exit(
            root,
            "short-flow-through",
            base,
            2,
            artifact_updates={"flow_throughs": 0.2},
        )
        expect_exit(
            root,
            "wrong-reference-mass",
            "uniform.guo-velocity-io.d2q9",
            2,
            artifact_updates={"physical_weighted_reference_mass": 110.0},
        )
        expect_exit(root, "nonfinite", base, 2, artifact_updates={"E_2_u": "nan"})
        expect_exit(root, "negative-error", base, 2, artifact_updates={"E_2_u": -1e-12})
        expect_exit(root, "zero-rows", base, 2, rows=0)
        expect_exit(root, "two-rows", base, 2, rows=2)
        expect_exit(root, "extra-row-value", base, 2, extra_value=True)
        expect_exit(
            root,
            "missing-column",
            base,
            2,
            columns=qualifier.EXPECTED_COLUMNS[:-1],
        )
        expect_exit(
            root,
            "extra-column",
            base,
            2,
            columns=(*qualifier.EXPECTED_COLUMNS, "unexpected"),
        )
        expect_exit(
            root,
            "reordered-columns",
            base,
            2,
            columns=(
                qualifier.EXPECTED_COLUMNS[1],
                qualifier.EXPECTED_COLUMNS[0],
                *qualifier.EXPECTED_COLUMNS[2:],
            ),
        )
        expect_exit(
            root,
            "wrong-variant",
            base,
            2,
            log_replacements=((f"case_variant = {base}", "case_variant = smoke"),),
        )
        expect_exit(
            root,
            "wrong-grid",
            base,
            2,
            log_replacements=(("valid_nodes = (10, 10, 1)", "valid_nodes = (11, 10, 1)"),),
        )
        expect_exit(
            root,
            "wrong-envelope",
            base,
            2,
            log_replacements=(("AMReX envelope prob_lo = (0, 0, 0)", "AMReX envelope prob_lo = (-0.05, 0, 0)"),),
        )
        expect_exit(
            root,
            "wrong-target",
            base,
            2,
            log_replacements=(("./build/lbm_uniform_flow ", "./build/lbm_invalid_target "),),
        )
        expect_exit(
            root,
            "root-legacy-input",
            base,
            2,
            log_replacements=(("inputs/inputs.d2q9_hbb_velocity_io ", "../../inputs/inputs.uniform "),),
        )
        expect_exit(
            root,
            "noncanonical-launcher",
            d3,
            2,
            ranks=2,
            log_replacements=(("--report-bindings ", "--report-bindings --nooversubscribe "),),
        )
        expect_exit(
            root,
            "thread-mismatch",
            base,
            2,
            log_replacements=(("amrex.omp_threads=1\n", "amrex.omp_threads=2\n"),),
        )
        expect_exit(
            root,
            "wrong-output-owner",
            base,
            2,
            log_replacements=(
                (
                    f"output_directory = outputs/baseline/{base}/attempt-synthetic",
                    "output_directory = outputs/wrong",
                ),
            ),
        )
        expect_exit(
            root,
            "completion-mismatch",
            base,
            2,
            log_replacements=(("steps=10000, descriptor=D2Q9", "steps=9999, descriptor=D2Q9"),),
        )

    print("uniform-flow baseline qualifier synthetic checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
