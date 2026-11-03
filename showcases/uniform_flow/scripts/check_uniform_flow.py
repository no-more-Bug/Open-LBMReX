#!/usr/bin/env python3
"""Qualify one approved 100-flow-through direct uniform-flow attempt."""

from __future__ import annotations

import argparse
import csv
import math
import re
import shlex
import sys
from pathlib import Path

EXPECTED_COLUMNS = (
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
)
DEFAULT_E2_U_TOL = 1.0e-8
DEFAULT_EINF_U_TOL = 1.0e-7
DEFAULT_STD_UX_TOL = 1.0e-8
DEFAULT_TRANSVERSE_TOL = 1.0e-8
DEFAULT_MASS_TOL = 1.0e-8
DEFAULT_EXPECTED_TOL = 1.0e-10


def _finite_float(raw: str, field: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"{field} is not a number: {raw!r}") from exc
    if not math.isfinite(value):
        raise ArtifactError(f"{field} must be finite")
    return value


def read_final_row(path: Path) -> dict[str, float]:
    try:
        stream = path.open(newline="", encoding="utf-8")
    except OSError as exc:
        raise ArtifactError(f"cannot read metrics CSV {path}: {exc}") from exc

    with stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != EXPECTED_COLUMNS:
            raise ArtifactError(
                f"{path}: metrics CSV schema changed; expected "
                + ",".join(EXPECTED_COLUMNS)
            )
        raw_rows = list(reader)

    if len(raw_rows) != 1:
        raise ArtifactError(f"{path}: metrics CSV must contain exactly one row")
    raw = raw_rows[0]
    if None in raw:
        raise ArtifactError(f"{path}: data row has unexpected extra values")
    if any(raw.get(field) in (None, "") for field in EXPECTED_COLUMNS):
        raise ArtifactError(f"{path}: empty required value")
    row = {
        field: _finite_float(raw[field], f"{path}:{field}")
        for field in EXPECTED_COLUMNS
    }
    if float(int(row["step"])) != row["step"] or row["step"] < 0.0:
        raise ArtifactError(f"{path}: step must be a nonnegative integer")
    for field in (
        "flow_throughs",
        "E_2_u",
        "E_inf_u",
        "std_ux_over_u0",
        "max_abs_uy_over_u0",
        "max_abs_uz_over_u0",
        "relative_mass_error",
        "rho_std_over_rho0",
        "raw_lattice_mass",
        "raw_lattice_reference_mass",
        "physical_weighted_mass",
        "physical_weighted_reference_mass",
        "physical_weighted_relative_mass_error",
    ):
        if row[field] < 0.0:
            raise ArtifactError(f"{path}: {field} must be nonnegative")
    if (
        row["raw_lattice_reference_mass"] <= 0.0
        or row["physical_weighted_reference_mass"] <= 0.0
    ):
        raise ArtifactError(f"{path}: mass references must be positive")
    return row


def _require_le(
    failures: list[str],
    row: dict[str, float],
    column: str,
    limit: float,
) -> None:
    if row[column] > limit:
        failures.append(f"{column}={row[column]:.17g} > {limit:.17g}")


def evaluate_metrics(row: dict[str, float]) -> list[str]:
    failures: list[str] = []
    _require_le(failures, row, "E_2_u", DEFAULT_E2_U_TOL)
    _require_le(failures, row, "E_inf_u", DEFAULT_EINF_U_TOL)
    _require_le(failures, row, "std_ux_over_u0", DEFAULT_STD_UX_TOL)
    _require_le(
        failures,
        row,
        "max_abs_uy_over_u0",
        DEFAULT_TRANSVERSE_TOL,
    )
    _require_le(
        failures,
        row,
        "max_abs_uz_over_u0",
        DEFAULT_TRANSVERSE_TOL,
    )
    _require_le(failures, row, "relative_mass_error", DEFAULT_MASS_TOL)
    _require_le(
        failures,
        row,
        "physical_weighted_relative_mass_error",
        DEFAULT_MASS_TOL,
    )
    return failures

EXPECTED_STEP = 10000
EXPECTED_FLOW_THROUGHS = 100.0
EXPECTED_RHO = 1.0
EXPECTED_U = 0.1
EXPECTED_NU = 1.0 / 12.0
EXPECTED_TAU = 0.75
EXPECTED_OMEGA = 4.0 / 3.0

IDENTITIES = {
    "uniform.hbb-velocity-io.d2q9": {
        "descriptor": "D2Q9",
        "depth": 1,
        "target": "./build/lbm_uniform_flow",
        "input": "inputs/inputs.d2q9_hbb_velocity_io",
        "condition": "velocity_inlet -> velocity_outlet",
        "scheme": "halfway_bounce_back",
        "valid_nodes": (10, 10, 1),
        "prob_lo": (0.0, 0.0, 0.0),
        "prob_hi": (1.0, 1.0, 0.1),
        "raw_mass": 100.0,
        "weighted_mass": 100.0,
    },
    "uniform.hbb-velocity-io.d3q27": {
        "descriptor": "D3Q27",
        "depth": 4,
        "target": "./build/lbm_uniform_flow_d3q27",
        "input": "inputs/inputs.d2q9_hbb_velocity_io",
        "condition": "velocity_inlet -> velocity_outlet",
        "scheme": "halfway_bounce_back",
        "valid_nodes": (10, 10, 4),
        "prob_lo": (0.0, 0.0, 0.0),
        "prob_hi": (1.0, 1.0, 0.4),
        "raw_mass": 400.0,
        "weighted_mass": 400.0,
    },
    "uniform.guo-velocity-io.d2q9": {
        "descriptor": "D2Q9",
        "depth": 1,
        "target": "./build/lbm_uniform_flow",
        "input": "inputs/inputs.d2q9_guo_velocity_io",
        "condition": "velocity_inlet -> velocity_outlet",
        "scheme": "guo_extrapolation",
        "valid_nodes": (11, 10, 1),
        "prob_lo": (-0.05, 0.0, 0.0),
        "prob_hi": (1.05, 1.0, 0.1),
        "raw_mass": 110.0,
        "weighted_mass": 100.0,
    },
    "uniform.guo-velocity-io.d3q27": {
        "descriptor": "D3Q27",
        "depth": 4,
        "target": "./build/lbm_uniform_flow_d3q27",
        "input": "inputs/inputs.d2q9_guo_velocity_io",
        "condition": "velocity_inlet -> velocity_outlet",
        "scheme": "guo_extrapolation",
        "valid_nodes": (11, 10, 4),
        "prob_lo": (-0.05, 0.0, 0.0),
        "prob_hi": (1.05, 1.0, 0.4),
        "raw_mass": 440.0,
        "weighted_mass": 400.0,
    },
    "uniform.guo-free-outflow.d2q9": {
        "descriptor": "D2Q9",
        "depth": 1,
        "target": "./build/lbm_uniform_flow",
        "input": "inputs/inputs.d2q9_guo_free_outflow",
        "condition": "velocity_inlet -> free_outflow",
        "scheme": "guo_extrapolation",
        "valid_nodes": (11, 10, 1),
        "prob_lo": (-0.05, 0.0, 0.0),
        "prob_hi": (1.05, 1.0, 0.1),
        "raw_mass": 110.0,
        "weighted_mass": 100.0,
    },
    "uniform.guo-free-outflow.d3q27": {
        "descriptor": "D3Q27",
        "depth": 4,
        "target": "./build/lbm_uniform_flow_d3q27",
        "input": "inputs/inputs.d2q9_guo_free_outflow",
        "condition": "velocity_inlet -> free_outflow",
        "scheme": "guo_extrapolation",
        "valid_nodes": (11, 10, 4),
        "prob_lo": (-0.05, 0.0, 0.0),
        "prob_hi": (1.05, 1.0, 0.4),
        "raw_mass": 440.0,
        "weighted_mass": 400.0,
    },
}


class ArtifactError(RuntimeError):
    """The observed files do not prove the declared long-run identity."""


def _close(actual: float, expected: float, *, absolute: float = 1.0e-12) -> bool:
    return math.isclose(actual, expected, rel_tol=1.0e-10, abs_tol=absolute)


def _resolve(raw: str | Path, workdir: Path) -> Path:
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (workdir / path).resolve()


def _require_under(path: Path, root: Path, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ArtifactError(f"{label} must stay under {root}, got {path}") from exc


def _single_value(lines: list[str], prefix: str, path: Path) -> str:
    values = [line[len(prefix):].strip() for line in lines if line.startswith(prefix)]
    if len(values) != 1:
        raise ArtifactError(
            f"{path}: expected exactly one {prefix.strip()!r} line, got {len(values)}"
        )
    return values[0]


def _float_value(lines: list[str], prefix: str, path: Path) -> float:
    raw = _single_value(lines, prefix, path)
    try:
        value = float(raw)
    except ValueError as exc:
        raise ArtifactError(f"{path}: {prefix.strip()} is not numeric") from exc
    if not math.isfinite(value):
        raise ArtifactError(f"{path}: {prefix.strip()} must be finite")
    return value


def _tuple_value(lines: list[str], prefix: str, path: Path) -> tuple[float, ...]:
    raw = _single_value(lines, prefix, path)
    match = re.fullmatch(r"\(([^()]*)\)", raw)
    if match is None:
        raise ArtifactError(f"{path}: malformed tuple for {prefix.strip()}")
    try:
        values = tuple(float(part.strip()) for part in match.group(1).split(","))
    except ValueError as exc:
        raise ArtifactError(f"{path}: nonnumeric tuple for {prefix.strip()}") from exc
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ArtifactError(f"{path}: {prefix.strip()} must contain three finite values")
    return values


def _read_log(
    path: Path,
) -> tuple[list[str], dict[str, str], list[str], tuple[int, str]]:
    try:
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    except OSError as exc:
        raise ArtifactError(f"cannot read raw log {path}: {exc}") from exc

    command_lines = [line for line in lines if line.startswith("command:")]
    if len(command_lines) != 1:
        raise ArtifactError(f"{path}: expected exactly one command line")
    try:
        command = shlex.split(command_lines[0].removeprefix("command:").strip())
    except ValueError as exc:
        raise ArtifactError(f"{path}: malformed command line: {exc}") from exc

    runtime_lines = [line for line in lines if line.startswith("runtime_context:")]
    if len(runtime_lines) != 1:
        raise ArtifactError(f"{path}: expected exactly one runtime_context")
    runtime_match = re.fullmatch(
        r"runtime_context: mpi_ranks=([0-9]+) "
        r"OMP_NUM_THREADS=([0-9]+) OMP_DYNAMIC=FALSE "
        r"OMP_PROC_BIND=(\S+) OMP_PLACES=(\S+) "
        r"amrex\.omp_threads=([0-9]+)",
        runtime_lines[0],
    )
    if runtime_match is None:
        raise ArtifactError(f"{path}: malformed runtime_context")
    runtime = dict(
        zip(
            ("ranks", "threads", "proc_bind", "places", "amrex_threads"),
            runtime_match.groups(),
        )
    )

    completion_lines = [
        line for line in lines if line.startswith("LBM run complete:")
    ]
    if len(completion_lines) != 1:
        raise ArtifactError(f"{path}: expected exactly one completion marker")
    completion_match = re.fullmatch(
        r"LBM run complete: steps=([0-9]+), descriptor=([A-Za-z0-9]+), .*",
        completion_lines[0],
    )
    if completion_match is None:
        raise ArtifactError(f"{path}: malformed completion marker")
    completion = (int(completion_match.group(1)), completion_match.group(2))
    return lines, runtime, command, completion


def _expected_command(
    identity_name: str,
    identity: dict[str, object],
    runtime: dict[str, str],
    output_directory: str,
    log_directory: str,
) -> list[str]:
    ranks = int(runtime["ranks"])
    threads = int(runtime["threads"])
    expected = [
        "env",
        "TMPDIR=/tmp",
        "TEMP=/tmp",
        "TMP=/tmp",
        "OMPI_MCA_orte_tmpdir_base=/tmp",
        f"OMP_NUM_THREADS={threads}",
        "OMP_DYNAMIC=FALSE",
        f"OMP_PROC_BIND={runtime['proc_bind']}",
        f"OMP_PLACES={runtime['places']}",
    ]
    if ranks > 1:
        mapping = "core" if threads == 1 else f"ppr:{ranks}:node:PE={threads}"
        expected.extend(
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
    expected.extend(
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
    return expected


def _check_observation_summary(
    lines: list[str],
    path: Path,
    artifact_raw: str,
) -> None:
    summary = _single_value(lines, "uniform_flow diagnostics:", path)
    expected_numbers = {
        "write_uniform_flow_metrics": 1.0,
        "uniform_flow_assert": 0.0,
        "min_flow_throughs": 100.0,
        "E_2_u_tol": DEFAULT_E2_U_TOL,
        "E_inf_u_tol": DEFAULT_EINF_U_TOL,
        "std_ux/u0_tol": DEFAULT_STD_UX_TOL,
        "transverse_u/u0_tol": DEFAULT_TRANSVERSE_TOL,
        "relative_mass_tol(raw_lattice)": DEFAULT_MASS_TOL,
    }
    for label, expected in expected_numbers.items():
        match = re.search(
            rf"(?:^|, )\s*{re.escape(label)} = ([^,]+)",
            summary,
        )
        if match is None:
            raise ArtifactError(f"{path}: missing {label} in observation summary")
        try:
            actual = float(match.group(1))
        except ValueError as exc:
            raise ArtifactError(f"{path}: nonnumeric {label}") from exc
        if not math.isfinite(actual) or not _close(actual, expected):
            raise ArtifactError(
                f"{path}: {label}={actual:.17g}, expected {expected:.17g}"
            )
    expected_metrics = f"metrics_csv = {artifact_raw}"
    if expected_metrics not in summary:
        raise ArtifactError(
            f"{path}: observation summary does not name {artifact_raw}"
        )


def qualify(
    *,
    identity_name: str,
    artifact_path: Path,
    log_path: Path,
    workdir: Path,
) -> tuple[dict[str, float], list[str]]:
    resolved_workdir = workdir.resolve()
    if not resolved_workdir.is_dir():
        raise ArtifactError(f"working directory does not exist: {resolved_workdir}")
    identity = IDENTITIES[identity_name]
    target = resolved_workdir / str(identity["target"])
    control = resolved_workdir / str(identity["input"])
    if not target.is_file():
        raise ArtifactError(f"direct executable does not exist: {target}")
    if not control.is_file():
        raise ArtifactError(f"approved input does not exist: {control}")

    artifact = _resolve(artifact_path, resolved_workdir)
    log = _resolve(log_path, resolved_workdir)
    _require_under(artifact, resolved_workdir / "outputs", "artifact")
    _require_under(log, resolved_workdir / "log", "raw log")

    row = read_final_row(artifact)
    lines, runtime, command, completion = _read_log(log)
    if int(runtime["ranks"]) <= 0 or int(runtime["threads"]) <= 0:
        raise ArtifactError(f"{log}: ranks and threads must be positive")
    if runtime["threads"] != runtime["amrex_threads"]:
        raise ArtifactError(f"{log}: AMReX and environment thread counts differ")

    expected_strings = {
        "descriptor = ": str(identity["descriptor"]),
        "case_name = ": "uniform_flow",
        "case_variant = ": identity_name,
        "condition = ": str(identity["condition"]),
        "scheme = ": str(identity["scheme"]),
        "profile = ": "uniform",
        "location_source = ": "xlo/xhi case composition",
        "physical_intervals_x = ": "10",
        "max_step = ": str(EXPECTED_STEP),
        "diagnostics_interval = ": str(EXPECTED_STEP),
        "plot_interval = ": "0",
        "convergence_row_written = ": "false",
        "final_diagnostic_step = ": str(EXPECTED_STEP),
    }
    for prefix, expected in expected_strings.items():
        actual = _single_value(lines, prefix, log)
        if actual != expected:
            raise ArtifactError(
                f"{log}: {prefix.strip()}={actual!r}, expected {expected!r}"
            )

    expected_numbers = {
        "rho_ref_lb = ": EXPECTED_RHO,
        "u_ref_lb = ": EXPECTED_U,
        "nu_lb = ": EXPECTED_NU,
        "tau = ": EXPECTED_TAU,
        "omega = ": EXPECTED_OMEGA,
    }
    for prefix, expected in expected_numbers.items():
        actual = _float_value(lines, prefix, log)
        if not _close(actual, expected):
            raise ArtifactError(
                f"{log}: {prefix.strip()}={actual:.17g}, expected {expected:.17g}"
            )

    for prefix, expected in (
        ("valid_nodes = ", identity["valid_nodes"]),
        ("AMReX envelope prob_lo = ", identity["prob_lo"]),
        ("AMReX envelope prob_hi = ", identity["prob_hi"]),
    ):
        actual = _tuple_value(lines, prefix, log)
        expected_tuple = tuple(float(value) for value in expected)
        if any(
            not _close(observed, wanted)
            for observed, wanted in zip(actual, expected_tuple)
        ):
            raise ArtifactError(f"{log}: {prefix.strip()}={actual}, expected {expected_tuple}")

    raw_workdir = _single_value(lines, "working_directory = ", log)
    if Path(raw_workdir).resolve() != resolved_workdir:
        raise ArtifactError(f"{log}: working_directory does not match case root")
    raw_output = _single_value(lines, "output_directory = ", log)
    raw_log_directory = _single_value(lines, "terminal_log_directory = ", log)
    if Path(raw_output).is_absolute() or Path(raw_log_directory).is_absolute():
        raise ArtifactError(f"{log}: output and log paths must be case-relative")
    if _resolve(raw_output, resolved_workdir) != artifact.parent:
        raise ArtifactError(f"{log}: output_directory does not own artifact")
    if _resolve(raw_log_directory, resolved_workdir) != log.parent:
        raise ArtifactError(f"{log}: terminal_log_directory does not own raw log")
    metrics_raw = _single_value(lines, "uniform_flow_metrics_csv = ", log)
    if _resolve(metrics_raw, resolved_workdir) != artifact:
        raise ArtifactError(f"{log}: metrics path does not match artifact")
    _check_observation_summary(lines, log, metrics_raw)

    expected_command = _expected_command(
        identity_name,
        identity,
        runtime,
        raw_output,
        raw_log_directory,
    )
    if command != expected_command:
        raise ArtifactError(f"{log}: command does not match approved direct command")
    if completion != (EXPECTED_STEP, str(identity["descriptor"])):
        raise ArtifactError(
            f"{log}: completion={completion}, expected "
            f"({EXPECTED_STEP}, {identity['descriptor']})"
        )

    if int(row["step"]) != EXPECTED_STEP:
        raise ArtifactError(
            f"{artifact}: step={row['step']:.17g}, expected {EXPECTED_STEP}"
        )
    if not math.isclose(
        row["flow_throughs"],
        EXPECTED_FLOW_THROUGHS,
        rel_tol=0.0,
        abs_tol=DEFAULT_EXPECTED_TOL,
    ):
        raise ArtifactError(f"{artifact}: flow_throughs must equal 100")

    for column, expected in (
        ("raw_lattice_reference_mass", float(identity["raw_mass"])),
        (
            "physical_weighted_reference_mass",
            float(identity["weighted_mass"]),
        ),
    ):
        if not _close(row[column], expected):
            raise ArtifactError(
                f"{artifact}: {column}={row[column]:.17g}, expected {expected:.17g}"
            )

    failures = evaluate_metrics(row)
    return row, failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--identity", choices=tuple(IDENTITIES), required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args()

    try:
        row, failures = qualify(
            identity_name=args.identity,
            artifact_path=args.artifact,
            log_path=args.log,
            workdir=args.workdir,
        )
    except (ArtifactError, csv.Error, OSError, ValueError) as exc:
        print(f"invalid artifact: {exc}", file=sys.stderr)
        return 2

    if failures:
        for failure in failures:
            print(f"scientific failure: {failure}", file=sys.stderr)
        return 1

    print(
        f"uniform baseline qualifier passed: identity={args.identity}, "
        f"step={int(row['step'])}, flow_throughs={row['flow_throughs']:.12g}, "
        f"E_2_u={row['E_2_u']:.12g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
