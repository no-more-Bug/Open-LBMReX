#!/usr/bin/env python3
"""Qualify one approved three-resolution Poiseuille acceptance group."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shlex
import sys
from pathlib import Path

EXPECTED_RESOLUTIONS = (10, 20, 50)
EXPECTED_METRICS_COLUMNS = (
    "step",
    "nx",
    "ny",
    "nz",
    "n_intervals_y",
    "on_lattice_boundary_nodes",
    "u_max_lb",
    "E_2_profile",
    "E_inf_profile",
    "E_2_profile_full",
    "E_inf_profile_full",
    "max_abs_uy_over_umax",
    "max_abs_uz_over_umax",
    "relative_mass_error",
    "raw_lattice_mass",
    "raw_lattice_reference_mass",
    "physical_weighted_mass",
    "physical_weighted_reference_mass",
    "physical_weighted_relative_mass_error",
    "rho_mean_minus_rho0",
    "rho_std_over_rho0",
    "rho_min",
    "rho_max",
    "rho_streamwise_xlo",
    "rho_streamwise_xhi",
    "pressure_streamwise_xlo",
    "pressure_streamwise_xhi",
    "rho_streamwise_slope",
    "pressure_streamwise_slope",
    "rho_streamwise_linear_r2",
    "pressure_driven",
    "prescribed_state",
    "pressure_scaling_code",
    "re_effective",
    "rho_in_target_lb",
    "rho_out_target_lb",
    "density_delta_target_lb",
    "density_delta_expected_lb",
    "density_delta_over_rho_ref",
    "mach",
    "max_abs_pressure_boundary_rho_error",
    "rho_streamwise_delta_measured",
    "rho_streamwise_delta_relative_error",
    "max_abs_profile_error_all_x_over_umax",
    "max_streamwise_ux_spread_over_umax",
)
EXPECTED_PROFILE_COLUMNS = (
    "j",
    "y_phys",
    "eta",
    "ux_exact",
    "ux_center_zmean",
    "uy_abs_max",
    "uz_abs_max",
    "sample_i",
    "is_physical_boundary_node",
)
EXPECTED_DENSITY_COLUMNS = (
    "i",
    "x_phys",
    "x_over_L",
    "rho_cross_section_weighted_mean",
    "pressure_lb_cross_section_weighted_mean",
)

EXPECTED_RE = 12.0
EXPECTED_TAU = 0.75
EXPECTED_NU = 1.0 / 12.0
EXPECTED_OMEGA = 4.0 / 3.0
STEADY_TOL = 5.0e-8
SLOPE_MIN = 1.8
SLOPE_MAX = 2.2
E2_TOL = 5.0e-2
EINF_TOL = 8.0e-2
TRANSVERSE_TOL = 1.0e-2
MASS_TOL = 1.0e-5
PRESSURE_RHO_TOL = 1.0e-10
PRESSURE_R2_MIN = 0.99

GROUPS = {
    "poiseuille.hbb-velocity.d2q9.slope": {
        "route": "hbb-velocity",
        "descriptor": "D2Q9",
        "depth": 1,
        "target": "./build/lbm_poiseuille_flow",
        "input": "inputs/inputs.d2q9_hbb_velocity_profile",
        "entry_prefix": "poiseuille.hbb-velocity.d2q9",
        "scheme": "halfway_bounce_back",
        "condition": "velocity_inlet -> velocity_outlet",
        "profile": "poiseuille",
        "guo": False,
        "pressure": False,
    },
    "poiseuille.hbb-velocity.d3q27.slope": {
        "route": "hbb-velocity",
        "descriptor": "D3Q27",
        "depth": 4,
        "target": "./build/lbm_poiseuille_flow_d3q27",
        "input": "inputs/inputs.d2q9_hbb_velocity_profile",
        "entry_prefix": "poiseuille.hbb-velocity.d3q27",
        "scheme": "halfway_bounce_back",
        "condition": "velocity_inlet -> velocity_outlet",
        "profile": "poiseuille",
        "guo": False,
        "pressure": False,
    },
    "poiseuille.guo-velocity.d2q9.slope": {
        "route": "guo-velocity",
        "descriptor": "D2Q9",
        "depth": 1,
        "target": "./build/lbm_poiseuille_flow",
        "input": "inputs/inputs.d2q9_guo_velocity",
        "entry_prefix": "poiseuille.guo-velocity.d2q9",
        "scheme": "guo_extrapolation",
        "condition": "velocity_inlet -> velocity_outlet",
        "profile": "poiseuille",
        "guo": True,
        "pressure": False,
    },
    "poiseuille.guo-velocity.d3q27.slope": {
        "route": "guo-velocity",
        "descriptor": "D3Q27",
        "depth": 4,
        "target": "./build/lbm_poiseuille_flow_d3q27",
        "input": "inputs/inputs.d2q9_guo_velocity",
        "entry_prefix": "poiseuille.guo-velocity.d3q27",
        "scheme": "guo_extrapolation",
        "condition": "velocity_inlet -> velocity_outlet",
        "profile": "poiseuille",
        "guo": True,
        "pressure": False,
    },
    "poiseuille.guo-pressure.d2q9.slope": {
        "route": "guo-pressure",
        "descriptor": "D2Q9",
        "depth": 1,
        "target": "./build/lbm_poiseuille_flow",
        "input": "inputs/inputs.d2q9_guo_pressure",
        "entry_prefix": "poiseuille.guo-pressure.d2q9",
        "scheme": "guo_extrapolation",
        "condition": "pressure_inlet -> pressure_outlet",
        "profile": "constant_density",
        "guo": True,
        "pressure": True,
    },
    "poiseuille.guo-pressure.d3q27.slope": {
        "route": "guo-pressure",
        "descriptor": "D3Q27",
        "depth": 4,
        "target": "./build/lbm_poiseuille_flow_d3q27",
        "input": "inputs/inputs.d2q9_guo_pressure",
        "entry_prefix": "poiseuille.guo-pressure.d3q27",
        "scheme": "guo_extrapolation",
        "condition": "pressure_inlet -> pressure_outlet",
        "profile": "constant_density",
        "guo": True,
        "pressure": True,
    },
}

ENTRY_DEFINITIONS = {
    f"{group['entry_prefix']}.n{n}": (group_id, n)
    for group_id, group in GROUPS.items()
    for n in EXPECTED_RESOLUTIONS
}


class EvidenceError(RuntimeError):
    """The supplied files do not prove the declared direct-run identity."""


def _finite(raw: object, field: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise EvidenceError(f"{field} is not a number: {raw!r}") from exc
    if not math.isfinite(value):
        raise EvidenceError(f"{field} must be finite")
    return value


def _close(actual: float, expected: float, *, absolute: float = 1.0e-12) -> bool:
    return math.isclose(actual, expected, rel_tol=1.0e-10, abs_tol=absolute)


def _integer(value: float, field: str) -> int:
    result = int(value)
    if float(result) != value:
        raise EvidenceError(f"{field} must be an integer")
    return result


def parse_specs(raw_specs: list[str], option: str) -> dict[int, Path]:
    paths: dict[int, Path] = {}
    for raw in raw_specs:
        resolution, separator, path = raw.partition("=")
        if not separator or not resolution.isdigit() or not path:
            raise EvidenceError(
                f"invalid {option} {raw!r}; expected RESOLUTION=PATH"
            )
        n = int(resolution)
        if n in paths:
            raise EvidenceError(f"duplicate {option} for N={n}")
        paths[n] = Path(path)
    if tuple(sorted(paths)) != EXPECTED_RESOLUTIONS:
        raise EvidenceError(f"{option} must name exactly N=10,20,50")
    return paths


def _read_csv(path: Path, columns: tuple[str, ...]) -> list[dict[str, float]]:
    try:
        stream = path.open(newline="", encoding="utf-8")
    except OSError as exc:
        raise EvidenceError(f"cannot read CSV {path}: {exc}") from exc
    with stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != columns:
            raise EvidenceError(
                f"{path}: CSV schema changed; expected " + ",".join(columns)
            )
        raw_rows = list(reader)
    rows: list[dict[str, float]] = []
    for row_index, raw in enumerate(raw_rows, start=1):
        if None in raw:
            raise EvidenceError(f"{path}: row {row_index} has extra values")
        if any(raw.get(field) in (None, "") for field in columns):
            raise EvidenceError(f"{path}: row {row_index} has an empty value")
        rows.append(
            {
                field: _finite(raw[field], f"{path}:row {row_index}:{field}")
                for field in columns
            }
        )
    return rows


def read_metrics(path: Path) -> dict[str, float]:
    rows = _read_csv(path, EXPECTED_METRICS_COLUMNS)
    if len(rows) != 1:
        raise EvidenceError(f"{path}: metrics CSV must contain exactly one row")
    return rows[0]


def _single_prefixed(lines: list[str], prefix: str, path: Path) -> str:
    values = [line[len(prefix):].strip() for line in lines if line.startswith(prefix)]
    if len(values) != 1:
        raise EvidenceError(
            f"{path}: expected exactly one {prefix.strip()!r} line, got {len(values)}"
        )
    return values[0]

def _resolve(path: Path | str, workdir: Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (workdir / value).resolve()


def _require_under(path: Path, root: Path, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise EvidenceError(f"{label} must stay under {root}, got {path}") from exc


def _relative(path: Path, root: Path, label: str) -> str:
    _require_under(path, root, label)
    return path.relative_to(root).as_posix()


def _parse_tuple(raw: str, field: str) -> tuple[float, ...]:
    match = re.fullmatch(r"\(([^()]*)\)", raw.strip())
    if not match:
        raise EvidenceError(f"{field} must be a parenthesized tuple")
    parts = [part.strip() for part in match.group(1).split(",")]
    if not parts or any(not part for part in parts):
        raise EvidenceError(f"{field} contains an empty item")
    return tuple(_finite(part, field) for part in parts)


def read_log(path: Path) -> dict[str, object]:
    try:
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    except OSError as exc:
        raise EvidenceError(f"cannot read raw log {path}: {exc}") from exc

    runtime: dict[str, str] | None = None
    command: list[str] | None = None
    metadata: dict[str, str] = {}
    completions: list[tuple[int, str]] = []
    steady_markers: list[tuple[int, float, int]] = []
    for line in lines:
        runtime_match = re.fullmatch(
            r"runtime_context: mpi_ranks=([0-9]+) "
            r"OMP_NUM_THREADS=([0-9]+) OMP_DYNAMIC=FALSE "
            r"OMP_PROC_BIND=(\S+) OMP_PLACES=(\S+) "
            r"amrex\.omp_threads=([0-9]+)",
            line,
        )
        if runtime_match:
            observed = dict(
                zip(
                    ("ranks", "threads", "proc_bind", "places", "amrex_threads"),
                    runtime_match.groups(),
                )
            )
            if runtime is not None:
                raise EvidenceError(f"{path}: multiple runtime_context lines")
            runtime = observed
        if line.startswith("command:"):
            if command is not None:
                raise EvidenceError(f"{path}: multiple command lines")
            try:
                command = shlex.split(line.removeprefix("command:").strip())
            except ValueError as exc:
                raise EvidenceError(f"{path}: malformed command line: {exc}") from exc
        metadata_match = re.fullmatch(r"([A-Za-z0-9_]+) = (.*)", line)
        if metadata_match:
            key, value = metadata_match.groups()
            if key in metadata and metadata[key] != value:
                raise EvidenceError(f"{path}: conflicting {key} values")
            metadata[key] = value
        completion = re.search(
            r"LBM run complete: steps=([0-9]+), descriptor=([A-Za-z0-9]+)",
            line,
        )
        if completion:
            completions.append((int(completion.group(1)), completion.group(2)))
        steady = re.fullmatch(
            r"poiseuille_steady reached step=([0-9]+) "
            r"relative_l2_delta_u=(\S+) delta_step=([0-9]+)",
            line,
        )
        if steady:
            steady_markers.append(
                (
                    int(steady.group(1)),
                    _finite(steady.group(2), f"{path}:steady residual"),
                    int(steady.group(3)),
                )
            )

    if runtime is None:
        raise EvidenceError(f"{path}: missing canonical runtime_context")
    if command is None:
        raise EvidenceError(f"{path}: missing command line")
    if len(completions) != 1:
        raise EvidenceError(f"{path}: expected exactly one completion marker")
    if len(steady_markers) > 1:
        raise EvidenceError(f"{path}: multiple steady reached markers")
    return {
        "lines": lines,
        "text": "\n".join(lines),
        "runtime": runtime,
        "command": command,
        "metadata": metadata,
        "completion": completions[0],
        "steady": steady_markers[0] if steady_markers else None,
    }


def _required(metadata: dict[str, str], key: str, path: Path) -> str:
    if key not in metadata:
        raise EvidenceError(f"{path}: missing {key} metadata")
    return metadata[key]


def _expected_value(n: int) -> tuple[float, int, int]:
    return 1.0 / n, 200 * n * n, 100 if n == 10 else 200 if n == 20 else 500


def _expected_command_tail(
    group: dict[str, object],
    n: int,
    output_directory: str,
    log_directory: str,
    threads: int,
) -> list[str]:
    u_max, max_step, check_interval = _expected_value(n)
    entry_id = f"{group['entry_prefix']}.n{n}"
    tail = [
        str(group["target"]),
        str(group["input"]),
        f"poiseuille_flow.intervals={4 * n}",
        str(n),
        f"poiseuille_flow.depth_cells={group['depth']}",
        f"poiseuille_flow.u_max_lb={u_max:.17g}",
        "poiseuille_flow.tau=0.75",
    ]
    if bool(group["pressure"]):
        density_delta = 8.0 / (n * n)
        tail.extend(
            (
                f"poiseuille_flow.rho_in_lb={1.0 + 0.5 * density_delta:.17g}",
                f"poiseuille_flow.rho_out_lb={1.0 - 0.5 * density_delta:.17g}",
                "poiseuille_flow.pressure_scaling=fixed_re_fixed_tau",
            )
        )
    tail.extend(
        (
            "mesh.max_grid_size=32",
            f"run.variant={entry_id}",
            "run.kind=convergence",
            f"run.max_step={max_step}",
            f"diagnostics.interval={check_interval}",
            f"diagnostics.steady_check_interval={check_interval}",
            "diagnostics.steady_l2_tolerance=5.0e-8",
            "diagnostics.stop_on_steady=1",
            "poiseuille_validation.write_profile=1",
            "poiseuille_validation.assert=0",
            "poiseuille_validation.mass_assert=0",
            "output.plot_interval=0",
            "output.write_final_plotfile=0",
            f"output.directory={output_directory}",
            f"output.log_directory={log_directory}",
            f"amrex.omp_threads={threads}",
        )
    )
    return tail


def _validate_command(
    group: dict[str, object],
    n: int,
    log_path: Path,
    log: dict[str, object],
    workdir: Path,
    output_directory: str,
    log_directory: str,
) -> None:
    runtime = log["runtime"]
    command = log["command"]
    assert isinstance(runtime, dict)
    assert isinstance(command, list)
    ranks = int(runtime["ranks"])
    threads = int(runtime["threads"])
    if ranks <= 0 or threads <= 0:
        raise EvidenceError(f"{log_path}: ranks and threads must be positive")
    if runtime["amrex_threads"] != runtime["threads"]:
        raise EvidenceError(f"{log_path}: amrex.omp_threads mismatch")

    prefix = [
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
        prefix.extend(
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
    expected = prefix + _expected_command_tail(
        group, n, output_directory, log_directory, threads
    )
    if command != expected:
        raise EvidenceError(
            f"{log_path}: command does not match the approved direct N={n} command"
        )

    target = _resolve(str(group["target"]), workdir)
    control = _resolve(str(group["input"]), workdir)
    _require_under(target, workdir / "build", "direct executable")
    _require_under(control, workdir / "inputs", "case control")
    if not target.is_file():
        raise EvidenceError(f"{log_path}: direct executable is missing: {target}")
    if not control.is_file():
        raise EvidenceError(f"{log_path}: case-local input is missing: {control}")


def _expect_text(
    actual: str,
    expected: str,
    field: str,
    path: Path,
) -> None:
    if actual != expected:
        raise EvidenceError(
            f"{path}: {field}={actual!r}, expected {expected!r}"
        )


def _expect_number(
    actual: str,
    expected: float,
    field: str,
    path: Path,
) -> None:
    value = _finite(actual, f"{path}:{field}")
    if not _close(value, expected):
        raise EvidenceError(
            f"{path}: {field}={value:.17g}, expected {expected:.17g}"
        )


def validate_log_identity(
    group: dict[str, object],
    n: int,
    log_path: Path,
    log: dict[str, object],
    workdir: Path,
    metrics_path: Path,
) -> dict[str, object]:
    metadata = log["metadata"]
    lines = log["lines"]
    text = log["text"]
    completion = log["completion"]
    assert isinstance(metadata, dict)
    assert isinstance(lines, list)
    assert isinstance(text, str)
    assert isinstance(completion, tuple)

    output_dir = metrics_path.parent
    log_dir = log_path.parent
    output_relative = _relative(output_dir, workdir, "output directory")
    log_relative = _relative(log_dir, workdir, "log directory")
    if not output_relative.startswith("outputs/"):
        raise EvidenceError(f"{metrics_path}: output must stay in case-local outputs")
    if not log_relative.startswith("log/"):
        raise EvidenceError(f"{log_path}: log must stay in case-local log")
    _validate_command(
        group, n, log_path, log, workdir, output_relative, log_relative
    )

    expected_text = {
        "working_directory": str(workdir),
        "descriptor": str(group["descriptor"]),
        "case_name": "poiseuille_flow",
        "case_variant": f"{group['entry_prefix']}.n{n}",
        "condition": str(group["condition"]),
        "scheme": str(group["scheme"]),
        "profile": str(group["profile"]),
        "initialization": "poiseuille",
        "location_source": "xlo/xhi drive and ylo/yhi wall case composition",
        "output_directory": output_relative,
        "terminal_log_directory": log_relative,
        "convergence_row_written": "false",
        "run_output_dir": output_relative,
        "poiseuille_profile_csv": f"{output_relative}/poiseuille_profile.csv",
        "poiseuille_profile_metrics_csv": (
            f"{output_relative}/poiseuille_profile_metrics.csv"
        ),
        "poiseuille_density_pressure_csv": (
            f"{output_relative}/poiseuille_density_pressure.csv"
        ),
        "poiseuille_mass_gate": "record_only",
    }
    for field, expected in expected_text.items():
        _expect_text(_required(metadata, field, log_path), expected, field, log_path)

    u_max, max_step, check_interval = _expected_value(n)
    expected_numbers = {
        "rho_ref_lb": 1.0,
        "u_ref_lb": u_max,
        "nu_lb": EXPECTED_NU,
        "tau": EXPECTED_TAU,
        "omega": EXPECTED_OMEGA,
        "max_step": float(max_step),
        "diagnostics_interval": float(check_interval),
        "steady_check_interval": float(check_interval),
        "steady_l2_tolerance": STEADY_TOL,
        "stop_on_steady": 1.0,
        "plot_interval": 0.0,
    }
    for field, expected in expected_numbers.items():
        _expect_number(_required(metadata, field, log_path), expected, field, log_path)

    physical = _parse_tuple(
        _single_prefixed(lines, "physical_intervals = ", log_path),
        f"{log_path}:physical_intervals",
    )
    valid = _parse_tuple(
        _single_prefixed(lines, "valid_nodes = ", log_path),
        f"{log_path}:valid_nodes",
    )
    prob_lo = _parse_tuple(
        _single_prefixed(lines, "AMReX envelope prob_lo = ", log_path),
        f"{log_path}:prob_lo",
    )
    prob_hi = _parse_tuple(
        _single_prefixed(lines, "AMReX envelope prob_hi = ", log_path),
        f"{log_path}:prob_hi",
    )
    guo = bool(group["guo"])
    depth = int(group["depth"])
    expected_valid = (4 * n + int(guo), n + int(guo), depth)
    expected_lo = (-0.5 / n, -0.5 / n, 0.0) if guo else (0.0, 0.0, 0.0)
    expected_hi = (
        (4.0 + 0.5 / n, 1.0 + 0.5 / n, depth / n)
        if guo
        else (4.0, 1.0, depth / n)
    )
    for actual, expected, field in (
        (physical, (4 * n, n), "physical_intervals"),
        (valid, expected_valid, "valid_nodes"),
        (prob_lo, expected_lo, "prob_lo"),
        (prob_hi, expected_hi, "prob_hi"),
    ):
        if len(actual) != len(expected) or any(
            not _close(value, float(reference))
            for value, reference in zip(actual, expected)
        ):
            raise EvidenceError(
                f"{log_path}: {field}={actual}, expected {expected}"
            )

    profile_geometry = "valid_node_center" if guo else "cell_center_inside_domain_face"
    fragments = (
        f"profile_geometry = {profile_geometry}",
        f"profile_valid_nodes = {n + int(guo)}",
        f"profile_lattice_intervals = {n}",
        "profile_assert = 0",
        "mass_assert = 0",
        "E_2_profile_tol = 0.05",
        "E_inf_profile_tol = 0.08",
        "transverse_u/Umax_tol = 0.01",
        "pressure_boundary_rho_tol = 1e-10",
    )
    for fragment in fragments:
        if fragment not in text:
            raise EvidenceError(f"{log_path}: missing diagnostic identity {fragment!r}")

    completion_step, completion_descriptor = completion
    if completion_descriptor != group["descriptor"]:
        raise EvidenceError(f"{log_path}: completion descriptor mismatch")
    if completion_step <= 0 or completion_step > max_step:
        raise EvidenceError(f"{log_path}: completion step outside steady/cap horizon")
    return {
        "completion_step": completion_step,
        "steady": log["steady"],
        "metadata": metadata,
    }

def validate_artifacts(
    group: dict[str, object],
    n: int,
    metrics_path: Path,
    log_evidence: dict[str, object],
) -> tuple[dict[str, float], list[str]]:
    row = read_metrics(metrics_path)
    guo = bool(group["guo"])
    pressure = bool(group["pressure"])
    expected_integers = {
        "nx": 4 * n + int(guo),
        "ny": n + int(guo),
        "nz": int(group["depth"]),
        "n_intervals_y": n,
        "on_lattice_boundary_nodes": int(guo),
        "pressure_driven": int(pressure),
        "prescribed_state": 0,
        "pressure_scaling_code": 2 if pressure else 0,
    }
    for field, expected in expected_integers.items():
        actual = _integer(row[field], f"{metrics_path}:{field}")
        if actual != expected:
            raise EvidenceError(
                f"{metrics_path}: {field}={actual}, expected {expected}"
            )

    completion_step = int(log_evidence["completion_step"])
    step = _integer(row["step"], f"{metrics_path}:step")
    if step != completion_step:
        raise EvidenceError(
            f"{metrics_path}: artifact step {step} != completion step {completion_step}"
        )
    metadata = log_evidence["metadata"]
    assert isinstance(metadata, dict)
    if _integer(
        _finite(
            _required(metadata, "final_diagnostic_step", metrics_path),
            f"{metrics_path}:final_diagnostic_step",
        ),
        f"{metrics_path}:final_diagnostic_step",
    ) != step:
        raise EvidenceError(f"{metrics_path}: final diagnostic step mismatch")

    u_max = 1.0 / n
    fixed_values = {
        "u_max_lb": u_max,
        "re_effective": EXPECTED_RE,
    }
    if pressure:
        density_delta = 8.0 / (n * n)
        fixed_values.update(
            {
                "rho_in_target_lb": 1.0 + 0.5 * density_delta,
                "rho_out_target_lb": 1.0 - 0.5 * density_delta,
                "density_delta_target_lb": density_delta,
                "density_delta_expected_lb": density_delta,
                "density_delta_over_rho_ref": density_delta,
                "mach": u_max / math.sqrt(1.0 / 3.0),
            }
        )
    for field, expected in fixed_values.items():
        if not _close(row[field], expected, absolute=2.0e-12):
            raise EvidenceError(
                f"{metrics_path}: {field}={row[field]:.17g}, "
                f"expected {expected:.17g}"
            )

    for field in (
        "E_2_profile",
        "E_inf_profile",
        "E_2_profile_full",
        "E_inf_profile_full",
        "max_abs_uy_over_umax",
        "max_abs_uz_over_umax",
        "relative_mass_error",
        "raw_lattice_mass",
        "raw_lattice_reference_mass",
        "physical_weighted_mass",
        "physical_weighted_reference_mass",
        "physical_weighted_relative_mass_error",
        "rho_std_over_rho0",
        "max_abs_pressure_boundary_rho_error",
        "max_abs_profile_error_all_x_over_umax",
        "max_streamwise_ux_spread_over_umax",
    ):
        if row[field] < 0.0:
            raise EvidenceError(f"{metrics_path}: {field} must be nonnegative")
    if row["E_2_profile"] <= 0.0:
        raise EvidenceError(f"{metrics_path}: E_2_profile must be positive")
    if (
        row["raw_lattice_reference_mass"] <= 0.0
        or row["physical_weighted_reference_mass"] <= 0.0
    ):
        raise EvidenceError(f"{metrics_path}: reference masses must be positive")

    profile_path = metrics_path.with_name("poiseuille_profile.csv")
    density_path = metrics_path.with_name("poiseuille_density_pressure.csv")
    profile = _read_csv(profile_path, EXPECTED_PROFILE_COLUMNS)
    density = _read_csv(density_path, EXPECTED_DENSITY_COLUMNS)
    if len(profile) != expected_integers["ny"]:
        raise EvidenceError(
            f"{profile_path}: expected {expected_integers['ny']} rows, "
            f"found {len(profile)}"
        )
    if len(density) != expected_integers["nx"]:
        raise EvidenceError(
            f"{density_path}: expected {expected_integers['nx']} rows, "
            f"found {len(density)}"
        )
    for index, value in enumerate(profile):
        if _integer(value["j"], f"{profile_path}:j") != index:
            raise EvidenceError(f"{profile_path}: j sequence is not contiguous")
        sample_i = _integer(value["sample_i"], f"{profile_path}:sample_i")
        if sample_i != expected_integers["nx"] // 2:
            raise EvidenceError(
                f"{profile_path}: sample_i does not identify the center profile")
        boundary = _integer(
            value["is_physical_boundary_node"],
            f"{profile_path}:is_physical_boundary_node",
        )
        expected_boundary = int(guo and index in (0, len(profile) - 1))
        if boundary != expected_boundary:
            raise EvidenceError(f"{profile_path}: boundary-node identity mismatch")
        expected_eta = index / n if guo else (index + 0.5) / n
        if not _close(value["eta"], expected_eta):
            raise EvidenceError(f"{profile_path}: eta identity mismatch at j={index}")
        if not _close(value["y_phys"], expected_eta):
            raise EvidenceError(
                f"{profile_path}: y_phys normalization mismatch at j={index}")
    for index, value in enumerate(density):
        if _integer(value["i"], f"{density_path}:i") != index:
            raise EvidenceError(f"{density_path}: i sequence is not contiguous")
        expected_xi = index / (4 * n) if guo else (index + 0.5) / (4 * n)
        if not _close(value["x_over_L"], expected_xi):
            raise EvidenceError(f"{density_path}: x_over_L identity mismatch at i={index}")
        if not _close(value["x_phys"], 4.0 * expected_xi):
            raise EvidenceError(
                f"{density_path}: x_phys normalization mismatch at i={index}")

    failures: list[str] = []
    limits = {
        "E_2_profile": E2_TOL,
        "E_inf_profile": EINF_TOL,
        "E_2_profile_full": E2_TOL,
        "E_inf_profile_full": EINF_TOL,
        "max_abs_uy_over_umax": TRANSVERSE_TOL,
        "max_abs_uz_over_umax": TRANSVERSE_TOL,
    }
    for field, limit in limits.items():
        if row[field] > limit:
            failures.append(f"N={n} {field}={row[field]:.12g} > {limit:.12g}")
    if group["route"] == "hbb-velocity":
        for field in (
            "relative_mass_error",
            "physical_weighted_relative_mass_error",
        ):
            if row[field] > MASS_TOL:
                failures.append(f"N={n} {field}={row[field]:.12g} > {MASS_TOL:.12g}")
    if pressure:
        if row["max_abs_pressure_boundary_rho_error"] > PRESSURE_RHO_TOL:
            failures.append(
                f"N={n} pressure-face error="
                f"{row['max_abs_pressure_boundary_rho_error']:.12g} "
                f"> {PRESSURE_RHO_TOL:.12g}"
            )
        if row["rho_streamwise_linear_r2"] < PRESSURE_R2_MIN:
            failures.append(
                f"N={n} rho_streamwise_linear_r2="
                f"{row['rho_streamwise_linear_r2']:.12g} < {PRESSURE_R2_MIN:.12g}"
            )

    steady_text = _required(metadata, "poiseuille_steady_reached", metrics_path)
    if steady_text not in ("true", "false"):
        raise EvidenceError(f"{metrics_path}: invalid poiseuille_steady_reached")
    residual = _finite(
        _required(metadata, "poiseuille_final_relative_l2_delta_u", metrics_path),
        f"{metrics_path}:steady residual",
    )
    steady_marker = log_evidence["steady"]
    reached_step_record = _integer(
        _finite(
            _required(metadata, "poiseuille_steady_reached_step", metrics_path),
            f"{metrics_path}:poiseuille_steady_reached_step",
        ),
        f"{metrics_path}:poiseuille_steady_reached_step",
    )
    _, max_step, _ = _expected_value(n)
    if steady_text == "true":
        if steady_marker is None:
            raise EvidenceError(f"{metrics_path}: missing steady reached marker")
        reached_step, reached_residual, delta_step = steady_marker
        if reached_step != step or delta_step != 1:
            raise EvidenceError(f"{metrics_path}: inconsistent steady marker")
        if reached_step_record != step:
            raise EvidenceError(
                f"{metrics_path}: recorded steady step does not match final step"
            )
        if not _close(residual, reached_residual):
            raise EvidenceError(f"{metrics_path}: steady residual mismatch")
        if residual >= STEADY_TOL:
            raise EvidenceError(f"{metrics_path}: true steady marker exceeds tolerance")
    else:
        if steady_marker is not None:
            raise EvidenceError(f"{metrics_path}: false steady flag has reached marker")
        if reached_step_record != -1:
            raise EvidenceError(
                f"{metrics_path}: false steady flag must record reached step -1"
            )
        if step != max_step:
            raise EvidenceError(
                f"{metrics_path}: non-steady run did not complete its declared horizon"
            )
        failures.append(
            f"N={n} steady gate not reached; final residual={residual:.12g}"
        )
    return row, failures


def _fit(rows: dict[int, dict[str, float]]) -> tuple[float, float, float]:
    xs = [math.log(1.0 / n) for n in EXPECTED_RESOLUTIONS]
    ys = [math.log(rows[n]["E_2_profile"]) for n in EXPECTED_RESOLUTIONS]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    sxx = sum((value - x_mean) ** 2 for value in xs)
    slope = sum(
        (x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)
    ) / sxx
    intercept = y_mean - slope * x_mean
    fitted = [slope * x + intercept for x in xs]
    ss_res = sum((y - estimate) ** 2 for y, estimate in zip(ys, fitted))
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    r2 = 1.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    return slope, intercept, r2


def write_result(
    group_id: str,
    workdir: Path,
    output_dir: Path,
    rows: dict[int, dict[str, float]],
    artifacts: dict[int, Path],
    logs: dict[int, Path],
    log_evidence: dict[int, dict[str, object]],
    slope: float,
    intercept: float,
    fit_r2: float,
    failures: list[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "poiseuille_convergence.csv"
    summary_path = output_dir / "poiseuille_convergence_summary.txt"
    if csv_path.exists() or summary_path.exists():
        raise EvidenceError(f"{output_dir}: refusing to overwrite qualifier output")

    csv_tmp = output_dir / ".poiseuille_convergence.csv.tmp"
    summary_tmp = output_dir / ".poiseuille_convergence_summary.txt.tmp"
    group = GROUPS[group_id]
    with csv_tmp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "acceptance_group_id",
                "entry_id",
                "n",
                "h",
                "nx",
                "ny",
                "nz",
                "u_max_lb",
                "E_2_profile",
                "E_inf_profile",
                "steady_reached_step",
                "final_relative_l2_delta_u",
                "max_abs_pressure_boundary_rho_error",
                "rho_streamwise_linear_r2",
                "metrics_path",
                "profile_path",
                "density_path",
                "raw_log_path",
            )
        )
        for n in EXPECTED_RESOLUTIONS:
            metadata = log_evidence[n]["metadata"]
            assert isinstance(metadata, dict)
            metrics_path = artifacts[n]
            writer.writerow(
                (
                    group_id,
                    f"{group['entry_prefix']}.n{n}",
                    n,
                    1.0 / n,
                    int(rows[n]["nx"]),
                    int(rows[n]["ny"]),
                    int(rows[n]["nz"]),
                    rows[n]["u_max_lb"],
                    rows[n]["E_2_profile"],
                    rows[n]["E_inf_profile"],
                    metadata["poiseuille_steady_reached_step"],
                    metadata["poiseuille_final_relative_l2_delta_u"],
                    rows[n]["max_abs_pressure_boundary_rho_error"],
                    rows[n]["rho_streamwise_linear_r2"],
                    _relative(metrics_path, workdir, "metrics path"),
                    _relative(
                        metrics_path.with_name("poiseuille_profile.csv"),
                        workdir,
                        "profile path",
                    ),
                    _relative(
                        metrics_path.with_name("poiseuille_density_pressure.csv"),
                        workdir,
                        "density path",
                    ),
                    _relative(logs[n], workdir, "raw log path"),
                )
            )
    status = "pass" if not failures else "numerical-fail"
    summary_tmp.write_text(
        f"acceptance_group_id={group_id}\n"
        f"entry_count=3\n"
        f"resolutions=10,20,50\n"
        f"slope={slope:.12g}\n"
        f"intercept={intercept:.12g}\n"
        f"fit_r2={fit_r2:.12g}\n"
        f"slope_min={SLOPE_MIN:.12g}\n"
        f"slope_max={SLOPE_MAX:.12g}\n"
        f"scientific_status={status}\n"
        f"scientific_failures={' | '.join(failures) if failures else 'none'}\n",
        encoding="utf-8",
    )
    csv_tmp.replace(csv_path)
    summary_tmp.replace(summary_path)


def qualify(
    group_id: str,
    workdir: Path,
    artifact_specs: list[str],
    log_specs: list[str],
    output_dir: Path,
) -> list[str]:
    if group_id not in GROUPS:
        raise EvidenceError(f"unknown acceptance group {group_id!r}")
    resolved_workdir = workdir.resolve()
    if not resolved_workdir.is_dir():
        raise EvidenceError(f"workdir is not a directory: {resolved_workdir}")
    artifacts = {
        n: _resolve(path, resolved_workdir)
        for n, path in parse_specs(artifact_specs, "--artifact").items()
    }
    logs = {
        n: _resolve(path, resolved_workdir)
        for n, path in parse_specs(log_specs, "--log").items()
    }
    resolved_output = _resolve(output_dir, resolved_workdir)
    _require_under(resolved_output, resolved_workdir / "outputs", "qualifier output")

    group = GROUPS[group_id]
    rows: dict[int, dict[str, float]] = {}
    evidence: dict[int, dict[str, object]] = {}
    failures: list[str] = []
    for n in EXPECTED_RESOLUTIONS:
        log = read_log(logs[n])
        evidence[n] = validate_log_identity(
            group, n, logs[n], log, resolved_workdir, artifacts[n]
        )
        rows[n], entry_failures = validate_artifacts(
            group, n, artifacts[n], evidence[n]
        )
        failures.extend(entry_failures)

    slope, intercept, fit_r2 = _fit(rows)
    if not SLOPE_MIN <= slope <= SLOPE_MAX:
        failures.append(
            f"three-point slope={slope:.12g} outside [{SLOPE_MIN},{SLOPE_MAX}]"
        )
    write_result(
        group_id,
        resolved_workdir,
        resolved_output,
        rows,
        artifacts,
        logs,
        evidence,
        slope,
        intercept,
        fit_r2,
        failures,
    )
    return failures


def qualify_entry(
    entry_id: str,
    workdir: Path,
    artifact_path: Path,
    log_path: Path,
) -> dict[str, object]:
    if entry_id not in ENTRY_DEFINITIONS:
        raise EvidenceError(f"unknown Poiseuille entry {entry_id!r}")
    resolved_workdir = workdir.resolve()
    if not resolved_workdir.is_dir():
        raise EvidenceError(f"workdir is not a directory: {resolved_workdir}")

    group_id, n = ENTRY_DEFINITIONS[entry_id]
    group = GROUPS[group_id]
    artifact = _resolve(artifact_path, resolved_workdir)
    raw_log = _resolve(log_path, resolved_workdir)
    log = read_log(raw_log)
    evidence = validate_log_identity(
        group, n, raw_log, log, resolved_workdir, artifact
    )
    row, local_limit_observations = validate_artifacts(
        group, n, artifact, evidence
    )
    metadata = evidence["metadata"]
    assert isinstance(metadata, dict)
    return {
        "mode": "entry-evidence",
        "evidence_status": "valid",
        "entry_id": entry_id,
        "acceptance_group_id": group_id,
        "n": n,
        "artifact_path": _relative(
            artifact, resolved_workdir, "entry artifact path"
        ),
        "raw_log_path": _relative(raw_log, resolved_workdir, "entry raw log path"),
        "completion_step": int(evidence["completion_step"]),
        "steady_reached": metadata["poiseuille_steady_reached"] == "true",
        "final_relative_l2_delta_u": _finite(
            metadata["poiseuille_final_relative_l2_delta_u"],
            f"{artifact}:steady residual",
        ),
        "observations": {
            "E_2_profile": row["E_2_profile"],
            "E_inf_profile": row["E_inf_profile"],
            "max_abs_uy_over_umax": row["max_abs_uy_over_umax"],
            "max_abs_uz_over_umax": row["max_abs_uz_over_umax"],
            "relative_mass_error": row["relative_mass_error"],
            "physical_weighted_relative_mass_error": (
                row["physical_weighted_relative_mass_error"]
            ),
            "max_abs_pressure_boundary_rho_error": (
                row["max_abs_pressure_boundary_rho_error"]
            ),
            "rho_streamwise_linear_r2": row["rho_streamwise_linear_r2"],
        },
        "local_limit_observations": local_limit_observations,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    identity = parser.add_mutually_exclusive_group(required=True)
    identity.add_argument("--group", choices=tuple(GROUPS))
    identity.add_argument("--entry", choices=tuple(ENTRY_DEFINITIONS))
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        metavar="SPEC",
        help="PATH for --entry; N=PATH for --group",
    )
    parser.add_argument(
        "--log",
        action="append",
        default=[],
        metavar="SPEC",
        help="PATH for --entry; N=PATH for --group",
    )
    args = parser.parse_args()
    try:
        if args.entry is not None:
            if args.output is not None:
                raise EvidenceError("--entry does not accept --output")
            if len(args.artifact) != 1 or len(args.log) != 1:
                raise EvidenceError(
                    "--entry requires exactly one --artifact PATH and one --log PATH"
                )
            observation = qualify_entry(
                args.entry,
                args.workdir,
                Path(args.artifact[0]),
                Path(args.log[0]),
            )
            print(json.dumps(observation, sort_keys=True, separators=(",", ":")))
            return 0

        if args.output is None:
            raise EvidenceError("--group requires --output")
        failures = qualify(
            args.group,
            args.workdir,
            args.artifact,
            args.log,
            args.output,
        )
    except EvidenceError as exc:
        print(f"poiseuille_qualifier invalid: {exc}", file=sys.stderr)
        return 2
    except (OSError, UnicodeError, csv.Error) as exc:
        print(
            f"poiseuille_qualifier invalid: evidence I/O failure: {exc}",
            file=sys.stderr,
        )
        return 2
    if failures:
        print(
            f"poiseuille_qualifier numerical-fail group={args.group}: "
            + " | ".join(failures),
            file=sys.stderr,
        )
        return 1
    print(f"poiseuille_qualifier pass group={args.group}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
