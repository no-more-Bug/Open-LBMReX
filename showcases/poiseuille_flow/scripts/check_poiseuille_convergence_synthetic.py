#!/usr/bin/env python3
"""Synthetic positive and negative fixtures for the Poiseuille qualifier."""

from __future__ import annotations

import csv
import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import check_poiseuille_convergence as qualifier

CHECKER = Path(__file__).with_name("check_poiseuille_convergence.py").resolve()
CASE_DIR = Path(__file__).resolve().parents[1]
RUNNERS = (
    CASE_DIR / "scripts" / "run_poiseuille_velocity.sh",
    CASE_DIR / "scripts" / "run_guo_poiseuille.sh",
)


def _format_tuple(values: tuple[float, ...]) -> str:
    return "(" + ", ".join(f"{value:.12g}" for value in values) + ")"


def _runtime_prefix(ranks: int, threads: int) -> list[str]:
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
    return command


def _write_metrics(
    path: Path,
    group: dict[str, object],
    n: int,
    step: int,
    *,
    updates: dict[str, object] | None = None,
    columns: tuple[str, ...] | None = None,
    extra_value: bool = False,
) -> None:
    guo = bool(group["guo"])
    pressure = bool(group["pressure"])
    u_max = 1.0 / n
    density_delta = 8.0 / (n * n) if pressure else 0.0
    values: dict[str, object] = {
        "step": step,
        "nx": 4 * n + int(guo),
        "ny": n + int(guo),
        "nz": int(group["depth"]),
        "n_intervals_y": n,
        "on_lattice_boundary_nodes": int(guo),
        "u_max_lb": u_max,
        "E_2_profile": 0.01 * (10.0 / n) ** 2,
        "E_inf_profile": 0.015 * (10.0 / n) ** 2,
        "E_2_profile_full": 0.011 * (10.0 / n) ** 2,
        "E_inf_profile_full": 0.016 * (10.0 / n) ** 2,
        "max_abs_uy_over_umax": 1.0e-8,
        "max_abs_uz_over_umax": 2.0e-8,
        "relative_mass_error": 1.0e-8,
        "raw_lattice_mass": 100.0,
        "raw_lattice_reference_mass": 100.0,
        "physical_weighted_mass": 100.0,
        "physical_weighted_reference_mass": 100.0,
        "physical_weighted_relative_mass_error": 1.0e-8,
        "rho_mean_minus_rho0": 0.0,
        "rho_std_over_rho0": 1.0e-6,
        "rho_min": 1.0 - 0.5 * density_delta,
        "rho_max": 1.0 + 0.5 * density_delta,
        "rho_streamwise_xlo": 1.0 + 0.5 * density_delta,
        "rho_streamwise_xhi": 1.0 - 0.5 * density_delta,
        "pressure_streamwise_xlo": (1.0 + 0.5 * density_delta) / 3.0,
        "pressure_streamwise_xhi": (1.0 - 0.5 * density_delta) / 3.0,
        "rho_streamwise_slope": -density_delta,
        "pressure_streamwise_slope": -density_delta / 3.0,
        "rho_streamwise_linear_r2": 0.999999,
        "pressure_driven": int(pressure),
        "prescribed_state": 0,
        "pressure_scaling_code": 2 if pressure else 0,
        "re_effective": 12.0,
        "rho_in_target_lb": 1.0 + 0.5 * density_delta if pressure else 0.0,
        "rho_out_target_lb": 1.0 - 0.5 * density_delta if pressure else 0.0,
        "density_delta_target_lb": density_delta,
        "density_delta_expected_lb": density_delta,
        "density_delta_over_rho_ref": density_delta,
        "mach": u_max / (1.0 / 3.0) ** 0.5 if pressure else 0.0,
        "max_abs_pressure_boundary_rho_error": 1.0e-12 if pressure else 0.0,
        "rho_streamwise_delta_measured": density_delta,
        "rho_streamwise_delta_relative_error": 1.0e-8,
        "max_abs_profile_error_all_x_over_umax": 0.01,
        "max_streamwise_ux_spread_over_umax": 0.01,
    }
    if updates:
        values.update(updates)
    header = columns or qualifier.EXPECTED_METRICS_COLUMNS
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        row = [values.get(field, 0.0) for field in header]
        writer.writerow((*row, "unexpected") if extra_value else row)


def _write_profile(path: Path, group: dict[str, object], n: int) -> None:
    guo = bool(group["guo"])
    ny = n + int(guo)
    nx = 4 * n + int(guo)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(qualifier.EXPECTED_PROFILE_COLUMNS)
        for j in range(ny):
            eta = j / n if guo else (j + 0.5) / n
            writer.writerow(
                (
                    j,
                    eta,
                    eta,
                    0.0,
                    0.0,
                    1.0e-12,
                    2.0e-12,
                    nx // 2,
                    int(guo and j in (0, ny - 1)),
                )
            )


def _write_density(path: Path, group: dict[str, object], n: int) -> None:
    guo = bool(group["guo"])
    nx = 4 * n + int(guo)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(qualifier.EXPECTED_DENSITY_COLUMNS)
        for i in range(nx):
            xi = i / (4 * n) if guo else (i + 0.5) / (4 * n)
            writer.writerow((i, 4.0 * xi, xi, 1.0, 1.0 / 3.0))


def _write_log(
    path: Path,
    *,
    workdir: Path,
    group_id: str,
    n: int,
    metrics_path: Path,
    ranks: int,
    threads: int,
    steady: bool = True,
    command_override: list[str] | None = None,
) -> None:
    group = qualifier.GROUPS[group_id]
    guo = bool(group["guo"])
    depth = int(group["depth"])
    u_max, max_step, check_interval = qualifier._expected_value(n)
    step = check_interval * 3 if steady else max_step
    output_relative = metrics_path.parent.relative_to(workdir).as_posix()
    log_relative = path.parent.relative_to(workdir).as_posix()
    command = _runtime_prefix(ranks, threads) + qualifier._expected_command_tail(
        group, n, output_relative, log_relative, threads
    )
    if command_override is not None:
        command = command_override

    valid = (4 * n + int(guo), n + int(guo), depth)
    prob_lo = (-0.5 / n, -0.5 / n, 0.0) if guo else (0.0, 0.0, 0.0)
    prob_hi = (
        (4.0 + 0.5 / n, 1.0 + 0.5 / n, depth / n)
        if guo
        else (4.0, 1.0, depth / n)
    )
    residual = 1.0e-9 if steady else 1.0e-5
    reached_step = step if steady else -1
    lines = [
        (
            f"runtime_context: mpi_ranks={ranks} OMP_NUM_THREADS={threads} "
            "OMP_DYNAMIC=FALSE OMP_PROC_BIND=spread OMP_PLACES=cores "
            f"amrex.omp_threads={threads}"
        ),
        f"working_directory = {workdir}",
        f"command: {shlex.join(command)}",
        f"descriptor = {group['descriptor']}",
        "case_name = poiseuille_flow",
        f"case_variant = {group['entry_prefix']}.n{n}",
        f"condition = {group['condition']}",
        f"scheme = {group['scheme']}",
        f"profile = {group['profile']}",
        "initialization = poiseuille",
        "location_source = xlo/xhi drive and ylo/yhi wall case composition",
        f"valid_nodes = {_format_tuple(valid)}",
        f"physical_intervals = ({4 * n}, {n})",
        f"AMReX envelope prob_lo = {_format_tuple(prob_lo)}",
        f"AMReX envelope prob_hi = {_format_tuple(prob_hi)}",
        "rho_ref_lb = 1",
        f"u_ref_lb = {u_max:.17g}",
        "nu_lb = 0.0833333333333",
        "tau = 0.75",
        "omega = 1.33333333333",
        f"max_step = {max_step}",
        f"diagnostics_interval = {check_interval}",
        f"steady_check_interval = {check_interval}",
        "steady_l2_tolerance = 5e-08",
        "stop_on_steady = 1",
        "plot_interval = 0",
        f"output_directory = {output_relative}",
        f"terminal_log_directory = {log_relative}",
        (
            "poiseuille_flow diagnostics: U_max_lb = "
            f"{u_max:.12g}, profile_axis = y, "
            f"profile_geometry = {'valid_node_center' if guo else 'cell_center_inside_domain_face'}, "
            f"profile_valid_nodes = {n + int(guo)}, "
            f"profile_lattice_intervals = {n}, write_profile = 1, "
            "profile_assert = 0, E_2_profile_tol = 0.05, "
            "E_inf_profile_tol = 0.08, transverse_u/Umax_tol = 0.01, "
            "mass_assert = 0, relative_mass_tol = 1e-05, "
            "pressure_boundary_rho_tol = 1e-10"
        ),
    ]
    if steady:
        lines.append(
            f"poiseuille_steady reached step={step} "
            f"relative_l2_delta_u={residual:.12g} delta_step=1"
        )
    lines.extend(
        (
            f"poiseuille_profile_csv = {output_relative}/poiseuille_profile.csv",
            (
                "poiseuille_profile_metrics_csv = "
                f"{output_relative}/poiseuille_profile_metrics.csv"
            ),
            (
                "poiseuille_density_pressure_csv = "
                f"{output_relative}/poiseuille_density_pressure.csv"
            ),
            f"poiseuille_steady_reached = {str(steady).lower()}",
            f"poiseuille_steady_reached_step = {reached_step}",
            f"poiseuille_final_relative_l2_delta_u = {residual:.12g}",
            "poiseuille_mass_gate = record_only",
            f"final_diagnostic_step = {step}",
            "convergence_row_written = false",
            f"run_output_dir = {output_relative}",
            (
                f"LBM run complete: steps={step}, "
                f"descriptor={group['descriptor']}, omega=1.33333333333"
            ),
            "",
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")

def make_fixture(
    root: Path,
    group_id: str,
    *,
    metric_updates: dict[int, dict[str, object]] | None = None,
    steady_false: set[int] | None = None,
    bad_columns: dict[int, tuple[str, ...]] | None = None,
    extra_values: set[int] | None = None,
) -> tuple[Path, dict[int, Path], dict[int, Path], Path]:
    group = qualifier.GROUPS[group_id]
    workdir = (root / "case").resolve()
    target = workdir / str(group["target"])
    control = workdir / str(group["input"])
    target.parent.mkdir(parents=True, exist_ok=True)
    control.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("synthetic direct executable\n", encoding="utf-8")
    control.write_text("synthetic case-local control\n", encoding="utf-8")

    output = workdir / "outputs" / "baseline" / group_id / "attempt-synthetic"
    artifacts: dict[int, Path] = {}
    logs: dict[int, Path] = {}
    ranks = 2 if group["descriptor"] == "D3Q27" else 1
    threads = 2 if group["descriptor"] == "D3Q27" else 1
    for n in qualifier.EXPECTED_RESOLUTIONS:
        _, max_step, check_interval = qualifier._expected_value(n)
        steady = n not in (steady_false or set())
        step = check_interval * 3 if steady else max_step
        metrics = output / f"N{n}" / "poiseuille_profile_metrics.csv"
        log = (
            workdir
            / "log"
            / "baseline"
            / group_id
            / "attempt-synthetic"
            / f"N{n}"
            / "console.log"
        )
        _write_metrics(
            metrics,
            group,
            n,
            step,
            updates=(metric_updates or {}).get(n),
            columns=(bad_columns or {}).get(n),
            extra_value=n in (extra_values or set()),
        )
        _write_profile(metrics.with_name("poiseuille_profile.csv"), group, n)
        _write_density(
            metrics.with_name("poiseuille_density_pressure.csv"), group, n
        )
        _write_log(
            log,
            workdir=workdir,
            group_id=group_id,
            n=n,
            metrics_path=metrics,
            ranks=ranks,
            threads=threads,
            steady=steady,
        )
        artifacts[n] = metrics
        logs[n] = log
    return workdir, artifacts, logs, output


def invoke(
    workdir: Path,
    group_id: str,
    artifacts: dict[int, Path],
    logs: dict[int, Path],
    output: Path,
    *,
    resolutions: tuple[int, ...] = qualifier.EXPECTED_RESOLUTIONS,
    extra_args: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "-S",
        "-B",
        str(CHECKER),
        "--group",
        group_id,
        "--workdir",
        str(workdir),
        "--output",
        str(output.relative_to(workdir)),
    ]
    for n in resolutions:
        command.extend(("--artifact", f"{n}={artifacts[n].relative_to(workdir)}"))
    for n in resolutions:
        command.extend(("--log", f"{n}={logs[n].relative_to(workdir)}"))
    command.extend(extra_args)
    return subprocess.run(command, text=True, capture_output=True, check=False)


def invoke_entry(
    workdir: Path,
    entry_id: str,
    artifact: Path,
    log: Path,
    *,
    extra_args: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "-S",
        "-B",
        str(CHECKER),
        "--entry",
        entry_id,
        "--workdir",
        str(workdir),
        "--artifact",
        str(artifact.relative_to(workdir)),
        "--log",
        str(log.relative_to(workdir)),
    ]
    command.extend(extra_args)
    return subprocess.run(command, text=True, capture_output=True, check=False)


def expect_code(
    result: subprocess.CompletedProcess[str],
    expected: int,
    label: str,
) -> None:
    if result.returncode != expected:
        raise AssertionError(
            f"{label}: expected exit {expected}, got {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def check_runner_contract() -> None:
    canonical_launcher = "mpirun\n                -np"
    working_token = "printf 'working_directory = %s\\n'"
    command_token = "printf 'command:'"
    arguments_token = "printf ' %q'"
    for runner in RUNNERS:
        text = runner.read_text(encoding="utf-8")
        try:
            working = text.index(working_token)
            command = text.index(command_token, working)
            arguments = text.index(arguments_token, command)
        except ValueError as error:
            raise AssertionError(f"{runner}: missing formal log token") from error
        if not working < command < arguments:
            raise AssertionError(
                f"{runner}: formal log must write working_directory before command"
            )
        if text.count(canonical_launcher) != 2 or "--nooversubscribe" in text:
            raise AssertionError(
                f"{runner}: MPI branches must use the C1 canonical launcher grammar"
            )


def run_positive(root: Path) -> None:
    if len(qualifier.GROUPS) * len(qualifier.EXPECTED_RESOLUTIONS) != 18:
        raise AssertionError("approved Poiseuille matrix must contain exactly 18 entries")
    for index, group_id in enumerate(qualifier.GROUPS):
        workdir, artifacts, logs, output = make_fixture(
            root / f"positive-{index}", group_id
        )
        result = invoke(workdir, group_id, artifacts, logs, output)
        expect_code(result, 0, group_id)
        summary = (output / "poiseuille_convergence_summary.txt").read_text(
            encoding="utf-8"
        )
        if (
            f"acceptance_group_id={group_id}" not in summary
            or "scientific_status=pass" not in summary
            or "slope=2" not in summary
        ):
            raise AssertionError(f"{group_id}: incomplete pass summary")
        with (output / "poiseuille_convergence.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            if len(list(csv.DictReader(stream))) != 3:
                raise AssertionError(f"{group_id}: expected three convergence rows")


def run_entry_evidence_fixtures(root: Path) -> None:
    seen: set[str] = set()
    for index, (group_id, group) in enumerate(qualifier.GROUPS.items()):
        workdir, artifacts, logs, output = make_fixture(
            root / f"entry-positive-{index}", group_id
        )
        for n in qualifier.EXPECTED_RESOLUTIONS:
            entry_id = f"{group['entry_prefix']}.n{n}"
            result = invoke_entry(
                workdir, entry_id, artifacts[n], logs[n]
            )
            expect_code(result, 0, f"{entry_id} entry evidence")
            try:
                observation = json.loads(result.stdout)
            except json.JSONDecodeError as exc:
                raise AssertionError(
                    f"{entry_id}: entry stdout is not JSON: {result.stdout!r}"
                ) from exc
            if (
                observation.get("mode") != "entry-evidence"
                or observation.get("evidence_status") != "valid"
                or observation.get("entry_id") != entry_id
                or observation.get("acceptance_group_id") != group_id
                or observation.get("n") != n
                or observation.get("completion_step") <= 0
                or observation.get("local_limit_observations") != []
            ):
                raise AssertionError(
                    f"{entry_id}: incomplete entry-local observation {observation!r}"
                )
            if "slope" in observation or "scientific_status" in observation:
                raise AssertionError(
                    f"{entry_id}: entry mode must not own group scientific verdict"
                )
            expected_observations = {
                "E_2_profile",
                "E_inf_profile",
                "max_abs_uy_over_umax",
                "max_abs_uz_over_umax",
                "relative_mass_error",
                "physical_weighted_relative_mass_error",
                "max_abs_pressure_boundary_rho_error",
                "rho_streamwise_linear_r2",
            }
            if set(observation.get("observations", {})) != expected_observations:
                raise AssertionError(
                    f"{entry_id}: unexpected observation schema"
                )
            seen.add(entry_id)
        for derived in (
            output / "poiseuille_convergence.csv",
            output / "poiseuille_convergence_summary.txt",
        ):
            if derived.exists():
                raise AssertionError(f"entry mode wrote derived group result: {derived}")

    if seen != set(qualifier.ENTRY_DEFINITIONS) or len(seen) != 18:
        raise AssertionError("entry mode did not cover the approved 18-entry matrix")

    hbb = "poiseuille.hbb-velocity.d2q9.slope"
    entry_id = "poiseuille.hbb-velocity.d2q9.n20"
    workdir, artifacts, logs, _ = make_fixture(
        root / "entry-local-limit",
        hbb,
        metric_updates={20: {"E_2_profile": 0.051}},
    )
    result = invoke_entry(workdir, entry_id, artifacts[20], logs[20])
    expect_code(result, 0, "entry-local numerical limit observation")
    observation = json.loads(result.stdout)
    if not observation["local_limit_observations"]:
        raise AssertionError("entry mode omitted a local limit observation")

    entry_id = "poiseuille.hbb-velocity.d2q9.n50"
    workdir, artifacts, logs, _ = make_fixture(
        root / "entry-nonsteady",
        hbb,
        steady_false={50},
    )
    result = invoke_entry(workdir, entry_id, artifacts[50], logs[50])
    expect_code(result, 0, "entry-local non-steady observation")
    observation = json.loads(result.stdout)
    if observation["steady_reached"] or not observation["local_limit_observations"]:
        raise AssertionError("entry mode omitted the non-steady local observation")

    columns = qualifier.EXPECTED_METRICS_COLUMNS[:-1]
    entry_id = "poiseuille.hbb-velocity.d2q9.n20"
    workdir, artifacts, logs, _ = make_fixture(
        root / "entry-schema",
        hbb,
        bad_columns={20: columns},
    )
    expect_code(
        invoke_entry(workdir, entry_id, artifacts[20], logs[20]),
        2,
        "entry schema drift",
    )

    workdir, artifacts, logs, output = make_fixture(
        root / "entry-output",
        hbb,
    )
    expect_code(
        invoke_entry(
            workdir,
            entry_id,
            artifacts[20],
            logs[20],
            extra_args=("--output", str(output.relative_to(workdir))),
        ),
        2,
        "entry derived output exclusion",
    )


def run_invalid_fixtures(root: Path) -> None:
    hbb = "poiseuille.hbb-velocity.d2q9.slope"
    pressure = "poiseuille.guo-pressure.d2q9.slope"

    workdir, artifacts, logs, output = make_fixture(root / "missing-n", hbb)
    expect_code(
        invoke(workdir, hbb, artifacts, logs, output, resolutions=(10, 20)),
        2,
        "missing resolution",
    )

    columns = qualifier.EXPECTED_METRICS_COLUMNS[:-1]
    workdir, artifacts, logs, output = make_fixture(
        root / "schema", hbb, bad_columns={20: columns}
    )
    expect_code(invoke(workdir, hbb, artifacts, logs, output), 2, "schema drift")

    workdir, artifacts, logs, output = make_fixture(
        root / "extra", hbb, extra_values={20}
    )
    expect_code(invoke(workdir, hbb, artifacts, logs, output), 2, "extra CSV value")

    workdir, artifacts, logs, output = make_fixture(
        root / "nan", hbb, metric_updates={20: {"E_2_profile": "nan"}}
    )
    expect_code(invoke(workdir, hbb, artifacts, logs, output), 2, "non-finite")

    workdir, artifacts, logs, output = make_fixture(root / "invalid-utf8", hbb)
    artifacts[20].write_bytes(b"\xff")
    expect_code(
        invoke(workdir, hbb, artifacts, logs, output),
        2,
        "invalid UTF-8 evidence",
    )

    workdir, artifacts, logs, output = make_fixture(
        root / "profile-sample-identity", hbb
    )
    profile_path = artifacts[20].with_name("poiseuille_profile.csv")
    with profile_path.open(newline="", encoding="utf-8") as stream:
        profile_rows = list(csv.reader(stream))
    profile_rows[1][7] = "0"
    with profile_path.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerows(profile_rows)
    expect_code(
        invoke(workdir, hbb, artifacts, logs, output),
        2,
        "profile sample identity",
    )

    workdir, artifacts, logs, output = make_fixture(
        root / "grid", hbb, metric_updates={20: {"ny": 21}}
    )
    expect_code(invoke(workdir, hbb, artifacts, logs, output), 2, "wrong grid")

    workdir, artifacts, logs, output = make_fixture(
        root / "prescribed",
        pressure,
        metric_updates={20: {"pressure_driven": 0, "prescribed_state": 1}},
    )
    expect_code(
        invoke(workdir, pressure, artifacts, logs, output),
        2,
        "prescribed-state exclusion",
    )

    workdir, artifacts, logs, output = make_fixture(
        root / "fixed-mach",
        pressure,
        metric_updates={20: {"pressure_scaling_code": 1}},
    )
    expect_code(
        invoke(workdir, pressure, artifacts, logs, output),
        2,
        "fixed-Mach exclusion",
    )

    workdir, artifacts, logs, output = make_fixture(root / "command", hbb)
    log = logs[20]
    text = log.read_text(encoding="utf-8").replace(
        "./build/lbm_poiseuille_flow",
        "./build/lbm_single_level",
        1,
    )
    log.write_text(text, encoding="utf-8")
    expect_code(invoke(workdir, hbb, artifacts, logs, output), 2, "compat target")

    workdir, artifacts, logs, output = make_fixture(root / "missing-profile", hbb)
    artifacts[20].with_name("poiseuille_profile.csv").unlink()
    expect_code(invoke(workdir, hbb, artifacts, logs, output), 2, "missing profile")

    workdir, artifacts, logs, output = make_fixture(root / "wrong-output", hbb)
    outside = workdir / "tests" / "numerical" / "poiseuille"
    expect_code(invoke(workdir, hbb, artifacts, logs, outside), 2, "test output")


def run_scientific_fixtures(root: Path) -> None:
    hbb = "poiseuille.hbb-velocity.d2q9.slope"
    pressure = "poiseuille.guo-pressure.d2q9.slope"

    workdir, artifacts, logs, output = make_fixture(
        root / "slope",
        hbb,
        metric_updates={50: {"E_2_profile": 0.02}},
    )
    result = invoke(workdir, hbb, artifacts, logs, output)
    expect_code(result, 1, "slope scientific failure")
    if "scientific_status=numerical-fail" not in (
        output / "poiseuille_convergence_summary.txt"
    ).read_text(encoding="utf-8"):
        raise AssertionError("slope failure did not produce numerical-fail record")

    workdir, artifacts, logs, output = make_fixture(
        root / "steady", hbb, steady_false={50}
    )
    expect_code(invoke(workdir, hbb, artifacts, logs, output), 1, "steady failure")

    workdir, artifacts, logs, output = make_fixture(
        root / "profile",
        hbb,
        metric_updates={20: {"E_2_profile": 0.051}},
    )
    expect_code(invoke(workdir, hbb, artifacts, logs, output), 1, "profile failure")

    workdir, artifacts, logs, output = make_fixture(
        root / "pressure-rho",
        pressure,
        metric_updates={20: {"max_abs_pressure_boundary_rho_error": 1.1e-10}},
    )
    expect_code(
        invoke(workdir, pressure, artifacts, logs, output),
        1,
        "pressure-face failure",
    )

    workdir, artifacts, logs, output = make_fixture(
        root / "pressure-r2",
        pressure,
        metric_updates={20: {"rho_streamwise_linear_r2": 0.98}},
    )
    expect_code(
        invoke(workdir, pressure, artifacts, logs, output),
        1,
        "pressure R2 failure",
    )

    workdir, artifacts, logs, output = make_fixture(root / "collision", hbb)
    expect_code(invoke(workdir, hbb, artifacts, logs, output), 0, "first write")
    expect_code(invoke(workdir, hbb, artifacts, logs, output), 2, "output collision")


def main() -> int:
    check_runner_contract()
    with tempfile.TemporaryDirectory(prefix="poiseuille-qualifier-") as raw:
        root = Path(raw)
        run_positive(root)
        run_entry_evidence_fixtures(root)
        run_invalid_fixtures(root)
        run_scientific_fixtures(root)
    print("poiseuille_convergence_synthetic: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
