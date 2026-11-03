#!/usr/bin/env python3
"""Qualify the approved Taylor--Green per-entry artifacts and raw logs."""

from __future__ import annotations

import argparse
import csv
import math
import re
import shlex
import sys
from pathlib import Path

EXPECTED_COLUMNS = (
    "N",
    "h",
    "Re",
    "target_time_phys",
    "time_phys",
    "time_lb",
    "step",
    "u0_lb",
    "nu_lb",
    "omega",
    "l2_ux",
    "l2_uy",
    "l2_uz",
    "l2_u",
    "l2_u_exact",
    "relative_l2_u",
    "linf_u",
    "linf_uz",
    "relative_mass_error",
)
EXPECTED_RESOLUTIONS = (50, 75, 100)
EXPECTED_STEPS = {50: 600, 75: 1350, 100: 2400}
EXPECTED_RE = 10.0
EXPECTED_TAU = 0.75
EXPECTED_NU_LB = 1.0 / 12.0
EXPECTED_OMEGA = 1.0 / EXPECTED_TAU
EXPECTED_TIME_PHYS = 0.2
SLOPE_MIN = 1.8
SLOPE_MAX = 2.2
TRANSVERSE_LIMIT = 1.0e-12

IDENTITIES = {
    "D2Q9": {
        "variant": "d2q9_periodic_2d",
        "depth": 1,
        "target": "./build/lbm_taylor_green_vortex",
        "input": "inputs/inputs.d2q9_tau_convergence",
        "entry_prefix": "tgv.d2q9",
        "historical_stage2_slope": 1.99957067873,
        "historical_stage4_bro_slope": 1.99957067873,
    },
    "D3Q27": {
        "variant": "d3q27_quasi_2d",
        "depth": 4,
        "target": "./build/lbm_taylor_green_vortex_d3q27",
        "input": "inputs/inputs.d3q27_tau_convergence",
        "entry_prefix": "tgv.d3q27",
        "historical_stage2_slope": 1.99957067868,
        "historical_stage4_bro_slope": 1.99957067871,
    },
}

ENTRY_IDENTITIES = {
    f"{identity['entry_prefix']}.n{n}": (descriptor, n)
    for descriptor, identity in IDENTITIES.items()
    for n in EXPECTED_RESOLUTIONS
}


class ArtifactError(RuntimeError):
    """The observed files do not prove the declared benchmark identity."""


def _close(actual: float, expected: float, *, absolute: float = 1.0e-12) -> bool:
    return math.isclose(actual, expected, rel_tol=1.0e-10, abs_tol=absolute)


def _finite_float(raw: str, field: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise ArtifactError(f"{field} is not a number: {raw!r}") from exc
    if not math.isfinite(value):
        raise ArtifactError(f"{field} must be finite")
    return value


def linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    count = len(xs)
    x_bar = sum(xs) / count
    y_bar = sum(ys) / count
    sxx = sum((x - x_bar) ** 2 for x in xs)
    sxy = sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys))
    if sxx == 0.0:
        raise ArtifactError("cannot fit convergence: all h values are identical")
    slope = sxy / sxx
    intercept = y_bar - slope * x_bar
    fitted = [slope * x + intercept for x in xs]
    ss_res = sum((y - estimate) ** 2 for y, estimate in zip(ys, fitted))
    ss_tot = sum((y - y_bar) ** 2 for y in ys)
    r2 = 1.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    return slope, intercept, r2


def parse_specs(raw_specs: list[str], option: str) -> dict[int, Path]:
    paths: dict[int, Path] = {}
    for raw in raw_specs:
        resolution, separator, path = raw.partition("=")
        if not separator or not resolution.isdigit() or not path:
            raise ArtifactError(
                f"invalid {option} {raw!r}; expected RESOLUTION=PATH"
            )
        n = int(resolution)
        if n in paths:
            raise ArtifactError(f"duplicate {option} for N={n}")
        paths[n] = Path(path).resolve()
    if tuple(sorted(paths)) != EXPECTED_RESOLUTIONS:
        raise ArtifactError(f"{option} must name exactly N=50,75,100")
    return paths


def read_single_row(csv_path: Path, expected_n: int) -> dict[str, float]:
    try:
        stream = csv_path.open(newline="", encoding="utf-8")
    except OSError as exc:
        raise ArtifactError(f"cannot read convergence CSV {csv_path}: {exc}") from exc

    with stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != EXPECTED_COLUMNS:
            raise ArtifactError(
                f"{csv_path}: convergence CSV schema changed; expected "
                + ",".join(EXPECTED_COLUMNS)
            )
        raw_rows = list(reader)

    if len(raw_rows) != 1:
        raise ArtifactError(
            f"{csv_path}: each entry artifact must contain exactly one data row"
        )
    raw = raw_rows[0]
    if None in raw:
        raise ArtifactError(f"{csv_path}: data row has unexpected extra values")

    if any(raw.get(field) in (None, "") for field in EXPECTED_COLUMNS):
        raise ArtifactError(f"{csv_path}: empty required value")

    row = {
        field: _finite_float(raw[field], f"{csv_path}:{field}")
        for field in EXPECTED_COLUMNS
    }
    n_value = row["N"]
    if float(int(n_value)) != n_value:
        raise ArtifactError(f"{csv_path}: N must be an integer")
    if int(n_value) != expected_n:
        raise ArtifactError(
            f"{csv_path}: artifact declared as N={expected_n} but row is N={n_value:g}"
        )
    return row


def validate_row_identity(n: int, row: dict[str, float]) -> None:
    expected_step = EXPECTED_STEPS[n]
    expected_values = {
        "h": 1.0 / n,
        "Re": EXPECTED_RE,
        "target_time_phys": EXPECTED_TIME_PHYS,
        "time_phys": EXPECTED_TIME_PHYS,
        "time_lb": float(expected_step),
        "step": float(expected_step),
        "u0_lb": 5.0 / (6.0 * n),
        "nu_lb": EXPECTED_NU_LB,
        "omega": EXPECTED_OMEGA,
    }
    for field, expected in expected_values.items():
        if not _close(row[field], expected):
            raise ArtifactError(
                f"N={n}: {field}={row[field]:.17g}, expected {expected:.17g}"
            )

    for field in (
        "l2_ux",
        "l2_uy",
        "l2_uz",
        "l2_u",
        "l2_u_exact",
        "relative_l2_u",
        "linf_u",
        "linf_uz",
        "relative_mass_error",
    ):
        if row[field] < 0.0:
            raise ArtifactError(f"N={n}: {field} must be nonnegative")
    if row["h"] <= 0.0 or row["l2_u_exact"] <= 0.0:
        raise ArtifactError(f"N={n}: h and l2_u_exact must be positive")
    if row["relative_l2_u"] <= 0.0:
        raise ArtifactError(f"N={n}: relative_l2_u must be positive")


def _read_log(
    path: Path,
) -> tuple[
    dict[str, str],
    dict[str, str],
    list[list[str]],
    list[str],
    tuple[int, str],
]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ArtifactError(f"cannot read raw log {path}: {exc}") from exc

    metadata: dict[str, str] = {}
    runtime: dict[str, str] = {}
    execution_prefixes: list[list[str]] = []
    command_tokens: list[str] = []
    completion: tuple[int, str] | None = None
    for raw_line in lines:
        line = raw_line.strip()
        if line.startswith("command:"):
            if command_tokens:
                raise ArtifactError(f"{path}: multiple command lines")
            try:
                command_tokens = shlex.split(line.removeprefix("command:").strip())
            except ValueError as exc:
                raise ArtifactError(f"{path}: malformed command line: {exc}") from exc
        if line.startswith("execution_prefix:"):
            try:
                execution_prefixes.append(
                    shlex.split(line.removeprefix("execution_prefix:").strip())
                )
            except ValueError as exc:
                raise ArtifactError(
                    f"{path}: malformed execution prefix: {exc}"
                ) from exc
        runtime_match = re.fullmatch(
            r"runtime_context: mpi_ranks=([0-9]+) "
            r"OMP_NUM_THREADS=([0-9]+) OMP_DYNAMIC=FALSE "
            r"OMP_PROC_BIND=(\S+) OMP_PLACES=(\S+) "
            r"amrex\.omp_threads=([0-9]+)",
            line,
        )
        if runtime_match:
            observed_runtime = dict(
                zip(
                    ("ranks", "threads", "proc_bind", "places", "amrex_threads"),
                    runtime_match.groups(),
                )
            )
            if runtime and runtime != observed_runtime:
                raise ArtifactError(f"{path}: conflicting runtime_context lines")
            runtime = observed_runtime
        match = re.fullmatch(r"([A-Za-z0-9_]+) = (.*)", line)
        if match:
            key, value = match.groups()
            if key in metadata and metadata[key] != value:
                raise ArtifactError(f"{path}: conflicting {key} values")
            metadata[key] = value
        match = re.search(
            r"LBM run complete: steps=([0-9]+), descriptor=([A-Za-z0-9]+)",
            line,
        )
        if match:
            observed = (int(match.group(1)), match.group(2))
            if completion is not None and completion != observed:
                raise ArtifactError(f"{path}: conflicting completion markers")
            completion = observed

    if not command_tokens:
        raise ArtifactError(f"{path}: missing command line")
    if not runtime:
        raise ArtifactError(f"{path}: missing canonical runtime_context")
    if completion is None:
        raise ArtifactError(f"{path}: missing completion marker")
    return metadata, runtime, execution_prefixes, command_tokens, completion


def _required(metadata: dict[str, str], key: str, path: Path) -> str:
    if key not in metadata:
        raise ArtifactError(f"{path}: missing {key} metadata")
    return metadata[key]


def _metadata_float(metadata: dict[str, str], key: str, path: Path) -> float:
    return _finite_float(_required(metadata, key, path), f"{path}:{key}")


def _resolve_from_workdir(raw_path: str, workdir: Path) -> Path:
    path = Path(raw_path)
    return path.resolve() if path.is_absolute() else (workdir / path).resolve()


def _require_under(path: Path, root: Path, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ArtifactError(
            f"{label} must stay under {root}, got {path}"
        ) from exc


def validate_command(
    *,
    path: Path,
    command_tokens: list[str],
    runtime: dict[str, str],
    metadata: dict[str, str],
    descriptor: str,
    n: int,
    formal_entry: bool,
) -> None:
    identity = IDENTITIES[descriptor]
    ranks = int(runtime["ranks"])
    threads = int(runtime["threads"])
    if ranks <= 0 or threads <= 0:
        raise ArtifactError(f"{path}: ranks and threads must be positive")
    if runtime["amrex_threads"] != runtime["threads"]:
        raise ArtifactError(
            f"{path}: amrex.omp_threads does not match OMP_NUM_THREADS"
        )

    if formal_entry:
        expected_runtime = {
            "ranks": "4",
            "threads": "1",
            "proc_bind": "close",
            "places": "cores",
            "amrex_threads": "1",
        }
        if runtime != expected_runtime:
            raise ArtifactError(
                f"{path}: formal entry runtime does not match canonical 4x1 layout"
            )

    expected: list[str] = []
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

    depth = int(identity["depth"])
    expected.extend(
        (
            str(identity["target"]),
            str(identity["input"]),
            f"taylor_green.n_cell={n}",
            str(n),
            str(depth),
            "mesh.max_grid_size=25",
            f"output.directory={_required(metadata, 'output_directory', path)}",
            "run.convergence_enabled=1",
            "run.kind=convergence",
            "run.convergence_reset=1",
            "output.convergence_directory=.",
            (
                "output.log_directory="
                + _required(metadata, "terminal_log_directory", path)
            ),
            "run.final_time_phys=0.2",
            "taylor_green.u_ref_phys=1.0",
            "output.plot_interval=0",
            f"amrex.omp_threads={threads}",
        )
    )
    if command_tokens != expected:
        raise ArtifactError(
            f"{path}: command does not match the approved direct command"
        )


def validate_log_identity(
    *,
    path: Path,
    artifact_path: Path,
    n: int,
    row: dict[str, float],
    descriptor: str,
    workdir: Path,
    formal_entry: bool,
) -> None:
    identity = IDENTITIES[descriptor]
    metadata, runtime, execution_prefixes, command_tokens, completion = _read_log(
        path
    )
    if formal_entry:
        if len(execution_prefixes) != 1:
            raise ArtifactError(
                f"{path}: formal entry requires exactly one execution_prefix"
            )
        prefix = execution_prefixes[0]
        expected_prefix = [
            "timeout",
            "--foreground",
            "--signal=TERM",
            "--kill-after=30s",
        ]
        if (
            len(prefix) != 5
            or prefix[:4] != expected_prefix
            or re.fullmatch(r"[1-9][0-9]*s", prefix[4]) is None
        ):
            raise ArtifactError(f"{path}: invalid formal execution_prefix")

    expected_text = {
        "descriptor": descriptor,
        "case_name": "taylor_green_vortex",
        "case_variant": str(identity["variant"]),
        "run_kind": "convergence",
        "convergence_mode": "1",
        "convergence_row_written": "true",
    }
    for key, expected in expected_text.items():
        actual = _required(metadata, key, path)
        if actual != expected:
            raise ArtifactError(
                f"{path}: {key}={actual!r}, expected {expected!r}"
            )

    for key in ("solver_exit_code", "tee_exit_code"):
        actual = _required(metadata, key, path)
        if actual != "0":
            raise ArtifactError(
                f"{path}: {key}={actual!r}, expected successful evidence"
            )

    actual_workdir = Path(_required(metadata, "working_directory", path)).resolve()
    if actual_workdir != workdir:
        raise ArtifactError(
            f"{path}: working_directory={actual_workdir}, expected {workdir}"
        )

    validate_command(
        path=path,
        command_tokens=command_tokens,
        runtime=runtime,
        metadata=metadata,
        descriptor=descriptor,
        n=n,
        formal_entry=formal_entry,
    )

    depth = int(identity["depth"])
    expected_n_cell = f"({n}, {n}, {depth})"
    actual_n_cell = _required(metadata, "n_cell", path)
    if actual_n_cell != expected_n_cell:
        raise ArtifactError(
            f"{path}: n_cell={actual_n_cell!r}, expected {expected_n_cell!r}"
        )

    raw_output = _required(metadata, "output_directory", path)
    raw_log_dir = _required(metadata, "terminal_log_directory", path)
    if Path(raw_output).is_absolute() or Path(raw_log_dir).is_absolute():
        raise ArtifactError(
            f"{path}: output/log command paths must be case-relative"
        )

    expected_output = artifact_path.parent.resolve()
    actual_output = _resolve_from_workdir(raw_output, workdir)
    if actual_output != expected_output:
        raise ArtifactError(
            f"{path}: output_directory={actual_output}, expected {expected_output}"
        )

    expected_log_dir = path.parent.resolve()
    actual_log_dir = _resolve_from_workdir(raw_log_dir, workdir)
    if actual_log_dir != expected_log_dir:
        raise ArtifactError(
            f"{path}: terminal_log_directory={actual_log_dir}, "
            f"expected {expected_log_dir}"
        )

    expected_step = EXPECTED_STEPS[n]
    if completion != (expected_step, descriptor):
        raise ArtifactError(
            f"{path}: completion={completion}, expected "
            f"({expected_step}, {descriptor})"
        )

    numeric_expectations = {
        "u_ref_lb": 5.0 / (6.0 * n),
        "nu_lb": EXPECTED_NU_LB,
        "tau": EXPECTED_TAU,
        "omega": EXPECTED_OMEGA,
        "Re": EXPECTED_RE,
        "max_step": float(expected_step),
        "final_diagnostic_step": float(expected_step),
        "final_time_phys": EXPECTED_TIME_PHYS,
        "final_relative_l2_u": row["relative_l2_u"],
    }
    for key, expected in numeric_expectations.items():
        actual = _metadata_float(metadata, key, path)
        absolute = 1.0e-14 if key == "final_relative_l2_u" else 1.0e-12
        if not _close(actual, expected, absolute=absolute):
            raise ArtifactError(
                f"{path}: {key}={actual:.17g}, expected {expected:.17g}"
            )


def validate_entry_evidence(
    *,
    artifact_path: Path,
    log_path: Path,
    descriptor: str,
    n: int,
    workdir: Path,
    formal_entry: bool,
) -> dict[str, float]:
    """Validate one entry's identity and evidence without a scientific verdict."""

    resolved_workdir = workdir.resolve()
    if not resolved_workdir.is_dir():
        raise ArtifactError(f"working directory does not exist: {resolved_workdir}")

    identity = IDENTITIES[descriptor]
    control = resolved_workdir / str(identity["input"])
    target = resolved_workdir / str(identity["target"])
    if not control.is_file():
        raise ArtifactError(f"approved input does not exist: {control}")
    if not target.is_file():
        raise ArtifactError(f"direct executable does not exist: {target}")

    artifact = _resolve_from_workdir(str(artifact_path), resolved_workdir)
    log = _resolve_from_workdir(str(log_path), resolved_workdir)
    _require_under(artifact, (resolved_workdir / "outputs").resolve(), "artifact")
    _require_under(log, (resolved_workdir / "log").resolve(), "raw log")

    row = read_single_row(artifact, n)
    validate_row_identity(n, row)
    validate_log_identity(
        path=log,
        artifact_path=artifact,
        n=n,
        row=row,
        descriptor=descriptor,
        workdir=resolved_workdir,
        formal_entry=formal_entry,
    )
    return row


def qualify(
    *,
    artifact_specs: list[str],
    log_specs: list[str],
    descriptor: str,
    workdir: Path,
) -> tuple[float, float, list[str]]:
    artifacts = parse_specs(artifact_specs, "--artifact")
    logs = parse_specs(log_specs, "--log")

    rows: dict[int, dict[str, float]] = {}
    for n in EXPECTED_RESOLUTIONS:
        rows[n] = validate_entry_evidence(
            artifact_path=artifacts[n],
            log_path=logs[n],
            descriptor=descriptor,
            n=n,
            workdir=workdir,
            formal_entry=False,
        )

    xs = [math.log(rows[n]["h"]) for n in EXPECTED_RESOLUTIONS]
    ys = [math.log(rows[n]["relative_l2_u"]) for n in EXPECTED_RESOLUTIONS]
    slope, _, r2 = linear_fit(xs, ys)
    failures: list[str] = []
    if not SLOPE_MIN <= slope <= SLOPE_MAX:
        failures.append(
            f"relative_l2_u slope {slope:.12g} outside "
            f"[{SLOPE_MIN}, {SLOPE_MAX}]"
        )
    if descriptor == "D3Q27":
        for n, row in rows.items():
            for field in ("l2_uz", "linf_uz"):
                if row[field] > TRANSVERSE_LIMIT:
                    failures.append(
                        f"N={n} {field}={row[field]:.12g} exceeds "
                        f"{TRANSVERSE_LIMIT:.12g}"
                    )
    return slope, r2, failures


def write_summary(
    *,
    path: Path,
    descriptor: str,
    slope: float,
    r2: float,
    failures: list[str],
) -> None:
    identity = IDENTITIES[descriptor]
    stage2 = float(identity["historical_stage2_slope"])
    stage4_bro = float(identity["historical_stage4_bro_slope"])
    entry_prefix = str(identity["entry_prefix"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        stream.write(f"descriptor = {descriptor}\n")
        stream.write(f"variant = {identity['variant']}\n")
        stream.write(
            "entries = "
            + ",".join(f"{entry_prefix}.n{n}" for n in EXPECTED_RESOLUTIONS)
            + "\n"
        )
        stream.write("resolutions = 50,75,100\n")
        stream.write(f"relative_l2_u_slope = {slope:.17g}\n")
        stream.write(f"r2 = {r2:.17g}\n")
        stream.write(f"slope_min = {SLOPE_MIN:.17g}\n")
        stream.write(f"slope_max = {SLOPE_MAX:.17g}\n")
        stream.write(f"historical_stage2_slope = {stage2:.17g}\n")
        stream.write(
            f"delta_vs_historical_stage2 = {slope - stage2:.17g}\n"
        )
        stream.write(f"historical_stage4_bro_slope = {stage4_bro:.17g}\n")
        stream.write(
            f"delta_vs_historical_stage4_bro = {slope - stage4_bro:.17g}\n"
        )
        stream.write(f"scientific_pass = {str(not failures).lower()}\n")
        for failure in failures:
            stream.write(f"failure = {failure}\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", action="append", required=True)
    parser.add_argument("--log", action="append", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--descriptor", choices=tuple(IDENTITIES))
    mode.add_argument("--entry-id", choices=tuple(ENTRY_IDENTITIES))
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    if args.entry_id is not None:
        if args.summary is not None:
            parser.error("--summary belongs only to three-point group mode")
        if len(args.artifact) != 1 or len(args.log) != 1:
            parser.error(
                "single-entry mode requires exactly one --artifact and --log"
            )
        descriptor, n = ENTRY_IDENTITIES[args.entry_id]
        try:
            validate_entry_evidence(
                artifact_path=Path(args.artifact[0]).resolve(),
                log_path=Path(args.log[0]).resolve(),
                descriptor=descriptor,
                n=n,
                workdir=args.workdir,
                formal_entry=True,
            )
        except (ArtifactError, csv.Error, OSError, ValueError) as exc:
            print(f"invalid artifact: {exc}", file=sys.stderr)
            return 2

        print(f"TGV entry evidence valid: entry_id={args.entry_id}")
        return 0

    if args.summary is None:
        parser.error("three-point group mode requires --summary")

    try:
        slope, r2, failures = qualify(
            artifact_specs=args.artifact,
            log_specs=args.log,
            descriptor=args.descriptor,
            workdir=args.workdir,
        )
        write_summary(
            path=args.summary,
            descriptor=args.descriptor,
            slope=slope,
            r2=r2,
            failures=failures,
        )
    except (ArtifactError, csv.Error, OSError, ValueError) as exc:
        print(f"invalid artifact: {exc}", file=sys.stderr)
        return 2

    if failures:
        for failure in failures:
            print(f"scientific failure: {failure}", file=sys.stderr)
        return 1

    print(
        f"TGV qualifier passed: descriptor={args.descriptor}, "
        f"slope={slope:.12g}, r2={r2:.12g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
