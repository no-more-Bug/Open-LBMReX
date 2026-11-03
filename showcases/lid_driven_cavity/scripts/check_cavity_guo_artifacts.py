#!/usr/bin/env python3
"""Validate geometry and diagnostic artifacts for an all-Guo cavity run."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def read_metadata(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    with path.open() as stream:
        for line in stream:
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def read_numeric_csv(path: Path) -> tuple[list[str], list[dict[str, float]]]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise SystemExit(f"{path}: missing CSV header")
        rows = [
            {key: float(value) for key, value in row.items()}
            for row in reader
        ]
    if not rows:
        raise SystemExit(f"{path}: no data rows")
    for row_index, row in enumerate(rows):
        for key, value in row.items():
            if not math.isfinite(value):
                raise SystemExit(
                    f"{path}: row {row_index} {key}={value} is not finite"
                )
    return list(reader.fieldnames), rows


def require_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=2.0e-14):
        raise SystemExit(f"{label}: expected {expected:.17g}, got {actual:.17g}")


def require_nonnegative(value: float, label: str) -> None:
    if value < 0.0:
        raise SystemExit(f"{label}: expected a nonnegative value, got {value:.17g}")


def check_centerline(
    path: Path,
    coordinate_name: str,
    intervals: int,
) -> None:
    _, rows = read_numeric_csv(path)
    if len(rows) != intervals + 1:
        raise SystemExit(
            f"{path}: expected {intervals + 1} boundary-inclusive nodes, "
            f"found {len(rows)}"
        )
    for index, row in enumerate(rows):
        require_close(
            row[coordinate_name],
            index / intervals,
            f"{path}: coordinate row {index}",
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--require-boundary-uz", action="store_true")
    parser.add_argument("--require-steady", action="store_true")
    parser.add_argument(
        "--require-mass",
        action="store_true",
        help="enable the optional closed-cavity mass diagnostic gate",
    )
    parser.add_argument("--boundary-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--uz-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--mass-tolerance", type=float, default=1.0e-8)
    args = parser.parse_args()
    for label, value in (
        ("boundary tolerance", args.boundary_tolerance),
        ("uz tolerance", args.uz_tolerance),
        ("mass tolerance", args.mass_tolerance),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise SystemExit(f"{label} must be finite and nonnegative: {value}")

    required_artifacts = (
        "cavity_centerline_metadata.txt",
        "cavity_centerline_x0p5_ux.csv",
        "cavity_centerline_y0p5_uy.csv",
        "cavity_validation_metrics.csv",
        "diagnostics.csv",
    )
    missing_artifacts = [
        name for name in required_artifacts if not (args.run_dir / name).is_file()
    ]
    if missing_artifacts:
        raise SystemExit(
            "missing required cavity artifact(s): "
            + ", ".join(missing_artifacts)
        )

    metadata = read_metadata(args.run_dir / "cavity_centerline_metadata.txt")
    expected_text = {
        "boundary_geometry": "on_lattice_valid_node",
        "sampling": "direct_node",
        "interpolation": "false",
        "endpoint_injection": "false",
        "valid_nodes_x": "129",
        "valid_nodes_y": "129",
        "lattice_intervals_x": "128",
        "lattice_intervals_y": "128",
    }
    for key, expected in expected_text.items():
        actual = metadata.get(key)
        if actual != expected:
            raise SystemExit(f"metadata {key}: expected {expected!r}, got {actual!r}")

    for key, expected in {
        "physical_lo_x": 0.0,
        "physical_hi_x": 1.0,
        "physical_lo_y": 0.0,
        "physical_hi_y": 1.0,
        "envelope_lo_x": -1.0 / 256.0,
        "envelope_hi_x": 1.0 + 1.0 / 256.0,
        "envelope_lo_y": -1.0 / 256.0,
        "envelope_hi_y": 1.0 + 1.0 / 256.0,
        "x_center_coordinate": 0.5,
        "y_center_coordinate": 0.5,
    }.items():
        require_close(float(metadata[key]), expected, f"metadata {key}")

    check_centerline(
        args.run_dir / "cavity_centerline_x0p5_ux.csv", "y_over_L", 128
    )
    check_centerline(
        args.run_dir / "cavity_centerline_y0p5_uy.csv", "x_over_L", 128
    )

    metrics_fields, metrics_rows = read_numeric_csv(
        args.run_dir / "cavity_validation_metrics.csv"
    )
    required_metrics = {
        "step",
        "relative_mass_error",
        "physical_weighted_relative_mass_error",
        "rho_min",
        "rho_max",
        "max_static_wall_velocity_error_over_Ulid",
        "max_moving_lid_velocity_error_over_Ulid",
        "max_moving_lid_tangential_error_over_Ulid",
        "max_moving_lid_normal_velocity_over_Ulid",
        "max_abs_uz_over_Ulid",
    }
    missing_metrics = required_metrics.difference(metrics_fields)
    if missing_metrics:
        raise SystemExit(
            "cavity_validation_metrics.csv is missing required columns: "
            + ", ".join(sorted(missing_metrics))
        )
    if len(metrics_rows) != 1:
        raise SystemExit("cavity metrics must contain exactly one final row")
    metrics = metrics_rows[0]
    metrics_step = metrics["step"]
    if metrics_step != int(metrics_step):
        raise SystemExit(f"cavity metrics step is not an integer: {metrics_step}")
    if int(metrics_step) != int(metadata["step"]):
        raise SystemExit("cavity metrics and centerline metadata steps differ")
    boundary_error_keys = (
        "max_static_wall_velocity_error_over_Ulid",
        "max_moving_lid_velocity_error_over_Ulid",
        "max_moving_lid_tangential_error_over_Ulid",
        "max_moving_lid_normal_velocity_over_Ulid",
    )
    for key in boundary_error_keys:
        require_nonnegative(metrics[key], key)
    require_nonnegative(metrics["max_abs_uz_over_Ulid"], "max_abs_uz_over_Ulid")
    require_nonnegative(metrics["relative_mass_error"], "relative_mass_error")
    require_nonnegative(
        metrics["physical_weighted_relative_mass_error"],
        "physical_weighted_relative_mass_error",
    )
    rho_min = metrics["rho_min"]
    rho_max = metrics["rho_max"]
    if rho_min <= 0.0 or rho_max <= 0.0 or rho_min > rho_max:
        raise SystemExit(
            f"invalid density range [{rho_min:.17g},{rho_max:.17g}]"
        )

    diagnostics_fields, diagnostics_rows = read_numeric_csv(
        args.run_dir / "diagnostics.csv"
    )
    required_diagnostics = {"step", "l2_uz", "linf_uz"}
    missing_diagnostics = required_diagnostics.difference(diagnostics_fields)
    if missing_diagnostics:
        raise SystemExit(
            "diagnostics.csv is missing required columns: "
            + ", ".join(sorted(missing_diagnostics))
        )
    diagnostic_steps = [row["step"] for row in diagnostics_rows]
    if any(step_value != int(step_value) for step_value in diagnostic_steps):
        raise SystemExit("diagnostics.csv contains a non-integer step")
    if any(
        current <= previous
        for previous, current in zip(diagnostic_steps, diagnostic_steps[1:])
    ):
        raise SystemExit("diagnostics.csv steps are not strictly increasing")
    final_diagnostics = diagnostics_rows[-1]
    if int(final_diagnostics["step"]) != int(metadata["step"]):
        raise SystemExit("diagnostics and centerline metadata final steps differ")

    lid_speed = float(metadata["lid_speed_lb"])
    if not math.isfinite(lid_speed) or lid_speed <= 0.0:
        raise SystemExit(f"metadata lid_speed_lb must be positive and finite: {lid_speed}")
    require_nonnegative(final_diagnostics["l2_uz"], "diagnostics final l2_uz")
    require_nonnegative(final_diagnostics["linf_uz"], "diagnostics final linf_uz")
    l2_uz_over_ulid = final_diagnostics["l2_uz"] / lid_speed
    linf_uz_over_ulid = final_diagnostics["linf_uz"] / lid_speed

    print(
        "guo_cavity_artifact_integrity passed "
        f"run={args.run_dir} step={int(metrics['step'])} "
        f"raw_mass_drift={metrics['relative_mass_error']:.12g} "
        "weighted_mass_drift="
        f"{metrics['physical_weighted_relative_mass_error']:.12g} "
        f"rho=[{metrics['rho_min']:.12g},{metrics['rho_max']:.12g}] "
        f"l2_uz/Ulid={l2_uz_over_ulid:.12g} "
        f"linf_uz/Ulid={linf_uz_over_ulid:.12g} "
        "sampling=direct_node interpolation=false endpoint_injection=false",
        flush=True,
    )
    if args.require_boundary_uz or args.require_steady:
        for key in boundary_error_keys:
            if metrics[key] > args.boundary_tolerance:
                raise SystemExit(
                    f"{key}={metrics[key]:.17g} exceeds "
                    f"{args.boundary_tolerance:.17g}"
                )
        if metrics["max_abs_uz_over_Ulid"] > args.uz_tolerance:
            raise SystemExit(
                "max_abs_uz_over_Ulid="
                f"{metrics['max_abs_uz_over_Ulid']:.17g} exceeds "
                f"{args.uz_tolerance:.17g}"
            )
        for key, value in (
            ("l2_uz_over_Ulid", l2_uz_over_ulid),
            ("linf_uz_over_Ulid", linf_uz_over_ulid),
        ):
            if value > args.uz_tolerance:
                raise SystemExit(
                    f"{key}={value:.17g} exceeds {args.uz_tolerance:.17g}"
                )
        print(
            "guo_cavity_boundary_uz_gate passed "
            f"boundary_tolerance={args.boundary_tolerance:.12g} "
            f"uz_tolerance={args.uz_tolerance:.12g}"
        )
    raw_mass_drift = metrics["relative_mass_error"]
    weighted_mass_drift = metrics["physical_weighted_relative_mass_error"]
    if args.require_mass:
        for label, mass_drift in (
            ("raw lattice", raw_mass_drift),
            ("physical weighted", weighted_mass_drift),
        ):
            if mass_drift > args.mass_tolerance:
                raise SystemExit(
                    f"closed-cavity {label} mass drift {mass_drift:.17g} "
                    f"exceeds {args.mass_tolerance:.17g}"
                )
        print(
            "guo_cavity_mass_gate passed "
            f"raw_mass_drift={raw_mass_drift:.12g} "
            f"weighted_mass_drift={weighted_mass_drift:.12g} "
            f"mass_tolerance={args.mass_tolerance:.12g}"
        )
    if args.require_steady:
        if metadata.get("steady_reached") != "true":
            raise SystemExit("formal Guo cavity run did not reach the steady gate")
        residual = float(metadata["relative_l2_delta_u"])
        tolerance = float(metadata["steady_l2_tol"])
        require_nonnegative(residual, "relative_l2_delta_u")
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise SystemExit(
                f"steady_l2_tol must be finite and nonnegative: {tolerance}"
            )
        if not residual < tolerance:
            raise SystemExit(
                f"steady residual {residual:.17g} is not below {tolerance:.17g}"
            )
        print(
            "guo_cavity_steady_gate passed "
            f"relative_l2_delta_u={residual:.12g} "
            f"steady_l2_tol={tolerance:.12g} "
            f"raw_mass_drift={raw_mass_drift:.12g} "
            f"weighted_mass_drift={weighted_mass_drift:.12g} "
            "mass_policy=record_only"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
