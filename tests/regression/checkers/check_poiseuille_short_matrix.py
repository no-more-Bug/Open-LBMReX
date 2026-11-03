#!/usr/bin/env python3
"""Validate exact schemas and identities for direct Poiseuille short runs."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


DIAGNOSTICS_COLUMNS = [
    "step", "time_lb", "time_phys", "mass", "relative_mass_error",
    "l2_ux", "l2_uy", "l2_uz", "l2_u", "l2_u_exact",
    "relative_l2_u", "linf_u", "linf_uz", "max_u", "max_rho_error",
]
PROFILE_COLUMNS = [
    "j", "y_phys", "eta", "ux_exact", "ux_center_zmean",
    "uy_abs_max", "uz_abs_max", "sample_i", "is_physical_boundary_node",
]
DENSITY_COLUMNS = [
    "i", "x_phys", "x_over_L", "rho_cross_section_weighted_mean",
    "pressure_lb_cross_section_weighted_mean",
]
METRICS_COLUMNS = [
    "step", "nx", "ny", "nz", "n_intervals_y",
    "on_lattice_boundary_nodes", "u_max_lb", "E_2_profile",
    "E_inf_profile", "E_2_profile_full", "E_inf_profile_full",
    "max_abs_uy_over_umax", "max_abs_uz_over_umax",
    "relative_mass_error", "raw_lattice_mass",
    "raw_lattice_reference_mass", "physical_weighted_mass",
    "physical_weighted_reference_mass",
    "physical_weighted_relative_mass_error", "rho_mean_minus_rho0",
    "rho_std_over_rho0", "rho_min", "rho_max", "rho_streamwise_xlo",
    "rho_streamwise_xhi", "pressure_streamwise_xlo",
    "pressure_streamwise_xhi", "rho_streamwise_slope",
    "pressure_streamwise_slope", "rho_streamwise_linear_r2",
    "pressure_driven", "prescribed_state", "pressure_scaling_code",
    "re_effective", "rho_in_target_lb", "rho_out_target_lb",
    "density_delta_target_lb", "density_delta_expected_lb",
    "density_delta_over_rho_ref", "mach",
    "max_abs_pressure_boundary_rho_error",
    "rho_streamwise_delta_measured",
    "rho_streamwise_delta_relative_error",
    "max_abs_profile_error_all_x_over_umax",
    "max_streamwise_ux_spread_over_umax",
]


def read_numeric_csv(path: Path, expected: list[str]) -> list[dict[str, float]]:
    if not path.is_file():
        raise SystemExit(f"missing artifact: {path}")
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        header = reader.fieldnames or []
        if (
            not header
            or any(not name for name in header)
            or len(set(header)) != len(header)
        ):
            raise SystemExit(f"{path}: empty or duplicate column name")
        if header != expected:
            raise SystemExit(
                f"{path}: header mismatch; expected {expected}, found {header}"
            )
        rows = list(reader)
    if not rows:
        raise SystemExit(f"{path}: no data rows")
    values: list[dict[str, float]] = []
    for row_index, row in enumerate(rows):
        if None in row or any(value is None for value in row.values()):
            raise SystemExit(f"{path}: row {row_index} width mismatch")
        try:
            numeric = {key: float(value) for key, value in row.items()}
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"{path}: row {row_index} contains non-numeric data") from exc
        for name, value in numeric.items():
            if not math.isfinite(value):
                raise SystemExit(
                    f"{path}: row {row_index} {name}={value} is not finite"
                )
        values.append(numeric)
    return values


def require_equal(label: str, actual: float, expected: float) -> None:
    if actual != expected:
        raise SystemExit(f"{label}: expected {expected}, found {actual}")

def require_close(
    label: str, actual: float, expected: float, tolerance: float = 1.0e-13
) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise SystemExit(f"{label}: expected {expected}, found {actual}")



def check_run(
    label: str,
    directory: Path,
    *,
    on_lattice: int,
    pressure_driven: int,
    prescribed_state: int,
    nz: int,
) -> None:
    diagnostics = read_numeric_csv(
        directory / "diagnostics.csv", DIAGNOSTICS_COLUMNS
    )
    profile = read_numeric_csv(
        directory / "poiseuille_profile.csv", PROFILE_COLUMNS
    )
    density = read_numeric_csv(
        directory / "poiseuille_density_pressure.csv", DENSITY_COLUMNS
    )
    metrics_rows = read_numeric_csv(
        directory / "poiseuille_profile_metrics.csv", METRICS_COLUMNS
    )
    if len(metrics_rows) != 1:
        raise SystemExit(f"{label}: expected one metrics row")
    metrics = metrics_rows[0]
    nx = 41 if on_lattice else 40
    ny = 11 if on_lattice else 10
    require_equal(f"{label}.step", metrics["step"], 2)
    require_equal(f"{label}.nx", metrics["nx"], nx)
    require_equal(f"{label}.ny", metrics["ny"], ny)
    require_equal(f"{label}.nz", metrics["nz"], nz)
    require_equal(f"{label}.n_intervals_y", metrics["n_intervals_y"], 10)
    require_equal(
        f"{label}.on_lattice_boundary_nodes",
        metrics["on_lattice_boundary_nodes"],
        on_lattice,
    )
    require_equal(
        f"{label}.pressure_driven",
        metrics["pressure_driven"],
        pressure_driven,
    )
    require_equal(
        f"{label}.prescribed_state",
        metrics["prescribed_state"],
        prescribed_state,
    )
    expected_scaling = 2 if pressure_driven or prescribed_state else 0
    require_equal(
        f"{label}.pressure_scaling_code",
        metrics["pressure_scaling_code"],
        expected_scaling,
    )
    require_close(f"{label}.u_max_lb", metrics["u_max_lb"], 0.1)
    require_close(f"{label}.re_effective", metrics["re_effective"], 12.0)
    if pressure_driven or prescribed_state:
        require_close(f"{label}.rho_in_target_lb", metrics["rho_in_target_lb"], 1.04)
        require_close(f"{label}.rho_out_target_lb", metrics["rho_out_target_lb"], 0.96)
        require_close(
            f"{label}.density_delta_target_lb",
            metrics["density_delta_target_lb"],
            0.08,
        )
        require_close(
            f"{label}.density_delta_expected_lb",
            metrics["density_delta_expected_lb"],
            0.08,
        )
    if len(profile) != ny:
        raise SystemExit(f"{label}: expected {ny} profile rows, found {len(profile)}")
    if len(density) != nx:
        raise SystemExit(f"{label}: expected {nx} density rows, found {len(density)}")
    if diagnostics[-1]["step"] != 2:
        raise SystemExit(f"{label}: final diagnostic step is not 2")

    strict_profile = profile[1:-1] if on_lattice else profile
    strict_error_squared = sum(
        (row["ux_center_zmean"] - row["ux_exact"]) ** 2
        for row in strict_profile
    )
    strict_exact_squared = sum(row["ux_exact"] ** 2 for row in strict_profile)
    full_error_squared = sum(
        (row["ux_center_zmean"] - row["ux_exact"]) ** 2
        for row in profile
    )
    full_exact_squared = sum(row["ux_exact"] ** 2 for row in profile)
    expected_e2 = math.sqrt(strict_error_squared / strict_exact_squared)
    expected_einf = max(
        abs(row["ux_center_zmean"] - row["ux_exact"])
        for row in strict_profile
    ) / metrics["u_max_lb"]
    expected_e2_full = math.sqrt(full_error_squared / full_exact_squared)
    expected_einf_full = max(
        abs(row["ux_center_zmean"] - row["ux_exact"])
        for row in profile
    ) / metrics["u_max_lb"]
    for field, expected in (
        ("E_2_profile", expected_e2),
        ("E_inf_profile", expected_einf),
        ("E_2_profile_full", expected_e2_full),
        ("E_inf_profile_full", expected_einf_full),
    ):
        require_close(f"{label}.{field}", metrics[field], expected, 1.0e-12)

    cs2 = 1.0 / 3.0
    for row_index, density_row in enumerate(density):
        require_close(
            f"{label}.pressure[{row_index}]",
            density_row["pressure_lb_cross_section_weighted_mean"],
            cs2 * density_row["rho_cross_section_weighted_mean"],
            1.0e-12,
        )
    rho_xlo = density[0]["rho_cross_section_weighted_mean"]
    rho_xhi = density[-1]["rho_cross_section_weighted_mean"]
    require_close(f"{label}.rho_streamwise_xlo", metrics["rho_streamwise_xlo"], rho_xlo)
    require_close(f"{label}.rho_streamwise_xhi", metrics["rho_streamwise_xhi"], rho_xhi)
    require_close(
        f"{label}.pressure_streamwise_xlo",
        metrics["pressure_streamwise_xlo"],
        cs2 * rho_xlo,
    )
    require_close(
        f"{label}.pressure_streamwise_xhi",
        metrics["pressure_streamwise_xhi"],
        cs2 * rho_xhi,
    )
    xi = [row["x_over_L"] for row in density]
    rho = [row["rho_cross_section_weighted_mean"] for row in density]
    xi_mean = sum(xi) / len(xi)
    rho_mean = sum(rho) / len(rho)
    xi_variance = sum((value - xi_mean) ** 2 for value in xi)
    rho_variance = sum((value - rho_mean) ** 2 for value in rho)
    covariance = sum(
        (x_value - xi_mean) * (rho_value - rho_mean)
        for x_value, rho_value in zip(xi, rho)
    )
    rho_slope = covariance / xi_variance
    rho_r2 = covariance * covariance / (xi_variance * rho_variance) \
        if rho_variance > 0.0 else 1.0
    require_close(f"{label}.rho_streamwise_slope", metrics["rho_streamwise_slope"], rho_slope)
    require_close(
        f"{label}.pressure_streamwise_slope",
        metrics["pressure_streamwise_slope"],
        cs2 * rho_slope,
    )
    require_close(f"{label}.rho_streamwise_linear_r2", metrics["rho_streamwise_linear_r2"], rho_r2)
    require_close(
        f"{label}.rho_streamwise_delta_measured",
        metrics["rho_streamwise_delta_measured"],
        rho_xlo - rho_xhi,
    )
    x_weights = [0.5] + [1.0] * (nx - 2) + [0.5] if on_lattice else [1.0] * nx
    physical_mass = sum(
        weight * rho_value for weight, rho_value in zip(x_weights, rho)
    ) * metrics["n_intervals_y"] * nz
    require_close(
        f"{label}.physical_weighted_mass",
        metrics["physical_weighted_mass"],
        physical_mass,
        1.0e-10,
    )
    expected_weighted_error = abs(
        physical_mass - metrics["physical_weighted_reference_mass"]
    ) / metrics["physical_weighted_reference_mass"]
    require_close(
        f"{label}.physical_weighted_relative_mass_error",
        metrics["physical_weighted_relative_mass_error"],
        expected_weighted_error,
        1.0e-12,
    )
    if not (
        metrics["rho_min"] <= min(rho)
        and metrics["rho_max"] >= max(rho)
        and metrics["max_abs_profile_error_all_x_over_umax"] + 1.0e-13
            >= metrics["E_inf_profile"]
    ):
        raise SystemExit(f"{label}: full-field metrics do not bound sampled artifacts")
    if on_lattice:
        require_close(f"{label}.eta_first", profile[0]["eta"], 0.0)
        require_close(f"{label}.eta_last", profile[-1]["eta"], 1.0)
        require_close(f"{label}.x_over_L_first", density[0]["x_over_L"], 0.0)
        require_close(f"{label}.x_over_L_last", density[-1]["x_over_L"], 1.0)
        require_equal(
            f"{label}.physical_node_first",
            profile[0]["is_physical_boundary_node"],
            1,
        )
        require_equal(
            f"{label}.physical_node_last",
            profile[-1]["is_physical_boundary_node"],
            1,
        )
    else:
        require_close(f"{label}.eta_first", profile[0]["eta"], 0.05)
        require_close(f"{label}.eta_last", profile[-1]["eta"], 0.95)
        require_close(f"{label}.x_over_L_first", density[0]["x_over_L"], 0.0125)
        require_close(f"{label}.x_over_L_last", density[-1]["x_over_L"], 0.9875)
        if any(row["is_physical_boundary_node"] != 0 for row in profile):
            raise SystemExit(
                f"{label}: HBB profile must not mark valid physical nodes"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hbb-d2q9", required=True, type=Path)
    parser.add_argument("--guo-velocity-d2q9", required=True, type=Path)
    parser.add_argument("--guo-pressure-d2q9", required=True, type=Path)
    parser.add_argument("--guo-prescribed-d2q9", required=True, type=Path)
    parser.add_argument("--representative-d3q27", required=True, type=Path)
    args = parser.parse_args()
    check_run(
        "hbb_d2q9", args.hbb_d2q9,
        on_lattice=0, pressure_driven=0, prescribed_state=0, nz=1,
    )
    check_run(
        "guo_velocity_d2q9", args.guo_velocity_d2q9,
        on_lattice=1, pressure_driven=0, prescribed_state=0, nz=1,
    )
    check_run(
        "guo_pressure_d2q9", args.guo_pressure_d2q9,
        on_lattice=1, pressure_driven=1, prescribed_state=0, nz=1,
    )
    check_run(
        "guo_prescribed_d2q9", args.guo_prescribed_d2q9,
        on_lattice=1, pressure_driven=0, prescribed_state=1, nz=1,
    )
    check_run(
        "representative_d3q27", args.representative_d3q27,
        on_lattice=1, pressure_driven=1, prescribed_state=0, nz=4,
    )
    print("poiseuille_short_matrix passed: four D2 variants and D3 Guo pressure")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
