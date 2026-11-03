#!/usr/bin/env python3
"""Render optional Poiseuille plots without owning scientific acceptance."""

from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from pathlib import Path

def _pyplot():
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "matplotlib-codex"),
    )
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Matplotlib is optional and required only for requested plots"
        ) from exc
    return pyplot


def read_single_row(path: Path) -> dict[str, float]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise SystemExit(f"expected one row in {path}, found {len(rows)}")
    return {key: float(value) for key, value in rows[0].items()}


def read_profile(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as f:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(f)]


def require_finite(values: dict[str, float], path: Path) -> None:
    for key, value in values.items():
        if not math.isfinite(value):
            raise SystemExit(f"{path}: {key}={value} is not finite")


def drive_flags(metrics: dict[str, float]) -> tuple[bool, bool]:
    pressure_flag = int(metrics["pressure_driven"])
    prescribed_flag = int(metrics["prescribed_state"])
    if pressure_flag not in (0, 1) or prescribed_flag not in (0, 1):
        raise SystemExit("Poiseuille drive flags must be zero or one")
    if pressure_flag + prescribed_flag > 1:
        raise SystemExit("Poiseuille run cannot be pressure- and prescribed-state driven")
    return pressure_flag == 1, prescribed_flag == 1


def plot_density_pressure(run_dir: Path, metrics: dict[str, float]) -> None:
    plt = _pyplot()
    data_path = run_dir / "poiseuille_density_pressure.csv"
    if not data_path.exists():
        raise SystemExit(f"missing required artifact {data_path}")
    rows = read_profile(data_path)
    if len(rows) != int(metrics["nx"]):
        raise SystemExit(
            f"{data_path}: expected nx={int(metrics['nx'])} rows, "
            f"found {len(rows)}"
        )
    if not rows:
        raise SystemExit(f"no density/pressure rows found in {data_path}")
    for row in rows:
        require_finite(row, data_path)

    x_over_l = [row["x_over_L"] for row in rows]
    if any(right <= left for left, right in zip(x_over_l, x_over_l[1:])):
        raise SystemExit(f"{data_path}: x_over_L must be strictly increasing")
    rho = [row["rho_cross_section_weighted_mean"] for row in rows]
    pressure = [
        row["pressure_lb_cross_section_weighted_mean"] for row in rows
    ]

    fig, axes = plt.subplots(
        2, 1, figsize=(6.8, 6.3), sharex=True, constrained_layout=True
    )
    axes[0].plot(x_over_l, rho, "o-", markersize=3.2, color="#2b6cb0")
    axes[0].set_ylabel(r"cross-section $\bar{\rho}$")
    pressure_driven, prescribed_state = drive_flags(metrics)
    drive = (
        "Prescribed-state"
        if prescribed_state
        else "Pressure-driven" if pressure_driven else "Velocity-driven"
    )
    axes[0].set_title(f"{drive} Poiseuille streamwise density and pressure")
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(x_over_l, pressure, "o-", markersize=3.2, color="#c05621")
    axes[1].set_xlabel(r"$(x-x_{lo})/L_x$")
    axes[1].set_ylabel(r"$\bar{p}_{lb}=c_s^2\bar{\rho}$")
    axes[1].grid(True, alpha=0.25)
    out_path = run_dir / "poiseuille_density_pressure.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"density_pressure_plot={out_path}")


def plot_profile(run_dir: Path) -> dict[str, float]:
    plt = _pyplot()
    metrics_path = run_dir / "poiseuille_profile_metrics.csv"
    profile_path = run_dir / "poiseuille_profile.csv"
    metrics = read_single_row(metrics_path)
    profile = read_profile(profile_path)
    require_finite(metrics, metrics_path)
    if not profile:
        raise SystemExit(f"no profile rows found in {profile_path}")
    for row in profile:
        require_finite(row, profile_path)

    n_intervals_y = int(metrics["n_intervals_y"])
    on_lattice = int(metrics["on_lattice_boundary_nodes"]) != 0
    if on_lattice:
        if len(profile) != n_intervals_y + 1:
            raise SystemExit(
                f"{profile_path}: expected N+1={n_intervals_y + 1} rows, "
                f"found {len(profile)}"
            )
        if not math.isclose(profile[0]["eta"], 0.0, abs_tol=1.0e-13):
            raise SystemExit(f"{profile_path}: first boundary node is not eta=0")
        if not math.isclose(profile[-1]["eta"], 1.0, abs_tol=1.0e-13):
            raise SystemExit(f"{profile_path}: last boundary node is not eta=1")

    eta_points = [row["eta"] for row in profile]
    ux_exact = [row["ux_exact"] for row in profile]
    ux_sample_key = "ux_center_zmean"
    ux_samples = [row[ux_sample_key] for row in profile]
    eta_curve = [i / 400.0 for i in range(401)]
    u_max = metrics["u_max_lb"]
    ux_curve = [4.0 * u_max * eta * (1.0 - eta) for eta in eta_curve]

    fig, ax = plt.subplots(figsize=(6.4, 4.4), constrained_layout=True)
    ax.plot(ux_curve, eta_curve, label="theory", color="black", linewidth=1.8)
    ax.scatter(
        ux_samples,
        eta_points,
        label="LBM center-x samples",
        s=32,
        color="#2b6cb0",
    )
    ax.set_xlabel(r"$u_x$ (lattice units)")
    ax.set_ylabel(r"$y/H$")
    pressure_driven, prescribed_state = drive_flags(metrics)
    drive = (
        "Prescribed-state"
        if prescribed_state
        else "Pressure-driven" if pressure_driven else "Velocity-profile"
    )
    ax.set_title(
        f"{drive} Poiseuille, N_y={n_intervals_y}, "
        f"E2={metrics['E_2_profile']:.3g}"
    )
    ax.grid(True, alpha=0.25)
    ax.legend()
    out_path = run_dir / "poiseuille_profile.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"profile_plot={out_path}")
    plot_density_pressure(run_dir, metrics)
    return metrics


def fit_log_slope(points: list[tuple[float, float]]) -> tuple[float, float]:
    xs = [math.log(h) for h, _ in points]
    ys = [math.log(err) for _, err in points]
    n = len(points)
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom == 0.0:
        raise SystemExit("cannot fit convergence slope with identical h values")
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denom
    intercept = y_mean - slope * x_mean
    return slope, intercept


def write_convergence(
    metrics_rows: list[dict[str, float]],
    output_dir: Path,
    scaling_label: str | None,
) -> None:
    plt = _pyplot()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(
        metrics_rows,
        key=lambda row: row["n_intervals_y"],
    )
    points = [
        (1.0 / row["n_intervals_y"], row["E_2_profile"])
        for row in rows
    ]
    slope, intercept = fit_log_slope(points)
    csv_path = output_dir / "poiseuille_convergence_plot_data.csv"
    streamwise_fields = [
        "rho_streamwise_xlo",
        "rho_streamwise_xhi",
        "pressure_streamwise_xlo",
        "pressure_streamwise_xhi",
        "rho_streamwise_slope",
        "pressure_streamwise_slope",
        "rho_streamwise_linear_r2",
    ]
    pressure_fields = [
        "pressure_driven",
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
    ]
    pressure_scaling_fields = [
        "pressure_scaling_code",
        "re_effective",
    ]
    prescribed_state_fields = ["prescribed_state"]
    pressure_flags = {int(row["pressure_driven"]) for row in rows}
    if not pressure_flags <= {0, 1} or len(pressure_flags) != 1:
        raise SystemExit(
            "Poiseuille convergence cannot mix pressure drive flags"
        )
    pressure_driven = pressure_flags == {1}
    prescribed_flags = {int(row["prescribed_state"]) for row in rows}
    if not prescribed_flags <= {0, 1} or len(prescribed_flags) != 1:
        raise SystemExit(
            "Poiseuille convergence cannot mix prescribed-state drive flags"
        )
    prescribed_state = prescribed_flags == {1}
    if pressure_driven and prescribed_state:
        raise SystemExit(
            "Poiseuille convergence cannot be pressure- and prescribed-state driven"
        )
    scaling_codes = {
        "fixed_mach_characterization": 1,
        "fixed_re_fixed_tau": 2,
    }
    resolved_scaling = "not_applicable"
    if pressure_driven:
        codes = {int(row["pressure_scaling_code"]) for row in rows}
        if len(codes) != 1:
            raise SystemExit(
                "Poiseuille convergence cannot mix pressure scaling codes"
            )
        reverse_codes = {value: key for key, value in scaling_codes.items()}
        code = next(iter(codes))
        if code not in reverse_codes:
            raise SystemExit(f"unsupported pressure scaling code {code}")
        resolved_scaling = (
            scaling_label if scaling_label is not None else reverse_codes[code]
        )
        expected_code = scaling_codes.get(resolved_scaling)
        if expected_code is None:
            raise SystemExit(
                f"unsupported pressure scaling label {resolved_scaling}"
            )
        for row in rows:
            actual_code = int(row["pressure_scaling_code"])
            if actual_code != expected_code:
                raise SystemExit(
                    "pressure scaling label/code mismatch: "
                    f"label={resolved_scaling}, code={actual_code}"
                )
    elif scaling_label is not None:
        resolved_scaling = scaling_label
    if resolved_scaling == "fixed_re_fixed_tau":
        re_values = [row["re_effective"] for row in rows]
        re_scale = max(1.0, *(abs(value) for value in re_values))
        if min(re_values) <= 0.0 or max(re_values) - min(re_values) > 2.0e-12 * re_scale:
            raise SystemExit(
                "fixed_re_fixed_tau convergence requires one positive Re_effective"
            )
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        header = [
            "ny",
            "valid_nodes_y",
            "h",
            "u_max_lb",
            "E_2_profile",
            "E_inf_profile",
            "E_2_profile_full",
            "E_inf_profile_full",
            "max_abs_uy_over_umax",
            "max_abs_uz_over_umax",
            "relative_mass_error",
            "physical_weighted_relative_mass_error",
        ]
        if resolved_scaling != "not_applicable":
            header.append("convergence_scaling")
        header.extend(streamwise_fields)
        if pressure_driven:
            header.append("pressure_scaling")
        header.extend(pressure_fields)
        header.extend(pressure_scaling_fields)
        header.extend(prescribed_state_fields)
        writer.writerow(header)
        for row in rows:
            n_intervals_y = row["n_intervals_y"]
            values = [
                int(n_intervals_y),
                int(row["ny"]),
                1.0 / n_intervals_y,
                row["u_max_lb"],
                row["E_2_profile"],
                row["E_inf_profile"],
                row["E_2_profile_full"],
                row["E_inf_profile_full"],
                row["max_abs_uy_over_umax"],
                row["max_abs_uz_over_umax"],
                row["relative_mass_error"],
                row["physical_weighted_relative_mass_error"],
            ]
            if resolved_scaling != "not_applicable":
                values.append(resolved_scaling)
            values.extend(row[field] for field in streamwise_fields)
            if pressure_driven:
                values.append(resolved_scaling)
            values.extend(row[field] for field in pressure_fields)
            values.extend(row[field] for field in pressure_scaling_fields)
            values.extend(row[field] for field in prescribed_state_fields)
            writer.writerow(values)

    fig, ax = plt.subplots(figsize=(6.2, 4.3), constrained_layout=True)
    hs = [h for h, _ in points]
    errs = [err for _, err in points]
    ax.loglog(hs, errs, "o-", label=f"fit slope={slope:.3f}")
    ref_h0 = hs[0]
    ref_e0 = errs[0]
    ref = [ref_e0 * (h / ref_h0) ** 2 for h in hs]
    ax.loglog(hs, ref, "--", color="gray", label="2nd-order reference")
    ax.set_xlabel(r"$h=1/N_y$")
    ax.set_ylabel(r"$E_{2,profile}$")
    if prescribed_state:
        scaling_title = resolved_scaling.replace("_", " ")
        drive = "Prescribed-state" if resolved_scaling == "not_applicable" else (
            f"Prescribed-state ({scaling_title})"
        )
    elif pressure_driven:
        scaling_title = resolved_scaling.replace("_", " ")
        drive = f"Pressure-driven ({scaling_title})"
    else:
        scaling_title = resolved_scaling.replace("_", " ")
        drive = "Velocity-profile" if resolved_scaling == "not_applicable" else (
            f"Velocity-profile ({scaling_title})"
        )
    ax.set_title(f"{drive} Poiseuille fixed-tau convergence")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    png_path = output_dir / "poiseuille_convergence.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    summary_path = output_dir / "poiseuille_convergence_plot_metadata.txt"
    pressure_scaling = resolved_scaling if pressure_driven else "not_applicable"
    summary_path.write_text(
        f"slope={slope:.12g}\n"
        f"intercept={intercept:.12g}\n"
        f"pressure_driven={str(pressure_driven).lower()}\n"
        f"prescribed_state={str(prescribed_state).lower()}\n"
        f"pressure_scaling={pressure_scaling}\n"
        f"convergence_scaling={resolved_scaling}\n"
        "visualization_only=true\n",
        encoding="utf-8",
    )
    print(f"plot_data_csv={csv_path}")
    print(f"convergence_plot={png_path}")
    print(f"plot_metadata={summary_path}")
    print(f"convergence_slope_visualization={slope:.12g}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--convergence-output", type=Path)
    parser.add_argument(
        "--scaling-label",
        choices=(
            "fixed_mach_characterization",
            "fixed_re_fixed_tau",
        ),
        help="series scaling metadata written to convergence outputs",
    )
    args = parser.parse_args()

    metrics_rows = [plot_profile(run_dir) for run_dir in args.run_dirs]
    if len(metrics_rows) > 1:
        output_dir = args.convergence_output
        if output_dir is None:
            output_dir = args.run_dirs[0].parent / "convergence"
        write_convergence(
            metrics_rows,
            output_dir,
            args.scaling_label,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
