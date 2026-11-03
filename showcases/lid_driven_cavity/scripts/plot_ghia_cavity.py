#!/usr/bin/env python3
"""Plot Stage-4 lid-driven cavity centerlines against Ghia Re=1000 data."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_xy(path: Path, x_name: str, y_name: str) -> list[tuple[float, float]]:
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        values = [(float(row[x_name]), float(row[y_name])) for row in reader]
    values.sort(key=lambda item: item[0])
    return values


def read_reference(path: Path) -> list[tuple[float, float]]:
    with path.open(newline="") as fh:
        reader = csv.reader(fh)
        next(reader)
        values = [(float(row[0]), float(row[1])) for row in reader if row]
    values.sort(key=lambda item: item[0])
    return values


def read_metadata(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    with path.open() as fh:
        for line in fh:
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def add_or_replace_endpoint(
    values: list[tuple[float, float]],
    coord: float,
    velocity: float,
) -> list[tuple[float, float]]:
    filtered = [(x, y) for x, y in values if abs(x - coord) > 1.0e-14]
    filtered.append((coord, velocity))
    filtered.sort(key=lambda item: item[0])
    return filtered


def add_wall_endpoints(
    ux: list[tuple[float, float]],
    uy: list[tuple[float, float]],
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    ux = add_or_replace_endpoint(ux, 0.0, 0.0)
    ux = add_or_replace_endpoint(ux, 1.0, 1.0)
    uy = add_or_replace_endpoint(uy, 0.0, 0.0)
    uy = add_or_replace_endpoint(uy, 1.0, 0.0)
    return ux, uy


def split_xy(values: list[tuple[float, float]]) -> tuple[list[float], list[float]]:
    return [x for x, _ in values], [y for _, y in values]


def infer_computed_label(run_dir: Path) -> str:
    text = str(run_dir).lower()
    if "d3q27" in text:
        return "D3Q27"
    if "d2q9" in text:
        return "D2Q9"
    return "computed"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docs" / "reference-data",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output figure path. Defaults to run_dir/ghia_centerline_comparison.png.",
    )
    parser.add_argument(
        "--no-wall-endpoints",
        action="store_true",
        help="Plot only cell-center samples without appending known wall velocities.",
    )
    parser.add_argument(
        "--computed-label",
        help="Legend label for computed centerline data. Defaults to D2Q9/D3Q27 from run_dir.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    output = args.output or run_dir / "ghia_centerline_comparison.png"
    metadata = read_metadata(run_dir / "cavity_centerline_metadata.txt")
    sampling = metadata.get("sampling", "interpolated_cell_center")
    computed_label = args.computed_label or infer_computed_label(run_dir)
    if sampling == "direct_node":
        computed_label += " (direct nodes)"

    ux_computed = read_xy(
        run_dir / "cavity_centerline_x0p5_ux.csv",
        "y_over_L",
        "ux_over_Ulid",
    )
    uy_computed = read_xy(
        run_dir / "cavity_centerline_y0p5_uy.csv",
        "x_over_L",
        "uy_over_Ulid",
    )
    if sampling != "direct_node" and not args.no_wall_endpoints:
        ux_computed, uy_computed = add_wall_endpoints(ux_computed, uy_computed)

    ux_reference = read_reference(args.reference_dir / "Re1000_x0.5_Ux_Ghia.csv")
    uy_reference = read_reference(args.reference_dir / "Re1000_y0.5_Uy_Ghia.csv")

    ux_y, ux = split_xy(ux_computed)
    ux_ref_y, ux_ref = split_xy(ux_reference)
    uy_x, uy = split_xy(uy_computed)
    uy_ref_x, uy_ref = split_xy(uy_reference)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), constrained_layout=True)

    axes[0].plot(ux, ux_y, "-", color="#1f77b4", linewidth=1.8, label=computed_label)
    axes[0].scatter(
        ux_ref,
        ux_ref_y,
        marker="s",
        facecolors="none",
        edgecolors="#1f77b4",
        linewidths=1.2,
        s=38,
        label="Ghia Re=1000",
    )
    axes[0].set_xlabel(r"$u_x / U_{lid}$")
    axes[0].set_ylabel(r"$y / L$")
    axes[0].set_title(r"$x/L = 0.5$")
    axes[0].grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
    axes[0].legend(frameon=False)

    axes[1].plot(uy_x, uy, "-", color="#d62728", linewidth=1.8, label=computed_label)
    axes[1].scatter(
        uy_ref_x,
        uy_ref,
        marker="s",
        facecolors="none",
        edgecolors="#d62728",
        linewidths=1.2,
        s=38,
        label="Ghia Re=1000",
    )
    axes[1].set_xlabel(r"$x / L$")
    axes[1].set_ylabel(r"$u_y / U_{lid}$")
    axes[1].set_title(r"$y/L = 0.5$")
    axes[1].grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
    axes[1].legend(frameon=False)

    sampling_title = (
        "on-lattice direct-node sampling"
        if sampling == "direct_node"
        else "cell-center interpolation with wall endpoints"
    )
    fig.suptitle(f"Lid-driven cavity centerlines, Re=1000\n{sampling_title}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
