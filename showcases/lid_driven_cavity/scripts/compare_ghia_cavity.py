#!/usr/bin/env python3
"""Compare Stage-4 lid-driven cavity centerlines with Ghia Re=1000 data."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def read_xy(path: Path, x_name: str, y_name: str) -> list[tuple[float, float]]:
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        values = []
        for row in reader:
            coordinate = float(row[x_name])
            value = float(row[y_name])
            if not math.isfinite(coordinate) or not math.isfinite(value):
                raise RuntimeError(f"{path}: non-finite computed profile value")
            values.append((coordinate, value))
    values.sort(key=lambda item: item[0])
    return values


def read_reference(path: Path) -> list[tuple[float, float]]:
    with path.open(newline="") as fh:
        reader = csv.reader(fh)
        next(reader)
        values = []
        for row in reader:
            if not row:
                continue
            coordinate = float(row[0])
            value = float(row[1])
            if not math.isfinite(coordinate) or not math.isfinite(value):
                raise RuntimeError(f"{path}: non-finite reference profile value")
            values.append((coordinate, value))
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


def parse_bool(value: str | None) -> bool:
    return value is not None and value.lower() in {"1", "true", "yes"}


def add_or_replace_endpoint(
    values: list[tuple[float, float]],
    coord: float,
    velocity: float,
) -> list[tuple[float, float]]:
    filtered = [(x, y) for x, y in values if abs(x - coord) > 1.0e-14]
    filtered.append((coord, velocity))
    filtered.sort(key=lambda item: item[0])
    return filtered


def interpolate(values: list[tuple[float, float]], x: float) -> float:
    if x <= values[0][0]:
        return values[0][1]
    if x >= values[-1][0]:
        return values[-1][1]
    for left, right in zip(values, values[1:]):
        x0, y0 = left
        x1, y1 = right
        if x0 <= x <= x1:
            if x1 == x0:
                return y0
            weight = (x - x0) / (x1 - x0)
            return y0 + weight * (y1 - y0)
    raise RuntimeError(f"failed to interpolate x={x}")


def direct_node_value(
    values: list[tuple[float, float]],
    coord: float,
    lattice_intervals: int,
) -> float:
    node_index = round(lattice_intervals * coord)
    expected_coord = node_index / lattice_intervals
    coord_tol = 64.0 * math.ulp(max(1.0, abs(coord), abs(expected_coord)))
    if abs(coord - expected_coord) > coord_tol:
        raise RuntimeError(
            f"Ghia coordinate {coord:.17g} does not map to an on-lattice "
            f"node m/{lattice_intervals}"
        )

    matches = [value for x, value in values if abs(x - expected_coord) <= coord_tol]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one computed node at {node_index}/{lattice_intervals} "
            f"({expected_coord:.17g}), found {len(matches)}"
        )
    return matches[0]


def compare_profile(
    name: str,
    computed: list[tuple[float, float]],
    reference: list[tuple[float, float]],
    tolerance: float,
    zero_reference_tol: float,
    sampling: str,
    lattice_intervals: int | None,
) -> dict[str, float | str | int | bool]:
    if not computed or not reference:
        raise RuntimeError(f"{name}: computed/reference profile must not be empty")
    if any(
        not math.isfinite(coordinate) or not math.isfinite(value)
        for coordinate, value in computed + reference
    ):
        raise RuntimeError(f"{name}: computed/reference profile contains NaN or Inf")

    sum_sq = 0.0
    max_abs = -1.0
    max_metric = -1.0
    max_coord = 0.0
    max_ref = 0.0
    max_comp = 0.0

    for coord, ref in reference:
        if sampling == "direct_node":
            if lattice_intervals is None or lattice_intervals <= 0:
                raise RuntimeError(
                    "direct-node comparison requires positive lattice_intervals metadata"
                )
            comp = direct_node_value(computed, coord, lattice_intervals)
        else:
            comp = interpolate(computed, coord)
        err = comp - ref
        abs_err = abs(err)
        metric = abs_err / abs(ref) if abs(ref) > zero_reference_tol else abs_err
        sum_sq += err * err
        if abs_err > max_abs:
            max_abs = abs_err
        if metric > max_metric:
            max_metric = metric
            max_coord = coord
            max_ref = ref
            max_comp = comp

    l2 = math.sqrt(sum_sq / len(reference))
    return {
        "profile": name,
        "samples": len(reference),
        "l2": l2,
        "linf": max_abs,
        "max_metric": max_metric,
        "max_coord": max_coord,
        "max_reference": max_ref,
        "max_computed": max_comp,
        "sampling": sampling,
        "interpolation": sampling != "direct_node",
        "endpoint_injection": sampling != "direct_node",
        "pass": max_metric <= tolerance,
    }


def write_summary(path: Path, rows: list[dict[str, float | str | int | bool]]) -> None:
    fieldnames = [
        "profile",
        "samples",
        "l2",
        "linf",
        "max_metric",
        "max_coord",
        "max_reference",
        "max_computed",
        "sampling",
        "interpolation",
        "endpoint_injection",
        "pass",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docs" / "reference-data",
    )
    parser.add_argument("--tolerance", type=float, default=0.06)
    parser.add_argument("--zero-reference-tol", type=float, default=1.0e-12)
    parser.add_argument("--steady-l2-tol", type=float)
    parser.add_argument(
        "--ignore-steady",
        action="store_true",
        help="Skip the steady metadata gate for short smoke checks.",
    )
    parser.add_argument(
        "--record-only",
        action="store_true",
        help="Write comparison artifacts but do not fail the process on a numerical gate.",
    )
    args = parser.parse_args()

    metadata = read_metadata(args.run_dir / "cavity_centerline_metadata.txt")
    sampling = metadata.get("sampling", "interpolated_cell_center")
    if sampling not in {"direct_node", "interpolated_cell_center"}:
        raise RuntimeError(f"unsupported cavity sampling mode: {sampling}")
    lattice_intervals_x = (
        int(metadata["lattice_intervals_x"])
        if "lattice_intervals_x" in metadata
        else None
    )
    lattice_intervals_y = (
        int(metadata["lattice_intervals_y"])
        if "lattice_intervals_y" in metadata
        else None
    )

    ux_computed = read_xy(
        args.run_dir / "cavity_centerline_x0p5_ux.csv",
        "y_over_L",
        "ux_over_Ulid",
    )
    uy_computed = read_xy(
        args.run_dir / "cavity_centerline_y0p5_uy.csv",
        "x_over_L",
        "uy_over_Ulid",
    )

    if sampling != "direct_node":
        ux_computed = add_or_replace_endpoint(ux_computed, 0.0, 0.0)
        ux_computed = add_or_replace_endpoint(ux_computed, 1.0, 1.0)
        uy_computed = add_or_replace_endpoint(uy_computed, 0.0, 0.0)
        uy_computed = add_or_replace_endpoint(uy_computed, 1.0, 0.0)

    ux_reference = read_reference(args.reference_dir / "Re1000_x0.5_Ux_Ghia.csv")
    uy_reference = read_reference(args.reference_dir / "Re1000_y0.5_Uy_Ghia.csv")

    rows = [
        compare_profile(
            "x0.5_Ux",
            ux_computed,
            ux_reference,
            args.tolerance,
            args.zero_reference_tol,
            sampling,
            lattice_intervals_y,
        ),
        compare_profile(
            "y0.5_Uy",
            uy_computed,
            uy_reference,
            args.tolerance,
            args.zero_reference_tol,
            sampling,
            lattice_intervals_x,
        ),
    ]
    steady_reached = parse_bool(metadata.get("steady_reached"))
    relative_l2_delta_u = float(metadata.get("relative_l2_delta_u", "nan"))
    steady_l2_tol = (
        args.steady_l2_tol
        if args.steady_l2_tol is not None
        else float(metadata.get("steady_l2_tol", "5.0e-8"))
    )
    steady_pass = (
        args.ignore_steady or
        (steady_reached and relative_l2_delta_u < steady_l2_tol)
    )
    profile_pass = all(bool(row["pass"]) for row in rows)
    overall_pass = profile_pass and steady_pass

    summary_csv = args.run_dir / "ghia_comparison_summary.csv"
    write_summary(summary_csv, rows)
    summary_txt = args.run_dir / "ghia_comparison_summary.txt"
    with summary_txt.open("w") as fh:
        for row in rows:
            fh.write(
                f"{row['profile']}: pass={row['pass']} "
                f"l2={row['l2']:.12g} linf={row['linf']:.12g} "
                f"max_metric={row['max_metric']:.12g} "
                f"max_coord={row['max_coord']:.12g} "
                f"ref={row['max_reference']:.12g} "
                f"computed={row['max_computed']:.12g}\n"
            )
        fh.write(
            f"sampling: mode={sampling} "
            f"interpolation={sampling != 'direct_node'} "
            f"endpoint_injection={sampling != 'direct_node'} "
            f"lattice_intervals_x={lattice_intervals_x} "
            f"lattice_intervals_y={lattice_intervals_y}\n"
        )
        fh.write(
            f"steady_gate: pass={steady_pass} "
            f"ignore_steady={args.ignore_steady} "
            f"steady_reached={steady_reached} "
            f"relative_l2_delta_u={relative_l2_delta_u:.12g} "
            f"steady_l2_tol={steady_l2_tol:.12g}\n"
        )
        fh.write(
            f"overall: pass={overall_pass} "
            f"profile_pass={profile_pass} steady_pass={steady_pass} "
            f"record_only={args.record_only}\n"
        )
    for row in rows:
        print(
            f"{row['profile']}: pass={row['pass']} "
            f"l2={row['l2']:.12g} linf={row['linf']:.12g} "
            f"max_metric={row['max_metric']:.12g}"
        )
    print(
        f"sampling: mode={sampling} "
        f"interpolation={sampling != 'direct_node'} "
        f"endpoint_injection={sampling != 'direct_node'}"
    )
    print(
        f"steady_gate: pass={steady_pass} "
        f"steady_reached={steady_reached} "
        f"relative_l2_delta_u={relative_l2_delta_u:.12g}"
    )
    print(f"overall: pass={overall_pass}")
    return 0 if overall_pass or args.record_only else 1


if __name__ == "__main__":
    raise SystemExit(main())
