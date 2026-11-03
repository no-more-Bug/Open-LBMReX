#!/usr/bin/env python3
"""Compare all-Guo cavity centerlines and metrics across CPU layouts."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


OUTPUT_FILES = (
    "diagnostics.csv",
    "cavity_centerline_x0p5_ux.csv",
    "cavity_centerline_y0p5_uy.csv",
    "cavity_validation_metrics.csv",
)

TEXT_OUTPUT_FILES = ("cavity_centerline_metadata.txt",)


def read_numeric_csv(path: Path) -> tuple[list[str], list[list[float]]]:
    with path.open(newline="") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise SystemExit(f"{path}: empty CSV") from exc
        rows = [[float(value) for value in row] for row in reader if row]
    if not rows:
        raise SystemExit(f"{path}: no data rows")
    if any(not math.isfinite(value) for row in rows for value in row):
        raise SystemExit(f"{path}: contains a non-finite value")
    return header, rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--relative-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--absolute-tolerance", type=float, default=1.0e-12)
    args = parser.parse_args()
    if len(args.run_dirs) < 2:
        raise SystemExit("at least two run directories are required")

    max_abs = 0.0
    max_rel = 0.0
    for filename in TEXT_OUTPUT_FILES:
        reference = (args.run_dirs[0] / filename).read_bytes()
        for run_dir in args.run_dirs[1:]:
            candidate = (run_dir / filename).read_bytes()
            if candidate != reference:
                raise SystemExit(f"{run_dir / filename}: byte content mismatch")

    for filename in OUTPUT_FILES:
        reference_header, reference_rows = read_numeric_csv(
            args.run_dirs[0] / filename
        )
        for run_dir in args.run_dirs[1:]:
            header, rows = read_numeric_csv(run_dir / filename)
            if header != reference_header or len(rows) != len(reference_rows):
                raise SystemExit(f"{run_dir / filename}: schema/row-count mismatch")
            for row_index, (reference, candidate) in enumerate(
                zip(reference_rows, rows)
            ):
                if len(candidate) != len(reference):
                    raise SystemExit(
                        f"{run_dir / filename}: row {row_index} width mismatch"
                    )
                for column, (expected, actual) in enumerate(
                    zip(reference, candidate)
                ):
                    difference = abs(actual - expected)
                    relative = difference / max(abs(actual), abs(expected), 1.0)
                    max_abs = max(max_abs, difference)
                    max_rel = max(max_rel, relative)
                    if not math.isclose(
                        actual,
                        expected,
                        rel_tol=args.relative_tolerance,
                        abs_tol=args.absolute_tolerance,
                    ):
                        raise SystemExit(
                            f"{run_dir / filename}: row {row_index} column "
                            f"{column}={actual:.17g}, expected {expected:.17g}"
                        )

    print(
        "guo_cavity_parallel_check passed "
        f"runs={len(args.run_dirs)} max_abs_diff={max_abs:.12g} "
        f"max_rel_diff={max_rel:.12g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
