#!/usr/bin/env python3
"""Compare case-owned Poiseuille outputs across MPI/OpenMP layouts."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


OUTPUT_FILES = (
    "poiseuille_profile_metrics.csv",
    "poiseuille_profile.csv",
    "poiseuille_density_pressure.csv",
)


def read_numeric_csv(path: Path) -> tuple[list[str], list[dict[str, float]]]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise SystemExit(f"{path}: missing CSV header")
        fieldnames = list(reader.fieldnames)
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
    return fieldnames, rows


def compare_file(
    reference_path: Path,
    candidate_path: Path,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> tuple[float, float]:
    reference_fields, reference_rows = read_numeric_csv(reference_path)
    candidate_fields, candidate_rows = read_numeric_csv(candidate_path)
    if candidate_fields != reference_fields:
        raise SystemExit(
            f"{candidate_path}: header differs from {reference_path}"
        )
    if len(candidate_rows) != len(reference_rows):
        raise SystemExit(
            f"{candidate_path}: row count {len(candidate_rows)} differs from "
            f"{reference_path}: {len(reference_rows)}"
        )

    max_absolute_difference = 0.0
    max_relative_difference = 0.0
    for row_index, (reference, candidate) in enumerate(
        zip(reference_rows, candidate_rows)
    ):
        for key in reference_fields:
            expected = reference[key]
            actual = candidate[key]
            absolute_difference = abs(actual - expected)
            scale = max(abs(expected), abs(actual), 1.0)
            relative_difference = absolute_difference / scale
            max_absolute_difference = max(
                max_absolute_difference, absolute_difference
            )
            max_relative_difference = max(
                max_relative_difference, relative_difference
            )
            if not math.isclose(
                actual,
                expected,
                rel_tol=relative_tolerance,
                abs_tol=absolute_tolerance,
            ):
                raise SystemExit(
                    f"{candidate_path}: row {row_index} {key}={actual:.17g} "
                    f"differs from reference {expected:.17g}"
                )
    return max_absolute_difference, max_relative_difference


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--relative-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--absolute-tolerance", type=float, default=1.0e-12)
    args = parser.parse_args()
    if len(args.run_dirs) < 2:
        raise SystemExit("at least two run directories are required")

    reference_dir = args.run_dirs[0]
    overall_max_absolute = 0.0
    overall_max_relative = 0.0
    for candidate_dir in args.run_dirs[1:]:
        for filename in OUTPUT_FILES:
            max_absolute, max_relative = compare_file(
                reference_dir / filename,
                candidate_dir / filename,
                args.relative_tolerance,
                args.absolute_tolerance,
            )
            overall_max_absolute = max(overall_max_absolute, max_absolute)
            overall_max_relative = max(overall_max_relative, max_relative)

    print(
        "poiseuille_parallel_check passed "
        f"runs={len(args.run_dirs)} "
        f"max_abs_diff={overall_max_absolute:.12g} "
        f"max_rel_diff={overall_max_relative:.12g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
