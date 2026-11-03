#!/usr/bin/env python3
"""Compare public text and numeric artifacts across short regression runs."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def read_numeric(path: Path) -> tuple[list[str], list[list[float]]]:
    if not path.is_file():
        raise SystemExit(f"missing artifact: {path}")
    with path.open(newline="") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise SystemExit(f"{path}: empty CSV") from exc
        if any(not name for name in header) or len(set(header)) != len(header):
            raise SystemExit(f"{path}: empty or duplicate column name")
        rows = []
        for row_index, row in enumerate(reader):
            if not row:
                continue
            if len(row) != len(header):
                raise SystemExit(f"{path}: row {row_index} width mismatch")
            try:
                rows.append([float(value) for value in row])
            except ValueError as exc:
                raise SystemExit(
                    f"{path}: row {row_index} contains non-numeric data"
                ) from exc
    if not rows or any(not math.isfinite(value) for row in rows for value in row):
        raise SystemExit(f"{path}: missing rows or non-finite value")
    return header, rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--numeric", nargs="+", default=[])
    parser.add_argument("--text", nargs="+", default=[])
    parser.add_argument("--relative-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--absolute-tolerance", type=float, default=1.0e-12)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    args = parser.parse_args()
    if len(args.run_dirs) < 2:
        raise SystemExit("at least two run directories are required")
    reference_dir = args.run_dirs[0]
    max_difference = 0.0
    for name in args.text:
        expected = (reference_dir / name).read_bytes()
        for run_dir in args.run_dirs[1:]:
            if (run_dir / name).read_bytes() != expected:
                raise SystemExit(f"{run_dir / name}: text content mismatch")
    for name in args.numeric:
        expected_header, expected_rows = read_numeric(reference_dir / name)
        for run_dir in args.run_dirs[1:]:
            header, rows = read_numeric(run_dir / name)
            if header != expected_header or len(rows) != len(expected_rows):
                raise SystemExit(f"{run_dir / name}: schema/row-count mismatch")
            for row_index, (expected, actual) in enumerate(zip(expected_rows, rows)):
                if len(actual) != len(expected):
                    raise SystemExit(f"{run_dir / name}: row {row_index} width mismatch")
                for column, (lhs, rhs) in enumerate(zip(expected, actual)):
                    difference = abs(rhs - lhs)
                    max_difference = max(max_difference, difference)
                    if not math.isclose(
                        rhs, lhs,
                        rel_tol=args.relative_tolerance,
                        abs_tol=args.absolute_tolerance,
                    ):
                        raise SystemExit(
                            f"{run_dir / name}: row {row_index} column {column} "
                            f"{rhs:.17g} != {lhs:.17g}"
                        )
    print(
        f"public_artifact_parity passed runs={len(args.run_dirs)} "
        f"max_abs_diff={max_difference:.12g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
