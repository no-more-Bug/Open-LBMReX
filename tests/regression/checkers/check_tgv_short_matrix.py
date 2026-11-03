#!/usr/bin/env python3
"""Validate direct D2Q9/D3Q27 Taylor-Green exact-error artifacts."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


REQUIRED = {
    "step",
    "time_lb",
    "time_phys",
    "mass",
    "relative_mass_error",
    "l2_ux",
    "l2_uy",
    "l2_uz",
    "l2_u",
    "l2_u_exact",
    "relative_l2_u",
    "linf_u",
    "linf_uz",
    "max_u",
    "max_rho_error",
}


def check_run(label: str, run_dir: Path) -> int:
    path = run_dir / "diagnostics.csv"
    if not path.is_file():
        raise SystemExit(f"{label}: missing artifact {path}")
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        header = reader.fieldnames or []
        if len(header) != len(set(header)) or any(not name for name in header):
            raise SystemExit(f"{label}: empty or duplicate diagnostics column")
        missing = REQUIRED - set(header)
        if missing:
            raise SystemExit(f"{label}: missing exact-error columns {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise SystemExit(f"{label}: diagnostics has no rows")
    for row_index, row in enumerate(rows):
        if None in row or any(value is None for value in row.values()):
            raise SystemExit(f"{label}: row {row_index} width mismatch")
        for name, value in row.items():
            try:
                numeric = float(value)
            except ValueError as exc:
                raise SystemExit(
                    f"{label}: row {row_index} {name} is not numeric"
                ) from exc
            if not math.isfinite(numeric):
                raise SystemExit(
                    f"{label}: row {row_index} {name}={numeric} is not finite"
                )
    if int(rows[-1]["step"]) != 2:
        raise SystemExit(f"{label}: expected final step 2, got {rows[-1]['step']}")
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d2", type=Path, required=True)
    parser.add_argument("--d3", type=Path, required=True)
    args = parser.parse_args()
    counts = {
        "d2": check_run("d2", args.d2),
        "d3": check_run("d3", args.d3),
    }
    print(f"tgv_short_matrix passed rows={counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
