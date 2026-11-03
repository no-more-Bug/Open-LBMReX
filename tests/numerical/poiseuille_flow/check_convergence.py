#!/usr/bin/env python3
"""Check second-order Guo-pressure Poiseuille convergence."""

import csv
import math
import sys
from pathlib import Path

RESOLUTIONS = (10, 20, 50)


def read_error(path: Path, expected_n: int) -> float:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise ValueError(f"{path}: expected one row")
    row = rows[0]
    n_value = float(row["n_intervals_y"])
    step = int(row["step"])
    error = float(row["E_2_profile"])
    pressure = int(float(row["pressure_driven"]))
    scaling = int(float(row["pressure_scaling_code"]))
    if n_value != expected_n or pressure != 1 or scaling != 2:
        raise ValueError(f"{path}: wrong Guo-pressure identity")
    if step <= 0 or step > 200 * expected_n * expected_n:
        raise ValueError(f"{path}: invalid steady-run horizon")
    if not math.isfinite(error) or error <= 0.0:
        raise ValueError(f"{path}: invalid E_2_profile")
    return error


def check_log(path: Path, descriptor: str) -> None:
    text = path.read_text(encoding="utf-8")
    markers = (
        f"descriptor = {descriptor}",
        "case_name = poiseuille_flow",
        "case_variant = numerical_guo_pressure",
        "condition = pressure_inlet -> pressure_outlet",
        "scheme = guo_extrapolation",
        "profile = constant_density",
    )
    if any(marker not in text for marker in markers):
        raise ValueError(f"{path}: Guo-pressure identity mismatch")
    if "poiseuille_steady_reached = true" not in text:
        raise ValueError(f"{path}: steady completion is missing")


def main() -> int:
    if len(sys.argv) != 8 or sys.argv[1] not in {"d2q9", "d3q27"}:
        return 2
    try:
        errors = [read_error(Path(path), n) for path, n in zip(sys.argv[2:5], RESOLUTIONS)]
        descriptor = "D2Q9" if sys.argv[1] == "d2q9" else "D3Q27"
        for path in sys.argv[5:8]:
            check_log(Path(path), descriptor)
        xs = [math.log(1.0 / n) for n in RESOLUTIONS]
        ys = [math.log(error) for error in errors]
        xbar = sum(xs) / 3
        ybar = sum(ys) / 3
        order = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / sum(
            (x - xbar) ** 2 for x in xs
        )
    except (OSError, KeyError, ValueError, ZeroDivisionError) as error:
        print(f"invalid Poiseuille result: {error}", file=sys.stderr)
        return 2
    print(f"{sys.argv[1]} slope={order:.12g}")
    return 0 if 1.8 <= order <= 2.2 else 1


if __name__ == "__main__":
    sys.exit(main())
