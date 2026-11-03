#!/usr/bin/env python3
"""Check second-order Taylor-Green velocity convergence."""

import csv
import math
import sys
from pathlib import Path

RESOLUTIONS = (50, 75, 100)


def read_row(path: Path, expected_n: int) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise ValueError(f"{path}: expected one row")
    required = (
        "N",
        "h",
        "Re",
        "target_time_phys",
        "time_phys",
        "relative_l2_u",
        "linf_uz",
    )
    if any(name not in rows[0] or rows[0][name] == "" for name in required):
        raise ValueError(f"{path}: missing required columns")
    values = {name: float(rows[0][name]) for name in required}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError(f"{path}: non-finite value")
    if values["N"] != expected_n or not math.isclose(values["h"], 1.0 / expected_n):
        raise ValueError(f"{path}: wrong resolution identity")
    if not math.isclose(values["Re"], 10.0):
        raise ValueError(f"{path}: wrong Reynolds-number identity")
    if not math.isclose(values["target_time_phys"], 0.2) or not math.isclose(
        values["time_phys"], 0.2
    ):
        raise ValueError(f"{path}: incomplete physical-time horizon")
    if values["relative_l2_u"] <= 0.0 or values["linf_uz"] < 0.0:
        raise ValueError(f"{path}: invalid error value")
    return values


def slope(rows: list[dict[str, float]]) -> float:
    xs = [math.log(row["h"]) for row in rows]
    ys = [math.log(row["relative_l2_u"]) for row in rows]
    xbar = sum(xs) / len(xs)
    ybar = sum(ys) / len(ys)
    return sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / sum(
        (x - xbar) ** 2 for x in xs
    )


def check_log(path: Path, descriptor: str) -> None:
    text = path.read_text(encoding="utf-8")
    expected = (
        f"descriptor = {descriptor}",
        "case_name = taylor_green_vortex",
        f"case_variant = numerical_tgv_{descriptor.lower()}",
    )
    if any(marker not in text for marker in expected):
        raise ValueError(f"{path}: Taylor-Green identity mismatch")


def main() -> int:
    if len(sys.argv) != 8 or sys.argv[1] not in {"d2q9", "d3q27"}:
        return 2
    try:
        rows = [read_row(Path(path), n) for path, n in zip(sys.argv[2:5], RESOLUTIONS)]
        descriptor = "D2Q9" if sys.argv[1] == "d2q9" else "D3Q27"
        for path in sys.argv[5:8]:
            check_log(Path(path), descriptor)
        order = slope(rows)
        transverse = max(row["linf_uz"] for row in rows)
    except (OSError, ValueError, ZeroDivisionError) as error:
        print(f"invalid TGV result: {error}", file=sys.stderr)
        return 2
    passed = 1.8 <= order <= 2.2
    if sys.argv[1] == "d3q27":
        passed = passed and transverse <= 1.0e-12
    print(f"{sys.argv[1]} slope={order:.12g} max_linf_uz={transverse:.12g}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
