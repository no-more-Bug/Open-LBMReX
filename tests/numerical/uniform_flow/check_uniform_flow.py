#!/usr/bin/env python3
"""Check the two approved uniform-flow hard gates."""

import csv
import math
import sys
from pathlib import Path

LIMIT = 1.0e-8
IDENTITIES = {
    "hbb_velocity_io.d2q9": (
        "D2Q9", "numerical_uniform_hbb",
        "velocity_inlet -> velocity_outlet", "halfway_bounce_back",
    ),
    "hbb_velocity_io.d3q27": (
        "D3Q27", "numerical_uniform_hbb",
        "velocity_inlet -> velocity_outlet", "halfway_bounce_back",
    ),
    "guo_velocity_io.d2q9": (
        "D2Q9", "numerical_uniform_guo_velocity",
        "velocity_inlet -> velocity_outlet", "guo_extrapolation",
    ),
    "guo_velocity_io.d3q27": (
        "D3Q27", "numerical_uniform_guo_velocity",
        "velocity_inlet -> velocity_outlet", "guo_extrapolation",
    ),
    "guo_free_outflow.d2q9": (
        "D2Q9", "numerical_uniform_guo_free_outflow",
        "velocity_inlet -> free_outflow", "guo_extrapolation",
    ),
    "guo_free_outflow.d3q27": (
        "D3Q27", "numerical_uniform_guo_free_outflow",
        "velocity_inlet -> free_outflow", "guo_extrapolation",
    ),
}


def check_log(path: Path, expected: tuple[str, str, str, str]) -> None:
    descriptor, variant, condition, scheme = expected
    text = path.read_text(encoding="utf-8")
    markers = (
        f"descriptor = {descriptor}",
        "case_name = uniform_flow",
        f"case_variant = {variant}",
        f"condition = {condition}",
        f"scheme = {scheme}",
    )
    if any(marker not in text for marker in markers):
        raise ValueError(f"{path}: uniform-flow identity mismatch")


def main() -> int:
    if len(sys.argv) != 7:
        return 2
    failed = False
    observed: set[str] = set()
    try:
        for spec in sys.argv[1:]:
            parts = spec.split("|")
            if len(parts) != 3:
                raise ValueError(f"invalid artifact spec: {spec}")
            identity, raw_path, raw_log = parts
            if identity not in IDENTITIES or identity in observed:
                raise ValueError(f"unknown or duplicate identity: {identity}")
            observed.add(identity)
            check_log(Path(raw_log), IDENTITIES[identity])
            with Path(raw_path).open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            if len(rows) != 1:
                raise ValueError(f"{raw_path}: expected one row")
            step = int(rows[0]["step"])
            flow_throughs = float(rows[0]["flow_throughs"])
            e2 = float(rows[0]["E_2_u"])
            mass = float(rows[0]["relative_mass_error"])
            if step != 10000 or not math.isclose(flow_throughs, 100.0):
                raise ValueError(f"{raw_path}: incomplete 100-flow-through horizon")
            if not math.isfinite(e2) or not math.isfinite(mass) or e2 < 0.0 or mass < 0.0:
                raise ValueError(f"{raw_path}: invalid metric")
            passed = e2 <= LIMIT and mass <= LIMIT
            failed = failed or not passed
            print(f"{identity}: E_2_u={e2:.12g} relative_mass_error={mass:.12g}")
        if observed != set(IDENTITIES):
            raise ValueError("uniform-flow identity set is incomplete")
    except (OSError, KeyError, ValueError) as error:
        print(f"invalid uniform-flow result: {error}", file=sys.stderr)
        return 2
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
