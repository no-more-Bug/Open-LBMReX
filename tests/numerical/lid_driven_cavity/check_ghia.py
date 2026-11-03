#!/usr/bin/env python3
"""Compare HBB cavity centerlines with the test-owned Ghia Re=1000 data."""

import csv
import math
import sys
from pathlib import Path

TOLERANCE = 0.06
IDENTITIES = {
    "d2q9": ("D2Q9", "numerical_hbb_d2q9_re1000"),
    "d3q27": ("D3Q27", "numerical_hbb_d3q27_re1000"),
}


def read_key_values(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if " = " in line:
            key, value = line.split(" = ", 1)
            values[key.strip()] = value.strip()
    return values


def validate_identity(descriptor: str, log_path: Path, metadata_path: Path) -> None:
    expected_descriptor, expected_variant = IDENTITIES[descriptor]
    log = read_key_values(log_path)
    expected_log = {
        "descriptor": expected_descriptor,
        "case_name": "lid_driven_cavity",
        "case_variant": expected_variant,
        "wall_scheme": "halfway_bounce_back",
        "Re": "1000",
    }
    metadata = read_key_values(metadata_path)
    expected_metadata = {
        "boundary_geometry": "domain_face_halfway",
        "sampling": "interpolated_cell_center",
        "valid_nodes_x": "129",
        "valid_nodes_y": "129",
        "lattice_intervals_x": "129",
        "lattice_intervals_y": "129",
    }
    for key, expected in expected_log.items():
        if log.get(key) != expected:
            raise ValueError(f"{log_path}: expected {key} = {expected}")
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(f"{metadata_path}: expected {key} = {expected}")
    step = int(metadata["step"])
    steady = metadata.get("steady_reached")
    if step <= 0 or step > 200000 or steady not in {"true", "false"}:
        raise ValueError(f"{metadata_path}: invalid completion record")
    if steady != "true" and step != 200000:
        raise ValueError(f"{metadata_path}: run stopped before steady state or cap")


def read_computed(path: Path, coordinate: str, value: str) -> list[tuple[float, float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [(float(row[coordinate]), float(row[value])) for row in csv.DictReader(stream)]
    if not rows or not all(math.isfinite(x) and math.isfinite(y) for x, y in rows):
        raise ValueError(f"{path}: invalid computed profile")
    return sorted(rows)


def read_reference(path: Path) -> list[tuple[float, float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        next(reader)
        rows = [(float(row[0]), float(row[1])) for row in reader if row]
    if not rows or not all(math.isfinite(x) and math.isfinite(y) for x, y in rows):
        raise ValueError(f"{path}: invalid reference profile")
    return sorted(rows)


def interpolate(rows: list[tuple[float, float]], x: float) -> float:
    for (x0, y0), (x1, y1) in zip(rows, rows[1:]):
        if x0 <= x <= x1:
            return y0 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    if math.isclose(x, rows[0][0]):
        return rows[0][1]
    if math.isclose(x, rows[-1][0]):
        return rows[-1][1]
    raise ValueError(f"reference coordinate {x} is outside computed profile")


def compare(computed: list[tuple[float, float]], reference: list[tuple[float, float]]) -> float:
    maximum = 0.0
    for coordinate, expected in reference:
        actual = interpolate(computed, coordinate)
        error = abs(actual - expected)
        metric = error / abs(expected) if abs(expected) > 1.0e-12 else error
        maximum = max(maximum, metric)
    return maximum


def main() -> int:
    if len(sys.argv) != 5 or sys.argv[1] not in IDENTITIES:
        return 2
    descriptor = sys.argv[1]
    run_dir = Path(sys.argv[2])
    log_path = Path(sys.argv[3])
    reference_dir = Path(sys.argv[4])
    try:
        validate_identity(
            descriptor,
            log_path,
            run_dir / "cavity_centerline_metadata.txt",
        )
        ux = read_computed(run_dir / "cavity_centerline_x0p5_ux.csv", "y_over_L", "ux_over_Ulid")
        uy = read_computed(run_dir / "cavity_centerline_y0p5_uy.csv", "x_over_L", "uy_over_Ulid")
        ux = sorted([(x, y) for x, y in ux if not math.isclose(x, 0.0) and not math.isclose(x, 1.0)] + [(0.0, 0.0), (1.0, 1.0)])
        uy = sorted([(x, y) for x, y in uy if not math.isclose(x, 0.0) and not math.isclose(x, 1.0)] + [(0.0, 0.0), (1.0, 0.0)])
        ux_metric = compare(ux, read_reference(reference_dir / "Re1000_x0.5_Ux_Ghia.csv"))
        uy_metric = compare(uy, read_reference(reference_dir / "Re1000_y0.5_Uy_Ghia.csv"))
    except (OSError, KeyError, StopIteration, ValueError) as error:
        print(f"invalid cavity result: {error}", file=sys.stderr)
        return 2
    print(f"Ghia metrics: Ux={ux_metric:.12g} Uy={uy_metric:.12g}")
    return 0 if max(ux_metric, uy_metric) <= TOLERANCE else 1


if __name__ == "__main__":
    sys.exit(main())
