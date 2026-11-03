#!/usr/bin/env python3
"""Compare complete tests-owned v3 states: 0 pass, 1 numerical mismatch, 2 bad artifact."""

import argparse
import csv
import math
import os
from pathlib import Path

from static_hierarchy_artifacts import (NumericalMismatch, close, read_identity,
                                        require, validate_geometry, validate_state)


VARIABLE_IDENTITY = {"ranks", "backend", "max_grid_size"}


def columns(q):
    return ["level", "step", "time_phys", "i", "j", "k"] + [f"f{n}" for n in range(q)] + ["rho", "ux", "uy", "uz"]


def load_snapshot(directory, source_id=None):
    identity = read_identity(directory / "identity.txt")
    sample_directory = directory
    if sample_directory.name in {"amrcore", "single_level"}:
        sample_directory = sample_directory.parent
    if sample_directory.name.startswith("box") and sample_directory.name[3:].isdigit():
        sample_directory = sample_directory.parent
    require(identity["sample"] == sample_directory.name, "sample identity differs from requested artifact path")
    geometry = validate_geometry(identity, source_id)
    values = {}
    all_analytic = True
    with (directory / "states.csv").open(newline="") as stream:
        reader = csv.reader(stream)
        expected_columns = columns(geometry.q)
        require(next(reader, None) == expected_columns, "invalid states.csv columns")
        for row in reader:
            require(len(row) == len(expected_columns), "incomplete state row")
            level, step = int(row[0]), int(row[1])
            time = float(row[2])
            cell = tuple(map(int, row[3:6]))
            require(0 <= level < len(geometry.coverage) and step in geometry.times, "unexpected level/step")
            require(math.isfinite(time) and math.isclose(time, geometry.times[step], rel_tol=1e-12, abs_tol=1e-15),
                    "row physical time differs from identity")
            require(cell in geometry.coverage[level], "state outside actual valid Boxes")
            key = (level, step, *cell)
            require(key not in values, f"duplicate state coordinate: {key}")
            state = tuple(map(float, row[6:]))
            all_analytic = validate_state(geometry, key, state) and all_analytic
            values[key] = state
    # Every unique row lies in A; exact count therefore proves every A cell at every step.
    require(len(values) == len(geometry.steps)*sum(map(len, geometry.coverage)),
            "incomplete states: actual valid Boxes/steps contain missing cells")
    return identity, values, all_analytic


def physical_identity(identity):
    result = {key: value for key, value in identity.items()
              if key not in VARIABLE_IDENTITY and not key.startswith("boxes_")}
    if result["regions"] != "none":
        result["regions"] = tuple(sorted(set(result["regions"].split(";"))))
    return result


def compare_pair(left, right, source_id=None):
    a, av, a_analytic = load_snapshot(left, source_id)
    b, bv, b_analytic = load_snapshot(right, source_id)
    require(physical_identity(a) == physical_identity(b), "physical/source identity mismatch")
    if av.keys() != bv.keys():
        # Different legal AMReX coverage cannot be paired. Every state must have
        # passed an independent analytic oracle; moments alone are insufficient.
        require(a["coverage_contract"] == "minimum-repair" and a_analytic and b_analytic,
                "different coverage requires independent analytic values for every state")
        return 0, len(av)+len(bv), None
    maximum = 0.0
    for key in av:
        for x, y in zip(av[key], bv[key]):
            maximum = max(maximum, abs(x-y))
            close(x, y, f"paired state {key}")
    return 0, len(av), maximum


def discover(root):
    require(root.is_dir(), f"missing snapshot directory: {root}")
    return {path.parent.relative_to(root).as_posix() for path in root.rglob("identity.txt")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path)
    parser.add_argument("--right", type=Path)
    parser.add_argument("--samples", nargs="+")
    parser.add_argument("--source-id", default=os.environ.get("LBM_EXPECTED_GIT_COMMIT"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            from test_static_hierarchy_artifacts import run
            return run()
        require(args.left is not None and args.right is not None, "--left and --right are required")
        left, right = discover(args.left), discover(args.right)
        if args.samples:
            samples = set(args.samples)
            require(len(samples) == len(args.samples) and samples <= left and samples <= right,
                    "explicit sample list is duplicated or incomplete")
        else:
            require(left and left == right, "snapshot sample sets differ or are empty")
            samples = left
        status = 0
        for sample in sorted(samples):
            try:
                code, cells, maximum = compare_pair(args.left/sample, args.right/sample, args.source_id)
                comparison = "independent-analytic" if maximum is None else "full-coordinate"
                difference = "not-compared" if maximum is None else f"{maximum:.17g}"
                print(f"static_parity sample={sample} states={cells} comparison={comparison} max_abs={difference} status={code}")
            except NumericalMismatch as error:
                status = 1
                print(f"static_parity sample={sample} numerical_mismatch: {error}")
        return status
    except NumericalMismatch as error:
        print(f"static_parity numerical_mismatch: {error}")
        return 1
    except (OSError, ValueError, KeyError, csv.Error) as error:
        print(f"static_parity artifact_error: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
