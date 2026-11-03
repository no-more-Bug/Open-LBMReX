#!/usr/bin/env python3
"""Small synthetic contracts for the evolution consumer, not solver validation."""
import contextlib
import copy
import csv
import io
from pathlib import Path
import tempfile
import unittest

import check_amr_evolution as checker


def fixture(dimensions, levels):
    descriptor = "D2Q9" if dimensions == 2 else "D3Q27"
    directions = checker.DIRECTIONS[descriptor]
    q = len(directions)
    velocity = [0.01, 0.007, 0.005 if dimensions == 3 else 0.0]
    if dimensions == 2:
        weights = [4/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9]
    else:
        weights = [{0: 8/27, 1: 2/27, 2: 1/54, 3: 1/216}[sum(c*c for c in direction)] for direction in directions]
    populations = []
    for weight, direction in zip(weights, directions):
        cu = sum(c*u for c, u in zip(direction, velocity))
        populations.append(weight*(1+3*cu+4.5*cu*cu-1.5*sum(u*u for u in velocity)))
    meta = dict(zip(checker.IDENTITY, ["amr-evolution-v1", "synthetic", descriptor, "cpu", 53, 1, levels, 2,
                    4, 4, 1 if dimensions == 2 else 4, 1, 1, 1/16 if dimensions == 2 else 1,
                    0.005, 0.65, 2, 0.5, checker.REFERENCE]))
    layout, cells = [], []
    for lev in range(levels):
        low = (0, 2, 6)[lev]
        high = low+3
        lo, hi = (low, low, low if dimensions == 3 else 0), (high, high, high if dimensions == 3 else 0)
        layout.append(dict(zip(checker.LAYOUT, [lev, 0, 0, *lo, *hi])))
        cells.append([(i, j, k) for k in range(lo[2], hi[2]+1) for j in range(lo[1], hi[1]+1) for i in range(lo[0], hi[0]+1)])
    schedule = [("sync", 0, lev, 0, 0.0) for lev in range(levels)]
    # Literal two/three-level completion order, independent of checker.events().
    local = [("First", 1, 1, 0.5), ("Second", 1, 2, 1.0)] if levels == 2 else [
        ("First", 2, 1, 0.25), ("Second", 2, 2, 0.5), ("First", 1, 1, 0.5),
        ("First", 2, 3, 0.75), ("Second", 2, 4, 1.0), ("Second", 1, 2, 1.0)]
    for coarse in (1, 2):
        for phase, lev, relative_step, fraction in local:
            schedule.append((phase, coarse, lev, (coarse-1)*2**lev+relative_step, (coarse-1+fraction)*0.005))
        schedule += [("sync", coarse, lev, coarse*2**lev, coarse*0.005) for lev in range(levels)]
    states = []
    for event, (phase, coarse, lev, step, time) in enumerate(schedule):
        for point in cells[lev]:
            row = dict(zip(checker.CLOCK, [event, phase, coarse, lev, step, time, *point]))
            for kind in ("actual", "reference"):
                row.update({f"{kind}_f{n}": value for n, value in enumerate(populations)})
            row.update(dict.fromkeys(checker.MACRO+checker.NATIVE, ""))
            if phase == "sync":
                row.update(dict(zip(checker.MACRO, [1, *velocity])))
                row.update(dict(zip(checker.NATIVE, [0.5, *(u/0.5 for u in velocity)])))
            states.append(row)
    return meta, layout, states, checker.state_columns(q)


def write(directory, data):
    meta, layout, states, columns = data
    directory.mkdir(exist_ok=True)
    for name, fields, values in (("identity", checker.IDENTITY, [meta]), ("layout", checker.LAYOUT, layout), ("states", columns, states)):
        with (directory/f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(values)


class ConsumerTests(unittest.TestCase):
    def consume(self, root):
        with contextlib.redirect_stdout(io.StringIO()):
            return checker.main(["--root", str(root), "--samples", "synthetic"])

    def test_valid_d2_d3_two_three_levels(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as name:
            root = Path(name)
            for dimensions in (2, 3):
                for levels in (2, 3):
                    with self.subTest(dimensions=dimensions, levels=levels):
                        write(root/"synthetic", fixture(dimensions, levels))
                        self.assertEqual(self.consume(root), 0)

    def test_status_and_identity_failures(self):
        baseline = fixture(2, 3)
        first_fine = next(i for i, row in enumerate(baseline[2]) if row["phase"] == "First")
        def change_row(index, key, value):
            return lambda data: data[2][index].__setitem__(key, value)
        cases = [
            ("missing", lambda data: data[2].pop(0), 2),
            ("duplicate", lambda data: data[2].insert(1, dict(data[2][0])), 2),
            ("level", change_row(first_fine, "level", 0), 2),
            ("local_step", change_row(first_fine, "local_step", 9), 2),
            ("coarsest_step", change_row(first_fine, "coarsest_step", 9), 2),
            ("time", change_row(first_fine, "time_phys", 0.5), 2),
            ("phase", change_row(first_fine, "phase", "Second"), 2),
            ("coordinate", change_row(0, "i", 99), 2),
            ("reference", lambda data: data[0].__setitem__("reference", "uniform-finest-wrong-reference"), 2),
            ("horizon_identity", lambda data: data[0].__setitem__("coarse_steps", 1), 2),
            ("incomplete_horizon", lambda data: data[2].pop(), 2),
            ("extra_row", lambda data: data[2].append(dict(data[2][-1])), 2),
            ("owner", lambda data: data[1][0].__setitem__("owner", 8), 2),
            ("box_index", lambda data: data[1][0].__setitem__("box", -1), 2),
            ("anisotropic", lambda data: data[0].__setitem__("ly", 2.0), 2),
            ("nonfinite", change_row(0, "actual_f0", "nan"), 2),
            ("false_fine_macro", change_row(first_fine, "rho", 1.0), 2),
            ("population_error", change_row(0, "actual_f0", 0.5), 1),
            ("macro_error", change_row(0, "ux", 0.02), 1),
            ("native_error", change_row(0, "native_rho_norm", 0.6), 1),
            ("native_inverse_overflow", change_row(0, "native_rho_norm", 1e308), 1),
        ]
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as name:
            root = Path(name)
            for label, mutate, expected in cases:
                with self.subTest(label=label):
                    data = copy.deepcopy(baseline)
                    mutate(data)
                    write(root/"synthetic", data)
                    self.assertEqual(self.consume(root), expected)
            write(root/"synthetic", baseline)
            (root/"synthetic"/"layout.csv").unlink()
            self.assertEqual(self.consume(root), 2)


if __name__ == "__main__":
    unittest.main()
