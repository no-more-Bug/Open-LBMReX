#!/usr/bin/env python3
"""Small complete and deliberately broken advance artifacts for the pairing consumer."""
import csv
import tempfile
from pathlib import Path
from check_amr_advance_parity import BALANCE, IDENTITY, compare_case, compare_roots, expected_cases
from check_amr_evolution import DIRECTIONS, LAYOUT, MACRO


def write(path, fields, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def fixture(root, descriptor, backend="cpu", ranks=1):
    d = 2 if descriptor == "D2Q9" else 3
    q = len(DIRECTIONS[descriptor])
    identity = dict(zip(IDENTITY, ["amr-advance-v1", "fixture", descriptor, backend, 53, ranks, 2, 2,
                                    0.65, 0, 2, 2, 1 if d == 2 else 2, 1, 1, 0.25 if d == 2 else 1, 0.01]))
    write(root/"identity.csv", IDENTITY, [identity])
    boxes, states = [], []
    for level in (0, 1):
        n = 2*2**level
        nz = 1 if d == 2 else n
        boxes.append(dict(zip(LAYOUT, [level, 0, 0, 0, 0, 0, n-1, n-1, nz-1])))
        for k in range(nz):
            for j in range(n):
                for i in range(n):
                    states.append([level, i, j, k])
    write(root/"layout.csv", LAYOUT, boxes)
    fields = "level step time_phys i j k".split()+[f"f{n}" for n in range(q)]+MACRO
    balance = []
    volume = 0.25 if d == 2 else 1.0
    for step in range(3):
        records = []
        for level, i, j, k in states:
            values = [level, step*2**level, step*0.01, i, j, k]+[1.0]+[0.0]*(q-1)+[1.0, 0.0, 0.0, 0.0]
            records.append(dict(zip(fields, values)))
        write(root/f"step-{step}"/"states.csv", fields, records)
        balance.append(dict(zip(BALANCE, [step, step*0.01, 4**d, volume, volume, 0, 0, 0, volume, 0, 0, 0])))
    write(root/"balance.csv", BALANCE, balance)


def mutate(path, column=None, value=None, drop=False, duplicate=False):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields, records = reader.fieldnames, list(reader)
    if drop:
        records.pop()
    elif duplicate:
        records.append(dict(records[-1]))
    else:
        records[0][column] = value
    write(path, fields, records)


def status(left, right, descriptor):
    try:
        return int(bool(compare_case(left, right, descriptor, ("fixture", 2, 0.65, False), (("cpu", 1), ("cuda", 1)))))
    except (OSError, ValueError, OverflowError):
        return 2


def main():
    with tempfile.TemporaryDirectory(dir=".") as temporary:
        left, right = Path(temporary)/"left", Path(temporary)/"right"
        for descriptor in DIRECTIONS:
            fixture(left, descriptor)
            fixture(right, descriptor, "cuda", 1)
            assert status(left, right, descriptor) == 0
            for backend in ("cpu", "cuda"):
                fixture(left, descriptor, backend, 1)
                fixture(right, descriptor, backend, 2)
                assert not compare_case(left, right, descriptor, ("fixture", 2, 0.65, False),
                                        ((backend, 1), (backend, 2)))
            fixture(left, descriptor)
            tests = [
                ("identity.csv", "schema", "old-static-v3", False, False, 2),
                ("identity.csv", "coarse_steps", 1, False, False, 2),
                ("identity.csv", "backend", "cpu", False, False, 2),
                ("identity.csv", "ranks", 2, False, False, 2),
                ("identity.csv", "uniform", 1, False, False, 2),
                ("layout.csv", "owner", 2, False, False, 2),
                ("step-2/states.csv", "level", 1, False, False, 2),
                ("step-2/states.csv", "step", 3, False, False, 2),
                ("step-2/states.csv", "time_phys", 0.01, False, False, 2),
                ("step-2/states.csv", "i", 1, False, False, 2),
                ("step-2/states.csv", "f0", "nan", False, False, 2),
                ("step-2/states.csv", None, None, True, False, 2),
                ("step-2/states.csv", None, None, False, True, 2),
                ("step-2/states.csv", "f0", 1.00001, False, False, 1),
                ("step-2/states.csv", "ux", 0.00001, False, False, 1),
                ("balance.csv", "mass", 1.1, False, False, 1),
                ("balance.csv", "cells", 1, False, False, 2),
            ]
            for filename, column, value, drop, duplicate, expected in tests:
                fixture(right, descriptor, "cuda", 1)
                mutate(right/filename, column, value, drop, duplicate)
                assert status(left, right, descriptor) == expected, (descriptor, filename, column)
            fixture(right, descriptor, "cuda", 1)
            (right/"step-2"/"states.csv").unlink()
            assert status(left, right, descriptor) == 2
            for group, count in (("advance", 12), ("matrix", 13 if descriptor == "D2Q9" else 15)):
                assert len(expected_cases(descriptor, group)) == count
                try:
                    compare_roots(left, right, descriptor, group, (("cpu", 1), ("cuda", 1)))
                except ValueError:
                    pass
                else:
                    raise AssertionError("missing complete case set accepted")
    print("advance parity consumer: D2/D3 complete fixtures and malformed/numerical failures PASS")


if __name__ == "__main__":
    main()
