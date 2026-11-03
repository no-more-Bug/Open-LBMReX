#!/usr/bin/env python3
"""Narrow consumer for complete AMR evolution samples; exits 0/1/2."""
import argparse
import csv
import math
from pathlib import Path

EPS = 2.0**-52
REFERENCE = "independent-level-bgk-volume-v1"
IDENTITY = "schema sample descriptor backend real_digits ranks levels coarse_steps nx ny nz lx ly lz dt0 tau_finest rho_plot u_plot reference".split()
LAYOUT = "level box owner ilo jlo klo ihi jhi khi".split()
CLOCK = "event phase coarsest_step level local_step time_phys i j k".split()
MACRO = "rho ux uy uz".split()
NATIVE = "native_rho_norm native_ux_norm native_uy_norm native_uz_norm".split()
DIRECTIONS = {
    "D2Q9": [(0,0,0),(-1,1,0),(-1,0,0),(-1,-1,0),(0,-1,0),(1,-1,0),(1,0,0),(1,1,0),(0,1,0)],
    "D3Q27": [(0,0,0),(-1,0,0),(0,-1,0),(0,0,-1),(-1,-1,0),(-1,1,0),(-1,0,-1),(-1,0,1),
              (0,-1,-1),(0,-1,1),(-1,-1,-1),(-1,-1,1),(-1,1,-1),(-1,1,1),
              (1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,-1,0),(1,0,1),(1,0,-1),(0,1,1),(0,1,-1),
              (1,1,1),(1,1,-1),(1,-1,1),(1,-1,-1)],
}


class ArtifactError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ArtifactError(message)


def finite(text):
    value = float(text)
    require(math.isfinite(value), "nonfinite artifact value")
    return value


def rows(path, fields):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == fields, f"wrong CSV columns: {path.name}")
        for row in reader:
            require(None not in row and None not in row.values(), f"incomplete CSV row: {path.name}")
            yield row


def state_columns(q):
    return CLOCK + [f"{kind}_f{n}" for kind in ("actual", "reference") for n in range(q)] + MACRO + NATIVE


def within(actual, expected, scale=1.0):
    return (math.isfinite(actual) and math.isfinite(expected) and
            abs(actual-expected) <= 4096*EPS*max(scale, abs(actual), abs(expected)))


def events(levels, horizon, dt0):
    """Closed-form finest ticks: simultaneous completions are deepest first."""
    for lev in range(levels):
        yield "sync", 0, lev, 0, 0.0
    finest_ticks = 2**(levels-1)
    for coarse in range(1, horizon+1):
        for tick in range(1, finest_ticks+1):
            for lev in range(levels-1, 0, -1):
                stride = 2**(levels-1-lev)
                if tick % stride == 0:
                    step = (coarse-1)*2**lev + tick//stride
                    yield ("First" if step % 2 else "Second"), coarse, lev, step, step*dt0/2**lev
        for lev in range(levels):
            yield "sync", coarse, lev, coarse*2**lev, coarse*dt0


def moments(values, directions):
    rho = math.fsum(values)
    momentum = [math.fsum(v*c[d] for v, c in zip(values, directions)) for d in range(3)]
    return rho, momentum


def check_case(directory, sample, horizon=2):
    identity = list(rows(directory/"identity.csv", IDENTITY))
    require(len(identity) == 1, "identity must contain exactly one record")
    meta = identity[0]
    require(meta["schema"] == "amr-evolution-v1" and meta["sample"] == sample, "wrong schema/sample identity")
    require(meta["reference"] == REFERENCE, "wrong independent reference identity")
    require(meta["descriptor"] in DIRECTIONS and meta["backend"] in ("cpu", "cuda"), "unknown descriptor/backend")
    require(int(meta["real_digits"]) == 53 and int(meta["ranks"]) in (1, 2), "wrong precision/rank identity")
    levels, ranks = int(meta["levels"]), int(meta["ranks"])
    require(1 <= levels <= 4 and int(meta["coarse_steps"]) == horizon, "wrong level count/horizon identity")
    dimensions = 2 if meta["descriptor"] == "D2Q9" else 3
    directions = DIRECTIONS[meta["descriptor"]]
    q = len(directions)
    ratio = (2, 2, 1 if dimensions == 2 else 2)
    n0 = tuple(int(meta[key]) for key in ("nx", "ny", "nz"))
    length = tuple(finite(meta[key]) for key in ("lx", "ly", "lz"))
    dt0 = finite(meta["dt0"])
    require(all(n > 0 for n in n0) and all(x > 0 for x in length) and dt0 > 0, "invalid domain/time identity")
    require(dimensions != 2 or n0[2] == 1, "D2 must have one z cell")
    spacing = length[0]/n0[0]
    require(all(within(length[d]/n0[d], spacing, spacing) for d in range(dimensions)), "anisotropic active grid")
    require(within(dt0, 0.02*length[0]/n0[0], dt0) and finite(meta["tau_finest"]) == 0.65, "wrong physical configuration")
    rho_plot, u_plot = finite(meta["rho_plot"]), finite(meta["u_plot"])
    require(rho_plot == 2.0 and u_plot == 0.5, "wrong native normalization identity")
    cells = [set() for _ in range(levels)]
    box_ids = [set() for _ in range(levels)]
    for box in rows(directory/"layout.csv", LAYOUT):
        lev, box_id, owner = (int(box[k]) for k in ("level", "box", "owner"))
        require(0 <= lev < levels and 0 <= owner < ranks and box_id >= 0 and box_id not in box_ids[lev], "wrong layout level/owner/Box")
        box_ids[lev].add(box_id)
        lo = tuple(int(box[k]) for k in ("ilo", "jlo", "klo"))
        hi = tuple(int(box[k]) for k in ("ihi", "jhi", "khi"))
        require(all(0 <= lo[d] <= hi[d] < n0[d]*ratio[d]**lev for d in range(3)), "Box outside level domain")
        for k in range(lo[2], hi[2]+1):
            for j in range(lo[1], hi[1]+1):
                for i in range(lo[0], hi[0]+1):
                    point = (i, j, k)
                    require(point not in cells[lev], "overlapping actual valid Boxes")
                    cells[lev].add(point)
    require(len(cells[0]) == math.prod(n0) and all(cells), "missing actual valid coverage")
    covered = [{tuple(p[d]//ratio[d] for d in range(3)) for p in cells[lev+1]}
               if lev+1 < levels else set() for lev in range(levels)]
    volume = math.prod(length)
    measure = [volume/(math.prod(n0)*math.prod(ratio)**lev) for lev in range(levels)]
    count = sum(len(cells[lev]-covered[lev]) for lev in range(levels))
    require(within(sum(len(cells[lev]-covered[lev])*measure[lev] for lev in range(levels)), volume, volume),
            "composite layout does not cover the domain exactly once")
    ordered = [sorted(points, key=lambda p: (p[2], p[1], p[0])) for points in cells]
    stream = iter(rows(directory/"states.csv", state_columns(q)))
    failed = set()
    totals = {}  # Only synchronization frames participate in composite sums.
    maximum = 0.0
    event_count = 0
    for event, expected in enumerate(events(levels, horizon, dt0)):
        phase, coarse, lev, step, time = expected
        event_count += 1
        if phase == "sync" and coarse not in totals:
            totals[coarse] = [[[] for _ in range(4)] for _ in range(2)]
        for point in ordered[lev]:
            row = next(stream, None)
            require(row is not None, "missing sample or incomplete horizon")
            require((int(row["event"]), row["phase"], int(row["coarsest_step"]), int(row["level"]), int(row["local_step"]))
                    == (event, phase, coarse, lev, step), "wrong/duplicate event, phase, level or local step")
            require(tuple(int(row[k]) for k in ("i", "j", "k")) == point, "wrong/duplicate cell coordinate")
            require(within(finite(row["time_phys"]), time, dt0), "wrong physical time")
            actual = [finite(row[f"actual_f{n}"]) for n in range(q)]
            reference = [finite(row[f"reference_f{n}"]) for n in range(q)]
            for a, b in zip(actual, reference):
                maximum = max(maximum, abs(a-b)/max(1.0, abs(a), abs(b)))
                if not within(a, b):
                    failed.add("populations/reference")
            rho, momentum = moments(actual, directions)
            ref_rho, ref_momentum = moments(reference, directions)
            require(ref_rho > 0, "inadmissible reference density")
            if rho <= 0:
                failed.add("positive density")
            if phase == "sync":
                macro = [finite(row[key]) for key in MACRO]
                native = [finite(row[key])*(rho_plot if n == 0 else u_plot) for n, key in enumerate(NATIVE)]
                moment_values = [rho]+[p/rho if rho else 0.0 for p in momentum]
                reference_values = [ref_rho]+[p/ref_rho for p in ref_momentum]
                for n in range(4):
                    scale = 1.0 if n == 0 else 0.02
                    if not within(macro[n], moment_values[n], scale) or not within(macro[n], reference_values[n], scale):
                        failed.add("macroscopic moments/reference")
                    if not within(native[n], macro[n], scale):
                        failed.add("native readback")
                if point not in covered[lev]:
                    for source, values in enumerate(([rho]+momentum, [ref_rho]+ref_momentum)):
                        for n, value in enumerate(values):
                            totals[coarse][source][n].append(value*measure[lev])
            else:
                require(all(row[key] == "" for key in MACRO+NATIVE), "fine sample falsely publishes synchronized output")
    require(next(stream, None) is None, "extra/duplicate samples after the horizon")
    summed = {step: [[math.fsum(component) for component in source] for source in value]
              for step, value in totals.items()}
    initial = summed[0][0]
    require(initial[0] > 0, "nonpositive initial composite mass")
    eta = 4096*EPS*(1+math.ceil(math.log2(max(2, count))))
    for step, (actual, reference) in summed.items():
        for expected, label in ((reference, "composite/reference"), (initial, "composite conservation")):
            error_mass = abs(actual[0]-expected[0])/initial[0]
            error_momentum = math.sqrt(math.fsum((actual[n]-expected[n])**2 for n in range(1, 4)))/(initial[0]*0.02)
            if error_mass > eta or error_momentum > eta:
                failed.add(label)
    return {"sample": sample, "events": event_count, "composite_cells": count,
            "max_population_scaled_error": maximum, "failures": sorted(failed)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--samples", nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        require(args.steps > 0 and len(args.samples) == len(set(args.samples)), "invalid requested samples/horizon")
        results = [check_case(args.root/sample, sample, args.steps) for sample in args.samples]
    except (OSError, ValueError, OverflowError) as error:
        print(f"INVALID: {error}")
        return 2
    for result in results:
        print(result)
    return int(any(result["failures"] for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
