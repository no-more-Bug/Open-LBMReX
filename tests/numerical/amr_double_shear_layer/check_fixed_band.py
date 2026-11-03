#!/usr/bin/env python3
"""Check fixed-band DSL data and geometric relations; no scientific error gate."""
import argparse
import csv
import math
from pathlib import Path
import numpy as np

from analyze_dynamic_double_shear_layer import open_fixed_band
from check_dynamic_double_shear_layer import check as check_diagnostics
from amr_plotfile import ArtifactError, require, rows


def grid_history(run, layouts):
    history = {}
    ordered = []
    for row in rows(run.directory/"grid-history.csv"):
        require(set(row) == {"step", "time", "level", "valid_cells", "boxes"}, "wrong grid-history schema")
        step, lev, cells, boxes = (int(row[key]) for key in ("step", "level", "valid_cells", "boxes"))
        require((step, lev) not in history and 1 <= step <= int(run.id["requested_steps"]) and
                0 <= lev <= 2 and 0 < boxes <= cells and
                math.isclose(float(row["time"]), step*.1/run.n, rel_tol=1e-12, abs_tol=1e-14),
                "missing/duplicate/invalid grid-history row")
        n = run.n*2**lev
        domain_cells = n*n*(n//32 if run.dimensions == 3 else 1)
        require(cells <= domain_cells and (lev != 0 or cells == domain_cells), "invalid grid-history coverage count")
        history[step, lev] = cells, boxes
        ordered.append((step, lev))
    require(ordered == sorted(ordered), "grid-history rows are out of order")
    last = int(run.id["requested_steps"])
    for step in range(1, last+1):
        levels = [lev for lev in range(3) if (step, lev) in history]
        require(levels and levels == list(range(len(levels))), "missing grid-history step/level")
    for frame, step in enumerate(run.steps):
        if step == 0:
            continue
        expected = {(step, lev): (sum(math.prod(b-a for a, b in zip(lo, hi)) for lo, hi in boxes), len(boxes))
                    for lev, boxes in enumerate(layouts[frame])}
        require({key: value for key, value in history.items() if key[0] == step} == expected,
                "grid-history disagrees with the native saved frame")
    initial_l2 = sum(math.prod(b-a for a, b in zip(lo, hi)) for lo, hi in layouts[0][2]) if len(layouts[0]) == 3 else 0
    previous = initial_l2
    creations = decreases = l2_steps = 0
    for step in range(1, last+1):
        current = history.get((step, 2), (0, 0))[0]
        creations += current > 0 and previous == 0
        decreases += current < previous
        l2_steps += current > 0
        previous = current
    # These are observations, not an activity or scientific acceptance gate.
    return dict(grid_steps=last, l2_present_steps=l2_steps, l2_creation_events=creations,
                l2_cell_decreases=decreases)


def check(run, analysis):
    frames = check_diagnostics(run, analysis)
    # The data checker has matched every native plot BoxArray against these
    # layout ledgers. Reuse the small ledgers instead of reading all fields twice.
    layouts = [run.layout(frame) for frame in range(frames)]
    summary = dict(frames=frames, **grid_history(run, layouts))
    violations = []
    band_l2_frames = band_l2_max = 0
    nz = 8 if run.dimensions == 3 else 1
    for frame, (step, levels) in enumerate(zip(run.steps, layouts)):
        band = np.zeros((nz, 52, 256), dtype=bool)
        if len(levels) > 1:
            for lo, hi in levels[1]:
                low, high = max(lo[1], 166), min(hi[1], 218)
                if low < high:
                    band[lo[2]:hi[2], low-166:high-166, lo[0]:hi[0]] = True
        missing = int(band.size-band.sum())
        if missing:
            violations.append(dict(frame=frame, step=step, relation="missing_L1_fixed_band", cells=missing))
        band_l2_cells = 0
        if len(levels) > 2:
            for lo, hi in levels[2]:
                width = max(0, min(hi[1], 436)-max(lo[1], 332))
                band_l2_cells += width*(hi[0]-lo[0])*(hi[2]-lo[2])
        band_l2_frames += band_l2_cells > 0
        band_l2_max = max(band_l2_max, band_l2_cells)
        if run.band_max_level == 1 and band_l2_cells:
            violations.append(dict(frame=frame, step=step, relation="L2_inside_fixed_band", cells=band_l2_cells))
    summary.update(fixed_band_l2_frames=band_l2_frames, fixed_band_l2_cells_max=band_l2_max)
    return summary, violations


def check_run(directory, analysis, output, short=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    try:
        summary, violations = check(open_fixed_band(directory, short), analysis)
        result = 1 if violations else 0
        with (output/"violations.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, ("frame", "step", "relation", "cells"))
            writer.writeheader()
            writer.writerows(violations)
        (output/"result.txt").write_text(
            f"exit={result}\nscientific_gate=none\nclassification=engineering-and-data\n" +
            "".join(f"{key}={value}\n" for key, value in summary.items()) +
            f"violations={len(violations)}\n")
        print(f"FIXED BAND {'PASS' if result == 0 else 'RELATION FAILURE'}: {summary}; violations={len(violations)}")
        return result
    except (ArtifactError, OSError, ValueError, KeyError, csv.Error) as error:
        (output/"result.txt").write_text(f"exit=2\nerror={error}\n")
        print(f"INVALID fixed-band evidence: {error}")
        return 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--short", action="store_true")
    args = parser.parse_args()
    return check_run(args.run, args.analysis, args.output, args.short)


if __name__ == "__main__":
    raise SystemExit(main())
