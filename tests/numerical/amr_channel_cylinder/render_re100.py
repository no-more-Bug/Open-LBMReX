#!/usr/bin/env pvpython
"""Render checker-derived force history, qualified cycles, and native wake planes."""
import argparse
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"common"))
from amr_plotfile import identity, require, rows
from pv_native_planes import mosaic
from render_re20 import chart


def render(analysis, output):
    meta = identity(analysis)
    result = (analysis/"result.txt").read_text().splitlines()
    output.mkdir(parents=True, exist_ok=True)
    scope = next(line for line in result if line.startswith("scope="))
    title = f'{meta["descriptor"]} {meta["backend"]} {meta["scheme"]} {meta["layout"]}; {scope}'
    history = rows(analysis/"force-history.csv")
    times = np.array([float(row["time"]) for row in history])
    for field in ("Cd", "Cl"):
        values = np.array([float(row[field]) for row in history])
        chart(output/f"{field}-history.png", title+"; native midpoint samples", "physical time", field,
              [(field, times, values)])
        if (analysis/"statistics.csv").exists():
            statistics = rows(analysis/"statistics.csv")[0]
            begin, end = (float(statistics[key])/10 for key in ("stable_begin_t_star", "statistics_end_t_star"))
            selected = (times >= begin) & (times <= end)
            split = float(statistics["statistics_begin_t_star"])/10
            stable = selected & (times < split)
            used = selected & (times >= split)
            chart(output/f"{field}-window.png", title+f"; first 3+5 window, statistics starts at {split:.7g}",
                  "physical time", field, [("3 qualification cycles", times[stable], values[stable]),
                                           ("5 statistics cycles", times[used], values[used])])
    steps = sorted({int(path.name.split("-")[0][1:]) for path in (analysis/"planes").glob("s*-L0-xy.npz")})
    require(steps, "no checker-derived native planes")
    for step in steps:
        planes = []
        for lev in range(int(meta["levels"])):
            with np.load(analysis/"planes"/f"s{step}-L{lev}-xy.npz") as archive:
                plane = {key: archive[key] for key in archive.files}
            require(str(plane["source_id"]) == meta["source_id"], "render source differs")
            # Native level colors expose the actual refinement coverage.
            plane["refinement_level"] = np.full(plane["valid"].shape, lev)
            planes.append(plane)
        panels = [(planes, "speed", "speed / Umean", (0, 2.5), False),
                  (planes, "pressure", "(rho/rho0 - 1)/(3 Umean_lb^2)", (-1, 5), True),
                  (planes, "refinement_level", "actual native refinement level", (0, 2), False)]
        mosaic(output/f"s{step}-wake.png", panels, title+f'; t={float(planes[0]["time"]):g}',
               view=(1.1, .205, 1.4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.analysis, args.output)
