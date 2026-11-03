#!/usr/bin/env pvpython
"""Render the checker's native xy cells using Windows ParaView pvpython."""
import argparse
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"common"))
from amr_plotfile import identity, require
from pv_native_planes import mosaic


def render(analysis, output):
    meta = identity(analysis)
    result = (analysis/"result.txt").read_text().splitlines()
    scope = next(row for row in result if row.startswith("scope="))
    output.mkdir(parents=True, exist_ok=True)
    steps = sorted({int(path.name.split("-")[0][1:]) for path in (analysis/"planes").glob("s*-L0-xy.npz")})
    require(steps, "no checker-derived planes")
    for step in steps:
        planes = []
        for lev in range(3):
            with np.load(analysis/"planes"/f"s{step}-L{lev}-xy.npz") as archive:
                plane = {key: archive[key] for key in archive.files}
            require(str(plane["source_id"]) == meta["source_id"], "render source differs")
            plane["refinement_level"] = np.full(plane["valid"].shape, lev)
            planes.append(plane)
        def peak(field):
            return max(1e-14, max(float(np.max(np.abs(p[field][p["valid"]]))) for p in planes if p["valid"].any()))
        panels = [(planes, "ux", "ux / U0", (-1, 1), True),
                  (planes, "velocity_error", "|u - U0 ex| / U0", (0, peak("velocity_error")), False),
                  (planes, "rho_error", "rho/rho0 - 1", (-peak("rho_error"), peak("rho_error")), True),
                  (planes, "refinement_level", "actual native refinement level", (0, 2), False)]
        title = f'{meta["descriptor"]} {meta["backend"]} t={float(planes[0]["time"]):g}; {scope}; {result[0]}'
        mosaic(output/f"s{step}.png", panels, title)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.analysis, args.output)
