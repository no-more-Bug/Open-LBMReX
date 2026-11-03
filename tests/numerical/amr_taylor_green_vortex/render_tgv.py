#!/usr/bin/env pvpython
"""Render unsmoothed native TGV error fields and the actual interface band."""
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
        for lev in range(int(meta["levels"])):
            with np.load(analysis/"planes"/f"s{step}-L{lev}-xy.npz") as archive:
                plane = {key: archive[key] for key in archive.files}
            require(str(plane["source_id"]) == meta["source_id"], "render source differs")
            planes.append(plane)
        error_fields = ["error_analytic"] + (["error_reference"] if "error_reference" in planes[0] else [])
        scale = max(1e-14, max(float(np.max(p[field][p["valid"]])) for p in planes
                             for field in error_fields if p["valid"].any()))
        panels = [(planes, "ux", "u_x / U0", (-1, 1), True),
                  (planes, "analytic_ux", "analytic u_x / U0", (-1, 1), True)]
        panels += [(planes, field, field+" / U0 (initial)", (0, scale), False) for field in error_fields]
        if meta["layout"] == "amr":
            panels.append((planes, "interface_band", "distance to actual interface <= 1/16", (0, 1), False))
        title = f'{meta["descriptor"]} {meta["backend"]} {meta["layout"]} N0={meta["coarse_cells"]} t={float(planes[0]["time"]):g}; {scope}; {result[0]}'
        mosaic(output/f"s{step}.png", panels, title)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.analysis, args.output)
