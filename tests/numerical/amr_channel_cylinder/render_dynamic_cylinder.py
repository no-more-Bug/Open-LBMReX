#!/usr/bin/env pvpython
"""Render checker-produced native planes and raw force curves; no field diagnostics."""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np

from check_dynamic_cylinder import DT, LOW, sample_steps
from amr_plotfile import ArtifactError, identity, require, rows
from render_re20 import chart


def animation(output, name, frames):
    if not frames:
        return
    times = [step*DT for step in frames]
    spacing = float(np.median(np.diff(times))) if len(times) > 1 else 1
    items = [dict(image=f"s{step}-wake.png",time=time) for step,time in zip(frames,times)]
    (output/f"{name}.html").write_text("""<!doctype html><meta charset="utf-8">
<title>Dynamic cylinder native frames</title>
<style>body{font:16px sans-serif;margin:1em}img{max-width:100%;display:block}button{margin:.5em}</style>
<button id="toggle">Pause</button><span id="label"></span><img id="frame">
<script>
const frames="""+json.dumps(items)+""", scale="""+str(250/spacing)+""";
let index=0,playing=true;
document.querySelector('#toggle').onclick=()=>{playing=!playing;document.querySelector('#toggle').textContent=playing?'Pause':'Play'};
function tick(){const f=frames[index];document.querySelector('#frame').src=f.image;
document.querySelector('#label').textContent='physical time = '+f.time.toPrecision(7)+'; time scale '+scale.toPrecision(4)+' ms / physical second';
const next=(index+1)%frames.length,dt=next?frames[next].time-f.time:250/scale;
if(playing)index=next;setTimeout(tick,playing?scale*dt:100)}
tick();</script>""",encoding="utf-8")


def load_planes(analysis, meta, step):
    layouts = rows(analysis/f"layout-{step}.csv")
    levels = sorted({int(row["level"]) for row in layouts})
    require(levels == list(range(len(levels))),"noncontiguous render layout")
    planes, faces = [], []
    for lev in levels:
        with np.load(analysis/"planes"/f"s{step}-L{lev}-xy.npz",allow_pickle=False) as archive:
            plane = {key:archive[key] for key in archive.files}
        require(all(str(plane[key]) == meta[key] for key in ("source_id","descriptor","backend","variant","schema","scheme")) and
                str(plane["phase"]) == "Sync" and str(plane["plane"]) == "xy" and
                int(plane["level"]) == lev and int(plane["step"]) == step*2**lev and
                int(plane["coarsest_step"]) == step and math.isclose(float(plane["time"]),step*DT,abs_tol=1e-10),
                "render plane identity differs")
        for key in ("speed","pressure"):
            require(plane[key].shape == plane["valid"].shape and np.isfinite(plane[key][plane["valid"]]).all(),
                    "invalid render plane")
        plane["refinement_level"] = np.full(plane["valid"].shape,lev)
        planes.append(plane)
        if lev:
            h = .005/2**lev
            for row in layouts:
                if int(row["level"]) != lev or not int(row["klo"]) <= int(plane["slice_index"]) < int(row["khi"]):
                    continue
                x0,x1 = (LOW[0]+int(row[key])*h for key in ("ilo","ihi"))
                y0,y1 = (LOW[1]+int(row[key])*h for key in ("jlo","jhi"))
                z0,z1 = (int(row[key])*float(plane["spacing"][2]) for key in ("klo","khi"))
                faces.extend(((0,(x0,y0,z0),(x0,y1,z1)),(0,(x1,y0,z0),(x1,y1,z1)),
                              (1,(x0,y0,z0),(x1,y0,z1)),(1,(x0,y1,z0),(x1,y1,z1))))
    return planes,faces


def render(analysis, output, selected=None):
    from pv_native_planes import mosaic
    meta = identity(analysis)
    require(meta["schema"] == "dynamic-cylinder-v1","wrong dynamic render identity")
    result = (analysis/"result.txt").read_text().splitlines()
    require(result and result[0] in ("exit=0","exit=1"),"invalid/incomplete data cannot be rendered as completed results")
    steps = sample_steps(int(meta["requested_steps"]))
    frames = steps if selected is None else sorted(set(selected))
    require(frames and set(frames) <= set(steps),"requested render frame was not sampled")
    output.mkdir(parents=True,exist_ok=True)
    title = f'{meta["descriptor"]} {meta["backend"]} {meta["scheme"]} {meta["variant"]}; no scientific gate'
    records = []
    for step in frames:
        planes,faces = load_planes(analysis,meta,step)
        mosaic(output/f"s{step}-wake.png",
            [(planes,"speed","speed / Umean",(0,2.5),False),
             (planes,"pressure","(rho/rho0-1)/(3 Umean_lb^2)",(-1,5),True),
             (planes,"refinement_level","actual AMR level and fine FAB edges",(0,2),False)],
            title+f"; t={step*DT:.7g}",faces=faces,view=(1.1,.205,1.4))
        records.append(dict(step=step,time=step*DT,speed_min=0,speed_max=2.5,pressure_min=-1,pressure_max=5,
                            source_id=meta["source_id"],backend=meta["backend"]))
    with (output/"render-record.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,list(records[0]));writer.writeheader();writer.writerows(records)
    overview = [step for step in frames if step%4000 == 0]
    if frames[-1] < 4000 and frames[-1] not in overview:
        overview.append(frames[-1])
    detail = [step for step in frames if step >= 160000 and (step-160000)%400 == 0]
    animation(output,"overview",overview)
    animation(output,"detail",detail)
    # Full original midpoint samples remain in the CSV; rendering does not smooth them.
    forces = np.loadtxt(analysis/"force-history.csv",delimiter=",",skiprows=1,usecols=(0,4,5,6),ndmin=2)
    if forces.size:
        for field,column in (("Cd",2),("Cl",3)):
            series=[]
            for bid in np.unique(forces[:,0]):
                chosen=forces[:,0] == bid
                series.append((f"Boundary {int(bid)}",forces[chosen,1],forces[chosen,column]))
            chart(output/f"{field}-history.png",title+"; raw native samples","physical time",field,series)
    grid=np.loadtxt(analysis/"grid-history.csv",delimiter=",",skiprows=1,usecols=(1,2,3,4),ndmin=2)
    if grid.size:
        for field,column in (("valid_cells",2),("boxes",3)):
            series=[(f"L{lev}",grid[grid[:,1] == lev,0],grid[grid[:,1] == lev,column]) for lev in (0,1,2)]
            chart(output/f"grid-{field}.png",title,"physical time",field,series)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--steps",type=int,nargs="+")
    args=parser.parse_args()
    try:
        render(args.analysis,args.output,args.steps)
        return 0
    except (ArtifactError,OSError,ValueError,KeyError,csv.Error) as error:
        print(f"INVALID dynamic cylinder rendering: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
