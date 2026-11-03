#!/usr/bin/env python3
"""Render native DSL slices with pvpython, or curve comparisons with Matplotlib."""
import argparse
import csv
import math
import html
import json
from pathlib import Path
import numpy as np

from analyze_dynamic_double_shear_layer import DynamicRun, FixedBandRun, open_run, open_fixed_band, DERIVATIVE
from amr_plotfile import ArtifactError, identity, require, rows


def geometry(run, frame):
    boxes = run.layout(frame) if isinstance(run, DynamicRun) else [run.boxes[lev] for lev in range(run.levels)]
    coverage, faces = [], []
    for lev, level_boxes in enumerate(boxes):
        n = run.n*2**lev
        nz = n//32 if run.dimensions == 3 else 1
        mask = np.zeros((n, n), dtype=bool)
        for lo, hi in level_boxes:
            if not lo[2] <= nz//2 < hi[2]:
                continue
            mask[lo[1]:hi[1], lo[0]:hi[0]] = True
            if lev > 0:
                x0, y0, x1, y1 = lo[0]/n, lo[1]/n, hi[0]/n, hi[1]/n
                faces.extend(((0, (x0, y0, 0), (x0, y1, run.depth)),
                              (0, (x1, y0, 0), (x1, y1, run.depth)),
                              (1, (x0, y0, 0), (x1, y0, run.depth)),
                              (1, (x0, y1, 0), (x1, y1, run.depth))))
        coverage.append(mask)
    masks = [mask.copy() for mask in coverage]
    for lev in range(len(masks)-1):
        n = masks[lev].shape[0]
        masks[lev] &= ~coverage[lev+1].reshape(n, 2, n, 2).all(axis=(1, 3))
    return masks, faces


def load_planes(run, analysis, frame):
    masks, faces = geometry(run, frame)
    step = run.steps[frame]
    planes = []
    for lev, mask in enumerate(masks):
        path = Path(analysis)/'planes'/f'frame{frame}-L{lev}-xy.npz'
        with np.load(path, allow_pickle=False) as archive:
            sample = {key: archive[key] for key in archive.files}
        require(all(str(sample[k]) == run.id[k] for k in ('source_id', 'descriptor', 'backend', 'layout', 'base_cells')) and
                str(sample['derivative']) == DERIVATIVE and str(sample['phase']) == 'Sync' and
                str(sample['plane']) == 'xy' and int(sample['frame']) == frame and int(sample['level']) == lev and
                int(sample['step']) == step*2**lev and int(sample['coarsest_step']) == step and
                math.isclose(float(sample['time']), step*.1/run.n, rel_tol=1e-12, abs_tol=1e-14),
                'native slice source/frame differs')
        if isinstance(run, FixedBandRun):
            require(str(sample.get("schema", "")) == run.id["schema"] and
                    str(sample.get("case_variant", "")) == run.id["case_variant"],
                    "fixed-band slice identity differs")
        n = run.n*2**lev
        nz = n//32 if run.dimensions == 3 else 1
        require(np.array_equal(sample['valid'], mask) and int(sample['slice_index']) == nz//2 and
                math.isclose(float(sample['slice_coordinate']), run.depth/2, rel_tol=0, abs_tol=1e-15) and
                np.allclose(sample['spacing'], [1/n, 1/n, run.depth/nz], rtol=0, atol=1e-15),
                'native slice does not match this frame layout')
        for key in ('vorticity', 'rho', 'ux', 'uy', 'uz'):
            require(sample[key].shape == mask.shape and np.isfinite(sample[key][mask]).all(),
                    'invalid native slice field')
        sample['level_index'] = np.full(mask.shape, lev, dtype=float)
        planes.append(sample)
    return planes, faces


def render_fields(run, analysis, output, frames, bound):
    from pv_native_planes import mosaic
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if bound is None:
        bound = max(float(row['vorticity_peak']) for row in rows(Path(analysis)/'curves.csv'))
        bound = max(bound, 1e-12)
    require(math.isfinite(bound) and bound > 0, 'invalid vorticity color limit')
    records = []
    for frame in frames:
        require(0 <= frame < len(run.steps), 'frame outside saved samples')
        planes, faces = load_planes(run, analysis, frame)
        label = f'{run.id["descriptor"]} {run.id.get("case_variant", run.id["layout"])} t={run.steps[frame]*.1/run.n:.5g}'
        mosaic(output/f'frame-{frame}.png',
               [(planes, 'vorticity', 'vorticity z', (-bound, bound), True),
                (planes, 'level_index', 'active level; lines: fine FAB boxes',
                 (0, max(1, run.max_level if isinstance(run, DynamicRun) else run.levels-1)), False)],
               label, faces=faces)
        records.append(dict(frame=frame, step=run.steps[frame], time=run.steps[frame]*.1/run.n,
            schema=run.id['schema'], color_limit=bound,
            clipped=sum(int(np.count_nonzero(np.abs(p['vorticity'][p['valid']]) > bound)) for p in planes),
            source_id=run.id['source_id'], backend=run.id['backend']))
    with (output/'render-record.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, records[0].keys())
        writer.writeheader(); writer.writerows(records)



def write_animation(run, output, frames):
    """A local PNG viewer; native physical time determines each frame duration."""
    require(frames and frames == sorted(set(frames)), "animation frames must be ordered and unique")
    output = Path(output)
    records = rows(output/"render-record.csv")
    require([int(row["frame"]) for row in records] == frames and
            len({row["color_limit"] for row in records}) == 1,
            "animation requires one fixed color scale")
    samples = [dict(file=f"frame-{frame}.png", step=run.steps[frame],
                    time=run.steps[frame]*.1/run.n) for frame in frames]
    require(all((output/sample["file"]).is_file() for sample in samples), "animation PNG is missing")
    title = f'{run.id["descriptor"]} {run.id.get("case_variant", run.id["layout"])}'
    payload = json.dumps(dict(frames=samples, milliseconds_per_tc=5000, schema=run.id["schema"],
                              source_id=run.id["source_id"], color_limit=float(records[0]["color_limit"])))
    page = """<!doctype html><meta charset="utf-8"><title>""" + html.escape(title) + """</title>
<style>body{font:16px sans-serif;margin:16px;background:#eee}img{display:block;max-width:100%;height:auto}button{margin:8px}</style>
<h2>""" + html.escape(title) + """</h2><p id="label"></p><img id="frame" alt="Vorticity and native AMR grid">
<button id="toggle">Pause</button><p>Playback: 5 seconds per convective time; fixed color scale.</p>
<script>
const data=""" + payload + """;
let index=0, playing=true;
function draw() {
  const f=data.frames[index];
  document.getElementById('frame').src=f.file;
  document.getElementById('label').textContent='step='+f.step+'; t/tc='+f.time.toPrecision(6);
}
function tick() {
  const current=data.frames[index], next=data.frames[index+1];
  const dt=next ? next.time-current.time : (index ? current.time-data.frames[index-1].time : .05);
  setTimeout(function() { if(playing) {index=(index+1)%data.frames.length; draw();} tick();},
             Math.max(1,dt*data.milliseconds_per_tc));
}
document.getElementById('toggle').onclick=function(){playing=!playing;this.textContent=playing?'Pause':'Play';};
draw(); tick();
</script>"""
    (output/"animation.html").write_text(page, encoding="utf-8")

def render_curves(directories, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    groups = []
    for directory in directories:
        meta, data = identity(directory), rows(Path(directory)/'curves.csv')
        require(meta['derivative'] == DERIVATIVE and data, 'wrong or missing comparison diagnostics')
        depth = int(meta['max_level'])+1 if meta['layout'] == 'dynamic' else int(meta['levels'])
        label = f'{meta["descriptor"]} {meta.get("case_variant", meta["layout"])} L={depth} N0={meta["base_cells"]}'
        groups.append((meta, data, label))
    panels = [('energy-enstrophy', ('energy', 'enstrophy'), ('E*', 'Z*')),
              ('conservation', ('mass_drift', 'momentum_drift'), ('relative mass drift', 'normalized momentum drift')),
              ('velocity-error', ('velocity_l2', 'velocity_linf'), ('relative velocity L2', 'velocity Linf / Uref')),
              ('mesh', ('active_levels', 'composite_cells'), ('active levels', 'composite cells'))]
    for filename, keys, labels in panels:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        plotted = False
        for _, data, label in groups:
            for ax, key, ylabel in zip(axes, keys, labels):
                if not all(row[key] != '' for row in data):
                    continue
                time = np.array([float(row['time']) for row in data])
                values = np.array([float(row[key]) for row in data])
                require(np.isfinite(time).all() and np.isfinite(values).all(), 'nonfinite curve')
                ax.plot(time, values, label=label)
                ax.set_xlabel('t / tc'); ax.set_ylabel(ylabel); ax.grid(alpha=.25)
                plotted = True
        if plotted:
            for ax in axes:
                ax.legend(fontsize=7)
            fig.savefig(output/f'{filename}.png', dpi=180)
        plt.close(fig)
    with (output/'sources.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('directory', 'descriptor', 'layout', 'schema', 'case_variant', 'source_id', 'backend', 'derivative'))
        for directory, (meta, _, _) in zip(directories, groups):
            writer.writerow((str(directory), *(meta.get(k, '') for k in ('descriptor', 'layout', 'schema', 'case_variant', 'source_id', 'backend', 'derivative'))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    fields = modes.add_parser('fields')
    fields.add_argument('--run', type=Path, required=True)
    fields.add_argument('--analysis', type=Path, required=True)
    fields.add_argument('--output', type=Path, required=True)
    fields.add_argument('--frames', type=int, nargs='+', default=[0, 10, 20, 30, 40])
    fields.add_argument('--vorticity-limit', type=float)
    fields.add_argument('--short', action='store_true')
    fields.add_argument('--fixed-band', action='store_true')
    fields.add_argument('--all-frames', action='store_true')
    fields.add_argument('--animation', action='store_true')
    curves = modes.add_parser('curves')
    curves.add_argument('--analysis', type=Path, nargs='+', required=True)
    curves.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.mode == 'fields':
            opener = open_fixed_band if args.fixed_band else open_run
            run = opener(args.run, args.short)
            frames = list(range(len(run.steps))) if args.all_frames else args.frames
            render_fields(run, args.analysis, args.output, frames, args.vorticity_limit)
            if args.animation:
                write_animation(run, args.output, frames)
        else:
            render_curves(args.analysis, args.output)
        return 0
    except (ArtifactError, OSError, ValueError, KeyError, csv.Error) as error:
        print(f'INVALID DSL rendering: {error}')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
