#!/usr/bin/env pvpython
"""Render one DSL AMR/reference pair from analyzer-owned native cell planes.

Example: pvpython render_double_shear_layer.py --run RUN --reference REF
    --output FIGURES --frames 0 12 20 30 40
Omit --frames for all 41 frames. Optional fixed color limits permit comparison
across layouts; clipping counts are recorded. Exit 0 means rendering succeeded,
not that the scientific gates or the qualitative vortex inspection passed.
"""
import argparse
import csv
import math
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'common'))
from amr_plotfile import ArtifactError, identity, require
from analyze_double_shear_layer import DslRun, DERIVATIVE


def geometry(run):
    """Recover the actual xy coverage from layout Boxes at the saved z plane."""
    coverage = []
    for lev in range(run.levels):
        n = run.n*2**lev
        nz = n//32 if run.dimensions == 3 else 1
        mask = np.zeros((n, n), dtype=bool)
        for lo, hi in run.boxes[lev]:
            if lo[2] <= nz//2 < hi[2]:
                mask[lo[1]:hi[1], lo[0]:hi[0]] = True
        require(np.array_equal(mask, np.broadcast_to(mask[:, :1], mask.shape)),
                'DSL rendering requires complete x/z bands')
        coverage.append(mask)
    masks = [mask.copy() for mask in coverage]
    for lev in range(run.levels-1):
        n = run.n*2**lev
        covered = coverage[lev+1].reshape(n, 2, n, 2).all(axis=(1, 3))
        masks[lev] &= ~covered
    faces, records = [], []
    for lev in range(1, run.levels):
        active = coverage[lev][:, 0]
        for row in np.flatnonzero(active != np.roll(active, 1)):
            y = float(row)/len(active)
            # y=1 repeats the same physical periodic interface drawn at y=0.
            for shown_y in ((y, 1.) if row == 0 else (y,)):
                faces.append((1, np.array([0., shown_y, 0.]), np.array([1., shown_y, run.depth])))
                records.append(dict(coarse_level=lev-1, fine_level=lev, y=y,
                                    displayed_y=shown_y, periodic_copy=int(shown_y != y)))
    return masks, faces, records


def load_planes(run, frame, masks):
    result = []
    step = frame*run.n//2
    require(all((step, lev) in run.samples for lev in range(run.levels)), f'missing DSL frame {frame}')
    for lev, mask in enumerate(masks):
        path = run.directory/'analysis/planes'/f'frame{frame}-L{lev}-xy.npz'
        with np.load(path, allow_pickle=False) as archive:
            sample = {key: archive[key] for key in archive.files}
        for key in ('source_id', 'descriptor', 'backend', 'layout', 'base_cells'):
            require(str(sample[key]) == run.id[key], f'{path}: wrong {key}')
        require(str(sample['derivative']) == DERIVATIVE, f'{path}: unexpected vorticity reconstruction')
        require(str(sample['plane']) == 'xy' and str(sample['phase']) == 'Sync' and
                int(sample['frame']) == frame and int(sample['level']) == lev and
                int(sample['step']) == step*2**lev and int(sample['coarsest_step']) == step and
                math.isclose(float(sample['time']), frame/20, rel_tol=1e-12, abs_tol=1e-14),
                f'{path}: wrong frame/clock identity')
        n = run.n*2**lev
        nz = n//32 if run.dimensions == 3 else 1
        spacing = (1/n, 1/n, run.depth/nz)
        require(np.allclose(sample['spacing'], spacing, rtol=1e-13, atol=0) and
                float(sample['slice_coordinate']) == run.depth/2 and int(sample['slice_index']) == nz//2,
                f'{path}: wrong native slice lattice')
        require(sample['valid'].dtype == np.bool_ and np.array_equal(sample['valid'], mask),
                f'{path}: mask differs from actual uncovered coverage')
        for field in ('vorticity', 'mach'):
            require(sample[field].shape == mask.shape and np.isfinite(sample[field][mask]).all(),
                    f'{path}: invalid {field}')
        require((sample['mach'][mask] >= 0).all(), f'{path}: negative Mach number')
        sample['resolution'] = np.full(mask.shape, n, dtype=float)
        result.append(sample)
    return result


def peak(planes, field):
    return max(float(np.max(np.abs(p[field][p['valid']]))) for p in planes if p['valid'].any())


def clipped(planes, field, bound):
    return sum(int(np.count_nonzero(np.abs(p[field][p['valid']]) > bound)) for p in planes)


def render(run_directory, reference_directory, output, frames, vorticity_limit, mach_limit):
    import paraview
    from pv_native_planes import mosaic

    run, reference = DslRun(run_directory), DslRun(reference_directory)
    require(run.layout != 'uniform' and reference.layout == 'uniform', 'expected AMR and uniform roles')
    nref = 256 if run.layout == 'baseline' else 512
    require(reference.n == nref and all(run.id[key] == reference.id[key] for key in
            ('source_id', 'descriptor', 'backend', 'dimensions', 'reynolds', 'u0_lb', 'u0_phys',
             'viscosity', 'length_z', 'collision', 'initialization', 'periodic', 'real_digits')),
            'unmatched DSL rendering reference')
    for source in (run, reference):
        derived = identity(source.directory/'analysis')
        require(all(derived[key] == value for key, value in source.id.items()), 'analysis/run identity differs')
        require(derived['derivative'] == DERIVATIVE, 'unexpected vorticity reconstruction')
    masks, faces, interfaces = geometry(run)
    reference_masks, _, _ = geometry(reference)
    output.mkdir(parents=True, exist_ok=True)
    with (output/'interfaces.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, ('coarse_level', 'fine_level', 'y', 'displayed_y', 'periodic_copy'))
        writer.writeheader(); writer.writerows(interfaces)
    (output/'render-identity.txt').write_text(
        f'paraview={paraview.__version__}\nsource_id={run.id["source_id"]}\nbackend={run.id["backend"]}\n'
        f'descriptor={run.id["descriptor"]}\nlayout={run.layout}\nrun={run.directory.resolve()}\n'
        f'reference={reference.directory.resolve()}\nframes={",".join(map(str, frames))}\n'
        'geometry=native-cell-xy-plane\nmask=actual-uncovered-valid\ninterpolation=none\n'
        'interface_overlay=AMR-interface-guides-on-both-panels\n'
        f'vorticity=omega_z-L/u0\nderivative={DERIVATIVE}\nmach=local-speed/cs\n'
        'secondary_vortex_judgment=pending-image-review\nscientific_gates=not-evaluated\n')
    fields = ('frame', 'time', 'vorticity_bound', 'mach_bound', 'amr_vorticity_peak', 'reference_vorticity_peak',
              'amr_mach_peak', 'reference_mach_peak', 'amr_vorticity_clipped', 'reference_vorticity_clipped',
              'amr_mach_clipped', 'reference_mach_clipped')
    with (output/'render-settings.csv').open('w', newline='') as stream, \
            (output/'inspection-template.csv').open('w', newline='') as inspection:
        writer = csv.DictWriter(stream, fields); writer.writeheader()
        review = csv.writer(inspection)
        review.writerow(('frame', 'time', 'vorticity_image', 'mach_image', 'mesh_image', 'secondary_vortices', 'notes'))
        for frame in frames:
            try:
                actual = load_planes(run, frame, masks)
                uniform = load_planes(reference, frame, reference_masks)
                av, rv = peak(actual, 'vorticity'), peak(uniform, 'vorticity')
                am, rm = peak(actual, 'mach'), peak(uniform, 'mach')
                vb = vorticity_limit if vorticity_limit is not None else max(av, rv, 1e-12)
                mb = mach_limit if mach_limit is not None else max(am, rm, 1e-12)
                title = f'{run.id["descriptor"]} {run.layout} t/tc={frame/20:g}; black lines: AMR interfaces'
                labels = (f'AMR {run.layout}', f'Uniform N={nref}; AMR interface guides')
                for field, limits, signed in (('vorticity', (-vb, vb), True), ('mach', (0, mb), False)):
                    panels = [(planes, field, label, limits, signed) for planes, label in zip((actual, uniform), labels)]
                    mosaic(output/f'frame{frame}-{field}.png', panels, title, faces)
                limits = (128., 512.)
                panels = [(planes, 'resolution', label+'; native Nxy', limits, False)
                          for planes, label in zip((actual, uniform), labels)]
                mosaic(output/f'frame{frame}-mesh.png', panels, title, faces, mesh_edges=True)
                writer.writerow(dict(frame=frame, time=frame/20, vorticity_bound=vb, mach_bound=mb,
                    amr_vorticity_peak=av, reference_vorticity_peak=rv, amr_mach_peak=am, reference_mach_peak=rm,
                    amr_vorticity_clipped=clipped(actual, 'vorticity', vb), reference_vorticity_clipped=clipped(uniform, 'vorticity', vb),
                    amr_mach_clipped=clipped(actual, 'mach', mb), reference_mach_clipped=clipped(uniform, 'mach', mb)))
                stream.flush()
                review.writerow((frame, frame/20, f'frame{frame}-vorticity.png', f'frame{frame}-mach.png',
                                 f'frame{frame}-mesh.png', 'pending', ''))
                inspection.flush()
                print(f'DSL rendered frame={frame} time={frame/20:g}', flush=True)
            except (OSError, ValueError, KeyError, RuntimeError) as error:
                raise ArtifactError(f'frame={frame} nominal_time={frame/20}: {error}') from error
    (output/'render-result.txt').write_text(f'exit=0\nrendered_frames={len(frames)}\nimage_review=pending\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--frames', nargs='+', type=int, default=list(range(41)))
    parser.add_argument('--vorticity-limit', type=float, help='fixed symmetric color half-range; clipping is recorded')
    parser.add_argument('--mach-limit', type=float, help='fixed color upper limit; clipping is recorded')
    args = parser.parse_args()
    try:
        require(args.frames == sorted(set(args.frames)) and all(0 <= f <= 40 for f in args.frames),
                'frames must be unique, increasing integers from 0 to 40')
        require(all(v is None or (math.isfinite(v) and v > 0) for v in (args.vorticity_limit, args.mach_limit)),
                'color limits must be positive and finite')
        render(args.run, args.reference, args.output, args.frames, args.vorticity_limit, args.mach_limit)
        return 0
    except (OSError, ValueError, KeyError, RuntimeError, ImportError, csv.Error) as error:
        print(f'INVALID DSL render: {error}')
        try:
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output/'render-result.txt').write_text(f'exit=2\nerror={error}\n')
        except OSError as output_error:
            print(f'Cannot save render failure: {output_error}')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
