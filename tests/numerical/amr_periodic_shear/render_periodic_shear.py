#!/usr/bin/env pvpython
"""Render checker-derived native cells with pvpython; never trust external AMR masking."""
import argparse
import csv
import math
from itertools import product
from pathlib import Path
import sys
import numpy as np
import paraview

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'common'))
from amr_plotfile import identity, require, rows
from pv_native_planes import mosaic
from shear_metrics import U, AMPLITUDE, mode_vectors


def load(path, meta, plane, **expected):
    with np.load(path) as archive:
        sample = {key: archive[key] for key in archive.files}
    for key in ('source_id', 'descriptor', 'backend', 'layout', 'mode', 'coarse_cells'):
        require(str(sample[key]) == meta[key], f'wrong rendering identity: {key}')
    require(str(sample['plane']) == plane, 'wrong native slice plane')
    for key, value in expected.items():
        if isinstance(value, float):
            require(math.isclose(float(sample[key]), value, rel_tol=1e-12, abs_tol=1e-14), f'wrong rendering {key}')
        else:
            require(str(sample[key]) == str(value), f'wrong rendering {key}')
    lev = int(sample['level']); n = int(meta['coarse_cells'])*2**lev
    dx = (1/n, 1/n, 1/n if meta['dimensions'] == '3' else 1/16)
    require(np.allclose(sample['spacing'], dx, rtol=1e-13, atol=0) and sample['valid'].shape == (n, n),
            'wrong native slice lattice')
    position = .5001 if meta['dimensions'] == '3' else 1/32
    hidden = 2 if plane == 'xy' else 1
    require(float(sample['slice_coordinate']) == position and int(sample['slice_index']) == int(position/dx[hidden]),
            'wrong native slice position')
    return sample


def peak(planes, fields):
    return max(float(np.max(np.abs(p[field][p['valid']]))) for p in planes for field in fields if p['valid'].any())


def render(run, out):
    meta = identity(run); dimensions = int(meta['dimensions'])
    out.mkdir(parents=True, exist_ok=True)
    faces = [(int(row['axis']), np.array([float(row[k]) for k in ('xlo', 'ylo', 'zlo')]),
              np.array([float(row[k]) for k in ('xhi', 'yhi', 'zhi')])) for row in rows(run/'analysis/interfaces.csv')]
    # Show both edges for a canonical periodic interface stored at coordinate zero.
    for axis, lo, hi in list(faces):
        if lo[axis] == hi[axis] == 0:
            shift = np.zeros(3); shift[axis] = 1
            faces.append((axis, lo+shift, hi+shift))
    planes_root = run/'analysis/planes'
    records = []
    nc = int(meta['coarse_cells'])
    for t in (1, 2, 4):
        for plane in (('xy',) if dimensions == 2 else ('xy', 'xz')):
            native = [load(planes_root/f't{t}-L{lev}-{plane}.npz', meta, plane, time=float(t), level=lev,
                           step=t*nc*2**lev, coarsest_step=t*nc, phase='Sync') for lev in (0, 1)]
            raw_range = max(1, peak(native, ('amr', 'analytic', 'reference')))
            error_range = max(1e-14, peak(native, ('error_analytic', 'error_reference')))
            pressure_range = max(1e-14, peak(native, ('pressure',)))
            panels = [(native, f, label, (-raw_range, raw_range), True) for f, label in (
                ('amr', 'AMR: polarized perturbation / initial amplitude'), ('analytic', 'Analytic cell average'),
                ('reference', 'Uniform reference mapped by density/momentum'))]
            panels += [(native, 'error_analytic', 'norm(AMR - analytic) / initial amplitude', (0, error_range), False),
                       (native, 'error_reference', 'norm(AMR - reference) / initial amplitude', (0, error_range), False),
                       (native, 'pressure', 'Pressure residual / (rho0 cs amplitude)', (-pressure_range, pressure_range), True)]
            title = f'{meta["descriptor"]} {meta["mode"]} {meta["layout"]} Nc={meta["coarse_cells"]} t={t} {plane}'
            mosaic(out/f't{t}-{plane}-fields.png', panels, title, faces)
            mosaic(out/f't{t}-{plane}-mesh.png', [panels[0]], title, faces, mesh_edges=True)
            # Detail around this native slice's largest AMR/reference error.
            worst = max((sample for sample in native if sample['valid'].any()),
                        key=lambda sample: float(np.max(sample['error_reference'][sample['valid']])))
            j, i = np.unravel_index(np.argmax(np.where(worst['valid'], worst['error_reference'], -np.inf)), worst['valid'].shape)
            width = max(1/8, 8/nc)
            shown = (0, 1) if plane == 'xy' else (0, 2)
            hidden = 3-sum(shown)
            center = np.array([(i+.5)*worst['spacing'][shown[0]], (j+.5)*worst['spacing'][shown[1]]])
            intersections = [np.clip(center, lo[list(shown)]+shift, hi[list(shown)]+shift)
                             for _, lo, hi in faces for shift in map(np.array, product((-1, 0, 1), repeat=2))
                             if lo[hidden] <= float(worst['slice_coordinate']) <= hi[hidden]]
            detail = 'native error maximum (no interface in this slice)'
            if intersections:
                center = np.mod(min(intersections, key=lambda point: np.linalg.norm(point-center)), 1)
                detail = 'coarse/fine interface selected by periodic distance; domain-side view'
            x, y = np.clip(center, width/2, 1-width/2)
            mosaic(out/f't{t}-{plane}-interface-detail.png', [panels[0], panels[4], panels[5]],
                   title+'\n'+detail, faces, view=(float(x), float(y), .64*width))
            records.append((t, plane, float(native[0]['slice_coordinate']), raw_range, error_range, pressure_range))
            fine = []
            for j in range(4):
                step = 2*nc*t-3+j
                fine.append(load(planes_root/f'fine-t{t}-j{j}-{plane}.npz', meta, plane,
                                 time=step/(2*nc), level=1, step=step, coarsest_step=(step+1)//2,
                                 phase='First' if step % 2 else 'Second', endpoint=t))
            m, p = mode_vectors(dimensions, meta['mode']); background = U*m/np.linalg.norm(m)
            for sample in fine:
                sample['raw_polarized'] = sum(p[d]*(sample[f'raw_{d+1}']-background[d]) for d in range(3))/AMPLITUDE
                sample['residual_polarized'] = sum(p[d]*sample[f'residual_{d+1}'] for d in range(3))
            temporal = load(planes_root/f'temporal-t{t}-{plane}.npz', meta, plane, time=float(t), level=1)
            temporal['polarized'] = sum(p[d]*temporal[f'coefficient_{d+1}'] for d in range(3))
            bound = max(1e-14, peak(fine, ('residual_polarized',)), peak([temporal], ('polarized',)))
            panels = [([sample], 'residual_polarized',
                f'fine step={int(sample["step"])} target L0={int(sample["coarsest_step"])} {str(sample["phase"])} t={float(sample["time"]):.12g}',
                (-bound, bound), True) for sample in fine]
            panels += [([temporal], 'polarized', 'Signed temporal four-point velocity coefficient', (-bound, bound), True),
                       ([temporal], 'coefficient_0', 'Signed temporal pressure coefficient',
                        (-max(1e-14, peak([temporal], ('coefficient_0',))), max(1e-14, peak([temporal], ('coefficient_0',)))), True)]
            mosaic(out/f't{t}-{plane}-fine-residuals.png', panels, title, faces)
            raw_bound = max(1, peak(fine, ('raw_polarized',)))
            mosaic(out/f't{t}-{plane}-fine-raw.png', [([sample], 'raw_polarized',
                f'Raw fine step={int(sample["step"])} {str(sample["phase"])}', (-raw_bound, raw_bound), True)
                for sample in fine], title, faces)
            spatial = [load(planes_root/f'spatial-t{t}-axis{axis}-{plane}.npz', meta, plane, time=float(t), level=1, axis=axis) for axis in range(dimensions)]
            for sample in spatial:
                sample['polarized'] = sum(p[d]*sample[f'coefficient_{d+1}'] for d in range(3))
            bounds = {field: max(1e-14, peak(spatial, (field,))) for field in ('polarized', 'coefficient_0')}
            panels = []
            for axis, sample in enumerate(spatial):
                for field, quantity in (('polarized', 'velocity'), ('coefficient_0', 'pressure')):
                    bound = bounds[field]
                    panels.append(([sample], field, f'Spatial axis={axis} signed {quantity} coefficient', (-bound, bound), True))
            mosaic(out/f't{t}-{plane}-spatial.png', panels, title, faces)
    with (out/'render-settings.csv').open('w', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(('time', 'plane', 'slice_coordinate', 'raw_bound', 'velocity_error_bound', 'pressure_bound'))
        writer.writerows(records)
    (out/'render-identity.txt').write_text(f'paraview={paraview.__version__}\nsource_id={meta["source_id"]}\nrun={run.resolve()}\n'
        'geometry=native-cell-planes\nmask=actual-uncovered-valid\ninterpolation=none\n'
        'error_colors=shared-actual-peak-per-quantity-per-figure; fixed-normalization; see CSV limits\n'
        'external_reader_mask=not-used; existing thin-AMR reader issue remains open\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path); parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        render(args.run, args.output)
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f'INVALID render: {error}')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
