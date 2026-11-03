#!/usr/bin/env python3
"""Compute DSL energy/enstrophy on the original uncovered valid cells."""
import argparse
import csv
import math
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'common'))
from amr_plotfile import ArtifactError, require, identity, rows, read_plot, composite_masks, totals
from native_planes import dump_planes
from dsl_vorticity import CurlPlan

LAYOUTS = ('baseline', '3l', '2r', 'p1', 'p2', 'p3')
COLUMNS = ('frame', 'time', 'step', 'energy', 'enstrophy', 'mass', 'px', 'py', 'pz',
           'mass_drift', 'momentum_drift', 'uz_peak', 'uz_rms', 'z_rho', 'z_ux', 'z_uy', 'z_uz',
           'vorticity_peak', 'mach_peak')
DERIVATIVE = 'central4-and-periodic-quartic-lsq'


def requested_rows(n, layout, lev):
    y = (np.arange(n)+.5)/n
    if lev == 0:
        return np.ones(n, dtype=bool)
    if lev == 1:
        if layout == '2r':
            return ((y >= 1/8) & (y < 3/8)) | ((y >= 9/16) & (y < 15/16))
        return (y >= .5) | ((y < 3/8) if layout.startswith('p') else False)
    return ((y >= 5/8) & (y < (1 if layout in ('p2', 'p3') else 7/8))) | ((y < 1/8) if layout == 'p3' else False)


class DslRun:
    def __init__(self, directory):
        self.directory = Path(directory); self.id = identity(directory)
        self.dimensions = int(self.id['dimensions']); self.layout = self.id['layout']
        self.n = int(self.id['base_cells']); self.levels = int(self.id['levels'])
        require(self.id['schema'] == 'double-shear-v1' and self.id['backend'] in ('cpu', 'cuda') and
                int(self.id['real_digits']) == 53 and (self.id['descriptor'], self.dimensions) in
                (('D2Q9', 2), ('D3Q27', 3)), 'invalid DSL identity')
        require(self.id['collision'] == 'BGK' and self.id['periodic'] == '1,1,1' and
                self.id['initialization'] == 'cell-center-equilibrium', 'wrong DSL composition')
        require((self.layout == 'uniform' and self.levels == 1 and self.n in (128, 256, 512)) or
                (self.layout in LAYOUTS and self.n == 128 and self.levels == (2 if self.layout == 'baseline' else 3)),
                'wrong DSL layout/grid')
        require(int(self.id['full_steps']) == int(self.id['requested_steps']) == 20*self.n and
                int(self.id['frame_stride']) == self.n//2, 'incomplete DSL horizon')
        for key, value in (('reynolds', 10000), ('u0_lb', .1), ('u0_phys', 1), ('viscosity', .0001),
                           ('kappa', 80), ('delta', .05), ('rho_reference', 1), ('velocity_reference', .1),
                           ('gap_fill_radius', 4), ('coarse_nesting_margin', 4), ('grid_efficiency', 1)):
            require(math.isclose(float(self.id[key]), value, rel_tol=1e-13), f'wrong DSL {key}')
        self.depth = 1/32 if self.dimensions == 3 else 1/16
        require(float(self.id['length_z']) == self.depth and int(self.id['base_z']) == (self.n//32 if self.dimensions == 3 else 1),
                'wrong quasi-2D thickness/cells')
        for lev in range(self.levels):
            n = self.n*2**lev
            require(math.isclose(float(self.id[f'dt_{lev}']), .1/n, rel_tol=1e-13) and
                    math.isclose(float(self.id[f'tau_{lev}']), .5+3*.1*n/10000, rel_tol=1e-13), 'wrong DSL scaling')
        self.samples = {}
        for row in rows(self.directory/'samples.csv'):
            key = int(row['coarsest_step']), int(row['level'])
            require(key not in self.samples and row['kind'] == 'sync' and row['phase'] == 'Sync' and
                    int(row['step']) == key[0]*2**key[1] and
                    math.isclose(float(row['time']), key[0]*.1/self.n, rel_tol=1e-12, abs_tol=1e-14), 'invalid DSL sample')
            self.samples[key] = row
        expected = {(frame*self.n//2, lev) for frame in range(41) for lev in range(self.levels)}
        self.expected_samples = expected
        require(self.samples.keys() <= expected, 'unexpected DSL frames or levels')
        self.boxes = {}
        for row in rows(self.directory/'layout.csv'):
            lo = tuple(int(row[key]) for key in ('ilo', 'jlo', 'klo'))
            hi = tuple(int(row[key]) for key in ('ihi', 'jhi', 'khi'))
            self.boxes.setdefault(int(row['level']), []).append((lo, hi))

    def require_complete(self):
        require(self.samples.keys() == self.expected_samples, 'missing DSL frames or levels')

    def plot(self, frame):
        step = frame*self.n//2
        require(all((step, lev) in self.samples for lev in range(self.levels)), f'missing DSL frame {frame}')
        path = Path(self.samples[step, 0]['path'])
        require(not path.is_absolute() and '..' not in path.parts, 'invalid DSL sample path')
        plot = read_plot(self.directory/path)
        require(len(plot.levels) == self.levels and plot.low == (0, 0, 0) and plot.high == (1, 1, self.depth) and
                math.isclose(plot.time, frame/20, rel_tol=1e-12, abs_tol=1e-14), 'wrong native DSL domain/time')
        for lev, level in enumerate(plot.levels):
            n = self.n*2**lev
            require(level.valid.shape == (n//32 if self.dimensions == 3 else 1, n, n) and
                    level.step == step*2**lev and sorted(level.boxes) == sorted(self.boxes[lev]), 'native DSL layout/step differs')
            expected = requested_rows(n, self.layout, lev)
            require(np.array_equal(level.valid, np.broadcast_to(expected[None, :, None], level.valid.shape)),
                    'actual A differs from the prescribed DSL bands')
        return plot


def statistics(plot, masks, curls, dimensions):
    volume = math.prod(b-a for a, b in zip(plot.low, plot.high))
    energy = enstrophy = uz2 = 0.0
    uz_peak = vorticity_peak = mach_peak = 0.0
    deviations = np.zeros(4)
    for level, mask, curl in zip(plot.levels, masks, curls):
        dv = math.prod(level.spacing)
        velocity = level.values[1:, mask]
        speed2 = np.sum(velocity*velocity, axis=0)
        energy += .5*np.sum(speed2)*dv/volume
        enstrophy += np.sum(curl[:, mask]**2)*dv/volume
        uz2 += np.sum(level.values[3][mask]**2)*dv/volume
        uz_peak = max(uz_peak, float(np.max(np.abs(level.values[3][mask]))))
        vorticity_peak = max(vorticity_peak, float(np.max(np.abs(curl[-1][mask]))))
        mach_peak = max(mach_peak, float(np.sqrt(3)*.1*np.sqrt(np.max(speed2))))
        if dimensions == 3:
            columns = level.values.reshape(4, level.valid.shape[0], -1)[:, :, mask[0].ravel()]
            deviations = np.maximum(deviations, np.max(np.abs(columns-columns.mean(axis=1, keepdims=True)), axis=(1, 2)))
    result = dict(energy=float(energy), enstrophy=float(enstrophy), uz_peak=uz_peak, uz_rms=math.sqrt(uz2),
                  z_rho=deviations[0], z_ux=deviations[1], z_uy=deviations[2], z_uz=deviations[3],
                  vorticity_peak=vorticity_peak, mach_peak=mach_peak)
    require(all(math.isfinite(v) for v in result.values()), 'nonfinite DSL statistics')
    return result


def analyze(run, output):
    output.mkdir(parents=True, exist_ok=True)
    (output/'identity.txt').write_text(''.join(f'{key}={value}\n' for key, value in run.id.items())+
        'energy_definition=mean_u_norm_squared/2\nenstrophy_definition=mean_curl_phys_u_norm_squared\n'
        f'integration=native-uncovered-valid\nderivative={DERIVATIVE}\n')
    initial = run.plot(0); masks = composite_masks(initial)
    plan = CurlPlan(initial, masks, run.dimensions)
    m0, p0 = totals(initial, masks)
    plane_identity = {key: run.id[key] for key in ('source_id', 'descriptor', 'backend', 'layout', 'base_cells')}
    plane_identity['derivative'] = DERIVATIVE
    with (output/'curves.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, COLUMNS); writer.writeheader()
        for frame in range(41):
            try:
                plot = initial if frame == 0 else run.plot(frame)
                curls = plan.curl(plot)
                result = statistics(plot, masks, curls, run.dimensions)
                mass, momentum = totals(plot, masks)
                dm = abs(mass-m0)/m0; dp = float(np.linalg.norm(momentum-p0)/m0)
                require(math.isfinite(dm) and math.isfinite(dp), 'nonfinite DSL conservation drift')
                writer.writerow(dict(frame=frame, time=plot.time, step=plot.levels[0].step, **result,
                    mass=mass, px=momentum[0], py=momentum[1], pz=momentum[2], mass_drift=dm, momentum_drift=dp))
                stream.flush()
                print(f'DSL analyzed frame={frame} time={plot.time:.12g} E={result["energy"]:.12g} Z={result["enstrophy"]:.12g}', flush=True)
                for lev, (level, mask, curl) in enumerate(zip(plot.levels, masks, curls)):
                    fields = dict(vorticity=curl[-1], mach=np.sqrt(3)*.1*np.sqrt(np.sum(level.values[1:]**2, axis=0)),
                                  rho=level.values[0], ux=level.values[1], uy=level.values[2], uz=level.values[3])
                    dump_planes(output/'planes', f'frame{frame}-L{lev}', fields, mask, level.spacing, run.dimensions,
                                time=plot.time, frame=frame, level=lev, step=level.step,
                                coarsest_step=plot.levels[0].step, phase='Sync', **plane_identity)
            except (ArtifactError, OSError, ValueError, KeyError, csv.Error, np.linalg.LinAlgError) as error:
                raise ArtifactError(f'frame={frame} nominal_time={frame/20}: {error}') from error
    (output/'analysis-result.txt').write_text('exit=0\nframes=41\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path); parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        analyze(DslRun(args.run), args.output)
        return 0
    except (ArtifactError, OSError, ValueError, KeyError, csv.Error, np.linalg.LinAlgError) as error:
        print(f'INVALID DSL analysis: {error}')
        try:
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output/'analysis-result.txt').write_text(f'exit=2\nerror={error}\n')
        except OSError as output_error:
            print(f'Cannot save invalid-result record: {output_error}')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
