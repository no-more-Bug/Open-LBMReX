#!/usr/bin/env python3
"""Observe periodic DSL on native AMR cells with second-order composite curl."""
import argparse
import csv
import math
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'common'))
from amr_plotfile import ArtifactError, require, identity, rows, read_plot, composite_masks, totals
from native_planes import dump_planes
from amr_validation import common_identity
from analyze_double_shear_layer import DslRun, COLUMNS
from dynamic_vorticity import curls, restricted

DERIVATIVE = 'central2-native-composite-amrex-mc'
OUTPUT_COLUMNS = (*COLUMNS, 'active_levels', 'composite_cells', 'velocity_l2', 'velocity_linf')


class DynamicRun:
    schema = "dynamic-double-shear-v1"

    def __init__(self, directory, short=False):
        self.directory = Path(directory)
        self.id = identity(directory)
        self.dimensions = common_identity(self.id, self.schema)
        self.max_level = int(self.id['max_level'])
        self.n = int(self.id['base_cells'])
        self.depth = 1/32 if self.dimensions == 3 else 1/16
        require(self.id['layout'] == 'dynamic' and self.max_level in (1, 2) and self.n == 128 and
                int(self.id['max_grid_size']) >= 8 and int(self.id['max_grid_size'])%2 == 0,
                'invalid dynamic DSL grid identity')
        require(self.id['periodic'] == '1,1,1' and
                self.id['initialization'] == 'cell-center-equilibrium', 'wrong dynamic composition')
        for key, value in (('reynolds', 10000), ('u0_lb', .1), ('u0_phys', 1), ('viscosity', .0001),
                           ('kappa', 80), ('delta', .05), ('rho_reference', 1), ('velocity_reference', .1),
                           ('gap_fill_radius', 4), ('coarse_nesting_margin', 4)):
            require(math.isclose(float(self.id[key]), value, rel_tol=1e-13), f'wrong DSL {key}')
        efficiency = float(self.id['grid_efficiency'])
        require(math.isfinite(efficiency) and 0 < efficiency <= 1, 'invalid DSL grid_efficiency')
        require(float(self.id['length_z']) == self.depth and
                int(self.id['base_z']) == (4 if self.dimensions == 3 else 1), 'wrong DSL thickness')
        last = int(self.id['requested_steps'])
        require(int(self.id['full_steps']) == 20*self.n and int(self.id['frame_stride']) == self.n//2 and
                0 <= last <= 20*self.n and (short or last == 20*self.n), 'incomplete DSL horizon')
        self.steps = list(range(0, last+1, self.n//2))
        if self.steps[-1] != last:
            self.steps.append(last)
        require(int(self.id['regrid_interval']) >= 0 and int(self.id['ranks']) > 0, 'invalid runtime controls')
        for lev in range(self.max_level+1):
            require(math.isclose(float(self.id[f'dt_{lev}']), .1/(self.n*2**lev), rel_tol=1e-13) and
                    math.isclose(float(self.id[f'tau_{lev}']), .5+3*.1*self.n*2**lev/10000, rel_tol=1e-13),
                    'wrong fixed allowed-level scaling')
            require(int(self.id[f'tag_buffer_{lev}']) >= 0, 'invalid tag buffer')
            if lev < self.max_level:
                refine, keep = (float(self.id[f'theta_{name}_{lev}']) for name in ('refine', 'keep'))
                require(math.isfinite(refine) and 0 <= keep < refine, 'invalid hysteresis')
        self.samples = {}
        for row in rows(self.directory/'samples.csv'):
            step, lev = int(row['coarsest_step']), int(row['level'])
            require((step, lev) not in self.samples and step in self.steps and 0 <= lev <= self.max_level and
                    row['kind'] == 'sync' and row['phase'] == 'Sync' and row['path'] == f'sync-{step}' and
                    int(row['step']) == step*2**lev and
                    math.isclose(float(row['time']), step*.1/self.n, abs_tol=1e-14, rel_tol=1e-12),
                    'invalid dynamic sample')
            self.samples[step, lev] = row
        for step in self.steps:
            levels = sorted(lev for s, lev in self.samples if s == step)
            require(levels and levels == list(range(len(levels))), 'missing frame or noncontiguous levels')

    def layout(self, frame):
        step = self.steps[frame]
        levels = sum(s == step for s, _ in self.samples)
        boxes = {}
        for row in rows(self.directory/f'layout-{step}.csv'):
            lev = int(row['level'])
            require(0 <= lev < levels and 0 <= int(row['rank']) < int(self.id['ranks']), 'invalid layout owner/level')
            entries = boxes.setdefault(lev, [])
            require(int(row['box']) == len(entries), 'missing/duplicate layout box')
            entries.append((tuple(int(row[k]) for k in ('ilo', 'jlo', 'klo')),
                            tuple(int(row[k]) for k in ('ihi', 'jhi', 'khi'))))
        require(set(boxes) == set(range(levels)), 'missing layout level')
        return [boxes[lev] for lev in range(levels)]

    def plot(self, frame):
        step = self.steps[frame]
        boxes = self.layout(frame)
        plot = read_plot(self.directory/f'sync-{step}')
        require(len(plot.levels) == len(boxes) and plot.low == (0, 0, 0) and plot.high == (1, 1, self.depth) and
                math.isclose(plot.time, step*.1/self.n, rel_tol=1e-12, abs_tol=1e-14),
                'native dynamic domain/time/levels differ')
        for lev, level in enumerate(plot.levels):
            n = self.n*2**lev
            require(level.step == step*2**lev and
                    level.valid.shape == (n//32 if self.dimensions == 3 else 1, n, n) and
                    level.boxes == boxes[lev], 'native dynamic layout/step differs')
        return plot



class FixedBandRun(DynamicRun):
    def __init__(self, directory, short=False):
        self.schema = identity(directory).get("schema")
        require(self.schema in ("fixed-band-double-shear-v1", "fixed-band-double-shear-v2"),
                "unsupported fixed-band schema")
        super().__init__(directory, short)
        if self.schema == "fixed-band-double-shear-v1":
            self.band_max_level = 1
            require(self.id.get("case_variant") == "fixed-band" and
                    int(self.id["fixed_band_level"]) == 1, "wrong legacy fixed-band bounds")
        else:
            self.band_max_level = 2
            require(self.id.get("case_variant") == "minimum-band" and
                    int(self.id["fixed_band_min_level"]) == 1 and
                    int(self.id["fixed_band_max_level"]) == 2, "wrong minimum-band bounds")
        require(self.max_level == 2 and
                int(self.id["fixed_band_parent_jlo"]) == 83 and
                int(self.id["fixed_band_parent_jhi"]) == 109, "wrong fixed-band composition")
        for key, value in (("fixed_band_y_lo", .65), ("fixed_band_y_hi", .85),
                           ("fixed_band_aligned_y_lo", 83/128),
                           ("fixed_band_aligned_y_hi", 109/128),
                           ("grid_efficiency", .8)):
            require(math.isclose(float(self.id[key]), value, rel_tol=0, abs_tol=1e-14),
                    f"wrong fixed-band {key}")
        require(int(self.id["regrid_interval"]) == 8 and
                all(int(self.id[f"tag_buffer_{lev}"]) == 2 for lev in range(3)) and
                all(float(self.id[f"theta_refine_{lev}"]) == 20 and
                    float(self.id[f"theta_keep_{lev}"]) == 10 for lev in range(2)),
                "wrong fixed-band tagging controls")


def open_fixed_band(directory, short=False):
    status = identity(directory, "exit.txt")
    require(status.get("solver_exit") == "0" and status.get("tee_exit") == "0",
            "solver/tee did not finish successfully")
    return FixedBandRun(directory, short)

def open_run(directory, short=False):
    status = identity(directory, 'exit.txt')
    require(status.get('solver_exit') == '0' and status.get('tee_exit') == '0',
            'solver/tee did not finish successfully')
    if identity(directory)['schema'] == 'dynamic-double-shear-v1':
        return DynamicRun(directory, short)
    require(not short, 'short mode is only for dynamic engineering runs')
    run = DslRun(directory)
    run.require_complete()
    run.steps = [frame*run.n//2 for frame in range(41)]
    return run


def statistics(plot, masks, vorticity, dimensions):
    volume = math.prod(b-a for a, b in zip(plot.low, plot.high))
    energy = enstrophy = uz2 = 0.0
    uz_peak = vorticity_peak = mach_peak = 0.0
    deviations = np.zeros(4)
    for level, mask, curl in zip(plot.levels, masks, vorticity):
        if not np.any(mask):
            continue
        dv = math.prod(level.spacing)
        velocity = level.values[1:, mask]
        speed2 = np.sum(velocity**2, axis=0)
        energy += .5*np.sum(speed2)*dv/volume
        enstrophy += np.sum(curl[:, mask]**2)*dv/volume
        uz2 += np.sum(velocity[2]**2)*dv/volume
        uz_peak = max(uz_peak, float(np.max(np.abs(velocity[2]))))
        vorticity_peak = max(vorticity_peak, float(np.max(np.sqrt(np.sum(curl[:, mask]**2, axis=0)))))
        mach_peak = max(mach_peak, float(np.sqrt(3)*.1*np.sqrt(np.max(speed2))))
        if dimensions == 3:
            count = mask.sum(axis=0)
            mean = np.where(mask[None], level.values, 0).sum(axis=1)/np.maximum(count, 1)[None]
            difference = np.where(mask[None], np.abs(level.values-mean[:, None]), 0)
            deviations = np.maximum(deviations, difference.max(axis=(1, 2, 3)))
    result = dict(energy=float(energy), enstrophy=float(enstrophy), uz_peak=uz_peak, uz_rms=math.sqrt(uz2),
                  z_rho=deviations[0], z_ux=deviations[1], z_uy=deviations[2], z_uz=deviations[3],
                  vorticity_peak=vorticity_peak, mach_peak=mach_peak)
    require(all(math.isfinite(v) for v in result.values()), 'nonfinite composite diagnostics')
    return result


def velocity_error(plot, reference, masks):
    require(len(reference.levels) == 1 and reference.low == plot.low and reference.high == plot.high and
            math.isclose(plot.time, reference.time, abs_tol=1e-14, rel_tol=1e-12), 'reference domain/time mismatch')
    fine = reference.levels[0].values[1:]
    numerator = denominator = peak = 0.0
    for level, mask in zip(plot.levels, masks):
        ratio = tuple(a//b for a, b in zip(fine.shape[1:], level.valid.shape))
        require(all(r >= 1 and a == r*b for r, a, b in zip(ratio, fine.shape[1:], level.valid.shape)),
                'reference grid does not match native cells')
        matched = restricted(fine, ratio)
        diff = level.values[1:, mask]-matched[:, mask]
        dv = math.prod(level.spacing)
        numerator += float(np.sum(diff**2))*dv
        denominator += float(np.sum(matched[:, mask]**2))*dv
        if diff.size:
            peak = max(peak, float(np.max(np.sqrt(np.sum(diff**2, axis=0)))))
    require(denominator > 0, 'zero reference velocity norm')
    return math.sqrt(numerator/denominator), peak


def analyze(run, output, reference=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = dict(run.id, derivative=DERIVATIVE, integration='native-uncovered-valid',
                    energy_definition='mean_u_norm_squared/2',
                    enstrophy_definition='mean_curl_phys_u_norm_squared')
    if reference:
        require(reference.id['layout'] == 'uniform' and reference.id['descriptor'] == run.id['descriptor'] and
                reference.n == run.n*2**(run.max_level if isinstance(run, DynamicRun) else run.levels-1),
                'wrong matching uniform reference')
        metadata.update(reference_directory=str(reference.directory), reference_source_id=reference.id['source_id'],
                        reference_backend=reference.id['backend'])
    (output/'identity.txt').write_text(''.join(f'{key}={value}\n' for key, value in metadata.items()))
    m0 = p0 = None
    with (output/'curves.csv').open('w', newline='') as stream, (output/'mesh.csv').open('w', newline='') as mesh:
        writer = csv.DictWriter(stream, OUTPUT_COLUMNS)
        writer.writeheader()
        layout = csv.writer(mesh)
        layout.writerow(('frame', 'time', 'level', 'valid_cells', 'composite_cells', 'boxes'))
        for frame, step in enumerate(run.steps):
            plot = run.plot(frame)
            masks = composite_masks(plot)
            vorticity = curls(plot, run.dimensions)
            result = statistics(plot, masks, vorticity, run.dimensions)
            mass, momentum = totals(plot, masks)
            if m0 is None:
                m0, p0 = mass, momentum
            errors = velocity_error(plot, reference.plot(frame), masks) if reference else ('', '')
            writer.writerow(dict(frame=frame, time=plot.time, step=step, **result, mass=mass,
                px=momentum[0], py=momentum[1], pz=momentum[2], mass_drift=abs(mass-m0)/m0,
                momentum_drift=float(np.linalg.norm(momentum-p0)/m0), active_levels=len(plot.levels),
                composite_cells=sum(int(mask.sum()) for mask in masks),
                velocity_l2=errors[0], velocity_linf=errors[1]))
            stream.flush()
            for lev, (level, mask, curl) in enumerate(zip(plot.levels, masks, vorticity)):
                layout.writerow((frame, plot.time, lev, int(level.valid.sum()), int(mask.sum()), len(level.boxes)))
                fields = dict(vorticity=curl[2], rho=level.values[0], ux=level.values[1],
                              uy=level.values[2], uz=level.values[3])
                dump_planes(output/'planes', f'frame{frame}-L{lev}', fields, mask, level.spacing, run.dimensions,
                    frame=frame, time=plot.time, level=lev, step=level.step, coarsest_step=step,
                    phase='Sync', derivative=DERIVATIVE, **{k: run.id[k] for k in ('source_id', 'descriptor', 'backend', 'layout', 'base_cells', 'schema', 'case_variant') if k in run.id})
            print(f'native DSL frame={frame} time={plot.time:.12g} levels={len(plot.levels)} '
                  f'E={result["energy"]:.12g} Z={result["enstrophy"]:.12g}', flush=True)
    (output/'analysis-result.txt').write_text(f'exit=0\nframes={len(run.steps)}\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--short', action='store_true')
    parser.add_argument('--fixed-band', action='store_true')
    args = parser.parse_args()
    try:
        opener = open_fixed_band if args.fixed_band else open_run
        analyze(opener(args.run, args.short), args.output,
                open_run(args.reference) if args.reference else None)
        return 0
    except (ArtifactError, OSError, ValueError, KeyError, csv.Error) as error:
        print(f'INVALID native DSL analysis: {error}')
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output/'analysis-result.txt').write_text(f'exit=2\nerror={error}\n')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
