#!/usr/bin/env python3
"""Check a complete periodic-shear AMR run against its matching uniform run."""
import argparse
import csv
import math
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'common'))
from amr_plotfile import (ArtifactError, require, identity, rows, read_plot, composite_masks,
                          conservative_average, totals)
from shear_metrics import (U, AMPLITUDE, NU, exact, residual, interface_faces, face_distance,
                           four_point, norm, merge_norms, visual_fields, expected_refined)
from native_planes import dump_planes

SYNC = (0, 1/8, 1/4, 1/2, 1, 2, 4)
MAIN = (1, 2, 4)


class Run:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.id = identity(directory)
        self.nc = int(self.id['coarse_cells'])
        self.dimensions = int(self.id['dimensions'])
        self.uniform = self.id['layout'] == 'uniform'
        self.n0 = self.nc*(2 if self.uniform else 1)
        self.mode = self.id['mode']
        require(self.id['collision'] == 'BGK' and self.id['periodic'] == '1,1,1' and
                self.id['initialization'] == 'cell-center-equilibrium', 'wrong collision/boundary/initialization')
        require(self.id['schema'] == 'periodic-shear-v1' and self.mode in ('axis', 'diagonal'), 'wrong shear identity')
        require((self.id['descriptor'], self.dimensions) in (('D2Q9', 2), ('D3Q27', 3)) and
                self.id['backend'] in ('cpu', 'cuda') and int(self.id['real_digits']) == 53, 'descriptor/backend/precision mismatch')
        grids = (64, 128, 256) if self.dimensions == 2 else (32, 64, 128)
        require(self.nc in grids and int(self.id['base_cells']) == self.n0 and
                int(self.id['levels']) == (1 if self.uniform else 2), 'wrong grid identity')
        require(int(self.id['full_steps']) == int(self.id['requested_steps']) == 4*self.n0, 'incomplete requested horizon')
        for key, expected in [('viscosity', NU), ('background', U), ('amplitude', AMPLITUDE),
                              ('rho_reference', 1), ('velocity_reference', 1), ('gap_fill_radius', 4),
                              ('coarse_nesting_margin', 4), ('grid_efficiency', 1)]:
            require(math.isclose(float(self.id[key]), expected, rel_tol=1e-14), f'wrong {key}')
        for lev in range(int(self.id['levels'])):
            cells = self.n0*2**lev
            require(math.isclose(float(self.id[f'dt_{lev}']), 1/cells, rel_tol=1e-13) and
                    math.isclose(float(self.id[f'tau_{lev}']), 0.5+3*NU*cells, rel_tol=1e-13), 'wrong level scaling')
        self.samples = {}
        for row in rows(self.directory/'samples.csv'):
            kind, lev, step = row['kind'], int(row['level']), int(row['step'])
            key = kind, lev, step
            require(key not in self.samples and kind in ('sync', 'fine'), 'duplicate/unknown sample')
            require(lev in range(int(self.id['levels'])), 'invalid sample level')
            t = step/(self.n0*2**lev)
            require(math.isclose(float(row['time']), t, rel_tol=1e-12, abs_tol=1e-14), 'wrong sample time')
            if kind == 'fine':
                require(not self.uniform and lev == 1 and int(row['coarsest_step']) == (step+1)//2 and
                        row['phase'] == ('First' if step % 2 else 'Second'), 'wrong fine phase/coarsest step')
            else:
                require(row['phase'] == 'Sync' and step == int(row['coarsest_step'])*2**lev, 'wrong synchronization identity')
            self.samples[key] = row
        for t in SYNC:
            for lev in range(int(self.id['levels'])):
                require(('sync', lev, round(t*self.n0*2**lev)) in self.samples, f'missing sync sample at {t}')
        for t in MAIN:
            for j in range(4):
                key = ('sync', 0, t*2*self.nc-j) if self.uniform else ('fine', 1, t*2*self.nc-j)
                require(key in self.samples, f'missing four-point sample {key}')
        self.layout = {}
        for row in rows(self.directory/'layout.csv'):
            lev = int(row['level'])
            bounds = tuple(int(row[k]) for k in ('ilo', 'jlo', 'klo', 'ihi', 'jhi', 'khi'))
            self.layout.setdefault(lev, []).append((bounds[:3], bounds[3:]))

    def plot(self, step, fine=False):
        key = ('fine', 1, step) if fine else ('sync', 0, step)
        row = self.samples[key]
        path = Path(row['path'])
        require(not path.is_absolute() and '..' not in path.parts, 'invalid sample path')
        plot = read_plot(self.directory/path)
        require(math.isclose(plot.time, float(row['time']), rel_tol=1e-12, abs_tol=1e-14), 'plot/index time differs')
        require(len(plot.levels) == (1 if fine else int(self.id['levels'])), 'plot lost a level')
        for lev, level in enumerate(plot.levels):
            source_level = 1 if fine else lev
            expected_step = step if fine else step*2**lev
            require(level.step == expected_step and sorted(level.boxes) == sorted(self.layout[source_level]),
                    'native steps/coverage differ from layout')
            n = self.n0*2**source_level
            shape = (n if self.dimensions == 3 else 1, n, n)
            require(level.valid.shape == shape, 'native grid shape differs')
        expected_hi = (1., 1., 1. if self.dimensions == 3 else 1/16)
        require(plot.low == (0., 0., 0.) and plot.high == expected_hi, 'wrong physical domain')
        return plot


def scientific_failures(run, records, balances):
    """Apply the case budgets to complete, validated statistics."""
    failed = set()
    for row in balances:
        if max(float(row['mass_drift']), float(row['momentum_drift'])) > 1e-10:
            failed.add(f"{row['role']}: conservation at t={float(row['time']):g}")
    if run.id['layout'] == 'half' and run.nc != (256 if run.dimensions == 2 else 128):
        return sorted(failed)
    d3_complex = (run.dimensions == 3 and run.mode == 'diagonal' and run.nc == 64 and
                  run.id['layout'] in ('corner', 'shell', 'separated'))
    interface, pressure, four_velocity = (0.06, 0.015, 0.03) if d3_complex else (0.05, 0.01, 0.01)
    limits = {
        'analytic-domain': ('l2', 0.01, 'velocity analytic'),
        'reference-domain': ('l2', 0.01, 'velocity reference'),
        'analytic-interface': ('peak', interface, 'velocity analytic'),
        'reference-interface': ('peak', interface, 'velocity reference'),
        'pressure': ('peak', pressure, 'pressure'),
        'spatial-velocity': ('peak', four_velocity, 'spatial velocity'),
        'spatial-pressure': ('peak', 0.01, 'spatial pressure'),
        'temporal-velocity': ('peak', four_velocity, 'temporal velocity'),
        'temporal-pressure': ('peak', 0.01, 'temporal pressure'),
    }
    for row in records:
        time = float(row['time'])
        comparison = row['comparison']
        if time not in MAIN or comparison not in limits:
            continue
        field, limit, label = limits[comparison]
        if float(row[field]) > limit:
            if comparison.startswith('spatial-'):
                label += f" axis={row['region'].removeprefix('axis-')}"
            failed.add(f'{label} t={time:g}')
    return sorted(failed)


def report_case(amr, reference, output):
    output.mkdir(parents=True, exist_ok=True)
    require(not amr.uniform and reference.uniform and amr.id['layout'] in ('half', 'corner', 'shell', 'separated'), 'wrong run roles')
    require(all(amr.id[k] == reference.id[k] for k in ('descriptor', 'mode', 'coarse_cells', 'source_id', 'backend')), 'reference identity differs')
    if amr.id['layout'] != 'half':
        require(amr.mode == 'diagonal' and amr.nc == (128 if amr.dimensions == 2 else 64), 'wrong complex configuration')
    initial = amr.plot(0)
    masks = composite_masks(initial)
    require(initial.levels[0].valid.all(), 'incomplete coarse storage')
    refined = initial.levels[0].valid & ~masks[0]
    require(np.array_equal(refined, expected_refined(refined.shape, initial.levels[0].spacing,
                                                   amr.dimensions, amr.id['layout'])),
            'actual coverage differs from the prescribed shear layout')
    faces = interface_faces(refined, initial.levels[0].spacing, amr.dimensions)
    require(faces, 'missing actual coarse/fine interfaces')
    with (output/'interfaces.csv').open('w', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(('face', 'axis', 'xlo', 'ylo', 'zlo', 'xhi', 'yhi', 'zhi'))
        for n, (axis, lo, hi) in enumerate(faces):
            writer.writerow((n, axis, *lo, *hi))
    band_masks = []
    for level in initial.levels:
        distance = np.full(level.valid.shape, np.inf)
        for face in faces:
            np.minimum(distance, face_distance(level.valid.shape, level.spacing, initial.high, face, amr.dimensions), out=distance)
        band_masks.append(distance <= 1/16+1e-15)
    balance_initial = {'amr': totals(initial, masks)}
    ref0 = reference.plot(0)
    balance_initial['reference'] = totals(ref0, composite_masks(ref0))
    del initial, ref0
    plane_identity = dict(source_id=amr.id['source_id'], descriptor=amr.id['descriptor'], backend=amr.id['backend'],
                          layout=amr.id['layout'], mode=amr.mode, coarse_cells=amr.nc,
                          reference_directory=str(reference.directory.resolve()))
    records = []
    balances = []
    profiles = []
    for t in SYNC:
        actual = amr.plot(round(t*amr.n0)); ref = reference.plot(round(t*reference.n0))
        ref_values = ref.levels[0].values
        for role, plot in (('amr', actual), ('reference', ref)):
            mass, momentum = totals(plot, composite_masks(plot))
            m0, p0 = balance_initial[role]
            dm, dp = abs(mass-m0)/m0, float(np.linalg.norm(momentum-p0)/(m0*U))
            require(math.isfinite(dm) and math.isfinite(dp), 'nonfinite conservation drift')
            balances.append(dict(role=role, time=t, mass=mass, px=momentum[0], py=momentum[1], pz=momentum[2], mass_drift=dm, momentum_drift=dp))
        ref_exact = exact(ref_values.shape[1:], ref.levels[0].spacing, t, amr.dimensions, amr.mode)
        ref_error = np.sqrt(np.sum(residual(ref_values, ref_exact)[1:]**2, axis=0))
        records.append(dict(time=t, comparison='reference-analytic', region='domain', level=0,
                            **norm(ref_error, ref.levels[0].valid, ref.levels[0].spacing, amr.dimensions)))
        del ref_exact, ref_error
        aggregates = {key: [] for key in ('analytic-domain', 'reference-domain', 'analytic-interface', 'reference-interface', 'pressure')}
        for lev, (level, mask, band) in enumerate(zip(actual.levels, masks, band_masks)):
            analytic = exact(mask.shape, level.spacing, t, amr.dimensions, amr.mode)
            mapped = conservative_average(ref_values, mask.shape)
            ra, rr = residual(level.values, analytic), residual(level.values, mapped)
            for label, difference in (('analytic', ra), ('reference', rr)):
                error = np.sqrt(np.sum(difference[1:]**2, axis=0))
                for region, selected in (('domain', mask), ('interface', mask & band)):
                    result = norm(error, selected, level.spacing, amr.dimensions)
                    if result:
                        aggregates[f'{label}-{region}'].append(result)
                        records.append(dict(time=t, comparison=f'amr-{label}', region=region, level=lev, **result))
                if t in MAIN:
                    for index, face in enumerate(faces):
                        near = mask & (face_distance(mask.shape, level.spacing, actual.high, face, amr.dimensions) <= 1/16+1e-15)
                        result = norm(error, near, level.spacing, amr.dimensions)
                        if result:
                            records.append(dict(time=t, comparison=f'amr-{label}', region=f'face-{index}', level=lev, **result))
                            if label == 'reference':
                                cell = [result['k'], result['j'], result['i']]
                                axis = face[0]
                                for position in range(mask.shape[2-axis]):
                                    cell[2-axis] = position
                                    z, y, x = cell
                                    if not mask[z, y, x]:
                                        continue
                                    row = dict(time=t, level=lev, face=index, axis=axis, i=x, j=y, k=z,
                                               x=(x+0.5)*level.spacing[0], y=(y+0.5)*level.spacing[1], z=(z+0.5)*level.spacing[2])
                                    for prefix, values in (('amr', level.values), ('analytic', analytic), ('reference', mapped)):
                                        for q, variable in enumerate(('rho', 'ux', 'uy', 'uz')):
                                            row[f'{prefix}_{variable}'] = float(values[q, z, y, x])
                                    profiles.append(row)
            pressure = norm(np.abs(rr[0]), mask, level.spacing, amr.dimensions)
            aggregates['pressure'].append(pressure)
            records.append(dict(time=t, comparison='pressure-reference', region='domain', level=lev, **pressure))
            if t in MAIN:
                fields = visual_fields(level.values, analytic, mapped, amr.dimensions, amr.mode, ra, rr)
                dump_planes(output/'planes', f't{t}-L{lev}', fields, mask, level.spacing, amr.dimensions,
                            level=lev, time=t, step=level.step, coarsest_step=actual.levels[0].step, phase='Sync', **plane_identity)
                if lev == 1:
                    for axis, coefficient, valid in four_point(rr, level.valid, amr.dimensions):
                        v = norm(np.sqrt(np.sum(coefficient[1:]**2, axis=0)), valid, level.spacing, amr.dimensions)
                        p = norm(np.abs(coefficient[0]), valid, level.spacing, amr.dimensions)
                        require(v is not None, 'no legal spatial four-point windows')
                        for quantity, metric in (('velocity', v), ('pressure', p)):
                            records.append(dict(time=t, comparison=f'spatial-{quantity}', region=f'axis-{axis}', level=lev, **metric))
                        dump_planes(output/'planes', f'spatial-t{t}-axis{axis}',
                            {f'coefficient_{q}': coefficient[q] for q in range(4)}, valid, level.spacing,
                            amr.dimensions, time=t, level=lev, axis=axis, **plane_identity)
            del analytic, mapped, ra, rr
        for key, parts in aggregates.items():
            records.append(dict(time=t, comparison=key, region='composite', level=-1, **merge_norms(parts)))
        del actual, ref, ref_values
    for t in MAIN:
        coefficient = None
        for j, weight in enumerate((1, -3, 3, -1)):
            step = t*2*amr.nc-3+j
            actual = amr.plot(step, fine=True); ref = reference.plot(step)
            level = actual.levels[0]
            r = residual(level.values, ref.levels[0].values)
            coefficient = weight*r if coefficient is None else coefficient+weight*r
            row = amr.samples['fine', 1, step]
            dump_planes(output/'planes', f'fine-t{t}-j{j}',
                        {**{f'residual_{q}': r[q] for q in range(4)}, **{f'raw_{q}': level.values[q] for q in range(4)}},
                        level.valid, level.spacing, amr.dimensions, time=actual.time, level=1, step=step,
                        coarsest_step=int(row['coarsest_step']), phase=row['phase'], endpoint=t, **plane_identity)
        coefficient /= 8
        for quantity, field in (('velocity', np.sqrt(np.sum(coefficient[1:]**2, axis=0))), ('pressure', np.abs(coefficient[0]))):
            metric = norm(field, level.valid, level.spacing, amr.dimensions)
            records.append(dict(time=t, comparison=f'temporal-{quantity}', region='fine', level=1, **metric))
        dump_planes(output/'planes', f'temporal-t{t}', {f'coefficient_{q}': coefficient[q] for q in range(4)},
                    level.valid, level.spacing, amr.dimensions, time=t, level=1, **plane_identity)
        del actual, ref, coefficient
    failed = scientific_failures(amr, records, balances)
    write_csv(output/'metrics.csv', records)
    write_csv(output/'conservation.csv', balances)
    write_csv(output/'interface-profiles.csv', profiles)
    (output/'result.txt').write_text(f'exit={int(bool(failed))}\namr={amr.directory.resolve()}\nreference={reference.directory.resolve()}\nsource_id={amr.id["source_id"]}\n'+''.join(f'failure={s}\n' for s in failed))
    print(f'{amr.directory.name}: {"FAIL" if failed else "PASS"} failures={len(failed)}')
    return int(bool(failed))


def write_csv(path, records):
    fields = list(dict.fromkeys(key for row in records for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields); writer.writeheader(); writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--amr', required=True, type=Path)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        return report_case(Run(args.amr), Run(args.reference), args.output)
    except (ArtifactError, OSError, ValueError, KeyError, StopIteration, csv.Error) as error:
        print(f'INVALID: {error}')
        try:
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output/'result.txt').write_text(f'exit=2\nerror={error}\n')
        except OSError as output_error:
            print(f'Cannot save invalid-result record: {output_error}')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
