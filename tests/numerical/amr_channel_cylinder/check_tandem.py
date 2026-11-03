#!/usr/bin/env python3
"""Tandem engineering checks: 0 pass, 1 numerical mismatch, 2 invalid evidence."""
import argparse
import csv
from pathlib import Path
import re
import sys
import numpy as np
from check_re20 import time_matches

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
from amr_plotfile import ArtifactError, require, identity, rows, read_plot, composite_masks

EPS = np.finfo(float).eps
DT = .000875
STEPS = 8
FORCE_FIELDS = ('kind', 'id', 'level', 'population_step', 'coarsest_step', 'begin_time',
                'end_time', 'sample_time', 'Fx_lb', 'Fy_lb', 'Fz_lb')


def close(actual, expected):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    return (a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all() and
            np.all(np.abs(a-b) <= 4096*EPS*(1+np.abs(b))))


def comparison(name, actual, expected):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    require(a.shape == b.shape and a.size and np.isfinite(a).all() and np.isfinite(b).all(),
            f'{name}: missing/nonfinite comparison values')
    error = float(np.max(np.abs(a-b)/(4096*EPS*(1+np.abs(b)))))
    return dict(check=name, max_scaled_error=error, passed=int(error <= 1))


def vector(text, cast=float):
    return tuple(cast(value) for value in text.split(','))


def boundary_map(records, layout, grouping):
    require(len(records) == 2 and {row['name'] for row in records} == {'cylinder_a', 'cylinder_b'},
            'expected two named boundaries')
    result = {}
    for row in records:
        name = row['name']
        bid, oid, level = (int(row[key]) for key in ('boundary_id', 'object_id', 'level'))
        expected_level = 1 if name == 'cylinder_b' and layout == 'cross' else 2
        require(bid >= 0 and oid >= 0 and level == expected_level and bid not in result,
                'wrong boundary identity/ownership')
        expected = (.2 if name == 'cylinder_a' else .4, .2, .005, .05)
        require(close([float(row[k]) for k in ('center_x', 'center_y', 'center_z', 'radius')], expected),
                'wrong tandem geometry')
        result[bid] = dict(name=name, object=oid, level=level)
    require(len({value['object'] for value in result.values()}) == (1 if grouping == 'shared' else 2),
            'wrong Object grouping')
    return result


def object_levels(boundaries):
    result = {}
    for value in boundaries.values():
        result.setdefault(value['object'], set()).add(value['level'])
    return result


def read_forces(records, boundaries, steps=STEPS):
    expected = set()
    for bid, value in boundaries.items():
        expected.update(('boundary', bid, value['level'], n)
                        for n in range(1, steps*(2**value['level'])+1))
    for oid, levels in object_levels(boundaries).items():
        if len(levels) == 1:
            level = next(iter(levels))
            expected.update(('object_step', oid, level, n) for n in range(1, steps*(2**level)+1))
        else:
            expected.update(('object_interval', oid, None, n) for n in range(1, steps+1))
    result = {}
    for row in records:
        require(set(row) == set(FORCE_FIELDS), 'wrong force CSV columns')
        kind, entity, coarse = row['kind'], int(row['id']), int(row['coarsest_step'])
        if kind == 'object_interval':
            require(row['level'] == row['population_step'] == row['sample_time'] == '',
                    'interval average must not claim a native step/sample time')
            level, step, dt = None, coarse, DT
        else:
            require(kind in ('boundary', 'object_step'), 'unknown force record kind')
            level, step = int(row['level']), int(row['population_step'])
            require(level in (1, 2) and coarse == (step-1)//(2**level)+1, 'wrong force step label')
            dt = DT/(2**level)
            require(time_matches(float(row['sample_time']), (step-.5)*dt, step, dt), 'wrong force midpoint label')
        require(time_matches(float(row['begin_time']), (step-1)*dt, step, dt) and
                time_matches(float(row['end_time']), step*dt, step, dt),
                'wrong force interval')
        key = (kind, entity, level, step)
        require(key in expected and key not in result, 'unexpected or duplicate force identity')
        value = np.array([float(row[k]) for k in ('Fx_lb', 'Fy_lb', 'Fz_lb')])
        require(np.isfinite(value).all(), 'nonfinite force')
        result[key] = value
    require(set(result) == expected, 'missing force samples or final batch')
    return result


def force_checks(boundaries, forces):
    checks = []
    for bid, value in sorted(boundaries.items()):
        peak = max(abs(force[0]) for key, force in forces.items() if key[:2] == ('boundary', bid))
        checks.append(dict(check=value['name']+'_nonzero_Fx', max_scaled_error='', passed=int(peak > 4096*EPS)))
    actual, expected = [], []
    for key, value in sorted(forces.items()):
        kind, oid, level, step = key
        if kind == 'boundary':
            continue
        total = np.zeros(3)
        for bid, boundary in boundaries.items():
            if boundary['object'] != oid:
                continue
            if kind == 'object_step':
                total += forces['boundary', bid, level, step]
            else:
                # Independently reconstruct the specified 1/4 and 1/2 weights.
                count = 2**boundary['level']
                for n in range((step-1)*count+1, step*count+1):
                    total += forces['boundary', bid, boundary['level'], n]/count
        actual.append(value)
        expected.append(total)
    checks.append(comparison('Object_from_Boundary', actual, expected))
    return checks


def validate_layout(records, plot, ranks):
    boxes = [[] for _ in plot.levels]
    for row in records:
        level, index, rank = (int(row[key]) for key in ('level', 'box', 'rank'))
        require(0 <= level < len(boxes) and index == len(boxes[level]) and 0 <= rank < ranks,
                'invalid layout level/Box/rank')
        boxes[level].append((tuple(int(row[k]) for k in ('ilo', 'jlo', 'klo')),
                             tuple(int(row[k]) for k in ('ihi', 'jhi', 'khi'))))
    require(all(actual and actual == data.boxes for actual, data in zip(boxes, plot.levels)),
            'layout does not match native field Boxes')


class TandemRun:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.id = identity(directory)
        meta = self.id
        self.dimensions = int(meta['dimensions'])
        require(meta['schema'] == 'tandem-v1' and meta['backend'] in ('cpu', 'cuda') and
                (meta['descriptor'], self.dimensions) in (('D2Q9', 2), ('D3Q27', 3)) and
                re.fullmatch(r'[0-9a-f]{40}', meta['source_id']) and int(meta['real_digits']) == 53,
                'wrong tandem source/backend/descriptor/precision')
        require(meta['scheme'] in ('YMS', 'Bouzidi') and meta['layout'] in ('cross', 'same') and
                meta['grouping'] in ('shared', 'distinct') and meta['collision'] == 'BGK' and
                meta['periodic'] == '0,0,1' and meta['initialization'] == 'uniform-equilibrium-then-Guo' and
                meta['force_units'] == 'finest-lattice', 'wrong tandem composition')
        require(int(meta['full_steps']) == STEPS and int(meta['levels']) == 3 and
                int(meta['ranks']) > 0 and int(meta['omp_threads']) == 1 and
                int(meta['max_grid_size']) >= 2 and int(meta['max_grid_size']) % 2 == 0,
                'wrong run/parallel identity')
        for key, value in (('reynolds', 20), ('u0_lb', .035), ('u0_phys', .2), ('viscosity', .001),
                           ('rho_reference', 1), ('velocity_reference', .035), ('comparison_dt', DT)):
            require(close(float(meta[key]), value), f'wrong {key}')
        require(close(vector(meta['physical_lo']), (0, 0, 0)) and
                close(vector(meta['physical_hi']), (2.2, .41, .01)), 'wrong physical channel')
        exits = dict(line.split('=', 1) for line in (self.directory/'exit.txt').read_text().splitlines())
        require(exits == {'solver_exit': '0', 'tee_exit': '0'}, 'solver/output did not complete')
        self.boundaries = boundary_map(rows(self.directory/'boundaries.csv'), meta['layout'], meta['grouping'])
        self.forces = read_forces(rows(self.directory/'forces.csv'), self.boundaries)
        self.checks = force_checks(self.boundaries, self.forces)
        samples = rows(self.directory/'samples.csv')
        require(len(samples) == 3 and {int(row['level']) for row in samples} == {0, 1, 2},
                'missing/duplicate terminal level records')
        for row in samples:
            level = int(row['level'])
            require(row['kind'] == 'sync' and row['phase'] == 'Sync' and row['path'] == 'sync-8' and
                    int(row['step']) == STEPS*(2**level) and int(row['coarsest_step']) == STEPS and
                    close(float(row['time']), STEPS*DT), 'wrong terminal step/time')
        self.plot = read_plot(self.directory/'sync-8')
        require(len(self.plot.levels) == 3 and close(self.plot.time, STEPS*DT) and
                close(self.plot.low, (-.0025, -.0025, 0)) and
                close(self.plot.high, (2.2025, .4125, .01)), 'wrong terminal field/envelope')
        for level, data in enumerate(self.plot.levels):
            ratio = 2**level
            cells = (441*ratio, 83*ratio, 2*ratio if self.dimensions == 3 else 1)
            spacing = (.005/ratio, .005/ratio, .005/ratio if self.dimensions == 3 else .01)
            require(data.step == STEPS*ratio and data.values.shape[1:] == tuple(reversed(cells)) and
                    vector(meta[f'cells_{level}'], int) == cells and close(data.spacing, spacing) and
                    close(vector(meta[f'dx_{level}']), spacing) and close(float(meta[f'dt_{level}']), DT/ratio),
                    'wrong native level geometry/clock')
            # Acoustic scaling: finest tau=.5+(.035*80/20)/(1/3)=.92.
            require(close(float(meta[f'tau_{level}']), .5+.42/(2**(2-level))), 'wrong level relaxation')
        require(self.plot.levels[0].valid.all(), 'incomplete L0 field')
        self.masks = composite_masks(self.plot)
        self.layout = rows(self.directory/'layout.csv')
        validate_layout(self.layout, self.plot, int(meta['ranks']))


def paired_checks(shared, distinct):
    require(shared.id['grouping'] == 'shared' and distinct.id['grouping'] == 'distinct',
            'pair must be shared versus distinct')
    require({k: v for k, v in shared.id.items() if k != 'grouping'} ==
            {k: v for k, v in distinct.id.items() if k != 'grouping'}, 'pair identities differ beyond grouping')
    require(shared.layout == distinct.layout, 'pair actual Box/rank layouts differ')
    require({bid: (b['name'], b['level']) for bid, b in shared.boundaries.items()} ==
            {bid: (b['name'], b['level']) for bid, b in distinct.boundaries.items()}, 'pair boundary identities differ')
    keys = sorted(key for key in shared.forces if key[0] == 'boundary')
    checks = [comparison('paired_Boundary', [shared.forces[k] for k in keys],
                         [distinct.forces[k] for k in keys])]
    for lev, (a, b) in enumerate(zip(shared.plot.levels, distinct.plot.levels)):
        require(np.array_equal(a.valid, b.valid), 'pair valid coverage differs')
        nx, ny = a.values.shape[3], a.values.shape[2]
        x = shared.plot.low[0]+(np.arange(nx)+.5)*a.spacing[0]
        y = shared.plot.low[1]+(np.arange(ny)+.5)*a.spacing[1]
        fluid = ((x[None, :]-.2)**2+(y[:, None]-.2)**2 >= .05**2)
        fluid &= ((x[None, :]-.4)**2+(y[:, None]-.2)**2 >= .05**2)
        mask = shared.masks[lev] & fluid[None, :, :]
        checks.append(comparison(f'paired_field_L{lev}', a.values[:, mask], b.values[:, mask]))
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--reference', type=Path, help='explicit matching distinct run for a shared run')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        run = TandemRun(args.run)
        checks = list(run.checks)
        if run.id['grouping'] == 'shared':
            require(args.reference is not None, 'shared run needs an explicit distinct reference')
            reference = TandemRun(args.reference)
            checks.extend(dict(row, check='reference_'+row['check']) for row in reference.checks)
            checks.extend(paired_checks(run, reference))
            (args.output/'reference-identity.txt').write_bytes((args.reference/'identity.txt').read_bytes())
            (args.output/'reference.txt').write_text(str(args.reference)+'\n')
        else:
            require(args.reference is None, 'distinct run does not take a reference')
        with (args.output/'checks.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, ('check', 'max_scaled_error', 'passed'))
            writer.writeheader(); writer.writerows(checks)
        result = int(any(not row['passed'] for row in checks))
        message = f'exit={result}\nforce_rows={len(run.forces)}\nchecks={len(checks)}\n'
    except (ArtifactError, OSError, ValueError, KeyError, OverflowError) as error:
        result = 2
        message = f'exit=2\nerror={error}\n'
    (args.output/'result.txt').write_text(message)
    print(message, end='')
    return result


if __name__ == '__main__':
    raise SystemExit(main())
