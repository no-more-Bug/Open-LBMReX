#!/usr/bin/env python3
"""C8 D2Q9/YMS layout observation: 0 means complete data, 2 means invalid data.

No similarity or steady-state acceptance threshold is applied.
"""
import argparse
import csv
from pathlib import Path
import re
import numpy as np

from check_tandem import (DT, boundary_map, close, read_forces, validate_layout, vector, time_matches,
                          ArtifactError, require, identity, rows, read_plot, composite_masks)

STEPS = 20000
SNAPSHOTS = (0, 100, 500, 1000, 2000, 4000, 8000, 12000, 16000, 20000)


def coefficients(boundaries, forces, steps, meta):
    """Average native Boundary samples over each L0 interval; forces already share units."""
    # D2 force is per unit span. Storage thickness is not part of its reference area.
    diameter_lb = .1/vector(meta['dx_2'])[0]
    denominator = .5*float(meta['rho_reference'])*float(meta['velocity_reference'])**2*diameter_lb
    result = {}
    for bid, boundary in boundaries.items():
        level = boundary['level']
        count = 2**level
        native = np.array([forces['boundary', bid, level, n]
                           for n in range(1, steps*count+1)])
        result[boundary['name']] = native.reshape(steps, count, 3).mean(axis=1)[:, :2]/denominator
    return result


def sample_index(records):
    expected = {(step, level) for step in SNAPSHOTS for level in range(3)}
    seen = set()
    for row in records:
        step, level = int(row['coarsest_step']), int(row['level'])
        require((step, level) in expected and (step, level) not in seen, 'unexpected/duplicate snapshot')
        require(row['kind'] == 'sync' and row['phase'] == 'Sync' and row['path'] == f'sync-{step}' and
                int(row['step']) == step*2**level and time_matches(float(row['time']), step*DT, step*2**level, DT/2**level),
                'wrong snapshot step/time/path')
        seen.add((step, level))
    require(seen == expected, 'missing snapshot')


class ComparisonRun:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.id = identity(self.directory)
        meta = self.id
        fixed = dict(schema='tandem-compare-v1', descriptor='D2Q9', dimensions='2', scheme='YMS',
                     grouping='shared', collision='BGK', periodic='0,0,1',
                     initialization='uniform-equilibrium-then-Guo', force_units='finest-lattice',
                     full_steps=str(STEPS), levels='3', real_digits='53', omp_threads='1', max_grid_size='64')
        require(all(meta[k] == v for k, v in fixed.items()), 'wrong long-run composition/budget')
        require(meta['layout'] in ('same', 'cross') and meta['backend'] in ('cpu', 'cuda') and
                re.fullmatch('[0-9a-f]{40}', meta['source_id']) and int(meta['ranks']) > 0,
                'wrong source/layout/parallel identity')
        for key, value in (('reynolds', 20), ('u0_lb', .035), ('u0_phys', .2), ('viscosity', .001),
                           ('rho_reference', 1), ('velocity_reference', .035), ('comparison_dt', DT)):
            require(close(float(meta[key]), value), f'wrong {key}')
        require(close(vector(meta['physical_lo']), (0, 0, 0)) and
                close(vector(meta['physical_hi']), (2.2, .41, .01)), 'wrong physical channel')
        for lev in range(3):
            ratio = 2**lev
            require(vector(meta[f'cells_{lev}'], int) == (441*ratio, 83*ratio, 1) and
                    close(vector(meta[f'dx_{lev}']), (.005/ratio, .005/ratio, .01)) and
                    close(float(meta[f'dt_{lev}']), DT/ratio) and
                    close(float(meta[f'tau_{lev}']), .5+.105*ratio), 'wrong level geometry/clock/relaxation')
        exits = dict(line.split('=', 1) for line in (self.directory/'exit.txt').read_text().splitlines())
        require(exits == {'solver_exit': '0', 'tee_exit': '0'}, 'solver/output did not complete')
        self.boundaries = boundary_map(rows(self.directory/'boundaries.csv'), meta['layout'], 'shared')
        forces = read_forces(rows(self.directory/'forces.csv'), self.boundaries, STEPS)
        self.force_rows = len(forces)
        self.coefficients = coefficients(self.boundaries, forces, STEPS, meta)
        sample_index(rows(self.directory/'samples.csv'))
        self.layout = rows(self.directory/'layout.csv')

    def plot(self, step):
        plot = read_plot(self.directory/f'sync-{step}')
        require(len(plot.levels) == 3 and time_matches(plot.time, step*DT, step*4, DT/4) and
                close(plot.low, (-.0025, -.0025, 0)) and close(plot.high, (2.2025, .4125, .01)),
                'wrong native snapshot clock/envelope')
        for lev, data in enumerate(plot.levels):
            require(data.step == step*2**lev and data.values.shape[1:] == (1, 83*2**lev, 441*2**lev) and
                    close(data.spacing, vector(self.id[f'dx_{lev}'])), 'wrong native level geometry/step')
        require(plot.levels[0].valid.all(), 'incomplete L0 field')
        validate_layout(self.layout, plot, int(self.id['ranks']))
        return plot


def pair(same, cross):
    require(same.id['layout'] == 'same' and cross.id['layout'] == 'cross', 'expected same/cross pair')
    require({k: v for k, v in same.id.items() if k != 'layout'} ==
            {k: v for k, v in cross.id.items() if k != 'layout'}, 'identities differ beyond layout')
    # The shared L1 request does not imply identical final Box partitions.
    # Each run is checked against its own native Boxes in plot().


def fluid_masks(plot):
    masks = composite_masks(plot)
    for data, mask in zip(plot.levels, masks):
        x = plot.low[0]+(np.arange(data.values.shape[3])+.5)*data.spacing[0]
        y = plot.low[1]+(np.arange(data.values.shape[2])+.5)*data.spacing[1]
        for center in (.2, .4):
            mask &= ((x[None, :] - center)**2+(y[:, None]-.2)**2 >= .05**2)[None, :, :]
        mask &= ((x >= 0) & (x <= 2.2))[None, None, :]
        mask &= ((y >= 0) & (y <= .41))[None, :, None]
    return masks


def write_csv(path, fields, records):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(records)


def analyze(same, cross, output):
    pair(same, cross)
    for run in (same, cross):
        for step in SNAPSHOTS:
            run.plot(step)
        def records():
            for name, values in run.coefficients.items():
                for n, (cd, cl) in enumerate(values, 1):
                    yield dict(cylinder=name, coarsest_step=n, begin_time=(n-1)*DT,
                               end_time=n*DT, midpoint_time=(n-.5)*DT, Cd=cd, Cl=cl)
        write_csv(output/f"{run.id['layout']}-coefficients.csv",
                  ('cylinder', 'coarsest_step', 'begin_time', 'end_time', 'midpoint_time', 'Cd', 'Cl'), records())
    statistics = []
    for name in ('cylinder_a', 'cylinder_b'):
        for component, index in (('Cd', 0), ('Cl', 1)):
            a, b = same.coefficients[name][:, index], cross.coefficients[name][:, index]
            for layout, values in (('same', a), ('cross', b), ('cross-minus-same', b-a)):
                statistics.append(dict(cylinder=name, component=component, layout=layout,
                    terminal=values[-1], last_2000_mean=np.mean(values[-2000:]),
                    last_2000_min=np.min(values[-2000:]), last_2000_max=np.max(values[-2000:]),
                    last_1000_mean_change=np.mean(values[-1000:])-np.mean(values[-2000:-1000])))
    write_csv(output/'statistics.csv', tuple(statistics[0]), statistics)
    (output/'pair.txt').write_text(f'same={same.directory.resolve()}\ncross={cross.directory.resolve()}\n'
                                   f"source_id={same.id['source_id']}\nbackend={same.id['backend']}\n"
                                   f'full_steps={STEPS}\nphysical_end={STEPS*DT}\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--same', type=Path, required=True)
    parser.add_argument('--cross', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        same, cross = ComparisonRun(args.same), ComparisonRun(args.cross)
        analyze(same, cross, args.output)
        message = (f'exit=0\nassessment=observation-only\n'
                   f'force_rows_same={same.force_rows}\nforce_rows_cross={cross.force_rows}\n'
                   f'snapshots_per_run={len(SNAPSHOTS)}\nintervals_per_cylinder={STEPS}\n')
        result = 0
    except (ArtifactError, OSError, ValueError, KeyError, OverflowError) as error:
        result, message = 2, f'exit=2\nerror={error}\n'
    (args.output/'result.txt').write_text(message)
    print(message, end='')
    return result


if __name__ == '__main__':
    raise SystemExit(main())
