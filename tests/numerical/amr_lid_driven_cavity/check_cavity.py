#!/usr/bin/env python3
"""Check the six Stage-6-04 cavity configurations from native synchronized fields.

Exit 0: all applicable scientific gates pass; 1: complete numerical evidence
fails a gate; 2: incomplete/invalid evidence or an unqualified AMR reference.
"""
import argparse
import csv
import math
from pathlib import Path
import re
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
from amr_plotfile import (ArtifactError, require, identity, rows, read_plot,
                          composite_masks, conservative_average, totals)

DELTA = 1 / 1280
STEADY = 5e-8
GHIA_LIMIT = .065
REFERENCE = Path(__file__).resolve().parents[1] / 'lid_driven_cavity/reference-data'
OBSERVATIONS = ('step', 'time', 'comparison_begin_time', 'relative_l2', 'mass',
                'mass_relative_change', 'uz_max', 'rho_z_max', 'velocity_z_max')


def frame_steps(n):
    stride = 10 * n
    return sorted({t * stride for t in range(0, 201, 20)} |
                  {t * stride - stride // 1280 for t in (198, 199, 200)} |
                  {198 * stride, 199 * stride})


def time_matches(actual, expected, steps, dt):
    """Bound accumulated FP64 additions; distinguish neighboring physical steps."""
    unit = np.finfo(float).eps / 2
    gamma = steps * unit / (1 - steps * unit)
    tolerance = min(dt / 4, 4 * gamma * abs(expected) + 16 * unit * max(1, abs(expected)))
    return math.isfinite(actual) and abs(actual - expected) <= tolerance


def agree(actual, expected, message):
    require(math.isfinite(actual) and math.isfinite(expected) and
            abs(actual - expected) <= 4096 * np.finfo(float).eps * (1 + abs(expected)), message)


def write_csv(path, fields, records):
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(records)


class CavityRun:
    def __init__(self, directory, complete=True):
        self.directory = Path(directory)
        self.id = identity(directory)
        meta = self.id
        self.dimensions = int(meta['dimensions'])
        self.layout = meta['layout']
        self.n = int(meta['base_cells'])
        self.levels = int(meta['levels'])
        self.dt = .1 / self.n
        self.depth = 1 / 16 if self.dimensions == 2 else 1 / 32
        require(meta['schema'] == 'cavity-v1' and meta['backend'] in ('cpu', 'cuda') and
                (meta['descriptor'], self.dimensions) in (('D2Q9', 2), ('D3Q27', 3)) and
                int(meta['real_digits']) == 53 and re.fullmatch(r'[0-9a-f]{40}', meta['source_id']),
                'invalid cavity source/descriptor/backend/precision identity')
        require(meta['collision'] == 'BGK' and meta['periodic'] == '0,0,1', 'wrong cavity composition')
        require((self.layout, self.n, self.levels) in
                (('coarse', 128, 1), ('fine', 256, 1), ('amr', 128, 2)), 'wrong cavity layout')
        require(int(meta['ranks']) > 0 and int(meta['max_grid_size']) > 0, 'invalid parallel layout identity')
        require(int(meta['base_z']) == (1 if self.dimensions == 2 else self.n // 32) and
                float(meta['length_z']) == self.depth, 'wrong cavity thickness')
        for key, expected in (('reynolds', 1000), ('u0_lb', .1), ('u0_phys', 1), ('viscosity', .001),
                              ('rho_reference', 1), ('velocity_reference', .1)):
            require(math.isclose(float(meta[key]), expected, rel_tol=1e-13), f'wrong cavity {key}')
        for lev in range(self.levels):
            n = self.n * 2**lev
            require(math.isclose(float(meta[f'dt_{lev}']), .1 / n, rel_tol=1e-13) and
                    math.isclose(float(meta[f'tau_{lev}']), .5 + .0003 * n, rel_tol=1e-13),
                    'wrong cavity level scaling')
        self.full_steps = 2000 * self.n
        self.requested_steps = int(meta['requested_steps'])
        require(int(meta['full_steps']) == self.full_steps and 0 <= self.requested_steps <= self.full_steps and
                int(meta['comparison_steps']) == (2 if self.layout == 'fine' else 1), 'wrong horizon/comparison interval')
        if complete:
            require(self.requested_steps == self.full_steps, 'short run is not complete scientific evidence')
            exits = (self.directory / 'exit.txt').read_text().splitlines()
            require(len(exits) == 2 and set(exits) == {'solver_exit=0', 'tee_exit=0'},
                    'solver/tee did not finish successfully')
        self.initial_mass = float(meta['initial_mass'])
        require(math.isfinite(self.initial_mass) and self.initial_mass > 0, 'invalid initial mass')
        self.boxes = [[] for _ in range(self.levels)]
        for row in rows(self.directory / 'layout.csv'):
            lev, index, rank = (int(row[key]) for key in ('level', 'box', 'rank'))
            require(0 <= lev < self.levels and index == len(self.boxes[lev]) and
                    0 <= rank < int(meta['ranks']), 'invalid layout level/Box/rank')
            lo = tuple(int(row[key]) for key in ('ilo', 'jlo', 'klo'))
            hi = tuple(int(row[key]) for key in ('ihi', 'jhi', 'khi'))
            self.boxes[lev].append((lo, hi))
        require(all(self.boxes), 'missing native level layout')
        self.samples = {}
        for row in rows(self.directory / 'samples.csv'):
            step, lev = int(row['coarsest_step']), int(row['level'])
            key = step, lev
            require(key not in self.samples and 0 <= lev < self.levels and
                    0 <= step <= self.requested_steps and row['kind'] == 'sync' and row['phase'] == 'Sync' and
                    int(row['step']) == step * 2**lev and row['path'] == f'sync-{step}' and
                    time_matches(float(row['time']), step * self.dt, step, self.dt), 'invalid cavity sample index')
            self.samples[key] = row
        expected = {(step, lev) for step in frame_steps(self.n) for lev in range(self.levels)}
        if complete:
            require(self.samples.keys() == expected, 'missing/extra cavity frames or levels')
        else:
            require(all((step, lev) in self.samples for step, _ in self.samples for lev in range(self.levels)),
                    'incomplete synchronized frame')

    def plot(self, step):
        row = self.samples[step, 0]
        plot = read_plot(self.directory / row['path'])
        require(len(plot.levels) == self.levels and plot.low == (0, 0, 0) and
                plot.high == (1, 1, self.depth) and time_matches(plot.time, step * self.dt, step, self.dt),
                'wrong native cavity domain/time')
        agree(plot.time, float(row['time']), 'native/sample index time differs')
        for lev, level in enumerate(plot.levels):
            n = self.n * 2**lev
            nz = 1 if self.dimensions == 2 else n // 32
            require(level.valid.shape == (nz, n, n) and level.step == step * 2**lev and
                    sorted(level.boxes) == sorted(self.boxes[lev]), 'native cavity layout/step differs')
            agree(plot.time, float(self.samples[step, lev]['time']), 'level/sample time differs')
            require(np.array_equal(level.valid, np.broadcast_to(level.valid[0], level.valid.shape)),
                    'cavity valid coverage must extend through z')
            if lev == 0:
                require(level.valid.all(), 'incomplete L0 coverage')
            else:
                require(level.valid[:, n//4:3*n//4, n//4:3*n//4].all() and
                        not (level.valid[:, 0, :].any() or level.valid[:, -1, :].any() or
                             level.valid[:, :, 0].any() or level.valid[:, :, -1].any()),
                        'actual refinement misses the center request or reaches a physical wall')
        masks = composite_masks(plot)
        totals(plot, masks)  # Validates complete, nonduplicated physical volume.
        return plot

    def observations(self):
        data = rows(self.directory / 'observations.csv')
        require(len(data) == 200, 'expected 200 scalar observations')
        result = {}
        for t, record in enumerate(data, 1):
            row = {key: float(record[key]) for key in OBSERVATIONS}
            step = t * 10 * self.n
            require(all(math.isfinite(value) for value in row.values()) and row['step'] == step and
                    time_matches(row['time'], t, step, self.dt) and
                    time_matches(row['comparison_begin_time'], t - DELTA, step, self.dt) and
                    abs(row['time'] - row['comparison_begin_time'] - DELTA) <=
                    16 * np.finfo(float).eps * max(1, t), 'invalid scalar observation time/interval')
            require(row['mass'] > 0 and all(row[key] >= 0 for key in
                    ('relative_l2', 'mass_relative_change', 'uz_max', 'rho_z_max', 'velocity_z_max')),
                    'invalid scalar norm/mass')
            agree(row['mass_relative_change'], abs(row['mass'] - self.initial_mass) / self.initial_mass,
                  'reported mass drift disagrees with mass')
            result[t] = row
        return result


def statistics(plot, masks):
    mass, _ = totals(plot, masks)
    uz = rho_z = velocity_z = 0.
    for level, mask in zip(plot.levels, masks):
        if not mask.any():
            continue
        require(np.array_equal(mask, np.broadcast_to(mask[0], mask.shape)), 'partial z column in cavity composite')
        columns = level.values[:, :, mask[0]]
        difference = columns - columns.mean(axis=1, keepdims=True)
        uz = max(uz, float(np.max(np.abs(columns[3]))))
        rho_z = max(rho_z, float(np.max(np.abs(difference[0]))))
        velocity_z = max(velocity_z, float(np.max(np.sqrt(np.sum(difference[1:]**2, axis=0)))))
    return dict(mass=mass, uz_max=uz, rho_z_max=rho_z, velocity_z_max=velocity_z)


def relative_l2(previous, current, masks):
    require(len(previous.levels) == len(current.levels) == len(masks), 'residual lost a level')
    numerator = denominator = 0.
    for old, new, mask in zip(previous.levels, current.levels, masks):
        require(old.spacing == new.spacing and np.array_equal(old.valid, new.valid), 'residual layout changed')
        velocity = new.values[1:, mask]
        difference = velocity - old.values[1:, mask]
        volume = math.prod(new.spacing)
        numerator += float(np.sum(difference**2)) * volume
        denominator += float(np.sum(velocity**2)) * volume
    return math.sqrt(numerator / denominator) if denominator else (0. if numerator == 0 else math.inf)


def centerlines(plot, masks):
    result = {'ux': [], 'uy': []}
    for lev, (level, mask) in enumerate(zip(plot.levels, masks)):
        mean = level.values.mean(axis=1)
        for field, axis, component in (('ux', 0, 1), ('uy', 1, 2)):
            coordinate = (.5 - plot.low[axis]) / level.spacing[axis] - .5
            first = math.floor(coordinate)
            weight = coordinate - first
            if axis == 0:
                selected = mask[0, :, first] & mask[0, :, first + 1]
                values = (1 - weight) * mean[component, :, first] + weight * mean[component, :, first + 1]
            else:
                selected = mask[0, first, :] & mask[0, first + 1, :]
                values = (1 - weight) * mean[component, first, :] + weight * mean[component, first + 1, :]
            for index in np.flatnonzero(selected):
                position = plot.low[1-axis] + (index + .5) * level.spacing[1-axis]
                result[field].append((float(position), float(values[index]), lev))
    for field, values in result.items():
        values.extend(((0., 0., -1), (1., 1. if field == 'ux' else 0., -1)))
        values.sort()
        require(all(math.isfinite(v) for _, v, _ in values) and
                all(a[0] < b[0] for a, b in zip(values, values[1:])), 'invalid composite centerline')
    return result


def ghia_comparison(lines):
    comparisons = []
    for field, filename in (('ux', 'Re1000_x0.5_Ux_Ghia.csv'), ('uy', 'Re1000_y0.5_Uy_Ghia.csv')):
        reference = np.loadtxt(REFERENCE / filename, delimiter=',', skiprows=1)
        require(reference.shape == (17, 2) and np.isfinite(reference).all(), 'invalid tests-owned Ghia reference')
        computed = np.asarray(lines[field])
        for coordinate, expected in reference:
            actual = float(np.interp(coordinate, computed[:, 0], computed[:, 1]))
            error = abs(actual - expected)
            metric = error / abs(expected) if abs(expected) > 1e-12 else error
            comparisons.append(dict(component=field, coordinate=coordinate, actual=actual,
                                    reference=expected, metric=metric, passed=int(metric <= GHIA_LIMIT)))
    return comparisons


def interfaces(plot, masks):
    """Exact face adjacency on both native lattices; never same-level Box seams."""
    selected = [np.zeros_like(mask) for mask in masks]
    faces = []
    if len(masks) == 1:
        return selected, faces
    require(len(masks) == 2, 'cavity interface definition expects two levels')
    covered = plot.levels[0].valid & ~masks[0]
    for axis in (1, 2):  # array y/x; z is a complete periodic extrusion
        left = [slice(None)] * 3; right = left.copy()
        left[axis] = slice(None, -1); right[axis] = slice(1, None)
        left, right = tuple(left), tuple(right)
        change = covered[left] != covered[right]
        selected[0][left] |= change & ~covered[left]
        selected[0][right] |= change & ~covered[right]
        fine = plot.levels[1].valid
        change = fine[left] != fine[right]
        selected[1][left] |= change & fine[left]
        selected[1][right] |= change & fine[right]
    footprint = covered[0]
    h = plot.levels[0].spacing
    for direction in (0, 1):
        axis = 1 - direction
        left = [slice(None)] * 2; right = left.copy()
        left[axis] = slice(None, -1); right[axis] = slice(1, None)
        for j, i in zip(*np.nonzero(footprint[tuple(left)] != footprint[tuple(right)])):
            lo = np.array([i * h[0], j * h[1], 0.])
            hi = lo + np.array([h[0], h[1], plot.high[2]])
            lo[direction] += h[direction]
            hi[direction] = lo[direction]
            faces.append((direction, lo, hi))
    return selected, faces


def field_comparison(plot, reference, masks):
    require(len(reference.levels) == 1 and reference.levels[0].valid.all() and
            plot.low == reference.low and plot.high == reference.high, 'invalid uniform reference domain')
    boundary, _ = interfaces(plot, masks)
    error2 = volume = peak = 0.
    interface_cells = 0
    for level, mask, near in zip(plot.levels, masks, boundary):
        mapped = conservative_average(reference.levels[0].values, mask.shape)
        error = np.sqrt(np.sum((level.values[1:] - mapped[1:])**2, axis=0))
        dv = math.prod(level.spacing)
        error2 += float(np.sum(error[mask]**2)) * dv
        volume += int(mask.sum()) * dv
        active = near & mask
        if active.any():
            peak = max(peak, float(np.max(error[active])))
            interface_cells += int(active.sum())
    require(volume > 0 and (len(masks) == 1 or interface_cells > 0), 'missing comparison volume/interface')
    return math.sqrt(error2 / volume), peak, interface_cells


def analyze(run, output):
    output.mkdir(parents=True, exist_ok=True)
    observations = run.observations()
    frame_records = []
    residuals = {}
    previous = None
    for step in frame_steps(run.n):
        plot = run.plot(step)
        masks = composite_masks(plot)
        stats = statistics(plot, masks)
        stats['mass_relative_change'] = abs(stats['mass'] - run.initial_mass) / run.initial_mass
        nominal = step * run.dt
        if step == 0:
            agree(stats['mass'], run.initial_mass, 'initial native/identity mass differs')
            agree(stats['mass'], run.depth, 'initial mass is not rho0 times physical volume')
            for level, mask in zip(plot.levels, masks):
                require(np.max(np.abs(level.values[0][mask] - 1)) <= 1e-12 and
                        np.max(np.abs(level.values[1:, mask])) <= 1e-12, 'initial state is not constant rest')
        if step % (10 * run.n) == 0 and step:
            row = observations[step // (10 * run.n)]
            for key, value in stats.items():
                agree(value, row[key], f'native/scalar {key} differs at t={nominal:g}')
        if step in {t * 10 * run.n for t in (198, 199, 200)}:
            require(previous is not None and previous.levels[0].step == step - int(run.id['comparison_steps']),
                    'missing previous coarse-interval snapshot')
            residual = relative_l2(previous, plot, masks)
            t = step // (10 * run.n)
            agree(residual, observations[t]['relative_l2'], f'native/scalar residual differs at t={t}')
            residuals[t] = residual
        frame_records.append(dict(step=step, time=plot.time, nominal_time=nominal, **stats))
        previous = plot
    end = previous
    lines = centerlines(end, composite_masks(end))
    for field, coordinate, value in (('ux', 'y_over_L', 'ux_over_Ulid'), ('uy', 'x_over_L', 'uy_over_Ulid')):
        write_csv(output / f'centerline_{field}.csv', (coordinate, value, 'level', 'wall_endpoint'),
                  [{coordinate: p, value: v, 'level': lev, 'wall_endpoint': int(lev < 0)} for p, v, lev in lines[field]])
    ghia = ghia_comparison(lines)
    write_csv(output / 'ghia.csv', ('component', 'coordinate', 'actual', 'reference', 'metric', 'passed'), ghia)
    write_csv(output / 'frames.csv', tuple(frame_records[0]), frame_records)
    gates = [dict(metric=f'relative_l2_t{t}', value=residuals[t], limit=STEADY,
                  passed=int(residuals[t] < STEADY), gate=1) for t in (198, 199, 200)]
    for field in ('ux', 'uy'):
        value = max(row['metric'] for row in ghia if row['component'] == field)
        gates.append(dict(metric=f'ghia_{field}', value=value, limit=GHIA_LIMIT, passed=int(value <= GHIA_LIMIT), gate=1))
    for key, limit in [('mass_relative_change', 1e-8)] + (
            [(key, 1e-10) for key in ('uz_max', 'rho_z_max', 'velocity_z_max')] if run.dimensions == 3 else []):
        value = max(row[key] for row in observations.values())
        gates.append(dict(metric=key, value=value, limit=limit, passed=int(value <= limit), gate=1))
    write_csv(output / 'gates.csv', ('metric', 'value', 'limit', 'passed', 'gate'), gates)
    (output / 'identity.txt').write_text(''.join(f'{key}={value}\n' for key, value in run.id.items()), encoding='utf-8')
    return dict(end=end, lines=lines, gates=gates, steady=all(residuals[t] < STEADY for t in (198, 199, 200)),
                failed=[row['metric'] for row in gates if not row['passed']])


def check(run_directory, reference_directory, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    local = None
    invalid = None
    reference_status = 'not-requested'
    try:
        run = CavityRun(run_directory)
        local = analyze(run, output)
        if run.layout == 'amr':
            require(reference_directory is not None, 'AMR requires a matching uniform-fine reference')
        if reference_directory is not None:
            reference = CavityRun(reference_directory)
            require(run.layout != 'fine' and reference.layout == 'fine' and
                    all(run.id[key] == reference.id[key] for key in ('descriptor', 'source_id', 'backend')),
                    'reference identity/role differs')
            fine = analyze(reference, output / 'reference')
            reference_status = str(int(bool(fine['failed'])))
            rms, peak, count = field_comparison(local['end'], fine['end'], composite_masks(local['end']))
            qualified = fine['steady']
            comparisons = [dict(metric='velocity_volume_rms', value=rms, limit=.01, passed=int(rms <= .01),
                                gate=int(run.layout == 'amr' and qualified)),
                           dict(metric='velocity_interface_peak', value=peak, limit=.03, passed=int(peak <= .03),
                                gate=int(run.layout == 'amr' and qualified))]
            for field in ('ux', 'uy'):
                a, b = np.asarray(local['lines'][field]), np.asarray(fine['lines'][field])
                value = float(np.max(np.abs(a[:, 1] - np.interp(a[:, 0], b[:, 0], b[:, 1]))))
                comparisons.append(dict(metric=f'centerline_{field}_peak', value=value, limit='', passed='', gate=0))
            write_csv(output / 'comparison.csv', ('metric', 'value', 'limit', 'passed', 'gate'), comparisons)
            (output / 'reference-status.txt').write_text(
                f'steady_qualified={int(qualified)}\nreference_exit={reference_status}\ninterface_cells={count}\n', encoding='utf-8')
            local['failed'].extend(row['metric'] for row in comparisons if row['gate'] and not row['passed'])
            if run.layout == 'amr':
                require(qualified, 'uniform-fine reference has not passed its last three steady checks')
    except (ArtifactError, OSError, ValueError, KeyError, IndexError, csv.Error) as error:
        invalid = str(error)
    status = 2 if invalid is not None else int(bool(local['failed']))
    text = f'exit={status}\nlocal_exit={int(bool(local["failed"])) if local else 2}\nreference_exit={reference_status}\n'
    if invalid is not None:
        text += f'invalid={invalid}\n'
    if local:
        text += ''.join(f'failed={name}\n' for name in local['failed'])
    (output / 'result.txt').write_text(text, encoding='utf-8')
    print(f'cavity checker: exit={status}' + (f' invalid={invalid}' if invalid else ''))
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    return check(args.run, args.reference, args.output)


if __name__ == '__main__':
    raise SystemExit(main())
