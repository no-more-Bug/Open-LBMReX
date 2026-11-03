#!/usr/bin/env python3
"""Native-field Re20 verification: 0 pass, 1 scientific failure, 2 invalid evidence.

Guo face nodes and the independent fine lattice have different envelopes.
Profiles compare physical point values; pressure uses the prescribed fluid-side
extrapolation. Raw force records retain their own intervals and lattice units.
"""
import argparse
import csv
import math
from pathlib import Path
import re
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
from amr_plotfile import ArtifactError, require, identity, rows, read_plot, composite_masks, totals

DELTA = .000875
OBSERVATION_COUNT = 800
FINAL_CHECKS = range(OBSERVATION_COUNT-2, OBSERVATION_COUNT+1)
ULB = .035
EPS = np.finfo(float).eps
HEIGHT = .41
DEPTH = .01
REFERENCE = Path(__file__).resolve().parents[1] / 'channel_cylinder/reference-data/dfg_re20_reference.csv'
OBSERVATIONS = ('step', 'common_step', 'time', 'comparison_begin_time', 'relative_l2',
                'force_begin_time', 'force_end_time', 'Cd', 'Cl', 'rho_front', 'rho_rear',
                'Cdp', 'phi_in', 'phi_out', 'flux_imbalance', 'uz_max', 'rho_z_max', 'velocity_z_max')
FORCES = ('kind', 'id', 'level', 'population_step', 'coarsest_step', 'begin_time', 'end_time',
          'sample_time', 'Fx_lb', 'Fy_lb', 'Fz_lb', 'Cd', 'Cl', 'Cz')
PROFILE_FIELDS = ('line', 'coordinate', 'x', 'y', 'ux_norm', 'uy_norm')
LINES = {'x0.05': (0, .05, 0., .41, 329), 'x0.30': (0, .30, 0., .41, 329),
         'x0.60': (0, .60, 0., .41, 329), 'x1.50': (0, 1.50, 0., .41, 329),
         'y0.20': (1, .20, .30, 2.10, 1441)}


def frame_steps(factor):
    common = set(range(0, 100*OBSERVATION_COUNT+1, 10*OBSERVATION_COUNT)) | {100, 1000}
    common.update(100*n+offset for n in FINAL_CHECKS for offset in (-1, 0))
    return [m * factor for m in sorted(common)]


def time_matches(actual, expected, steps, dt):
    unit = EPS / 2
    gamma = steps * unit / (1 - steps * unit)
    tolerance = min(dt / 4, 4 * gamma * abs(expected) + 16 * unit * max(1, abs(expected)))
    return math.isfinite(actual) and abs(actual - expected) <= tolerance


def agree(actual, expected, message):
    require(math.isfinite(actual) and math.isfinite(expected) and
            abs(actual - expected) <= 4096 * EPS * (1 + abs(expected)), message)


def write_csv(path, fields, records):
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fields); writer.writeheader(); writer.writerows(records)


def vector(value, cast=float):
    return tuple(cast(item) for item in value.split(','))


class Re20Run:
    def __init__(self, directory, complete=True):
        self.directory = Path(directory)
        self.id = identity(directory)
        meta = self.id
        self.dimensions = int(meta['dimensions'])
        self.layout = meta['layout']
        self.levels = int(meta['levels'])
        self.factor = 4 if self.layout == 'fine' else 1
        self.h = .00125 if self.layout == 'fine' else .005
        self.dt = DELTA / self.factor
        self.owner_level = 2 if self.layout == 'amr' else 0
        self.full_steps = 100 * OBSERVATION_COUNT * self.factor
        self.requested_steps = int(meta['requested_steps'])
        require(meta['schema'] == 're20-v1' and meta['backend'] in ('cpu', 'cuda') and
                (meta['descriptor'], self.dimensions) in (('D2Q9', 2), ('D3Q27', 3)) and
                re.fullmatch(r'[0-9a-f]{40}', meta['source_id']) and int(meta['real_digits']) == 53,
                'invalid Re20 source/backend/descriptor/precision identity')
        require(meta['scheme'] in ('YMS', 'Bouzidi') and meta['collision'] == 'BGK' and
                meta['periodic'] == '0,0,1' and (self.layout, self.levels) in
                (('coarse', 1), ('fine', 1), ('amr', 3)), 'wrong Re20 composition')
        require(int(meta['ranks']) > 0 and int(meta['max_grid_size']) >= 2 and
                int(meta['max_grid_size']) % 2 == 0, 'invalid rank/Box identity')
        self.low = (-self.h / 2, -self.h / 2, 0.)
        self.high = (2.2 + self.h / 2, HEIGHT + self.h / 2, DEPTH)
        for key, expected in (('physical_lo', (0., 0., 0.)), ('physical_hi', (2.2, HEIGHT, DEPTH)),
                              ('envelope_lo', self.low), ('envelope_hi', self.high),
                              ('cylinder_center', (.2, .2))):
            actual = vector(meta[key]); require(len(actual) == len(expected), f'wrong {key}')
            for a, b in zip(actual, expected): agree(a, b, f'wrong {key}')
        for key, expected in (('reynolds', 20), ('u0_lb', ULB), ('u0_phys', .2), ('viscosity', .001),
                              ('rho_reference', 1), ('velocity_reference', ULB), ('cylinder_radius', .05),
                              ('comparison_dt', DELTA)):
            agree(float(meta[key]), expected, f'wrong Re20 {key}')
        self.cells = []
        for lev in range(self.levels):
            ratio = 2**lev
            h = self.h / ratio
            cells = ((round(2.2/self.h)+1)*ratio, (round(HEIGHT/self.h)+1)*ratio,
                     1 if self.dimensions == 2 else round(DEPTH/h))
            dx = (h, h, DEPTH if self.dimensions == 2 else h)
            require(vector(meta[f'cells_{lev}'], int) == cells, 'wrong native lattice shape')
            actual = vector(meta[f'dx_{lev}']); require(len(actual) == 3, 'wrong native spacing')
            for a, b in zip(actual, dx): agree(a, b, 'wrong native spacing')
            agree(float(meta[f'dt_{lev}']), self.dt/ratio, 'wrong level physical dt')
            agree(float(meta[f'tau_{lev}']), .5 + 3*.001*(self.dt/ratio)/h**2, 'wrong BGK tau')
            self.cells.append(cells)
        self.finest_h = self.h / 2**(self.levels-1)
        self.denominator = .5 * ULB**2 * .1/self.finest_h
        if self.dimensions == 3: self.denominator *= DEPTH/self.finest_h
        agree(float(meta['diameter_lb']), .1/self.finest_h, 'wrong force diameter scale')
        agree(float(meta['span_lb']), DEPTH/self.finest_h, 'wrong force span scale')
        agree(float(meta['force_denominator']), self.denominator, 'wrong force normalization')
        require(int(meta['owner_level']) == self.owner_level and int(meta['boundary_id']) >= 0 and
                int(meta['object_id']) >= 0 and int(meta['full_steps']) == self.full_steps and
                int(meta['comparison_steps']) == self.factor and 0 <= self.requested_steps <= self.full_steps,
                'wrong owner/horizon/comparison identity')
        if complete:
            require(self.requested_steps == self.full_steps, 'short run is not complete scientific evidence')
            exits = (self.directory/'exit.txt').read_text().splitlines()
            require(len(exits) == 2 and set(exits) == {'solver_exit=0', 'tee_exit=0'},
                    'solver/tee did not finish successfully')
        self.boxes = [[] for _ in range(self.levels)]
        for row in rows(self.directory/'layout.csv'):
            lev, index, rank = (int(row[key]) for key in ('level', 'box', 'rank'))
            require(0 <= lev < self.levels and index == len(self.boxes[lev]) and
                    0 <= rank < int(meta['ranks']), 'invalid layout level/Box/rank')
            self.boxes[lev].append((tuple(int(row[k]) for k in ('ilo', 'jlo', 'klo')),
                                    tuple(int(row[k]) for k in ('ihi', 'jhi', 'khi'))))
        require(all(self.boxes), 'missing level layout')
        self.samples = {}
        for row in rows(self.directory/'samples.csv'):
            step, lev = int(row['coarsest_step']), int(row['level'])
            require((step, lev) not in self.samples and 0 <= lev < self.levels and
                    0 <= step <= self.requested_steps and row['kind'] == 'sync' and row['phase'] == 'Sync' and
                    int(row['step']) == step*2**lev and row['path'] == f'sync-{step}' and
                    time_matches(float(row['time']), step*self.dt, step, self.dt), 'invalid sample identity/time')
            self.samples[step, lev] = row
        if complete:
            require(self.samples.keys() == {(step, lev) for step in frame_steps(self.factor)
                    for lev in range(self.levels)}, 'missing/extra synchronized frames')
        else:
            require(all((step, lev) in self.samples for step, _ in self.samples for lev in range(self.levels)),
                    'incomplete synchronized frame')

    def plot(self, step):
        plot = read_plot(self.directory/self.samples[step, 0]['path'])
        require(len(plot.levels) == self.levels and
                time_matches(plot.time, step*self.dt, step, self.dt), 'wrong native level count/time')
        for a, b in zip((*plot.low, *plot.high), (*self.low, *self.high)): agree(a, b, 'wrong native envelope')
        agree(plot.time, float(self.samples[step, 0]['time']), 'native/index time differs')
        for lev, data in enumerate(plot.levels):
            require(data.valid.shape == tuple(reversed(self.cells[lev])) and data.step == step*2**lev and
                    sorted(data.boxes) == sorted(self.boxes[lev]), 'wrong native layout/step')
            agree(plot.time, float(self.samples[step, lev]['time']), 'level/index time differs')
            require(np.array_equal(data.valid, np.broadcast_to(data.valid[0], data.valid.shape)),
                    'Re20 layout does not span periodic z')
            if lev == 0:
                require(data.valid.all(), 'incomplete L0')
            else:
                bounds = ((.0475, 1.2025, .0475, .3525), (.1, 1., .1, .3))[lev-1]
                x, y = coordinates(plot, lev)
                requested = ((y[:, None] > bounds[2]) & (y[:, None] < bounds[3]) &
                             (x[None, :] > bounds[0]) & (x[None, :] < bounds[1]))
                require(data.valid[:, requested].all() and not
                        (data.valid[:, 0, :].any() or data.valid[:, -1, :].any() or
                         data.valid[:, :, 0].any() or data.valid[:, :, -1].any()),
                        'refinement misses its request or reaches a Guo boundary')
        totals(plot, composite_masks(plot))  # Completeness uses storage volume, not cut-cell fluid volume.
        return plot

    def observations(self):
        records = rows(self.directory/'observations.csv')
        require(len(records) == OBSERVATION_COUNT+1, f'expected initialization and {OBSERVATION_COUNT} observations')
        result = []
        interval = ('comparison_begin_time', 'relative_l2', 'force_begin_time', 'force_end_time', 'Cd', 'Cl')
        zfields = ('uz_max', 'rho_z_max', 'velocity_z_max')
        for index, record in enumerate(records):
            absent = set(interval if index == 0 else ()) | set(zfields if self.dimensions == 2 else ())
            require(all(record[key] == '' for key in absent), 'inapplicable observation must be empty')
            row = {key: float(record[key]) for key in OBSERVATIONS if key not in absent}
            step = index*100*self.factor
            require(all(math.isfinite(v) for v in row.values()) and row['step'] == step and
                    row['common_step'] == index*100 and time_matches(row['time'], step*self.dt, step, self.dt),
                    'invalid observation identity/time/value')
            if index:
                for key in ('comparison_begin_time', 'force_begin_time'):
                    require(time_matches(row[key], index*100*DELTA-DELTA, step, self.dt), 'wrong comparison interval')
                    agree(row['time']-row[key], DELTA, 'wrong comparison duration')
                agree(row['force_end_time'], row['time'], 'wrong force interval endpoint')
                require(row['relative_l2'] >= 0, 'negative residual')
            require(row['rho_front'] > 0 and row['rho_rear'] > 0 and row['flux_imbalance'] >= 0,
                    'invalid pressure/flow observation')
            agree(row['Cdp'], (row['rho_front']-row['rho_rear'])/(3*ULB**2), 'wrong pressure coefficient')
            agree(row['flux_imbalance'], abs(row['phi_out']-row['phi_in']), 'wrong flux imbalance')
            if self.dimensions == 3: require(all(row[k] >= 0 for k in zfields), 'negative z deviation')
            result.append(row)
        return result


def coordinates(plot, lev):
    level = plot.levels[lev]
    return tuple(plot.low[d] + (np.arange(level.valid.shape[2-d])+.5)*level.spacing[d] for d in (0, 1))


def fluid_masks(plot):
    masks = composite_masks(plot)
    band = 64*np.finfo(float).eps*max(.2, .05, *(abs(v) for v in (*plot.low, *plot.high)))
    for lev, mask in enumerate(masks):
        x, y = coordinates(plot, lev)
        outside = np.hypot(x[None, :]-.2, y[:, None]-.2) >= .05-band
        mask &= outside[None, :, :]
    return masks


def relative_l2(previous, current, masks):
    require(len(previous.levels) == len(current.levels) == len(masks), 'residual lost a level')
    numerator = denominator = 0.
    for old, new, mask in zip(previous.levels, current.levels, masks):
        require(old.spacing == new.spacing and np.array_equal(old.valid, new.valid), 'residual layout changed')
        velocity = new.values[1:, mask]
        difference = velocity-old.values[1:, mask]
        volume = math.prod(new.spacing)
        numerator += float(np.sum(difference**2))*volume
        denominator += float(np.sum(velocity**2))*volume
    require(denominator > 0, 'zero Re20 velocity norm')
    return math.sqrt(numerator/denominator)


def z_statistics(plot, masks):
    result = dict(uz_max=0., rho_z_max=0., velocity_z_max=0.)
    for level, mask in zip(plot.levels, masks):
        require(np.array_equal(mask, np.broadcast_to(mask[0], mask.shape)), 'partial composite z column')
        if not mask.any(): continue
        values = level.values[:, :, mask[0]]
        difference = values-values.mean(axis=1, keepdims=True)
        result['uz_max'] = max(result['uz_max'], float(np.max(np.abs(values[3]))))
        result['rho_z_max'] = max(result['rho_z_max'], float(np.max(np.abs(difference[0]))))
        result['velocity_z_max'] = max(result['velocity_z_max'], float(np.max(np.linalg.norm(difference[1:], axis=0))))
    return result


def pressure(plot):
    lev = len(plot.levels)-1
    data = plot.levels[lev]; x, y = coordinates(plot, lev)
    values = []
    for location, direction in ((.15, -1), (.25, 1)):
        if lev == 0:
            i = round((location-plot.low[0])/data.spacing[0]-.5)
            j = round((.2-plot.low[1])/data.spacing[1]-.5)
            agree(x[i], location, 'uniform pressure probe moved'); agree(y[j], .2, 'uniform pressure probe moved')
            require(data.valid[:, j, i].all(), 'missing pressure probe')
            value = data.values[0, :, j, i].mean()
        else:
            q = (location-plot.low[0])/data.spacing[0]-.5
            near = math.floor(q) if direction < 0 else math.ceil(q)
            far = near+direction
            j = math.floor((.2-plot.low[1])/data.spacing[1]-.5)
            weight = (.2-y[j])/(y[j+1]-y[j])
            samples = []
            for i in (near, far):
                require(data.valid[:, j:j+2, i].all() and
                        np.hypot(x[i]-.2, y[j:j+2]-.2).min() >= .05,
                        'pressure support is not valid fluid')
                samples.append(((1-weight)*data.values[0, :, j, i]+weight*data.values[0, :, j+1, i]).mean())
            s1, s2 = abs(x[near]-location), abs(x[far]-location)
            value = (s2*samples[0]-s1*samples[1])/(s2-s1)
        values.append(float(value))
    return dict(rho_front=values[0], rho_rear=values[1], Cdp=(values[0]-values[1])/(3*ULB**2))


def flux(plot):
    level = plot.levels[0]; x, y = coordinates(plot, 0)
    agree(x[0], 0., 'inlet is not a Guo node'); agree(x[-1], 2.2, 'outlet is not a Guo node')
    agree(y[0], 0., 'lower wall is not a Guo node'); agree(y[-1], HEIGHT, 'upper wall is not a Guo node')
    values = []
    for i in (0, -1):
        product = (level.values[0, :, :, i]*level.values[1, :, :, i]).mean(axis=0)
        values.append(float(np.sum((product[:-1]+product[1:])*.5*np.diff(y))/HEIGHT))
    return dict(phi_in=values[0], phi_out=values[1], flux_imbalance=abs(values[1]-values[0]))


def profiles(plot, masks):
    result, native = {}, {}
    for name, (axis, position, start, end, count) in LINES.items():
        samples = []
        for lev, (data, mask) in enumerate(zip(plot.levels, masks)):
            coords = coordinates(plot, lev)
            location = (position-plot.low[axis])/data.spacing[axis]-.5
            nearest = round(location)
            if abs(location-nearest) < 128*np.finfo(float).eps*max(1, abs(location)):
                first = second = nearest; weight = 0.
            else:
                first = math.floor(location); second = first+1; weight = location-first
            require(0 <= first <= second < len(coords[axis]), 'profile lies outside native domain')
            mean = data.values[1:3].mean(axis=1)
            if axis == 0:
                valid = mask[0, :, first] & mask[0, :, second]
                velocity = (1-weight)*mean[:, :, first]+weight*mean[:, :, second]
            else:
                valid = mask[0, first, :] & mask[0, second, :]
                velocity = (1-weight)*mean[:, first, :]+weight*mean[:, second, :]
            for j in np.flatnonzero(valid):
                samples.append((float(coords[1-axis][j]), *velocity[:, j], lev))
        samples.sort(); values = np.asarray(samples)
        require(len(values) >= 2 and np.isfinite(values).all() and np.all(np.diff(values[:, 0]) > 0) and
                values[0, 0] <= start+1e-12 and values[-1, 0] >= end-1e-12, 'incomplete composite profile')
        common = np.linspace(start, end, count)
        result[name] = np.column_stack((common, *(np.interp(common, values[:, 0], values[:, j]) for j in (1, 2))))
        native[name] = values
    return result, native


def profile_records(values):
    for name, data in values.items():
        axis, position, *_ = LINES[name]
        for coordinate, ux, uy in data:
            yield dict(line=name, coordinate=coordinate, x=position if axis == 0 else coordinate,
                       y=coordinate if axis == 0 else position, ux_norm=ux, uy_norm=uy)


def profile_errors(actual, reference):
    result = []
    for name, a in actual.items():
        b = reference[name]
        require(np.array_equal(a[:, 0], b[:, 0]), 'profile comparison coordinates differ')
        error = np.linalg.norm(a[:, 1:]-b[:, 1:], axis=1)
        weights = np.ones(len(error)); weights[[0, -1]] = .5
        result.extend(((name+'_rms', math.sqrt(float(np.sum(weights*error**2)/weights.sum())), .01),
                       (name+'_peak', float(error.max()), .03)))
    return result


def force_steps(run, count):
    ratio = 2**run.owner_level
    dt = run.dt/ratio
    coefficients = {kind: np.empty((count, 3)) for kind in ('boundary', 'object')}
    durations = {kind: np.empty(count) for kind in coefficients}
    ends = {kind: np.empty(count) for kind in coefficients}
    seen = dict(boundary=0, object=0)
    with (run.directory/'forces.csv').open(newline='') as stream:
        reader = csv.DictReader(stream)
        require(tuple(reader.fieldnames or ()) == FORCES, 'wrong force schema')
        for row in reader:
            require(None not in row and all(value is not None for value in row.values()), 'incomplete/extra force fields')
            kind = row['kind']; require(kind in seen, 'unknown force kind')
            step = int(row['population_step'])
            require(step == seen[kind]+1 and step <= count and int(row['level']) == run.owner_level and
                    int(row['id']) == int(run.id['boundary_id' if kind == 'boundary' else 'object_id']) and
                    int(row['coarsest_step']) == (step+ratio-1)//ratio, 'missing/repeated/wrong force identity')
            begin, end, sample = (float(row[k]) for k in ('begin_time', 'end_time', 'sample_time'))
            require(time_matches(begin, (step-1)*dt, step, dt) and time_matches(end, step*dt, step, dt),
                    'wrong native force interval')
            agree(end-begin, dt, 'wrong native force duration'); agree(sample, (begin+end)/2, 'wrong force midpoint')
            if step > 1: agree(begin,float(ends[kind][step-2]),'discontinuous native force intervals')
            durations[kind][step-1] = end-begin
            ends[kind][step-1] = end
            for d, (raw, coefficient) in enumerate(zip(('Fx_lb', 'Fy_lb', 'Fz_lb'), ('Cd', 'Cl', 'Cz'))):
                value = float(row[raw])/run.denominator
                agree(float(row[coefficient]), value, 'force coefficient has wrong scale/span')
                coefficients[kind][step-1, d] = value
            seen[kind] = step
    require(all(value == count for value in seen.values()), 'incomplete native force history')
    difference = np.abs(coefficients['boundary']-coefficients['object'])
    require(np.all(difference <= 4096*np.finfo(float).eps*(1+np.abs(coefficients['boundary']))),
            'single-boundary Object differs from Boundary force')
    return coefficients, durations, ends


def force_intervals(run, observations):
    ratio = 2**run.owner_level
    coefficients, durations, ends = force_steps(run, run.full_steps*ratio)
    interval_count = run.factor*ratio
    result = []
    for index, observation in enumerate(observations[1:], 1):
        end = index*100*interval_count
        selected = slice(end-interval_count,end)
        for kind in coefficients:
            first = end-interval_count
            agree(float(ends[kind][first]-durations[kind][first]),observation['force_begin_time'],
                  'raw/observation force beginning differs')
            agree(float(ends[kind][end-1]),observation['force_end_time'],'raw/observation force endpoint differs')
        average = np.sum(coefficients['boundary'][selected]*durations['boundary'][selected,None],axis=0)/DELTA
        object_average = np.sum(coefficients['object'][selected]*durations['object'][selected,None],axis=0)/DELTA
        agree(float(durations['boundary'][selected].sum()),DELTA,'incomplete force comparison interval')
        for a,b in zip(average,object_average): agree(float(a),float(b),'Object/Boundary interval weights differ')
        for key, value in zip(('Cd', 'Cl'), average[:2]):
            agree(observation[key], float(value), f'raw/observation {key} interval differs')
        result.append(dict(common_step=index*100, begin_time=observation['force_begin_time'],
                           end_time=observation['force_end_time'], Cd=average[0], Cl=average[1], Cz=average[2]))
    return result, max(float(np.max(np.abs(values[:, 2]))) for values in coefficients.values())


def coefficient_errors(actual, reference):
    result = []
    for key,limit in (('Cd',.01),('Cl',.002),('Cdp',.03)):
        difference = abs(actual[key]-reference[key])
        denominator = abs(reference[key]) if key != 'Cl' else 1
        value = difference/denominator if denominator else (0. if difference == 0 else math.inf)
        result.append((key,value,limit))
    return result


def analyze(run, output):
    output.mkdir(parents=True, exist_ok=True)
    observations = run.observations()
    intervals, cz = force_intervals(run, observations)
    write_csv(output/'force-intervals.csv', tuple(intervals[0]), intervals)
    frames, residuals = [], {}
    previous = None
    for step in frame_steps(run.factor):
        plot = run.plot(step); masks = fluid_masks(plot)
        metrics = {**pressure(plot), **flux(plot), **z_statistics(plot, masks)}
        common_step = step//run.factor
        if common_step % 100 == 0:
            row = observations[common_step//100]
            for key, value in metrics.items():
                if key in row: agree(value, row[key], f'native/scalar {key} differs at m={common_step}')
        if common_step in (100*n for n in FINAL_CHECKS):
            require(previous is not None and previous.levels[0].step == step-run.factor,
                    'missing previous common-interval snapshot')
            value = relative_l2(previous, plot, masks)
            agree(value, observations[common_step//100]['relative_l2'], 'native/scalar residual differs')
            residuals[common_step] = value
        frames.append(dict(step=step, common_step=common_step, time=plot.time, **metrics))
        previous = plot
    final_profiles, native = profiles(previous, fluid_masks(previous))
    saved = rows(run.directory/'profiles.csv'); expected = list(profile_records(final_profiles))
    require(len(saved) == len(expected), 'incomplete final profiles')
    for row, target in zip(saved, expected):
        require(row['line'] == target['line'], 'wrong profile identity/order')
        for key in PROFILE_FIELDS[1:]: agree(float(row[key]), target[key], f'native/CSV profile {key} differs')
    write_csv(output/'profiles.csv', PROFILE_FIELDS, expected)
    write_csv(output/'profile-native.csv', ('line', 'coordinate', 'ux_norm', 'uy_norm', 'level'),
              (dict(line=name, coordinate=p, ux_norm=ux, uy_norm=uy, level=int(lev))
               for name, data in native.items() for p, ux, uy, lev in data))
    write_csv(output/'frames.csv', tuple(frames[0]), frames)
    gates = []
    def gate(name, value, limit, active=True, strict=False):
        gates.append(dict(metric=name, value=value, limit=limit, passed=int(value < limit if strict else value <= limit), gate=int(active)))
    for m, value in residuals.items(): gate(f'relative_l2_m{m}', value, 5e-7, strict=True)
    for index in FINAL_CHECKS: gate(f'flux_imbalance_m{100*index}', observations[index]['flux_imbalance'], 1e-3)
    if run.dimensions == 3:
        for key in ('uz_max', 'rho_z_max', 'velocity_z_max'):
            gate(key, max(row[key] for row in observations), 1e-10)
        gate('max_abs_Cz', cz, 1e-10)
    external = {row['metric']: float(row['value']) for row in rows(REFERENCE)}
    external = dict(Cd=external['Cd_final'], Cl=external['Cl_final'], Cdp=external['C_delta_p'])
    coefficients = dict(Cd=intervals[-1]['Cd'], Cl=intervals[-1]['Cl'], Cdp=frames[-1]['Cdp'])
    for key, value, limit in coefficient_errors(coefficients, external):
        gate('external_'+key, value, limit, run.layout == 'fine')
    write_csv(output/'gates.csv', ('metric', 'value', 'limit', 'passed', 'gate'), gates)
    (output/'identity.txt').write_text(''.join(f'{key}={value}\n' for key, value in run.id.items()), encoding='utf-8')
    return dict(coefficients=coefficients, profiles=final_profiles, gates=gates,
                failed=[row['metric'] for row in gates if row['gate'] and not row['passed']])


def check(run_directory, reference_directory, output, d2_directory=None):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    local = None; invalid = None; reference_exit = 'not-requested'; d2_exit = 'not-requested'
    try:
        run = Re20Run(run_directory); local = analyze(run, output)
        if run.layout == 'amr': require(reference_directory is not None, 'AMR requires a matching uniform-fine reference')
        for path, mode in ((reference_directory, 'fine'), (d2_directory, 'd2')):
            if path is None: continue
            if mode == 'fine': reference_exit = '2'
            else: d2_exit = '2'
            other = Re20Run(path)
            require(all(run.id[k] == other.id[k] for k in ('source_id', 'backend', 'scheme')),
                    f'{mode} reference source/backend/scheme differs')
            if mode == 'fine':
                require(run.layout != 'fine' and other.layout == 'fine' and run.dimensions == other.dimensions,
                        'wrong fine reference role/descriptor')
            else:
                require(run.dimensions == 3 and other.dimensions == 2 and run.layout == other.layout,
                        'wrong D3/D2 reference role/layout')
            report = analyze(other, output/mode)
            status = int(bool(report['failed'])); qualified = status == 0
            if mode == 'fine': reference_exit = str(status)
            else: d2_exit = str(status)
            active = mode == 'fine' and run.layout == 'amr' and qualified
            comparisons = [dict(metric=name, value=value, limit=limit, passed=int(value <= limit), gate=int(active))
                           for name, value, limit in
                           coefficient_errors(local['coefficients'], report['coefficients']) +
                           profile_errors(local['profiles'], report['profiles'])]
            write_csv(output/f'{mode}-comparison.csv', ('metric', 'value', 'limit', 'passed', 'gate'), comparisons)
            local['failed'].extend(mode+'_'+row['metric'] for row in comparisons if row['gate'] and not row['passed'])
            if mode == 'fine' and run.layout == 'amr' and not qualified:
                invalid = 'uniform-fine reference has failed its scientific gates'
    except (ArtifactError, OSError, ValueError, KeyError, IndexError, TypeError, csv.Error) as error:
        invalid = str(error)
    status = 2 if invalid is not None else int(bool(local['failed']))
    result = f'exit={status}\nlocal_exit={int(bool(local["failed"])) if local else 2}\nreference_exit={reference_exit}\nd2_local_exit={d2_exit}\n'
    if invalid is not None: result += f'invalid={invalid}\n'
    if local: result += ''.join(f'failed={item}\n' for item in local['failed'])
    (output/'result.txt').write_text(result, encoding='utf-8')
    print(f'Re20 checker: exit={status}' + (f' invalid={invalid}' if invalid else ''))
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--d2-reference', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        return check(args.run, args.reference, args.output, args.d2_reference)
    except OSError as error:
        print(f'INVALID Re20 output: {error}',file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
