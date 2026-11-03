#!/usr/bin/env python3
"""Summarize explicit DSL run directories without adding scientific gates.

Supply the three uniform references and six AMR runs for one Descriptor, or
all 18 runs, with --runs RUN... --output SUMMARY. --plots additionally needs
Matplotlib and writes E/Z figures. With both Descriptors, native plotfiles are
read at common valid times to compare every D3 z plane with D2 at identical xy cells;
the original z data are retained, never averaged before differencing.
Exit 0 means valid summary artifacts were produced; scientific results retain
their original status. Incomplete runs retain their continuous valid prefixes;
invalid/missing runs or layouts return 2 without discarding the other runs.
This script does not run the solver or alter its output.
"""
import argparse
import csv
import hashlib
import math
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'common'))
from amr_plotfile import ArtifactError, require, identity, rows, composite_masks
from analyze_double_shear_layer import DslRun, COLUMNS, LAYOUTS, DERIVATIVE
from check_double_shear_layer import minion, REFERENCE

SPECTRAL = Path(__file__).resolve().parent/'reference-data/fig5-6-pspect-auxiliary.csv'


def label(meta):
    return f'uniform-{meta["base_cells"]}' if meta['layout'] == 'uniform' else meta['layout']


def curve_prefix(run, record):
    """Retain only successive, finite analyzer rows starting at frame zero."""
    values = {}
    try:
        derived = identity(run.directory/'analysis')
        require(all(derived[key] == value for key, value in run.id.items()), 'analysis/run identity differs')
        require(derived['energy_definition'] == 'mean_u_norm_squared/2' and
                derived['enstrophy_definition'] == 'mean_curl_phys_u_norm_squared' and
                derived['integration'] == 'native-uncovered-valid' and
                derived['derivative'] == DERIVATIVE, 'wrong E/Z definition or derivative')
        with (run.directory/'analysis/curves.csv').open(newline='') as stream:
            reader = csv.DictReader(stream)
            require(reader.fieldnames and set(COLUMNS) <= set(reader.fieldnames) and
                    len(reader.fieldnames) == len(set(reader.fieldnames)), 'missing/duplicate curve columns')
            for row in reader:
                frame = len(values)
                require(None not in row and int(row['frame']) == frame < 41 and
                        int(row['step']) == frame*run.n//2 and
                        math.isclose(float(row['time']), frame/20, rel_tol=1e-12, abs_tol=1e-14) and
                        all((frame*run.n//2, lev) in run.samples for lev in range(run.levels)),
                        'nonconsecutive curve or missing source sample')
                numeric = {key: float(row[key]) for key in COLUMNS}
                require(all(math.isfinite(value) for value in numeric.values()) and
                        all(numeric[key] >= 0 for key in ('energy', 'enstrophy', 'mass_drift', 'momentum_drift',
                            'uz_peak', 'uz_rms', 'z_rho', 'z_ux', 'z_uy', 'z_uz')), 'invalid curve values')
                values[frame] = numeric
        require(len(values) == 41, 'incomplete analyzer curve')
    except (OSError, ValueError, KeyError, TypeError, csv.Error) as error:
        record.update(summary_exit=2, error=f'frame={len(values)} nominal_time={len(values)/20}: {error}')
    record.update(valid_frames=len(values), last_valid_frame=len(values)-1 if values else '',
                  last_valid_time=(len(values)-1)/20 if values else '',
                  first_unavailable_frame=len(values) if len(values) < 41 else '',
                  first_unavailable_time=len(values)/20 if len(values) < 41 else '')
    return values


def load_runs(directories):
    groups, records = {}, []
    common = None
    for directory in directories:
        record = dict(run=str(directory.resolve()), valid_frames=0, summary_exit=0,
                      scientific_exit='not-recorded', analysis_exit='not-recorded', error='')
        records.append(record)
        try:
            run = DslRun(directory); meta = run.id
            record.update(descriptor=meta['descriptor'], case=label(meta), source_id=meta['source_id'], backend=meta['backend'])
            signature = tuple(meta[key] for key in ('source_id', 'backend', 'collision', 'reynolds', 'u0_lb',
                                                   'u0_phys', 'viscosity', 'kappa', 'delta', 'initialization', 'periodic'))
            require(common is None or common == signature, 'summary runs have different source/backend/physics')
            common = signature
            entries = groups.setdefault(meta['descriptor'], {})
            name = label(meta)
            require(name not in entries, f'duplicate summary run: {meta["descriptor"]} {name}')
            values = curve_prefix(run, record)
            entries[name] = (directory, meta, values)
            for role, result_path in (('analysis', directory/'analysis/analysis-result.txt'), ('scientific', directory/'check/result.txt')):
                if result_path.is_file():
                    lines = result_path.read_text().splitlines()
                    statuses = [line.removeprefix('exit=') for line in lines if line.startswith('exit=')]
                    require(len(statuses) == 1 and statuses[0] in ('0', '1', '2'), f'invalid {role} exit record')
                    record[role+'_exit'] = statuses[0]
                    record[role+'_details'] = '; '.join(line for line in lines if line.startswith(('error=', 'failure=')))
                    if statuses[0] == '2':
                        record['summary_exit'] = 2
                    elif role == 'scientific':
                        require(f'derivative={DERIVATIVE}' in lines, 'wrong scientific-result derivative')
        except (OSError, ValueError, KeyError, TypeError, csv.Error) as error:
            record.update(summary_exit=2, error=str(error))
    expected = set(LAYOUTS) | {'uniform-128', 'uniform-256', 'uniform-512'}
    for descriptor, entries in groups.items():
        for name in sorted(expected-entries.keys()):
            records.append(dict(descriptor=descriptor, case=name, valid_frames=0, summary_exit=2, error='missing configuration'))
    return groups, records


def reference_data(output):
    main = minion()
    auxiliary = rows(SPECTRAL)
    require(len(auxiliary) == 21, 'expected 21 auxiliary PSpect samples')
    for index, row in enumerate(auxiliary):
        require(math.isclose(float(row['time']), index/20, rel_tol=0, abs_tol=1e-14) and
                all(math.isfinite(float(row[key])) and float(row[key]) > 0 for key in
                    ('energy', 'enstrophy', 'energy_readout_abs', 'enstrophy_readout_abs')),
                'invalid auxiliary PSpect sample')
    fields = ('role', 'time', 'energy', 'energy_readout_abs', 'enstrophy', 'enstrophy_readout_abs', 'method')
    with (output/'literature-points.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields); writer.writeheader()
        for role, data in (('Minion-open-circle-main', main), ('PSpect-auxiliary', auxiliary)):
            for row in data:
                writer.writerow(dict(role=role, **{key: row[key] for key in fields[1:]}))
    return main, auxiliary


def write_curves(groups, records, output):
    identity_fields = ('descriptor', 'layout', 'base_cells', 'source_id', 'backend', 'run', 'scientific_exit')
    with (output/'curves.csv').open('w', newline='') as stream, (output/'inputs.csv').open('w', newline='') as inputs:
        writer = csv.DictWriter(stream, (*identity_fields, *COLUMNS)); writer.writeheader()
        sources = csv.DictWriter(inputs, ('run', 'descriptor', 'case', 'source_id', 'backend', 'valid_frames',
            'last_valid_frame', 'last_valid_time', 'first_unavailable_frame', 'first_unavailable_time',
            'analysis_exit', 'scientific_exit', 'summary_exit', 'error', 'analysis_details', 'scientific_details'))
        sources.writeheader(); sources.writerows(records)
        statuses = {}
        for record in records:
            statuses.setdefault(record.get('run'), record.get('scientific_exit', 'not-recorded'))
        for descriptor, entries in groups.items():
            for name, (directory, meta, values) in entries.items():
                source = dict(descriptor=descriptor, layout=meta['layout'], base_cells=meta['base_cells'],
                              source_id=meta['source_id'], backend=meta['backend'], run=str(directory.resolve()),
                              scientific_exit=statuses[str(directory.resolve())])
                for frame in values:
                    writer.writerow(dict(**source, **{key: values[frame][key] for key in COLUMNS}))


def plots(groups, output, paper, spectral):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    for descriptor, entries in groups.items():
        for case in ('uniform', *LAYOUTS):
            if case != 'uniform' and (case not in entries or not entries[case][2]):
                continue
            if case == 'uniform' and not any(name.startswith('uniform-') and entry[2] for name, entry in entries.items()):
                continue
            figure, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
            matched = 256 if case == 'baseline' else 512
            for axis, quantity, symbol in zip(axes, ('energy', 'enstrophy'), ('E*', 'Z*')):
                for n, color, style in ((128, '.65', ':'), (256, '.35', '--'), (512, '.1', '-')):
                    if f'uniform-{n}' not in entries or not entries[f'uniform-{n}'][2]:
                        continue
                    values = entries[f'uniform-{n}'][2]
                    role = 'matched' if case != 'uniform' and n == matched else 'auxiliary' if n == 128 or case != 'uniform' else 'reference'
                    suffix = f'; prefix to {max(values)/20:g}' if len(values) < 41 else ''
                    axis.plot([row['time'] for row in values.values()], [row[quantity] for row in values.values()],
                              color=color, linestyle=style, marker='.', markersize=3, label=f'Uniform {n} ({role}{suffix})')
                if case != 'uniform':
                    values = entries[case][2]
                    suffix = f'; prefix to {max(values)/20:g}' if len(values) < 41 else ''
                    axis.plot([row['time'] for row in values.values()], [row[quantity] for row in values.values()],
                              color='#2563a6', marker='.', markersize=3, label=f'AMR {case}{suffix}')
                axis.errorbar([float(row['time']) for row in paper], [float(row[quantity]) for row in paper],
                    yerr=[float(row[quantity+'_readout_abs']) for row in paper], fmt='o', markersize=4,
                    markerfacecolor='white', markeredgecolor='black', ecolor='black', capsize=2,
                    label='Minion open circles (main)')
                axis.plot([float(row['time']) for row in spectral], [float(row[quantity]) for row in spectral],
                          color='#bb6a22', linestyle='-.', label='PSpect digitized (auxiliary)')
                axis.set(xlim=(0, 2), xlabel='t / tc', ylabel=symbol)
                axis.grid(alpha=.2); axis.legend(fontsize=7)
            figure.suptitle(f'{descriptor} {case}; available prefixes; literature ends at t/tc=1; no extrapolation')
            for extension in ('png', 'svg'):
                figure.savefig(output/f'{descriptor}-{case}-curves.{extension}', dpi=180)
            plt.close(figure)


def compare_descriptors(groups, output):
    invalid = False
    fields = ('layout', 'base_cells', 'frame', 'time', 'level', 'step', 'k', 'z', 'quantity',
              'cell_count', 'mean_difference', 'l1', 'rms', 'peak', 'peak_i', 'peak_j', 'peak_x', 'peak_y')
    whole_fields = ('layout', 'base_cells', 'frame', 'time', 'quantity', 'l1', 'rms', 'peak')
    with (output/'d2-d3-per-z.csv').open('w', newline='') as stream, \
            (output/'d2-d3-composite.csv').open('w', newline='') as whole, \
            (output/'d2-d3-pairs.csv').open('w', newline='') as pairs:
        writer = csv.DictWriter(stream, fields); writer.writeheader()
        aggregate = csv.DictWriter(whole, whole_fields); aggregate.writeheader()
        pair_writer = csv.writer(pairs)
        pair_writer.writerow(('case', 'D2_run', 'D3_run', 'common_frames', 'compared_frames', 'exit',
                              'failure_frame', 'failure_time', 'error'))
        for name in sorted(groups['D2Q9'].keys() & groups['D3Q27'].keys()):
            directory2, meta, values2 = groups['D2Q9'][name]
            directory3, _, values3 = groups['D3Q27'][name]
            frames = sorted(values2.keys() & values3.keys())
            compared = 0
            failure_frame = len(frames) if len(frames) < 41 else ''
            problem = 'incomplete common time range' if len(frames) < 41 else ''
            d2, d3 = DslRun(directory2), DslRun(directory3)
            require(d2.n == d3.n and d2.levels == d3.levels and d2.layout == d3.layout, 'unmatched D2/D3 geometry')
            for frame in frames:
                try:
                    two, three = d2.plot(frame), d3.plot(frame)
                    masks2, masks3 = composite_masks(two), composite_masks(three)
                    total_abs, total_square, maximum = np.zeros(4), np.zeros(4), np.zeros(4)
                    volume = 0.
                    frame_rows = []
                    for lev, (a, b, mask2, mask3) in enumerate(zip(two.levels, three.levels, masks2, masks3)):
                        selected = mask2[0]
                        require(np.array_equal(mask3, np.broadcast_to(selected, mask3.shape)) and
                                a.spacing[:2] == b.spacing[:2] and a.step == b.step, 'different native xy coverage/clock')
                        js, ix = np.nonzero(selected)
                        dv = math.prod(b.spacing)
                        baseline = a.values[:, 0][:, selected]
                        for k in range(b.valid.shape[0]):
                            difference = b.values[:, k][:, selected]-baseline
                            absolute = np.abs(difference)
                            sum_abs = absolute.sum(axis=1); sum_square = (difference*difference).sum(axis=1)
                            mean = difference.mean(axis=1); peaks = absolute.max(axis=1)
                            rms = np.sqrt(sum_square/len(ix))
                            require(np.isfinite(np.concatenate((sum_abs, sum_square, mean, peaks, rms))).all(),
                                    'nonfinite D2/D3 macro differences')
                            total_abs += sum_abs*dv; total_square += sum_square*dv
                            maximum = np.maximum(maximum, peaks); volume += len(ix)*dv
                            for q, quantity in enumerate(('rho/rho0', 'ux/u0', 'uy/u0', 'uz/u0')):
                                location = int(np.argmax(absolute[q]))
                                frame_rows.append(dict(layout=meta['layout'], base_cells=d2.n, frame=frame, time=frame/20,
                                    level=lev, step=a.step, k=k, z=(k+.5)*b.spacing[2], quantity=quantity, cell_count=len(ix),
                                    mean_difference=mean[q], l1=sum_abs[q]/len(ix), rms=rms[q], peak=peaks[q],
                                    peak_i=ix[location], peak_j=js[location],
                                    peak_x=(ix[location]+.5)*b.spacing[0], peak_y=(js[location]+.5)*b.spacing[1]))
                    writer.writerows(frame_rows)
                    for q, quantity in enumerate(('rho/rho0', 'ux/u0', 'uy/u0', 'uz/u0')):
                        aggregate.writerow(dict(layout=meta['layout'], base_cells=d2.n, frame=frame, time=frame/20,
                            quantity=quantity, l1=total_abs[q]/volume, rms=math.sqrt(total_square[q]/volume), peak=maximum[q]))
                    stream.flush(); whole.flush()
                    compared += 1
                    print(f'D2/D3 {name}: frame={frame} compared every native z layer', flush=True)
                except (OSError, ValueError, KeyError) as error:
                    failure_frame, problem = frame, str(error)
                    break
            status = 2 if problem else 0
            invalid |= status == 2
            pair_writer.writerow((name, str(directory2.resolve()), str(directory3.resolve()), len(frames), compared,
                                  status, failure_frame, failure_frame/20 if failure_frame != '' else '', problem))
            pairs.flush()
    return invalid


def summarize(directories, output, make_plots):
    groups, records = load_runs(directories)
    output.mkdir(parents=True, exist_ok=True)
    write_curves(groups, records, output)
    paper, spectral = reference_data(output)
    meta = next((entry[1] for entries in groups.values() for entry in entries.values()), {})
    (output/'summary-identity.txt').write_text(
        f'source_id={meta.get("source_id", "unavailable")}\nbackend={meta.get("backend", "unavailable")}\ninput_runs={len(directories)}\n'
        f'derivative={DERIVATIVE}\ndescriptors={",".join(groups)}\nmain_reference_sha256={hashlib.sha256(REFERENCE.read_bytes()).hexdigest()}\n'
        f'auxiliary_reference_sha256={hashlib.sha256(SPECTRAL.read_bytes()).hexdigest()}\n'
        'paper_time_window=0..1\npaper_lines=visual-connectors-only\nreference_extrapolation=none\n'
        'macro_difference=D3(z)-D2-at-identical-native-xy-cells\nmacro_normalization=rho0-and-u0\n'
        'macro_aggregation=native-uncovered-volume-after-per-z-differencing\n'
        'original_z_data=retained-in-input-native-plotfiles\nscientific_gates=not-added-or-reevaluated\n')
    if make_plots:
        plots(groups, output, paper, spectral)
    invalid = any(record['summary_exit'] == 2 for record in records)
    if set(groups) == {'D2Q9', 'D3Q27'}:
        invalid |= compare_descriptors(groups, output)
    status = 2 if invalid else 0
    (output/'summary-result.txt').write_text(
        f'exit={status}\ninput_runs={len(directories)}\nplots={int(make_plots)}\n'
        f'cross_descriptor={int(len(groups) == 2)}\nscientific_status=see-inputs.csv\n')
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', required=True, nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--plots', action='store_true', help='also write PNG/SVG curves using Matplotlib')
    args = parser.parse_args()
    try:
        return summarize(args.runs, args.output, args.plots)
    except (OSError, ValueError, KeyError, ImportError, csv.Error) as error:
        print(f'INVALID DSL summary: {error}')
        try:
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output/'summary-result.txt').write_text(f'exit=2\nerror={error}\n')
        except OSError as output_error:
            print(f'Cannot save summary failure: {output_error}')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
