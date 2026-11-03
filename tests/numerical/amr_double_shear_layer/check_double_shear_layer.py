#!/usr/bin/env python3
"""DSL scientific gates: complete curves, Minion open circles and uniform reference."""
import argparse
import csv
import hashlib
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'common'))
from amr_plotfile import ArtifactError, require, identity, rows
from analyze_double_shear_layer import DslRun, DERIVATIVE

REFERENCE = Path(__file__).resolve().parent/'reference-data/fig5-6-open-circle-reference.csv'


def curve(run):
    run = Path(run); scientific_run = DslRun(run); scientific_run.require_complete()
    meta = scientific_run.id
    analysis = run/'analysis'
    derived = identity(analysis)
    require(all(derived[key] == value for key, value in meta.items()), 'analysis/run identity differs')
    require(derived['energy_definition'] == 'mean_u_norm_squared/2' and
            derived['enstrophy_definition'] == 'mean_curl_phys_u_norm_squared' and
            derived['integration'] == 'native-uncovered-valid', 'wrong energy/enstrophy definition')
    require(derived['derivative'] == DERIVATIVE, 'unexpected DSL derivative reconstruction')
    result = {}
    for row in rows(analysis/'curves.csv'):
        frame = int(row['frame'])
        require(frame not in result and 0 <= frame <= 40 and int(row['step']) == frame*int(meta['base_cells'])//2 and
                math.isclose(float(row['time']), frame/20, rel_tol=1e-12, abs_tol=1e-14), 'wrong DSL curve sample')
        values = {key: float(value) for key, value in row.items()}
        require(all(math.isfinite(value) for value in values.values()), 'nonfinite DSL curve')
        require(all(values[key] >= 0 for key in ('energy', 'enstrophy', 'mass_drift', 'momentum_drift', 'uz_peak', 'uz_rms',
                                                'z_rho', 'z_ux', 'z_uy', 'z_uz')), 'negative norm or squared integral')
        result[frame] = values
    require(result.keys() == set(range(41)), 'incomplete DSL curve')
    return meta, result


def minion():
    data = rows(REFERENCE)
    require(len(data) == 11 and all(row['method'] == 'open_circle_center' for row in data), 'wrong Minion reference identity')
    for frame, row in enumerate(data):
        require(math.isclose(float(row['time']), frame/10, abs_tol=1e-14, rel_tol=0), 'wrong Minion nominal time')
        require(all(math.isfinite(float(row[key])) and float(row[key]) > 0 for key in
                    ('energy', 'enstrophy', 'energy_readout_abs', 'enstrophy_readout_abs')), 'invalid Minion reference')
    return data


def local_failures(meta, values):
    failure = []
    for frame, row in values.items():
        keys = ['mass_drift', 'momentum_drift']
        if meta['dimensions'] == '3':
            keys += ['uz_peak', 'z_rho', 'z_ux', 'z_uy', 'z_uz']
        failure.extend(f'frame={frame} {key}' for key in keys if row[key] > 1e-10)
    return failure


def check_uniform(directory, output):
    meta, values = curve(directory)
    require(meta['layout'] == 'uniform', 'expected uniform reference')
    failure = local_failures(meta, values)
    # N128 is auxiliary; N256/N512 are matched main references.
    main_reference = int(meta['base_cells']) in (256, 512)
    comparisons = []
    for index, paper in enumerate(minion()):
        frame = 2*index
        for quantity, tolerance in (('energy', .01), ('enstrophy', .10)):
            expected = float(paper[quantity]); actual = values[frame][quantity]
            error = abs(actual-expected); budget = tolerance*expected+float(paper[quantity+'_readout_abs'])
            comparisons.append(dict(frame=frame, time=frame/20, quantity=quantity, actual=actual,
                reference=expected, absolute_error=error, budget=budget, gate=int(main_reference), passed=int(error <= budget)))
            if main_reference and error > budget:
                failure.append(f'Minion frame={frame} {quantity}')
    output.mkdir(parents=True, exist_ok=True)
    with (output/'comparisons.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, list(comparisons[0])); writer.writeheader(); writer.writerows(comparisons)
    status = int(bool(failure))
    (output/'result.txt').write_text(f'exit={status}\nrole={"matched-reference" if main_reference else "auxiliary"}\n'
        f'derivative={DERIVATIVE}\n'+
        ''.join('failure='+item+'\n' for item in failure))
    return status


def compare(amr_directory, reference_directory, output):
    a, amr = curve(amr_directory); r, reference = curve(reference_directory)
    require(a['schema'] == r['schema'] == 'double-shear-v1' and a['layout'] != 'uniform' and r['layout'] == 'uniform', 'wrong DSL comparison roles')
    require(all(a[key] == r[key] for key in ('source_id', 'descriptor', 'backend', 'reynolds', 'u0_lb', 'u0_phys',
                                           'viscosity', 'length_z', 'collision', 'initialization', 'periodic', 'real_digits', 'dimensions')), 'unmatched DSL reference')
    nref = 256 if a['layout'] == 'baseline' else 512
    require(int(r['base_cells']) == nref, 'wrong matched uniform resolution')
    failure = []
    comparisons = []
    for role, values, meta in (('amr', amr, a), ('reference', reference, r)):
        failure.extend(role+' '+item for item in local_failures(meta, values))
    for frame in range(41):
        for quantity, tolerance in (('energy', .01), ('enstrophy', .05)):
            expected = reference[frame][quantity]
            require(expected > 0, f'unusable {quantity} uniform reference')
            error = abs(amr[frame][quantity]-expected)
            budget = tolerance*abs(expected)
            comparisons.append(dict(comparison='amr-uniform', frame=frame, time=frame/20, quantity=quantity,
                                    actual=amr[frame][quantity], reference=expected, absolute_error=error, budget=budget, passed=int(error <= budget)))
            if error > budget:
                failure.append(f'amr-uniform frame={frame} {quantity}')
    for index, paper in enumerate(minion()):
        frame = 2*index
        for role, values in (('amr', amr), ('reference', reference)):
            for quantity, tolerance in (('energy', .01), ('enstrophy', .10)):
                expected = float(paper[quantity]); error = abs(values[frame][quantity]-expected)
                budget = tolerance*expected+float(paper[quantity+'_readout_abs'])
                comparisons.append(dict(comparison=role+'-Minion-open-circle', frame=frame, time=frame/20,
                    quantity=quantity, actual=values[frame][quantity], reference=expected,
                    absolute_error=error, budget=budget, passed=int(error <= budget)))
                if error > budget:
                    failure.append(f'{role}-Minion frame={frame} {quantity}')
    output.mkdir(parents=True, exist_ok=True)
    with (output/'comparisons.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, list(comparisons[0])); writer.writeheader(); writer.writerows(comparisons)
    status = int(bool(failure))
    (output/'result.txt').write_text(f'exit={status}\namr={Path(amr_directory).resolve()}\nreference={Path(reference_directory).resolve()}\n'
        f'source_id={a["source_id"]}\nderivative={DERIVATIVE}\nminion_sha256={hashlib.sha256(REFERENCE.read_bytes()).hexdigest()}\n'+
        ''.join('failure='+item+'\n' for item in failure))
    print(f'DSL {a["descriptor"]} {a["layout"]}: {"FAIL" if failure else "PASS"}; failed_conditions={len(failure)}')
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--amr', type=Path); selection.add_argument('--uniform', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.uniform is not None:
            return check_uniform(args.uniform, args.output)
        require(args.reference is not None, 'a matching uniform reference is required')
        return compare(args.amr, args.reference, args.output)
    except (ArtifactError, OSError, ValueError, KeyError, csv.Error) as error:
        print(f'INVALID DSL comparison: {error}')
        try:
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output/'result.txt').write_text(f'exit=2\nerror={error}\n')
        except OSError as output_error:
            print(f'Cannot save invalid-result record: {output_error}')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
