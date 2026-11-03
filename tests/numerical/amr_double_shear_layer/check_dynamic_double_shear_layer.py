#!/usr/bin/env python3
"""Check DSL artifact completeness and finite data; apply no scientific error gate."""
import argparse
import csv
import math
from pathlib import Path

from analyze_dynamic_double_shear_layer import open_run, DERIVATIVE, OUTPUT_COLUMNS
from amr_plotfile import ArtifactError, require, identity, rows, composite_masks, totals


def check(run, analysis):
    result = identity(analysis, 'analysis-result.txt')
    require(result.get('exit') == '0' and int(result.get('frames', -1)) == len(run.steps), 'analysis did not finish')
    metadata = identity(analysis)
    require(all(metadata.get(key) == value for key, value in run.id.items()) and
            metadata.get('derivative') == DERIVATIVE and metadata.get('integration') == 'native-uncovered-valid',
            'analysis identity differs from run')
    curves = rows(analysis/'curves.csv')
    require(len(curves) == len(run.steps), 'missing/extra diagnostic frames')
    for frame, (step, row) in enumerate(zip(run.steps, curves)):
        require(set(row) == set(OUTPUT_COLUMNS) and int(row['frame']) == frame and int(row['step']) == step,
                'invalid diagnostic frame schema/order')
        for key, value in row.items():
            if key in ('velocity_l2', 'velocity_linf') and 'reference_directory' not in metadata:
                require(value == '', 'unbound reference error')
            else:
                require(math.isfinite(float(value)), f'nonfinite diagnostic {key}')
        plot = run.plot(frame)
        masks = composite_masks(plot)
        mass, momentum = totals(plot, masks)
        require(math.isclose(float(row['time']), plot.time, abs_tol=1e-14, rel_tol=1e-12) and
                int(row['active_levels']) == len(plot.levels) and
                int(row['composite_cells']) == sum(int(mask.sum()) for mask in masks),
                'diagnostic time/layout differs')
        require(math.isclose(float(row['mass']), mass, rel_tol=1e-12) and
                all(math.isclose(float(row[key]), value, abs_tol=1e-12, rel_tol=1e-12)
                    for key, value in zip(('px', 'py', 'pz'), momentum)), 'diagnostic integral differs')
    return len(curves)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--analysis', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--short', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        frames = check(open_run(args.run, args.short), args.analysis)
        (args.output/'result.txt').write_text(f'exit=0\nframes={frames}\nscientific_gate=none\nclassification=observation-only\n')
        print(f'DATA VALID: {frames} frames; no scientific error gate')
        return 0
    except (ArtifactError, OSError, ValueError, KeyError, csv.Error) as error:
        (args.output/'result.txt').write_text(f'exit=2\nerror={error}\n')
        print(f'INVALID DSL evidence: {error}')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
