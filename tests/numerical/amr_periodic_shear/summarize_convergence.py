#!/usr/bin/env python3
"""Report base-grid orders without imposing monotonicity or an order gate."""
import csv
import math
from pathlib import Path
import sys


def main():
    root = Path(sys.argv[1]); grids = list(map(int, sys.argv[2:]))
    fields = ('time', 'comparison', 'region', 'level')
    try:
        previous = None
        with (root/'orders.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(('coarse_N', 'fine_N', *fields, 'norm', 'error_coarse', 'error_fine', 'order'))
            for n in grids:
                with (root/f'half-N{n}/analysis/metrics.csv').open(newline='') as data:
                    current = {}
                    for row in csv.DictReader(data):
                        if row['region'] != 'composite' and row['comparison'] != 'reference-analytic':
                            continue
                        key = (float(row['time']), row['comparison'], row['region'], int(row['level']))
                        if key in current:
                            raise ValueError('duplicate convergence sample')
                        if any(not math.isfinite(float(row[k])) or float(row[k]) < 0 for k in ('l1', 'l2', 'peak')):
                            raise ValueError('nonfinite/negative error norm')
                        current[key] = row
                    expected = {(t, label, 'composite', -1) for t in (0, .125, .25, .5, 1, 2, 4)
                                for label in ('analytic-domain', 'reference-domain', 'analytic-interface', 'reference-interface', 'pressure')}
                    expected |= {(t, 'reference-analytic', 'domain', 0) for t in (0, .125, .25, .5, 1, 2, 4)}
                    if current.keys() != expected:
                        raise ValueError('incomplete convergence sampling')
                if previous is not None:
                    old_n, old = previous
                    if current.keys() != old.keys():
                        raise ValueError('incomplete convergence sampling')
                    for key in current:
                        for metric in ('l1', 'l2', 'peak'):
                            a, b = float(old[key][metric]), float(current[key][metric])
                            order = math.log2(a/b) if a > 0 and b > 0 else 'undefined-zero-error'
                            writer.writerow((old_n, n, *key, metric, a, b, order))
                previous = n, current
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f'INVALID convergence evidence: {error}')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
