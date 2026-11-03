#!/usr/bin/env python3
"""Report observed orders only for matching physical refinement families."""
import argparse
from pathlib import Path
import sys
import numpy as np
from check_tgv import TgvRun
from tgv_metrics import observed_order
from amr_plotfile import ArtifactError, require, composite_masks
from amr_validation import write_csv


def same_family(first, second):
    coarse = ~composite_masks(first)[0]
    fine = ~composite_masks(second)[0]
    expanded = coarse
    for axis in range(3):
        ratio = fine.shape[axis]//coarse.shape[axis]
        expanded = np.repeat(expanded, ratio, axis=axis)
    return expanded.shape == fine.shape and np.array_equal(expanded, fine)


def summarize(directories):
    runs = [TgvRun(path) for path in directories]
    indexed = {(run.amr, run.coarse_cells): run for run in runs}
    require(len(runs) == len(indexed) == 6 and set(indexed) == {(amr, n) for amr in (False, True) for n in (20, 40, 80)},
            "expected the AMR20/40/80 and uniform40/80/160 families")
    for key in ("descriptor", "backend", "source_id"):
        require(len({run.meta[key] for run in runs}) == 1, "mixed convergence identity: "+key)
    result = []
    for amr in (False, True):
        for first_n, second_n in ((20, 40), (40, 80)):
            first, second = indexed[amr, first_n], indexed[amr, second_n]
            matching = not amr or same_family(first.plots[0], second.plots[0])
            e1, e2 = first.records[-1]["analytic_e2"], second.records[-1]["analytic_e2"]
            order = observed_order(e1, e2) if matching else None
            reason = "" if order is not None else ("refinement geometry differs" if not matching else "zero error")
            result.append(dict(layout="amr" if amr else "uniform", coarse_first=first_n, coarse_second=second_n,
                               error_first=e1, error_second=e2, observed_order=order, reason=reason))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs=6, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        records = summarize(args.runs)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_csv(args.output, list(records[0]), records)
    except (ArtifactError, OSError, ValueError, KeyError, IndexError) as error:
        print(error, file=sys.stderr)
        sys.exit(2)
