#!/usr/bin/env python3
"""Re100 engineering readback and independent cycle/literature/AMR conclusions."""
import argparse
import hashlib
from pathlib import Path
import sys
import time
from re100_artifacts import Re100Run
from re100_cycles import compare_metrics
from analyze_re100 import REFERENCE_SHA256
from amr_plotfile import ArtifactError, require, rows
from amr_validation import write_csv

REFERENCE = Path(__file__).resolve().parents[1]/"channel_cylinder/reference-data/dfg_re100_reference.csv"
AMR_TOLERANCES = dict(St=.02, Cd_min=.02, Cd_max=.02, Cd_midrange=.02,
                      Cd_peak_to_peak=.10, Cl_min=.05, Cl_max=.05, Cl_peak_to_peak=.05)


def literature(metrics):
    require(hashlib.sha256(REFERENCE.read_bytes()).hexdigest() == REFERENCE_SHA256, "frozen reference changed")
    reference = rows(REFERENCE)
    return compare_metrics(metrics, {row["metric"]: float(row["value"]) for row in reference},
                           {row["metric"]: float(row["relative_tolerance"]) for row in reference})


def check(run, reference_path, output, engineering=False):
    run.export(output)
    conclusions = dict(period_qualification=2, quasi_state=int(not run.quasi_pass),
                       literature="not_evaluated", amr_comparison="not_applicable")
    write_csv(output/"conclusions.csv", list(conclusions), [conclusions])
    if engineering:
        conclusions["scope"] = "engineering"
        return 0, conclusions
    require(run.metrics is not None, "no qualified continuous 3+5 window")
    conclusions["period_qualification"] = 0
    external = literature(run.metrics)
    write_csv(output/"literature.csv", list(external[0]), external)
    conclusions["literature"] = max(row["status"] for row in external)
    if run.layout == "amr":
        conclusions["amr_comparison"] = 2
        # Preserve current-run conclusions even if its required reference is missing.
        write_csv(output/"conclusions.csv", list(conclusions), [conclusions])
        require(reference_path is not None, "matching uniform80 reference is required")
        reference = Re100Run(reference_path)
        require(reference.layout == "fine" and all(run.id[key] == reference.id[key] for key in
                ("source_id", "descriptor", "backend", "scheme")), "wrong uniform80 reference identity")
        require(reference.metrics is not None, "uniform80 has no qualified continuous 3+5 window")
        # Literature failure does not remove a numerically qualified reference.
        comparison = compare_metrics(run.metrics, reference.metrics, AMR_TOLERANCES)
        write_csv(output/"amr-comparison.csv", list(comparison[0]), comparison)
        conclusions["amr_comparison"] = max(row["status"] for row in comparison)
        write_csv(output/"mean-comparison.csv", ("metric", "value", "reference", "absolute_difference"),
                  (dict(metric=key, value=run.metrics[key], reference=reference.metrics[key],
                        absolute_difference=abs(run.metrics[key]-reference.metrics[key]))
                   for key in ("Cd_mean", "Cl_mean")))
    values = [value for value in conclusions.values() if isinstance(value, int)]
    return max(values), conclusions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engineering", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    begin = time.perf_counter()
    status, error = 2, ""
    try:
        run = Re100Run(args.run, args.engineering)
        status, conclusions = check(run, args.reference, args.output, args.engineering)
        write_csv(args.output/"conclusions.csv", list(conclusions), [conclusions])
    except (ArtifactError, OSError, ValueError, KeyError, IndexError) as exception:
        error = str(exception)
    (args.output/"result.txt").write_text(
        f"exit={status}\nscope={'engineering' if args.engineering else 'scientific'}\n"
        f"checker_seconds={time.perf_counter()-begin:.9g}\nerror={error}\n")
    print(f"re100_checker exit={status} {error}")
    return status


if __name__ == "__main__":
    sys.exit(main())
