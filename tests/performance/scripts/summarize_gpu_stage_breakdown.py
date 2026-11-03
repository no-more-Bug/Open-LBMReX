#!/usr/bin/env python3
"""Summarize the explicit 24-record GPU stage-breakdown matrix into eight rows."""

import argparse
from performance_expectations import GPU_FIELDS, add_expected_arguments, expected_values
import csv
from pathlib import Path
import statistics

from check_gpu_performance_csv import (
    CRITICAL_PER_STEP_FIELDS,
    INDEPENDENT_MAX_FIELDS,
    SHARE_FIELDS,
    ValidationError,
)
from gpu_summary_common import (
    IDENTITY_FIELDS,
    load_fixed_matrix,
    repeat_provenance,
    require,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("records", nargs="+", type=Path)
    add_expected_arguments(parser, GPU_FIELDS)
    args = parser.parse_args()
    expected = expected_values(args, GPU_FIELDS)
    try:
        groups = load_fixed_matrix(args.records, "synchronized_stage_breakdown", expected)
        require(not args.output.exists(), f"refusing to overwrite {args.output}")
        rows = []
        for (workload, layout), group in groups:
            reference = group[0]
            totals = [float(record["critical_rank_total_s_per_step"]) for record in group]
            mean = sum(totals) / len(totals)
            cv = statistics.pstdev(totals) / mean
            ratio = max(totals) / min(totals)
            row = {
                "schema_version": 1,
                **{field: reference[field] for field in IDENTITY_FIELDS if field not in {"measurement_mode", "run_mode", "workload", "layout"}},
                "measurement_mode": reference["measurement_mode"],
                "run_mode": reference["run_mode"],
                "workload": workload,
                "layout": layout,
                **repeat_provenance(group),
                "repeat_count": 3,
                "noise_status": "shared-node-noisy" if cv > 0.05 or ratio > 1.10 else "baseline",
                "diagnostic_total_min_s_per_step": min(totals),
                "diagnostic_total_median_s_per_step": statistics.median(totals),
                "diagnostic_total_max_s_per_step": max(totals),
                "diagnostic_total_cv": cv,
                "diagnostic_mlups_median": statistics.median(float(record["diagnostic_mlups"]) for record in group),
            }
            for field in CRITICAL_PER_STEP_FIELDS + SHARE_FIELDS:
                row[f"{field}_median"] = statistics.median(float(record[field]) for record in group)
            nsteps = int(reference["nsteps"])
            for field in INDEPENDENT_MAX_FIELDS:
                suffix = "_window_s"
                stem = field[:-len(suffix)] if field.endswith(suffix) else field
                row[f"{stem}_median_s_per_step"] = statistics.median(float(record[field]) / nsteps for record in group)
            rows.append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    except ValidationError as exc:
        print(f"FAIL: {exc}")
        return 1
    except (OSError, csv.Error, ZeroDivisionError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(f"gpu_stage_breakdown passed rows=8 output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
