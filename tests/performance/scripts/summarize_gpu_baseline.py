#!/usr/bin/env python3
"""Summarize the explicit 24-record GPU throughput matrix into eight rows."""

import argparse
from performance_expectations import GPU_FIELDS, add_expected_arguments, expected_values
import csv
from pathlib import Path
import statistics

from check_gpu_performance_csv import ValidationError
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
        groups = load_fixed_matrix(args.records, "throughput", expected)
        require(not args.output.exists(), f"refusing to overwrite {args.output}")
        rows = []
        for (workload, layout), group in groups:
            reference = group[0]
            totals = [float(record["total_s_per_step"]) for record in group]
            mlups = [float(record["mlups"]) for record in group]
            mean = sum(totals) / len(totals)
            cv = statistics.pstdev(totals) / mean
            ratio = max(totals) / min(totals)
            rows.append({
                "schema_version": 1,
                **{field: reference[field] for field in IDENTITY_FIELDS if field not in {"measurement_mode", "run_mode", "workload", "layout"}},
                "measurement_mode": reference["measurement_mode"],
                "run_mode": reference["run_mode"],
                "workload": workload,
                "layout": layout,
                **repeat_provenance(group),
                "repeat_count": 3,
                "noise_status": "shared-node-noisy" if cv > 0.05 or ratio > 1.10 else "baseline",
                "total_min_s_per_step": min(totals),
                "total_median_s_per_step": statistics.median(totals),
                "total_max_s_per_step": max(totals),
                "total_cv": cv,
                "mlups_min": min(mlups),
                "mlups_median": statistics.median(mlups),
                "mlups_max": max(mlups),
            })
        by_key = {(row["workload"], row["layout"]): row for row in rows}
        for workload in {row["workload"] for row in rows}:
            one = by_key[(workload, "1mig")]
            four = by_key[(workload, "4mig")]
            speedup = four["mlups_median"] / one["mlups_median"]
            for row in (one, four):
                row["speedup_1_to_4"] = speedup
                row["efficiency_1_to_4"] = speedup / 4.0
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
    print(f"gpu_baseline passed rows=8 output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
