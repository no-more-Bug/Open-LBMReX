#!/usr/bin/env python3
"""Summarize one fixed workload or the complete four-workload baseline."""

import argparse
from performance_expectations import CPU_FIELDS, add_expected_arguments, expected_values
import csv
from pathlib import Path
import statistics
from typing import Dict, List, Union

from check_performance_csv import TIMING_FIELDS, ValidationError, validate_path


WORKLOAD_ORDER = ["tgv_d2q9", "tgv_d3q27", "dfg_d2q9", "dfg_d3q27"]
STAGE_FIELDS = [
    name
    for name in TIMING_FIELDS
    if name.endswith("_window_s") and name != "synchronized_window_s"
]
CONTEXT_FIELDS = [
    "descriptor",
    "git_commit",
    "amrex_source_commit",
    "amrex_prefix",
    "amrex_build_type",
    "build_type",
    "compiler",
    "mpi",
    "ucx_version",
    "ucx_tls",
    "node",
    "partition",
    "qos",
    "grid_x",
    "grid_y",
    "grid_z",
    "global_valid_cells",
    "max_grid_size",
    "box_count",
    "mpi_ranks",
    "omp_threads",
    "cpuset",
    "binding_policy",
    "sample_start",
    "sample_end",
    "nsteps",
    "timing_reduction",
]

GLOBAL_CONTEXT_FIELDS = [
    "amrex_source_commit",
    "amrex_prefix",
    "amrex_build_type",
    "build_type",
    "compiler",
    "mpi",
    "ucx_version",
    "ucx_tls",
    "partition",
    "qos",
    "mpi_ranks",
    "omp_threads",
    "binding_policy",
    "timing_reduction",
]

# Each workload is frozen and repeated at one commit. The complete matrix may
# span tests-owned identity commits, whose per-row provenance remains explicit.


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def summarize(
    records: List[Dict[str, str]]
) -> List[Dict[str, Union[str, float, int]]]:
    require(
        len(records) in {3, 12},
        f"expected one workload (3 records) or the full matrix (12), found {len(records)}",
    )
    selected = {record["workload"] for record in records}
    if len(records) == 3:
        require(len(selected) == 1, "three records must belong to one workload")
    else:
        require(
            selected == set(WORKLOAD_ORDER),
            "twelve records must cover all four fixed workloads",
        )
    grouped = {workload: [] for workload in WORKLOAD_ORDER if workload in selected}
    for record in records:
        require(record["mode"] == "baseline", "summary accepts baseline mode only")
        grouped[record["workload"]].append(record)

    shared = records[0]
    for record in records[1:]:
        for field in GLOBAL_CONTEXT_FIELDS:
            require(
                record[field] == shared[field],
                f"baseline matrix context drift in {field}",
            )

    output = []
    for workload in WORKLOAD_ORDER:
        if workload not in grouped:
            continue
        group = grouped[workload]
        require(len(group) == 3, f"{workload}: expected three repeats")
        require(
            {int(record["repeat"]) for record in group} == {1, 2, 3},
            f"{workload}: repeats must be 1, 2 and 3",
        )
        reference = group[0]
        for record in group[1:]:
            for field in CONTEXT_FIELDS:
                require(
                    record[field] == reference[field],
                    f"{workload}: context drift in {field}",
                )

        totals = [float(record["total_s_per_step"]) for record in group]
        mlups = [float(record["mlups"]) for record in group]
        total_mean = sum(totals) / len(totals)
        total_cv = statistics.pstdev(totals) / total_mean
        ratio = max(totals) / min(totals)
        row = {
            "schema_version": 2,
            "workload": workload,
            **{field: reference[field] for field in CONTEXT_FIELDS},
            "repeat_count": 3,
            "noise_status": (
                "shared-node-noisy"
                if total_cv > 0.05 or ratio > 1.10
                else "baseline"
            ),
            "total_min_s_per_step": min(totals),
            "total_median_s_per_step": statistics.median(totals),
            "total_max_s_per_step": max(totals),
            "total_cv": total_cv,
            "mlups_min": min(mlups),
            "mlups_median": statistics.median(mlups),
            "mlups_max": max(mlups),
        }
        nsteps = int(reference["nsteps"])
        for field in STAGE_FIELDS:
            stem = field[: -len("_window_s")]
            row[f"{stem}_median_s_per_step"] = statistics.median(
                float(record[field]) / nsteps for record in group
            )
        output.append(row)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("records", nargs="+", type=Path)
    add_expected_arguments(parser, CPU_FIELDS)
    args = parser.parse_args()
    expected = expected_values(args, CPU_FIELDS)
    try:
        records = [validate_path(path, expected) for path in args.records]
        rows = summarize(records)
        require(not args.output.exists(), f"refusing to overwrite {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    except ValidationError as exc:
        print(f"FAIL: {exc}")
        return 1
    except (OSError, csv.Error) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(f"cpu_mpi_baseline passed rows={len(rows)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
