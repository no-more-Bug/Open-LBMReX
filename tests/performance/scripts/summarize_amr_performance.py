#!/usr/bin/env python3
"""Summarize explicit AMR CSVs; optionally propose fixed windows from calibration."""
import argparse
from collections import defaultdict
import csv
import math
from pathlib import Path
from statistics import median
from check_amr_performance_csv import ValidationError, validate, require, close, add_amrex_arguments


def write(path, records):
    if not records: return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)


def mesh_signature(record):
    return tuple(tuple(box[key] for key in ("level","box","lo_x","lo_y","lo_z","hi_x","hi_y","hi_z"))
                 for box in record["boxes"])


def comparison_groups(records):
    groups = defaultdict(list)
    for record in records:
        require(record["finite_final_state"] == "1", "nonfinite artifact cannot be summarized")
        key = tuple(record[field] for field in ("run_mode","measurement_mode","backend","workload"))
        groups[key].append(record)
    for key, group in groups.items():
        first = group[0]
        same = ("git_commit","amrex_source_commit","amrex_prefix","build_type","compiler","mpi","cpu_model")
        require(all(tuple(row[field] for field in same) == tuple(first[field] for field in same) for row in group),
                "comparison mixes platform/build identity")
        for row in group:
            close(row["physical_begin"], first["physical_begin"], "paired physical begin")
            close(row["physical_end"], first["physical_end"], "paired physical end")
        expected = {(layout, ranks) for layout in ("amr","uniform")
                    for ranks in ((16,) if first["backend"] == "cpu" else (1,4))}
        require({(row["layout"], int(row["mpi_ranks"])) for row in group} == expected,
                "comparison group is incomplete")
        for layout in ("amr","uniform"):
            variants = [row for row in group if row["layout"] == layout]
            require(all(mesh_signature(row) == mesh_signature(variants[0]) for row in variants),
                    "same layout has different Boxes")
        if first["backend"] == "cuda":
            identities = {(resource["device_name"], resource["device_total_bytes"])
                          for row in group for resource in row["resources"]}
            require(len(identities) == 1, "comparison mixes MIG specifications")
            require(len({row["gpu_parent_uuid"] for row in group}) == 1, "comparison mixes parent GPUs")
        for layout, ranks in expected:
            variants = [row for row in group if row["layout"] == layout and int(row["mpi_ranks"]) == ranks]
            repeats = [int(row["repeat"]) for row in variants]
            wanted = {1,2,3} if first["run_mode"] == "baseline" and first["measurement_mode"] == "throughput" else {1}
            require(len(repeats) == len(wanted) and set(repeats) == wanted, "missing or duplicate independent repeats")
    return groups


def summarize(records, output, calibrate=False):
    groups = comparison_groups(records)
    output.mkdir(parents=True, exist_ok=True)
    summaries, comparisons, stages, plans = [], [], [], []
    for key, group in groups.items():
        mode, measurement, backend, workload = key
        first = group[0]
        medians = {}
        for layout, ranks in sorted({(row["layout"], int(row["mpi_ranks"])) for row in group}):
            variants = [row for row in group if row["layout"] == layout and int(row["mpi_ranks"]) == ranks]
            values = [float(row["window_s"]) for row in variants]
            duration = float(first["physical_end"])-float(first["physical_begin"])
            medians[layout,ranks] = median(values)
            summaries.append(dict(run_mode=mode,measurement_mode=measurement,backend=backend,workload=workload,
                layout=layout,ranks=ranks,repeats=len(values),window_median_s=median(values),
                window_min_s=min(values),window_max_s=max(values),physical_begin=first["physical_begin"],
                physical_end=first["physical_end"],seconds_per_physical_time=median(values)/duration,
                mlups=median([float(row["mlups"]) for row in variants]) if measurement == "throughput" else "",
                source=first["git_commit"]))
        if measurement == "throughput":
            for ranks in sorted({ranks for layout,ranks in medians}):
                comparisons.append(dict(run_mode=mode,backend=backend,workload=workload,comparison="AMR",
                    resource=str(ranks),speedup=medians["uniform",ranks]/medians["amr",ranks],efficiency=""))
            if backend == "cuda":
                for layout in ("amr","uniform"):
                    speedup = medians[layout,1]/medians[layout,4]
                    comparisons.append(dict(run_mode=mode,backend=backend,workload=workload,comparison="1-to-4-MIG",
                        resource=layout,speedup=speedup,efficiency=speedup/4))
        for row in group:
            for item in row["stages"]:
                stages.append(dict(run_mode=mode,backend=backend,workload=workload,layout=row["layout"],
                    ranks=row["mpi_ranks"],critical_rank=row["critical_rank"],**item))
            if row["stages"]:
                total = float(row["critical_local_s"])
                remaining = total-sum(float(item["critical_s"]) for item in row["stages"])
                stages.append(dict(run_mode=mode,backend=backend,workload=workload,layout=row["layout"],
                    ranks=row["mpi_ranks"],critical_rank=row["critical_rank"],owner="other",level=-1,
                    stage="other_and_timing",critical_s=remaining,calls="",rank_max_s="",rank_max_calls="",
                    share_percent=100*remaining/total,mean_call_s="",mean_level_step_s=""))
        if calibrate:
            require(mode == "calibration", "window planning requires calibration artifacts")
            ratio = int(first["uniform_step_ratio"])
            base_counts = {int(row["nsteps"])//(ratio if row["layout"] == "uniform" else 1) for row in group}
            base_warmups = {int(row["warmup_steps"])//(ratio if row["layout"] == "uniform" else 1) for row in group}
            require(len(base_counts) == len(base_warmups) == 1, "calibration base windows differ")
            base_count, base_warmup = base_counts.pop(), base_warmups.pop()
            # 20% timing margin above the 600-second minimum; this is an estimate,
            # and the formal checker still checks the actual completed duration.
            formal_count = math.ceil(base_count*720/min(float(row["window_s"]) for row in group)) if measurement == "throughput" else base_count
            budget = 6000
            for row in group:
                factor = ratio if row["layout"] == "uniform" else 1
                count, warmup = formal_count*factor, base_warmup*factor
                estimate = float(row["initialization_s"])+(count+warmup)*float(row["window_s"])/int(row["nsteps"])+60
                per_job = min(3, int(budget//estimate)) if measurement == "throughput" else int(estimate <= budget)
                batching = {0:"over_budget",1:"1+1+1",2:"2+1",3:"3"}[per_job] if measurement == "throughput" else ("one" if per_job else "over_budget")
                dt = float(row["levels"][0]["dt"])
                plans.append(dict(backend=backend,workload=workload,layout=row["layout"],ranks=row["mpi_ranks"],
                    measurement_mode=measurement,warmup_steps=warmup,measurement_steps=count,max_steps=warmup+count,
                    physical_begin=warmup*dt,physical_end=(warmup+count)*dt,estimated_run_s=estimate,
                    working_budget_s=budget,repeats_per_job=per_job,batching=batching,source=row["git_commit"]))
    write(output/"summary.csv", summaries)
    write(output/"comparisons.csv", comparisons)
    write(output/"stages.csv", stages)
    write(output/"window-plan.csv", plans)
    return plans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--calibrate", action="store_true")
    add_amrex_arguments(parser)
    args = parser.parse_args()
    try:
        plans = summarize([validate(path, args.expected_amrex_source_commit, args.expected_amrex_prefix) for path in args.csv], args.output, args.calibrate)
        over = sum(row["batching"] == "over_budget" for row in plans)
        print(f"exit=0 summaries written; proposed configurations over budget={over}")
        return 0
    except (OSError, ValueError, KeyError, IndexError, ValidationError) as error:
        print("exit=2 "+str(error)); return 2


if __name__ == "__main__":
    raise SystemExit(main())
