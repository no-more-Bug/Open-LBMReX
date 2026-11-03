#!/usr/bin/env python3
"""Check one AMR performance artifact; no speed or scaling acceptance threshold."""
import argparse
import csv
import math
from pathlib import Path
import re
from check_gpu_mapping_log import ValidationError, cpu_set, require

LEVEL_STAGES = ("collision", "communication", "streaming", "boundary_pre", "boundary_post", "management")
PAIR_STAGES = ("coarse_to_fine_communication", "fine_to_coarse_communication", "explode", "coalesce")


def rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def artifact(path, suffix):
    return path.with_name(path.stem+"-"+suffix+".csv")


def number(row, name):
    value = float(row[name])
    require(math.isfinite(value), "nonfinite "+name)
    return value


def close(actual, expected, name):
    require(math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-13), "inconsistent "+name)


def validate(path, expected_source, expected_prefixes, source=None):
    path = Path(path)
    records = rows(path)
    require(len(records) == 1, "expected one window")
    record = records[0]
    require(record["schema"] == "amr-performance-v1", "wrong schema")
    workload = record["workload"]
    require(workload in ("tgv_d2q9", "tgv_d3q27", "dfg_d2q9", "dfg_d3q27"), "wrong workload")
    require(record["descriptor"].lower() == workload.split("_")[1], "descriptor differs")
    require(record["layout"] in ("amr", "uniform") and record["backend"] in ("cpu", "cuda"), "wrong layout/backend")
    require(record["run_mode"] in ("smoke", "calibration", "baseline"), "wrong run mode")
    require(record["measurement_mode"] in ("throughput", "stage_breakdown"), "wrong measurement mode")
    require(re.fullmatch("[0-9a-f]{40}", record["git_commit"]), "missing source identity")
    require(source is None or record["git_commit"] == source, "source differs")
    require(record["build_type"] == "Release" and record["omp_threads"] == "1", "wrong build/thread identity")
    require(record["cpu_model"] not in ("", "unknown") and record["mpi"] not in ("", "unknown"), "missing platform identity")
    require(re.fullmatch("[0-9a-f]{40}", record["amrex_source_commit"]), "invalid AMReX source identity")
    require(record["amrex_source_commit"] == expected_source, "AMReX source differs from explicit expectation")
    require(Path(record["amrex_prefix"]).is_absolute() and record["amrex_prefix"] in expected_prefixes,
            "AMReX prefix differs from explicit expectations")
    ranks = int(record["mpi_ranks"])
    require((record["backend"] == "cpu" and ranks == 16) or (record["backend"] == "cuda" and ranks in (1, 4)), "wrong ranks")
    require(int(record["repeat"]) > 0, "invalid repeat")
    warmup, count = int(record["warmup_steps"]), int(record["nsteps"])
    require(warmup >= 0 and count > 0 and int(record["sample_start"]) == warmup+1 and
            int(record["sample_end"]) == warmup+count, "incomplete window")
    require(number(record, "window_s") > 0 and number(record, "initialization_s") >= 0, "invalid timing")
    if record["run_mode"] == "baseline" and record["measurement_mode"] == "throughput":
        require(number(record, "window_s") >= 600, "formal Throughput window is shorter than 600 seconds")
    require(record["finite_final_state"] in ("0", "1"), "missing finite-state check")
    tgv, d3 = workload.startswith("tgv"), workload.endswith("d3q27")
    levels = rows(artifact(path, "levels"))
    expected_levels = 1 if record["layout"] == "uniform" else (2 if tgv else 3)
    require(len(levels) == expected_levels, "hierarchy size differs")
    base = (160 if record["layout"] == "uniform" else 80) if tgv else (80 if record["layout"] == "uniform" else 20)
    step_ratio = 2 if tgv else 4
    boxes = rows(artifact(path, "boxes"))
    updates = 0
    for lev, level in enumerate(levels):
        n = base*2**lev
        shape = (n, n, n//5 if d3 else 1) if tgv else (
            (22*base+1)*2**lev, (41*base//10+1)*2**lev, n//10 if d3 else 1)
        require(int(level["level"]) == lev and tuple(int(level[key]) for key in ("nx","ny","nz")) == shape,
                "level grid differs")
        h = 1/n if tgv else .1/n
        dt = (5/(6*160) if tgv else .035)*h
        close(level["dt"], dt, "level dt")
        close(level["tau"], .5+3*(.1 if tgv else .001)*dt/h**2, "level tau")
        require(int(level["measured_steps"]) == count*2**lev, "level steps differ")
        selected = [box for box in boxes if int(box["level"]) == lev]
        require(len(selected) == int(level["boxes"]) and
                {int(box["box"]) for box in selected} == set(range(len(selected))), "Box count/ids differ")
        cells = 0
        for box in selected:
            require(0 <= int(box["rank"]) < ranks, "invalid Box owner")
            lengths = []
            for axis, extent in zip("xyz", shape):
                lo, hi = int(box["lo_"+axis]), int(box["hi_"+axis])
                require(0 <= lo <= hi < extent, "Box outside level domain")
                lengths.append(hi-lo+1)
            cells += lengths[0]*lengths[1]*lengths[2]
        require(cells > 0 and cells == int(level["valid_cells"]), "valid-cell count differs")
        if lev == 0: require(cells == shape[0]*shape[1]*shape[2], "L0 does not cover the full workload")
        require(int(level["valid_updates"]) == cells*count*2**lev, "level work differs")
        updates += int(level["valid_updates"])
    require(len(boxes) == sum(int(level["boxes"]) for level in levels), "unexpected Box level")
    close(record["physical_begin"], warmup*float(levels[0]["dt"]), "physical begin")
    close(record["physical_end"], (warmup+count)*float(levels[0]["dt"]), "physical end")
    require(updates == int(record["valid_updates"]), "total work differs")
    stages = []
    if record["measurement_mode"] == "throughput":
        require(record["critical_local_s"] == "" and not artifact(path, "stages").exists(), "Throughput has stage data")
        close(record["mlups"], updates/(1e6*float(record["window_s"])), "MLUPS")
    else:
        require(record["mlups"] == "", "diagnostic timing cannot produce formal MLUPS")
        total = number(record, "critical_local_s")
        require(total > 0 and 0 <= int(record["critical_rank"]) < ranks, "invalid critical rank/total")
        stages = rows(artifact(path, "stages"))
        expected = {("level", lev, stage) for lev in range(expected_levels) for stage in LEVEL_STAGES}
        expected |= {("pair", lev, stage) for lev in range(expected_levels-1) for stage in PAIR_STAGES}
        expected.add(("hierarchy", -1, "management"))
        require(len(stages) == len(expected) and
                {(row["owner"], int(row["level"]), row["stage"]) for row in stages} == expected,
                "stage ownership differs")
        for item in stages:
            lev, owner, stage = int(item["level"]), item["owner"], item["stage"]
            seconds, calls = number(item, "critical_s"), int(item["calls"])
            require(seconds >= 0 and number(item, "rank_max_s")+1e-13 >= seconds, "invalid stage time")
            expected_calls = count*2 if owner == "hierarchy" else count*2**lev
            if owner == "level" and stage == "management": expected_calls *= 2
            if owner == "level" and stage == "boundary_pre" and lev > 0: expected_calls = 0
            require(calls == expected_calls and int(item["rank_max_calls"]) == calls, "stage call count differs")
            close(item["share_percent"], 100*seconds/total, "critical-rank share")
            if calls: close(item["mean_call_s"], seconds/calls, "mean call")
            else: require(item["mean_call_s"] == "" and seconds == 0, "uncalled stage has measurements")
            if owner == "level": close(item["mean_level_step_s"], seconds/(count*2**lev), "mean layer step")
            else: require(item["mean_level_step_s"] == "", "pair/hierarchy is not a level step")
        require(sum(float(item["critical_s"]) for item in stages) <= total+1e-10, "exclusive times exceed local total")
    resources = [rows(artifact(path, "rank-"+str(rank)))[0] for rank in range(ranks)]
    used = set()
    for rank, resource in enumerate(resources):
        cpus = cpu_set(resource["cpuset"])
        width = 1 if record["backend"] == "cpu" else (8 if ranks == 1 else 3)
        require(int(resource["rank"]) == rank and resource["omp_threads"] == "1" and len(cpus) == width and
                not cpus & used, "rank CPU placement differs")
        used |= cpus
        require(number(resource, "host_process_peak_rss_kib") > 0 and
                resource["device_peak_status"] == "not_measured", "memory scope differs")
        if record["backend"] == "cuda":
            require(resource["device_count"] == "1" and resource["device_visible"].startswith("MIG-") and
                    "1g.5gb" in resource["device_name"], "not one expected MIG per rank")
            require(0 < number(resource, "after_init_used_bytes") <= number(resource, "device_total_bytes") and
                    0 < number(resource, "after_window_used_bytes") <= number(resource, "device_total_bytes"), "invalid device snapshots")
    if record["backend"] == "cuda":
        visible = {resource["device_visible"] for resource in resources}
        require(len(visible) == ranks, "duplicate MIG allocation")
        parent = None
        parents = {}
        for line in (path.parent/"nvidia-smi.txt").read_text().splitlines():
            if line.startswith("GPU "): parent = re.search(r"UUID: (GPU-[^)]+)", line).group(1)
            if "MIG " in line and "UUID:" in line:
                uuid = re.search(r"UUID: (MIG-[^)]+)", line).group(1)
                parents[uuid] = parent
        require(visible <= parents.keys() and len({parents[uuid] for uuid in visible}) == 1 and
                None not in {parents[uuid] for uuid in visible}, "MIGs do not share one recorded parent GPU")
        record["gpu_parent_uuid"] = parents[next(iter(visible))]
    record["levels"] = levels
    record["stages"] = stages
    record["boxes"] = boxes
    record["resources"] = resources
    record["path"] = str(path)
    record["uniform_step_ratio"] = step_ratio
    return record


def add_amrex_arguments(parser):
    parser.add_argument("--expected-amrex-source-commit", required=True)
    parser.add_argument("--expected-amrex-prefix", required=True, action="append",
                        help="independently approved full prefix; repeat for CPU/CUDA packages")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--source")
    add_amrex_arguments(parser)
    args = parser.parse_args()
    try:
        record = validate(args.csv, args.expected_amrex_source_commit, args.expected_amrex_prefix, args.source)
        if record["finite_final_state"] == "0":
            print("exit=1 nonfinite final state"); return 1
        print("exit=0 valid performance artifact; no speed threshold")
        return 0
    except (OSError, ValueError, KeyError, IndexError, AttributeError, ValidationError) as error:
        print("exit=2 "+str(error)); return 2


if __name__ == "__main__":
    raise SystemExit(main())
