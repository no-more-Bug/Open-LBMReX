#!/usr/bin/env python3
"""Generate fixed synthetic GPU records and exercise their narrow consumers."""

import csv
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Dict, List

from check_gpu_performance_csv import (
    CRITICAL_PER_STEP_FIELDS,
    CRITICAL_WINDOW_FIELDS,
    IDENTITIES,
    INDEPENDENT_MAX_FIELDS,
    SHARE_FIELDS,
    STAGE_BREAKDOWN_COLUMNS,
    THROUGHPUT_COLUMNS,
)


# Independent synthetic package/platform values, not imported checker defaults.
EXPECTED_CONTEXT = {
    "amrex_source_commit": "4ceb814db7d79c5858b13dab153fb04ecc3b129d",
    "amrex_prefix": "/opt/test-amrex/cuda",
    "amrex_config_sha256": "c" * 64,
    "compiler": "GNU-13.2.0",
    "nvcc": "12.4.131",
    "cuda_toolkit": "12.4.131",
    "mpi": "test-openmpi",
    "mpi_cuda_build_version": "12.4",
    "partition": "test-gpu",
    "qos": "test-qos",
    "mig_profile": "1g.5gb",
}
EXPECTED_ARGS = [
    "--expected-" + field.replace("_", "-") + "=" + value
    for field, value in EXPECTED_CONTEXT.items()
]


WORKLOADS = ["tgv_d2q9", "tgv_d3q27", "dfg_d2q9", "dfg_d3q27"]


def common(workload: str, layout: str, repeat: int, nsteps: int) -> Dict[str, str]:
    descriptor, grid, max_grid_size, box_count = IDENTITIES[workload]
    population_count = 9 if descriptor == "D2Q9" else 27
    ranks = 1 if layout == "1mig" else 4
    rank_map = ";".join(
        "r{0}=global_rank={0}|local_rank={0}|hostname=gpu-test|"
        "cuda_visible_devices=MIG-{0}|slurm_step_gpus=unset|"
        "allocation_cuda_visible_devices={3}|device_assignment_source={4}|"
        "cuda_device_count=1|cuda_device_id=0|amrex_device_id=0|"
        "device_name=NVIDIA_A100_MIG|visible_mig_uuid=MIG-{0}|"
        "cuda_parent_uuid={1}|"
        "compute_capability=8.0|device_total_memory=5368709120|"
        "pci_bus_id=0000:00:00.0|cpu_cpuset={0}|"
        "num_devices_used=1|num_device_partners={2}".format(
            rank,
            "1" * 32,
            ranks,
            ",".join("MIG-{}".format(index) for index in range(ranks)),
            "slurm-task-local-cvd",
        )
        for rank in range(ranks)
    )
    cpuset = ";".join(f"r{rank}={rank}" for rank in range(ranks))
    return {
        **EXPECTED_CONTEXT,
        "schema_version": "2",
        "run_mode": "baseline",
        "workload": workload,
        "descriptor": descriptor,
        "layout": layout,
        "repeat": str(repeat),
        "git_commit": "a" * 40,
        "amrex_build_type": "Release",
        "build_type": "Release",
        "cuda_runtime": "12.2",
        "driver": "12.2",
        "mpi_cuda_support_reported": "true",
        "mpi_cuda_query_support": "true",
        "job_id": str(1000 + repeat),
        "node": "gpu-test",
        "grid_x": str(grid[0]),
        "grid_y": str(grid[1]),
        "grid_z": str(grid[2]),
        "global_valid_cells": str(grid[0] * grid[1] * grid[2]),
        "max_grid_size": str(max_grid_size),
        "box_count": str(box_count),
        "mpi_ranks": str(ranks),
        "mig_instance_count": str(ranks),
        "parent_gpu_count": "1",
        "boxes_per_rank": str(box_count // ranks),
        "omp_threads": "1",
        "amrex_use_gpu_aware_mpi": "0",
        "communication_mode": "pinned_staging",
        "rank_device_map": rank_map,
        "slurm_job_gpus": ",".join(str(rank) for rank in range(ranks)),
        "cpu_cpuset": cpuset,
        "binding_policy": "srun_one-task-per-mig",
        "parent_gpu_uuids": "GPU-11111111-1111-1111-1111-111111111111",
        "num_devices_used": "1",
        "num_device_partners": str(ranks),
        "warmup_steps": "100",
        "sample_start": "101",
        "sample_end": str(100 + nsteps),
        "nsteps": str(nsteps),
        "final_population_component_sums": ";".join(
            "c{}={}".format(population, 1000 + population)
            for population in range(population_count)
        ),
        "final_population_l2": "100.0",
    }


def write_row(path: Path, columns: List[str], row: Dict[str, str]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerow(row)


def throughput_row(workload: str, layout: str, repeat: int) -> Dict[str, str]:
    nsteps = 1000 if layout == "1mig" else 2000
    row = common(workload, layout, repeat, nsteps)
    window = 600.0 + repeat
    cells = int(row["global_valid_cells"])
    row.update({
        "measurement_mode": "throughput",
        "synchronized_throughput_window_s": str(window),
        "total_s_per_step": str(window / nsteps),
        "mlups": str(cells * nsteps / window / 1.0e6),
        "formal_mlups": "true",
    })
    return row


def stage_row(workload: str, layout: str, repeat: int) -> Dict[str, str]:
    nsteps = 100 if layout == "1mig" else 200
    row = common(workload, layout, repeat, nsteps)
    critical = [10.0, 20.0, 15.0, 2.0, 3.0, 15.0, 2.0, 3.0, 10.0, 5.0, 60.0]
    row["measurement_mode"] = "synchronized_stage_breakdown"
    row["critical_rank"] = "0"
    for name, value in zip(CRITICAL_WINDOW_FIELDS, critical):
        row[name] = str(value)
    for name, source in zip(
        CRITICAL_PER_STEP_FIELDS,
        [critical[index] for index in (0, 1, 5, 8, 9, 10)],
    ):
        row[name] = str(source / nsteps)
    for name, value in zip(INDEPENDENT_MAX_FIELDS, critical):
        row[name] = str(value)
    row.update({
        "collision_share_pct": str(100.0 * 10.0 / 60.0),
        "pre_stream_share_pct": str(100.0 * 20.0 / 60.0),
        "streaming_share_pct": str(100.0 * 15.0 / 60.0),
        "post_stream_share_pct": str(100.0 * 10.0 / 60.0),
        "diagnostics_share_pct": str(100.0 * 5.0 / 60.0),
        "other_or_timing_overhead_share_pct": "0.0",
        "fill_boundary_share_of_pre_stream_pct": "75.0",
        "domain_share_of_post_stream_pct": "20.0",
        "internal_solid_share_of_post_stream_pct": "30.0",
        "diagnostic_synchronized_window_s": str(60.0 + repeat),
        "diagnostic_mlups": str(
            int(row["global_valid_cells"]) * nsteps / (60.0 + repeat) / 1.0e6
        ),
        "formal_mlups": "false",
    })
    return row


def run(expected: int, *command: str) -> None:
    completed = subprocess.run(command, check=False)
    if completed.returncode != expected:
        raise RuntimeError(
            f"expected exit {expected}, got {completed.returncode}: {command}"
        )


def prepare_workdir(candidate: Path) -> Path:
    current = Path.cwd().resolve()
    if candidate.name != "artifacts" or candidate.parent.resolve() != current:
        raise ValueError("workdir must be the artifacts child of the current directory")
    workdir = current / "artifacts"
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir()
    return workdir


def main() -> int:
    if len(sys.argv) != 2:
        raise ValueError("expected one artifacts workdir")
    sentinel = Path.cwd() / "consumer-path-sentinel"
    if sentinel.exists():
        raise ValueError("refusing to overwrite workdir safety sentinel")
    sentinel.write_text("preserve\n")
    try:
        try:
            prepare_workdir(Path.cwd())
        except ValueError:
            pass
        else:
            raise RuntimeError("unsafe workdir was accepted")
        if sentinel.read_text() != "preserve\n":
            raise RuntimeError("unsafe workdir check changed its sentinel")
    finally:
        sentinel.unlink()
    workdir = prepare_workdir(Path(sys.argv[1]))
    script_dir = Path(__file__).resolve().parent
    forbidden_python36_tokens = tuple(
        left + right for left, right in (
            ("di", "ct["), ("li", "st["), ("tu", "ple["),
            ("se", "t["), (".remove", "suffix("),
            (".remove", "prefix("), ("math.", "prod("),
        )
    )
    for script in script_dir.glob("*.py"):
        source = script.read_text()
        for token in forbidden_python36_tokens:
            if token in source:
                raise RuntimeError(
                    "{} uses Python >3.6 syntax/API: {}".format(script, token)
                )

    throughput_paths = []
    stage_paths = []
    for workload in WORKLOADS:
        for layout in ("1mig", "4mig"):
            for repeat in (1, 2, 3):
                throughput = workdir / f"throughput-{workload}-{layout}-r{repeat}.csv"
                stage = workdir / f"stage-{workload}-{layout}-r{repeat}.csv"
                write_row(
                    throughput,
                    THROUGHPUT_COLUMNS,
                    throughput_row(workload, layout, repeat),
                )
                write_row(
                    stage,
                    STAGE_BREAKDOWN_COLUMNS,
                    stage_row(workload, layout, repeat),
                )
                throughput_paths.append(throughput)
                stage_paths.append(stage)

    checker = script_dir / "check_gpu_performance_csv.py"
    run(0, sys.executable, str(checker), *EXPECTED_ARGS, *(str(path) for path in throughput_paths + stage_paths))

    for field in EXPECTED_CONTEXT:
        mismatched = throughput_row("tgv_d2q9", "1mig", 1)
        mismatched[field] = "d" * len(mismatched[field]) if field in {"amrex_source_commit", "amrex_config_sha256"} else "/different/package" if field == "amrex_prefix" else "different"
        mismatch_path = workdir / ("mismatched-" + field + ".csv")
        write_row(mismatch_path, THROUGHPUT_COLUMNS, mismatched)
        run(1, sys.executable, str(checker), *EXPECTED_ARGS, str(mismatch_path))
    run(2, sys.executable, str(checker), str(throughput_paths[0]))

    invalid = workdir / "invalid.csv"
    bad = throughput_row("tgv_d2q9", "1mig", 1)
    bad["mlups"] = "-1"
    write_row(invalid, THROUGHPUT_COLUMNS, bad)
    run(1, sys.executable, str(checker), *EXPECTED_ARGS, str(invalid))

    bad_mapping = workdir / "invalid-duplicate-device.csv"
    bad = throughput_row("tgv_d2q9", "4mig", 1)
    bad["rank_device_map"] = bad["rank_device_map"].replace(
        "cuda_visible_devices=MIG-1",
        "cuda_visible_devices=MIG-0",
        1,
    ).replace(
        "visible_mig_uuid=MIG-1",
        "visible_mig_uuid=MIG-0",
        1,
    )
    write_row(bad_mapping, THROUGHPUT_COLUMNS, bad)
    run(1, sys.executable, str(checker), *EXPECTED_ARGS, str(bad_mapping))

    bad_rank = workdir / "invalid-local-rank.csv"
    bad = throughput_row("tgv_d2q9", "4mig", 1)
    bad["rank_device_map"] = bad["rank_device_map"].replace(
        "r1=global_rank=1|local_rank=1|",
        "r1=global_rank=1|local_rank=0|",
    )
    write_row(bad_rank, THROUGHPUT_COLUMNS, bad)
    run(1, sys.executable, str(checker), *EXPECTED_ARGS, str(bad_rank))

    bad_source = workdir / "invalid-device-source.csv"
    bad = throughput_row("tgv_d2q9", "4mig", 1)
    bad["rank_device_map"] = bad["rank_device_map"].replace(
        "device_assignment_source=slurm-task-local-cvd",
        "device_assignment_source=unknown",
        1,
    )
    write_row(bad_source, THROUGHPUT_COLUMNS, bad)
    run(1, sys.executable, str(checker), *EXPECTED_ARGS, str(bad_source))

    bad_allocation = workdir / "invalid-allocation-device-list.csv"
    bad = throughput_row("tgv_d2q9", "4mig", 1)
    bad["rank_device_map"] = bad["rank_device_map"].replace(
        "allocation_cuda_visible_devices=MIG-0,MIG-1,MIG-2,MIG-3",
        "allocation_cuda_visible_devices=MIG-0,MIG-x,MIG-2,MIG-3",
        1,
    )
    write_row(bad_allocation, THROUGHPUT_COLUMNS, bad)
    run(1, sys.executable, str(checker), *EXPECTED_ARGS, str(bad_allocation))

    bad_parent = workdir / "invalid-parent-device.csv"
    bad = throughput_row("tgv_d2q9", "4mig", 1)
    bad["parent_gpu_uuids"] = "GPU-22222222-2222-2222-2222-222222222222"
    write_row(bad_parent, THROUGHPUT_COLUMNS, bad)
    run(1, sys.executable, str(checker), *EXPECTED_ARGS, str(bad_parent))

    valid_lines = throughput_paths[0].read_text().splitlines()
    extra_field = workdir / "invalid-extra-field.csv"
    extra_field.write_text("{}\n{},extra\n".format(valid_lines[0], valid_lines[1]))
    run(1, sys.executable, str(checker), *EXPECTED_ARGS, str(extra_field))
    missing_field = workdir / "invalid-missing-field.csv"
    missing_field.write_text(
        "{}\n{}\n".format(valid_lines[0], valid_lines[1].rsplit(",", 1)[0])
    )
    run(1, sys.executable, str(checker), *EXPECTED_ARGS, str(missing_field))
    run(2, sys.executable, str(checker), *EXPECTED_ARGS, str(workdir / "missing.csv"))

    communication_off = throughput_row("tgv_d2q9", "4mig", 1)
    communication_off.update({
        "run_mode": "smoke",
        "job_id": "2000",
        "warmup_steps": "0",
        "sample_start": "1",
        "sample_end": "20",
        "nsteps": "20",
        "synchronized_throughput_window_s": "20.0",
        "total_s_per_step": "1.0",
        "mlups": str(
            int(communication_off["global_valid_cells"]) / 1.0e6
        ),
    })
    communication_on = dict(communication_off)
    communication_on.update({
        "job_id": "2001",
        "amrex_use_gpu_aware_mpi": "1",
        "communication_mode": "device_buffer",
    })
    communication_off_path = workdir / "communication-off.csv"
    communication_on_path = workdir / "communication-on.csv"
    write_row(
        communication_off_path, THROUGHPUT_COLUMNS, communication_off)
    write_row(
        communication_on_path, THROUGHPUT_COLUMNS, communication_on)
    communication_checker = script_dir / "check_gpu_communication_pair.py"
    run(
        0,
        sys.executable,
        str(communication_checker), *EXPECTED_ARGS,
        "--off-job-id",
        "2000",
        "--on-job-id",
        "2001",
        str(communication_off_path),
        str(communication_on_path),
    )
    unsupported_on = dict(communication_on)
    unsupported_on["mpi_cuda_query_support"] = "false"
    unsupported_on_path = workdir / "communication-on-unsupported.csv"
    write_row(
        unsupported_on_path, THROUGHPUT_COLUMNS, unsupported_on)
    run(
        1,
        sys.executable,
        str(checker), *EXPECTED_ARGS,
        str(unsupported_on_path),
    )
    run(
        1,
        sys.executable,
        str(communication_checker), *EXPECTED_ARGS,
        "--off-job-id",
        "2000",
        "--on-job-id",
        "2001",
        str(communication_off_path),
        str(unsupported_on_path),
    )
    inconsistent_on = dict(communication_on)
    inconsistent_on["final_population_component_sums"] = (
        inconsistent_on["final_population_component_sums"].replace(
            "c0=1000", "c0=1001", 1
        )
    )
    inconsistent_on_path = workdir / "communication-on-inconsistent.csv"
    write_row(
        inconsistent_on_path, THROUGHPUT_COLUMNS, inconsistent_on)
    run(
        1,
        sys.executable,
        str(communication_checker), *EXPECTED_ARGS,
        "--off-job-id",
        "2000",
        "--on-job-id",
        "2001",
        str(communication_off_path),
        str(inconsistent_on_path),
    )

    mapping_checker = script_dir / "check_gpu_mapping_log.py"
    launcher = script_dir / "hpc-slurm/gpu-rank-launch.sh"
    allowed_cpus = sorted(os.sched_getaffinity(0))
    fixture_cpu_start = next(
        (
            cpu for cpu in allowed_cpus
            if all(cpu + offset in allowed_cpus for offset in range(4))
        ),
        None,
    )
    if fixture_cpu_start is None:
        raise RuntimeError("launcher fixture requires four contiguous CPUs")
    fixture_cpuset = "{}-{}".format(
        fixture_cpu_start, fixture_cpu_start + 3
    )
    launcher_environment = os.environ.copy()
    launcher_environment.update({
        "SLURM_PROCID": "1",
        "SLURM_LOCALID": "1",
        "SLURM_NTASKS": "4",
        "SLURM_CPUS_PER_TASK": "1",
        "CUDA_VISIBLE_DEVICES": "MIG-b",
        "LBM_GPU_ALLOCATION_CUDA_VISIBLE_DEVICES":
            "MIG-a,MIG-b,MIG-c,MIG-d",
        "LBM_GPU_ALLOCATION_CPUSET": fixture_cpuset,
        "SLURM_JOB_GPUS": "0,1,2,3",
    })
    launcher_environment.pop("SLURM_STEP_GPUS", None)
    launcher_environment.pop("LBM_REPO_DIR", None)
    launcher_check = subprocess.run(
        [
            str(launcher),
            shutil.which("printenv"),
            "CUDA_VISIBLE_DEVICES",
            "SLURM_LOCALID",
            "LBM_GPU_ALLOCATION_CUDA_VISIBLE_DEVICES",
            "LBM_GPU_DEVICE_ASSIGNMENT_SOURCE",
        ],
        check=False,
        env=launcher_environment,
        stdout=subprocess.PIPE,
        universal_newlines=True,
    )
    require_launcher_lines = launcher_check.stdout.splitlines()
    if launcher_check.returncode != 0 or require_launcher_lines[-4:] != [
        "MIG-b",
        "1",
        "MIG-a,MIG-b,MIG-c,MIG-d",
        "slurm-task-local-cvd",
    ] or "global_rank=1 local_rank=1" not in require_launcher_lines[0] \
            or "slurm_step_gpus=unset" not in require_launcher_lines[0] \
            or "cpuset={}".format(fixture_cpu_start + 1) \
            not in require_launcher_lines[0]:
        raise RuntimeError("Slurm rank/device launcher fixture failed")

    # The same public launcher also supports a noncontiguous Slurm allocation.
    launcher_environment.update({
        "SLURM_NTASKS": "2",
        "LBM_GPU_ALLOCATION_CUDA_VISIBLE_DEVICES": "MIG-a,MIG-b",
        "LBM_GPU_ALLOCATION_CPUSET": "{},{}".format(fixture_cpu_start, fixture_cpu_start+2),
    })
    noncontiguous = subprocess.run(
        [str(launcher), shutil.which("printenv"), "CUDA_VISIBLE_DEVICES"],
        check=False, env=launcher_environment, stdout=subprocess.PIPE, universal_newlines=True,
    )
    if noncontiguous.returncode != 0 or noncontiguous.stdout.splitlines()[-1] != "MIG-b" or \
            "cpuset={}".format(fixture_cpu_start+2) not in noncontiguous.stdout.splitlines()[0]:
        raise RuntimeError("Noncontiguous rank CPU fixture failed")

    mapping_lines = []
    for rank in range(4):
        mapping_lines.append(
            "gpu_rank_launch global_rank={0} local_rank={0} "
            "cuda_visible_devices=MIG-{0} "
            "allocation_cuda_visible_devices=MIG-0,MIG-1,MIG-2,MIG-3 "
            "device_assignment_source=slurm-task-local-cvd "
            "slurm_step_gpus=unset "
            "cpuset={0} memset=0".format(rank)
        )
        mapping_lines.append(
            "gpu_mapping_status=pass global_rank={0} local_rank={0} "
            "amrex_local_rank={0} "
            "ompi_local_rank={0} cuda_visible_devices=MIG-{0} "
            "slurm_job_gpus=0,1,2,3 slurm_step_gpus=unset "
            "visible_device_count=1 cuda_device_id=0 amrex_device_id=0 "
            "num_devices_used=1 num_device_partners=4 device_name=NVIDIA "
            "visible_mig_uuid=MIG-{0} cuda_parent_uuid={1} "
            "compute_capability=8.0 total_memory_bytes=5368709120".format(
                rank, "1" * 32
            )
        )
    mapping_lines.append("gpu_mapping_global_status=0")
    mapping_log = workdir / "mapping.log"
    mapping_log.write_text("\n".join(mapping_lines) + "\n")
    run(
        0, sys.executable, str(mapping_checker), "--expected-ranks", "4",
        "--expected-cpus-per-rank", "1",
        "--expected-parent-gpus", "1",
        "--expected-parent-uuid", "GPU-11111111-1111-1111-1111-111111111111",
        "--allocation-cpuset", "0-3", str(mapping_log),
    )
    overlapping_mapping = workdir / "mapping-overlap.log"
    overlapping_mapping.write_text(
        mapping_log.read_text().replace("cpuset=1 memset=0", "cpuset=0 memset=0")
    )
    run(
        1, sys.executable, str(mapping_checker), "--expected-ranks", "4",
        "--expected-cpus-per-rank", "1",
        "--expected-parent-gpus", "1",
        "--expected-parent-uuid", "GPU-11111111-1111-1111-1111-111111111111",
        "--allocation-cpuset", "0-3", str(overlapping_mapping),
    )
    run(
        1, sys.executable, str(mapping_checker), "--expected-ranks", "4",
        "--expected-cpus-per-rank", "2",
        "--expected-parent-gpus", "1",
        "--expected-parent-uuid", "GPU-11111111-1111-1111-1111-111111111111",
        "--allocation-cpuset", "0-3", str(mapping_log),
    )
    run(
        2, sys.executable, str(mapping_checker), "--expected-ranks", "4",
        "--expected-cpus-per-rank", "1",
        "--expected-parent-gpus", "1",
        "--expected-parent-uuid", "GPU-11111111-1111-1111-1111-111111111111",
        "--allocation-cpuset", "0-3", str(workdir / "missing-mapping.log"),
    )

    ab_throughput = throughput_row("tgv_d2q9", "1mig", 1)
    ab_throughput.update({
        "run_mode": "smoke",
        "warmup_steps": "0",
        "sample_start": "1",
        "sample_end": "20",
        "nsteps": "20",
        "synchronized_throughput_window_s": "20.0",
        "total_s_per_step": "1.0",
        "mlups": str(int(ab_throughput["global_valid_cells"]) * 20 / 20.0 / 1.0e6),
    })
    ab_stage = stage_row("tgv_d2q9", "1mig", 1)
    ab_stage.update({
        "run_mode": "smoke",
        "warmup_steps": "0",
        "sample_start": "1",
        "sample_end": "20",
        "nsteps": "20",
    })
    critical = [float(ab_stage[name]) for name in CRITICAL_WINDOW_FIELDS]
    for name, source in zip(
        CRITICAL_PER_STEP_FIELDS,
        [critical[index] for index in (0, 1, 5, 8, 9, 10)],
    ):
        ab_stage[name] = str(source / 20)
    ab_stage["diagnostic_mlups"] = str(
        int(ab_stage["global_valid_cells"]) * 20 /
        float(ab_stage["diagnostic_synchronized_window_s"]) / 1.0e6
    )
    ab_throughput_path = workdir / "ab-throughput.csv"
    ab_stage_path = workdir / "ab-stage.csv"
    write_row(ab_throughput_path, THROUGHPUT_COLUMNS, ab_throughput)
    write_row(ab_stage_path, STAGE_BREAKDOWN_COLUMNS, ab_stage)
    run(
        0,
        sys.executable,
        str(script_dir / "compare_gpu_timing_modes.py"), *EXPECTED_ARGS,
        "--output",
        str(workdir / "gpu_timing_ab.csv"),
        str(ab_throughput_path),
        str(ab_stage_path),
    )

    baseline = workdir / "gpu_baseline.csv"
    breakdown = workdir / "gpu_stage_breakdown.csv"
    run(
        0,
        sys.executable,
        str(script_dir / "summarize_gpu_baseline.py"), *EXPECTED_ARGS,
        "--output",
        str(baseline),
        *(str(path) for path in throughput_paths),
    )
    run(
        0,
        sys.executable,
        str(script_dir / "summarize_gpu_stage_breakdown.py"), *EXPECTED_ARGS,
        "--output",
        str(breakdown),
        *(str(path) for path in stage_paths),
    )
    run(
        0,
        sys.executable,
        str(script_dir / "check_gpu_summary_csv.py"), *EXPECTED_ARGS,
        str(baseline),
        str(breakdown),
    )
    unknown_baseline = workdir / "unknown-compiler-baseline.csv"
    unknown_breakdown = workdir / "unknown-compiler-breakdown.csv"
    unknown_baseline.write_text(baseline.read_text().replace(EXPECTED_CONTEXT["compiler"], "unknown"))
    unknown_breakdown.write_text(breakdown.read_text().replace(EXPECTED_CONTEXT["compiler"], "unknown"))
    unknown_args = [value if not value.startswith("--expected-compiler=") else "--expected-compiler=unknown"
                    for value in EXPECTED_ARGS]
    run(1, sys.executable, str(script_dir / "check_gpu_summary_csv.py"),
        *unknown_args, str(unknown_baseline), str(unknown_breakdown))
    on_throughput_paths = []
    for workload in WORKLOADS:
        for layout in ("1mig", "4mig"):
            for repeat in (1, 2, 3):
                row = throughput_row(workload, layout, repeat)
                row["amrex_use_gpu_aware_mpi"] = "1"
                row["communication_mode"] = "device_buffer"
                path = workdir / "on-throughput-{}-{}-r{}.csv".format(
                    workload, layout, repeat
                )
                write_row(path, THROUGHPUT_COLUMNS, row)
                on_throughput_paths.append(path)
    run(
        1,
        sys.executable,
        str(script_dir / "summarize_gpu_baseline.py"), *EXPECTED_ARGS,
        "--output",
        str(workdir / "invalid-on-gpu-baseline.csv"),
        *(str(path) for path in on_throughput_paths),
    )
    invalid_summary = workdir / "invalid-gpu-baseline.csv"
    invalid_rows = baseline.read_text().splitlines()
    invalid_rows[1] = invalid_rows[1].replace(
        ",pinned_staging,", ",device_buffer,", 1
    )
    invalid_summary.write_text("\n".join(invalid_rows) + "\n")
    run(
        1,
        sys.executable,
        str(script_dir / "check_gpu_summary_csv.py"), *EXPECTED_ARGS,
        str(invalid_summary),
        str(breakdown),
    )
    run(
        2,
        sys.executable,
        str(script_dir / "check_gpu_summary_csv.py"), *EXPECTED_ARGS,
        str(workdir / "missing-gpu-baseline.csv"),
        str(breakdown),
    )
    run(
        0,
        sys.executable,
        str(script_dir / "write_gpu_aware_mpi_validation.py"),
        "--output",
        str(workdir / "gpu_aware_mpi_validation.csv"),
        "--git-commit",
        "a" * 40,
        "--off-job-id",
        "100",
        "--off-status",
        "pass",
        "--off-evidence",
        "/ignored/off",
        "--on-job-id",
        "101",
        "--on-status",
        "fail",
        "--on-evidence",
        "/ignored/on",
    )
    for path in (baseline, breakdown):
        with path.open(newline="") as stream:
            if len(list(csv.DictReader(stream))) != 8:
                raise RuntimeError(f"{path} does not contain eight rows")
    print("gpu performance consumers: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
