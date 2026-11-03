"""Shared fixed-matrix checks for the two GPU summary consumers."""

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from check_gpu_performance_csv import ValidationError, validate_path


WORKLOAD_ORDER = ["tgv_d2q9", "tgv_d3q27", "dfg_d2q9", "dfg_d3q27"]
LAYOUT_ORDER = ["1mig", "4mig"]
GROUP_ORDER = [
    (workload, layout)
    for workload in WORKLOAD_ORDER
    for layout in LAYOUT_ORDER
]
IDENTITY_FIELDS = [
    "measurement_mode",
    "run_mode",
    "workload",
    "descriptor",
    "layout",
    "git_commit",
    "amrex_source_commit",
    "amrex_prefix",
    "amrex_config_sha256",
    "amrex_build_type",
    "build_type",
    "compiler",
    "nvcc",
    "cuda_toolkit",
    "cuda_runtime",
    "driver",
    "mpi",
    "mpi_cuda_build_version",
    "mpi_cuda_support_reported",
    "mpi_cuda_query_support",
    "partition",
    "qos",
    "grid_x",
    "grid_y",
    "grid_z",
    "global_valid_cells",
    "max_grid_size",
    "box_count",
    "mpi_ranks",
    "mig_instance_count",
    "parent_gpu_count",
    "boxes_per_rank",
    "omp_threads",
    "amrex_use_gpu_aware_mpi",
    "communication_mode",
    "binding_policy",
    "mig_profile",
    "num_devices_used",
    "num_device_partners",
    "warmup_steps",
    "sample_start",
    "sample_end",
    "nsteps",
]
GLOBAL_FIELDS = [
    "git_commit",
    "amrex_source_commit",
    "amrex_prefix",
    "amrex_config_sha256",
    "amrex_build_type",
    "build_type",
    "compiler",
    "nvcc",
    "cuda_toolkit",
    "cuda_runtime",
    "driver",
    "mpi",
    "mpi_cuda_build_version",
    "mpi_cuda_support_reported",
    "mpi_cuda_query_support",
    "partition",
    "qos",
    "omp_threads",
    "amrex_use_gpu_aware_mpi",
    "communication_mode",
    "binding_policy",
    "mig_profile",
]
REPEAT_PROVENANCE_FIELDS = [
    "job_id",
    "node",
    "rank_device_map",
    "slurm_job_gpus",
    "cpu_cpuset",
    "parent_gpu_uuids",
]
FORMAL_AMREX_USE_GPU_AWARE_MPI = "0"
FORMAL_COMMUNICATION_MODE = "pinned_staging"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def load_fixed_matrix(
    paths: Iterable[Path], measurement_mode: str, expected: Dict[str, str]
) -> List[Tuple[Tuple[str, str], List[Dict[str, str]]]]:
    records = [validate_path(path, expected) for path in paths]
    require(len(records) == 24, f"expected 24 explicit raw records, found {len(records)}")
    require(
        all(record["measurement_mode"] == measurement_mode for record in records),
        "summary input contains the wrong measurement mode",
    )
    require(all(record["run_mode"] == "baseline" for record in records), "summary accepts baseline records only")
    require(
        all(
            record["amrex_use_gpu_aware_mpi"]
            == FORMAL_AMREX_USE_GPU_AWARE_MPI
            and record["communication_mode"] == FORMAL_COMMUNICATION_MODE
            for record in records
        ),
        "formal GPU summaries require the frozen pinned-staging identity",
    )

    grouped = {key: [] for key in GROUP_ORDER}
    for record in records:
        key = (record["workload"], record["layout"])
        require(key in grouped, f"unexpected workload/layout {key}")
        grouped[key].append(record)

    shared = records[0]
    for record in records[1:]:
        for field in GLOBAL_FIELDS:
            require(record[field] == shared[field], f"matrix context drift in {field}")

    ordered = []
    for key in GROUP_ORDER:
        group = grouped[key]
        require(len(group) == 3, f"{key}: expected three repeats")
        require({int(record["repeat"]) for record in group} == {1, 2, 3}, f"{key}: repeats must be 1, 2 and 3")
        reference = group[0]
        for record in group[1:]:
            for field in IDENTITY_FIELDS:
                require(record[field] == reference[field], f"{key}: identity drift in {field}")
        ordered.append((key, group))
    return ordered


def repeat_provenance(records: List[Dict[str, str]]) -> Dict[str, str]:
    values = {}
    for record in sorted(records, key=lambda item: int(item["repeat"])):
        repeat = record["repeat"]
        for field in REPEAT_PROVENANCE_FIELDS:
            values["repeat{}_{}".format(repeat, field)] = record[field]
    return values
