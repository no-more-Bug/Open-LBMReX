#!/usr/bin/env python3
"""Compare one explicit 4-MIG TGV GPU-aware MPI OFF/ON smoke pair."""

import argparse
from performance_expectations import GPU_FIELDS, add_expected_arguments, expected_values
import math
from pathlib import Path

from check_gpu_performance_csv import (
    ValidationError,
    parse_population_component_sums,
    validate_path,
)


IDENTITY_FIELDS = [
    "measurement_mode",
    "run_mode",
    "workload",
    "descriptor",
    "layout",
    "repeat",
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
    "mig_instance_count",
    "parent_gpu_count",
    "boxes_per_rank",
    "omp_threads",
    "rank_device_map",
    "slurm_job_gpus",
    "cpu_cpuset",
    "binding_policy",
    "mig_profile",
    "parent_gpu_uuids",
    "num_devices_used",
    "num_device_partners",
    "warmup_steps",
    "sample_start",
    "sample_end",
    "nsteps",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def same_result(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=5.0e-13, abs_tol=5.0e-13)


def relative_delta(lhs: float, rhs: float) -> float:
    return abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--off-job-id", required=True)
    parser.add_argument("--on-job-id", required=True)
    parser.add_argument("off_record", type=Path)
    parser.add_argument("on_record", type=Path)
    add_expected_arguments(parser, GPU_FIELDS)
    args = parser.parse_args()
    expected = expected_values(args, GPU_FIELDS)
    try:
        off = validate_path(args.off_record, expected)
        on = validate_path(args.on_record, expected)
        require(
            off["measurement_mode"] == on["measurement_mode"] == "throughput",
            "communication comparison requires throughput records",
        )
        require(
            off["run_mode"] == on["run_mode"] == "smoke",
            "communication comparison accepts smoke records only",
        )
        require(
            off["workload"] == on["workload"] == "tgv_d2q9" and
            off["layout"] == on["layout"] == "4mig",
            "communication comparison requires the 4-MIG TGV D2Q9 identity",
        )
        require(
            off["job_id"] == args.off_job_id and
            on["job_id"] == args.on_job_id,
            "communication job id does not match its raw record",
        )
        require(
            off["amrex_use_gpu_aware_mpi"] == "0" and
            off["communication_mode"] == "pinned_staging",
            "OFF record is not the pinned-staging identity",
        )
        require(
            on["amrex_use_gpu_aware_mpi"] == "1" and
            on["communication_mode"] == "device_buffer" and
            on["mpi_cuda_query_support"] == "true",
            "ON record is not the device-buffer identity",
        )
        for field in IDENTITY_FIELDS:
            require(off[field] == on[field], f"OFF/ON identity drift in {field}")

        off_components = parse_population_component_sums(off, args.off_record)
        on_components = parse_population_component_sums(on, args.on_record)
        require(
            all(
                same_result(off_value, on_value)
                for off_value, on_value in zip(off_components, on_components)
            ),
            "OFF/ON final population component sums differ",
        )
        off_l2 = float(off["final_population_l2"])
        on_l2 = float(on["final_population_l2"])
        require(
            same_result(off_l2, on_l2),
            "OFF/ON final population L2 values differ",
        )
        max_component_delta = max(
            relative_delta(off_value, on_value)
            for off_value, on_value in zip(off_components, on_components)
        )
        print(
            "gpu_communication_pair passed "
            f"max_component_relative_delta={max_component_delta:.17g} "
            f"l2_relative_delta={relative_delta(off_l2, on_l2):.17g}"
        )
    except ValidationError as exc:
        print(f"FAIL: {exc}")
        return 1
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
