#!/usr/bin/env python3
"""Compare one explicit throughput/stage-breakdown A/B pair."""

import argparse
from performance_expectations import GPU_FIELDS, add_expected_arguments, expected_values
import csv
from pathlib import Path

from check_gpu_performance_csv import ValidationError, validate_path


IDENTITY_FIELDS = [
    "workload",
    "descriptor",
    "layout",
    "git_commit",
    "amrex_source_commit",
    "amrex_prefix",
    "amrex_config_sha256",
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


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("throughput", type=Path)
    parser.add_argument("stage_breakdown", type=Path)
    add_expected_arguments(parser, GPU_FIELDS)
    args = parser.parse_args()
    expected = expected_values(args, GPU_FIELDS)
    try:
        throughput = validate_path(args.throughput, expected)
        breakdown = validate_path(args.stage_breakdown, expected)
        require(throughput["measurement_mode"] == "throughput", "first record is not throughput")
        require(
            breakdown["measurement_mode"] == "synchronized_stage_breakdown",
            "second record is not stage breakdown",
        )
        require(
            throughput["run_mode"] == breakdown["run_mode"] == "smoke",
            "A/B comparison accepts smoke records only",
        )
        for field in IDENTITY_FIELDS:
            require(throughput[field] == breakdown[field], f"A/B identity drift in {field}")
        require(not args.output.exists(), f"refusing to overwrite {args.output}")
        throughput_seconds = float(throughput["total_s_per_step"])
        breakdown_seconds = float(breakdown["critical_rank_total_s_per_step"])
        row = {
            "schema_version": 1,
            "workload": throughput["workload"],
            "layout": throughput["layout"],
            "git_commit": throughput["git_commit"],
            "mpi_ranks": throughput["mpi_ranks"],
            "mig_instance_count": throughput["mig_instance_count"],
            "amrex_use_gpu_aware_mpi": throughput["amrex_use_gpu_aware_mpi"],
            "throughput_job_id": throughput["job_id"],
            "throughput_node": throughput["node"],
            "throughput_parent_gpu_uuids": throughput["parent_gpu_uuids"],
            "stage_breakdown_job_id": breakdown["job_id"],
            "stage_breakdown_node": breakdown["node"],
            "stage_breakdown_parent_gpu_uuids": breakdown["parent_gpu_uuids"],
            "throughput_s_per_step": throughput_seconds,
            "throughput_mlups": float(throughput["mlups"]),
            "stage_breakdown_s_per_step": breakdown_seconds,
            "stage_breakdown_diagnostic_mlups": float(breakdown["diagnostic_mlups"]),
            "stage_breakdown_slowdown": breakdown_seconds / throughput_seconds,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
    except ValidationError as exc:
        print(f"FAIL: {exc}")
        return 1
    except (OSError, csv.Error, ZeroDivisionError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(f"gpu_timing_ab passed output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
