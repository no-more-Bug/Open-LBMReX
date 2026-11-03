#!/usr/bin/env python3
"""Validate explicit tests-owned GPU throughput or stage-breakdown records."""

import argparse
from performance_expectations import GPU_FIELDS, add_expected_arguments, expected_values
import csv
import math
from pathlib import Path
import re
from typing import Dict, List


IDENTITIES = {
    "tgv_d2q9": ("D2Q9", (1024, 1024, 1), 256, 16),
    "tgv_d3q27": ("D3Q27", (512, 512, 4), 128, 16),
    "dfg_d2q9": ("D2Q9", (1761, 329, 1), 224, 16),
    "dfg_d3q27": ("D3Q27", (881, 165, 4), 112, 16),
}

COMMON_COLUMNS = [
    "schema_version",
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
    "job_id",
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
    "amrex_use_gpu_aware_mpi",
    "communication_mode",
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

THROUGHPUT_COLUMNS = COMMON_COLUMNS + [
    "synchronized_throughput_window_s",
    "total_s_per_step",
    "mlups",
    "formal_mlups",
    "final_population_component_sums",
    "final_population_l2",
]

CRITICAL_WINDOW_FIELDS = [
    "critical_rank_collision_window_s",
    "critical_rank_pre_stream_window_s",
    "critical_rank_pre_stream_fill_boundary_window_s",
    "critical_rank_pre_stream_setup_window_s",
    "critical_rank_pre_stream_physical_ghosts_window_s",
    "critical_rank_streaming_window_s",
    "critical_rank_domain_post_stream_window_s",
    "critical_rank_internal_solid_reconstruction_window_s",
    "critical_rank_post_stream_window_s",
    "critical_rank_diagnostics_window_s",
    "critical_rank_total_window_s",
]
INDEPENDENT_MAX_FIELDS = [
    "independent_rank_max_collision_window_s",
    "independent_rank_max_pre_stream_window_s",
    "independent_rank_max_pre_stream_fill_boundary_window_s",
    "independent_rank_max_pre_stream_setup_window_s",
    "independent_rank_max_pre_stream_physical_ghosts_window_s",
    "independent_rank_max_streaming_window_s",
    "independent_rank_max_domain_post_stream_window_s",
    "independent_rank_max_internal_solid_reconstruction_window_s",
    "independent_rank_max_post_stream_window_s",
    "independent_rank_max_diagnostics_window_s",
    "independent_rank_max_total_window_s",
]
CRITICAL_PER_STEP_FIELDS = [
    "critical_rank_collision_s_per_step",
    "critical_rank_pre_stream_s_per_step",
    "critical_rank_streaming_s_per_step",
    "critical_rank_post_stream_s_per_step",
    "critical_rank_diagnostics_s_per_step",
    "critical_rank_total_s_per_step",
]
SHARE_FIELDS = [
    "collision_share_pct",
    "pre_stream_share_pct",
    "streaming_share_pct",
    "post_stream_share_pct",
    "diagnostics_share_pct",
    "other_or_timing_overhead_share_pct",
    "fill_boundary_share_of_pre_stream_pct",
    "domain_share_of_post_stream_pct",
    "internal_solid_share_of_post_stream_pct",
]
STAGE_BREAKDOWN_COLUMNS = (
    COMMON_COLUMNS
    + ["critical_rank"]
    + CRITICAL_WINDOW_FIELDS
    + CRITICAL_PER_STEP_FIELDS
    + INDEPENDENT_MAX_FIELDS
    + SHARE_FIELDS
    + [
        "diagnostic_synchronized_window_s",
        "diagnostic_mlups",
        "formal_mlups",
        "final_population_component_sums",
        "final_population_l2",
    ]
)


class ValidationError(Exception):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def integer(row: Dict[str, str], name: str, path: Path) -> int:
    try:
        return int(row[name])
    except ValueError as exc:
        raise ValidationError(f"{path}: {name} is not an integer") from exc


def number(row: Dict[str, str], name: str, path: Path) -> float:
    try:
        value = float(row[name])
    except ValueError as exc:
        raise ValidationError(f"{path}: {name} is not numeric") from exc
    require(math.isfinite(value), f"{path}: {name} is not finite")
    return value


def close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=1.0e-11, abs_tol=1.0e-12)


def parse_population_component_sums(
    row: Dict[str, str], path: Path
) -> List[float]:
    expected_count = 9 if row["descriptor"] == "D2Q9" else 27
    tokens = row["final_population_component_sums"].split(";")
    require(
        len(tokens) == expected_count,
        f"{path}: final population component count mismatch",
    )
    values = []
    for population, token in enumerate(tokens):
        name, separator, text = token.partition("=")
        require(
            separator == "=" and name == f"c{population}",
            f"{path}: malformed final population component sums",
        )
        try:
            value = float(text)
        except ValueError as exc:
            raise ValidationError(
                f"{path}: final population component sum is not numeric"
            ) from exc
        require(
            math.isfinite(value),
            f"{path}: final population component sum is not finite",
        )
        values.append(value)
    return values


def product(values: List[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def map_integer(fields: Dict[str, str], name: str, path: Path) -> int:
    try:
        return int(fields[name])
    except ValueError as exc:
        raise ValidationError(
            f"{path}: rank map field {name} is not an integer"
        ) from exc


def parse_rank_device_map(
    text: str, ranks: int, path: Path
) -> List[Dict[str, str]]:
    required = {
        "global_rank",
        "local_rank",
        "hostname",
        "cuda_visible_devices",
        "slurm_step_gpus",
        "allocation_cuda_visible_devices",
        "device_assignment_source",
        "cuda_device_count",
        "cuda_device_id",
        "amrex_device_id",
        "device_name",
        "visible_mig_uuid",
        "cuda_parent_uuid",
        "compute_capability",
        "device_total_memory",
        "pci_bus_id",
        "cpu_cpuset",
        "num_devices_used",
        "num_device_partners",
    }
    entries = text.split(";") if text else []
    require(len(entries) == ranks, f"{path}: rank map entry count mismatch")
    parsed = []
    for expected_rank, entry in enumerate(entries):
        prefix, separator, payload = entry.partition("=")
        require(separator == "=" and prefix == f"r{expected_rank}", f"{path}: rank map order mismatch")
        fields = {}
        for token in payload.split("|"):
            name, token_separator, value = token.partition("=")
            require(token_separator == "=" and name and name not in fields, f"{path}: malformed rank map token")
            fields[name] = value
        require(set(fields) == required, f"{path}: rank map fields mismatch")
        require(map_integer(fields, "global_rank", path) == expected_rank, f"{path}: global rank mismatch")
        require(map_integer(fields, "local_rank", path) == expected_rank, f"{path}: local rank mismatch")
        require(fields["hostname"] not in {"", "unset"}, f"{path}: hostname missing")
        require(fields["cuda_visible_devices"] not in {"", "unset"} and "," not in fields["cuda_visible_devices"], f"{path}: rank sees multiple/no CUDA devices")
        allocation_devices = fields["allocation_cuda_visible_devices"].split(",")
        require(len(allocation_devices) == ranks and all(allocation_devices), f"{path}: allocation CUDA device list mismatch")
        require(fields["device_assignment_source"] == "slurm-task-local-cvd", f"{path}: device assignment provenance invalid")
        require(fields["cuda_visible_devices"] in allocation_devices, f"{path}: rank device is outside allocation")
        require(map_integer(fields, "cuda_device_count", path) == 1, f"{path}: visible device count mismatch")
        require(map_integer(fields, "cuda_device_id", path) == 0, f"{path}: CUDA device id mismatch")
        require(map_integer(fields, "amrex_device_id", path) == 0, f"{path}: AMReX device id mismatch")
        require(fields["device_name"] not in {"", "unset"}, f"{path}: device name missing")
        require(re.fullmatch(r"MIG-[0-9a-f-]+", fields["visible_mig_uuid"]) is not None, f"{path}: visible MIG UUID invalid")
        require(fields["visible_mig_uuid"] == fields["cuda_visible_devices"], f"{path}: visible MIG/CUDA identity mismatch")
        require(re.fullmatch(r"[0-9a-f]{32}", fields["cuda_parent_uuid"]) is not None, f"{path}: CUDA parent UUID invalid")
        require(re.fullmatch(r"[0-9]+\.[0-9]+", fields["compute_capability"]) is not None, f"{path}: compute capability invalid")
        require(map_integer(fields, "device_total_memory", path) > 0, f"{path}: device memory invalid")
        require(fields["pci_bus_id"] not in {"", "unset"}, f"{path}: PCI identity missing")
        require(fields["cpu_cpuset"] not in {"", "unset"}, f"{path}: rank cpuset missing")
        require(map_integer(fields, "num_devices_used", path) == 1, f"{path}: AMReX parent-device count mismatch")
        require(map_integer(fields, "num_device_partners", path) == ranks, f"{path}: AMReX parent partner count mismatch")
        parsed.append(fields)
    require(
        len({fields["visible_mig_uuid"] for fields in parsed}) == ranks,
        f"{path}: visible MIG UUIDs are not unique",
    )
    require(
        len({fields["cuda_parent_uuid"] for fields in parsed}) == 1,
        f"{path}: CUDA parent UUID identity changed",
    )
    require(
        len({fields["allocation_cuda_visible_devices"] for fields in parsed}) == 1,
        f"{path}: ranks disagree on allocation CUDA device list",
    )
    require(
        set(parsed[0]["allocation_cuda_visible_devices"].split(","))
        == {fields["visible_mig_uuid"] for fields in parsed},
        f"{path}: visible MIG set does not cover allocation",
    )
    return parsed


def validate_common(row: Dict[str, str], path: Path, expected: Dict[str, str]) -> None:
    require(integer(row, "schema_version", path) == 2, f"{path}: wrong schema")
    require(row["run_mode"] in {"smoke", "calibration", "baseline"}, f"{path}: bad run mode")
    workload = row["workload"]
    require(workload in IDENTITIES, f"{path}: unknown workload")
    descriptor, grid, max_grid_size, boxes = IDENTITIES[workload]
    require(row["descriptor"] == descriptor, f"{path}: descriptor mismatch")
    component_sums = parse_population_component_sums(row, path)
    require(
        math.fsum(component_sums) > 0.0,
        f"{path}: final population total is not positive",
    )
    require(
        number(row, "final_population_l2", path) > 0.0,
        f"{path}: final population L2 is not positive",
    )
    require(
        tuple(integer(row, name, path) for name in ("grid_x", "grid_y", "grid_z")) == grid,
        f"{path}: grid mismatch",
    )
    require(integer(row, "max_grid_size", path) == max_grid_size, f"{path}: max_grid_size mismatch")
    require(integer(row, "box_count", path) == boxes, f"{path}: box count mismatch")
    require(
        integer(row, "global_valid_cells", path) == product(list(grid)),
        f"{path}: global cell count mismatch",
    )

    ranks = integer(row, "mpi_ranks", path)
    require(ranks in {1, 4}, f"{path}: MPI ranks must be 1 or 4")
    require(row["layout"] == f"{ranks}mig", f"{path}: layout mismatch")
    require(integer(row, "mig_instance_count", path) == ranks, f"{path}: MIG count mismatch")
    require(integer(row, "boxes_per_rank", path) == boxes // ranks, f"{path}: boxes/rank mismatch")
    require(integer(row, "omp_threads", path) == 1, f"{path}: OpenMP threads changed")
    require(integer(row, "num_devices_used", path) == 1, f"{path}: AMReX parent-device count mismatch")
    require(integer(row, "num_device_partners", path) == ranks, f"{path}: AMReX parent partner count mismatch")
    parent_count = integer(row, "parent_gpu_count", path)
    require(parent_count == 1, f"{path}: baseline requires one parent A100")
    parent_uuids = [value for value in row["parent_gpu_uuids"].split(";") if value]
    require(len(set(parent_uuids)) == parent_count, f"{path}: parent GPU UUID count mismatch")

    use_gpu_aware = integer(row, "amrex_use_gpu_aware_mpi", path)
    require(use_gpu_aware in {0, 1}, f"{path}: invalid GPU-aware MPI flag")
    expected_communication = "device_buffer" if use_gpu_aware else "pinned_staging"
    require(row["communication_mode"] == expected_communication, f"{path}: communication identity mismatch")
    require(row["binding_policy"] not in {"", "unknown"}, f"{path}: binding policy missing")
    require(row["slurm_job_gpus"] not in {"", "unknown"}, f"{path}: Slurm GPU identity missing")
    require(row["cpu_cpuset"].count("r") == ranks, f"{path}: CPU cpuset map incomplete")
    rank_map = parse_rank_device_map(row["rank_device_map"], ranks, path)
    normalized_parent_uuid = parent_uuids[0].replace("GPU-", "", 1).replace("-", "")
    require(
        {mapping["cuda_parent_uuid"] for mapping in rank_map}
        == {normalized_parent_uuid},
        f"{path}: CUDA and nvidia-smi parent UUIDs disagree",
    )
    require(
        integer(row, "num_device_partners", path)
        == map_integer(rank_map[0], "num_device_partners", path),
        f"{path}: top-level AMReX partner count mismatch",
    )
    cpuset_entries = row["cpu_cpuset"].split(";")
    require(len(cpuset_entries) == ranks, f"{path}: CPU cpuset map entry count mismatch")
    for rank, (cpuset_entry, mapping) in enumerate(zip(cpuset_entries, rank_map)):
        require(
            cpuset_entry == f"r{rank}={mapping['cpu_cpuset']}",
            f"{path}: CPU cpuset map disagrees with rank map",
        )

    required_strings = [
        "git_commit", "amrex_source_commit", "amrex_prefix", "amrex_config_sha256",
        "amrex_build_type", "build_type", "compiler", "nvcc", "cuda_toolkit",
        "cuda_runtime", "driver", "mpi", "mpi_cuda_build_version",
        "mpi_cuda_support_reported", "job_id", "node",
    ]
    for name in required_strings:
        require(row[name] not in {"", "unknown", "unset"}, f"{path}: missing {name}")
    require(re.fullmatch(r"[0-9a-f]{40}", row["git_commit"]) is not None, f"{path}: invalid Git SHA")
    require(re.fullmatch(r"[0-9a-f]{40}", row["amrex_source_commit"]) is not None, f"{path}: invalid AMReX SHA")
    require(re.fullmatch(r"[0-9a-f]{64}", row["amrex_config_sha256"]) is not None, f"{path}: invalid AMReX config hash")
    require(Path(row["amrex_prefix"]).is_absolute(), f"{path}: AMReX prefix must be absolute")
    for field in GPU_FIELDS:
        require(row[field] not in {"", "unknown", "unset"}, f"{path}: missing {field}")
        require(row[field] == expected[field], f"{path}: {field} differs from explicit expectation")
    require(row["amrex_build_type"] == "Release", f"{path}: AMReX build type mismatch")
    require(row["build_type"] == "Release", f"{path}: application build type mismatch")
    require(row["mpi_cuda_support_reported"] == "true", f"{path}: MPI CUDA support report mismatch")
    require(
        row["mpi_cuda_query_support"] in {"true", "false"},
        f"{path}: invalid MPIX CUDA capability",
    )
    require(
        use_gpu_aware == 0 or row["mpi_cuda_query_support"] == "true",
        f"{path}: device-buffer communication lacks runtime MPIX CUDA support",
    )

    repeat = integer(row, "repeat", path)
    warmup = integer(row, "warmup_steps", path)
    sample_start = integer(row, "sample_start", path)
    sample_end = integer(row, "sample_end", path)
    nsteps = integer(row, "nsteps", path)
    require(repeat > 0 and warmup >= 0 and nsteps > 0, f"{path}: invalid repeat/window")
    require(sample_start == warmup + 1, f"{path}: sample start mismatch")
    require(sample_end - sample_start + 1 == nsteps, f"{path}: incomplete sample")


def validate_throughput(row: Dict[str, str], path: Path) -> None:
    require(row["measurement_mode"] == "throughput", f"{path}: not throughput")
    require(row["formal_mlups"] == "true", f"{path}: throughput must be formal")
    window = number(row, "synchronized_throughput_window_s", path)
    seconds_per_step = number(row, "total_s_per_step", path)
    mlups = number(row, "mlups", path)
    require(window > 0.0 and seconds_per_step > 0.0 and mlups > 0.0, f"{path}: non-positive throughput")
    nsteps = integer(row, "nsteps", path)
    cells = integer(row, "global_valid_cells", path)
    require(close(seconds_per_step, window / nsteps), f"{path}: seconds/step mismatch")
    require(close(mlups, cells * nsteps / window / 1.0e6), f"{path}: MLUPS mismatch")
    if row["run_mode"] == "baseline":
        require(window >= 600.0, f"{path}: formal throughput window is shorter than 600 s")


def validate_stage_breakdown(row: Dict[str, str], path: Path) -> None:
    require(
        row["measurement_mode"] == "synchronized_stage_breakdown",
        f"{path}: not stage breakdown",
    )
    require(row["formal_mlups"] == "false", f"{path}: breakdown cannot be formal")
    ranks = integer(row, "mpi_ranks", path)
    critical_rank = integer(row, "critical_rank", path)
    require(0 <= critical_rank < ranks, f"{path}: invalid critical rank")
    critical = [number(row, name, path) for name in CRITICAL_WINDOW_FIELDS]
    maxima = [number(row, name, path) for name in INDEPENDENT_MAX_FIELDS]
    require(all(value >= 0.0 for value in critical + maxima), f"{path}: negative duration")
    require(critical[-1] > 0.0, f"{path}: non-positive critical total")
    for actual, maximum in zip(critical, maxima):
        require(maximum + 1.0e-12 >= actual, f"{path}: independent maximum below critical value")

    nsteps = integer(row, "nsteps", path)
    per_step_sources = [critical[index] for index in (0, 1, 5, 8, 9, 10)]
    for name, source in zip(CRITICAL_PER_STEP_FIELDS, per_step_sources):
        require(close(number(row, name, path), source / nsteps), f"{path}: {name} mismatch")

    top_level = [number(row, name, path) for name in SHARE_FIELDS[:6]]
    require(close(sum(top_level), 100.0), f"{path}: top-level shares do not sum to 100")
    expected_top = [
        100.0 * critical[index] / critical[-1] for index in (0, 1, 5, 8, 9)
    ]
    residual = critical[-1] - sum(critical[index] for index in (0, 1, 5, 8, 9))
    expected_top.append(100.0 * residual / critical[-1])
    for actual, expected in zip(top_level, expected_top):
        require(close(actual, expected), f"{path}: critical-rank share mismatch")

    substage_specs = [(2, 1), (6, 8), (7, 8)]
    for name, (child, parent) in zip(SHARE_FIELDS[6:], substage_specs):
        expected = 100.0 * critical[child] / critical[parent] if critical[parent] > 0.0 else 0.0
        require(close(number(row, name, path), expected), f"{path}: {name} mismatch")

    window = number(row, "diagnostic_synchronized_window_s", path)
    diagnostic_mlups = number(row, "diagnostic_mlups", path)
    require(window > 0.0 and diagnostic_mlups > 0.0, f"{path}: invalid diagnostic window")
    require(window + 1.0e-12 >= critical[-1], f"{path}: diagnostic window below critical total")
    cells = integer(row, "global_valid_cells", path)
    require(close(diagnostic_mlups, cells * nsteps / window / 1.0e6), f"{path}: diagnostic MLUPS mismatch")
    if row["run_mode"] == "baseline":
        require(60.0 <= window <= 120.0, f"{path}: formal breakdown window outside 60-120 s")


def validate_path(path: Path, expected: Dict[str, str]) -> Dict[str, str]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        columns = reader.fieldnames
    require(len(rows) == 1, f"{path}: expected exactly one row")
    row = rows[0]
    require(None not in row, f"{path}: row has extra fields")
    require(all(value is not None for value in row.values()), f"{path}: row has missing fields")
    mode = row.get("measurement_mode", "")
    expected_columns = THROUGHPUT_COLUMNS if mode == "throughput" else STAGE_BREAKDOWN_COLUMNS
    require(columns == expected_columns, f"{path}: schema columns mismatch")
    validate_common(row, path, expected)
    if mode == "throughput":
        validate_throughput(row, path)
    else:
        validate_stage_breakdown(row, path)
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", nargs="+", type=Path)
    add_expected_arguments(parser, GPU_FIELDS)
    args = parser.parse_args()
    expected = expected_values(args, GPU_FIELDS)
    try:
        for path in args.records:
            validate_path(path, expected)
    except ValidationError as exc:
        print(f"FAIL: {exc}")
        return 1
    except (OSError, csv.Error) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(f"gpu_performance_csv passed records={len(args.records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
