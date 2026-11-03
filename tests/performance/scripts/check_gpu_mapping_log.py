#!/usr/bin/env python3
"""Validate one explicit single-node rank-to-MIG mapping log."""

import argparse
from pathlib import Path
import re
from typing import Dict, List, Set


class ValidationError(Exception):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def fields(line: str) -> Dict[str, str]:
    return dict(re.findall(r"([a-z_]+)=([^ ]+)", line))


def integer(record: Dict[str, str], name: str) -> int:
    try:
        return int(record[name])
    except (KeyError, ValueError) as exc:
        raise ValidationError("invalid integer field: {}".format(name)) from exc


def cpu_set(text: str) -> Set[int]:
    require(text not in {"", "unset"}, "CPU set is missing")
    result = set()
    try:
        for group in text.split(","):
            bounds = group.split("-")
            require(len(bounds) in {1, 2}, "CPU set has invalid range")
            begin = int(bounds[0])
            end = int(bounds[-1])
            require(begin >= 0 and end >= begin, "CPU set has invalid bounds")
            result.update(range(begin, end + 1))
    except ValueError as exc:
        raise ValidationError("CPU set is not numeric") from exc
    require(bool(result), "CPU set is empty")
    return result


def normalized_parent_uuid(text: str) -> str:
    value = text[4:] if text.startswith("GPU-") else text
    value = value.replace("-", "").lower()
    require(re.fullmatch(r"[0-9a-f]{32}", value) is not None, "parent GPU UUID is invalid")
    return value


def indexed(records: List[Dict[str, str]], expected_ranks: int, kind: str) -> List[Dict[str, str]]:
    require(len(records) == expected_ranks, "{} record count mismatch".format(kind))
    by_rank = {}
    for record in records:
        rank = integer(record, "global_rank")
        require(rank not in by_rank, "{} contains duplicate rank".format(kind))
        by_rank[rank] = record
    require(set(by_rank) == set(range(expected_ranks)), "{} rank set mismatch".format(kind))
    return [by_rank[rank] for rank in range(expected_ranks)]


def validate(
    path: Path,
    expected_ranks: int,
    expected_cpus_per_rank: int,
    expected_parent_gpus: int,
    expected_parent_uuid: str,
    allocation: Set[int],
) -> None:
    lines = path.read_text().splitlines()
    launches = indexed(
        [fields(line) for line in lines if line.startswith("gpu_rank_launch ")],
        expected_ranks,
        "launcher",
    )
    mappings = indexed(
        [fields(line) for line in lines if line.startswith("gpu_mapping_status=pass ")],
        expected_ranks,
        "mapping",
    )
    require(
        lines.count("gpu_mapping_global_status=0") == 1,
        "global mapping status is missing or duplicated",
    )

    used_cpus = set()
    mig_uuids = set()
    parent_uuids = set()
    allocation_device_lists = set()
    for rank in range(expected_ranks):
        launch = launches[rank]
        mapping = mappings[rank]
        require(integer(launch, "local_rank") == rank, "launcher local rank mismatch")
        require(integer(mapping, "local_rank") == rank, "mapping local rank mismatch")
        require(integer(mapping, "amrex_local_rank") == rank, "AMReX local rank mismatch")
        require(
            mapping.get("ompi_local_rank", "unset") in {"unset", str(rank)},
            "Open MPI local-rank environment mismatch",
        )
        for record in (launch, mapping):
            require(
                record.get("cuda_visible_devices", "unset") not in {"", "unset"}
                and "," not in record["cuda_visible_devices"],
                "rank sees multiple or no CUDA devices",
            )
        allocation_devices = launch.get(
            "allocation_cuda_visible_devices", ""
        ).split(",")
        require(
            len(allocation_devices) == expected_ranks
            and all(allocation_devices),
            "allocation CUDA device list does not match rank count",
        )
        allocation_device_lists.add(tuple(allocation_devices))
        require(
            launch.get("device_assignment_source")
            == "slurm-task-local-cvd",
            "device assignment provenance is invalid",
        )
        require(
            launch["cuda_visible_devices"] in allocation_devices,
            "rank-local device is outside allocation list",
        )
        require(
            mapping["cuda_visible_devices"] == launch["cuda_visible_devices"],
            "application and launcher CUDA identities disagree",
        )
        require(
            mapping.get("slurm_step_gpus", "unset")
            == launch.get("slurm_step_gpus", "unset"),
            "application and launcher Slurm step identities disagree",
        )
        require(integer(mapping, "visible_device_count") == 1, "visible device count mismatch")
        require(integer(mapping, "cuda_device_id") == 0, "CUDA device id mismatch")
        require(integer(mapping, "amrex_device_id") == 0, "AMReX device id mismatch")
        require(integer(mapping, "num_devices_used") == expected_parent_gpus, "AMReX parent-device count mismatch")
        require(integer(mapping, "num_device_partners") == expected_ranks, "AMReX parent partner count mismatch")
        mig_uuid = mapping.get("visible_mig_uuid", "")
        require(re.fullmatch(r"MIG-[0-9a-f-]+", mig_uuid) is not None, "visible MIG UUID is invalid")
        require(mig_uuid == mapping["cuda_visible_devices"], "visible MIG/CUDA identity mismatch")
        require(mig_uuid not in mig_uuids, "visible MIG UUIDs are not unique")
        mig_uuids.add(mig_uuid)
        parent_uuid = mapping.get("cuda_parent_uuid", "")
        require(re.fullmatch(r"[0-9a-f]{32}", parent_uuid) is not None, "CUDA parent UUID is invalid")
        parent_uuids.add(parent_uuid)

        rank_cpus = cpu_set(launch.get("cpuset", ""))
        require(
            len(rank_cpus) == expected_cpus_per_rank,
            "rank CPU set size mismatch",
        )
        require(rank_cpus <= allocation, "rank CPU set is outside allocation")
        require(not (rank_cpus & used_cpus), "rank CPU sets overlap")
        used_cpus.update(rank_cpus)
        require(launch.get("memset", "unset") not in {"", "unset"}, "rank NUMA memory set is missing")
    require(len(allocation_device_lists) == 1, "ranks disagree on allocation device list")
    require(
        set(next(iter(allocation_device_lists))) == mig_uuids,
        "visible MIG set does not cover allocation",
    )
    require(len(parent_uuids) == expected_parent_gpus, "CUDA parent GPU count mismatch")
    require(
        parent_uuids == {normalized_parent_uuid(expected_parent_uuid)},
        "CUDA and nvidia-smi parent UUIDs disagree",
    )
    require(used_cpus == allocation, "rank CPU sets do not cover the allocation")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-ranks", type=int, required=True)
    parser.add_argument("--expected-cpus-per-rank", type=int, required=True)
    parser.add_argument("--expected-parent-gpus", type=int, required=True)
    parser.add_argument("--expected-parent-uuid", required=True)
    parser.add_argument("--allocation-cpuset", required=True)
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    try:
        require(args.expected_ranks > 0, "expected rank count must be positive")
        require(
            args.expected_cpus_per_rank > 0,
            "expected CPUs per rank must be positive",
        )
        require(args.expected_parent_gpus == 1, "baseline requires one parent GPU")
        validate(
            args.log,
            args.expected_ranks,
            args.expected_cpus_per_rank,
            args.expected_parent_gpus,
            args.expected_parent_uuid,
            cpu_set(args.allocation_cpuset),
        )
    except ValidationError as exc:
        print("FAIL: {}".format(exc))
        return 1
    except OSError as exc:
        print("ERROR: {}".format(exc))
        return 2
    print("gpu_mapping_log passed ranks={}".format(args.expected_ranks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
