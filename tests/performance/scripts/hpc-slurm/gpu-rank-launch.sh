#!/usr/bin/env bash

# Slurm selects the MIG. Bind each local rank to its real allocation slice,
# including noncontiguous CPU IDs; retain the existing mapping log schema.
set -euo pipefail
(( $# > 0 )) || { printf 'error: executable is required\n' >&2; exit 2; }
exec python3 -B - "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" "$@" <<'PY'
import os
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
from check_gpu_mapping_log import cpu_set, require

rank, local = int(os.environ["SLURM_PROCID"]), int(os.environ["SLURM_LOCALID"])
tasks, width = int(os.environ["SLURM_NTASKS"]), int(os.environ["SLURM_CPUS_PER_TASK"])
visible = os.environ["CUDA_VISIBLE_DEVICES"]
allocated = os.environ["LBM_GPU_ALLOCATION_CUDA_VISIBLE_DEVICES"]
allocation = sorted(cpu_set(os.environ["LBM_GPU_ALLOCATION_CPUSET"]))
require(visible and "," not in visible and allocated, "task-local device identity is missing")
require(len(allocation) == tasks*width and 0 <= local < tasks, "CPU allocation differs")
selected = set(allocation[local*width:(local+1)*width])
require(selected <= os.sched_getaffinity(0), "rank CPU slice is outside the allowed set")
os.sched_setaffinity(0, selected)
cpuset = ",".join(map(str, sorted(os.sched_getaffinity(0))))
memset = next(line.split(":",1)[1].strip() for line in Path("/proc/self/status").read_text().splitlines()
              if line.startswith("Mems_allowed_list:"))
os.environ["LBM_GPU_DEVICE_ASSIGNMENT_SOURCE"] = "slurm-task-local-cvd"
print(f"gpu_rank_launch global_rank={rank} local_rank={local} cuda_visible_devices={visible} "
      f"allocation_cuda_visible_devices={allocated} device_assignment_source=slurm-task-local-cvd "
      f"slurm_step_gpus={os.environ.get('SLURM_STEP_GPUS','unset')} cpuset={cpuset} memset={memset}", flush=True)
os.execvp(sys.argv[2],sys.argv[2:])
PY
