#!/usr/bin/env bash
# One bounded long run. Compare two explicit outputs separately; never rerun a reference.
set -uo pipefail
if (( $# != 4 )); then
    printf 'Usage: %s <d2-exe> <work-dir> <mpiexec> <same|cross>\n' "$0" >&2
    exit 2
fi
exe="$1" work="$2" launcher="$3" layout="$4"
case "$layout" in same|cross) ;; *) exit 2 ;; esac
locality="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_id="$("${LBM_TEST_PYTHON:-python3}" -B "$locality/../../scripts/source_identity.py" --strict)" || exit 2
ranks="${LBM_TEST_AMR_MPI_RANKS:-8}"
[[ "$ranks" =~ ^[1-9][0-9]*$ ]] || exit 2
attempt="$work/attempt-$(date +%Y%m%d-%H%M%S)-$$"
output="$attempt/run"
mkdir -p "$output" || exit 2
printf '%s\n' "$output" > "$work/latest-run.txt" || exit 2
export OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
command=("$launcher" -np "$ranks" --map-by core --bind-to core --rank-by slot --report-bindings
    "$exe" "$locality/inputs/d2q9" run.mode=compare "case.layout=$layout" case.scheme=yms
    case.grouping=shared mesh.max_grid_size=64 "run.source_id=$source_id"
    "output.directory=$output" amrex.omp_threads=1 amrex.use_gpu_aware_mpi=0)
printf '%q ' "${command[@]}" > "$output/command.txt"; printf '\n' >> "$output/command.txt"
"${command[@]}" 2>&1 | tee "$output/console.log"
statuses=("${PIPESTATUS[@]}")
printf 'solver_exit=%s\ntee_exit=%s\n' "${statuses[0]}" "${statuses[1]}" > "$output/exit.txt" || exit 2
result=0
if (( statuses[0] != 0 || statuses[1] != 0 )); then result=2; fi
printf 'layout=%s\nsource_id=%s\nranks=%s\nexit=%s\n' "$layout" "$source_id" "$ranks" "$result" > "$attempt/result.txt" || exit 2
exit "$result"
