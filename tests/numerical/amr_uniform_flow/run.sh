#!/usr/bin/env bash
# One case, one artifact directory, one explicit checker invocation.
set -uo pipefail
(( $# == 5 )) || { printf 'Usage: %s <d2-exe> <d3-exe> <work> <launcher> <d2|d3>\n' "$0" >&2; exit 2; }
d2="$1" d3="$2" work="$3" launcher="$4" entry="$5"
locality="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_id="$("${LBM_TEST_PYTHON:-python3}" -B "$locality/../../scripts/source_identity.py" --strict)" || exit 2
ranks="${LBM_TEST_AMR_MPI_RANKS:-4}" python="${LBM_TEST_PYTHON:-python3}"
limit="${LBM_TEST_MAX_STEP:--1}"
[[ "$ranks" =~ ^[1-9][0-9]*$ && "$limit" =~ ^(-1|[0-9]+)$ ]] || exit 2
case "$entry" in d2) exe="$d2"; descriptor=d2q9 ;; d3) exe="$d3"; descriptor=d3q27 ;; *) exit 2 ;; esac
attempt="$work/attempt-$(date +%Y%m%d-%H%M%S)-$$"
output="$attempt/run"
mkdir -p "$output" || exit 2
printf '%s\n' "$output" > "$work/latest-run.txt" || exit 2
export OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
command=("$launcher" -np "$ranks" --map-by core --bind-to core --rank-by slot --report-bindings
    "$exe" "$locality/inputs/$descriptor" "run.max_step=$limit" "run.source_id=$source_id"
    "output.directory=$output" amrex.omp_threads=1 amrex.use_gpu_aware_mpi=0)
printf '%q ' "${command[@]}" > "$output/command.txt"; printf '\n' >> "$output/command.txt"
"${command[@]}" 2>&1 | tee "$output/console.log"
statuses=("${PIPESTATUS[@]}")
printf 'solver_exit=%s\ntee_exit=%s\n' "${statuses[0]}" "${statuses[1]}" > "$output/exit.txt" || exit 2
if (( statuses[0] != 0 || statuses[1] != 0 )); then exit 2; fi
check=(--run "$output" --output "$output/analysis")
if [[ "${LBM_TEST_ENGINEERING:-0}" == 1 ]]; then check+=(--engineering); fi
"$python" -B "$locality/check_uniform_flow.py" "${check[@]}" 2>&1 | tee "$output/checker.log"
statuses=("${PIPESTATUS[@]}")
result="${statuses[0]}"
if (( result > 2 || statuses[1] != 0 )) || ! grep -Fxq "exit=$result" "$output/analysis/result.txt"; then result=2; fi
printf 'entry=%s\nsource_id=%s\nexit=%s\n' "$entry" "$source_id" "$result" > "$attempt/result.txt" || exit 2
exit "$result"
