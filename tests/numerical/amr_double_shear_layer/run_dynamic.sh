#!/usr/bin/env bash
# One tests-owned dynamic DSL solver/artifact/analysis/checker path.
set -uo pipefail
if (( $# < 5 || $# > 6 )); then
    printf 'Usage: %s <d2-exe> <d3-exe> <work-dir> <mpiexec> <d2|d3>-<2l|3l|fixed-band> [matching-uniform-run]\n' "$0" >&2
    exit 2
fi
d2="$1" d3="$2" work="$3" launcher="$4" entry="$5"
locality="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_id="$("${LBM_TEST_PYTHON:-python3}" -B "$locality/../../scripts/source_identity.py" --strict)" || exit 2
ranks="${LBM_TEST_AMR_MPI_RANKS:-2}"; python="${LBM_TEST_PYTHON:-python3}"
limit="${LBM_TEST_MAX_STEP:--1}"
[[ "$ranks" =~ ^[1-9][0-9]*$ && "$limit" =~ ^-?[0-9]+$ ]] || exit 2
case "${entry%%-*}" in d2) exe="$d2"; descriptor=d2q9 ;; d3) exe="$d3"; descriptor=d3q27 ;; *) exit 2 ;; esac
input="dynamic_$descriptor"; variant=(); checker=check_dynamic_double_shear_layer.py
case "${entry#*-}" in
    2l) level=1 ;;
    3l) level=2 ;;
    fixed-band) level=2; input="fixed_band_$descriptor"; variant=(--fixed-band); checker=check_fixed_band.py ;;
    *) exit 2 ;;
esac
attempt="$work/attempt-$(date +%Y%m%d-%H%M%S)-$$"
output="$attempt/run"
mkdir -p "$output" || exit 2
printf '%s\n' "$output" > "$work/latest-run.txt" || exit 2
export OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
command=("$launcher" -np "$ranks" --map-by core --bind-to core --rank-by slot --report-bindings
    "$exe" "$locality/inputs/$input" "case.max_level=$level"
    "run.max_step=$limit" "run.source_id=$source_id" "output.directory=$output"
    amrex.omp_threads=1 amrex.use_gpu_aware_mpi=0)
printf '%q ' "${command[@]}" > "$output/command.txt"; printf '\n' >> "$output/command.txt"
"${command[@]}" 2>&1 | tee "$output/console.log"
statuses=("${PIPESTATUS[@]}")
printf 'solver_exit=%s\ntee_exit=%s\n' "${statuses[0]}" "${statuses[1]}" > "$output/exit.txt" || exit 2
if (( statuses[0] != 0 || statuses[1] != 0 )); then
    printf 'exit=2\nerror=solver failed\n' > "$attempt/result.txt"
    exit 2
fi
reference=()
if (( $# == 6 )); then reference=(--reference "$6"); fi
short=()
if (( limit >= 0 && limit < 2560 )); then short=(--short); fi
"$python" -B "$locality/analyze_dynamic_double_shear_layer.py" --run "$output" --output "$output/analysis" "${reference[@]}" "${short[@]}" "${variant[@]}" 2>&1 | tee "$output/analyzer.log"
statuses=("${PIPESTATUS[@]}")
result=0
if (( statuses[0] != 0 || statuses[1] != 0 )); then result=2; fi
"$python" -B "$locality/$checker" --run "$output" --analysis "$output/analysis" --output "$output/check" "${short[@]}" 2>&1 | tee "$output/checker.log"
statuses=("${PIPESTATUS[@]}")
if (( statuses[1] != 0 || statuses[0] > 1 )); then
    result=2
elif (( statuses[0] == 1 )); then
    if [[ "$checker" == check_fixed_band.py ]] && (( result == 0 )); then result=1
    else result=2
    fi
fi
printf 'entry=%s\nsource_id=%s\nranks=%s\nexit=%s\nscientific_gate=none\n' "$entry" "$source_id" "$ranks" "$result" > "$attempt/result.txt" || exit 2
exit "$result"
