#!/usr/bin/env bash
# One complete solver run per entry; references are explicit read-only artifacts.
set -uo pipefail
if (( $# < 5 || $# > 6 )); then
    printf 'Usage: %s <d2-exe> <d3-exe> <work-dir> <mpiexec> <d2|d3>-<ref128|ref256|ref512|baseline|3l|2r|p1|p2|p3> [reference-work-dir]\n' "$0" >&2
    exit 2
fi
d2="$1" d3="$2" work="$3" launcher="$4" entry="$5"
locality="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_id="$("${LBM_TEST_PYTHON:-python3}" -B "$locality/../../scripts/source_identity.py" --strict)" || exit 2
ranks="${LBM_TEST_AMR_MPI_RANKS:-4}"; python="${LBM_TEST_PYTHON:-python3}"
[[ "$ranks" =~ ^[1-9][0-9]*$ ]] || exit 2
case "${entry%%-*}" in d2) exe="$d2"; descriptor=d2q9 ;; d3) exe="$d3"; descriptor=d3q27 ;; *) exit 2 ;; esac
layout="${entry#*-}"; reference=0; cells=128
case "$layout" in
    ref128|ref256|ref512) cells="${layout#ref}"; layout=uniform; reference=1 ;;
    baseline|3l|2r|p1|p2|p3) (( $# == 6 )) || exit 2 ;;
    *) exit 2 ;;
esac
attempt="$work/attempt-$(date +%Y%m%d-%H%M%S)-$$"
output="$attempt/run"
mkdir -p "$output" || exit 2
# The producer alone updates the latest-run index.
if (( reference )); then printf '%s\n' "$output" > "$work/latest-run.txt" || exit 2; fi
"$python" -c 'import numpy; print("numpy="+numpy.__version__)' > "$attempt/python.txt" 2>&1 || {
    printf 'exit=2\nerror=NumPy unavailable\n' > "$attempt/result.txt"; exit 2;
}
export OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
command=("$launcher" -np "$ranks" --map-by core --bind-to core --rank-by slot --report-bindings
    "$exe" "$locality/inputs/$descriptor" "case.layout=$layout" "case.uniform_cells=$cells"
    run.max_step=-1 "run.source_id=$source_id" "output.directory=$output" amrex.omp_threads=1 amrex.use_gpu_aware_mpi=0)
printf '%q ' "${command[@]}" > "$output/command.txt"; printf '\n' >> "$output/command.txt"
"${command[@]}" 2>&1 | tee "$output/console.log"
statuses=("${PIPESTATUS[@]}")
printf 'solver_exit=%s\ntee_exit=%s\n' "${statuses[0]}" "${statuses[1]}" > "$output/exit.txt" || exit 2
result=0
if (( statuses[0] != 0 || statuses[1] != 0 )); then result=2; fi
# Even a failed solver can leave useful synchronized frames; analysis preserves them.
"$python" -B "$locality/analyze_double_shear_layer.py" --run "$output" --output "$output/analysis" 2>&1 | tee "$output/analyzer.log"
statuses=("${PIPESTATUS[@]}")
if (( statuses[0] != 0 || statuses[1] != 0 )); then result=2; fi
if (( reference )); then
    check=(--uniform "$output")
else
    reference_work="$6"
    if ! IFS= read -r matched < "$reference_work/latest-run.txt" || [[ -z "$matched" ]] ||
       ! grep -Exq 'exit=[01]' "${matched%/run}/result.txt"; then
        printf 'Invalid evidence: matching reference is missing, incomplete or failed to run\n' | tee "$output/checker.log"
        printf 'entry=%s\nsource_id=%s\nexit=2\nerror=invalid reference\n' "$entry" "$source_id" > "$attempt/result.txt"
        exit 2
    fi
    printf 'reference=%s\n' "$matched" > "$output/reference.txt"
    check=(--amr "$output" --reference "$matched")
fi
"$python" -B "$locality/check_double_shear_layer.py" "${check[@]}" --output "$output/check" 2>&1 | tee "$output/checker.log"
statuses=("${PIPESTATUS[@]}")
if (( statuses[1] != 0 || statuses[0] > 1 )) ||
   ! grep -Fxq "exit=${statuses[0]}" "$output/check/result.txt"; then result=2
elif (( statuses[0] == 1 && result == 0 )); then result=1; fi
printf 'entry=%s\nsource_id=%s\nranks=%s\nexit=%s\n' "$entry" "$source_id" "$ranks" "$result" > "$attempt/result.txt" || exit 2
exit "$result"
