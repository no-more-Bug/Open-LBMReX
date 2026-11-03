#!/usr/bin/env bash
# Execute one fixed Re20 identity; comparison artifacts are explicit read-only inputs.
set -uo pipefail
if (( $# < 5 || $# > 6 )); then
    printf 'Usage: %s <d2-exe> <d3-exe> <work-dir> <mpiexec> <d2|d3>-<yms|bouzidi>-<coarse|fine|amr> [fine-work-dir]\n' "$0" >&2
    exit 2
fi
d2="$1" d3="$2" work="$3" launcher="$4" entry="$5"
locality="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_id="$("${LBM_TEST_PYTHON:-python3}" -B "$locality/../../scripts/source_identity.py" --strict)" || exit 2
ranks="${LBM_TEST_AMR_MPI_RANKS:-4}"; python="${LBM_TEST_PYTHON:-python3}"
box="${LBM_RE20_MAX_GRID_SIZE:-64}"
[[ "$ranks" =~ ^[1-9][0-9]*$ ]] || exit 2
IFS=- read -r dimension scheme layout extra <<< "$entry"
[[ -z "$extra" ]] || exit 2
case "$dimension" in d2) exe="$d2"; descriptor=d2q9 ;; d3) exe="$d3"; descriptor=d3q27 ;; *) exit 2 ;; esac
case "$scheme" in yms|bouzidi) ;; *) exit 2 ;; esac
case "$layout" in fine) (( $# == 5 )) || exit 2 ;; coarse|amr) (( $# == 6 )) || exit 2 ;; *) exit 2 ;; esac
attempt="$work/attempt-$(date +%Y%m%d-%H%M%S)-$$"
output="$attempt/run"
mkdir -p "$output" || exit 2
printf '%s\n' "$output" > "$work/latest-run.txt" || exit 2
"$python" -c 'import numpy; print("numpy="+numpy.__version__)' > "$attempt/python.txt" 2>&1 || {
    printf 'exit=2\nerror=NumPy unavailable\n' > "$attempt/result.txt"; exit 2;
}
export OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
command=("$launcher" -np "$ranks" --map-by core --bind-to core --rank-by slot --report-bindings
    "$exe" "$locality/inputs/$descriptor" "case.scheme=$scheme" "case.layout=$layout"
    "mesh.max_grid_size=$box" run.max_step=-1 "run.source_id=$source_id"
    "output.directory=$output" amrex.omp_threads=1 amrex.use_gpu_aware_mpi=0)
printf '%q ' "${command[@]}" > "$output/command.txt"; printf '\n' >> "$output/command.txt"
"${command[@]}" 2>&1 | tee "$output/console.log"
statuses=("${PIPESTATUS[@]}")
printf 'solver_exit=%s\ntee_exit=%s\n' "${statuses[0]}" "${statuses[1]}" > "$output/exit.txt" || exit 2
result=0
if (( statuses[0] != 0 || statuses[1] != 0 )); then result=2; fi
check=(--run "$output" --output "$output/analysis")
if [[ "$layout" != fine ]]; then
    if IFS= read -r matched < "$6/latest-run.txt" && [[ -n "$matched" ]]; then
        check+=(--reference "$matched")
        printf 'reference=%s\n' "$matched" > "$output/reference.txt"
    else
        result=2
        printf 'Matching fine reference is missing\n' > "$output/reference.txt"
    fi
fi
if [[ "$dimension" == d3 && -n "${LBM_RE20_D2_REFERENCE:-}" ]]; then
    if IFS= read -r matched < "$LBM_RE20_D2_REFERENCE/latest-run.txt" && [[ -n "$matched" ]]; then
        check+=(--d2-reference "$matched")
        printf 'reference=%s\n' "$matched" > "$output/d2-reference.txt"
    else
        result=2
        printf 'Matching D2 comparison is missing\n' > "$output/d2-reference.txt"
    fi
fi
"$python" -B "$locality/check_re20.py" "${check[@]}" 2>&1 | tee "$output/checker.log"
statuses=("${PIPESTATUS[@]}")
if (( statuses[1] != 0 || statuses[0] > 1 )) ||
   ! grep -Fxq "exit=${statuses[0]}" "$output/analysis/result.txt"; then result=2
elif (( statuses[0] == 1 && result == 0 )); then result=1; fi
printf 'entry=%s\nsource_id=%s\nranks=%s\nexit=%s\n' "$entry" "$source_id" "$ranks" "$result" > "$attempt/result.txt" || exit 2
exit "$result"
