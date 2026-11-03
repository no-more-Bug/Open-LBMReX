#!/usr/bin/env bash
# One bounded group owns its references and actual artifacts. No shared runner.
set -uo pipefail
if (( $# != 5 )); then
    printf 'Usage: %s <d2-exe> <d3-exe> <work-dir> <mpiexec> <d2-axis|d2-diagonal|d3-axis|d3-diagonal>\n' "$0" >&2
    exit 2
fi
d2="$1" d3="$2" work="$3" launcher="$4" group="$5"
case "$group" in d2-axis|d2-diagonal|d3-axis|d3-diagonal) ;; *) exit 2 ;; esac
locality="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_id="$("${LBM_TEST_PYTHON:-python3}" -B "$locality/../../scripts/source_identity.py" --strict)" || exit 2
ranks="${LBM_TEST_AMR_MPI_RANKS:-4}"
[[ "$ranks" =~ ^[1-9][0-9]*$ ]] || exit 2
python="${LBM_TEST_PYTHON:-python3}"
attempt="$work/attempt-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$attempt" || exit 2
"$python" -c 'import numpy; print("numpy="+numpy.__version__)' > "$attempt/python.txt" 2>&1 || {
    printf 'exit=2\nerror=NumPy unavailable\n' > "$attempt/result.txt"; exit 2;
}
mode="${group#*-}"
if [[ "$group" == d2-* ]]; then
    exe="$d2"; descriptor=d2q9; grids=(64 128 256)
else
    exe="$d3"; descriptor=d3q27; grids=(32 64 128)
fi
export OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
run_case() {
    local layout="$1" nc="$2" directory="$attempt/${1}-N${2}"
    mkdir -p "$directory" || return 2
    local command=("$launcher" -np "$ranks" --map-by core --bind-to core --rank-by slot --report-bindings
        "$exe" "$locality/inputs/$descriptor" "case.mode=$mode" "case.layout=$layout" "case.coarse_cells=$nc"
        run.max_step=-1 "run.source_id=$source_id" "output.directory=$directory" amrex.omp_threads=1 amrex.use_gpu_aware_mpi=0)
    printf '%q ' "${command[@]}" > "$directory/command.txt"
    printf '\n' >> "$directory/command.txt"
    "${command[@]}" 2>&1 | tee "$directory/console.log"
    local statuses=("${PIPESTATUS[@]}")
    printf 'solver_exit=%s\ntee_exit=%s\n' "${statuses[0]}" "${statuses[1]}" > "$directory/exit.txt" || return 2
    (( statuses[0] == 0 && statuses[1] == 0 )) || return 2
}
result=0
for nc in "${grids[@]}"; do
    run_case uniform "$nc" || result=2
    layouts=(half)
    if [[ "$mode" == diagonal && "$nc" == "${grids[1]}" ]]; then layouts+=(corner shell separated); fi
    for layout in "${layouts[@]}"; do
        run_case "$layout" "$nc" || result=2
        "$python" -B "$locality/check_periodic_shear.py" --amr "$attempt/$layout-N$nc" \
            --reference "$attempt/uniform-N$nc" --output "$attempt/$layout-N$nc/analysis" 2>&1 | tee "$attempt/$layout-N$nc/checker.log"
        statuses=("${PIPESTATUS[@]}")
        status="${statuses[0]}"
        if (( statuses[1] != 0 || status > 1 )) ||
           ! grep -Fxq "exit=$status" "$attempt/$layout-N$nc/analysis/result.txt"; then result=2
        elif (( status == 1 && result == 0 )); then result=1; fi
    done
done
"$python" -B "$locality/summarize_convergence.py" "$attempt" "${grids[@]}" || result=2
printf 'group=%s\nsource_id=%s\nranks=%s\nexit=%s\n' "$group" "$source_id" "$ranks" "$result" > "$attempt/result.txt" || exit 2
exit "$result"
