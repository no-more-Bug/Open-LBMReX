#!/usr/bin/env bash
set -uo pipefail

if (( $# != 4 )); then
    printf 'Usage: %s <d2-executable> <d3-executable> <work-dir> <mpiexec>\n' "$0" >&2
    exit 2
fi
d2_exe="$1"; d3_exe="$2"; work_dir="$3"; mpiexec="$4"
test_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
attempt_dir="${work_dir}/attempt-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "${attempt_dir}" || exit 2
scientific_exit=0

for descriptor in d2q9 d3q27; do
    if [[ "${descriptor}" == d2q9 ]]; then executable="${d2_exe}"; input="${test_dir}/inputs/d2q9_hbb_re1000"; else executable="${d3_exe}"; input="${test_dir}/inputs/d3q27_hbb_re1000"; fi
    run_dir="${attempt_dir}/${descriptor}"
    log_dir="${attempt_dir}/log/${descriptor}"
    mkdir -p "${run_dir}" "${log_dir}" || exit 2
    command=(
        "${mpiexec}" -np 4 --map-by core --bind-to core --rank-by slot
        --report-bindings "${executable}" "${input}"
        run.max_step=200000 diagnostics.interval=5000
        diagnostics.steady_check_interval=5000
        diagnostics.steady_l2_tolerance=5e-8 diagnostics.stop_on_steady=1
        diagnostics.write_centerlines=1 diagnostics.write_metrics=0
        output.plot_interval=0 output.write_final_plotfile=0
        output.directory="${run_dir}" output.log_directory="${log_dir}"
        amrex.omp_threads=1 amrex.use_gpu_aware_mpi=0)
    printf 'command:' >"${log_dir}/console.log" || exit 2
    printf ' %q' "${command[@]}" >>"${log_dir}/console.log"
    printf '\n' >>"${log_dir}/console.log"
    set +e
    env OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores \
        "${command[@]}" 2>&1 | tee -a "${log_dir}/console.log"
    status=("${PIPESTATUS[@]}")
    printf 'solver_exit=%d tee_exit=%d\n' "${status[0]}" "${status[1]}" \
        >>"${log_dir}/console.log" || exit 2
    (( status[0] == 0 && status[1] == 0 )) || exit 2
    set +e
    python3 -B "${test_dir}/check_ghia.py" "${descriptor}" "${run_dir}" \
        "${log_dir}/console.log" "${test_dir}/reference-data" 2>&1 | \
        tee -a "${attempt_dir}/checker.log"
    checker_status=("${PIPESTATUS[@]}")
    printf '%s checker_exit=%d tee_exit=%d\n' "${descriptor}" \
        "${checker_status[0]}" "${checker_status[1]}" \
        >>"${attempt_dir}/checker.log" || exit 2
    (( checker_status[1] == 0 )) || exit 2
    checker_exit="${checker_status[0]}"
    if (( checker_exit == 1 )); then
        scientific_exit=1
    elif (( checker_exit != 0 )); then
        exit 2
    fi
done
exit "${scientific_exit}"
