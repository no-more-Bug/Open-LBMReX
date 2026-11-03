#!/usr/bin/env bash
set -uo pipefail

if (( $# != 4 )); then
    printf 'Usage: %s <d2-executable> <d3-executable> <work-dir> <mpiexec>\n' "$0" >&2
    exit 2
fi

d2_exe="$1"
d3_exe="$2"
work_dir="$3"
mpiexec="$4"
test_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
attempt="attempt-$(date +%Y%m%d-%H%M%S)-$$"
attempt_dir="${work_dir}/${attempt}"

run_solver() {
    local executable="$1" input="$2" descriptor="$3" n="$4" depth="$5"
    local run_dir="${attempt_dir}/${descriptor}/N${n}"
    local log_dir="${attempt_dir}/log/${descriptor}/N${n}"
    mkdir -p "${run_dir}" "${log_dir}" || return 2
    local command=(
        "${mpiexec}" -np 4 --map-by core --bind-to core --rank-by slot
        --report-bindings "${executable}" "${input}"
        taylor_green.n_cell="${n}" "${n}" "${depth}"
        mesh.max_grid_size=25
        run.final_time_phys=0.2
        run.convergence_enabled=1
        run.convergence_reset=1
        diagnostics.interval=100
        output.directory="${run_dir}"
        output.log_directory="${log_dir}"
        output.convergence_directory=.
        output.plot_interval=0
        output.write_final_plotfile=0
        amrex.omp_threads=1 amrex.use_gpu_aware_mpi=0)
    printf 'command:' >"${log_dir}/console.log" || return 2
    printf ' %q' "${command[@]}" >>"${log_dir}/console.log"
    printf '\n' >>"${log_dir}/console.log"
    set +e
    env OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores \
        "${command[@]}" 2>&1 | tee -a "${log_dir}/console.log"
    local status=("${PIPESTATUS[@]}")
    printf 'solver_exit=%d tee_exit=%d\n' "${status[0]}" "${status[1]}" \
        >>"${log_dir}/console.log" || return 2
    (( status[0] == 0 && status[1] == 0 )) || return 2
}

mkdir -p "${attempt_dir}" || exit 2
for descriptor in d2q9 d3q27; do
    if [[ "${descriptor}" == d2q9 ]]; then
        executable="${d2_exe}"; input="${test_dir}/inputs/d2q9"; depth=1
    else
        executable="${d3_exe}"; input="${test_dir}/inputs/d3q27"; depth=4
    fi
    for n in 50 75 100; do
        run_solver "${executable}" "${input}" "${descriptor}" "${n}" "${depth}" || exit 2
    done
done

scientific_exit=0
for descriptor in d2q9 d3q27; do
    set +e
    python3 -B "${test_dir}/check_convergence.py" "${descriptor}" \
        "${attempt_dir}/${descriptor}/N50/convergence.csv" \
        "${attempt_dir}/${descriptor}/N75/convergence.csv" \
        "${attempt_dir}/${descriptor}/N100/convergence.csv" \
        "${attempt_dir}/log/${descriptor}/N50/console.log" \
        "${attempt_dir}/log/${descriptor}/N75/console.log" \
        "${attempt_dir}/log/${descriptor}/N100/console.log" 2>&1 | \
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
