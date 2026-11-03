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

artifacts=()
for variant in hbb_velocity_io guo_velocity_io guo_free_outflow; do
    for descriptor in d2q9 d3q27; do
        if [[ "${descriptor}" == d2q9 ]]; then executable="${d2_exe}"; depth=1; else executable="${d3_exe}"; depth=4; fi
        run_dir="${attempt_dir}/${variant}/${descriptor}"
        log_dir="${attempt_dir}/log/${variant}/${descriptor}"
        mkdir -p "${run_dir}" "${log_dir}" || exit 2
        command=(
            "${mpiexec}" -np 1 --map-by core --bind-to core --rank-by slot
            --report-bindings "${executable}" "${test_dir}/inputs/${variant}"
            uniform_flow.depth_cells="${depth}"
            run.max_step=10000 diagnostics.interval=10000
            uniform_validation.write_metrics=1 uniform_validation.assert=0
            uniform_validation.min_flow_throughs=100
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
        artifacts+=("${variant}.${descriptor}|${run_dir}/uniform_flow_metrics.csv|${log_dir}/console.log")
    done
done
set +e
python3 -B "${test_dir}/check_uniform_flow.py" "${artifacts[@]}" 2>&1 | \
    tee "${attempt_dir}/checker.log"
checker_status=("${PIPESTATUS[@]}")
printf 'checker_exit=%d tee_exit=%d\n' "${checker_status[0]}" \
    "${checker_status[1]}" >>"${attempt_dir}/checker.log" || exit 2
(( checker_status[1] == 0 )) || exit 2
checker_exit="${checker_status[0]}"
(( checker_exit == 0 || checker_exit == 1 )) && exit "${checker_exit}"
exit 2
