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
    if [[ "${descriptor}" == d2q9 ]]; then executable="${d2_exe}"; depth=1; else executable="${d3_exe}"; depth=4; fi
    artifacts=()
    for n in 10 20 50; do
        nx=$((4 * n)); max_step=$((200 * n * n)); check_interval=100
        (( n >= 20 )) && check_interval=200
        (( n >= 50 )) && check_interval=500
        read -r u_max rho_in rho_out < <(python3 -c \
            'import sys; n=float(sys.argv[1]); print(1/n, 1+4/n**2, 1-4/n**2)' "${n}")
        run_dir="${attempt_dir}/${descriptor}/N${n}"
        log_dir="${attempt_dir}/log/${descriptor}/N${n}"
        mkdir -p "${run_dir}" "${log_dir}" || exit 2
        command=(
            "${mpiexec}" -np 4 --map-by core --bind-to core --rank-by slot
            --report-bindings "${executable}" "${test_dir}/inputs/guo_pressure"
            poiseuille_flow.intervals="${nx}" "${n}"
            poiseuille_flow.depth_cells="${depth}"
            poiseuille_flow.u_max_lb="${u_max}"
            poiseuille_flow.tau=0.75
            poiseuille_flow.rho_in_lb="${rho_in}"
            poiseuille_flow.rho_out_lb="${rho_out}"
            poiseuille_flow.pressure_scaling=fixed_re_fixed_tau
            mesh.max_grid_size=32 run.max_step="${max_step}"
            diagnostics.interval="${check_interval}"
            diagnostics.steady_check_interval="${check_interval}"
            diagnostics.stop_on_steady=1
            poiseuille_validation.assert=0 poiseuille_validation.mass_assert=0
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
        artifacts+=("${run_dir}/poiseuille_profile_metrics.csv")
    done
    set +e
    python3 -B "${test_dir}/check_convergence.py" "${descriptor}" \
        "${artifacts[@]}" \
        "${attempt_dir}/log/${descriptor}/N10/console.log" \
        "${attempt_dir}/log/${descriptor}/N20/console.log" \
        "${attempt_dir}/log/${descriptor}/N50/console.log" 2>&1 | \
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
