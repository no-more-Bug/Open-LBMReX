#!/usr/bin/env bash
set -uo pipefail

if (( $# != 4 )); then
    printf 'Usage: %s <d2-executable> <d3-executable> <work-dir> <mpiexec>\n' "$0" >&2
    exit 2
fi
d2_exe="$1"; d3_exe="$2"; work_dir="$3"; mpiexec="$4"
test_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
attempt_dir="${work_dir}/attempt-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "${attempt_dir}/log" || exit 2

run_solver() {
    local name="$1" executable="$2" input="$3" ranks="$4" threads="$5"
    local run_dir="${attempt_dir}/${name}" log_dir="${attempt_dir}/log/${name}"
    local tmp_dir="${attempt_dir}/tmp/${name}"
    local mapping
    if [[ "${LBM_TEST_R5_MPI_RANKS:-0}" != 0 ]]; then
        [[ "${LBM_TEST_R5_MPI_RANKS}" =~ ^[1-9][0-9]*$ ]] || return 2
        ranks="${LBM_TEST_R5_MPI_RANKS}"
        threads=1
    fi
    mkdir -p "${run_dir}" "${log_dir}" "${tmp_dir}" || return 2
    if (( threads == 1 )); then
        mapping="core"
    else
        mapping="ppr:${ranks}:node:PE=${threads}"
    fi
    local -a command=(
        "${mpiexec}" -np "${ranks}" --map-by "${mapping}" --bind-to core
        --rank-by slot --report-bindings "${executable}" "${input}"
        output.directory="${run_dir}" output.log_directory="${log_dir}"
        amrex.omp_threads="${threads}" amrex.use_gpu_aware_mpi=0)
    {
        printf 'start_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf 'layout=%s_MPI_x_%s_OMP map=%s bind=core rank_by=slot\n' \
            "${ranks}" "${threads}" "${mapping}"
        printf 'runtime_tmp=%s\n' "${tmp_dir}"
        printf 'command:'
        printf ' %q' "${command[@]}"
        printf '\n'
        printf 'executable_sha256=%s\n' "$(sha256sum "${executable}" | awk '{print $1}')"
        printf 'input_sha256=%s\n' "$(sha256sum "${input}" | awk '{print $1}')"
    } >"${log_dir}/console.log" || return 2
    set +e
    env TMPDIR="${tmp_dir}" TEMP="${tmp_dir}" TMP="${tmp_dir}" \
        OMPI_MCA_orte_tmpdir_base="${tmp_dir}" \
        OMP_NUM_THREADS="${threads}" OMP_DYNAMIC=FALSE \
        OMP_PROC_BIND=close OMP_PLACES=cores \
        "${command[@]}" 2>&1 | tee -a "${log_dir}/console.log"
    local -a statuses=("${PIPESTATUS[@]}")
    set -e
    {
        printf 'solver_exit=%d tee_exit=%d\n' "${statuses[0]}" "${statuses[1]}"
        printf 'end_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    } >>"${log_dir}/console.log" || return 2
    (( statuses[0] == 0 && statuses[1] == 0 )) || return 2
}

run_solver re20_d2q9_nd20 "${d2_exe}" "${test_dir}/inputs/re20_d2q9_nd20" 1 18 || exit 2
run_solver re20_d3q27_nd20_nz4 "${d3_exe}" "${test_dir}/inputs/re20_d3q27_nd20_nz4" 1 18 || exit 2
run_solver re20_d2q9_nd40 "${d2_exe}" "${test_dir}/inputs/re20_d2q9_nd40" 8 2 || exit 2
run_solver re20_d2q9_nd80 "${d2_exe}" "${test_dir}/inputs/re20_d2q9_nd80" 8 2 || exit 2
run_solver re100_d2q9_nd40 "${d2_exe}" "${test_dir}/inputs/re100_d2q9_nd40" 18 1 || exit 2
run_solver re100_d2q9_nd80 "${d2_exe}" "${test_dir}/inputs/re100_d2q9_nd80" 9 2 || exit 2

scientific_exit=0
set +e
python3 -B "${test_dir}/check_re20.py" \
    "${test_dir}/reference-data/dfg_re20_reference.csv" \
    "${attempt_dir}/re20_d2q9_nd20" \
    "${attempt_dir}/re20_d3q27_nd20_nz4" \
    "${attempt_dir}/re20_d2q9_nd40" \
    "${attempt_dir}/re20_d2q9_nd80" 2>&1 | tee "${attempt_dir}/re20_checker.log"
checker_status=("${PIPESTATUS[@]}")
set -e
(( checker_status[1] == 0 )) || exit 2
if (( checker_status[0] == 1 )); then scientific_exit=1; elif (( checker_status[0] != 0 )); then exit 2; fi

for resolution in nd40 nd80; do
    history="${attempt_dir}/re100_d2q9_${resolution}/channel_cylinder_force_history.csv"
    result="${attempt_dir}/re100_d2q9_${resolution}/channel_cylinder_periodic_metrics.csv"
    set +e
    if [[ "${resolution}" == nd40 ]]; then steps=400000; diameter=40; else steps=800000; diameter=80; fi
    python3 -B "${test_dir}/analyze_re100.py" "${history}" \
        "${test_dir}/reference-data/dfg_re100_reference.csv" "${result}" 250 \
        "${steps}" "${diameter}" \
        2>&1 | tee "${attempt_dir}/re100_${resolution}_checker.log"
    checker_status=("${PIPESTATUS[@]}")
    set -e
    (( checker_status[1] == 0 )) || exit 2
    if (( checker_status[0] == 1 )); then scientific_exit=1; elif (( checker_status[0] != 0 )); then exit 2; fi
done

sha256sum "${test_dir}/check_re20.py" "${test_dir}/analyze_re100.py" \
    "${test_dir}/reference-data/dfg_re20_reference.csv" \
    "${test_dir}/reference-data/dfg_re100_reference.csv" \
    >"${attempt_dir}/authority-digests.sha256" || exit 2
exit "${scientific_exit}"
