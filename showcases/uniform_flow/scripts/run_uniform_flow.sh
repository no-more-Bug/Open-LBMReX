#!/usr/bin/env bash
set -euo pipefail

mode="${1:-smoke}"
selection="${2:-all}"

case_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${case_dir}"

np="${LBM_UNIFORM_NP:-1}"
jobs="${LBM_UNIFORM_BUILD_JOBS:-2}"
omp_threads="${LBM_UNIFORM_OMP_THREADS:-1}"
omp_proc_bind="${LBM_UNIFORM_OMP_PROC_BIND:-spread}"
omp_places="${LBM_UNIFORM_OMP_PLACES:-cores}"
timestamp="$(date +%Y-%m-%d_%H-%M-%S-%N)-$$"

for value_name in np omp_threads; do
    value="${!value_name}"
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        printf '%s must be a positive integer, got %s\n' \
            "${value_name}" "${value}" >&2
        exit 2
    fi
done

d2_exe="./build/lbm_uniform_flow"
d3_exe="./build/lbm_uniform_flow_d3q27"
hbb_input="inputs/inputs.d2q9_hbb_velocity_io"
guo_input="inputs/inputs.d2q9_guo_velocity_io"
free_input="inputs/inputs.d2q9_guo_free_outflow"
case_log_root="log"

run_logged() {
    local log_file="$1"
    shift
    local command=(
        env
        TMPDIR=/tmp
        TEMP=/tmp
        TMP=/tmp
        OMPI_MCA_orte_tmpdir_base=/tmp
        OMP_NUM_THREADS="${omp_threads}"
        OMP_DYNAMIC=FALSE
        OMP_PROC_BIND="${omp_proc_bind}"
        OMP_PLACES="${omp_places}"
    )
    if (( np > 1 )); then
        if (( omp_threads == 1 )); then
            command+=(
                mpirun
                -np "${np}"
                --map-by core
                --bind-to core
                --rank-by core
                --report-bindings
            )
        else
            command+=(
                mpirun
                -np "${np}"
                --map-by "ppr:${np}:node:PE=${omp_threads}"
                --bind-to core
                --rank-by core
                --report-bindings
            )
        fi
    fi
    command+=("$@" "amrex.omp_threads=${omp_threads}")

    mkdir -p "$(dirname "${log_file}")"
    {
        printf 'runtime_context: mpi_ranks=%s OMP_NUM_THREADS=%s OMP_DYNAMIC=FALSE OMP_PROC_BIND=%s OMP_PLACES=%s amrex.omp_threads=%s\n' \
            "${np}" "${omp_threads}" "${omp_proc_bind}" "${omp_places}" \
            "${omp_threads}"
        printf 'working_directory = %s\n' "${PWD}"
        printf 'command:'
        printf ' %q' "${command[@]}"
        printf '\n\n'
    } | tee "${log_file}"
    "${command[@]}" 2>&1 | tee -a "${log_file}"
}

configure_if_needed() {
    if [[ ! -f build/CMakeCache.txt ]]; then
        cmake -S ../.. -B build \
            -DAMReX_ROOT=../../../../amrex-install \
            -DCMAKE_BUILD_TYPE=RelWithDebInfo
    fi
    grep -F "CMAKE_BUILD_TYPE:STRING=RelWithDebInfo" build/CMakeCache.txt
}

build_targets() {
    configure_if_needed
    cmake --build build \
        --target lbm_uniform_flow lbm_uniform_flow_d3q27 \
        -j"${jobs}"
}

run_smoke_case() {
    local identity="$1"
    local descriptor="$2"
    local executable="$3"
    local input="$4"
    local steps="$5"
    local expected_outlet="$6"
    shift 6

    local run_dir="outputs/${identity}/${descriptor}/smoke/${timestamp}"
    local log_dir="${case_log_root}/${identity}/${descriptor}/smoke/${timestamp}"
    run_logged "${log_dir}/console.log" \
        "${executable}" "${input}" "$@" \
        run.max_step="${steps}" \
        diagnostics.interval="${steps}" \
        uniform_validation.min_flow_throughs=0 \
        run.kind=smoke \
        output.plot_interval=0 \
        output.write_final_plotfile=0 \
        output.directory="${run_dir}" \
        output.log_directory="${log_dir}"

    grep -F "condition = velocity_inlet -> ${expected_outlet}" \
        "${log_dir}/console.log"
    grep -F "profile = uniform" "${log_dir}/console.log"
    grep -F "uniform_flow check passed" "${log_dir}/console.log"
    grep -F "LBM run complete: steps=${steps}" "${log_dir}/console.log"
    printf '%s %s smoke output: %s\n' \
        "${descriptor}" "${identity}" "${run_dir}"
}

run_smoke() {
    run_smoke_case hbb_velocity_io d2q9 "${d2_exe}" "${hbb_input}" 2 \
        velocity_outlet run.variant=d2q9_hbb_velocity_io_smoke
    run_smoke_case guo_velocity_io d2q9 "${d2_exe}" "${guo_input}" 20 \
        velocity_outlet run.variant=d2q9_guo_velocity_io_smoke
    run_smoke_case guo_velocity_free_outflow d2q9 \
        "${d2_exe}" "${free_input}" 20 free_outflow \
        run.variant=d2q9_guo_velocity_free_outflow_smoke
    run_smoke_case guo_velocity_io d3q27 "${d3_exe}" "${guo_input}" 20 \
        velocity_outlet uniform_flow.depth_cells=4 \
        run.variant=d3q27_guo_velocity_io_smoke
    run_smoke_case guo_velocity_free_outflow d3q27 \
        "${d3_exe}" "${free_input}" 20 free_outflow \
        uniform_flow.depth_cells=4 \
        run.variant=d3q27_guo_velocity_free_outflow_smoke
}

run_baseline_case() {
    local identity="$1"
    local executable="$2"
    local input="$3"
    local depth="$4"
    local attempt="attempt-${timestamp}"
    local run_dir="outputs/baseline/${identity}/${attempt}"
    local log_dir="${case_log_root}/baseline/${identity}/${attempt}"

    mkdir -p "$(dirname "${run_dir}")" "$(dirname "${log_dir}")"
    if ! mkdir "${run_dir}" "${log_dir}"; then
        printf 'Refusing to reuse uniform-flow attempt path: %s or %s\n' \
            "${run_dir}" "${log_dir}" >&2
        return 2
    fi

    run_logged "${log_dir}/console.log" \
        "${executable}" "${input}" \
        uniform_flow.depth_cells="${depth}" \
        run.max_step=10000 \
        diagnostics.interval=10000 \
        uniform_validation.write_metrics=1 \
        uniform_validation.assert=0 \
        uniform_validation.min_flow_throughs=100 \
        run.kind=long-run \
        run.variant="${identity}" \
        output.plot_interval=0 \
        output.write_final_plotfile=0 \
        output.directory="${run_dir}" \
        output.log_directory="${log_dir}"

    python3 scripts/check_uniform_flow.py \
        --identity "${identity}" \
        --artifact "${run_dir}/uniform_flow_metrics.csv" \
        --log "${log_dir}/console.log" \
        --workdir .

    printf '%s baseline output: %s\n' "${identity}" "${run_dir}"
}

run_baseline_identity() {
    local identity="$1"
    case "${identity}" in
        uniform.hbb-velocity-io.d2q9)
            run_baseline_case "${identity}" "${d2_exe}" "${hbb_input}" 1
            ;;
        uniform.hbb-velocity-io.d3q27)
            run_baseline_case "${identity}" "${d3_exe}" "${hbb_input}" 4
            ;;
        uniform.guo-velocity-io.d2q9)
            run_baseline_case "${identity}" "${d2_exe}" "${guo_input}" 1
            ;;
        uniform.guo-velocity-io.d3q27)
            run_baseline_case "${identity}" "${d3_exe}" "${guo_input}" 4
            ;;
        uniform.guo-free-outflow.d2q9)
            run_baseline_case "${identity}" "${d2_exe}" "${free_input}" 1
            ;;
        uniform.guo-free-outflow.d3q27)
            run_baseline_case "${identity}" "${d3_exe}" "${free_input}" 4
            ;;
        *)
            printf 'Unknown uniform-flow baseline identity: %s\n' \
                "${identity}" >&2
            return 2
            ;;
    esac
}

run_baseline() {
    if [[ "${selection}" != all ]]; then
        run_baseline_identity "${selection}"
        return
    fi
    local identities=(
        uniform.hbb-velocity-io.d2q9
        uniform.hbb-velocity-io.d3q27
        uniform.guo-velocity-io.d2q9
        uniform.guo-velocity-io.d3q27
        uniform.guo-free-outflow.d2q9
        uniform.guo-free-outflow.d3q27
    )
    local identity
    for identity in "${identities[@]}"; do
        run_baseline_identity "${identity}"
    done
}

case "${mode}" in
    build)
        build_targets
        ;;
    smoke)
        build_targets
        run_smoke
        ;;
    baseline)
        build_targets
        run_baseline
        ;;
    all)
        build_targets
        run_smoke
        run_baseline
        ;;
    *)
        printf 'Usage: %s [build|smoke|baseline|all] [all|stable-id]\n' "$0" >&2
        exit 2
        ;;
esac
