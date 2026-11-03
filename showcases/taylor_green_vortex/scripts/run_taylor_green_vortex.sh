#!/usr/bin/env bash
set -euo pipefail

mode="${1:-smoke}"

case_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
lbm_root="$(cd "${case_dir}/../.." && pwd)"
cd "${case_dir}"

np="${LBM_TGV_NP:-8}"
jobs="${LBM_TGV_BUILD_JOBS:-2}"
omp_threads="${LBM_TGV_OMP_THREADS:-1}"
omp_proc_bind="${LBM_TGV_OMP_PROC_BIND:-spread}"
omp_places="${LBM_TGV_OMP_PLACES:-cores}"
timestamp="$(date +%Y-%m-%d_%H-%M-%S-%N)-$$"

require_positive_integer() {
    local value_name="$1"
    local value="$2"
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        printf '%s must be a positive integer, got %s\n' \
            "${value_name}" "${value}" >&2
        return 2
    fi
}

d2_exe="./build/lbm_taylor_green_vortex"
d3_exe="./build/lbm_taylor_green_vortex_d3q27"
d2_input="inputs/inputs.d2q9_periodic_2d"
d3_input="inputs/inputs.d3q27_quasi2d"
d2_quasi2d_input="inputs/inputs.d2q9_quasi2d"
d2_convergence_input="inputs/inputs.d2q9_tau_convergence"
d3_convergence_input="inputs/inputs.d3q27_tau_convergence"

d2_root="outputs/d2q9_periodic_2d"
d3_root="outputs/d3q27_quasi_2d"
case_log_root="log"

run_logged() {
    local log_file="$1"
    shift
    run_parallel_logged \
        "${log_file}" "${np}" "${omp_threads}" \
        "${omp_proc_bind}" "${omp_places}" "" "$@"
}

run_parallel_logged() {
    local log_file="$1"
    local ranks="$2"
    local threads="$3"
    local proc_bind="$4"
    local places="$5"
    local timeout_seconds="$6"
    shift 6

    require_positive_integer ranks "${ranks}" || return 2
    require_positive_integer threads "${threads}" || return 2

    local command=()
    if (( ranks > 1 )); then
        if (( threads == 1 )); then
            command+=(
                mpirun
                -np "${ranks}"
                --map-by core
                --bind-to core
                --rank-by core
                --report-bindings
            )
        else
            command+=(
                mpirun
                -np "${ranks}"
                --map-by "ppr:${ranks}:node:PE=${threads}"
                --bind-to core
                --rank-by core
                --report-bindings
            )
        fi
    fi
    command+=("$@" "amrex.omp_threads=${threads}")
    local prefix=()
    if [[ -n "${timeout_seconds}" ]]; then
        require_positive_integer timeout_seconds "${timeout_seconds}" || return 2
        prefix=(
            timeout
            --foreground
            --signal=TERM
            --kill-after=30s
            "${timeout_seconds}s"
        )
    fi
    local invocation=("${prefix[@]}" "${command[@]}")

    mkdir -p "$(dirname "${log_file}")"
    {
        printf 'runtime_context: mpi_ranks=%s OMP_NUM_THREADS=%s OMP_DYNAMIC=FALSE OMP_PROC_BIND=%s OMP_PLACES=%s amrex.omp_threads=%s\n' \
            "${ranks}" "${threads}" "${proc_bind}" "${places}" \
            "${threads}"
        printf 'working_directory = %s\n' "${PWD}"
        if (( ${#prefix[@]} > 0 )); then
            printf 'execution_prefix:'
            printf ' %q' "${prefix[@]}"
            printf '\n'
        fi
        printf 'command:'
        printf ' %q' "${command[@]}"
        printf '\n\n'
    } >"${log_file}"

    local pipeline_status=()
    set +e
    OMP_NUM_THREADS="${threads}" \
    OMP_DYNAMIC=FALSE \
    OMP_PROC_BIND="${proc_bind}" \
    OMP_PLACES="${places}" \
        "${invocation[@]}" 2>&1 | tee -a "${log_file}"
    pipeline_status=("${PIPESTATUS[@]}")
    set -e

    local solver_code="${pipeline_status[0]}"
    local tee_code="${pipeline_status[1]}"
    printf 'solver_exit_code = %s\ntee_exit_code = %s\n' \
        "${solver_code}" "${tee_code}" >>"${log_file}" || return 1
    if (( solver_code != 0 )); then
        return "${solver_code}"
    fi
    if (( tee_code != 0 )); then
        return "${tee_code}"
    fi
}

run_checker_logged() {
    local log_file="$1"
    shift
    local checker=("$@")
    {
        printf 'checker_working_directory = %s\n' "${lbm_root}"
        printf 'checker_command:'
        printf ' %q' "${checker[@]}"
        printf '\n'
    } >>"${log_file}" || return 1

    local pipeline_status=()
    set +e
    (
        cd "${lbm_root}"
        "${checker[@]}"
    ) 2>&1 | tee -a "${log_file}"
    pipeline_status=("${PIPESTATUS[@]}")
    set -e

    local checker_code="${pipeline_status[0]}"
    local tee_code="${pipeline_status[1]}"
    printf 'checker_exit_code = %s\nchecker_tee_exit_code = %s\n' \
        "${checker_code}" "${tee_code}" >>"${log_file}" || return 1
    if (( checker_code != 0 )); then
        return "${checker_code}"
    fi
    if (( tee_code != 0 )); then
        return "${tee_code}"
    fi
}

configure_if_needed() {
    if [[ ! -f build/CMakeCache.txt ]]; then
        cmake -S ../.. -B build \
            -DAMReX_ROOT=../../../../amrex-install \
            -DCMAKE_BUILD_TYPE=RelWithDebInfo
    fi
    grep -F "CMAKE_BUILD_TYPE:STRING=RelWithDebInfo" \
        build/CMakeCache.txt
}

build_targets() {
    configure_if_needed
    cmake --build build \
        --target lbm_taylor_green_vortex lbm_taylor_green_vortex_d3q27 \
        -j"${jobs}"
}

run_smoke() {
    local d2_log_dir="${case_log_root}/d2q9_periodic_2d/smoke/${timestamp}"
    local d3_log_dir="${case_log_root}/d3q27_quasi_2d/smoke/${timestamp}"
    run_logged \
        "${d2_log_dir}/console.log" \
        "${d2_exe}" "${d2_input}" \
        taylor_green.n_cell=16 16 1 \
        mesh.max_grid_size=16 \
        run.max_step=2 \
        output.plot_interval=2 \
        diagnostics.interval=1 \
        run.kind=smoke \
        output.directory="${d2_root}/smoke/${timestamp}" \
        output.log_directory="${d2_log_dir}"

    run_logged \
        "${d3_log_dir}/console.log" \
        "${d3_exe}" "${d3_input}" \
        taylor_green.n_cell=16 16 4 \
        mesh.max_grid_size=16 \
        run.max_step=2 \
        output.plot_interval=2 \
        diagnostics.interval=1 \
        run.kind=smoke \
        output.directory="${d3_root}/smoke/${timestamp}" \
        output.log_directory="${d3_log_dir}"
}

run_quasi2d_smoke() {
    local root="outputs/d2q9_quasi_2d/smoke/${timestamp}"
    local log_dir="${case_log_root}/d2q9_quasi_2d/smoke/${timestamp}"
    run_logged \
        "${log_dir}/console.log" \
        "${d2_exe}" "${d2_quasi2d_input}" \
        run.max_step=2 \
        output.plot_interval=0 \
        diagnostics.interval=1 \
        output.write_final_plotfile=0 \
        run.kind=smoke \
        output.directory="${root}" \
        output.log_directory="${log_dir}"
    grep -F "LBM run complete: steps=2" \
        "${log_dir}/console.log"
    printf 'TGV D2Q9 quasi-2D smoke output: %s\n' "${root}"
}

resolve_entry_identity() {
    local entry_id="$1"
    if [[ ! "${entry_id}" =~ ^tgv\.(d2q9|d3q27)\.n(50|75|100)$ ]]; then
        printf 'Unknown TGV baseline entry: %s\n' "${entry_id}" >&2
        return 2
    fi

    resolved_entry_id="${entry_id}"
    local family="${BASH_REMATCH[1]}"
    resolved_n="${BASH_REMATCH[2]}"
    resolved_label="$(printf 'N%03d' "${resolved_n}")"
    case "${family}" in
        d2q9)
            resolved_exe="${d2_exe}"
            resolved_input="${d2_convergence_input}"
            resolved_root="${d2_root}"
            resolved_depth=1
            resolved_sequence="tau0p75_Re10_ft0p2_N50_75_100"
            resolved_variant="d2q9_periodic_2d"
            ;;
        d3q27)
            resolved_exe="${d3_exe}"
            resolved_input="${d3_convergence_input}"
            resolved_root="${d3_root}"
            resolved_depth=4
            resolved_sequence="tau0p75_Re10_ft0p2_N50_75_100_nz4"
            resolved_variant="d3q27_quasi_2d"
            ;;
    esac
}

resolve_attempt_paths() {
    local attempt_id="$1"
    resolved_output_attempt_root="${resolved_root}/convergence/${resolved_sequence}/${attempt_id}"
    resolved_log_attempt_root="${case_log_root}/${resolved_variant}/convergence/${resolved_sequence}/${attempt_id}"
    resolved_run_dir="${resolved_output_attempt_root}/${resolved_label}"
    resolved_log_dir="${resolved_log_attempt_root}/${resolved_label}"
    resolved_log_file="${resolved_log_dir}/console.log"
    resolved_artifact="${resolved_run_dir}/convergence.csv"
}

create_attempt_roots() {
    mkdir -p \
        "$(dirname "${resolved_output_attempt_root}")" \
        "$(dirname "${resolved_log_attempt_root}")"
    if ! mkdir "${resolved_output_attempt_root}" "${resolved_log_attempt_root}"; then
        printf 'Refusing to reuse TGV attempt path: %s or %s\n' \
            "${resolved_output_attempt_root}" "${resolved_log_attempt_root}" >&2
        return 2
    fi
}

run_resolved_convergence_entry() {
    local ranks="$1"
    local threads="$2"
    local proc_bind="$3"
    local places="$4"
    local timeout_seconds="$5"
    run_parallel_logged \
        "${resolved_log_file}" "${ranks}" "${threads}" \
        "${proc_bind}" "${places}" "${timeout_seconds}" \
        "${resolved_exe}" "${resolved_input}" \
        taylor_green.n_cell="${resolved_n}" "${resolved_n}" "${resolved_depth}" \
        mesh.max_grid_size=25 \
        output.directory="${resolved_run_dir}" \
        run.convergence_enabled=1 \
        run.kind=convergence \
        run.convergence_reset=1 \
        output.convergence_directory=. \
        output.log_directory="${resolved_log_dir}" \
        run.final_time_phys=0.2 \
        taylor_green.u_ref_phys=1.0 \
        output.plot_interval=0
}

preflight_formal_entry() {
    local entry_id="$1"
    local attempt_id="$2"
    local timeout_seconds="$3"

    resolve_entry_identity "${entry_id}" || return 2
    if [[ ! "${attempt_id}" =~ ^attempt-[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{9}-[0-9]+$ ]]; then
        printf 'Invalid TGV attempt ID: %s\n' "${attempt_id}" >&2
        return 2
    fi
    require_positive_integer timeout_seconds "${timeout_seconds}" || return 2
    resolve_attempt_paths "${attempt_id}"

    if [[ ! -x "${resolved_exe}" ]]; then
        printf 'TGV direct executable is not executable: %s\n' "${resolved_exe}" >&2
        return 2
    fi
    if [[ ! -f "${resolved_input}" ]]; then
        printf 'TGV control source does not exist: %s\n' "${resolved_input}" >&2
        return 2
    fi
    if [[ ! -f "${lbm_root}/showcases/taylor_green_vortex/scripts/check_tgv_convergence.py" ]]; then
        printf 'TGV entry checker does not exist\n' >&2
        return 2
    fi
    local tool
    for tool in mpirun timeout tee python3; do
        if ! command -v "${tool}" >/dev/null 2>&1; then
            printf 'Required TGV entry tool is unavailable: %s\n' "${tool}" >&2
            return 2
        fi
    done
    if [[ -e "${resolved_output_attempt_root}" ||
          -L "${resolved_output_attempt_root}" ||
          -e "${resolved_log_attempt_root}" ||
          -L "${resolved_log_attempt_root}" ]]; then
        printf 'Refusing to reuse TGV attempt path: %s or %s\n' \
            "${resolved_output_attempt_root}" "${resolved_log_attempt_root}" >&2
        return 2
    fi
}

run_formal_entry() {
    local entry_id="$1"
    local attempt_id="$2"
    local timeout_seconds="$3"
    preflight_formal_entry "${entry_id}" "${attempt_id}" "${timeout_seconds}" ||
        return 2
    create_attempt_roots || return 2
    mkdir "${resolved_run_dir}" "${resolved_log_dir}"

    local run_code=0
    run_resolved_convergence_entry 4 1 close cores "${timeout_seconds}" ||
        run_code=$?
    if (( run_code != 0 )); then
        return "${run_code}"
    fi

    local checker_artifact="showcases/taylor_green_vortex/${resolved_artifact}"
    local checker_log="showcases/taylor_green_vortex/${resolved_log_file}"
    local checker=(
        python3
        -B
        showcases/taylor_green_vortex/scripts/check_tgv_convergence.py
        --entry-id
        "${resolved_entry_id}"
        --workdir
        showcases/taylor_green_vortex
        --artifact
        "${checker_artifact}"
        --log
        "${checker_log}"
    )
    local checker_code=0
    run_checker_logged "${resolved_log_file}" "${checker[@]}" ||
        checker_code=$?
    if (( checker_code != 0 )); then
        return "${checker_code}"
    fi

    printf '%s formal output: %s\n' "${resolved_entry_id}" "${resolved_run_dir}"
}

run_convergence_case() {
    local descriptor="$1"
    local entry_prefix
    case "${descriptor}" in
        D2Q9) entry_prefix="tgv.d2q9" ;;
        D3Q27) entry_prefix="tgv.d3q27" ;;
        *) printf 'Unknown TGV descriptor: %s\n' "${descriptor}" >&2; return 2 ;;
    esac
    local attempt_name="attempt-${timestamp}"
    local artifact_args=()
    local log_args=()

    resolve_entry_identity "${entry_prefix}.n50"
    resolve_attempt_paths "${attempt_name}"
    create_attempt_roots || return 2
    local summary="${resolved_output_attempt_root}/fit_relative_l2_u.txt"

    for n in 50 75 100; do
        resolve_entry_identity "${entry_prefix}.n${n}"
        resolve_attempt_paths "${attempt_name}"
        mkdir "${resolved_run_dir}" "${resolved_log_dir}"
        run_resolved_convergence_entry \
            "${np}" "${omp_threads}" "${omp_proc_bind}" "${omp_places}" ""
        artifact_args+=(--artifact "${n}=${resolved_artifact}")
        log_args+=(--log "${n}=${resolved_log_file}")
    done

    python3 scripts/check_tgv_convergence.py \
        --descriptor "${descriptor}" \
        --workdir . \
        --summary "${summary}" \
        "${artifact_args[@]}" \
        "${log_args[@]}"

    printf '%s convergence output: %s\n' \
        "${descriptor}" "${resolved_output_attempt_root}"
}

run_convergence() {
    run_convergence_case D2Q9
    run_convergence_case D3Q27
}

case "${mode}" in
    entry)
        if (( $# != 4 )); then
            printf 'Usage: %s entry <entry-id> <attempt-id> <timeout-seconds>\n' \
                "$0" >&2
            exit 2
        fi
        run_formal_entry "$2" "$3" "$4"
        ;;
    smoke)
        build_targets
        run_smoke
        ;;
    quasi2d-smoke)
        build_targets
        run_quasi2d_smoke
        ;;
    convergence)
        build_targets
        run_convergence
        ;;
    all)
        build_targets
        run_smoke
        run_quasi2d_smoke
        run_convergence
        ;;
    *)
        printf 'Usage: %s [smoke|quasi2d-smoke|convergence|all] or\n' "$0" >&2
        printf '       %s entry <entry-id> <attempt-id> <timeout-seconds>\n' "$0" >&2
        exit 2
        ;;
esac
