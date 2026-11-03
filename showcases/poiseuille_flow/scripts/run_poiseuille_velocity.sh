#!/usr/bin/env bash
set -euo pipefail

mode="${1:-d2-convergence}"
case_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
demo_dir="$(cd "${case_dir}/../.." && pwd)"
cd "${demo_dir}"

jobs="${LBM_POISEUILLE_BUILD_JOBS:-2}"
mpi_ranks="${LBM_POISEUILLE_MPI_RANKS:-1}"
omp_threads="${LBM_POISEUILLE_OMP_THREADS:-1}"
omp_proc_bind="${LBM_POISEUILLE_OMP_PROC_BIND:-spread}"
omp_places="${LBM_POISEUILLE_OMP_PLACES:-cores}"
timestamp="$(date +%Y-%m-%d_%H-%M-%S-%N)-$$"
build_dir="showcases/poiseuille_flow/build"
case_log_root="showcases/poiseuille_flow/log"

for value_name in mpi_ranks omp_threads; do
    value="${!value_name}"
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        printf '%s must be a positive integer, got %s\n' \
            "${value_name}" "${value}" >&2
        exit 2
    fi
done

d2_exe="./${build_dir}/lbm_poiseuille_flow"
d3_exe="./${build_dir}/lbm_poiseuille_flow_d3q27"
input="showcases/poiseuille_flow/inputs/inputs.d2q9_hbb_velocity_profile"
velocity_nu_lb="0.083333333333333333"
velocity_tau="0.75"
velocity_re_target="12.0"
RUN_CASE_DIR=""

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
    if (( mpi_ranks > 1 )); then
        if (( omp_threads == 1 )); then
            command+=(
                mpirun
                -np "${mpi_ranks}"
                --map-by core
                --bind-to core
                --rank-by core
                --report-bindings
            )
        else
            command+=(
                mpirun
                -np "${mpi_ranks}"
                --map-by "ppr:${mpi_ranks}:node:PE=${omp_threads}"
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
            "${mpi_ranks}" "${omp_threads}" "${omp_proc_bind}" \
            "${omp_places}" "${omp_threads}"
        printf 'working_directory = %s\n' "${PWD}"
        printf 'command:'
        printf ' %q' "${command[@]}"
        printf '\n\n'
    } | tee "${log_file}"
    "${command[@]}" 2>&1 | tee -a "${log_file}"
}

configure_if_needed() {
    if [[ ! -f "${build_dir}/CMakeCache.txt" ]]; then
        cmake -S . -B "${build_dir}" \
            -DAMReX_ROOT=../../amrex-install \
            -DCMAKE_BUILD_TYPE=RelWithDebInfo
    fi
    grep -F "CMAKE_BUILD_TYPE:STRING=RelWithDebInfo" "${build_dir}/CMakeCache.txt"
}

build_targets() {
    configure_if_needed
    cmake --build "${build_dir}" --target lbm_poiseuille_flow lbm_poiseuille_flow_d3q27 -j"${jobs}"
}

physical_value() {
    python3 -c 'import sys; print(f"{float(sys.argv[1]):.17g}")' "$1"
}

run_case() {
    local descriptor="$1"
    local exe="$2"
    local ny="$3"
    local nz="$4"
    local run_kind="$5"
    local nx=$((4 * ny))
    local max_step=$((200 * ny * ny))
    local check_int=100
    if (( ny >= 20 )); then
        check_int=200
    fi
    if (( ny >= 50 )); then
        check_int=500
    fi
    local run_dir="showcases/poiseuille_flow/outputs/velocity_profile/${descriptor}/${run_kind}/N${ny}/${timestamp}"
    local log_dir="${case_log_root}/velocity_profile/${descriptor}/${run_kind}/N${ny}/${timestamp}"
    local variant="${descriptor}_velocity_profile_poiseuille_N${ny}"
    local u_max_lb
    # Refine one physical problem: fixed Re and fixed tau imply the diffusive
    # scaling U_ref,lb = Re * nu_lb / N.  The historical fixed-U sequence
    # changed Re=12/24/60 and is characterization rather than a formal order
    # test.
    u_max_lb="$(physical_value "$(python3 -c \
        'import sys; print(float(sys.argv[1])*float(sys.argv[2])/int(sys.argv[3]))' \
        "${velocity_re_target}" "${velocity_nu_lb}" "${ny}")")"
    run_logged "${log_dir}/console.log" "${exe}" "${input}" \
        poiseuille_flow.intervals="${nx}" "${ny}" \
        poiseuille_flow.depth_cells="${nz}" \
        poiseuille_flow.u_max_lb="${u_max_lb}" \
        poiseuille_flow.tau="${velocity_tau}" \
        mesh.max_grid_size=32 \
        run.variant="${variant}" \
        run.kind="${run_kind}" \
        run.max_step="${max_step}" \
        output.plot_interval="${max_step}" \
        diagnostics.interval="${check_int}" \
        diagnostics.steady_check_interval="${check_int}" \
        output.directory="${run_dir}" \
        output.log_directory="${log_dir}"
    grep -q "poiseuille_flow check passed" "${log_dir}/console.log"
    grep -F "physical_intervals = (${nx}, ${ny})" "${log_dir}/console.log"
    grep -F "nu_lb = 0.0833333333333" "${log_dir}/console.log"
    python3 -c '
import csv
import math
import sys

path, expected_u_text, expected_re_text = sys.argv[1:]
with open(path, newline="") as stream:
    rows = list(csv.DictReader(stream))
if len(rows) != 1:
    raise SystemExit(f"{path}: expected one metrics row, found {len(rows)}")
row = rows[0]
for key, expected in (
    ("u_max_lb", float(expected_u_text)),
    ("re_effective", float(expected_re_text)),
):
    value = float(row[key])
    if not math.isfinite(value) or not math.isclose(
        value, expected, rel_tol=2.0e-12, abs_tol=2.0e-14
    ):
        raise SystemExit(
            f"{path}: {key}={value} does not match expected {expected}"
        )
' "${run_dir}/poiseuille_profile_metrics.csv" \
        "${u_max_lb}" "${velocity_re_target}"
    test -d "${run_dir}/plt00000"
    local final_plot
    final_plot="$(find "${run_dir}" -maxdepth 1 -type d \
        -regextype posix-extended -regex '.*/plt[0-9]+' | sort -V | tail -n 1)"
    test -n "${final_plot}"
    test "${final_plot}" != "${run_dir}/plt00000"
    python3 showcases/poiseuille_flow/scripts/plot_poiseuille_profiles.py "${run_dir}"
    RUN_CASE_DIR="${run_dir}"
    printf '%s\n' "${run_dir}"
}

run_convergence() (
    local group="$1"
    local executable="$2"
    local nz="$3"
    cd "${case_dir}"

    local attempt="attempt-${timestamp}"
    local root="outputs/baseline/${group}/${attempt}"
    local log_root="log/baseline/${group}/${attempt}"
    mkdir -p "$(dirname "${root}")" "$(dirname "${log_root}")"
    if ! mkdir "${root}" "${log_root}"; then
        printf 'Refusing to reuse Poiseuille attempt path: %s or %s\n' \
            "${root}" "${log_root}" >&2
        return 2
    fi

    local artifact_args=()
    local log_args=()
    local ny
    for ny in 10 20 50; do
        local nx=$((4 * ny))
        local max_step=$((200 * ny * ny))
        local check_int=100
        if (( ny >= 20 )); then
            check_int=200
        fi
        if (( ny >= 50 )); then
            check_int=500
        fi
        local entry="${group%.slope}.n${ny}"
        local run_dir="${root}/N${ny}"
        local log_dir="${log_root}/N${ny}"
        local u_max_lb
        u_max_lb="$(physical_value "$(python3 -c \
            'import sys; print(float(sys.argv[1])*float(sys.argv[2])/int(sys.argv[3]))' \
            "${velocity_re_target}" "${velocity_nu_lb}" "${ny}")")"
        mkdir "${run_dir}"
        run_logged "${log_dir}/console.log" \
            "${executable}" "inputs/inputs.d2q9_hbb_velocity_profile" \
            poiseuille_flow.intervals="${nx}" "${ny}" \
            poiseuille_flow.depth_cells="${nz}" \
            poiseuille_flow.u_max_lb="${u_max_lb}" \
            poiseuille_flow.tau="${velocity_tau}" \
            mesh.max_grid_size=32 \
            run.variant="${entry}" \
            run.kind=convergence \
            run.max_step="${max_step}" \
            diagnostics.interval="${check_int}" \
            diagnostics.steady_check_interval="${check_int}" \
            diagnostics.steady_l2_tolerance=5.0e-8 \
            diagnostics.stop_on_steady=1 \
            poiseuille_validation.write_profile=1 \
            poiseuille_validation.assert=0 \
            poiseuille_validation.mass_assert=0 \
            output.plot_interval=0 \
            output.write_final_plotfile=0 \
            output.directory="${run_dir}" \
            output.log_directory="${log_dir}"
        artifact_args+=(--artifact "${ny}=${run_dir}/poiseuille_profile_metrics.csv")
        log_args+=(--log "${ny}=${log_dir}/console.log")
    done

    python3 scripts/check_poiseuille_convergence.py \
        --group "${group}" \
        --workdir . \
        --output "${root}" \
        "${artifact_args[@]}" \
        "${log_args[@]}"
    printf '%s Poiseuille convergence output: %s\n' "${group}" "${root}"
)

run_d2_convergence() {
    run_convergence \
        poiseuille.hbb-velocity.d2q9.slope \
        ./build/lbm_poiseuille_flow \
        1
}

run_d3_convergence() {
    run_convergence \
        poiseuille.hbb-velocity.d3q27.slope \
        ./build/lbm_poiseuille_flow_d3q27 \
        4
}

run_d3_steady() {
    run_case "d3q27" "${d3_exe}" 10 4 "steady"
}

case "${mode}" in
    build)
        build_targets
        ;;
    d2-steady)
        build_targets
        run_case "d2q9" "${d2_exe}" 10 1 "steady"
        ;;
    d2-convergence)
        build_targets
        run_d2_convergence
        ;;
    d3-steady)
        build_targets
        run_d3_steady
        ;;
    d3-convergence)
        build_targets
        run_d3_convergence
        ;;
    all)
        build_targets
        run_d2_convergence
        run_d3_convergence
        ;;
    *)
        echo "Usage: $0 [build|d2-steady|d2-convergence|d3-steady|d3-convergence|all]" >&2
        exit 2
        ;;
esac
