#!/usr/bin/env bash
set -euo pipefail

mode="${1:-velocity-d2-convergence}"
case_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
demo_dir="$(cd "${case_dir}/../.." && pwd)"
cd "${demo_dir}"

jobs="${LBM_POISEUILLE_BUILD_JOBS:-4}"
timestamp="$(date +%Y-%m-%d_%H-%M-%S-%N)-$$"
build_dir="showcases/poiseuille_flow/build"
case_log_root="showcases/poiseuille_flow/log"
mpi_ranks="${LBM_POISEUILLE_MPI_RANKS:-1}"
omp_threads="${LBM_POISEUILLE_OMP_THREADS:-1}"
omp_dynamic="${LBM_POISEUILLE_OMP_DYNAMIC:-FALSE}"
omp_proc_bind="${LBM_POISEUILLE_OMP_PROC_BIND:-spread}"
omp_places="${LBM_POISEUILLE_OMP_PLACES:-cores}"

if [[ "${omp_dynamic}" != "FALSE" ]]; then
    printf 'LBM_POISEUILLE_OMP_DYNAMIC must remain FALSE, got %s\n' \
        "${omp_dynamic}" >&2
    exit 2
fi

d2_exe="./${build_dir}/lbm_poiseuille_flow"
d3_exe="./${build_dir}/lbm_poiseuille_flow_d3q27"
velocity_input="showcases/poiseuille_flow/inputs/inputs.d2q9_guo_velocity"
pressure_input="showcases/poiseuille_flow/inputs/inputs.d2q9_guo_pressure"
prescribed_state_input="showcases/poiseuille_flow/inputs/inputs.d2q9_guo_prescribed_state"
velocity_nu_lb="0.083333333333333333"
velocity_tau="0.75"
velocity_re_target="12.0"
pressure_nu_lb="0.083333333333333333"
pressure_tau="0.75"
pressure_re_target="12.0"
RUN_CASE_DIR=""

run_logged() {
    local log_file="$1"
    shift
    run_parallel_logged \
        "${log_file}" "${mpi_ranks}" "${omp_threads}" "$@"
}

run_parallel_logged() {
    local log_file="$1"
    local ranks="$2"
    local threads="$3"
    shift 3
    if [[ ! "${ranks}" =~ ^[1-9][0-9]*$ ]] ||
       [[ ! "${threads}" =~ ^[1-9][0-9]*$ ]]; then
        printf 'ranks and threads must be positive integers: %s %s\n' \
            "${ranks}" "${threads}" >&2
        return 2
    fi
    local command=(
        env
        TMPDIR=/tmp
        TEMP=/tmp
        TMP=/tmp
        OMPI_MCA_orte_tmpdir_base=/tmp
        OMP_NUM_THREADS="${threads}"
        OMP_DYNAMIC="${omp_dynamic}"
        OMP_PROC_BIND="${omp_proc_bind}"
        OMP_PLACES="${omp_places}"
    )
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

    mkdir -p "$(dirname "${log_file}")"
    {
        printf 'runtime_context: mpi_ranks=%s OMP_NUM_THREADS=%s OMP_DYNAMIC=%s OMP_PROC_BIND=%s OMP_PLACES=%s amrex.omp_threads=%s\n' \
            "${ranks}" "${threads}" "${omp_dynamic}" "${omp_proc_bind}" \
            "${omp_places}" "${threads}"
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

run_velocity_case() {
    local descriptor="$1"
    local executable="$2"
    local ny_intervals="$3"
    local nz="$4"
    local run_kind="$5"
    local ranks="${6:-${mpi_ranks}}"
    local threads="${7:-${omp_threads}}"
    local nx_intervals=$((4 * ny_intervals))
    local max_step=$((200 * ny_intervals * ny_intervals))
    local u_max_lb
    u_max_lb="$(physical_value "$(python3 -c \
        'import sys; print(float(sys.argv[1])*float(sys.argv[2])/int(sys.argv[3]))' \
        "${velocity_re_target}" "${velocity_nu_lb}" "${ny_intervals}")")"
    local check_int=100
    if (( ny_intervals >= 20 )); then
        check_int=200
    fi
    if (( ny_intervals >= 50 )); then
        check_int=500
    fi


    local run_dir="showcases/poiseuille_flow/outputs/guo_velocity/${descriptor}/${run_kind}/N${ny_intervals}/${timestamp}"
    local log_dir="${case_log_root}/guo_velocity/${descriptor}/${run_kind}/N${ny_intervals}/${timestamp}"
    local variant="${descriptor}_guo_velocity_poiseuille_N${ny_intervals}"
    run_parallel_logged "${log_dir}/console.log" "${ranks}" "${threads}" \
        "${executable}" "${velocity_input}" \
        poiseuille_flow.intervals="${nx_intervals}" "${ny_intervals}" \
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

    grep -F "poiseuille_flow check passed" "${log_dir}/console.log"
    grep -F "condition = velocity_inlet -> velocity_outlet" \
        "${log_dir}/console.log"
    grep -F "scheme = guo_extrapolation" "${log_dir}/console.log"
    grep -F "physical_intervals = (${nx_intervals}, ${ny_intervals})" \
        "${log_dir}/console.log"
    grep -F "profile_geometry = valid_node_center" "${log_dir}/console.log"
    grep -F "profile_valid_nodes = $((ny_intervals + 1))" "${log_dir}/console.log"
    grep -F "profile_lattice_intervals = ${ny_intervals}" \
        "${log_dir}/console.log"
    grep -F "poiseuille_steady_reached = true" "${log_dir}/console.log"
    grep -F "poiseuille_mass_gate = record_only" "${log_dir}/console.log"
    grep -F "physical_weighted_mass" "${run_dir}/poiseuille_profile_metrics.csv"
    grep -F "rho_streamwise_linear_r2" \
        "${run_dir}/poiseuille_profile_metrics.csv"
    test -f "${run_dir}/poiseuille_density_pressure.csv"
    grep -F "pressure_lb_cross_section_weighted_mean" \
        "${run_dir}/poiseuille_density_pressure.csv"
    test -d "${run_dir}/plt00000"
    local final_plot
    final_plot="$(find "${run_dir}" -maxdepth 1 -type d \
        -name 'plt[0-9][0-9][0-9][0-9][0-9]' | sort | tail -n 1)"
    test -n "${final_plot}"
    test "${final_plot}" != "${run_dir}/plt00000"
    python3 showcases/poiseuille_flow/scripts/plot_poiseuille_profiles.py "${run_dir}"
    test -f "${run_dir}/poiseuille_density_pressure.png"
    RUN_CASE_DIR="${run_dir}"
    printf '%s Guo velocity Poiseuille N=%s output: %s\n' \
        "${descriptor}" "${ny_intervals}" "${run_dir}"
}

run_formal_convergence() (
    local group="$1"
    local executable="$2"
    local input_file="$3"
    local nz="$4"
    local pressure_route="$5"
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
            "${pressure_re_target}" "${pressure_nu_lb}" "${ny}")")"
        local command=(
            "${executable}"
            "${input_file}"
            "poiseuille_flow.intervals=${nx}"
            "${ny}"
            "poiseuille_flow.depth_cells=${nz}"
            "poiseuille_flow.u_max_lb=${u_max_lb}"
            "poiseuille_flow.tau=${pressure_tau}"
        )
        if [[ "${pressure_route}" == "true" ]]; then
            local density_delta
            local rho_in
            local rho_out
            read -r density_delta rho_in rho_out < <(
                pressure_density_values "${ny}" "${u_max_lb}"
            )
            command+=(
                "poiseuille_flow.rho_in_lb=${rho_in}"
                "poiseuille_flow.rho_out_lb=${rho_out}"
                "poiseuille_flow.pressure_scaling=fixed_re_fixed_tau"
            )
        fi
        command+=(
            mesh.max_grid_size=32
            "run.variant=${entry}"
            run.kind=convergence
            "run.max_step=${max_step}"
            "diagnostics.interval=${check_int}"
            "diagnostics.steady_check_interval=${check_int}"
            diagnostics.steady_l2_tolerance=5.0e-8
            diagnostics.stop_on_steady=1
            poiseuille_validation.write_profile=1
            poiseuille_validation.assert=0
            poiseuille_validation.mass_assert=0
            output.plot_interval=0
            output.write_final_plotfile=0
            "output.directory=${run_dir}"
            "output.log_directory=${log_dir}"
        )
        mkdir "${run_dir}"
        run_parallel_logged \
            "${log_dir}/console.log" "${mpi_ranks}" "${omp_threads}" \
            "${command[@]}"
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


run_velocity_convergence() {
    local descriptor="$1"
    local executable="$2"
    local nz="$3"
    local group
    case "${descriptor}" in
        d2q9)
            group=poiseuille.guo-velocity.d2q9.slope
            ;;
        d3q27)
            group=poiseuille.guo-velocity.d3q27.slope
            ;;
        *)
            printf 'unsupported Guo velocity Descriptor: %s\n' \
                "${descriptor}" >&2
            return 2
            ;;
    esac
    local direct_executable="./${executable#./showcases/poiseuille_flow/}"
    run_formal_convergence \
        "${group}" "${direct_executable}" \
        inputs/inputs.d2q9_guo_velocity "${nz}" false
}

pressure_density_values() {
    local ny_intervals="$1"
    local u_max_lb="$2"
    python3 -c '
import sys
n = int(sys.argv[1])
rho0 = 1.0
u_max = float(sys.argv[2])
nu = float(sys.argv[3])
cs2 = 1.0 / 3.0
length = 4.0 * n
height = float(n)
delta = 8.0 * rho0 * nu * length * u_max / (cs2 * height * height)
print(f"{delta:.17g} {rho0 + 0.5 * delta:.17g} {rho0 - 0.5 * delta:.17g}")
' "${ny_intervals}" "${u_max_lb}" "${pressure_nu_lb}"
}

check_pressure_metrics() {
    local metrics_file="$1"
    local ny_intervals="$2"
    local density_delta="$3"
    local rho_in="$4"
    local rho_out="$5"
    local u_max_lb="$6"
    local scaling_code="$7"
    local re_effective="$8"
    local drive_kind="${9:-pressure}"
    python3 -c '
import csv
import math
import sys

(
    path,
    n_text,
    delta_text,
    rho_in_text,
    rho_out_text,
    u_max_text,
    scaling_code_text,
    re_text,
    drive_kind,
) = sys.argv[1:]
with open(path, newline="") as stream:
    rows = list(csv.DictReader(stream))
if len(rows) != 1:
    raise SystemExit(f"{path}: expected one metrics row, found {len(rows)}")
row = {key: float(value) for key, value in rows[0].items()}
for key, value in row.items():
    if not math.isfinite(value):
        raise SystemExit(f"{path}: {key}={value} is not finite")
n = int(n_text)
expected_delta = float(delta_text)
expected_rho_in = float(rho_in_text)
expected_rho_out = float(rho_out_text)
expected_u_max = float(u_max_text)
expected_scaling_code = int(scaling_code_text)
expected_re = float(re_text)
expected_mach = expected_u_max / math.sqrt(1.0 / 3.0)
if drive_kind not in ("pressure", "prescribed_state"):
    raise SystemExit(f"unsupported density diagnostic drive kind {drive_kind}")
expected_pressure_driven = 1.0 if drive_kind == "pressure" else 0.0
expected_prescribed_state = 1.0 if drive_kind == "prescribed_state" else 0.0
checks = {
    "pressure_driven": (row["pressure_driven"], expected_pressure_driven, 0.0),
    "prescribed_state": (row["prescribed_state"], expected_prescribed_state, 0.0),
    "n_intervals_y": (row["n_intervals_y"], float(n), 0.0),
    "rho_in_target_lb": (row["rho_in_target_lb"], expected_rho_in, 2.0e-14),
    "rho_out_target_lb": (row["rho_out_target_lb"], expected_rho_out, 2.0e-14),
    "density_delta_target_lb": (row["density_delta_target_lb"], expected_delta, 2.0e-14),
    "density_delta_expected_lb": (row["density_delta_expected_lb"], expected_delta, 2.0e-14),
    "density_delta_over_rho_ref": (row["density_delta_over_rho_ref"], expected_delta, 2.0e-14),
    "u_max_lb": (row["u_max_lb"], expected_u_max, 2.0e-14),
    "mach": (row["mach"], expected_mach, 2.0e-14),
    "pressure_scaling_code": (row["pressure_scaling_code"], float(expected_scaling_code), 0.0),
    "re_effective": (row["re_effective"], expected_re, 2.0e-12),
}
for label, (actual, expected, tolerance) in checks.items():
    if abs(actual - expected) > tolerance:
        raise SystemExit(
            f"{path}: {label}={actual:.17g}, expected {expected:.17g}"
        )
if row["max_abs_pressure_boundary_rho_error"] > 1.0e-10:
    raise SystemExit(
        f"{path}: pressure boundary rho error "
        f"{row['"'"'max_abs_pressure_boundary_rho_error'"'"']:.17g} exceeds 1e-10"
    )
if row["rho_streamwise_linear_r2"] < 0.99:
    raise SystemExit(
        f"{path}: density linear R2={row['"'"'rho_streamwise_linear_r2'"'"']:.17g} < 0.99"
    )
print(
    "density_metrics_pass "
    f"drive={drive_kind} "
    f"N={n} Umax={expected_u_max:.12g} "
    f"Re={row['"'"'re_effective'"'"']:.12g} delta_rho={expected_delta:.12g} "
    f"Mach={row['"'"'mach'"'"']:.12g} "
    f"boundary_rho_error={row['"'"'max_abs_pressure_boundary_rho_error'"'"']:.12g} "
    f"rho_R2={row['"'"'rho_streamwise_linear_r2'"'"']:.12g} "
    f"ux_spread/Umax={row['"'"'max_streamwise_ux_spread_over_umax'"'"']:.12g}"
)
' "${metrics_file}" "${ny_intervals}" "${density_delta}" "${rho_in}" "${rho_out}" \
    "${u_max_lb}" "${scaling_code}" "${re_effective}" "${drive_kind}"
}

run_pressure_case() {
    local descriptor="$1"
    local executable="$2"
    local ny_intervals="$3"
    local nz="$4"
    local run_kind="$5"
    local u_max_lb="${6:-0.1}"
    local scaling_label="${7:-fixed_re_fixed_tau}"
    local ranks="${8:-${mpi_ranks}}"
    local threads="${9:-${omp_threads}}"
    local scaling_code
    case "${scaling_label}" in
        fixed_mach_characterization)
            scaling_code=1
            ;;
        fixed_re_fixed_tau)
            scaling_code=2
            ;;
        *)
            printf 'unsupported pressure scaling label: %s\n' \
                "${scaling_label}" >&2
            exit 2
            ;;
    esac
    local nx_intervals=$((4 * ny_intervals))
    local max_step=$((200 * ny_intervals * ny_intervals))
    local check_int=100
    if (( ny_intervals >= 20 )); then
        check_int=200
    fi
    if (( ny_intervals >= 50 )); then
        check_int=500
    fi


    local density_delta
    local rho_in
    local rho_out
    local re_effective
    read -r density_delta rho_in rho_out < <(
        pressure_density_values "${ny_intervals}" "${u_max_lb}"
    )
    if [[ "${scaling_label}" == "fixed_re_fixed_tau" ]]; then
        re_effective="${pressure_re_target}"
    else
        re_effective="$(physical_value "$(python3 -c \
            'import sys; print(float(sys.argv[1])*int(sys.argv[2])/float(sys.argv[3]))' \
            "${u_max_lb}" "${ny_intervals}" "${pressure_nu_lb}")")"
    fi

    local run_dir="showcases/poiseuille_flow/outputs/guo_pressure/${descriptor}/${run_kind}/N${ny_intervals}/${timestamp}"
    local log_dir="${case_log_root}/guo_pressure/${descriptor}/${run_kind}/N${ny_intervals}/${timestamp}"
    local variant="${descriptor}_guo_pressure_${scaling_label}_N${ny_intervals}"
    run_parallel_logged "${log_dir}/console.log" "${ranks}" "${threads}" \
        "${executable}" "${pressure_input}" \
        poiseuille_flow.intervals="${nx_intervals}" "${ny_intervals}" \
        poiseuille_flow.depth_cells="${nz}" \
        poiseuille_flow.u_max_lb="${u_max_lb}" \
        poiseuille_flow.tau="${pressure_tau}" \
        poiseuille_flow.rho_in_lb="${rho_in}" \
        poiseuille_flow.rho_out_lb="${rho_out}" \
        poiseuille_flow.pressure_scaling="${scaling_label}" \
        mesh.max_grid_size=32 \
        run.variant="${variant}" \
        run.kind="${run_kind}" \
        run.max_step="${max_step}" \
        output.plot_interval="${max_step}" \
        diagnostics.interval="${check_int}" \
        diagnostics.steady_check_interval="${check_int}" \
        output.directory="${run_dir}" \
        output.log_directory="${log_dir}"

    grep -F "poiseuille_flow check passed" "${log_dir}/console.log"
    grep -F "physical_intervals = (${nx_intervals}, ${ny_intervals})" \
        "${log_dir}/console.log"
    grep -F "pressure-driven Poiseuille: pressure_driven = true" \
        "${log_dir}/console.log"
    grep -F "convergence_scaling = ${scaling_label}" \
        "${log_dir}/console.log"
    grep -F "profile = constant_density" "${log_dir}/console.log"
    grep -F "poiseuille_steady_reached = true" "${log_dir}/console.log"
    grep -F "poiseuille_mass_gate = record_only" "${log_dir}/console.log"
    test -f "${run_dir}/poiseuille_density_pressure.csv"
    check_pressure_metrics \
        "${run_dir}/poiseuille_profile_metrics.csv" \
        "${ny_intervals}" "${density_delta}" "${rho_in}" "${rho_out}" \
        "${u_max_lb}" "${scaling_code}" "${re_effective}"
    test -d "${run_dir}/plt00000"
    local final_plot
    final_plot="$(find "${run_dir}" -maxdepth 1 -type d \
        -name 'plt[0-9][0-9][0-9][0-9][0-9]' | sort | tail -n 1)"
    test -n "${final_plot}"
    test "${final_plot}" != "${run_dir}/plt00000"
    python3 showcases/poiseuille_flow/scripts/plot_poiseuille_profiles.py "${run_dir}"
    test -f "${run_dir}/poiseuille_profile.png"
    test -f "${run_dir}/poiseuille_density_pressure.png"
    RUN_CASE_DIR="${run_dir}"
    printf '%s Guo pressure Poiseuille N=%s Umax=%s Re=%s scaling=%s output: %s\n' \
        "${descriptor}" "${ny_intervals}" "${u_max_lb}" "${re_effective}" \
        "${scaling_label}" "${run_dir}"
}

run_prescribed_state_case() {
    local descriptor="$1"
    local executable="$2"
    local nz="$3"
    local ny_intervals=10
    local nx_intervals=$((4 * ny_intervals))
    local u_max_lb
    u_max_lb="$(physical_value "$(python3 -c \
        'import sys; print(float(sys.argv[1])*float(sys.argv[2])/int(sys.argv[3]))' \
        "${pressure_re_target}" "${pressure_nu_lb}" "${ny_intervals}")")"
    local density_delta
    local rho_in
    local rho_out
    read -r density_delta rho_in rho_out < <(
        pressure_density_values "${ny_intervals}" "${u_max_lb}"
    )
    local run_dir="showcases/poiseuille_flow/outputs/guo_prescribed_state/${descriptor}/experiment/N10/${timestamp}"
    local log_dir="${case_log_root}/guo_prescribed_state/${descriptor}/experiment/N10/${timestamp}"
    local variant="${descriptor}_guo_prescribed_state_poiseuille_N10"

    run_logged "${log_dir}/console.log" \
        "${executable}" "${prescribed_state_input}" \
        poiseuille_flow.intervals="${nx_intervals}" "${ny_intervals}" \
        poiseuille_flow.depth_cells="${nz}" \
        poiseuille_flow.u_max_lb="${u_max_lb}" \
        poiseuille_flow.tau="${pressure_tau}" \
        poiseuille_flow.rho_in_lb="${rho_in}" \
        poiseuille_flow.rho_out_lb="${rho_out}" \
        mesh.max_grid_size=32 \
        run.variant="${variant}" \
        output.directory="${run_dir}" \
        output.log_directory="${log_dir}"

    grep -F "prescribed-state Poiseuille: prescribed_state = true" \
        "${log_dir}/console.log"
    grep -F "condition = prescribed_state -> prescribed_state" \
        "${log_dir}/console.log"
    grep -F "profile_assert = 0" "${log_dir}/console.log"
    grep -F "poiseuille_mass_gate = record_only" "${log_dir}/console.log"
    check_pressure_metrics \
        "${run_dir}/poiseuille_profile_metrics.csv" \
        "${ny_intervals}" "${density_delta}" "${rho_in}" "${rho_out}" \
        "${u_max_lb}" 2 "${pressure_re_target}" prescribed_state
    test -f "${run_dir}/poiseuille_profile.csv"
    test -f "${run_dir}/poiseuille_density_pressure.csv"
    test -d "${run_dir}/plt00000"
    local final_plot
    final_plot="$(find "${run_dir}" -maxdepth 1 -type d \
        -name 'plt[0-9][0-9][0-9][0-9][0-9]' | sort | tail -n 1)"
    test -n "${final_plot}"
    test "${final_plot}" != "${run_dir}/plt00000"
    python3 showcases/poiseuille_flow/scripts/plot_poiseuille_profiles.py "${run_dir}"
    test -f "${run_dir}/poiseuille_profile.png"
    test -f "${run_dir}/poiseuille_density_pressure.png"
    RUN_CASE_DIR="${run_dir}"
    printf '%s Guo prescribed-state Poiseuille experiment output: %s\n' \
        "${descriptor}" "${run_dir}"
}

run_pressure_convergence() {
    local descriptor="$1"
    local executable="$2"
    local nz="$3"
    local scaling_label="${4:-fixed_re_fixed_tau}"
    if [[ "${scaling_label}" == "fixed_re_fixed_tau" ]]; then
        local group
        case "${descriptor}" in
            d2q9)
                group=poiseuille.guo-pressure.d2q9.slope
                ;;
            d3q27)
                group=poiseuille.guo-pressure.d3q27.slope
                ;;
            *)
                printf 'unsupported Guo pressure Descriptor: %s\n' \
                    "${descriptor}" >&2
                return 2
                ;;
        esac
        local direct_executable="./${executable#./showcases/poiseuille_flow/}"
        run_formal_convergence \
            "${group}" "${direct_executable}" \
            inputs/inputs.d2q9_guo_pressure "${nz}" true
        return
    fi
    if [[ "${scaling_label}" != "fixed_mach_characterization" ]]; then
        printf 'unsupported pressure scaling label: %s\n' \
            "${scaling_label}" >&2
        return 2
    fi

    local root="showcases/poiseuille_flow/outputs/guo_pressure/${descriptor}/characterization/${timestamp}"
    local runs=()
    local ny
    for ny in 10 20 50; do
        run_pressure_case \
            "${descriptor}" "${executable}" "${ny}" "${nz}" \
            characterization 0.1 fixed_mach_characterization
        runs+=("${RUN_CASE_DIR}")
    done
    python3 showcases/poiseuille_flow/scripts/plot_poiseuille_profiles.py "${runs[@]}" \
        --convergence-output "${root}" \
        --scaling-label fixed_mach_characterization
    grep -F "visualization_only=true" \
        "${root}/poiseuille_convergence_plot_metadata.txt"
    grep -F "pressure_scaling=fixed_mach_characterization" \
        "${root}/poiseuille_convergence_plot_metadata.txt"
    printf '%s Guo pressure Poiseuille fixed-Mach characterization output: %s\n' \
        "${descriptor}" "${root}"
}

run_velocity_parallel_check() {
    local descriptor="$1"
    local executable="$2"
    local nz="$3"
    local runs=()
    run_velocity_case \
        "${descriptor}" "${executable}" 10 "${nz}" parallel-singleton 1 1
    runs+=("${RUN_CASE_DIR}")
    run_velocity_case \
        "${descriptor}" "${executable}" 10 "${nz}" parallel-mpi2-omp1 2 1
    runs+=("${RUN_CASE_DIR}")
    run_velocity_case \
        "${descriptor}" "${executable}" 10 "${nz}" parallel-mpi2-omp2 2 2
    runs+=("${RUN_CASE_DIR}")
    python3 showcases/poiseuille_flow/scripts/check_poiseuille_parallel.py "${runs[@]}"
}

run_pressure_parallel_check() {
    local descriptor="$1"
    local executable="$2"
    local nz="$3"
    local runs=()
    run_pressure_case \
        "${descriptor}" "${executable}" 10 "${nz}" \
        parallel-singleton 0.1 fixed_re_fixed_tau 1 1
    runs+=("${RUN_CASE_DIR}")
    run_pressure_case \
        "${descriptor}" "${executable}" 10 "${nz}" \
        parallel-mpi2-omp1 0.1 fixed_re_fixed_tau 2 1
    runs+=("${RUN_CASE_DIR}")
    run_pressure_case \
        "${descriptor}" "${executable}" 10 "${nz}" \
        parallel-mpi2-omp2 0.1 fixed_re_fixed_tau 2 2
    runs+=("${RUN_CASE_DIR}")
    python3 showcases/poiseuille_flow/scripts/check_poiseuille_parallel.py "${runs[@]}"
}

case "${mode}" in
    build)
        build_targets
        ;;
    velocity-d2-steady)
        build_targets
        run_velocity_case d2q9 "${d2_exe}" 10 1 steady
        ;;
    velocity-d2-convergence)
        build_targets
        run_velocity_convergence d2q9 "${d2_exe}" 1
        ;;
    velocity-d3-steady)
        build_targets
        run_velocity_case d3q27 "${d3_exe}" 10 4 steady
        ;;
    velocity-d3-convergence)
        build_targets
        run_velocity_convergence d3q27 "${d3_exe}" 4
        ;;
    velocity-d2-parallel)
        build_targets
        run_velocity_parallel_check d2q9 "${d2_exe}" 1
        ;;
    velocity-d3-parallel)
        build_targets
        run_velocity_parallel_check d3q27 "${d3_exe}" 4
        ;;
    velocity-parallel)
        build_targets
        run_velocity_parallel_check d2q9 "${d2_exe}" 1
        run_velocity_parallel_check d3q27 "${d3_exe}" 4
        ;;
    pressure-d2-steady)
        build_targets
        run_pressure_case d2q9 "${d2_exe}" 10 1 steady
        ;;
    pressure-d2-convergence)
        build_targets
        run_pressure_convergence d2q9 "${d2_exe}" 1
        ;;
    pressure-d3-steady)
        build_targets
        run_pressure_case d3q27 "${d3_exe}" 10 4 steady
        ;;
    pressure-d3-convergence)
        build_targets
        run_pressure_convergence d3q27 "${d3_exe}" 4
        ;;
    pressure-d2-characterization)
        build_targets
        run_pressure_convergence \
            d2q9 "${d2_exe}" 1 fixed_mach_characterization
        ;;
    pressure-d3-characterization)
        build_targets
        run_pressure_convergence \
            d3q27 "${d3_exe}" 4 fixed_mach_characterization
        ;;
    pressure-d2-parallel)
        build_targets
        run_pressure_parallel_check d2q9 "${d2_exe}" 1
        ;;
    pressure-d3-parallel)
        build_targets
        run_pressure_parallel_check d3q27 "${d3_exe}" 4
        ;;
    pressure-parallel)
        build_targets
        run_pressure_parallel_check d2q9 "${d2_exe}" 1
        run_pressure_parallel_check d3q27 "${d3_exe}" 4
        ;;
    pressure-characterization)
        build_targets
        run_pressure_convergence \
            d2q9 "${d2_exe}" 1 fixed_mach_characterization
        run_pressure_convergence \
            d3q27 "${d3_exe}" 4 fixed_mach_characterization
        ;;
    pressure-all)
        build_targets
        run_pressure_convergence d2q9 "${d2_exe}" 1
        run_pressure_convergence d3q27 "${d3_exe}" 4
        ;;
    velocity-all)
        build_targets
        run_velocity_convergence d2q9 "${d2_exe}" 1
        run_velocity_convergence d3q27 "${d3_exe}" 4
        ;;
    prescribed-state-d2-experiment)
        build_targets
        run_prescribed_state_case d2q9 "${d2_exe}" 1
        ;;
    prescribed-state-d3-experiment)
        build_targets
        run_prescribed_state_case d3q27 "${d3_exe}" 4
        ;;
    prescribed-state-experiment)
        build_targets
        run_prescribed_state_case d2q9 "${d2_exe}" 1
        run_prescribed_state_case d3q27 "${d3_exe}" 4
        ;;
    *)
        printf 'Usage: %s [build|velocity-d2-steady|velocity-d2-convergence|velocity-d3-steady|velocity-d3-convergence|velocity-d2-parallel|velocity-d3-parallel|velocity-parallel|velocity-all|prescribed-state-d2-experiment|prescribed-state-d3-experiment|prescribed-state-experiment|pressure-d2-steady|pressure-d2-convergence|pressure-d3-steady|pressure-d3-convergence|pressure-d2-characterization|pressure-d3-characterization|pressure-characterization|pressure-d2-parallel|pressure-d3-parallel|pressure-parallel|pressure-all]\n' "$0" >&2
        exit 2
        ;;
esac
