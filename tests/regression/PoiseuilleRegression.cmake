# Direct Poiseuille replacement matrix. Test-owned fixtures never import case
# workflow inputs, scripts or reference data.
add_direct_case_run(
    regression_poiseuille_hbb_d2q9
    lbm_regression_poiseuille_d2q9
    hbb_velocity_d2q9_short.inputs
    poiseuille_direct)
set_property(
    TEST regression_poiseuille_hbb_d2q9
    APPEND PROPERTY FIXTURES_SETUP poiseuille_hbb_serial)
add_direct_case_run(
    regression_poiseuille_guo_velocity_d2q9
    lbm_regression_poiseuille_d2q9
    guo_velocity_d2q9_short.inputs
    poiseuille_direct)
add_direct_case_run(
    regression_poiseuille_guo_pressure_d2q9
    lbm_regression_poiseuille_d2q9
    guo_pressure_d2q9_short.inputs
    poiseuille_direct)
add_direct_case_run(
    regression_poiseuille_guo_prescribed_d2q9
    lbm_regression_poiseuille_d2q9
    guo_prescribed_d2q9_short.inputs
    poiseuille_direct)
add_direct_case_run(
    regression_poiseuille_representative_d3q27
    lbm_regression_poiseuille_d3q27
    guo_pressure_d3q27_short.inputs
    poiseuille_direct)

regression_artifact_dir(
    poiseuille_hbb_artifacts regression_poiseuille_hbb_d2q9)
regression_artifact_dir(
    poiseuille_guo_velocity_artifacts
    regression_poiseuille_guo_velocity_d2q9)
regression_artifact_dir(
    poiseuille_guo_pressure_artifacts
    regression_poiseuille_guo_pressure_d2q9)
regression_artifact_dir(
    poiseuille_guo_prescribed_artifacts
    regression_poiseuille_guo_prescribed_d2q9)
regression_artifact_dir(
    poiseuille_d3_artifacts
    regression_poiseuille_representative_d3q27)
add_test(
    NAME regression_poiseuille_schema
    COMMAND ${Python3_EXECUTABLE}
        ${REGRESSION_CHECKER_DIR}/check_poiseuille_short_matrix.py
        --hbb-d2q9 ${poiseuille_hbb_artifacts}
        --guo-velocity-d2q9 ${poiseuille_guo_velocity_artifacts}
        --guo-pressure-d2q9 ${poiseuille_guo_pressure_artifacts}
        --guo-prescribed-d2q9 ${poiseuille_guo_prescribed_artifacts}
        --representative-d3q27 ${poiseuille_d3_artifacts})
configure_lbm_test(regression_poiseuille_schema regression serial 1 1)
set_tests_properties(
    regression_poiseuille_schema PROPERTIES
    FIXTURES_REQUIRED poiseuille_direct)
set_property(TEST regression_poiseuille_schema APPEND PROPERTY LABELS short)

set(poiseuille_parallel_artifacts ${poiseuille_hbb_artifacts})
if(AMReX_OMP AND LBM_TEST_OPENMP)
    add_openmp_direct_case_run(
        regression_poiseuille_hbb_openmp
        lbm_regression_poiseuille_d2q9
        hbb_velocity_d2q9_short.inputs
        poiseuille_parallel
        2)
    regression_artifact_dir(
        poiseuille_openmp_artifacts regression_poiseuille_hbb_openmp)
    list(APPEND poiseuille_parallel_artifacts ${poiseuille_openmp_artifacts})
endif()
if(AMReX_MPI)
    add_mpi_direct_case_run(
        regression_poiseuille_hbb_mpi
        lbm_regression_poiseuille_d2q9
        hbb_velocity_d2q9_short.inputs
        poiseuille_parallel
        2)
    regression_artifact_dir(
        poiseuille_mpi_artifacts regression_poiseuille_hbb_mpi)
    list(APPEND poiseuille_parallel_artifacts ${poiseuille_mpi_artifacts})
endif()
list(LENGTH poiseuille_parallel_artifacts poiseuille_parallel_count)
if(poiseuille_parallel_count GREATER 1)
    add_test(
        NAME regression_poiseuille_parallel_parity
        COMMAND ${Python3_EXECUTABLE}
            ${REGRESSION_CHECKER_DIR}/compare_public_artifacts.py
            --numeric
                diagnostics.csv
                poiseuille_profile.csv
                poiseuille_profile_metrics.csv
                poiseuille_density_pressure.csv
            --
            ${poiseuille_parallel_artifacts})
    configure_lbm_test(
        regression_poiseuille_parallel_parity regression serial 1 1)
    set_tests_properties(
        regression_poiseuille_parallel_parity PROPERTIES
        FIXTURES_REQUIRED "poiseuille_hbb_serial;poiseuille_parallel")
    set_property(
        TEST regression_poiseuille_parallel_parity
        APPEND PROPERTY LABELS short parity)
endif()

lbm_test_command(failure_command lbm_regression_poiseuille_d2q9 1
        ${CMAKE_CURRENT_SOURCE_DIR}/direct/inputs/hbb_velocity_d2q9_short.inputs
        poiseuille_flow.variant=unsupported)
add_test(
    NAME regression_poiseuille_invalid_drive
    COMMAND ${Python3_EXECUTABLE}
        ${REGRESSION_CHECKER_DIR}/expect_process_failure.py
        --contains "poiseuille_flow.variant must be" --
        ${failure_command})
configure_lbm_test(
    regression_poiseuille_invalid_drive regression serial 1 1)
set_property(
    TEST regression_poiseuille_invalid_drive
    APPEND PROPERTY LABELS short negative)

lbm_test_command(failure_command lbm_regression_poiseuille_d2q9 1
        ${CMAKE_CURRENT_SOURCE_DIR}/direct/inputs/guo_pressure_d2q9_short.inputs
        poiseuille_flow.pressure_scaling=unsupported)
add_test(
    NAME regression_poiseuille_invalid_scaling
    COMMAND ${Python3_EXECUTABLE}
        ${REGRESSION_CHECKER_DIR}/expect_process_failure.py
        --contains "pressure_scaling must be" --
        ${failure_command})
configure_lbm_test(
    regression_poiseuille_invalid_scaling regression serial 1 1)
set_property(
    TEST regression_poiseuille_invalid_scaling
    APPEND PROPERTY LABELS short negative)

lbm_test_command(failure_command lbm_regression_poiseuille_d2q9 1
        ${CMAKE_CURRENT_SOURCE_DIR}/direct/inputs/hbb_velocity_d2q9_short.inputs
        poiseuille_flow.intervals=0 4)
add_test(
    NAME regression_poiseuille_invalid_grid
    COMMAND ${Python3_EXECUTABLE}
        ${REGRESSION_CHECKER_DIR}/expect_process_failure.py
        --contains "intervals, depth_cells and max_grid_size must be positive" --
        ${failure_command})
configure_lbm_test(
    regression_poiseuille_invalid_grid regression serial 1 1)
set_property(
    TEST regression_poiseuille_invalid_grid
    APPEND PROPERTY LABELS short negative)

lbm_test_command(failure_command lbm_regression_poiseuille_d2q9 1
        ${CMAKE_CURRENT_SOURCE_DIR}/direct/inputs/hbb_velocity_d2q9_short.inputs
        output.directory=${CMAKE_CURRENT_SOURCE_DIR}/fixtures/not_a_directory/child)
add_test(
    NAME regression_poiseuille_output_failure
    COMMAND ${Python3_EXECUTABLE}
        ${REGRESSION_CHECKER_DIR}/expect_process_failure.py
        --contains "Failed to prepare direct Poiseuille output" --
        ${failure_command})
configure_lbm_test(
    regression_poiseuille_output_failure regression serial 1 1)
set_property(
    TEST regression_poiseuille_output_failure
    APPEND PROPERTY LABELS short negative)
add_test(
    NAME regression_poiseuille_checker_failures
    COMMAND ${Python3_EXECUTABLE} -B
        ${REGRESSION_CHECKER_DIR}/check_poiseuille_short_matrix_synthetic.py)
configure_lbm_test(
    regression_poiseuille_checker_failures regression serial 1 1)
set_property(
    TEST regression_poiseuille_checker_failures
    APPEND PROPERTY LABELS short synthetic)
