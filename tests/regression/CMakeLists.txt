set(REGRESSION_CHECKER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/checkers)

function(add_regression_descriptor_targets target_stem source_file)
    add_lbm_test_target(${target_stem}_d2q9 ${source_file})
    target_include_directories(
        ${target_stem}_d2q9 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    add_lbm_test_target(${target_stem}_d3q27 ${source_file})
    target_include_directories(
        ${target_stem}_d3q27 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_definitions(
        ${target_stem}_d3q27 PRIVATE LBM_DESCRIPTOR_D3Q27=1)
endfunction()

add_regression_descriptor_targets(
    lbm_regression_taylor_green
    ${CMAKE_CURRENT_SOURCE_DIR}/src/taylor_green/main.cpp)
add_regression_descriptor_targets(
    lbm_regression_uniform
    ${CMAKE_CURRENT_SOURCE_DIR}/src/uniform/main.cpp)
add_regression_descriptor_targets(
    lbm_regression_poiseuille
    ${CMAKE_CURRENT_SOURCE_DIR}/src/poiseuille/main.cpp)
add_regression_descriptor_targets(
    lbm_regression_cavity
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cavity/main.cpp)
add_regression_descriptor_targets(
    lbm_regression_channel
    ${CMAKE_CURRENT_SOURCE_DIR}/src/channel/main.cpp)
function(add_checked_regression_process test_name parallel_kind processors omp_threads)
    add_test(
        NAME ${test_name}
        COMMAND ${Python3_EXECUTABLE}
            ${REGRESSION_CHECKER_DIR}/run_regression_process.py ${ARGN})
    configure_lbm_test(
        ${test_name} regression ${parallel_kind} ${processors} ${omp_threads})
endfunction()

function(regression_artifact_dir output test_name)
    set(${output} "${CMAKE_CURRENT_BINARY_DIR}/work/${test_name}/artifacts" PARENT_SCOPE)
endfunction()

function(regression_log_dir output test_name)
    set(${output} "${CMAKE_CURRENT_BINARY_DIR}/work/${test_name}/logs" PARENT_SCOPE)
endfunction()

function(configure_regression_run test_name fixture)
    set_tests_properties(${test_name} PROPERTIES FIXTURES_SETUP ${fixture})
    set_property(TEST ${test_name} APPEND PROPERTY LABELS short)
endfunction()

function(regression_direct_args output test_name input_name)
    regression_artifact_dir(artifact_dir ${test_name})
    regression_log_dir(log_dir ${test_name})
    set(${output}
        ${CMAKE_CURRENT_SOURCE_DIR}/direct/inputs/${input_name}
        output.directory=${artifact_dir}
        output.log_directory=${log_dir}
        output.plot_interval=0
        output.write_final_plotfile=0
        PARENT_SCOPE)
endfunction()

function(add_direct_case_run test_name target input_name fixture)
    regression_direct_args(arguments ${test_name} ${input_name})
    regression_artifact_dir(artifact_dir ${test_name})
    regression_log_dir(log_dir ${test_name})
    lbm_test_command(command ${target} 1 ${arguments})
    add_checked_regression_process(
        ${test_name} serial 1 1
        --clean ${artifact_dir} --clean ${log_dir} --
        ${command})
    configure_regression_run(${test_name} ${fixture})
endfunction()

function(add_openmp_direct_case_run
         test_name target input_name fixture threads)
    regression_direct_args(arguments ${test_name} ${input_name})
    regression_artifact_dir(artifact_dir ${test_name})
    regression_log_dir(log_dir ${test_name})
    add_checked_regression_process(
        ${test_name} openmp ${threads} ${threads}
        --clean ${artifact_dir} --clean ${log_dir} --
        $<TARGET_FILE:${target}> ${arguments})
    configure_regression_run(${test_name} ${fixture})
endfunction()

function(add_mpi_direct_case_run
         test_name target input_name fixture ranks)
    regression_direct_args(arguments ${test_name} ${input_name})
    regression_artifact_dir(artifact_dir ${test_name})
    regression_log_dir(log_dir ${test_name})
    lbm_test_command(command ${target} ${ranks} ${arguments})
    add_checked_regression_process(
        ${test_name} mpi ${ranks} 1
        --clean ${artifact_dir} --clean ${log_dir} --
        ${command})
    configure_regression_run(${test_name} ${fixture})
endfunction()

function(add_hybrid_direct_case_run
         test_name target input_name fixture ranks threads)
    regression_direct_args(arguments ${test_name} ${input_name})
    regression_artifact_dir(artifact_dir ${test_name})
    regression_log_dir(log_dir ${test_name})
    math(EXPR processor_count "${ranks} * ${threads}")
    add_checked_regression_process(
        ${test_name} hybrid ${processor_count} ${threads}
        --clean ${artifact_dir} --clean ${log_dir} --
        ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
        --map-by ppr:${ranks}:socket:PE=${threads}
        --bind-to core --rank-by slot --report-bindings
        ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${target}>
        ${MPIEXEC_POSTFLAGS} ${arguments})
    configure_regression_run(${test_name} ${fixture})
endfunction()

add_direct_case_run(
    regression_tgv_d2q9_direct
    lbm_regression_taylor_green_d2q9
    tgv_d2q9_short.inputs
    tgv_direct)
add_direct_case_run(
    regression_tgv_d2q9_quasi2d_setup
    lbm_regression_taylor_green_d2q9
    tgv_d2q9_quasi2d_short.inputs
    tgv_quasi2d)
add_direct_case_run(
    regression_tgv_d3q27_direct
    lbm_regression_taylor_green_d3q27
    tgv_d3q27_short.inputs
    tgv_direct)
regression_artifact_dir(tgv_d2_artifacts regression_tgv_d2q9_direct)
regression_artifact_dir(tgv_d3_artifacts regression_tgv_d3q27_direct)
add_test(
    NAME regression_tgv_schema
    COMMAND ${Python3_EXECUTABLE}
        ${REGRESSION_CHECKER_DIR}/check_tgv_short_matrix.py
        --d2 ${tgv_d2_artifacts}
        --d3 ${tgv_d3_artifacts})
configure_lbm_test(regression_tgv_schema regression serial 1 1)
set_tests_properties(
    regression_tgv_schema PROPERTIES FIXTURES_REQUIRED tgv_direct)
set_property(TEST regression_tgv_schema APPEND PROPERTY LABELS short)

add_direct_case_run(
    regression_uniform_serial
    lbm_regression_uniform_d2q9
    uniform_hbb_short.inputs
    uniform_parallel)
set_property(
    TEST regression_uniform_serial APPEND PROPERTY FIXTURES_SETUP uniform_serial)

set(uniform_parallel_artifacts)
regression_artifact_dir(uniform_serial_artifacts regression_uniform_serial)
list(
    APPEND uniform_parallel_artifacts
    ${uniform_serial_artifacts})
if(AMReX_OMP AND LBM_TEST_OPENMP)
    add_openmp_direct_case_run(
        regression_uniform_openmp
        lbm_regression_uniform_d2q9
        uniform_hbb_short.inputs
        uniform_parallel
        2)
    regression_artifact_dir(uniform_openmp_artifacts regression_uniform_openmp)
    list(APPEND uniform_parallel_artifacts ${uniform_openmp_artifacts})
endif()
if(AMReX_MPI)
    add_mpi_direct_case_run(
        regression_uniform_mpi
        lbm_regression_uniform_d2q9
        uniform_hbb_short.inputs
        uniform_parallel
        2)
    regression_artifact_dir(uniform_mpi_artifacts regression_uniform_mpi)
    list(APPEND uniform_parallel_artifacts ${uniform_mpi_artifacts})
endif()
if(AMReX_MPI AND AMReX_OMP AND LBM_TEST_OPENMP)
    add_hybrid_direct_case_run(
        regression_uniform_hybrid
        lbm_regression_uniform_d2q9
        uniform_hbb_short.inputs
        uniform_parallel
        2
        2)
    regression_artifact_dir(uniform_hybrid_artifacts regression_uniform_hybrid)
    list(APPEND uniform_parallel_artifacts ${uniform_hybrid_artifacts})
endif()

add_direct_case_run(
    regression_uniform_guo_velocity_d2q9
    lbm_regression_uniform_d2q9
    uniform_guo_io_short.inputs
    uniform_variants)
add_direct_case_run(
    regression_uniform_guo_free_d2q9
    lbm_regression_uniform_d2q9
    uniform_guo_free_short.inputs
    uniform_variants)
add_direct_case_run(
    regression_uniform_guo_velocity_d3q27
    lbm_regression_uniform_d3q27
    uniform_guo_d3q27_short.inputs
    uniform_variants)
add_direct_case_run(
    regression_uniform_guo_free_d3q27
    lbm_regression_uniform_d3q27
    uniform_guo_free_d3q27_short.inputs
    uniform_variants)

regression_artifact_dir(
    uniform_guo_velocity_d2_artifacts regression_uniform_guo_velocity_d2q9)
regression_artifact_dir(
    uniform_guo_free_d2_artifacts regression_uniform_guo_free_d2q9)
regression_artifact_dir(
    uniform_guo_velocity_d3_artifacts regression_uniform_guo_velocity_d3q27)
regression_artifact_dir(
    uniform_guo_free_d3_artifacts regression_uniform_guo_free_d3q27)
add_test(
    NAME regression_uniform_schema
    COMMAND ${Python3_EXECUTABLE}
        ${REGRESSION_CHECKER_DIR}/check_uniform_short_matrix.py
        --hbb ${uniform_serial_artifacts}
        --guo-velocity-d2q9 ${uniform_guo_velocity_d2_artifacts}
        --guo-free-d2q9 ${uniform_guo_free_d2_artifacts}
        --guo-velocity-d3q27 ${uniform_guo_velocity_d3_artifacts}
        --guo-free-d3q27 ${uniform_guo_free_d3_artifacts})
configure_lbm_test(regression_uniform_schema regression serial 1 1)
set_tests_properties(
    regression_uniform_schema PROPERTIES
    FIXTURES_REQUIRED "uniform_serial;uniform_variants")
set_property(TEST regression_uniform_schema APPEND PROPERTY LABELS short)

add_test(
    NAME regression_uniform_checker_failures
    COMMAND ${Python3_EXECUTABLE}
        ${REGRESSION_CHECKER_DIR}/check_uniform_short_matrix_synthetic.py)
configure_lbm_test(
    regression_uniform_checker_failures regression serial 1 1)
set_property(
    TEST regression_uniform_checker_failures APPEND PROPERTY LABELS short synthetic)

lbm_test_command(failure_command lbm_regression_uniform_d3q27 1
        ${CMAKE_CURRENT_SOURCE_DIR}/direct/inputs/uniform_guo_io_short.inputs
        uniform_flow.depth_cells=1)
add_test(
    NAME regression_uniform_invalid_d3_depth
    COMMAND ${Python3_EXECUTABLE}
        ${REGRESSION_CHECKER_DIR}/expect_process_failure.py
        --contains
        "[case_validation:invalid_uniform_flow_depth:mesh.n_cell.z]" --
        ${failure_command})
configure_lbm_test(
    regression_uniform_invalid_d3_depth regression serial 1 1)
set_property(
    TEST regression_uniform_invalid_d3_depth
    APPEND PROPERTY LABELS short negative)

lbm_test_command(failure_command lbm_regression_uniform_d2q9 1
        ${CMAKE_CURRENT_SOURCE_DIR}/direct/inputs/uniform_hbb_short.inputs
        output.directory=${CMAKE_CURRENT_SOURCE_DIR}/fixtures/not_a_directory/child)
add_test(
    NAME regression_uniform_output_failure
    COMMAND ${Python3_EXECUTABLE}
        ${REGRESSION_CHECKER_DIR}/expect_process_failure.py
        --contains "Failed to prepare direct uniform-flow output" --
        ${failure_command})
configure_lbm_test(
    regression_uniform_output_failure regression serial 1 1)
set_property(
    TEST regression_uniform_output_failure
    APPEND PROPERTY LABELS short negative)

add_direct_case_run(
    regression_channel_single_yms_d2q9
    lbm_regression_channel_d2q9
    channel_single_yms_d2_short.inputs
    channel_single)
add_direct_case_run(
    regression_channel_single_bouzidi_d2q9
    lbm_regression_channel_d2q9
    channel_single_bouzidi_d2_short.inputs
    channel_single)
add_direct_case_run(
    regression_channel_single_yms_d3q27
    lbm_regression_channel_d3q27
    channel_single_yms_d3_short.inputs
    channel_single)

set(channel_single_parallel_artifacts)
if(AMReX_OMP AND LBM_TEST_OPENMP)
    add_openmp_direct_case_run(
        regression_channel_single_yms_openmp
        lbm_regression_channel_d2q9
        channel_single_yms_d2_short.inputs
        channel_single
        2)
    regression_artifact_dir(
        channel_single_openmp_artifacts
        regression_channel_single_yms_openmp)
    list(
        APPEND channel_single_parallel_artifacts
        ${channel_single_openmp_artifacts})
endif()
if(AMReX_MPI)
    add_mpi_direct_case_run(
        regression_channel_single_yms_mpi
        lbm_regression_channel_d2q9
        channel_single_yms_d2_short.inputs
        channel_single
        2)
    regression_artifact_dir(
        channel_single_mpi_artifacts
        regression_channel_single_yms_mpi)
    list(
        APPEND channel_single_parallel_artifacts
        ${channel_single_mpi_artifacts})
endif()

regression_artifact_dir(
    channel_single_yms_d2_artifacts
    regression_channel_single_yms_d2q9)
regression_artifact_dir(
    channel_single_bouzidi_d2_artifacts
    regression_channel_single_bouzidi_d2q9)
regression_artifact_dir(
    channel_single_yms_d3_artifacts
    regression_channel_single_yms_d3q27)
add_test(
    NAME regression_channel_schema
    COMMAND ${Python3_EXECUTABLE}
        ${REGRESSION_CHECKER_DIR}/check_channel_single_short_matrix.py
        --yms-d2 ${channel_single_yms_d2_artifacts}
        --bouzidi-d2 ${channel_single_bouzidi_d2_artifacts}
        --yms-d3 ${channel_single_yms_d3_artifacts}
        --parallel ${channel_single_parallel_artifacts})
configure_lbm_test(
    regression_channel_schema regression serial 1 1)
set_tests_properties(
    regression_channel_schema PROPERTIES FIXTURES_REQUIRED channel_single)
set_property(
    TEST regression_channel_schema APPEND PROPERTY LABELS short)

add_direct_case_run(
    regression_channel_re20_nd20
    lbm_regression_channel_d2q9
    channel_re20_nd20_short.inputs
    channel_re20)
add_direct_case_run(
    regression_channel_re20_d3q27_nd20
    lbm_regression_channel_d3q27
    channel_re20_d3q27_nd20_short.inputs
    channel_re20)
regression_artifact_dir(
    channel_re20_d2_artifacts regression_channel_re20_nd20)
regression_artifact_dir(
    channel_re20_d3_artifacts regression_channel_re20_d3q27_nd20)
add_test(
    NAME regression_channel_re20_reference
    COMMAND ${Python3_EXECUTABLE}
        ${PROJECT_SOURCE_DIR}/tests/regression/checkers/channel_cylinder/check_re20_reference.py
        --reference
        ${PROJECT_SOURCE_DIR}/tests/regression/reference-data/channel_cylinder/dfg_re20_reference.csv
        --d2 ${channel_re20_d2_artifacts}
        --d3 ${channel_re20_d3_artifacts})
configure_lbm_test(
    regression_channel_re20_reference regression serial 1 1)
set_tests_properties(
    regression_channel_re20_reference PROPERTIES
    FIXTURES_REQUIRED channel_re20)
set_property(
    TEST regression_channel_re20_reference
    APPEND PROPERTY LABELS short reference)

add_direct_case_run(
    regression_channel_re100_nd40_short
    lbm_regression_channel_d2q9
    channel_re100_nd40_short.inputs
    channel_re100)
regression_artifact_dir(
    channel_re100_artifacts regression_channel_re100_nd40_short)
add_test(
    NAME regression_channel_re100_insufficient_cycles
    COMMAND ${Python3_EXECUTABLE}
        ${PROJECT_SOURCE_DIR}/tests/regression/checkers/channel_cylinder/check_re100_short_artifact.py
        --artifacts ${channel_re100_artifacts})
configure_lbm_test(
    regression_channel_re100_insufficient_cycles regression serial 1 1)
set_tests_properties(
    regression_channel_re100_insufficient_cycles PROPERTIES
    FIXTURES_REQUIRED channel_re100)
set_property(
    TEST regression_channel_re100_insufficient_cycles
    APPEND PROPERTY LABELS short reference)

lbm_test_command(failure_command lbm_regression_channel_d2q9 1
        ${CMAKE_CURRENT_SOURCE_DIR}/direct/inputs/channel_single_yms_d2_short.inputs
        channel_cylinder.variant=unsupported)
add_test(
    NAME regression_channel_invalid_variant
    COMMAND ${Python3_EXECUTABLE}
        ${REGRESSION_CHECKER_DIR}/expect_process_failure.py
        --contains
        "channel_cylinder.variant must be single_yms, single_bouzidi, tandem_distinct, tandem_shared, re20_yms or re100_yms" --
        ${failure_command})
configure_lbm_test(
    regression_channel_invalid_variant regression serial 1 1)
set_property(
    TEST regression_channel_invalid_variant
    APPEND PROPERTY LABELS short negative)

lbm_test_command(failure_command lbm_regression_channel_d3q27 1
        ${CMAKE_CURRENT_SOURCE_DIR}/direct/inputs/channel_single_yms_d3_short.inputs
        channel_cylinder.depth_cells=1)
add_test(
    NAME regression_channel_invalid_d3_depth
    COMMAND ${Python3_EXECUTABLE}
        ${REGRESSION_CHECKER_DIR}/expect_process_failure.py
        --contains
        "[case_validation:invalid_channel_cylinder_depth:mesh.n_cell.z]" --
        ${failure_command})
configure_lbm_test(
    regression_channel_invalid_d3_depth regression serial 1 1)
set_property(
    TEST regression_channel_invalid_d3_depth
    APPEND PROPERTY LABELS short negative)

add_direct_case_run(
    regression_channel_tandem_distinct_direct
    lbm_regression_channel_d2q9
    channel_tandem_distinct_short.inputs
    channel_tandem_direct)
add_direct_case_run(
    regression_channel_tandem_shared_direct
    lbm_regression_channel_d2q9
    channel_tandem_shared_short.inputs
    channel_tandem_direct)
regression_artifact_dir(
    tandem_distinct_direct_artifacts
    regression_channel_tandem_distinct_direct)
regression_artifact_dir(
    tandem_shared_direct_artifacts
    regression_channel_tandem_shared_direct)

set(tandem_parallel_checker_arguments)
if(AMReX_OMP AND LBM_TEST_OPENMP)
    add_openmp_direct_case_run(
        regression_channel_tandem_distinct_openmp
        lbm_regression_channel_d2q9
        channel_tandem_distinct_short.inputs
        channel_tandem_direct
        2)
    regression_artifact_dir(
        tandem_distinct_openmp_artifacts
        regression_channel_tandem_distinct_openmp)
    list(
        APPEND tandem_parallel_checker_arguments
        --distinct-parity ${tandem_distinct_openmp_artifacts})
endif()
if(AMReX_MPI)
    add_mpi_direct_case_run(
        regression_channel_tandem_shared_mpi
        lbm_regression_channel_d2q9
        channel_tandem_shared_short.inputs
        channel_tandem_direct
        2)
    regression_artifact_dir(
        tandem_shared_mpi_artifacts
        regression_channel_tandem_shared_mpi)
    list(
        APPEND tandem_parallel_checker_arguments
        --shared-parity ${tandem_shared_mpi_artifacts})
endif()

add_test(
    NAME regression_channel_tandem_multi_object
    COMMAND ${Python3_EXECUTABLE}
        ${REGRESSION_CHECKER_DIR}/check_channel_cylinder_multi_object.py
        --distinct-history
        ${tandem_distinct_direct_artifacts}/channel_cylinder_force_history.csv
        --distinct-metrics
        ${tandem_distinct_direct_artifacts}/channel_cylinder_metrics.csv
        --distinct-diagnostics
        ${tandem_distinct_direct_artifacts}/diagnostics.csv
        --shared-history
        ${tandem_shared_direct_artifacts}/channel_cylinder_force_history.csv
        --shared-metrics
        ${tandem_shared_direct_artifacts}/channel_cylinder_metrics.csv
        --shared-diagnostics
        ${tandem_shared_direct_artifacts}/diagnostics.csv
        ${tandem_parallel_checker_arguments}
        --expected-final-step 3)
configure_lbm_test(
    regression_channel_tandem_multi_object regression serial 1 1)
set_tests_properties(
    regression_channel_tandem_multi_object PROPERTIES
    FIXTURES_REQUIRED channel_tandem_direct)
set_property(
    TEST regression_channel_tandem_multi_object APPEND PROPERTY LABELS short)

add_test(
    NAME regression_channel_tandem_checker_failures
    COMMAND ${Python3_EXECUTABLE}
        ${REGRESSION_CHECKER_DIR}/check_channel_cylinder_multi_object_synthetic.py)
configure_lbm_test(
    regression_channel_tandem_checker_failures regression serial 1 1)
set_property(
    TEST regression_channel_tandem_checker_failures
    APPEND PROPERTY LABELS short synthetic)

add_direct_case_run(
    regression_cavity_direct
    lbm_regression_cavity_d2q9
    cavity_d2q9_guo_short.inputs
    cavity_parity)
add_direct_case_run(
    regression_cavity_d2q9_hbb_direct
    lbm_regression_cavity_d2q9
    cavity_d2q9_hbb_short.inputs
    cavity_variants)
add_direct_case_run(
    regression_cavity_d3q27_guo_direct
    lbm_regression_cavity_d3q27
    cavity_d3q27_guo_short.inputs
    cavity_variants)
add_direct_case_run(
    regression_cavity_d3q27_hbb_direct
    lbm_regression_cavity_d3q27
    cavity_d3q27_hbb_short.inputs
    cavity_variants)

regression_artifact_dir(direct_cavity_artifacts regression_cavity_direct)
regression_artifact_dir(d2_hbb_cavity_artifacts regression_cavity_d2q9_hbb_direct)
regression_artifact_dir(d3_guo_cavity_artifacts regression_cavity_d3q27_guo_direct)
regression_artifact_dir(d3_hbb_cavity_artifacts regression_cavity_d3q27_hbb_direct)
add_test(
    NAME regression_cavity_schema
    COMMAND ${Python3_EXECUTABLE} -B
        ${REGRESSION_CHECKER_DIR}/check_cavity_short_matrix.py
        --d2-guo ${direct_cavity_artifacts}
        --d2-hbb ${d2_hbb_cavity_artifacts}
        --d3-guo ${d3_guo_cavity_artifacts}
        --d3-hbb ${d3_hbb_cavity_artifacts})
configure_lbm_test(regression_cavity_schema regression serial 1 1)
set_tests_properties(
    regression_cavity_schema PROPERTIES
    FIXTURES_REQUIRED "cavity_parity;cavity_variants")
set_property(TEST regression_cavity_schema APPEND PROPERTY LABELS short)

list(LENGTH uniform_parallel_artifacts uniform_parallel_count)
if(uniform_parallel_count GREATER 1)
    add_test(
        NAME regression_uniform_parallel_parity
        COMMAND ${Python3_EXECUTABLE}
            ${REGRESSION_CHECKER_DIR}/compare_public_artifacts.py
            --numeric diagnostics.csv uniform_flow_metrics.csv --
            ${uniform_parallel_artifacts})
    configure_lbm_test(regression_uniform_parallel_parity regression parallel 1 1)
    set_tests_properties(
        regression_uniform_parallel_parity PROPERTIES
        FIXTURES_REQUIRED uniform_parallel)
    set_property(TEST regression_uniform_parallel_parity APPEND PROPERTY LABELS short)
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/PoiseuilleRegression.cmake)
