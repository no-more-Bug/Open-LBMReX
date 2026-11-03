if(LBM_TEST_MPI_LAUNCHER)
    set(numerical_mpiexec ${LBM_TEST_MPI_LAUNCHER})
else()
    set(numerical_mpiexec ${MPIEXEC_EXECUTABLE})
endif()

function(add_long_numerical_test test_name family d2_target d3_target processors timeout_seconds)
    set(work_dir "${CMAKE_CURRENT_BINARY_DIR}/work/${test_name}")
    set(tmp_dir "${work_dir}/tmp")
    file(MAKE_DIRECTORY "${tmp_dir}")
    math(EXPR processors "${processors} * ${LBM_TEST_CPUS_PER_RANK}")
    add_test(
        NAME ${test_name}
        COMMAND
            bash "${CMAKE_CURRENT_SOURCE_DIR}/${family}/run.sh"
            "$<TARGET_FILE:${d2_target}>"
            "$<TARGET_FILE:${d3_target}>"
            "${work_dir}"
            "${numerical_mpiexec}" ${ARGN})
    set_tests_properties(
        ${test_name}
        PROPERTIES
            COST 100
            ENVIRONMENT
                "${LBM_SOURCE_ENVIRONMENT};LBM_TEST_R5_MPI_RANKS=${LBM_TEST_R5_MPI_RANKS};LBM_TEST_AMR_MPI_RANKS=${LBM_TEST_AMR_MPI_RANKS};LBM_TEST_PYTHON=${Python3_EXECUTABLE};OMP_NUM_THREADS=1;OMP_DYNAMIC=FALSE;OMP_PROC_BIND=close;OMP_PLACES=cores;TMPDIR=${tmp_dir};TEMP=${tmp_dir};TMP=${tmp_dir};OMPI_MCA_orte_tmpdir_base=${tmp_dir}"
            LABELS "${LBM_TEST_BACKEND_LABEL};long;numerical"
            PROCESSORS ${processors}
            TIMEOUT ${timeout_seconds}
            WORKING_DIRECTORY "${work_dir}")
endfunction()

function(add_numerical_family_targets family target_base)
    add_lbm_test_target(${target_base} ${family}/src/main.cpp)
    add_lbm_test_target(${target_base}_d3q27 ${family}/src/main.cpp)
    target_compile_definitions(
        ${target_base}_d3q27 PRIVATE LBM_DESCRIPTOR_D3Q27=1)
endfunction()

add_numerical_family_targets(
    taylor_green_vortex lbm_numerical_taylor_green_vortex)
add_numerical_family_targets(
    uniform_flow lbm_numerical_uniform_flow)
add_numerical_family_targets(
    poiseuille_flow lbm_numerical_poiseuille_flow)
add_numerical_family_targets(
    lid_driven_cavity lbm_numerical_lid_driven_cavity)
add_numerical_family_targets(
    channel_cylinder lbm_numerical_channel_cylinder)

add_long_numerical_test(
    numerical_tgv_convergence
    taylor_green_vortex
    lbm_numerical_taylor_green_vortex
    lbm_numerical_taylor_green_vortex_d3q27
    4
    900)
add_long_numerical_test(
    numerical_uniform_flow
    uniform_flow
    lbm_numerical_uniform_flow
    lbm_numerical_uniform_flow_d3q27
    1
    600)
add_long_numerical_test(
    numerical_guo_pressure_poiseuille
    poiseuille_flow
    lbm_numerical_poiseuille_flow
    lbm_numerical_poiseuille_flow_d3q27
    4
    1800)
add_long_numerical_test(
    numerical_hbb_cavity_ghia
    lid_driven_cavity
    lbm_numerical_lid_driven_cavity
    lbm_numerical_lid_driven_cavity_d3q27
    4
    7200)
set(r5_processors 18)
if(LBM_TEST_R5_MPI_RANKS GREATER 0)
    set(r5_processors ${LBM_TEST_R5_MPI_RANKS})
endif()
add_long_numerical_test(
    numerical_channel_cylinder_baseline
    channel_cylinder
    lbm_numerical_channel_cylinder
    lbm_numerical_channel_cylinder_d3q27
    ${r5_processors}
    28800)

# Stage-6-03 families share production AMR seams and tests-owned native output.
add_numerical_family_targets(
    amr_periodic_shear lbm_numerical_amr_periodic_shear)

foreach(descriptor d2 d3)
    foreach(mode axis diagonal)
        add_long_numerical_test(
            numerical_amr_shear_${descriptor}_${mode}
            amr_periodic_shear
            lbm_numerical_amr_periodic_shear
            lbm_numerical_amr_periodic_shear_d3q27
            ${LBM_TEST_AMR_MPI_RANKS}
            3300
            ${descriptor}-${mode})
    endforeach()
endforeach()
# Stage-6-05 Re100 keeps its separate runner and the original Re20 identities.
add_lbm_test_target(lbm_numerical_amr_re100 amr_channel_cylinder/src/re100_main.cpp)
add_lbm_test_target(lbm_numerical_amr_re100_d3q27 amr_channel_cylinder/src/re100_main.cpp)
target_compile_definitions(lbm_numerical_amr_re100_d3q27 PRIVATE LBM_DESCRIPTOR_D3Q27=1)
set(re100_ranks 32)
if(LBM_ENABLE_CUDA)
    set(re100_ranks 6)
endif()
foreach(scope numerical integration)
    foreach(descriptor d2 d3)
        if(scope STREQUAL numerical AND descriptor STREQUAL d3 AND NOT LBM_ENABLE_CUDA)
            continue()
        endif()
        foreach(scheme yms bouzidi)
            foreach(layout fine coarse amr)
                set(test_name ${scope}_amr_re100_${descriptor}_${scheme}_${layout})
                set(reference_args)
                set(reference_test numerical_amr_re100_${descriptor}_${scheme}_fine)
                if(scope STREQUAL numerical AND layout STREQUAL amr)
                    list(APPEND reference_args "${CMAKE_CURRENT_BINARY_DIR}/work/${reference_test}")
                endif()
                add_test(NAME ${test_name}
                    COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/amr_channel_cylinder/run_re100.sh"
                        "$<TARGET_FILE:lbm_numerical_amr_re100>"
                        "$<TARGET_FILE:lbm_numerical_amr_re100_d3q27>"
                        "${CMAKE_CURRENT_BINARY_DIR}/work/${test_name}" "${numerical_mpiexec}"
                        ${descriptor}-${scheme}-${layout} ${reference_args})
                configure_lbm_test(${test_name} ${scope} mpi ${re100_ranks} 1)
                set_property(TEST ${test_name} APPEND PROPERTY ENVIRONMENT
                    "LBM_TEST_AMR_MPI_RANKS=${re100_ranks};LBM_TEST_PYTHON=${Python3_EXECUTABLE}")
                if(scope STREQUAL integration)
                    set_tests_properties(${test_name} PROPERTIES TIMEOUT 600
                        LABELS "${LBM_TEST_BACKEND_LABEL};short;integration;mpi")
                    set_property(TEST ${test_name} APPEND PROPERTY ENVIRONMENT
                        "LBM_TEST_MAX_STEP=8;LBM_TEST_ENGINEERING=1")
                else()
                    set_tests_properties(${test_name} PROPERTIES TIMEOUT 6900 COST 100
                        LABELS "${LBM_TEST_BACKEND_LABEL};long;numerical")
                    if(layout STREQUAL amr)
                        set_tests_properties(${test_name} PROPERTIES DEPENDS ${reference_test})
                    endif()
                endif()
            endforeach()
        endforeach()
    endforeach()
endforeach()
add_test(NAME integration_amr_re100_checker
    COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_SOURCE_DIR}/amr_channel_cylinder/test_re100_checker.py)
configure_lbm_test(integration_amr_re100_checker integration serial 1 1)
set_tests_properties(integration_amr_re100_checker PROPERTIES TIMEOUT 300)
add_test(NAME integration_amr_shear_checker
    COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_SOURCE_DIR}/amr_periodic_shear/test_periodic_shear.py)
configure_lbm_test(integration_amr_shear_checker integration serial 1 1)

add_numerical_family_targets(
    amr_double_shear_layer lbm_numerical_amr_double_shear_layer)
foreach(descriptor d2 d3)
    foreach(entry ref128 ref256 ref512 baseline 3l 2r p1 p2 p3)
        set(reference_args)
        if(entry STREQUAL baseline)
            set(reference_test numerical_amr_dsl_${descriptor}_ref256)
        elseif(NOT entry MATCHES "^ref")
            set(reference_test numerical_amr_dsl_${descriptor}_ref512)
        endif()
        if(NOT entry MATCHES "^ref")
            list(APPEND reference_args "${CMAKE_CURRENT_BINARY_DIR}/work/${reference_test}")
        endif()
        set(test_name numerical_amr_dsl_${descriptor}_${entry})
        add_long_numerical_test(
            ${test_name} amr_double_shear_layer
            lbm_numerical_amr_double_shear_layer lbm_numerical_amr_double_shear_layer_d3q27
            ${LBM_TEST_AMR_MPI_RANKS} 3300 ${descriptor}-${entry} ${reference_args})
        if(NOT entry MATCHES "^ref")
            # Ordering only: a reference scientific failure must not skip the AMR run.
            set_tests_properties(${test_name} PROPERTIES DEPENDS ${reference_test})
        endif()
    endforeach()
endforeach()
add_test(NAME integration_amr_dsl_checker
    COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_SOURCE_DIR}/amr_double_shear_layer/test_double_shear_layer.py)
configure_lbm_test(integration_amr_dsl_checker integration serial 1 1)
set_tests_properties(integration_amr_dsl_checker PROPERTIES TIMEOUT 300)

# Stage-6-04 HBB cavity: references finish first even when their scientific gate fails.
add_numerical_family_targets(
    amr_lid_driven_cavity lbm_numerical_amr_lid_driven_cavity)
foreach(descriptor d2 d3)
    foreach(entry fine coarse amr)
        set(reference_args)
        set(reference_test numerical_amr_cavity_${descriptor}_fine)
        if(NOT entry STREQUAL fine)
            list(APPEND reference_args "${CMAKE_CURRENT_BINARY_DIR}/work/${reference_test}")
        endif()
        set(test_name numerical_amr_cavity_${descriptor}_${entry})
        add_long_numerical_test(
            ${test_name} amr_lid_driven_cavity
            lbm_numerical_amr_lid_driven_cavity lbm_numerical_amr_lid_driven_cavity_d3q27
            ${LBM_TEST_AMR_MPI_RANKS} 6900 ${descriptor}-${entry} ${reference_args})
        if(NOT entry STREQUAL fine)
            set_tests_properties(${test_name} PROPERTIES DEPENDS ${reference_test})
        endif()
    endforeach()
endforeach()
add_test(NAME integration_amr_cavity_checker
    COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_SOURCE_DIR}/amr_lid_driven_cavity/test_checkers.py)
configure_lbm_test(integration_amr_cavity_checker integration serial 1 1)
set_tests_properties(integration_amr_cavity_checker PROPERTIES TIMEOUT 300)

# Re20 references remain explicit; dependencies order an already-selected matrix.
add_numerical_family_targets(
    amr_channel_cylinder lbm_numerical_amr_channel_cylinder)
foreach(descriptor d2 d3)
    foreach(scheme yms bouzidi)
        foreach(entry fine coarse amr)
            set(reference_args)
            set(dependencies)
            set(reference_test numerical_amr_re20_${descriptor}_${scheme}_fine)
            if(NOT entry STREQUAL fine)
                list(APPEND reference_args "${CMAKE_CURRENT_BINARY_DIR}/work/${reference_test}")
                list(APPEND dependencies ${reference_test})
            endif()
            set(test_name numerical_amr_re20_${descriptor}_${scheme}_${entry})
            add_long_numerical_test(
                ${test_name} amr_channel_cylinder
                lbm_numerical_amr_channel_cylinder lbm_numerical_amr_channel_cylinder_d3q27
                ${LBM_TEST_AMR_MPI_RANKS} 28800 ${descriptor}-${scheme}-${entry} ${reference_args})
            if(descriptor STREQUAL d3)
                set(d2_test numerical_amr_re20_d2_${scheme}_${entry})
                list(APPEND dependencies ${d2_test})
                set_property(TEST ${test_name} APPEND PROPERTY ENVIRONMENT
                    "LBM_RE20_D2_REFERENCE=${CMAKE_CURRENT_BINARY_DIR}/work/${d2_test}")
            endif()
            if(dependencies)
                set_tests_properties(${test_name} PROPERTIES DEPENDS "${dependencies}")
            endif()
        endforeach()
    endforeach()
endforeach()
add_test(NAME integration_amr_re20_checker
    COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_SOURCE_DIR}/amr_channel_cylinder/test_checkers.py)
configure_lbm_test(integration_amr_re20_checker integration serial 1 1)
set_tests_properties(integration_amr_re20_checker PROPERTIES TIMEOUT 600)

# Short Object-force engineering checks share the cylinder composition locality.
add_lbm_test_target(lbm_tandem_cylinder_integration amr_channel_cylinder/src/tandem_main.cpp)
add_lbm_test_target(lbm_tandem_cylinder_integration_d3q27 amr_channel_cylinder/src/tandem_main.cpp)
target_compile_definitions(lbm_tandem_cylinder_integration_d3q27 PRIVATE LBM_DESCRIPTOR_D3Q27=1)
foreach(descriptor d2 d3)
    foreach(scheme yms bouzidi)
        foreach(layout cross same)
            set(distinct_test integration_tandem_${descriptor}_${scheme}_${layout}_distinct)
            foreach(grouping distinct shared)
                set(test_name integration_tandem_${descriptor}_${scheme}_${layout}_${grouping})
                set(work_dir "${CMAKE_CURRENT_BINARY_DIR}/work/${test_name}")
                set(reference_args)
                if(grouping STREQUAL shared)
                    list(APPEND reference_args "${CMAKE_CURRENT_BINARY_DIR}/work/${distinct_test}")
                endif()
                add_test(NAME ${test_name}
                    COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/amr_channel_cylinder/run_tandem.sh"
                        "$<TARGET_FILE:lbm_tandem_cylinder_integration>"
                        "$<TARGET_FILE:lbm_tandem_cylinder_integration_d3q27>"
                        "${work_dir}" "${numerical_mpiexec}"
                        ${descriptor}-${scheme}-${layout}-${grouping} ${reference_args})
                configure_lbm_test(${test_name} integration mpi ${LBM_TEST_AMR_MPI_RANKS} 1)
                set_tests_properties(${test_name} PROPERTIES TIMEOUT 300
                    LABELS "${LBM_TEST_BACKEND_LABEL};short;integration;mpi")
                set_property(TEST ${test_name} APPEND PROPERTY ENVIRONMENT
                    "LBM_TEST_AMR_MPI_RANKS=${LBM_TEST_AMR_MPI_RANKS};LBM_TEST_PYTHON=${Python3_EXECUTABLE}")
                if(grouping STREQUAL shared)
                    set_tests_properties(${test_name} PROPERTIES DEPENDS ${distinct_test})
                endif()
            endforeach()
        endforeach()
    endforeach()
endforeach()
add_test(NAME integration_tandem_checker
    COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_SOURCE_DIR}/amr_channel_cylinder/test_tandem_checker.py)
configure_lbm_test(integration_tandem_checker integration serial 1 1)
add_test(NAME integration_tandem_compare_checker
    COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_SOURCE_DIR}/amr_channel_cylinder/test_tandem_compare.py)
configure_lbm_test(integration_tandem_compare_checker integration serial 1 1)

# Stage-6-05: full scientific runs and explicit eight-step engineering runs.
add_numerical_family_targets(amr_uniform_flow lbm_numerical_amr_uniform_flow)
add_numerical_family_targets(amr_taylor_green_vortex lbm_numerical_amr_tgv)
set(flow_ranks 4)
set(tgv_ranks 4)
if(LBM_ENABLE_CUDA)
    set(flow_ranks 1)
    set(tgv_ranks 2)
endif()
foreach(scope numerical integration)
    foreach(descriptor d2 d3)
        set(test_name ${scope}_amr_uniform_${descriptor})
        add_long_numerical_test(${test_name} amr_uniform_flow
            lbm_numerical_amr_uniform_flow lbm_numerical_amr_uniform_flow_d3q27
            ${flow_ranks} 6900 ${descriptor})
        set_property(TEST ${test_name} PROPERTY ENVIRONMENT_MODIFICATION
            "LBM_TEST_AMR_MPI_RANKS=set:${flow_ranks}")
        if(scope STREQUAL integration)
            set_tests_properties(${test_name} PROPERTIES TIMEOUT 300
                LABELS "${LBM_TEST_BACKEND_LABEL};short;integration;mpi")
            set_property(TEST ${test_name} APPEND PROPERTY ENVIRONMENT
                "LBM_TEST_MAX_STEP=8;LBM_TEST_ENGINEERING=1")
        endif()
        foreach(cells 20 40 80)
            foreach(layout uniform amr)
                set(test_name ${scope}_amr_tgv_${descriptor}_${cells}_${layout})
                set(reference_args)
                set(reference_test numerical_amr_tgv_${descriptor}_${cells}_uniform)
                if(scope STREQUAL numerical AND layout STREQUAL amr)
                    list(APPEND reference_args "${CMAKE_CURRENT_BINARY_DIR}/work/${reference_test}")
                endif()
                add_long_numerical_test(${test_name} amr_taylor_green_vortex
                    lbm_numerical_amr_tgv lbm_numerical_amr_tgv_d3q27
                    ${tgv_ranks} 6900 ${descriptor}-${cells}-${layout} ${reference_args})
                set_property(TEST ${test_name} PROPERTY ENVIRONMENT_MODIFICATION
                    "LBM_TEST_AMR_MPI_RANKS=set:${tgv_ranks}")
                if(scope STREQUAL integration)
                    set_tests_properties(${test_name} PROPERTIES TIMEOUT 300
                        LABELS "${LBM_TEST_BACKEND_LABEL};short;integration;mpi")
                    set_property(TEST ${test_name} APPEND PROPERTY ENVIRONMENT
                        "LBM_TEST_MAX_STEP=8;LBM_TEST_ENGINEERING=1")
                elseif(layout STREQUAL amr)
                    set_tests_properties(${test_name} PROPERTIES DEPENDS ${reference_test})
                endif()
            endforeach()
        endforeach()
    endforeach()
endforeach()
foreach(family uniform tgv)
    if(family STREQUAL uniform)
        set(locality amr_uniform_flow)
    else()
        set(locality amr_taylor_green_vortex)
    endif()
    add_test(NAME integration_amr_${family}_checker
        COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_SOURCE_DIR}/${locality}/test_checkers.py)
    configure_lbm_test(integration_amr_${family}_checker integration serial 1 1)
    set_tests_properties(integration_amr_${family}_checker PROPERTIES TIMEOUT 300)
endforeach()

# Stage-7-01 periodic dynamic DSL: four compositions share one producer.
add_lbm_test_target(lbm_numerical_dynamic_double_shear_layer amr_double_shear_layer/src/dynamic_main.cpp)
add_lbm_test_target(lbm_numerical_dynamic_double_shear_layer_d3q27 amr_double_shear_layer/src/dynamic_main.cpp)
target_compile_definitions(lbm_numerical_dynamic_double_shear_layer_d3q27 PRIVATE LBM_DESCRIPTOR_D3Q27=1)
foreach(descriptor d2 d3)
    foreach(depth 2l 3l)
        foreach(ranks 1 2)
            if(ranks EQUAL 2 AND NOT AMReX_MPI)
                continue()
            endif()
            set(test_name integration_dynamic_dsl_${descriptor}_${depth}_r${ranks})
            set(work_dir "${CMAKE_CURRENT_BINARY_DIR}/work/${test_name}")
            add_test(NAME ${test_name} COMMAND bash
                "${CMAKE_CURRENT_SOURCE_DIR}/amr_double_shear_layer/run_dynamic.sh"
                "$<TARGET_FILE:lbm_numerical_dynamic_double_shear_layer>"
                "$<TARGET_FILE:lbm_numerical_dynamic_double_shear_layer_d3q27>"
                "${work_dir}" "${numerical_mpiexec}" "${descriptor}-${depth}")
            if(ranks EQUAL 1)
                set(parallel_kind serial)
            else()
                set(parallel_kind mpi)
            endif()
            configure_lbm_test(${test_name} integration ${parallel_kind} ${ranks} 1)
            set_tests_properties(${test_name} PROPERTIES TIMEOUT 600)
            set_property(TEST ${test_name} APPEND PROPERTY ENVIRONMENT
                "LBM_TEST_AMR_MPI_RANKS=${ranks};LBM_TEST_PYTHON=${Python3_EXECUTABLE};LBM_TEST_MAX_STEP=16")
        endforeach()
    endforeach()
endforeach()
add_test(NAME integration_dynamic_dsl_checker COMMAND ${Python3_EXECUTABLE} -B
    ${CMAKE_CURRENT_SOURCE_DIR}/amr_double_shear_layer/test_dynamic_double_shear_layer.py)
configure_lbm_test(integration_dynamic_dsl_checker integration serial 1 1)
set_tests_properties(integration_dynamic_dsl_checker PROPERTIES TIMEOUT 300)


# Stage-7-02 fixed-band DSL uses an explicit composition and artifact identity.
add_lbm_test_target(lbm_fixed_band_dsl_d2 amr_double_shear_layer/src/dynamic_main.cpp)
add_lbm_test_target(lbm_fixed_band_dsl_d3 amr_double_shear_layer/src/dynamic_main.cpp)
target_compile_definitions(lbm_fixed_band_dsl_d2 PRIVATE LBM_DSL_FIXED_BAND=1)
target_compile_definitions(lbm_fixed_band_dsl_d3 PRIVATE LBM_DSL_FIXED_BAND=1 LBM_DESCRIPTOR_D3Q27=1)
set(stage702_ranks 1)
if(AMReX_MPI)
    list(APPEND stage702_ranks ${LBM_TEST_AMR_MPI_RANKS})
    list(REMOVE_DUPLICATES stage702_ranks)
endif()
foreach(descriptor d2 d3)
    foreach(ranks IN LISTS stage702_ranks)
        set(test_name integration_fixed_band_dsl_${descriptor}_r${ranks})
        set(work_dir "${CMAKE_CURRENT_BINARY_DIR}/work/${test_name}")
        add_test(NAME ${test_name} COMMAND bash
            "${CMAKE_CURRENT_SOURCE_DIR}/amr_double_shear_layer/run_dynamic.sh"
            "$<TARGET_FILE:lbm_fixed_band_dsl_d2>" "$<TARGET_FILE:lbm_fixed_band_dsl_d3>"
            "${work_dir}" "${numerical_mpiexec}" "${descriptor}-fixed-band")
        if(ranks EQUAL 1)
            set(parallel_kind serial)
        else()
            set(parallel_kind mpi)
        endif()
        configure_lbm_test(${test_name} integration ${parallel_kind} ${ranks} 1)
        set_tests_properties(${test_name} PROPERTIES TIMEOUT 1800)
        set_property(TEST ${test_name} APPEND PROPERTY ENVIRONMENT
            "LBM_TEST_AMR_MPI_RANKS=${ranks};LBM_TEST_PYTHON=${Python3_EXECUTABLE};LBM_TEST_MAX_STEP=128;LBM_TEST_DSL_PROFILE=1")
    endforeach()
endforeach()
add_test(NAME integration_fixed_band_dsl_checker COMMAND ${Python3_EXECUTABLE} -B
    ${CMAKE_CURRENT_SOURCE_DIR}/amr_double_shear_layer/test_fixed_band.py)
configure_lbm_test(integration_fixed_band_dsl_checker integration serial 1 1)
set_tests_properties(integration_fixed_band_dsl_checker PROPERTIES TIMEOUT 300)


# Stage-7-02 dynamic Re100 observations; full horizons stay explicitly unregistered.
add_lbm_test_target(lbm_dynamic_cylinder_d2 amr_channel_cylinder/src/dynamic_main.cpp)
add_lbm_test_target(lbm_dynamic_cylinder_d3 amr_channel_cylinder/src/dynamic_main.cpp)
target_compile_definitions(lbm_dynamic_cylinder_d3 PRIVATE LBM_DESCRIPTOR_D3Q27=1)
foreach(descriptor d2 d3)
    foreach(variant single tandem-a tandem-b)
        foreach(ranks IN LISTS stage702_ranks)
            set(test_name integration_dynamic_cylinder_${descriptor}_${variant}_r${ranks})
            set(work_dir "${CMAKE_CURRENT_BINARY_DIR}/work/${test_name}")
            add_test(NAME ${test_name} COMMAND bash
                "${CMAKE_CURRENT_SOURCE_DIR}/amr_channel_cylinder/run_dynamic.sh"
                "$<TARGET_FILE:lbm_dynamic_cylinder_d2>" "$<TARGET_FILE:lbm_dynamic_cylinder_d3>"
                "${work_dir}" "${numerical_mpiexec}" "${descriptor}-${variant}")
            if(ranks EQUAL 1)
                set(parallel_kind serial)
            else()
                set(parallel_kind mpi)
            endif()
            configure_lbm_test(${test_name} integration ${parallel_kind} ${ranks} 1)
            set_tests_properties(${test_name} PROPERTIES TIMEOUT 1800)
            set_property(TEST ${test_name} APPEND PROPERTY ENVIRONMENT
                "LBM_TEST_AMR_MPI_RANKS=${ranks};LBM_TEST_PYTHON=${Python3_EXECUTABLE};LBM_TEST_MAX_STEP=32;LBM_TEST_CYLINDER_PROFILE=1")
        endforeach()
    endforeach()
endforeach()
add_test(NAME integration_dynamic_cylinder_checker COMMAND ${Python3_EXECUTABLE} -B
    ${CMAKE_CURRENT_SOURCE_DIR}/amr_channel_cylinder/test_dynamic_cylinder.py)
configure_lbm_test(integration_dynamic_cylinder_checker integration serial 1 1)
set_tests_properties(integration_dynamic_cylinder_checker PROPERTIES TIMEOUT 300)
