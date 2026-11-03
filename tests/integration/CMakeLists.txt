foreach(descriptor IN ITEMS d2 d3)
    set(target lbm_dynamic_amr_integration_${descriptor})
    add_lbm_test_target(${target}
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/DynamicAmrIntegrationMain.cpp)
    target_compile_definitions(${target} PRIVATE
        LBM_DYNAMIC_GRID_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/inputs/dynamic_regrid_clustering")
    if(descriptor STREQUAL "d3")
        target_compile_definitions(${target} PRIVATE LBM_DESCRIPTOR_D3Q27=1)
    endif()
    add_lbm_serial_test(integration_dynamic_${descriptor}_prepare_serial ${target} integration)
    foreach(mode IN ITEMS tagging initialize velocity remap regrid relayout clustering constraints_prepare)
        add_lbm_serial_test(integration_dynamic_${descriptor}_${mode}_serial ${target} integration test.mode=${mode})
        set_tests_properties(integration_dynamic_${descriptor}_${mode}_serial PROPERTIES TIMEOUT 360)
        if(AMReX_MPI)
            add_lbm_mpi_test(integration_dynamic_${descriptor}_${mode}_mpi2 ${target} integration 2 test.mode=${mode})
            set_tests_properties(integration_dynamic_${descriptor}_${mode}_mpi2 PROPERTIES TIMEOUT 360)
        endif()
    endforeach()
    add_lbm_serial_test(integration_dynamic_${descriptor}_constraints_finalize_serial
        ${target} integration test.mode=constraints_finalize)
    set_tests_properties(integration_dynamic_${descriptor}_constraints_finalize_serial PROPERTIES TIMEOUT 360)
    if(AMReX_MPI)
        add_lbm_mpi_test(integration_dynamic_${descriptor}_constraints_finalize_mpi${LBM_TEST_AMR_MPI_RANKS}
            ${target} integration ${LBM_TEST_AMR_MPI_RANKS} test.mode=constraints_finalize)
        set_tests_properties(integration_dynamic_${descriptor}_constraints_finalize_mpi${LBM_TEST_AMR_MPI_RANKS}
            PROPERTIES TIMEOUT 360)
    endif()
    foreach(mode IN ITEMS constrained_tagging constrained_initialize constrained_regrid boundary_force boundary_relayout boundary_domain)
        add_lbm_serial_test(integration_dynamic_${descriptor}_${mode}_serial
            ${target} integration test.mode=${mode})
        set_tests_properties(integration_dynamic_${descriptor}_${mode}_serial PROPERTIES TIMEOUT 360)
        if(AMReX_MPI)
            add_lbm_mpi_test(integration_dynamic_${descriptor}_${mode}_mpi${LBM_TEST_AMR_MPI_RANKS}
                ${target} integration ${LBM_TEST_AMR_MPI_RANKS} test.mode=${mode})
            set_tests_properties(integration_dynamic_${descriptor}_${mode}_mpi${LBM_TEST_AMR_MPI_RANKS}
                PROPERTIES TIMEOUT 360)
        endif()
    endforeach()
    if(AMReX_MPI)
        add_lbm_mpi_test(integration_dynamic_${descriptor}_prepare_mpi2 ${target} integration 2)
    endif()
endforeach()

foreach(descriptor IN ITEMS d2 d3)
    set(target lbm_amr_boundary_integration_${descriptor})
    add_lbm_test_target(${target} ${CMAKE_CURRENT_SOURCE_DIR}/lbm/AmrBoundaryIntegrationMain.cpp)
    target_compile_definitions(${target} PRIVATE AMREX_BOUND_CHECK)
    if(descriptor STREQUAL "d3")
        target_compile_definitions(${target} PRIVATE LBM_DESCRIPTOR_D3Q27=1)
    endif()
    add_lbm_serial_test(integration_boundary_${descriptor}_prepare_serial ${target} integration)
    add_lbm_serial_test(integration_boundary_${descriptor}_evolve_serial ${target} integration test.mode=evolve)
    add_lbm_serial_test(integration_boundary_${descriptor}_force_serial ${target} integration test.mode=force)
    set_tests_properties(integration_boundary_${descriptor}_prepare_serial PROPERTIES TIMEOUT 180)
    set_tests_properties(integration_boundary_${descriptor}_evolve_serial PROPERTIES TIMEOUT 180)
    set_tests_properties(integration_boundary_${descriptor}_force_serial PROPERTIES TIMEOUT 180)
    if(AMReX_MPI)
        add_lbm_mpi_test(integration_boundary_${descriptor}_prepare_mpi2 ${target} integration 2)
        add_lbm_mpi_test(integration_boundary_${descriptor}_evolve_mpi2 ${target} integration 2 test.mode=evolve)
        add_lbm_mpi_test(integration_boundary_${descriptor}_force_mpi2 ${target} integration 2 test.mode=force)
        set_tests_properties(integration_boundary_${descriptor}_prepare_mpi2 PROPERTIES TIMEOUT 180)
        set_tests_properties(integration_boundary_${descriptor}_evolve_mpi2 PROPERTIES TIMEOUT 180)
        set_tests_properties(integration_boundary_${descriptor}_force_mpi2 PROPERTIES TIMEOUT 180)
    endif()
endforeach()

add_lbm_test_target(
    lbm_internal_solid_integration_d2q9
    ${CMAKE_CURRENT_SOURCE_DIR}/lbm/InternalSolidIntegrationMain.cpp)
target_include_directories(
    lbm_internal_solid_integration_d2q9 PRIVATE ${PROJECT_SOURCE_DIR}/tests)
add_lbm_test_target(
    lbm_internal_solid_integration_d3q27
    ${CMAKE_CURRENT_SOURCE_DIR}/lbm/InternalSolidIntegrationMain.cpp)
target_include_directories(
    lbm_internal_solid_integration_d3q27 PRIVATE ${PROJECT_SOURCE_DIR}/tests)
target_compile_definitions(
    lbm_internal_solid_integration_d3q27
    PRIVATE LBM_DESCRIPTOR_D3Q27=1)

add_lbm_test_target(
    lbm_single_level_preparation_integration_d2q9
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/SingleLevelPreparationIntegrationMain.cpp)
add_lbm_test_target(
    lbm_single_level_preparation_integration_d3q27
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/SingleLevelPreparationIntegrationMain.cpp)
target_compile_definitions(
    lbm_single_level_preparation_integration_d3q27
    PRIVATE LBM_DESCRIPTOR_D3Q27=1)

add_lbm_test_target(
    lbm_single_level_runtime_integration_d2q9
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/SingleLevelRuntimeIntegrationMain.cpp)
add_lbm_test_target(
    lbm_single_level_runtime_integration_d3q27
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/SingleLevelRuntimeIntegrationMain.cpp)
target_compile_definitions(
    lbm_single_level_runtime_integration_d3q27
    PRIVATE LBM_DESCRIPTOR_D3Q27=1)

add_lbm_test_target(
    lbm_fixed_thickness_integration
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/FixedThicknessIntegrationMain.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/FixedThicknessCylinderTests.cpp)

add_lbm_test_target(
    lbm_static_hierarchy_integration
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/StaticHierarchyIntegrationMain.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/StaticHierarchyLayoutTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/StaticHierarchy2DFixtures.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/StaticHierarchyLifecycleTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/StaticHierarchyOutputTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/StaticHierarchySnapshot.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/StaticHierarchyTopologyFixtures.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/StaticHierarchyTopologyOracle.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/StaticHierarchyTopologyTests.cpp)

add_lbm_test_target(
    lbm_static_hierarchy_integration_d3q27
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/StaticHierarchyIntegrationMain.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/StaticHierarchy3DTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/StaticHierarchy3DFixtures.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/StaticHierarchy3DLayoutTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/StaticHierarchy3DLifecycleTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/StaticHierarchy3DOutputTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/StaticHierarchySnapshot.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/StaticHierarchyTopologyFixtures.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/StaticHierarchyTopologyOracle.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/StaticHierarchyTopologyTests.cpp)
target_compile_definitions(lbm_static_hierarchy_integration_d3q27 PRIVATE LBM_DESCRIPTOR_D3Q27=1)

# Stage-6-03 C3: independent forward-push volume reference through real exchange.
foreach(descriptor IN ITEMS d2 d3)
    set(target lbm_coarse_fine_integration_${descriptor})
    add_lbm_test_target(${target}
        ${CMAKE_CURRENT_SOURCE_DIR}/lbm/CoarseFineCouplingIntegrationMain.cpp
        ${PROJECT_SOURCE_DIR}/tests/common/AmrPushReference.cpp)
    if(descriptor STREQUAL "d3")
        target_compile_definitions(${target} PRIVATE LBM_DESCRIPTOR_D3Q27=1)
    endif()
    set(serial integration_cf_${descriptor}_exchange_serial)
    add_lbm_serial_test(${serial} ${target} integration)
    set_tests_properties(${serial} PROPERTIES TIMEOUT 180)
    if(AMReX_MPI)
        set(mpi integration_cf_${descriptor}_exchange_mpi2)
        add_lbm_mpi_test(${mpi} ${target} integration 2)
        set_tests_properties(${mpi} PROPERTIES TIMEOUT 180)
    endif()
endforeach()

# Stage-6-03 lifecycle and short recursive advancement use the same real owner.
foreach(descriptor IN ITEMS d2 d3)
    set(target lbm_amr_evolution_integration_${descriptor})
    add_lbm_test_target(${target}
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/AmrEvolutionIntegrationMain.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/AmrEvolutionTests.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/AmrEvolutionTestCase.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/AmrAdvanceTests.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/AmrEvolutionSnapshot.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/AmrProfilingTests.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/AmrCompositeBalance.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/AmrAdvanceEvidence.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/AmrDonorEvidence.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/StaticHierarchyTopologyFixtures.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/StaticHierarchyTopologyOracle.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/StaticHierarchy2DFixtures.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/StaticHierarchy3DFixtures.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/AmrMatrixFixtures.cpp
        ${PROJECT_SOURCE_DIR}/tests/common/AmrPushReference.cpp
        ${PROJECT_SOURCE_DIR}/tests/common/AmrHierarchyReference.cpp)
    if(descriptor STREQUAL "d3")
        target_compile_definitions(${target} PRIVATE LBM_DESCRIPTOR_D3Q27=1)
    endif()
    foreach(mode IN ITEMS lifecycle advance output)
        set(mode_timeout 180)
        if(mode STREQUAL "output")
            set(mode_timeout 300)
        endif()
        set(serial integration_cf_${descriptor}_${mode}_serial)
        add_lbm_serial_test(${serial} ${target} integration test.mode=${mode})
        set_tests_properties(${serial} PROPERTIES TIMEOUT ${mode_timeout})
        if(AMReX_MPI)
            set(mpi integration_cf_${descriptor}_${mode}_mpi2)
            add_lbm_mpi_test(${mpi} ${target} integration 2 test.mode=${mode})
            set_tests_properties(${mpi} PROPERTIES TIMEOUT ${mode_timeout})
        endif()
    endforeach()
    add_lbm_serial_test(integration_amr_profiling_${descriptor}_serial ${target} integration test.mode=profiling)
    set_tests_properties(integration_amr_profiling_${descriptor}_serial PROPERTIES TIMEOUT 300)
    if(AMReX_MPI)
        add_lbm_mpi_test(integration_amr_profiling_${descriptor}_mpi2 ${target} integration 2 test.mode=profiling)
        set_tests_properties(integration_amr_profiling_${descriptor}_mpi2 PROPERTIES TIMEOUT 300)
    endif()
    set(matrix integration_cf_${descriptor}_advance_matrix_serial)
    add_lbm_serial_test(${matrix} ${target} integration test.mode=advance test.matrix=1)
    set_tests_properties(${matrix} PROPERTIES TIMEOUT 180)
    if(AMReX_MPI)
        set(matrix integration_cf_${descriptor}_advance_matrix_mpi2)
        add_lbm_mpi_test(${matrix} ${target} integration 2 test.mode=advance test.matrix=1)
        set_tests_properties(${matrix} PROPERTIES TIMEOUT 180)
    endif()
    if(AMReX_MPI)
        if(LBM_ENABLE_CUDA)
            set(parity_backend cuda)
        else()
            set(parity_backend cpu)
        endif()
        foreach(group IN ITEMS advance advance_matrix)
            set(serial integration_cf_${descriptor}_${group}_serial)
            set(mpi integration_cf_${descriptor}_${group}_mpi2)
            set_tests_properties(${serial} PROPERTIES FIXTURES_SETUP ${serial}_data)
            set_tests_properties(${mpi} PROPERTIES FIXTURES_SETUP ${mpi}_data)
            if(group STREQUAL "advance")
                set(artifact_group advance)
            else()
                set(artifact_group matrix)
            endif()
            if(descriptor STREQUAL "d2")
                set(descriptor_name D2Q9)
            else()
                set(descriptor_name D3Q27)
            endif()
            set(parity integration_cf_${descriptor}_${group}_parity)
            add_test(NAME ${parity}
                COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_SOURCE_DIR}/runtime/check_amr_advance_parity.py
                    --left ${CMAKE_CURRENT_BINARY_DIR}/work/${serial}/artifacts/evolution-${artifact_group}
                    --right ${CMAKE_CURRENT_BINARY_DIR}/work/${mpi}/artifacts/evolution-${artifact_group}
                    --descriptor ${descriptor_name} --group ${artifact_group} --backend ${parity_backend})
            configure_lbm_test(${parity} integration serial 1 1)
            set_tests_properties(${parity} PROPERTIES FIXTURES_REQUIRED "${serial}_data;${mpi}_data" TIMEOUT 300)
        endforeach()
    endif()
    if(descriptor STREQUAL "d2")
        set(output_samples D2-E1-box32 D2-E2-box8)
    else()
        set(output_samples D3-E1-box16 D3-E2-box8)
    endif()
    set(output_modes serial)
    if(AMReX_MPI)
        list(APPEND output_modes mpi2)
    endif()
    foreach(mode IN LISTS output_modes)
        set(producer integration_cf_${descriptor}_output_${mode})
        set(consumer ${producer}_check)
        set_tests_properties(${producer} PROPERTIES FIXTURES_SETUP ${producer}_data)
        add_test(NAME ${consumer}
            COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_SOURCE_DIR}/runtime/check_amr_evolution.py
                --root ${CMAKE_CURRENT_BINARY_DIR}/work/${producer}/artifacts/evolution-output
                --samples ${output_samples} --steps 2)
        configure_lbm_test(${consumer} integration serial 1 1)
        set_tests_properties(${consumer} PROPERTIES FIXTURES_REQUIRED ${producer}_data TIMEOUT 300)
    endforeach()
endforeach()
add_test(NAME integration_cf_output_checker
    COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_SOURCE_DIR}/runtime/test_amr_evolution_checker.py)
configure_lbm_test(integration_cf_output_checker integration serial 1 1)

add_test(NAME integration_cf_advance_checker
    COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_SOURCE_DIR}/runtime/test_amr_advance_parity.py)
configure_lbm_test(integration_cf_advance_checker integration serial 1 1)

add_lbm_test_target(
    lbm_domain_integration_d2q9
    ${CMAKE_CURRENT_SOURCE_DIR}/lbm/LbmDomainIntegrationMain.cpp)
target_include_directories(
    lbm_domain_integration_d2q9 PRIVATE ${PROJECT_SOURCE_DIR}/tests)
add_lbm_test_target(
    lbm_domain_integration_d3q27
    ${CMAKE_CURRENT_SOURCE_DIR}/lbm/LbmDomainIntegrationMain.cpp)
target_include_directories(
    lbm_domain_integration_d3q27 PRIVATE ${PROJECT_SOURCE_DIR}/tests)
target_compile_definitions(
    lbm_domain_integration_d3q27 PRIVATE LBM_DESCRIPTOR_D3Q27=1)
target_compile_definitions(lbm_domain_integration_d2q9 PRIVATE AMREX_BOUND_CHECK)
target_compile_definitions(lbm_domain_integration_d3q27 PRIVATE AMREX_BOUND_CHECK)

add_lbm_test_target(
    lbm_application_setup_integration_d2q9
    ${CMAKE_CURRENT_SOURCE_DIR}/app/ApplicationSetupIntegrationMain.cpp)
target_include_directories(
    lbm_application_setup_integration_d2q9
    PRIVATE ${PROJECT_SOURCE_DIR}/tests)
add_lbm_test_target(
    lbm_application_setup_integration_d3q27
    ${CMAKE_CURRENT_SOURCE_DIR}/app/ApplicationSetupIntegrationMain.cpp)
target_include_directories(
    lbm_application_setup_integration_d3q27
    PRIVATE ${PROJECT_SOURCE_DIR}/tests)
target_compile_definitions(
    lbm_application_setup_integration_d3q27
    PRIVATE LBM_DESCRIPTOR_D3Q27=1)

add_lbm_test_target(
    lbm_application_field_integration_d2q9
    ${CMAKE_CURRENT_SOURCE_DIR}/app/ApplicationFieldIntegrationMain.cpp)
target_include_directories(
    lbm_application_field_integration_d2q9
    PRIVATE ${PROJECT_SOURCE_DIR}/tests)

add_lbm_test_target(
    lbm_application_field_integration_d3q27
    ${CMAKE_CURRENT_SOURCE_DIR}/app/ApplicationFieldIntegrationMain.cpp)
target_include_directories(
    lbm_application_field_integration_d3q27
    PRIVATE ${PROJECT_SOURCE_DIR}/tests)
target_compile_definitions(
    lbm_application_field_integration_d3q27
    PRIVATE LBM_DESCRIPTOR_D3Q27=1)

add_lbm_test_target(
    lbm_application_run_support_integration
    ${CMAKE_CURRENT_SOURCE_DIR}/app/ApplicationRunSupportIntegrationMain.cpp)
target_include_directories(
    lbm_application_run_support_integration
    PRIVATE ${PROJECT_SOURCE_DIR}/tests)

if(LBM_ENABLE_CUDA)
    add_lbm_test_target(
        lbm_gpu_device_mapping_integration
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/GpuDeviceMappingIntegrationMain.cpp)
endif()

add_lbm_serial_test(
    lbm_internal_solid_d2q9_serial
    lbm_internal_solid_integration_d2q9
    integration)
add_lbm_serial_test(
    lbm_internal_solid_d3q27_serial
    lbm_internal_solid_integration_d3q27
    integration)
add_lbm_serial_test(
    single_level_prepare_direct_d2q9
    lbm_single_level_preparation_integration_d2q9
    integration)
add_lbm_serial_test(
    single_level_prepare_direct_d3q27
    lbm_single_level_preparation_integration_d3q27
    integration)
add_lbm_serial_test(
    regression_channel_re20_nd40_setup
    lbm_single_level_preparation_integration_d2q9
    integration
    ${PROJECT_SOURCE_DIR}/tests/regression/direct/inputs/channel_re20_nd40_short.inputs
    integration.mode=channel_re20_prepare
    integration.expected_diameter_intervals=40)
set_property(
    TEST regression_channel_re20_nd40_setup
    APPEND PROPERTY LABELS short setup)
add_lbm_serial_test(
    regression_channel_re20_nd80_setup
    lbm_single_level_preparation_integration_d2q9
    integration
    ${PROJECT_SOURCE_DIR}/tests/regression/direct/inputs/channel_re20_nd80_short.inputs
    integration.mode=channel_re20_prepare
    integration.expected_diameter_intervals=80)
set_property(
    TEST regression_channel_re20_nd80_setup
    APPEND PROPERTY LABELS short setup)
add_lbm_serial_test(
    regression_channel_re100_nd80_setup
    lbm_single_level_preparation_integration_d2q9
    integration
    ${PROJECT_SOURCE_DIR}/tests/regression/direct/inputs/channel_re100_nd80_short.inputs
    integration.mode=channel_re100_prepare
    integration.expected_diameter_intervals=80)
set_property(
    TEST regression_channel_re100_nd80_setup
    APPEND PROPERTY LABELS short setup)
add_lbm_serial_test(
    single_level_runtime_d2q9
    lbm_single_level_runtime_integration_d2q9
    integration)
add_lbm_serial_test(
    single_level_runtime_d3q27
    lbm_single_level_runtime_integration_d3q27
    integration)
add_lbm_serial_test(
    fixed_thickness_d2q9_serial
    lbm_fixed_thickness_integration
    integration)
add_lbm_serial_test(
    static_hierarchy_single_level_serial
    lbm_static_hierarchy_integration
    integration
    static_test.mode=single_level)
set_tests_properties(static_hierarchy_single_level_serial PROPERTIES TIMEOUT 180)
add_lbm_serial_test(static_hierarchy_layout_serial lbm_static_hierarchy_integration
    integration static_test.mode=layout)
foreach(mode IN ITEMS finest_region per_level)
    add_lbm_serial_test(static_hierarchy_${mode}_inputs lbm_static_hierarchy_integration
        integration ${CMAKE_CURRENT_SOURCE_DIR}/runtime/inputs/static_hierarchy/${mode}.inputs)
endforeach()
set_tests_properties(static_hierarchy_layout_serial static_hierarchy_finest_region_inputs
    static_hierarchy_per_level_inputs PROPERTIES TIMEOUT 180)

add_lbm_serial_test(
    lbm_domain_d2q9_serial
    lbm_domain_integration_d2q9
    integration)
add_lbm_serial_test(
    lbm_domain_d3q27_serial
    lbm_domain_integration_d3q27
    integration)
add_lbm_serial_test(
    application_setup_d2q9_serial
    lbm_application_setup_integration_d2q9
    integration)
add_lbm_serial_test(
    application_setup_d3q27_serial
    lbm_application_setup_integration_d3q27
    integration)
add_lbm_serial_test(
    application_field_d2q9_serial
    lbm_application_field_integration_d2q9
    integration)
add_lbm_serial_test(
    application_field_d3q27_serial
    lbm_application_field_integration_d3q27
    integration)
add_lbm_serial_test(
    application_run_support_serial
    lbm_application_run_support_integration
    integration)

if(LBM_ENABLE_CUDA)
    add_lbm_serial_test(
        gpu_device_mapping
        lbm_gpu_device_mapping_integration
        integration)
    set_property(
        TEST
            lbm_internal_solid_d2q9_serial
            lbm_internal_solid_d3q27_serial
            single_level_runtime_d2q9
            single_level_runtime_d3q27
            static_hierarchy_single_level_serial
            lbm_domain_d2q9_serial
            lbm_domain_d3q27_serial
            application_run_support_serial
            gpu_device_mapping
        APPEND PROPERTY LABELS gpu-focused)
    set_property(TEST gpu_device_mapping APPEND PROPERTY LABELS mapping)
endif()

if(AMReX_OMP AND LBM_TEST_OPENMP)
    add_lbm_openmp_test(
        single_level_runtime_d2q9_openmp
        lbm_single_level_runtime_integration_d2q9
        integration
        2)
    add_lbm_openmp_test(
        single_level_runtime_d3q27_openmp
        lbm_single_level_runtime_integration_d3q27
        integration
        2)
    add_lbm_openmp_test(
        lbm_internal_solid_d2q9_openmp
        lbm_internal_solid_integration_d2q9
        integration
        2)
    add_lbm_openmp_test(
        lbm_internal_solid_d3q27_openmp
        lbm_internal_solid_integration_d3q27
        integration
        2)
    add_lbm_openmp_test(
        lbm_domain_d2q9_openmp
        lbm_domain_integration_d2q9
        integration
        2)
    add_lbm_openmp_test(
        lbm_domain_d3q27_openmp
        lbm_domain_integration_d3q27
        integration
        2)
    add_lbm_openmp_test(
        application_field_d2q9_openmp
        lbm_application_field_integration_d2q9
        integration
        2)
    add_lbm_openmp_test(
        application_field_d3q27_openmp
        lbm_application_field_integration_d3q27
        integration
        2)
endif()

if(AMReX_MPI)
    if(NOT MPIEXEC_EXECUTABLE)
        message(FATAL_ERROR "AMReX MPI integration tests require MPIEXEC_EXECUTABLE")
    endif()
    add_lbm_mpi_test(
        fixed_thickness_d2q9_mpi
        lbm_fixed_thickness_integration
        integration
        2)
    add_lbm_mpi_test(
        static_hierarchy_single_level_mpi
        lbm_static_hierarchy_integration
        integration
        2
        static_test.mode=single_level)
    set_tests_properties(static_hierarchy_single_level_mpi PROPERTIES TIMEOUT 180)
    add_lbm_mpi_test(static_hierarchy_layout_mpi lbm_static_hierarchy_integration
        integration 2 static_test.mode=layout)
    set_tests_properties(static_hierarchy_layout_mpi PROPERTIES TIMEOUT 180)
    add_lbm_mpi_test(
        single_level_runtime_d2q9_mpi
        lbm_single_level_runtime_integration_d2q9
        integration
        2)
    add_lbm_mpi_test(
        single_level_runtime_d3q27_mpi
        lbm_single_level_runtime_integration_d3q27
        integration
        2)
    add_lbm_mpi_test(
        lbm_internal_solid_d2q9_mpi
        lbm_internal_solid_integration_d2q9
        integration
        2)
    add_lbm_mpi_test(
        lbm_internal_solid_d3q27_mpi
        lbm_internal_solid_integration_d3q27
        integration
        2)
    add_lbm_mpi_test(
        lbm_domain_d2q9_mpi
        lbm_domain_integration_d2q9
        integration
        2)
    add_lbm_mpi_test(
        lbm_domain_d3q27_mpi
        lbm_domain_integration_d3q27
        integration
        2)
    add_lbm_mpi_test(
        application_field_d2q9_mpi
        lbm_application_field_integration_d2q9
        integration
        2)
    add_lbm_mpi_test(
        application_field_d3q27_mpi
        lbm_application_field_integration_d3q27
        integration
        2)
    add_lbm_mpi_test(
        application_run_support_mpi
        lbm_application_run_support_integration
        integration
        2)
endif()

# Each parity consumer depends on the actual producers in this build tree.
set(static_parity_checker ${CMAKE_CURRENT_SOURCE_DIR}/runtime/check_static_hierarchy_parity.py)
set(static_tests static_hierarchy_single_level_serial static_hierarchy_layout_serial
    static_hierarchy_finest_region_inputs static_hierarchy_per_level_inputs)
foreach(mode IN ITEMS lifecycle output)
    add_lbm_serial_test(static_hierarchy_${mode}_serial lbm_static_hierarchy_integration
        integration static_test.mode=${mode})
    list(APPEND static_tests static_hierarchy_${mode}_serial)
    if(AMReX_MPI)
        add_lbm_mpi_test(static_hierarchy_${mode}_mpi lbm_static_hierarchy_integration
            integration 2 static_test.mode=${mode})
        list(APPEND static_tests static_hierarchy_${mode}_mpi)
    endif()
endforeach()
add_test(NAME static_hierarchy_parity_checker
    COMMAND ${Python3_EXECUTABLE} -B ${static_parity_checker} --self-test)
configure_lbm_test(static_hierarchy_parity_checker integration serial 1 1)
list(APPEND static_tests static_hierarchy_parity_checker)
set_tests_properties(static_hierarchy_single_level_serial PROPERTIES FIXTURES_SETUP static_single_serial)
set_tests_properties(static_hierarchy_layout_serial PROPERTIES FIXTURES_SETUP static_layout_serial)
set(layout_serial ${CMAKE_CURRENT_BINARY_DIR}/work/static_hierarchy_layout_serial/artifacts/layout)
foreach(pair IN ITEMS g3 corner)
    if(pair STREQUAL "g3")
        set(sample G3-P)
    else()
        set(sample P-corner-P)
    endif()
    add_test(NAME static_hierarchy_${pair}_box_parity
        COMMAND ${Python3_EXECUTABLE} -B ${static_parity_checker}
            --left ${layout_serial}/${sample}/box32 --right ${layout_serial}/${sample}/box8)
    configure_lbm_test(static_hierarchy_${pair}_box_parity integration serial 1 1)
    set_tests_properties(static_hierarchy_${pair}_box_parity PROPERTIES FIXTURES_REQUIRED static_layout_serial)
    list(APPEND static_tests static_hierarchy_${pair}_box_parity)
endforeach()
if(AMReX_MPI)
    set_tests_properties(static_hierarchy_single_level_mpi PROPERTIES FIXTURES_SETUP static_single_mpi)
    set_tests_properties(static_hierarchy_layout_mpi PROPERTIES FIXTURES_SETUP static_layout_mpi)
    add_test(NAME static_hierarchy_single_level_parity
        COMMAND ${Python3_EXECUTABLE} -B ${static_parity_checker}
            --left ${CMAKE_CURRENT_BINARY_DIR}/work/static_hierarchy_single_level_serial/artifacts
            --right ${CMAKE_CURRENT_BINARY_DIR}/work/static_hierarchy_single_level_mpi/artifacts
            --samples uniform/box32/amrcore uniform/box32/single_level uniform/box8/amrcore uniform/box8/single_level
                tgv/box32/amrcore tgv/box32/single_level tgv/box8/amrcore tgv/box8/single_level)
    configure_lbm_test(static_hierarchy_single_level_parity integration serial 1 1)
    set_tests_properties(static_hierarchy_single_level_parity PROPERTIES
        FIXTURES_REQUIRED "static_single_serial;static_single_mpi")
    add_test(NAME static_hierarchy_layout_parity
        COMMAND ${Python3_EXECUTABLE} -B ${static_parity_checker}
            --left ${layout_serial}
            --right ${CMAKE_CURRENT_BINARY_DIR}/work/static_hierarchy_layout_mpi/artifacts/layout
            --samples G2-A/box32 G2-P/box32 P-corner-A/box8 P-corner-P/box8 P3-A/box8 P3-P/box8
                P-through-x-A/box32 P-through-x-P/box32
                P-through-3-x-A/box32 P-through-3-x-P/box32)
    configure_lbm_test(static_hierarchy_layout_parity integration serial 1 1)
    set_tests_properties(static_hierarchy_layout_parity PROPERTIES
        FIXTURES_REQUIRED "static_layout_serial;static_layout_mpi")
    list(APPEND static_tests static_hierarchy_single_level_mpi static_hierarchy_layout_mpi
        static_hierarchy_single_level_parity static_hierarchy_layout_parity)
endif()
set_property(TEST ${static_tests} APPEND PROPERTY LABELS static-hierarchy)
set_tests_properties(${static_tests} PROPERTIES TIMEOUT 180)

# D3 exercises the same owner, with independent three-dimensional fixtures.
foreach(mode IN ITEMS single_level layout lifecycle output)
    add_lbm_serial_test(static_hierarchy_d3q27_${mode}_serial
        lbm_static_hierarchy_integration_d3q27 integration static_test.mode=${mode})
    set_tests_properties(static_hierarchy_d3q27_${mode}_serial PROPERTIES
        TIMEOUT 360)
    set_property(TEST static_hierarchy_d3q27_${mode}_serial APPEND PROPERTY LABELS static-hierarchy)
    if(AMReX_MPI)
        add_lbm_mpi_test(static_hierarchy_d3q27_${mode}_mpi
            lbm_static_hierarchy_integration_d3q27 integration 2 static_test.mode=${mode})
        set_tests_properties(static_hierarchy_d3q27_${mode}_mpi PROPERTIES
            TIMEOUT 360)
        set_property(TEST static_hierarchy_d3q27_${mode}_mpi APPEND PROPERTY LABELS static-hierarchy)
    endif()
endforeach()

foreach(mode IN ITEMS finest_region per_level)
    add_lbm_serial_test(static_hierarchy_d3q27_${mode}_inputs lbm_static_hierarchy_integration_d3q27
        integration ${CMAKE_CURRENT_SOURCE_DIR}/runtime/inputs/static_hierarchy/${mode}_3d.inputs)
    set_tests_properties(static_hierarchy_d3q27_${mode}_inputs PROPERTIES TIMEOUT 360)
    set_property(TEST static_hierarchy_d3q27_${mode}_inputs APPEND PROPERTY LABELS static-hierarchy)
endforeach()
set_tests_properties(static_hierarchy_d3q27_single_level_serial PROPERTIES FIXTURES_SETUP static_d3_single_serial)
set_tests_properties(static_hierarchy_d3q27_layout_serial PROPERTIES FIXTURES_SETUP static_d3_layout_serial)
set(d3_layout_serial ${CMAKE_CURRENT_BINARY_DIR}/work/static_hierarchy_d3q27_layout_serial/artifacts/layout)
foreach(sample IN ITEMS D3-G3-A D3-G3-P D3-Vertex-P D3-T3-z-P)
    add_test(NAME static_hierarchy_d3q27_${sample}_box_parity
        COMMAND ${Python3_EXECUTABLE} -B ${static_parity_checker}
            --left ${d3_layout_serial}/${sample}/box16 --right ${d3_layout_serial}/${sample}/box8)
    configure_lbm_test(static_hierarchy_d3q27_${sample}_box_parity integration serial 1 1)
    set_tests_properties(static_hierarchy_d3q27_${sample}_box_parity PROPERTIES
        FIXTURES_REQUIRED static_d3_layout_serial TIMEOUT 360)
    set_property(TEST static_hierarchy_d3q27_${sample}_box_parity APPEND PROPERTY LABELS static-hierarchy)
endforeach()
if(AMReX_MPI)
    set_tests_properties(static_hierarchy_d3q27_single_level_mpi PROPERTIES FIXTURES_SETUP static_d3_single_mpi)
    set_tests_properties(static_hierarchy_d3q27_layout_mpi PROPERTIES FIXTURES_SETUP static_d3_layout_mpi)
    add_test(NAME static_hierarchy_d3q27_single_level_parity
        COMMAND ${Python3_EXECUTABLE} -B ${static_parity_checker}
            --left ${CMAKE_CURRENT_BINARY_DIR}/work/static_hierarchy_d3q27_single_level_serial/artifacts
            --right ${CMAKE_CURRENT_BINARY_DIR}/work/static_hierarchy_d3q27_single_level_mpi/artifacts
            --samples D3-S0/box16/amrcore D3-S0/box16/single_level D3-S0/box8/amrcore D3-S0/box8/single_level
                D3-S1/box16/amrcore D3-S1/box16/single_level D3-S1/box8/amrcore D3-S1/box8/single_level)
    configure_lbm_test(static_hierarchy_d3q27_single_level_parity integration serial 1 1)
    set_tests_properties(static_hierarchy_d3q27_single_level_parity PROPERTIES
        FIXTURES_REQUIRED "static_d3_single_serial;static_d3_single_mpi" TIMEOUT 360)
    set(d3_mpi_samples)
    foreach(mode IN ITEMS A P)
        list(APPEND d3_mpi_samples D3-G2-${mode}/box16 D3-G3-${mode}/box8
            D3-Vertex-${mode}/box8 D3-Wrap3-${mode}/box8)
        foreach(axis IN ITEMS x y z)
            list(APPEND d3_mpi_samples D3-T2-${axis}-${mode}/box16 D3-T3-${axis}-${mode}/box16)
        endforeach()
    endforeach()
    add_test(NAME static_hierarchy_d3q27_layout_parity
        COMMAND ${Python3_EXECUTABLE} -B ${static_parity_checker}
            --left ${d3_layout_serial}
            --right ${CMAKE_CURRENT_BINARY_DIR}/work/static_hierarchy_d3q27_layout_mpi/artifacts/layout
            --samples ${d3_mpi_samples})
    configure_lbm_test(static_hierarchy_d3q27_layout_parity integration serial 1 1)
    set_tests_properties(static_hierarchy_d3q27_layout_parity PROPERTIES
        FIXTURES_REQUIRED "static_d3_layout_serial;static_d3_layout_mpi" TIMEOUT 360)
    set_property(TEST static_hierarchy_d3q27_single_level_parity static_hierarchy_d3q27_layout_parity
        APPEND PROPERTY LABELS static-hierarchy)
endif()

# Multi-region coverage uses the same two Descriptor executables and test-owned Spec adapter.
foreach(descriptor IN ITEMS d2q9 d3q27)
    if(descriptor STREQUAL "d2q9")
        set(topology_prefix static_hierarchy)
        set(topology_target lbm_static_hierarchy_integration)
        set(input_suffix "")
        set(topology_samples D2-Topo-Gap-8/box32 D2-Topo-Gap-8/box16
            D2-Topo-Through-x/box16 D2-Topo-Feedback/box32 D2-Topo-Feedback/box16 D2-Topo-CrossBox/box8)
        set(box_sample D2-Topo-Gap-8)
        set(box_left 32)
        set(box_right 16)
    else()
        set(topology_prefix static_hierarchy_d3q27)
        set(topology_target lbm_static_hierarchy_integration_d3q27)
        set(input_suffix _3d)
        set(topology_samples D3-Topo-Vertex/box8 D3-Topo-Vertex/box16 D3-Topo-Sparse-L1-L3/box16
            D3-Topo-Feedback/box32 D3-Topo-Feedback/box16 D3-Topo-Through-z/box16 D3-Topo-CrossBox/box8)
        set(box_sample D3-Topo-Vertex)
        set(box_left 16)
        set(box_right 8)
    endif()
    add_lbm_serial_test(${topology_prefix}_topology_serial ${topology_target}
        integration static_test.mode=topology)
    add_lbm_serial_test(${topology_prefix}_topology_cost_serial ${topology_target}
        integration static_test.mode=topology_cost)
    set_tests_properties(${topology_prefix}_topology_serial PROPERTIES FIXTURES_SETUP ${topology_prefix}_topology_serial)
    set(topology_tests ${topology_prefix}_topology_serial ${topology_prefix}_topology_cost_serial)
    foreach(fixture IN ITEMS union sparse)
        add_lbm_serial_test(${topology_prefix}_topology_${fixture}_inputs ${topology_target}
            integration ${CMAKE_CURRENT_SOURCE_DIR}/runtime/inputs/static_hierarchy/topology_${fixture}${input_suffix}.inputs)
        list(APPEND topology_tests ${topology_prefix}_topology_${fixture}_inputs)
    endforeach()
    set(topology_serial ${CMAKE_CURRENT_BINARY_DIR}/work/${topology_prefix}_topology_serial/artifacts/topology)
    add_test(NAME ${topology_prefix}_topology_box_parity
        COMMAND ${Python3_EXECUTABLE} -B ${static_parity_checker}
            --left ${topology_serial}/${box_sample}/box${box_left}
            --right ${topology_serial}/${box_sample}/box${box_right})
    configure_lbm_test(${topology_prefix}_topology_box_parity integration serial 1 1)
    set_tests_properties(${topology_prefix}_topology_box_parity PROPERTIES FIXTURES_REQUIRED ${topology_prefix}_topology_serial)
    list(APPEND topology_tests ${topology_prefix}_topology_box_parity)
    if(AMReX_MPI)
        add_lbm_mpi_test(${topology_prefix}_topology_mpi ${topology_target} integration 2 static_test.mode=topology)
        set_tests_properties(${topology_prefix}_topology_mpi PROPERTIES FIXTURES_SETUP ${topology_prefix}_topology_mpi)
        add_test(NAME ${topology_prefix}_topology_parity
            COMMAND ${Python3_EXECUTABLE} -B ${static_parity_checker}
                --left ${topology_serial}
                --right ${CMAKE_CURRENT_BINARY_DIR}/work/${topology_prefix}_topology_mpi/artifacts/topology
                --samples ${topology_samples})
        configure_lbm_test(${topology_prefix}_topology_parity integration serial 1 1)
        set_tests_properties(${topology_prefix}_topology_parity PROPERTIES
            FIXTURES_REQUIRED "${topology_prefix}_topology_serial;${topology_prefix}_topology_mpi")
        list(APPEND topology_tests ${topology_prefix}_topology_mpi ${topology_prefix}_topology_parity)
    endif()
    set_tests_properties(${topology_tests} PROPERTIES TIMEOUT 900)
    set_property(TEST ${topology_tests} APPEND PROPERTY LABELS static-hierarchy)
endforeach()

# The cross-level combinations have their own narrow producers; the original
# topology suite and its output identities retain their executed scope.
foreach(descriptor IN ITEMS d2q9 d3q27)
    if(descriptor STREQUAL "d2q9")
        set(prefix static_hierarchy)
        set(target lbm_static_hierarchy_integration)
    else()
        set(prefix static_hierarchy_d3q27)
        set(target lbm_static_hierarchy_integration_d3q27)
    endif()
    set(supplement ${prefix}_topology_supplement)
    add_lbm_serial_test(${supplement}_serial ${target} integration
        static_test.mode=topology static_test.supplement_only=1)
    set_tests_properties(${supplement}_serial PROPERTIES FIXTURES_SETUP ${supplement}_serial)
    set(supplement_tests ${supplement}_serial)
    if(descriptor STREQUAL "d3q27")
        add_lbm_serial_test(${supplement}_output_serial ${target} integration
            static_test.mode=output static_test.supplement_only=1)
        list(APPEND supplement_tests ${supplement}_output_serial)
        if(AMReX_MPI)
            add_lbm_mpi_test(${supplement}_mpi ${target} integration 2
                static_test.mode=topology static_test.supplement_only=1)
            set_tests_properties(${supplement}_mpi PROPERTIES FIXTURES_SETUP ${supplement}_mpi)
            add_lbm_mpi_test(${supplement}_output_mpi ${target} integration 2
                static_test.mode=output static_test.supplement_only=1)
            set(supplement_root ${CMAKE_CURRENT_BINARY_DIR}/work)
            add_test(NAME ${supplement}_parity
                COMMAND ${Python3_EXECUTABLE} -B ${static_parity_checker}
                    --left ${supplement_root}/${supplement}_serial/artifacts/topology
                    --right ${supplement_root}/${supplement}_mpi/artifacts/topology
                    --samples D3-Topo-L3Separated-L2Overlap-AutoL1-half-x/box32
                        D3-Topo-L3Separated-L2Overlap-AutoL1-half-x/box16)
            configure_lbm_test(${supplement}_parity integration serial 1 1)
            set_tests_properties(${supplement}_parity PROPERTIES
                FIXTURES_REQUIRED "${supplement}_serial;${supplement}_mpi")
            list(APPEND supplement_tests ${supplement}_mpi ${supplement}_output_mpi ${supplement}_parity)
        endif()
    endif()
    set_tests_properties(${supplement_tests} PROPERTIES TIMEOUT 900)
    set_property(TEST ${supplement_tests} APPEND PROPERTY LABELS static-hierarchy static-hierarchy-supplement)
endforeach()
