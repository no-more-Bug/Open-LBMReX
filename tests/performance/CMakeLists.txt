set(LBM_PERFORMANCE_GIT_COMMIT "${LBM_SOURCE_ID}")

set(LBM_PERFORMANCE_AMREX_GIT_VERSION "${AMReX_GIT_VERSION}")
string(REPLACE "\"" "" LBM_PERFORMANCE_AMREX_GIT_VERSION
       "${LBM_PERFORMANCE_AMREX_GIT_VERSION}")
if(NOT LBM_PERFORMANCE_AMREX_GIT_VERSION)
    set(LBM_PERFORMANCE_AMREX_GIT_VERSION "unknown")
endif()

set(LBM_PERFORMANCE_AMREX_BUILD_TYPE "${AMReX_BUILD_TYPE}")
if(NOT LBM_PERFORMANCE_AMREX_BUILD_TYPE)
    set(LBM_PERFORMANCE_AMREX_BUILD_TYPE "unknown")
endif()

get_filename_component(
    LBM_PERFORMANCE_AMREX_PREFIX "${AMReX_DIR}/../../.." ABSOLUTE)
set(LBM_PERFORMANCE_COMPILER
    "${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION}")
set(LBM_PERFORMANCE_MPI "${MPI_CXX_COMPILER}")
if(NOT LBM_PERFORMANCE_MPI)
    set(LBM_PERFORMANCE_MPI "unknown")
endif()

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/generated)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/PerformanceBuildInfo.H.in
    ${CMAKE_CURRENT_BINARY_DIR}/generated/PerformanceBuildInfo.H
    @ONLY)

function(add_lbm_performance_target target_name)
    add_lbm_test_target(
        ${target_name} ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)
    target_include_directories(
        ${target_name}
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_BINARY_DIR}/generated)
    if(ARGN)
        target_compile_definitions(${target_name} PRIVATE ${ARGN})
    endif()
endfunction()

if(LBM_ENABLE_CUDA)
    add_lbm_performance_target(
        lbm_gpu_performance_tgv_d2q9 LBM_GPU_PERFORMANCE=1)
    add_lbm_performance_target(
        lbm_gpu_performance_tgv_d3q27
        LBM_GPU_PERFORMANCE=1
        LBM_DESCRIPTOR_D3Q27=1)
    add_lbm_performance_target(
        lbm_gpu_performance_dfg_d2q9
        LBM_GPU_PERFORMANCE=1
        LBM_PERFORMANCE_DFG=1)
    add_lbm_performance_target(
        lbm_gpu_performance_dfg_d3q27
        LBM_GPU_PERFORMANCE=1
        LBM_DESCRIPTOR_D3Q27=1
        LBM_PERFORMANCE_DFG=1)

else()
    add_lbm_performance_target(lbm_performance_tgv_d2q9)
    add_lbm_performance_target(
        lbm_performance_tgv_d3q27 LBM_DESCRIPTOR_D3Q27=1)
    add_lbm_performance_target(
        lbm_performance_dfg_d2q9 LBM_PERFORMANCE_DFG=1)
    add_lbm_performance_target(
        lbm_performance_dfg_d3q27
        LBM_DESCRIPTOR_D3Q27=1
        LBM_PERFORMANCE_DFG=1)
endif()

add_test(
    NAME cpu_performance_consumers
    COMMAND ${Python3_EXECUTABLE} -B
        ${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_cpu_performance_consumers.py)
configure_lbm_test(cpu_performance_consumers performance serial 1 1)
set_property(TEST cpu_performance_consumers APPEND PROPERTY LABELS synthetic)

add_test(
    NAME gpu_performance_consumers
    COMMAND
        ${Python3_EXECUTABLE} -B
        ${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_gpu_performance_consumers.py
        ${CMAKE_CURRENT_BINARY_DIR}/work/gpu_performance_consumers/artifacts)
configure_lbm_test(
    gpu_performance_consumers performance serial 4 1)
set_property(
    TEST gpu_performance_consumers
    APPEND PROPERTY LABELS synthetic gpu-focused)

# AMR uses the same owner for refined and uniform-fine layouts. Old targets and
# CSV schemas above retain their single-level identities.
foreach(workload IN ITEMS tgv dfg)
    foreach(descriptor IN ITEMS d2q9 d3q27)
        set(target_name lbm_amr_performance_${workload}_${descriptor})
        add_lbm_test_target(${target_name}
            ${CMAKE_CURRENT_SOURCE_DIR}/src/amr_main.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/src/AmrPerformanceCsv.cpp)
        target_include_directories(${target_name} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_BINARY_DIR}/generated)
        target_compile_definitions(${target_name} PRIVATE LBM_AMR_PERFORMANCE=1)
        if(workload STREQUAL dfg)
            target_compile_definitions(${target_name} PRIVATE LBM_PERFORMANCE_DFG=1)
        endif()
        if(descriptor STREQUAL d3q27)
            target_compile_definitions(${target_name} PRIVATE LBM_DESCRIPTOR_D3Q27=1)
        endif()
    endforeach()
endforeach()

add_test(NAME amr_performance_consumers COMMAND ${Python3_EXECUTABLE} -B
    ${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_amr_performance_consumers.py)
configure_lbm_test(amr_performance_consumers performance serial 1 1)
