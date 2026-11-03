# Formal tests are composed here so production targets never depend on the
# repository testing structure.
find_package(Python3 3.9 REQUIRED COMPONENTS Interpreter)
include(${PROJECT_SOURCE_DIR}/cmake/SourceIdentity.cmake)

option(LBM_TEST_OPENMP "Register OpenMP and hybrid test layouts" ON)
set(LBM_TEST_MPI_LAUNCHER "" CACHE FILEPATH "Optional tests-owned MPI launcher")
set(LBM_TEST_CPUS_PER_RANK 1 CACHE STRING "Host CPUs reserved per test MPI rank")
set(LBM_TEST_R5_MPI_RANKS 0 CACHE STRING "Fixed pure-MPI R5 ranks; 0 keeps local layouts")
set(LBM_TEST_AMR_MPI_RANKS 4 CACHE STRING "Pure-MPI ranks for the AMR numerical groups")

if(LBM_ENABLE_CUDA)
    set(LBM_TEST_BACKEND_LABEL gpu)
else()
    set(LBM_TEST_BACKEND_LABEL cpu)
endif()

function(add_lbm_test_target target_name source_file)
    add_executable(${target_name} ${source_file} ${ARGN})
    if(LBM_ENABLE_CUDA)
        setup_target_for_cuda_compilation(${target_name})
    endif()
    target_link_libraries(${target_name} PRIVATE lbm_single_level_framework)
    target_include_directories(${target_name} PRIVATE ${PROJECT_SOURCE_DIR}/tests)
endfunction()

function(lbm_test_command output target_name ranks)
    if(LBM_TEST_MPI_LAUNCHER)
        set(command ${LBM_TEST_MPI_LAUNCHER} -np ${ranks})
    elseif(ranks GREATER 1)
        set(command ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
            --map-by core --bind-to core --rank-by slot --report-bindings
            ${MPIEXEC_PREFLAGS})
    else()
        set(command)
    endif()
    list(APPEND command $<TARGET_FILE:${target_name}> ${MPIEXEC_POSTFLAGS} ${ARGN})
    if(LBM_ENABLE_CUDA)
        list(APPEND command amrex.use_gpu_aware_mpi=0)
    endif()
    set(${output} ${command} PARENT_SCOPE)
endfunction()

function(configure_lbm_test test_name layer parallel_kind processors omp_threads)
    set(test_work_dir "${CMAKE_CURRENT_BINARY_DIR}/work/${test_name}")
    set(test_tmp_dir "${test_work_dir}/tmp")
    file(MAKE_DIRECTORY "${test_tmp_dir}")
    if(LBM_ENABLE_CUDA AND NOT layer STREQUAL "performance")
        math(EXPR processors "${processors} * ${LBM_TEST_CPUS_PER_RANK}")
    endif()

    set_tests_properties(
        ${test_name}
        PROPERTIES
            ENVIRONMENT
                "${LBM_SOURCE_ENVIRONMENT};OMP_NUM_THREADS=${omp_threads};OMP_DYNAMIC=FALSE;OMP_PROC_BIND=close;OMP_PLACES=cores;TMPDIR=${test_tmp_dir};TEMP=${test_tmp_dir};TMP=${test_tmp_dir}"
            LABELS "${LBM_TEST_BACKEND_LABEL};fast;${layer};${parallel_kind}"
            PROCESSORS ${processors}
            TIMEOUT 60
            WORKING_DIRECTORY "${test_work_dir}")
endfunction()

function(add_lbm_serial_test test_name target_name layer)
    lbm_test_command(command ${target_name} 1 ${ARGN})
    add_test(
        NAME ${test_name}
        COMMAND ${command})
    configure_lbm_test(${test_name} ${layer} serial 1 1)
endfunction()

function(add_lbm_openmp_test test_name target_name layer omp_threads)
    add_test(
        NAME ${test_name}
        COMMAND $<TARGET_FILE:${target_name}> ${ARGN})
    configure_lbm_test(
        ${test_name} ${layer} openmp ${omp_threads} ${omp_threads})
endfunction()

function(add_lbm_mpi_test test_name target_name layer mpi_ranks)
    lbm_test_command(command ${target_name} ${mpi_ranks} ${ARGN})
    add_test(
        NAME ${test_name}
        COMMAND ${command})
    configure_lbm_test(${test_name} ${layer} mpi ${mpi_ranks} 1)
endfunction()

add_test(
    NAME source_identity
    COMMAND ${Python3_EXECUTABLE} -B
        ${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_source_identity.py)
configure_lbm_test(source_identity integration serial 1 1)
set_property(TEST source_identity APPEND PROPERTY LABELS synthetic)

add_subdirectory(unit)
add_subdirectory(integration)
add_subdirectory(regression)
add_subdirectory(numerical)
add_subdirectory(performance EXCLUDE_FROM_ALL)
