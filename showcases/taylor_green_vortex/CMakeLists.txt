add_library(lbm_taylor_green_vortex_case INTERFACE)
target_include_directories(
    lbm_taylor_green_vortex_case
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(
    lbm_taylor_green_vortex_case
    INTERFACE lbm_single_level_framework)

add_executable(lbm_taylor_green_vortex src/main.cpp)
target_link_libraries(
    lbm_taylor_green_vortex PRIVATE lbm_taylor_green_vortex_case)
set_target_properties(
    lbm_taylor_green_vortex
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_executable(lbm_taylor_green_vortex_d3q27 src/main.cpp)
target_link_libraries(
    lbm_taylor_green_vortex_d3q27 PRIVATE lbm_taylor_green_vortex_case)
target_compile_definitions(
    lbm_taylor_green_vortex_d3q27 PRIVATE LBM_DESCRIPTOR_D3Q27=1)
set_target_properties(
    lbm_taylor_green_vortex_d3q27
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

if(LBM_ENABLE_CUDA)
    setup_target_for_cuda_compilation(lbm_taylor_green_vortex)
    setup_target_for_cuda_compilation(lbm_taylor_green_vortex_d3q27)
endif()
