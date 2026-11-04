add_library(lbm_poiseuille_flow_case INTERFACE)
target_include_directories(
    lbm_poiseuille_flow_case
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(
    lbm_poiseuille_flow_case
    INTERFACE lbm_single_level_framework)

add_executable(lbm_poiseuille_flow src/main.cpp)
target_link_libraries(lbm_poiseuille_flow PRIVATE lbm_poiseuille_flow_case)
set_target_properties(
    lbm_poiseuille_flow
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_executable(lbm_poiseuille_flow_d3q27 src/main.cpp)
target_link_libraries(
    lbm_poiseuille_flow_d3q27 PRIVATE lbm_poiseuille_flow_case)
target_compile_definitions(
    lbm_poiseuille_flow_d3q27 PRIVATE LBM_DESCRIPTOR_D3Q27=1)
set_target_properties(
    lbm_poiseuille_flow_d3q27
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

if(LBM_ENABLE_CUDA)
    setup_target_for_cuda_compilation(lbm_poiseuille_flow)
    setup_target_for_cuda_compilation(lbm_poiseuille_flow_d3q27)
endif()
