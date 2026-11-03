add_library(lbm_lid_driven_cavity_case INTERFACE)
target_include_directories(
    lbm_lid_driven_cavity_case
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(
    lbm_lid_driven_cavity_case
    INTERFACE lbm_single_level_framework)

add_executable(lbm_lid_driven_cavity src/main.cpp)
target_link_libraries(
    lbm_lid_driven_cavity PRIVATE lbm_lid_driven_cavity_case)
set_target_properties(
    lbm_lid_driven_cavity
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_executable(lbm_lid_driven_cavity_d3q27 src/main.cpp)
target_link_libraries(
    lbm_lid_driven_cavity_d3q27 PRIVATE lbm_lid_driven_cavity_case)
target_compile_definitions(
    lbm_lid_driven_cavity_d3q27 PRIVATE LBM_DESCRIPTOR_D3Q27=1)
set_target_properties(
    lbm_lid_driven_cavity_d3q27
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
