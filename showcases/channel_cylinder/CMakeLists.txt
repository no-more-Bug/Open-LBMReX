add_library(
    lbm_channel_cylinder_case STATIC
    src/channel_cylinder/observation/Output.cpp)
target_include_directories(
    lbm_channel_cylinder_case
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(
    lbm_channel_cylinder_case
    PUBLIC lbm_single_level_framework)

add_executable(lbm_channel_cylinder src/main.cpp)
target_link_libraries(
    lbm_channel_cylinder PRIVATE lbm_channel_cylinder_case)
set_target_properties(
    lbm_channel_cylinder
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_executable(lbm_channel_cylinder_d3q27 src/main.cpp)
target_link_libraries(
    lbm_channel_cylinder_d3q27 PRIVATE lbm_channel_cylinder_case)
target_compile_definitions(
    lbm_channel_cylinder_d3q27 PRIVATE LBM_DESCRIPTOR_D3Q27=1)
set_target_properties(
    lbm_channel_cylinder_d3q27
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

if(LBM_ENABLE_CUDA)
    setup_target_for_cuda_compilation(lbm_channel_cylinder_case)
    setup_target_for_cuda_compilation(lbm_channel_cylinder)
    setup_target_for_cuda_compilation(lbm_channel_cylinder_d3q27)
endif()
