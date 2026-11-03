function(add_lbm_descriptor_unit_targets target_stem source_file)
    add_lbm_test_target(
        ${target_stem}_d2q9
        ${source_file})
    target_include_directories(
        ${target_stem}_d2q9 PRIVATE ${CMAKE_SOURCE_DIR}/tests)

    add_lbm_test_target(
        ${target_stem}_d3q27
        ${source_file})
    target_include_directories(
        ${target_stem}_d3q27 PRIVATE ${CMAKE_SOURCE_DIR}/tests)
    target_compile_definitions(
        ${target_stem}_d3q27 PRIVATE LBM_DESCRIPTOR_D3Q27=1)
endfunction()

add_lbm_descriptor_unit_targets(
    lbm_unit_application
    ${CMAKE_CURRENT_SOURCE_DIR}/app/ApplicationUnitMain.cpp)
add_lbm_descriptor_unit_targets(
    lbm_unit_core
    ${CMAKE_CURRENT_SOURCE_DIR}/lbm/LbmCoreUnitMain.cpp)
add_lbm_descriptor_unit_targets(
    lbm_unit_boundary_formula
    ${CMAKE_CURRENT_SOURCE_DIR}/lbm/BoundaryFormulaUnitMain.cpp)

add_lbm_serial_test(
    unit_application_d2q9
    lbm_unit_application_d2q9
    unit)
add_lbm_serial_test(
    unit_application_d3q27
    lbm_unit_application_d3q27
    unit)
add_lbm_serial_test(
    unit_lbm_core_d2q9
    lbm_unit_core_d2q9
    unit)
add_lbm_serial_test(
    unit_lbm_core_d3q27
    lbm_unit_core_d3q27
    unit)
add_lbm_serial_test(
    unit_boundary_formula_d2q9
    lbm_unit_boundary_formula_d2q9
    unit)
add_lbm_serial_test(
    unit_boundary_formula_d3q27
    lbm_unit_boundary_formula_d3q27
    unit)

add_lbm_test_target(
    lbm_unit_public_header_link
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/PublicHeaderLinkMain.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/PublicHeaderLinkSecond.cpp)
add_lbm_serial_test(
    unit_public_header_link
    lbm_unit_public_header_link
    unit)
