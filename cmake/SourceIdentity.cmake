# Tests already require Python; share their identity rule with shell runners.
set(LBM_EXPECTED_GIT_COMMIT "" CACHE STRING
    "Expected source HEAD, or explicit public SHA for a source archive")
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -B
        "${PROJECT_SOURCE_DIR}/tests/scripts/source_identity.py"
        --expected "${LBM_EXPECTED_GIT_COMMIT}"
    RESULT_VARIABLE lbm_identity_status
    OUTPUT_VARIABLE LBM_SOURCE_ID
    ERROR_VARIABLE lbm_identity_error
    OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT lbm_identity_status EQUAL 0)
    message(FATAL_ERROR "${lbm_identity_error}")
endif()
set(LBM_SOURCE_ENVIRONMENT "")
if(NOT LBM_SOURCE_ID STREQUAL "unknown")
    set(LBM_SOURCE_ENVIRONMENT "LBM_EXPECTED_GIT_COMMIT=${LBM_SOURCE_ID}")
endif()
