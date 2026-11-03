include_guard(GLOBAL)

include(CMakePushCheckState)
include(CheckSourceCompiles)

function(require_amrex_eb amrex_target probe_language)
    if(NOT TARGET "${amrex_target}")
        message(FATAL_ERROR
            "LBMReX requires the canonical AMReX 3D target; "
            "missing target: ${amrex_target}")
    endif()
    if(NOT probe_language STREQUAL "CXX" AND
       NOT probe_language STREQUAL "CUDA")
        message(FATAL_ERROR
            "AMReX EB package probe language must be CXX or CUDA, got: "
            "${probe_language}")
    endif()

    if(NOT DEFINED AMReX_EB_FOUND OR NOT AMReX_EB_FOUND OR
       NOT DEFINED AMReX_EB OR NOT AMReX_EB)
        message(FATAL_ERROR
            "LBMReX requires an AMReX package built with "
            "AMReX_EB=ON. Reconfigure and reinstall AMReX with embedded-"
            "boundary support, then configure this case again.")
    endif()

    unset(LBM_AMREX_EB_PACKAGE_LINKS PARENT_SCOPE)
    unset(LBM_AMREX_EB_PACKAGE_LINKS)
    unset(LBM_AMREX_EB_PACKAGE_LINKS CACHE)
    if(probe_language STREQUAL "CUDA")
        try_compile(
            LBM_AMREX_EB_PACKAGE_LINKS
            PROJECT LbmAmrexEbProbe
            SOURCE_DIR ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/amrex-eb-probe
            BINARY_DIR ${CMAKE_BINARY_DIR}/CMakeFiles/LbmAmrexEbProbe
            TARGET lbm_amrex_eb_probe
            CMAKE_FLAGS
                -DAMReX_DIR:PATH=${AMReX_DIR}
                -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
                -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
                -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
                -DCMAKE_CUDA_COMPILER:FILEPATH=${CMAKE_CUDA_COMPILER}
                -DCMAKE_CUDA_HOST_COMPILER:FILEPATH=${CMAKE_CUDA_HOST_COMPILER}
                -DMPI_C_COMPILER:FILEPATH=${MPI_C_COMPILER}
                -DMPI_CXX_COMPILER:FILEPATH=${MPI_CXX_COMPILER}
            OUTPUT_VARIABLE probe_output)
    else()
        cmake_push_check_state(RESET)
        set(CMAKE_REQUIRED_LIBRARIES "${amrex_target}")
        set(CMAKE_TRY_COMPILE_TARGET_TYPE EXECUTABLE)
        check_source_compiles(CXX [=[
            #include <AMReX_EB2.H>
            #include <AMReX_EBFabFactory.H>

            #ifndef AMREX_USE_EB
            #error "AMReX package headers do not define AMREX_USE_EB"
            #endif

            int main()
            {
                auto const* index_space = amrex::EB2::TopIndexSpaceIfPresent();
                return index_space == nullptr ? 0 : 1;
            }
        ]=] LBM_AMREX_EB_PACKAGE_LINKS)
        cmake_pop_check_state()
    endif()

    if(NOT LBM_AMREX_EB_PACKAGE_LINKS)
        message(FATAL_ERROR
            "AMReX reports EB support, but the canonical 3D target failed "
            "the EB2 header/link probe. The package may be incomplete or "
            "ABI-inconsistent; inspect the CMake configure log before continuing.\n"
            "${probe_output}")
    endif()

    message(STATUS
        "AMReX EB package gate passed (${probe_language}): "
        "${amrex_target} provides EB2 symbols")
endfunction()
