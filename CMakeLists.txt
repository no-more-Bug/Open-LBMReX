cmake_minimum_required(VERSION 3.20)
project(LBMReX LANGUAGES C CXX)

option(LBM_ENABLE_CUDA "Compile selected LBM targets with CUDA" OFF)

if(LBM_ENABLE_CUDA)
    if(CMAKE_VERSION VERSION_LESS 3.26)
        message(FATAL_ERROR
            "LBM_ENABLE_CUDA requires CMake 3.26 or newer for the "
            "CUDA C++20 contract exported by AMReX")
    endif()
    enable_language(CUDA)
    find_package(AMReX REQUIRED 3D CUDA EB)
    include(AMReXTargetHelpers)
else()
    find_package(AMReX REQUIRED 3D)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(RequireAMReXEB)
if(LBM_ENABLE_CUDA)
    require_amrex_eb(AMReX::amrex_3d CUDA)
else()
    require_amrex_eb(AMReX::amrex_3d CXX)
endif()

set(LBM_INTERNAL_CORE_SOURCES
    src/lbm/profiling/AmrTimestepProfiler.cpp
    src/lbm/amr/HierarchyGeometry.cpp
    src/lbm/amr/RefinementConstraints.cpp
    src/lbm/amr/DynamicRefinement.cpp
    src/lbm/amr/DynamicGridFinalization.cpp
    src/lbm/amr/StaticRefinement.cpp
    src/lbm/amr/StaticRegionTopology.cpp
    src/lbm/boundary/domain/DomainBoundaryPolicy.cpp
    src/lbm/boundary/internal/AnalyticCylinderGeometry.cpp
    src/lbm/boundary/internal/EbGeometryBackend.cpp
    src/lbm/boundary/internal/InternalCylinderConstraints.cpp
    src/lbm/boundary/internal/InternalGeometryIntersection.cpp
    src/lbm/boundary/internal/InternalGeometryQuery.cpp
    src/lbm/boundary/metadata/InternalBoundaryMetadataDiagnostics.cpp
    src/lbm/boundary/metadata/InternalBoundaryMetadataStorage.cpp)

add_library(lbm_single_level_framework STATIC ${LBM_INTERNAL_CORE_SOURCES})
if(LBM_ENABLE_CUDA)
    setup_target_for_cuda_compilation(lbm_single_level_framework)
endif()
target_include_directories(
    lbm_single_level_framework PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(lbm_single_level_framework PUBLIC AMReX::amrex_3d)
target_compile_features(lbm_single_level_framework PUBLIC cxx_std_17)

include(CTest)
option(LBM_BUILD_SHOWCASES "Build independent scientific showcases" ON)

if(LBM_BUILD_SHOWCASES)
    add_subdirectory(showcases/lid_driven_cavity)
    add_subdirectory(showcases/taylor_green_vortex)
    add_subdirectory(showcases/uniform_flow)
    add_subdirectory(showcases/poiseuille_flow)
    add_subdirectory(showcases/channel_cylinder)
endif()

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
