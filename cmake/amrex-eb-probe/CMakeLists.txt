cmake_minimum_required(VERSION 3.26)
project(LbmAmrexEbProbe LANGUAGES C CXX CUDA)

find_package(AMReX REQUIRED 3D CUDA EB)
include(AMReXTargetHelpers)

add_executable(lbm_amrex_eb_probe main.cpp)
setup_target_for_cuda_compilation(lbm_amrex_eb_probe)
target_link_libraries(lbm_amrex_eb_probe PRIVATE AMReX::amrex_3d)
