#ifndef LBM_CASES_TAYLORGREENVORTEX_CASEVALIDATION_H
#define LBM_CASES_TAYLORGREENVORTEX_CASEVALIDATION_H

#include "runtime/SingleLevelProgram.H"
#include "runtime/SingleLevelSetupDiagnostic.H"

#include <vector>

namespace lbm::cases::taylor_green_vortex {

template <class Descriptor>
std::vector<runtime::SingleLevelCaseDiagnostic> validate_case_mesh(
    runtime::SingleLevelMeshIntent const& mesh)
{
    std::vector<runtime::SingleLevelCaseDiagnostic> diagnostics;
    if (mesh.nCell[0] != mesh.nCell[1]) {
        diagnostics.push_back({
            "invalid_taylor_green_mesh",
            "mesh.n_cell.xy",
            "Taylor-Green requires a square x-y grid"});
    }
    for (int axis = 0; axis < AMREX_SPACEDIM; ++axis) {
        if (mesh.isPeriodic[axis] == 0) {
            diagnostics.push_back({
                "invalid_taylor_green_periodicity",
                "mesh.periodicity",
                "Taylor-Green requires periodicity in every AMReX direction"});
            break;
        }
    }
    if constexpr (Descriptor::d == 3) {
        if (mesh.nCell[2] < 2) {
            diagnostics.push_back({
                "invalid_taylor_green_quasi2d_depth",
                "mesh.n_cell.z",
                "D3Q27 Taylor-Green quasi-2D setup requires at least two z cells"});
        }
    }
    return diagnostics;
}

} // namespace lbm::cases::taylor_green_vortex

#endif
