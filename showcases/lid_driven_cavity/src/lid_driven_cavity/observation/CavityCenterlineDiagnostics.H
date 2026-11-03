#ifndef LBM_CASES_LIDDRIVENCAVITY_CAVITYCENTERLINEDIAGNOSTICS_H
#define LBM_CASES_LIDDRIVENCAVITY_CAVITYCENTERLINEDIAGNOSTICS_H

#include "lbm/fields/MacroField.H"

#include <AMReX_Array.H>
#include <AMReX_BLassert.H>
#include <AMReX_Box.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_REAL.H>

#include <vector>

namespace lbm::cases::lid_driven_cavity {

struct CavityDiagnosticGeometry {
    int onLatticeBoundaryNodes = 0;
    amrex::Array<int, 2> validNodes{0, 0};
    amrex::Array<int, 2> latticeIntervals{0, 0};
    amrex::Array<amrex::Real, 2> physicalLo{0.0, 0.0};
    amrex::Array<amrex::Real, 2> physicalHi{1.0, 1.0};
};

/**
 * @brief Collective cavity-centerline samples without filesystem policy.
 *
 * Every rank receives identical coordinate/value arrays after entering the
 * same z-average reductions in index order. File paths, serialization and the
 * barriers surrounding host I/O belong to the application output layer.
 */
struct CavityCenterlineSamples {
    int centerI = 0;
    int centerJ = 0;
    amrex::Real centerXCoordinate = 0.0;
    amrex::Real centerYCoordinate = 0.0;
    std::vector<amrex::Real> yCoordinates;
    std::vector<amrex::Real> uxLb;
    std::vector<amrex::Real> xCoordinates;
    std::vector<amrex::Real> uyLb;
};

inline int box_num_cells(amrex::Box const& box, int dir) noexcept
{
    return box.bigEnd(dir) - box.smallEnd(dir) + 1;
}

inline amrex::Real normalized_cavity_coordinate(
    amrex::Geometry const& geom,
    CavityDiagnosticGeometry const& geometry,
    amrex::Box const& domain,
    int index,
    int dir) noexcept
{
    const amrex::Real cellCenter =
        geom.ProbLo(dir) +
        (static_cast<amrex::Real>(index - domain.smallEnd(dir)) + 0.5) *
            geom.CellSize(dir);
    return (cellCenter - geometry.physicalLo[dir]) /
           (geometry.physicalHi[dir] - geometry.physicalLo[dir]);
}

inline amrex::Real sample_component_z_average(
    MacroField const& macro,
    int i,
    int j,
    int component,
    amrex::Box const& domain)
{
    amrex::MultiFab const& mf = macro.multifab();
    amrex::Real localSum = 0.0;
    amrex::Real localCount = 0.0;

    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
        amrex::Box const& bx = mfi.validbox();
        auto const m = mf.const_array(mfi);
        for (int k = domain.smallEnd(2); k <= domain.bigEnd(2); ++k) {
            amrex::IntVect const iv(AMREX_D_DECL(i, j, k));
            if (bx.contains(iv)) {
                localSum += m(i, j, k, component);
                localCount += 1.0;
            }
        }
    }

    amrex::ParallelDescriptor::ReduceRealSum(localSum);
    amrex::ParallelDescriptor::ReduceRealSum(localCount);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        localCount > 0.0,
        "Failed to sample cavity centerline value on any rank");
    return localSum / localCount;
}

inline CavityCenterlineSamples sample_cavity_centerlines(
    MacroField const& macro,
    amrex::Geometry const& geom,
    amrex::Real lidSpeedLb,
    CavityDiagnosticGeometry const& diagnosticGeometry)
{
    amrex::Box const& domain = geom.Domain();
    const int nx = box_num_cells(domain, 0);
    const int ny = box_num_cells(domain, 1);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        nx % 2 == 1 && ny % 2 == 1,
        "Cavity centerline diagnostics require odd nx and ny so x=0.5/y=0.5 are cell centers");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        lidSpeedLb > 0.0,
        "Cavity centerline diagnostics require a positive lid speed");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        diagnosticGeometry.validNodes[0] == nx &&
            diagnosticGeometry.validNodes[1] == ny &&
            diagnosticGeometry.physicalHi[0] > diagnosticGeometry.physicalLo[0] &&
            diagnosticGeometry.physicalHi[1] > diagnosticGeometry.physicalLo[1],
        "Cavity diagnostic geometry is inconsistent with the AMReX domain");

    const int centerI = domain.smallEnd(0) + nx / 2;
    const int centerJ = domain.smallEnd(1) + ny / 2;

    CavityCenterlineSamples samples;
    samples.centerI = centerI;
    samples.centerJ = centerJ;
    samples.centerXCoordinate = normalized_cavity_coordinate(
        geom, diagnosticGeometry, domain, centerI, 0);
    samples.centerYCoordinate = normalized_cavity_coordinate(
        geom, diagnosticGeometry, domain, centerJ, 1);
    samples.yCoordinates.reserve(ny);
    samples.uxLb.reserve(ny);
    for (int j = domain.smallEnd(1); j <= domain.bigEnd(1); ++j) {
        samples.yCoordinates.push_back(normalized_cavity_coordinate(
            geom, diagnosticGeometry, domain, j, 1));
        samples.uxLb.push_back(sample_component_z_average(
            macro, centerI, j, MacroField::Ux, domain));
    }
    samples.xCoordinates.reserve(nx);
    samples.uyLb.reserve(nx);
    for (int i = domain.smallEnd(0); i <= domain.bigEnd(0); ++i) {
        samples.xCoordinates.push_back(normalized_cavity_coordinate(
            geom, diagnosticGeometry, domain, i, 0));
        samples.uyLb.push_back(sample_component_z_average(
            macro, i, centerJ, MacroField::Uy, domain));
    }
    return samples;
}

} // namespace lbm::cases::lid_driven_cavity

#endif
