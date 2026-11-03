#ifndef LBM_TESTS_AMR_UNIFORM_FLOW_OBSERVATION_HH
#define LBM_TESTS_AMR_UNIFORM_FLOW_OBSERVATION_HH
#include "common/HostFieldSnapshot.H"
#include <AMReX_BoxIterator.H>
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>

namespace amr_uniform_flow {
template<class D>
UniformFlowObservation<D>::UniformFlowObservation(std::string const& directory)
{
    if (amrex::ParallelDescriptor::IOProcessor()) {
        scalars_.open(directory+"/observations.csv");
        scalars_ << std::setprecision(17) << "step,time,mass,relative_l2,uz_max,rho_z_max,velocity_z_max\n";
        if (!scalars_) amrex::Abort("cannot create uniform-flow observations");
    }
}

template<class D>
void UniformFlowObservation<D>::record(lbm::amr::LbmAmrView<D> const& view)
{
    // Sum mass, error squared and measure; reduce the three z diagnostics by max.
    std::array<Real,6> values{};
    for (int lev=0; lev<int(view.levels.size()); ++lev) {
        auto const& level = view.levels[lev];
        auto const host = lbm::tests::host_field_snapshot(level.macro->multifab());
        amrex::BoxArray covered;
        if (lev+1<int(view.levels.size())) {
            covered = view.levels[lev+1].boxes;
            covered.coarsen(view.refRatios[lev]);
        }
        auto const h = level.geometry.CellSizeArray();
        Real const measure = h[0]*h[1]*(D::d==3 ? h[2] : 1);
        int const nx = level.geometry.Domain().length(0), ny = level.geometry.Domain().length(1);
        int const nz = level.geometry.Domain().length(2), planeCells = nx*ny;
        std::vector<Real> zMean;
        if constexpr (D::d==3) {
            zMean.assign(4*planeCells, 0);
            for (amrex::MFIter mfi(host, false); mfi.isValid(); ++mfi) {
                auto const a = host.const_array(mfi);
                for (amrex::BoxIterator cell(mfi.validbox()); cell.ok(); ++cell) {
                    auto const p = cell();
                    for (int c=0; c<4; ++c) zMean[c*planeCells+p[1]*nx+p[0]] += a(p,c)/nz;
                }
            }
            amrex::ParallelDescriptor::ReduceRealSum(zMean.data(), int(zMean.size()));
        }
        for (amrex::MFIter mfi(host, false); mfi.isValid(); ++mfi) {
            auto const a = host.const_array(mfi);
            for (amrex::BoxIterator cell(mfi.validbox()); cell.ok(); ++cell) {
                auto const p = cell();
                if (lev+1<int(view.levels.size()) && covered.contains(p)) continue;
                values[0] += measure*a(p,0);
                values[2] += measure;
                for (int c=1; c<4; ++c) {
                    Real const delta = a(p,c)/speed-(c==1 ? 1 : 0);
                    values[1] += measure*delta*delta;
                }
                values[3] = std::max(values[3], std::abs(a(p,3))/speed);
                if constexpr (D::d==3) {
                    int const xy = p[1]*nx+p[0];
                    values[4] = std::max(values[4], std::abs((a(p,0)-zMean[xy])/zMean[xy]));
                    Real square = 0;
                    for (int c=1; c<4; ++c) {
                        Real const delta = (a(p,c)-zMean[c*planeCells+xy])/speed;
                        square += delta*delta;
                    }
                    values[5] = std::max(values[5], std::sqrt(square));
                }
            }
        }
    }
    amrex::ParallelDescriptor::ReduceRealSum(values.data(), 3);
    amrex::ParallelDescriptor::ReduceRealMax(values.data()+3, 3);
    if (amrex::ParallelDescriptor::IOProcessor()) {
        scalars_ << view.step << ',' << view.physicalTime << ',' << values[0] << ','
                 << std::sqrt(values[1]/values[2]) << ',' << values[3] << ',' << values[4] << ',' << values[5] << '\n';
        scalars_.flush();
        if (!scalars_) amrex::Abort("cannot write uniform-flow observations");
    }
}
}
#endif
