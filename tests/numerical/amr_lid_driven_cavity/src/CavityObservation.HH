#ifndef LBM_TESTS_NUMERICAL_CAVITYOBSERVATION_HH
#define LBM_TESTS_NUMERICAL_CAVITYOBSERVATION_HH

#include "common/HostFieldSnapshot.H"
#include <AMReX_BoxIterator.H>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>

namespace amr_cavity {

template<class D>
CavityObservation<D>::CavityObservation(std::string const& directory)
{
    if (amrex::ParallelDescriptor::IOProcessor()) {
        scalars_.open(directory + "/observations.csv");
        scalars_ << std::setprecision(std::numeric_limits<Real>::max_digits10)
                 << "step,time,comparison_begin_time,relative_l2,mass,mass_relative_change,uz_max,rho_z_max,velocity_z_max\n";
        if (!scalars_) amrex::Abort("cannot create cavity observations");
    }
}

template<class D>
Real CavityObservation<D>::initialize(lbm::amr::LbmAmrView<D> const& view)
{
    covered_.resize(view.levels.size());
    for (int lev = 0; lev+1 < int(view.levels.size()); ++lev) {
        covered_[lev] = view.levels[lev+1].boxes;
        covered_[lev].coarsen(view.refRatios[lev]);
    }
    initialMass_ = measure(view, false)[0];
    return initialMass_;
}

template<class D>
void CavityObservation<D>::save(lbm::amr::LbmAmrView<D> const& view)
{
    previous_.clear();
    for (auto const& level : view.levels)
        previous_.push_back(lbm::tests::host_field_snapshot(level.macro->multifab()));
    beginTime_ = view.physicalTime;
}

template<class D>
std::array<Real,6> CavityObservation<D>::measure(lbm::amr::LbmAmrView<D> const& view, bool compare) const
{
    std::array<Real,6> values{};
    for (int lev = 0; lev < int(view.levels.size()); ++lev) {
        auto const& level = view.levels[lev];
        auto const host = lbm::tests::host_field_snapshot(level.macro->multifab());
        auto const h = level.geometry.CellSizeArray();
        Real const volume = h[0]*h[1]*h[2];
        int const nx = level.geometry.Domain().length(0), ny = level.geometry.Domain().length(1);
        int const nz = level.geometry.Domain().length(2), planeCells = nx*ny;
        std::vector<Real> zMean;
        if constexpr (D::d == 3) {
            zMean.assign(4*planeCells, 0);
            for (amrex::MFIter mfi(host, false); mfi.isValid(); ++mfi) {
                auto const a = host.const_array(mfi);
                for (amrex::BoxIterator cell(mfi.validbox()); cell.ok(); ++cell) {
                    auto const p = cell();
                    for (int c = 0; c < 4; ++c) zMean[c*planeCells+p[1]*nx+p[0]] += a(p,c)/nz;
                }
            }
            // Each xy column is fully present or absent in this z-invariant layout.
            amrex::ParallelDescriptor::ReduceRealSum(zMean.data(), int(zMean.size()));
        }
        for (amrex::MFIter mfi(host, false); mfi.isValid(); ++mfi) {
            auto const a = host.const_array(mfi);
            auto const old = compare ? previous_[lev].const_array(mfi) : amrex::Array4<Real const>{};
            for (amrex::BoxIterator cell(mfi.validbox()); cell.ok(); ++cell) {
                auto const p = cell();
                if (lev+1 < int(view.levels.size()) && covered_[lev].contains(p)) continue;
                values[0] += volume*a(p,0);
                for (int c = 1; c < 4; ++c) {
                    values[2] += volume*a(p,c)*a(p,c);
                    if (compare) {
                        Real const delta = a(p,c)-old(p,c);
                        values[1] += volume*delta*delta;
                    }
                }
                if constexpr (D::d == 3) {
                    int const xy = p[1]*nx+p[0];
                    values[3] = std::max(values[3], std::abs(a(p,3))/lidSpeed);
                    values[4] = std::max(values[4], std::abs(a(p,0)-zMean[xy]));
                    Real square = 0;
                    for (int c = 1; c < 4; ++c) {
                        Real const delta = (a(p,c)-zMean[c*planeCells+xy])/lidSpeed;
                        square += delta*delta;
                    }
                    values[5] = std::max(values[5], std::sqrt(square));
                }
            }
        }
    }
    amrex::ParallelDescriptor::ReduceRealSum(values.data(), 3);
    amrex::ParallelDescriptor::ReduceRealMax(values.data()+3, 3);
    return values;
}

template<class D>
void CavityObservation<D>::record(lbm::amr::LbmAmrView<D> const& view)
{
    auto const v = measure(view, true);
    Real const residual = v[2] > 0 ? std::sqrt(v[1]/v[2]) : (v[1] == 0 ? 0 : std::numeric_limits<Real>::infinity());
    if (amrex::ParallelDescriptor::IOProcessor()) {
        scalars_ << view.step << ',' << view.physicalTime << ',' << beginTime_ << ',' << residual
                 << ',' << v[0] << ',' << std::abs(v[0]-initialMass_)/initialMass_
                 << ',' << v[3] << ',' << v[4] << ',' << v[5] << '\n';
        scalars_.flush();
        if (!scalars_) amrex::Abort("cannot write cavity observations");
    }
}
} // namespace amr_cavity
#endif
