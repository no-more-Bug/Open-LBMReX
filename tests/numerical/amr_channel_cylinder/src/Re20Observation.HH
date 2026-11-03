#ifndef LBM_TESTS_NUMERICAL_RE20OBSERVATION_HH
#define LBM_TESTS_NUMERICAL_RE20OBSERVATION_HH

#include "common/HostFieldSnapshot.H"
#include <AMReX_BoxIterator.H>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>

namespace amr_re20 {

template<class D>
Re20Observation<D>::Re20Observation(std::string const& directory, Real forceDenominator,
                                    lbm::boundary::ObjectId objectId)
    : directory_(directory), forceDenominator_(forceDenominator), objectId_(objectId)
{
    if (amrex::ParallelDescriptor::IOProcessor()) {
        scalars_.open(directory + "/observations.csv");
        forces_.open(directory + "/forces.csv");
        scalars_ << std::setprecision(std::numeric_limits<Real>::max_digits10)
                 << "step,common_step,time,comparison_begin_time,relative_l2,force_begin_time,force_end_time,Cd,Cl,rho_front,rho_rear,Cdp,phi_in,phi_out,flux_imbalance,uz_max,rho_z_max,velocity_z_max\n";
        forces_ << std::setprecision(std::numeric_limits<Real>::max_digits10)
                << "kind,id,level,population_step,coarsest_step,begin_time,end_time,sample_time,Fx_lb,Fy_lb,Fz_lb,Cd,Cl,Cz\n";
        if (!scalars_ || !forces_) amrex::Abort("cannot create Re20 observations");
    }
}

template<class D>
bool Re20Observation<D>::fluid(amrex::IntVect const& cell, amrex::Geometry const& geometry) const
{
    Real const x = geometry.ProbLo(0)+(cell[0]+Real(0.5))*geometry.CellSize(0);
    Real const y = geometry.ProbLo(1)+(cell[1]+Real(0.5))*geometry.CellSize(1);
    Real const scale = std::max({Real(0.2), Real(0.05),
        std::abs(geometry.ProbLo(0)), std::abs(geometry.ProbLo(1)), std::abs(geometry.ProbLo(2)),
        std::abs(geometry.ProbHi(0)), std::abs(geometry.ProbHi(1)), std::abs(geometry.ProbHi(2)),
        geometry.CellSize(0), geometry.CellSize(1)});
    // Match the accepted OnSurface=Fluid band, including uniform-grid surface probes.
    return std::hypot(x-Real(0.2), y-Real(0.2)) >=
           Real(0.05)-64*std::numeric_limits<Real>::epsilon()*scale;
}

template<class D>
void Re20Observation<D>::initialize(lbm::amr::LbmAmrView<D> const& view)
{
    active_.resize(view.levels.size());
    for (int lev = 0; lev < int(view.levels.size()); ++lev) {
        auto const& level = view.levels[lev];
        int const nx = level.geometry.Domain().length(0), ny = level.geometry.Domain().length(1);
        amrex::BoxArray covered;
        if (lev+1 < int(view.levels.size())) {
            covered = view.levels[lev+1].boxes;
            covered.coarsen(view.refRatios[lev]);
        }
        active_[lev].assign(nx*ny, 0);
        // All case patches span the full periodic z column; cache only this static xy mask.
        for (int j = 0; j < ny; ++j) for (int i = 0; i < nx; ++i) {
            amrex::IntVect const cell(i,j,0);
            active_[lev][j*nx+i] = level.boxes.contains(cell) &&
                !(lev+1 < int(view.levels.size()) && covered.contains(cell)) && fluid(cell, level.geometry);
        }
    }
    auto const metrics = measure(view, false);
    if (amrex::ParallelDescriptor::IOProcessor()) {
        scalars_ << view.step << ",0," << view.physicalTime << ",,,,,,";
        writeFields(metrics);
    }
    finish();
}

template<class D>
void Re20Observation<D>::save(lbm::amr::LbmAmrView<D> const& view)
{
    previous_.clear();
    for (auto const& level : view.levels)
        previous_.push_back(lbm::tests::host_field_snapshot(level.macro->multifab()));
    beginTime_ = view.physicalTime;
    beginPopulationStep_ = view.levels.back().step;
    forceIntegral_ = {};
    intervalSamples_ = 0;
    collectingInterval_ = true;
}

template<class D>
void Re20Observation<D>::writeForce(char const* kind, int id, lbm::boundary::InternalForceStep const& step,
                                    amrex::GpuArray<Real,3> const& force)
{
    if (!amrex::ParallelDescriptor::IOProcessor()) return;
    forces_ << kind << ',' << id << ',' << step.level << ',' << step.populationStep << ',' << step.coarsestStep
            << ',' << step.beginPhysicalTime << ',' << step.endPhysicalTime << ',' << step.samplePhysicalTime();
    for (auto value : force) forces_ << ',' << value;
    for (auto value : force) forces_ << ',' << value/forceDenominator_;
    forces_ << '\n';
}

template<class D>
void Re20Observation<D>::consume(lbm::boundary::InternalForceBatch const& batch)
{
    for (auto const& sample : batch.boundarySteps)
        for (auto const& force : sample.boundaryForces)
            writeForce("boundary", force.boundaryId.value, sample.step, force.forceLb);
    for (auto const& sample : batch.objectSteps) for (auto const& force : sample.objectForces) {
        writeForce("object", force.objectId.value, sample.step, force.forceLb);
        if (collectingInterval_ && force.objectId.value == objectId_.value) {
            Real const dt = sample.step.endPhysicalTime-sample.step.beginPhysicalTime;
            for (int d = 0; d < 2; ++d) forceIntegral_[d] += dt*force.forceLb[d]/forceDenominator_;
            ++intervalSamples_;
        }
    }
    // This case's sole Object has one Boundary on one level, hence native step samples.
    if (!batch.objectIntervals.empty()) amrex::Abort("unexpected cross-level Re20 Object interval");
}

template<class D>
std::vector<Real> Re20Observation<D>::planeMean(amrex::MultiFab const& host, amrex::Geometry const& geometry) const
{
    int const nx = geometry.Domain().length(0), ny = geometry.Domain().length(1);
    int const nz = geometry.Domain().length(2), planeCells = nx*ny;
    std::vector<Real> result(4*planeCells, 0);
    for (amrex::MFIter mfi(host, false); mfi.isValid(); ++mfi) {
        auto const a = host.const_array(mfi);
        for (amrex::BoxIterator cell(mfi.validbox()); cell.ok(); ++cell) {
            auto const p = cell();
            for (int c = 0; c < 4; ++c) result[c*planeCells+p[1]*nx+p[0]] += a(p,c)/nz;
        }
    }
    amrex::ParallelDescriptor::ReduceRealSum(result.data(), int(result.size()));
    return result;
}

template<class D>
typename Re20Observation<D>::FieldMetrics
Re20Observation<D>::measure(lbm::amr::LbmAmrView<D> const& view, bool compare) const
{
    // First six entries are sums; the last three are D3 maxima.
    std::array<Real,9> values{};
    for (int lev = 0; lev < int(view.levels.size()); ++lev) {
        auto const& level = view.levels[lev];
        auto const host = lbm::tests::host_field_snapshot(level.macro->multifab());
        auto const h = level.geometry.CellSizeArray(), lo = level.geometry.ProbLoArray();
        Real const volume = h[0]*h[1]*h[2];
        int const nx = level.geometry.Domain().length(0), ny = level.geometry.Domain().length(1);
        int const nz = level.geometry.Domain().length(2), planeCells = nx*ny;
        std::vector<Real> zMean;
        if constexpr (D::d == 3) zMean = planeMean(host, level.geometry);
        for (amrex::MFIter mfi(host, false); mfi.isValid(); ++mfi) {
            auto const a = host.const_array(mfi);
            auto const old = compare ? previous_[lev].const_array(mfi) : amrex::Array4<Real const>{};
            for (amrex::BoxIterator cell(mfi.validbox()); cell.ok(); ++cell) {
                auto const p = cell();
                if (!active_[lev][p[1]*nx+p[0]]) continue;
                for (int c = 1; c < 4; ++c) {
                    values[1] += volume*a(p,c)*a(p,c);
                    if (compare) {
                        Real const delta = a(p,c)-old(p,c);
                        values[0] += volume*delta*delta;
                    }
                }
                if (lev == 0 && (p[0] == 0 || p[0] == nx-1)) {
                    Real const weight = p[1] == 0 || p[1] == ny-1 ? Real(0.5) : Real(1);
                    values[p[0] == 0 ? 4 : 5] += weight*h[1]*a(p,0)*a(p,1)/(nz*meanSpeed*Real(0.41));
                }
                if constexpr (D::d == 3) {
                    int const xy = p[1]*nx+p[0];
                    values[6] = std::max(values[6], std::abs(a(p,3))/meanSpeed);
                    values[7] = std::max(values[7], std::abs(a(p,0)-zMean[xy]));
                    Real square = 0;
                    for (int c = 1; c < 4; ++c) {
                        Real const delta = (a(p,c)-zMean[c*planeCells+xy])/meanSpeed;
                        square += delta*delta;
                    }
                    values[8] = std::max(values[8], std::sqrt(square));
                }
            }
            if (lev+1 == int(view.levels.size())) {
                // Uniform surface nodes are direct samples. AMR uses the fixed one-sided
                // linear extrapolation, with y interpolation on its two actual rows.
                int const columns = view.levels.size() == 1 ? 1 : 2;
                int const rows = columns;
                for (int side = 0; side < 2; ++side) for (int col = 0; col < columns; ++col)
                    for (int row = 0; row < rows; ++row) {
                        Real const x = (side == 0 ? Real(0.15) : Real(0.25)) +
                            (columns == 1 ? 0 : (side == 0 ? -1 : 1)*(Real(0.5)+col)*h[0]);
                        Real const y = Real(0.2)+(rows == 1 ? 0 : (row-Real(0.5))*h[1]);
                        int const i = int(std::llround((x-lo[0])/h[0]-Real(0.5)));
                        int const j = int(std::llround((y-lo[1])/h[1]-Real(0.5)));
                        Real const weight = columns == 1 ? Real(1) : (col == 0 ? Real(0.75) : Real(-0.25));
                        for (int k = mfi.validbox().smallEnd(2); k <= mfi.validbox().bigEnd(2); ++k) {
                            amrex::IntVect const point(i,j,k);
                            if (mfi.validbox().contains(point)) values[2+side] += weight*a(point,0)/nz;
                        }
                    }
            }
        }
    }
    amrex::ParallelDescriptor::ReduceRealSum(values.data(), 6);
    amrex::ParallelDescriptor::ReduceRealMax(values.data()+6, 3);
    return {values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7], values[8]};
}

template<class D>
void Re20Observation<D>::writeFields(FieldMetrics const& metrics)
{
    scalars_ << ',' << metrics.rhoFront << ',' << metrics.rhoRear
             << ',' << (metrics.rhoFront-metrics.rhoRear)/(3*meanSpeed*meanSpeed)
             << ',' << metrics.phiIn << ',' << metrics.phiOut << ',' << std::abs(metrics.phiOut-metrics.phiIn);
    if constexpr (D::d == 3)
        scalars_ << ',' << metrics.uzMax << ',' << metrics.rhoZMax << ',' << metrics.velocityZMax;
    else scalars_ << ",,,";
    scalars_ << '\n';
}

template<class D>
void Re20Observation<D>::record(lbm::amr::LbmAmrView<D> const& view, int commonStep)
{
    if (intervalSamples_ != view.levels.back().step-beginPopulationStep_)
        amrex::Abort("missing Re20 Object force steps in comparison interval");
    auto const metrics = measure(view, true);
    Real const residual = metrics.speedSquared > 0 ? std::sqrt(metrics.changeSquared/metrics.speedSquared) :
        (metrics.changeSquared == 0 ? 0 : std::numeric_limits<Real>::infinity());
    if (amrex::ParallelDescriptor::IOProcessor()) {
        scalars_ << view.step << ',' << commonStep << ',' << view.physicalTime << ',' << beginTime_ << ',' << residual
                 << ',' << beginTime_ << ',' << view.physicalTime << ',' << forceIntegral_[0]/comparisonDt
                 << ',' << forceIntegral_[1]/comparisonDt;
        writeFields(metrics);
    }
    collectingInterval_ = false;
    previous_.clear();
    finish();
}

template<class D>
void Re20Observation<D>::profiles(lbm::amr::LbmAmrView<D> const& view) const
{
    struct Point { Real coordinate, ux, uy; };
    std::array<std::vector<Point>,5> lines;
    std::array<Real,5> const locations{0.05, 0.30, 0.60, 1.50, 0.20};
    std::array<char const*,5> const names{"x0.05", "x0.30", "x0.60", "x1.50", "y0.20"};
    for (int lev = 0; lev < int(view.levels.size()); ++lev) {
        auto const& level = view.levels[lev];
        auto const host = lbm::tests::host_field_snapshot(level.macro->multifab());
        auto const mean = planeMean(host, level.geometry);
        int const nx = level.geometry.Domain().length(0), ny = level.geometry.Domain().length(1);
        int const planeCells = nx*ny;
        for (int line = 0; line < 5; ++line) {
            int const across = line == 4 ? 1 : 0, along = 1-across;
            Real const index = (locations[line]-level.geometry.ProbLo(across))/level.geometry.CellSize(across)-Real(0.5);
            Real const nearest = std::round(index);
            bool const exact = std::abs(index-nearest) <= 64*std::numeric_limits<Real>::epsilon()*std::max(Real(1), std::abs(index));
            int const first = exact ? int(nearest) : int(std::floor(index)), second = exact ? first : first+1;
            Real const weight = exact ? 0 : index-first;
            for (int n = 0; n < level.geometry.Domain().length(along); ++n) {
                int const i0 = across == 0 ? first : n, j0 = across == 0 ? n : first;
                int const i1 = across == 0 ? second : n, j1 = across == 0 ? n : second;
                if (i0 < 0 || i1 >= nx || j0 < 0 || j1 >= ny) continue;
                int const xy0 = j0*nx+i0, xy1 = j1*nx+i1;
                if (!active_[lev][xy0] || !active_[lev][xy1]) continue;
                Real const coordinate = level.geometry.ProbLo(along)+(n+Real(0.5))*level.geometry.CellSize(along);
                lines[line].push_back({coordinate,
                    ((1-weight)*mean[planeCells+xy0]+weight*mean[planeCells+xy1])/meanSpeed,
                    ((1-weight)*mean[2*planeCells+xy0]+weight*mean[2*planeCells+xy1])/meanSpeed});
            }
        }
    }
    if (!amrex::ParallelDescriptor::IOProcessor()) return;
    std::ofstream profile(directory_+"/profiles.csv");
    profile << std::setprecision(std::numeric_limits<Real>::max_digits10)
            << "line,coordinate,x,y,ux_norm,uy_norm\n";
    for (int line = 0; line < 5; ++line) {
        auto& points = lines[line];
        std::sort(points.begin(), points.end(), [](auto const& a, auto const& b) { return a.coordinate < b.coordinate; });
        int const count = line == 4 ? 1441 : 329;
        Real const start = line == 4 ? Real(0.30) : 0;
        for (int n = 0; n < count; ++n) {
            Real const coordinate = start+n*Real(0.00125);
            auto upper = std::lower_bound(points.begin(), points.end(), coordinate,
                [](auto const& point, Real value) { return point.coordinate < value; });
            Real ux = 0, uy = 0;
            if (upper != points.end() && std::abs(upper->coordinate-coordinate) < Real(1.e-12)) {
                ux = upper->ux; uy = upper->uy;
            } else if (upper != points.begin() && std::abs((upper-1)->coordinate-coordinate) < Real(1.e-12)) {
                ux = (upper-1)->ux; uy = (upper-1)->uy;
            } else {
                if (upper == points.begin() || upper == points.end()) amrex::Abort("Re20 velocity line lacks valid support");
                auto const& lower = *(upper-1);
                Real const weight = (coordinate-lower.coordinate)/(upper->coordinate-lower.coordinate);
                ux = (1-weight)*lower.ux+weight*upper->ux;
                uy = (1-weight)*lower.uy+weight*upper->uy;
            }
            profile << names[line] << ',' << coordinate << ',' << (line == 4 ? coordinate : locations[line])
                    << ',' << (line == 4 ? locations[line] : coordinate) << ',' << ux << ',' << uy << '\n';
        }
    }
    profile.flush();
    if (!profile) amrex::Abort("cannot write Re20 velocity profiles");
}

template<class D>
void Re20Observation<D>::finish()
{
    if (!amrex::ParallelDescriptor::IOProcessor()) return;
    scalars_.flush(); forces_.flush();
    if (!scalars_ || !forces_) amrex::Abort("cannot write Re20 observations");
}
} // namespace amr_re20
#endif
