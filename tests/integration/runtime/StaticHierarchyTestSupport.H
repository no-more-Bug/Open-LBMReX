#ifndef LBM_TESTS_STATICHIERARCHYTESTSUPPORT_H
#define LBM_TESTS_STATICHIERARCHYTESTSUPPORT_H

#include "common/HostFieldSnapshot.H"
#include "lbm/descriptors/D2Q9Descriptor.H"

#include <AMReX_BoxIterator.H>
#include <AMReX_BoxArray.H>
#include <AMReX_Geometry.H>
#include <AMReX_ParallelDescriptor.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace lbm::tests::static_hierarchy {

using Real = amrex::Real;
constexpr Real speed = 0.02;
constexpr Real dtPhys = 0.000625;

inline void require(bool condition, std::string const& message)
{
    if (!condition) amrex::Abort(message);
}

inline void close(Real actual, Real expected, std::string const& subject)
{
    const Real tolerance = std::max(Real(1.0e-12),
        Real(1.0e-12) * std::max(std::abs(actual), std::abs(expected)));
    if (!std::isfinite(actual) || !std::isfinite(expected) ||
        std::abs(actual - expected) > tolerance) {
        std::ostringstream message;
        message << std::setprecision(17) << subject << ": actual=" << actual
                << " expected=" << expected << " tolerance=" << tolerance;
        amrex::Abort(message.str());
    }
}

// Only actual valid cells are gathered. Domain-linear IDs give a stable order
// across Box decomposition without allocating q values for the entire domain.
struct ValidSnapshot {
    amrex::Box domain;
    int components = 0;
    amrex::BoxArray boxes;
    std::vector<std::size_t> cellIndices;
    std::vector<int> owners;
    std::vector<Real> values;

    std::size_t linear(amrex::IntVect const& cell) const
    {
        return (std::size_t(cell[2]-domain.smallEnd(2))*domain.length(1) +
            cell[1]-domain.smallEnd(1))*domain.length(0) + cell[0]-domain.smallEnd(0);
    }
    amrex::IntVect cell_at(std::size_t row) const
    {
        const auto id = cellIndices[row];
        const auto nx = std::size_t(domain.length(0)), ny = std::size_t(domain.length(1));
        return domain.smallEnd() + amrex::IntVect(id%nx,(id/nx)%ny,id/(nx*ny));
    }
    std::size_t offset(amrex::IntVect const& cell) const
    {
        if (!domain.contains(cell)) return owners.size();
        const auto id = linear(cell);
        const auto found = std::lower_bound(cellIndices.begin(),cellIndices.end(),id);
        return found != cellIndices.end() && *found == id
            ? std::size_t(found-cellIndices.begin()) : owners.size();
    }
};

inline ValidSnapshot snapshot(amrex::MultiFab const& source, amrex::Box const& domain)
{
    ValidSnapshot result;
    result.domain = domain;
    result.components = source.nComp();
    result.boxes = source.boxArray();
    require(result.boxes.isDisjoint(),"duplicate valid Boxes in static snapshot");
    result.cellIndices.reserve(result.boxes.numPts());
    for (int b = 0; b < result.boxes.size(); ++b) {
        require(domain.contains(result.boxes[b]),"valid Box outside snapshot domain");
        for (amrex::BoxIterator it(result.boxes[b]); it.ok(); ++it)
            result.cellIndices.push_back(result.linear(it()));
    }
    std::sort(result.cellIndices.begin(),result.cellIndices.end());
    const auto cells = result.cellIndices.size();
    result.owners.assign(cells,0);
    result.values.assign(cells*result.components,0.0);
    require(result.values.size() <= std::size_t(std::numeric_limits<int>::max()),
        "snapshot exceeds supported collective count");
    auto const host = host_field_snapshot(source);
    for (amrex::MFIter mfi(host); mfi.isValid(); ++mfi) {
        auto const data = host.const_array(mfi);
        for (amrex::BoxIterator it(mfi.validbox()); it.ok(); ++it) {
            const auto row = result.offset(it());
            ++result.owners[row];
            for (int n = 0; n < result.components; ++n)
                result.values[row*result.components+n] = data(it(),n);
        }
    }
    amrex::ParallelDescriptor::ReduceIntSum(result.owners.data(),static_cast<int>(cells));
    amrex::ParallelDescriptor::ReduceRealSum(result.values.data(),static_cast<int>(result.values.size()));
    for (int count : result.owners) require(count == 1,"missing or duplicate valid owner in static snapshot");
    return result;
}

inline void compare(ValidSnapshot const& a, ValidSnapshot const& b,
                    std::string const& subject, bool exact = false)
{
    require(a.domain == b.domain && a.components == b.components &&
        a.cellIndices == b.cellIndices && a.owners == b.owners,
        subject + ": coverage or component mismatch");
    for (std::size_t cell = 0; cell < a.owners.size(); ++cell) {
        if (a.owners[cell] == 0) continue;
        for (int n = 0; n < a.components; ++n) {
            const auto index = cell * a.components + n;
            if (exact) require(a.values[index] == b.values[index], subject + ": values changed");
            else close(a.values[index], b.values[index], subject + " cell=" +
                std::to_string(cell) + " component=" + std::to_string(n));
        }
    }
}

struct Frame {
    int step = 0;
    Real physicalTime = 0.0;
    ValidSnapshot populations;
    ValidSnapshot macro;
    int level = 0;
};

inline Frame capture(int step, Real time, amrex::Geometry const& geometry,
                     amrex::MultiFab const& populations, amrex::MultiFab const* macro)
{
    Frame result{step, time, snapshot(populations, geometry.Domain()), {}};
    if (macro) result.macro = snapshot(*macro, geometry.Domain());
    return result;
}

inline void compare(Frame const& a, Frame const& b, std::string const& subject)
{
    require(a.level == b.level && a.step == b.step, subject + ": level/step mismatch");
    close(a.physicalTime, b.physicalTime, subject + ": physical time");
    compare(a.populations, b.populations, subject + ": populations");
    compare(a.macro, b.macro, subject + ": macro");
}

// Independent tests-owned D2Q9 algebra, never production equilibrium/moments.
constexpr std::array<int, 9> cx{0, -1, -1, -1, 0, 1, 1, 1, 0};
constexpr std::array<int, 9> cy{0, 1, 0, -1, -1, -1, 0, 1, 1};
constexpr std::array<Real, 9> weights{4.0/9.0, 1.0/36.0, 1.0/9.0,
    1.0/36.0, 1.0/9.0, 1.0/36.0, 1.0/9.0, 1.0/36.0, 1.0/9.0};

inline void check_moments(Frame const& frame)
{
    auto const& f = frame.populations;
    auto const& m = frame.macro;
    require(f.components == 9 && m.components == 4 &&
        f.cellIndices == m.cellIndices && f.owners == m.owners,
        "moment snapshot shape or coverage mismatch");
    for (std::size_t cell = 0; cell < f.owners.size(); ++cell) {
        if (!f.owners[cell]) continue;
        Real rho = 0.0, jx = 0.0, jy = 0.0;
        for (int q = 0; q < 9; ++q) {
            const Real value = f.values[cell * 9 + q];
            require(std::isfinite(value), "nonfinite population");
            rho += value; jx += value * cx[q]; jy += value * cy[q];
        }
        require(std::isfinite(rho) && rho > 0.0, "nonpositive density");
        close(m.values[cell * 4], rho, "independent density moment");
        close(m.values[cell * 4 + 1], jx / rho, "independent ux moment");
        close(m.values[cell * 4 + 2], jy / rho, "independent uy moment");
        require(m.values[cell * 4 + 3] == 0.0, "inactive D2Q9 velocity changed");
    }
}

inline void write_frames(std::string const& directory, std::vector<Frame> const& frames)
{
    if (!amrex::ParallelDescriptor::IOProcessor()) return;
    std::filesystem::create_directories(directory);
    std::ofstream output(directory + "/states.csv");
    output << std::setprecision(17) << "level,step,time_phys,i,j,k";
    const int q = frames.front().populations.components;
    for (int n = 0; n < q; ++n) output << ",f" << n;
    output << ",rho,ux,uy,uz\n";
    for (auto const& frame : frames) {
        auto const& f = frame.populations;
        for (std::size_t cell = 0; cell < f.owners.size(); ++cell) {
            if (!f.owners[cell]) continue;
            const auto index = f.cell_at(cell);
            output << frame.level << ',' << frame.step << ',' << frame.physicalTime << ','
                   << index[0] << ',' << index[1] << ',' << index[2];
            for (int n = 0; n < q; ++n) output << ',' << f.values[cell * q + n];
            for (int n = 0; n < 4; ++n) output << ',' << frame.macro.values[cell * 4 + n];
            output << '\n';
        }
    }
    require(output.good(), "cannot write static hierarchy snapshots");
}

} // namespace lbm::tests::static_hierarchy

#endif
