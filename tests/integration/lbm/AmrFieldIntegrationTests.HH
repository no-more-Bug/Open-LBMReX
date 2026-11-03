#ifndef LBM_TESTS_INTEGRATION_LBM_AMRFIELDINTEGRATIONTESTS_HH
#define LBM_TESTS_INTEGRATION_LBM_AMRFIELDINTEGRATIONTESTS_HH

#include "common/HostFieldSnapshot.H"
#include "lbm/fields/IntField.H"
#include "lbm/fields/LatticeField.H"
#include "lbm/fields/RealField.H"
#include "lbm/operators/level/PullStreamingOperator.H"

#include <AMReX_BoxIterator.H>
#include <AMReX_FabArrayBase.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_RealBox.H>

#include <array>

namespace lbm::tests::integration {
namespace amr_field_detail {

constexpr amrex::Real sentinel = -1;

inline void require(bool condition, char const* message)
{
    if (!condition) amrex::Abort(message);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real marker(int i, int j, int k, int q) noexcept
{
    return amrex::Real(1 + 512*q + 64*i + 8*j + k);
}

inline amrex::IntVect wrapped(amrex::IntVect cell, amrex::Box const& domain)
{
    for (int d = 0; d < 3; ++d) {
        int const length = domain.length(d);
        int const offset = cell[d] - domain.smallEnd(d);
        cell[d] = domain.smallEnd(d) + (offset % length + length) % length;
    }
    return cell;
}

// Explicit test-owned direction order; the oracle does not call the pull kernel.
template<class Descriptor>
constexpr auto directions()
{
    if constexpr (Descriptor::d == 2) {
        return std::array<std::array<int,3>,9>{{
            {0,0,0}, {-1,1,0}, {-1,0,0}, {-1,-1,0}, {0,-1,0},
            {1,-1,0}, {1,0,0}, {1,1,0}, {0,1,0}
        }};
    } else {
        return std::array<std::array<int,3>,27>{{
            {0,0,0}, {-1,0,0}, {0,-1,0}, {0,0,-1}, {-1,-1,0}, {-1,1,0},
            {-1,0,-1}, {-1,0,1}, {0,-1,-1}, {0,-1,1}, {-1,-1,-1},
            {-1,-1,1}, {-1,1,-1}, {-1,1,1}, {1,0,0}, {0,1,0}, {0,0,1},
            {1,1,0}, {1,-1,0}, {1,0,1}, {1,0,-1}, {0,1,1}, {0,1,-1},
            {1,1,1}, {1,1,-1}, {1,-1,1}, {1,-1,-1}
        }};
    }
}

template<class Field>
void check_allocation(Field const& field, amrex::IntVect const& grow)
{
    auto const& data = field.multifab();
    require(field.nGrow() == grow[0] && data.nGrowVect() == grow,
            "C1 field ghost metadata/allocation mismatch");
    for (amrex::MFIter mfi(data, false); mfi.isValid(); ++mfi)
        require(data[mfi].box() == amrex::grow(mfi.validbox(), grow),
                "C1 field FAB extent does not match vector ghost width");
}

template<class Descriptor>
void initialize_markers(LatticeField<Descriptor>& field)
{
    auto& data = field.multifab();
    data.setVal(sentinel);
    for (amrex::MFIter mfi(data, false); mfi.isValid(); ++mfi) {
        auto const values = data.array(mfi);
        amrex::ParallelFor(mfi.validbox(), Descriptor::q,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int q) {
                values(i,j,k,q) = marker(i,j,k,q);
            });
    }
}

template<class Descriptor>
void check_fill(LatticeField<Descriptor> const& field, amrex::Box const& domain,
                amrex::IntVect const& fillGrow)
{
    auto host = host_field_snapshot(field.multifab());
    for (amrex::MFIter mfi(host, false); mfi.isValid(); ++mfi) {
        auto const values = host.const_array(mfi);
        auto const fillBox = amrex::grow(mfi.validbox(), fillGrow);
        for (amrex::BoxIterator it(host[mfi].box()); it.ok(); ++it) {
            auto const cell = it();
            bool const filled = fillBox.contains(cell);
            auto const donor = wrapped(cell, domain);
            for (int q = 0; q < Descriptor::q; ++q)
                require(values(cell,q) == (filled ? marker(donor[0],donor[1],donor[2],q) : sentinel),
                        "C1 fillBoundary changed an excluded cell or missed a periodic donor");
        }
    }
}

inline void check_source_unchanged(amrex::MultiFab const& source,
                                   amrex::MultiFab const& before)
{
    auto after = host_field_snapshot(source);
    for (amrex::MFIter mfi(after, false); mfi.isValid(); ++mfi) {
        auto const a = after.const_array(mfi);
        auto const b = before.const_array(mfi);
        for (amrex::BoxIterator it(after[mfi].box()); it.ok(); ++it)
            for (int q = 0; q < after.nComp(); ++q)
                require(a(it(),q) == b(it(),q), "C1 pull modified source populations");
    }
}

template<class Descriptor>
void check_pull(LatticeField<Descriptor> const& destination,
                amrex::Box const& domain, amrex::IntVect const& writeGrow)
{
    auto host = host_field_snapshot(destination.multifab());
    constexpr auto velocity = directions<Descriptor>();
    for (amrex::MFIter mfi(host, false); mfi.isValid(); ++mfi) {
        auto const values = host.const_array(mfi);
        auto const writeBox = amrex::grow(mfi.validbox(), writeGrow);
        for (amrex::BoxIterator it(host[mfi].box()); it.ok(); ++it) {
            auto const cell = it();
            for (int q = 0; q < Descriptor::q; ++q) {
                auto const donor = wrapped(cell - amrex::IntVect(
                    velocity[q][0], velocity[q][1], velocity[q][2]), domain);
                auto const expected = writeBox.contains(cell)
                    ? marker(donor[0],donor[1],donor[2],q) : sentinel;
                require(values(cell,q) == expected,
                        "C1 pull value or preserved outer ghost mismatch");
            }
        }
    }
}

} // namespace amr_field_detail

template<class Descriptor>
void run_amr_field_integration_tests()
{
    using namespace amr_field_detail;
    using Field = LatticeField<Descriptor>;
    amrex::Box const domain(amrex::IntVect(0),
        amrex::IntVect(7,7,Descriptor::d == 3 ? 7 : 0));
    amrex::RealBox physical({0.,0.,0.}, {1.,1.,Descriptor::d == 3 ? 1. : 0.125});
    std::array<int,3> const periodic{1,1,1};
    amrex::Geometry geometry(domain, &physical, amrex::CoordSys::cartesian, periodic.data());
    amrex::BoxArray boxes(domain);
    boxes.maxSize(4);
    int const rankCount = amrex::ParallelDescriptor::NProcs();
    amrex::Vector<int> owners(boxes.size());
    for (int b = 0; b < boxes.size(); ++b) {
        auto const box = boxes[b];
        auto const low = box.smallEnd();
        owners[b] = (low[0]/4 + low[1]/4 + low[2]/4) % rankCount;
    }
    amrex::DistributionMapping mapping(owners);
    amrex::IntVect const v(1,1,Descriptor::d == 3 ? 1 : 0);
    amrex::IntVect const allocation = v * 2;

    Field legacyDefault(boxes, mapping);
    Field legacyNamed("legacy_named", boxes, mapping);
    Field legacyScalar(boxes, mapping, 2);
    Field legacyNamedScalar("legacy_scalar", boxes, mapping, 2);
    check_allocation(legacyDefault, amrex::IntVect(Descriptor::vicinity));
    check_allocation(legacyNamed, amrex::IntVect(Descriptor::vicinity));
    check_allocation(legacyScalar, amrex::IntVect(2));
    check_allocation(legacyNamedScalar, amrex::IntVect(2));
    initialize_markers(legacyDefault);
    legacyDefault.fillBoundary(geometry);
    check_fill(legacyDefault, domain, legacyDefault.nGrowVect());
    initialize_markers(legacyNamed);
    legacyNamed.fillBoundary(geometry, false);
    check_fill(legacyNamed, domain, legacyNamed.nGrowVect());

    amrex::IntVect const otherGrow(2,1,0);
    RealField real(FieldMeta("vector_real", 2, otherGrow), boxes, mapping);
    IntField integer(FieldMeta("vector_integer", 3, otherGrow), boxes, mapping);
    check_allocation(real, otherGrow);
    check_allocation(integer, otherGrow);

    Field source(boxes, mapping, allocation);
    Field destination("amr_field_destination", boxes, mapping, allocation);
    check_allocation(source, allocation);
    check_allocation(destination, allocation);
    require(source.nGrowVect() == allocation && destination.nGrowVect() == allocation,
            "C1 LatticeField nGrowVect mismatch");
    for (auto const& fillGrow : {amrex::IntVect(0), amrex::IntVect(1,0,0), allocation}) {
        initialize_markers(source);
        source.fillBoundary(geometry, fillGrow);
        check_fill(source, domain, fillGrow);
    }
    initialize_markers(source);
    source.fillBoundary(geometry);
    check_fill(source, domain, allocation);

    int remote = 0, remotePeriodicZ = 0;
    for (amrex::MFIter mfi(source.multifab(), false); mfi.isValid(); ++mfi) {
        for (amrex::BoxIterator it(source.multifab()[mfi].box()); it.ok(); ++it) {
            auto const cell = it();
            if (mfi.validbox().contains(cell)) continue;
            auto const donor = wrapped(cell, domain);
            for (int b = 0; b < boxes.size(); ++b) {
                if (boxes[b].contains(donor) && owners[b] != amrex::ParallelDescriptor::MyProc()) {
                    ++remote;
                    if (donor[2] != cell[2]) ++remotePeriodicZ;
                }
            }
        }
    }
    amrex::ParallelDescriptor::ReduceIntSum(remote);
    amrex::ParallelDescriptor::ReduceIntSum(remotePeriodicZ);
    if (rankCount > 1) {
        require(remote > 0, "C1 fixture missed remote ghost donors");
        if constexpr (Descriptor::d == 3)
            require(remotePeriodicZ > 0, "C1 D3 fixture missed remote z-periodic donors");
    }

    auto const oldTileSize = amrex::FabArrayBase::mfiter_tile_size;
    bool const cpuTiles = amrex::Gpu::notInLaunchRegion();
    if (cpuTiles) amrex::FabArrayBase::mfiter_tile_size = amrex::IntVect(2);
    amrex::Vector<int> tiles(boxes.size(), 0);
    for (amrex::MFIter mfi(destination.multifab(), amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
        ++tiles[mfi.index()];
    int multiTileFabs = 0;
    for (int count : tiles) multiTileFabs += count > 1;
    amrex::ParallelDescriptor::ReduceIntSum(multiTileFabs);
    if (cpuTiles) require(multiTileFabs > 0, "C1 CPU fixture missed multiple tiles in one FAB");

    auto before = host_field_snapshot(source.multifab());
    operators::PullStreamingOperator<Descriptor> pull;
    for (auto const& writeGrow : {amrex::IntVect(0), v}) {
        destination.multifab().setVal(sentinel);
        if (writeGrow == amrex::IntVect(0)) pull.apply(source, destination);
        else pull.apply(source, destination, writeGrow);
        check_pull(destination, domain, writeGrow);
        check_source_unchanged(source.multifab(), before);
    }
    if (cpuTiles) amrex::FabArrayBase::mfiter_tile_size = oldTileSize;
    amrex::Print() << Descriptor::name() << " C1 vector fields/pull passed: ranks=" << rankCount
                   << " remote_donors=" << remote << " remote_periodic_z=" << remotePeriodicZ
                   << " multi_tile_fabs=" << multiTileFabs << '\n';
}

} // namespace lbm::tests::integration

#endif
