#ifndef LBM_FIELDS_LATTICEFIELD_H
#define LBM_FIELDS_LATTICEFIELD_H

#include "lbm/fields/FieldMeta.H"
#include "lbm/descriptors/StencilExtent.H"

#include <AMReX_Array4.H>
#include <AMReX_BLassert.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

#include <string>
#include <utility>

namespace lbm {

template <class Descriptor>
class LatticeField {
public:
    using DescriptorType = Descriptor;

    LatticeField(
        amrex::BoxArray const& ba,
        amrex::DistributionMapping const& dm,
        int nGrow = Descriptor::vicinity)
        : LatticeField("populations", ba, dm, nGrow)
    {}

    LatticeField(
        std::string name,
        amrex::BoxArray const& ba,
        amrex::DistributionMapping const& dm,
        int nGrow = Descriptor::vicinity)
        : LatticeField(std::move(name), ba, dm, amrex::IntVect(nGrow))
    {}

    LatticeField(
        amrex::BoxArray const& ba,
        amrex::DistributionMapping const& dm,
        amrex::IntVect const& nGrow)
        : LatticeField("populations", ba, dm, nGrow)
    {}

    LatticeField(
        std::string name,
        amrex::BoxArray const& ba,
        amrex::DistributionMapping const& dm,
        amrex::IntVect const& nGrow)
        : meta_(std::move(name), Descriptor::q, nGrow)
        , data_(ba, dm, Descriptor::q, nGrow)
    {
        AMREX_ALWAYS_ASSERT(data_.nComp() == Descriptor::q);
        AMREX_ALWAYS_ASSERT(nGrow.allGE(stencil_extent<Descriptor>()));

        // Pull streaming writes every valid destination.  For post-stream
        // full-q boundaries, reads from non-periodic physical ghosts are
        // disposable but must still be defined and finite until overwritten.
        data_.setVal(amrex::Real(0.0));
    }

    std::string const& name() const noexcept { return meta_.name; }
    char const* descriptorName() const noexcept { return Descriptor::name(); }

    amrex::MultiFab& multifab() noexcept { return data_; }
    amrex::MultiFab const& multifab() const noexcept { return data_; }

    int nComp() const noexcept { return meta_.nComp; }
    // Scalar compatibility accessor: x extent, as in AMReX.
    int nGrow() const noexcept { return meta_.nGrow[0]; }
    amrex::IntVect const& nGrowVect() const noexcept { return meta_.nGrow; }

    void fillBoundary(amrex::Geometry const& geom, bool cross = false)
    {
        data_.FillBoundary(geom.periodicity(), cross);
    }

    void fillBoundary(amrex::Geometry const& geom, amrex::IntVect const& fillGrow,
                      bool cross = false)
    {
        AMREX_ALWAYS_ASSERT(fillGrow.allGE(0));
        // AMReX checks fillGrow against the allocated width.
        data_.FillBoundary(fillGrow, geom.periodicity(), cross);
    }

private:
    FieldMeta meta_;
    amrex::MultiFab data_;
};

struct PopulationCell {
    amrex::Array4<amrex::Real> f;
    int i;
    int j;
    int k;

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real& operator[](int iPop) const noexcept
    {
        return f(i, j, k, iPop);
    }
};

struct ConstPopulationCell {
    amrex::Array4<amrex::Real const> f;
    int i;
    int j;
    int k;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real operator[](int iPop) const noexcept
    {
        return f(i, j, k, iPop);
    }
};

} // namespace lbm

#endif
