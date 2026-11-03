#ifndef LBM_FIELDS_REALFIELD_H
#define LBM_FIELDS_REALFIELD_H

#include "lbm/fields/FieldMeta.H"

#include <AMReX_BLassert.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#include <string>
#include <utility>

namespace lbm {

class RealField {
public:
    RealField(
        FieldMeta meta,
        amrex::BoxArray const& ba,
        amrex::DistributionMapping const& dm)
        : meta_(std::move(meta))
        , data_(ba, dm, meta_.nComp, meta_.nGrow)
    {
        meta_.validate();
        AMREX_ALWAYS_ASSERT(data_.nComp() == meta_.nComp);
        AMREX_ALWAYS_ASSERT(data_.nGrowVect() == meta_.nGrow);
    }

    std::string const& name() const noexcept { return meta_.name; }
    int nComp() const noexcept { return meta_.nComp; }
    int nGrow() const noexcept { return meta_.nGrow[0]; }
    FieldLocation location() const noexcept { return meta_.location; }

    amrex::MultiFab& multifab() noexcept { return data_; }
    amrex::MultiFab const& multifab() const noexcept { return data_; }

    void fillBoundary(amrex::Geometry const& geom)
    {
        data_.FillBoundary(geom.periodicity());
    }

protected:
    FieldMeta const& meta() const noexcept { return meta_; }

private:
    FieldMeta meta_;
    amrex::MultiFab data_;
};

class ScalarField : public RealField {
public:
    ScalarField(
        std::string name,
        amrex::BoxArray const& ba,
        amrex::DistributionMapping const& dm,
        int nGrow = 0)
        : RealField(FieldMeta(std::move(name), 1, nGrow), ba, dm)
    {}
};

class VectorField : public RealField {
public:
    VectorField(
        std::string name,
        amrex::BoxArray const& ba,
        amrex::DistributionMapping const& dm,
        int nGrow = 0)
        : RealField(FieldMeta(std::move(name), AMREX_SPACEDIM, nGrow), ba, dm)
    {}
};

} // namespace lbm

#endif
