#ifndef LBM_FIELDS_INTFIELD_H
#define LBM_FIELDS_INTFIELD_H

#include "lbm/fields/FieldMeta.H"

#include <AMReX_BLassert.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_iMultiFab.H>

#include <string>
#include <utility>

namespace lbm {

class IntField {
public:
    IntField(
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

    amrex::iMultiFab& multifab() noexcept { return data_; }
    amrex::iMultiFab const& multifab() const noexcept { return data_; }

    void fillBoundary(amrex::Geometry const& geom)
    {
        data_.FillBoundary(geom.periodicity());
    }

private:
    FieldMeta meta_;
    amrex::iMultiFab data_;
};

class MaskField : public IntField {
public:
    MaskField(
        std::string name,
        amrex::BoxArray const& ba,
        amrex::DistributionMapping const& dm,
        int nGrow = 0)
        : IntField(FieldMeta(std::move(name), 1, nGrow), ba, dm)
    {}
};

} // namespace lbm

#endif
