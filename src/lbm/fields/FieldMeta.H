#ifndef LBM_FIELDS_FIELDMETA_H
#define LBM_FIELDS_FIELDMETA_H

#include <AMReX_BLassert.H>
#include <AMReX_IntVect.H>

#include <string>
#include <utility>

namespace lbm {

enum class FieldLocation {
    Cell
};

struct FieldMeta {
    std::string name;
    int nComp = 1;
    amrex::IntVect nGrow{0};
    FieldLocation location = FieldLocation::Cell;

    FieldMeta() = default;

    FieldMeta(
        std::string fieldName,
        int numComp,
        int numGrow,
        FieldLocation fieldLocation = FieldLocation::Cell)
        : FieldMeta(std::move(fieldName), numComp, amrex::IntVect(numGrow), fieldLocation)
    {}

    FieldMeta(
        std::string fieldName,
        int numComp,
        amrex::IntVect const& numGrow,
        FieldLocation fieldLocation = FieldLocation::Cell)
        : name(std::move(fieldName))
        , nComp(numComp)
        , nGrow(numGrow)
        , location(fieldLocation)
    {
        validate();
    }

    void validate() const
    {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!name.empty(), "Field name must not be empty");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(nComp > 0, "Field nComp must be positive");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(nGrow.allGE(0), "Field nGrow must be non-negative");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            location == FieldLocation::Cell,
            "Stage 2 Field layer only supports cell-centered fields");
    }
};

} // namespace lbm

#endif
