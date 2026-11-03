#ifndef LBM_FIELDS_MACROFIELD_H
#define LBM_FIELDS_MACROFIELD_H

#include "lbm/fields/RealField.H"

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>

#include <string>
#include <utility>

namespace lbm {

class MacroField : public RealField {
public:
    static constexpr int Rho = 0;
    static constexpr int Ux = 1;
    static constexpr int Uy = 2;
    static constexpr int Uz = 3;
    static constexpr int NumComp = 4;

    MacroField(
        std::string name,
        amrex::BoxArray const& ba,
        amrex::DistributionMapping const& dm,
        int nGrow = 0)
        : RealField(FieldMeta(std::move(name), NumComp, nGrow), ba, dm)
    {}
};

} // namespace lbm

#endif
