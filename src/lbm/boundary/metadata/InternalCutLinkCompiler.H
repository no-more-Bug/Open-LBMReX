#ifndef LBM_BOUNDARY_METADATA_INTERNALCUTLINKCOMPILER_H
#define LBM_BOUNDARY_METADATA_INTERNALCUTLINKCOMPILER_H

#include "lbm/boundary/metadata/CanonicalCutLink.H"
#include "lbm/boundary/metadata/CenterPhaseField.H"

#include <AMReX_Geometry.H>

#include <string>
#include <vector>

namespace lbm::boundary {

/** @brief Compile local-valid Fluid-to-Solid descriptor links once at setup. */
template <class Descriptor>
class InternalCutLinkCompiler {
public:
    bool build(
        InternalGeometryQueryRegistry const& query,
        CenterPhaseField<Descriptor> const& phase,
        amrex::Geometry const& geom,
        amrex::BoxArray const& coveredCells,
        std::string& error);

    [[nodiscard]] std::vector<CanonicalCutLinkRecord> const& records()
        const noexcept
    {
        return records_;
    }
private:
    std::vector<CanonicalCutLinkRecord> records_;
};

} // namespace lbm::boundary

#include "lbm/boundary/metadata/InternalCutLinkCompiler.HH"

#endif
