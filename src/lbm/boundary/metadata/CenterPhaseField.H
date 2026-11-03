#ifndef LBM_BOUNDARY_METADATA_CENTERPHASEFIELD_H
#define LBM_BOUNDARY_METADATA_CENTERPHASEFIELD_H

#include "lbm/boundary/internal/InternalGeometryQuery.H"
#include "lbm/boundary/metadata/CanonicalCutLink.H"
#include "lbm/fields/IntField.H"

#include <AMReX_EBFabFactory.H>
#include <AMReX_GpuQualifiers.H>

#include <string>

namespace lbm::boundary {

enum class CenterPhase : int { Fluid = 0, Solid = 1 };

AMREX_GPU_HOST_DEVICE inline constexpr int center_phase_code(
    CenterPhase phase) noexcept
{
    return static_cast<int>(phase);
}

AMREX_GPU_HOST_DEVICE inline constexpr bool is_center_phase_code(
    int value) noexcept
{
    return value == center_phase_code(CenterPhase::Fluid) ||
        value == center_phase_code(CenterPhase::Solid);
}

struct CenterPhaseBuildReport {
    amrex::Long fluidCount = 0;
    amrex::Long solidCount = 0;
    amrex::Long onSurfaceCount = 0;
};

/** Verify only the fluid, solid and actual backward-fluid points used by links. */
inline bool check_internal_links_against_eb(
    amrex::EBFArrayBoxFactory const& factory,
    std::vector<CanonicalCutLinkRecord> const& records,
    std::string& error);

/** @brief Cell phase and provenance token on the canonical BA/DM. */
template <class Descriptor>
class CenterPhaseField {
public:
    CenterPhaseField(
        amrex::BoxArray const& ba,
        amrex::DistributionMapping const& dm)
        : phase_("internal_center_phase", ba, dm, Descriptor::vicinity)
        , provenance_(
              "internal_surface_provenance", ba, dm, Descriptor::vicinity)
    {}

    bool build(
        InternalGeometryQueryRegistry const& query,
        amrex::Geometry const& geom,
        std::string& error);

    [[nodiscard]] bool built() const noexcept { return built_; }
    [[nodiscard]] MaskField const& phase() const noexcept { return phase_; }
    [[nodiscard]] MaskField const& surfaceProvenance() const noexcept
    {
        return provenance_;
    }
    [[nodiscard]] CenterPhaseBuildReport const& report() const noexcept
    {
        return report_;
    }

private:
    MaskField phase_;
    MaskField provenance_;
    CenterPhaseBuildReport report_;
    bool built_ = false;
};

} // namespace lbm::boundary

#include "lbm/boundary/metadata/CenterPhaseField.HH"

#endif
