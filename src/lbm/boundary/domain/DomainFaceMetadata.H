#ifndef LBM_BOUNDARY_DOMAIN_DOMAINFACEMETADATA_H
#define LBM_BOUNDARY_DOMAIN_DOMAINFACEMETADATA_H

#include "lbm/boundary/core/DomainBoundaryCompatibility.H"
#include "lbm/boundary/core/ResolvedBoundary.H"

#include <AMReX_Array.H>
#include <AMReX_BLassert.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include <string>
#include <vector>

namespace lbm::boundary {

/**
 * @brief Trivially copyable, device-safe metadata for all oriented domain faces.
 *
 * Arrays are indexed by domain_side_index(). Disabled slots retain harmless
 * defaults and must be guarded by enabled before use. The view contains only
 * immutable scalar/profile data; strings and other host-only resolved metadata
 * deliberately stay out of kernels.
 */
struct DomainFaceTableView {
    amrex::GpuArray<int, DomainSideCount> enabled{};
    amrex::GpuArray<int, DomainSideCount> boundaryId{};
    amrex::GpuArray<BoundaryPhysical, DomainSideCount> physical{};
    amrex::GpuArray<BoundaryScheme, DomainSideCount> scheme{};
    amrex::GpuArray<BoundaryExecutionStage, DomainSideCount> stage{};
    amrex::GpuArray<BoundaryProfileKind, DomainSideCount> profile{};
    amrex::GpuArray<amrex::GpuArray<amrex::Real, 3>, DomainSideCount>
        prescribedULb{};
    amrex::GpuArray<amrex::Real, DomainSideCount> rhoLb{};
    amrex::GpuArray<int, DomainSideCount> profileAxis{};
    amrex::GpuArray<amrex::Real, DomainSideCount> uMaxLb{};
    amrex::GpuArray<amrex::Real, DomainSideCount> profileOrigin{};
    amrex::GpuArray<amrex::Real, DomainSideCount> profileHeight{};

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    bool is_enabled(DomainSide side) const noexcept
    {
        return enabled[domain_side_index(side)] != 0;
    }
};

/**
 * @brief Host-side owner of the device-safe domain-face view.
 *
 * The table is compiled once from resolved input. PhysBC copies the complete
 * value view into its device functor; post-stream dispatch reads the view on
 * the host and copies only the selected scalar/profile state into its
 * reconstruction functor. The table owns no AMReX field or geometry.
 */
struct DomainFaceTable {
    DomainFaceTableView data;

    DomainFaceTable()
    {
        for (int side = 0; side < DomainSideCount; ++side) {
            data.enabled[side] = 0;
            data.boundaryId[side] = -1;
            data.physical[side] = BoundaryPhysical::NoSlipWall;
            data.scheme[side] = BoundaryScheme::HalfwayBounceBack;
            data.stage[side] = BoundaryExecutionStage::PreStreamingPhysicalGhost;
            data.profile[side] = BoundaryProfileKind::None;
            data.prescribedULb[side] =
                amrex::GpuArray<amrex::Real, 3>{0.0, 0.0, 0.0};
            data.rhoLb[side] = 0.0;
            data.profileAxis[side] = -1;
            data.uMaxLb[side] = 0.0;
            data.profileOrigin[side] = 0.0;
            data.profileHeight[side] = 1.0;
        }
    }

    DomainFaceTableView view() const noexcept { return data; }
};

/**
 * @brief Compile resolved domain boundaries into fixed-index runtime metadata.
 *
 * @pre Every boundary has already passed resolver compatibility checks and its
 *      execution stage matches its scheme.
 * @note Assertions here defend the resolver/runtime contract; this function
 *       does not broaden the set of executable physical or numerical schemes.
 */
inline DomainFaceTable build_domain_face_table(
    std::vector<ResolvedDomainBoundary> const& boundaries,
    amrex::Geometry const& geom)
{
    DomainFaceTable table;
    amrex::Box const& domain = geom.Domain();
    for (ResolvedDomainBoundary const& boundary : boundaries) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            is_supported_domain_boundary_combination(
                boundary.physical, boundary.scheme, boundary.profile),
            "Domain face table received an unsupported physical/scheme/profile combination");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            execution_stage_for_scheme(boundary.scheme).has_value() &&
                *execution_stage_for_scheme(boundary.scheme) == boundary.stage,
            "Resolved domain boundary stage does not match its numerical scheme");

        const int side = domain_side_index(boundary.side);
        table.data.enabled[side] = 1;
        table.data.boundaryId[side] = boundary.boundaryId.value;
        table.data.physical[side] = boundary.physical;
        table.data.scheme[side] = boundary.scheme;
        table.data.stage[side] = boundary.stage;
        table.data.profile[side] = boundary.profile;
        table.data.prescribedULb[side] = amrex::GpuArray<amrex::Real, 3>{
            boundary.prescribedULb[0],
            boundary.prescribedULb[1],
            boundary.prescribedULb[2]};
        table.data.rhoLb[side] = boundary.rhoLb;
        table.data.profileAxis[side] = boundary.profileAxis;
        table.data.uMaxLb[side] = boundary.uMaxLb;
        if (boundary.profile == BoundaryProfileKind::Poiseuille) {
            const int axis = boundary.profileAxis;
            const bool onLattice =
                boundary.scheme == BoundaryScheme::GuoExtrapolation;
            const amrex::Real dx = geom.CellSize(axis);
            table.data.profileOrigin[side] =
                geom.ProbLo(axis) +
                (static_cast<amrex::Real>(domain.smallEnd(axis)) +
                 (onLattice ? amrex::Real(0.5) : amrex::Real(0.0))) * dx;
            table.data.profileHeight[side] =
                static_cast<amrex::Real>(
                    domain.length(axis) - (onLattice ? 1 : 0)) * dx;
        }
    }
    return table;
}

/**
 * @brief Require both faces of each non-periodic Descriptor axis to be configured.
 *
 * Periodic axes need no physical entry. AMREX_SPACEDIM-only axes outside
 * Descriptor::d are intentionally ignored.
 */
template <class Descriptor>
std::vector<std::string> validate_domain_face_coverage(
    DomainFaceTableView const& table,
    amrex::Geometry const& geom)
{
    std::vector<std::string> errors;
    for (int axis = 0; axis < Descriptor::d; ++axis) {
        if (geom.isPeriodic(axis)) {
            continue;
        }

        const int loSide = 2 * axis;
        const int hiSide = loSide + 1;
        if (table.enabled[loSide] == 0) {
            errors.push_back(
                "missing boundary.domain spec for non-periodic " +
                std::string(to_string(AllDomainSides[loSide])) + " face");
        }
        if (table.enabled[hiSide] == 0) {
            errors.push_back(
                "missing boundary.domain spec for non-periodic " +
                std::string(to_string(AllDomainSides[hiSide])) + " face");
        }
    }
    return errors;
}

} // namespace lbm::boundary

#endif
