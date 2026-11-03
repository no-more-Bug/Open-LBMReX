#ifndef LBM_BOUNDARY_DOMAIN_INITIALIZATION_GUOBOUNDARYINITIALIZATIONDISPATCH_H
#define LBM_BOUNDARY_DOMAIN_INITIALIZATION_GUOBOUNDARYINITIALIZATIONDISPATCH_H

#include "lbm/boundary/domain/DomainBoundaryPrescription.H"
#include "lbm/boundary/domain/DomainFaceMetadata.H"
#include "lbm/boundary/domain/LbmPostStreamingBoundary.H"

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

namespace lbm::boundary::initialization {

/**
 * @brief Complete macroscopic state used to initialize one valid boundary node.
 *
 * A state resolver receives the first strictly interior donor state, the
 * physical center of the valid boundary node, and the current
 * boundary-evaluation time. It replaces only the components prescribed by the
 * physical boundary and returns the complete state whose equilibrium
 * populations are written by the level Operator.
 */
struct ValidBoundaryMacroscopicState {
    amrex::Real rho = 0.0;
    amrex::Real ux = 0.0;
    amrex::Real uy = 0.0;
    amrex::Real uz = 0.0;
};

namespace detail {

struct NoSlipStateResolver {
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    ValidBoundaryMacroscopicState operator()(
        ValidBoundaryMacroscopicState const& donor,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const&,
        amrex::Real) const noexcept
    {
        return ValidBoundaryMacroscopicState{donor.rho, 0.0, 0.0, 0.0};
    }
};

template <class VelocityProfile>
struct VelocityStateResolver {
    VelocityProfile profile;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    ValidBoundaryMacroscopicState operator()(
        ValidBoundaryMacroscopicState const& donor,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& position,
        amrex::Real physicalTime) const noexcept
    {
        const amrex::GpuArray<amrex::Real, 3> velocity =
            profile(position, physicalTime);
        return ValidBoundaryMacroscopicState{
            donor.rho, velocity[0], velocity[1], velocity[2]};
    }
};

struct PressureStateResolver {
    ConstantDensityProfile profile;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    ValidBoundaryMacroscopicState operator()(
        ValidBoundaryMacroscopicState const& donor,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& position,
        amrex::Real physicalTime) const noexcept
    {
        const amrex::Real rho = profile(position, physicalTime);
        return ValidBoundaryMacroscopicState{
            rho, donor.ux, donor.uy, donor.uz};
    }
};

template <class VelocityProfile>
struct PrescribedStateResolver {
    ConstantDensityProfile densityProfile;
    VelocityProfile velocityProfile;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    ValidBoundaryMacroscopicState operator()(
        ValidBoundaryMacroscopicState const&,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& position,
        amrex::Real physicalTime) const noexcept
    {
        const amrex::Real rho = densityProfile(position, physicalTime);
        const amrex::GpuArray<amrex::Real, 3> velocity =
            velocityProfile(position, physicalTime);
        return ValidBoundaryMacroscopicState{
            rho, velocity[0], velocity[1], velocity[2]};
    }
};

struct FreeOutflowStateResolver {
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    ValidBoundaryMacroscopicState operator()(
        ValidBoundaryMacroscopicState const& donor,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const&,
        amrex::Real) const noexcept
    {
        return donor;
    }
};

AMREX_FORCE_INLINE
NoSlipStateResolver make_state_resolver(NoSlipPrescription) noexcept
{
    return {};
}

template <class VelocityProfile>
AMREX_FORCE_INLINE
VelocityStateResolver<VelocityProfile> make_state_resolver(
    VelocityPrescription<VelocityProfile> const& prescription) noexcept
{
    return VelocityStateResolver<VelocityProfile>{
        prescription.velocityProfile};
}

AMREX_FORCE_INLINE
PressureStateResolver make_state_resolver(
    PressurePrescription const& prescription) noexcept
{
    return PressureStateResolver{prescription.densityProfile};
}

template <class VelocityProfile>
AMREX_FORCE_INLINE
PrescribedStateResolver<VelocityProfile> make_state_resolver(
    PrescribedStatePrescription<VelocityProfile> const& prescription) noexcept
{
    return PrescribedStateResolver<VelocityProfile>{
        prescription.densityProfile,
        prescription.velocityProfile};
}

AMREX_FORCE_INLINE
FreeOutflowStateResolver make_state_resolver(
    FreeOutflowPrescription) noexcept
{
    return {};
}

} // namespace detail

/**
 * @brief Select the Guo initialization state rule for one compiled region.
 *
 * The shared domain-boundary prescription dispatcher owns the physical/profile
 * matrix. This adapter maps its typed prescription to a valid-node equilibrium
 * state resolver; it does not launch kernels or apply the runtime Guo
 * non-equilibrium extrapolation formula. The selected resolver is invoked
 * later with the boundary-node position, initial physical time, and inward donor's
 * macroscopic state.
 */
class GuoBoundaryInitializationDispatch {
public:
    template <class RegionLauncher>
    static void dispatch(
        DomainFaceTableView const& faces,
        PostStreamBoundaryRegion const& region,
        int sideIndex,
        RegionLauncher const& launchRegion)
    {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            faces.scheme[sideIndex] == BoundaryScheme::GuoExtrapolation,
            "Guo boundary initialization dispatch requires Guo face metadata");

        dispatch_domain_boundary_prescription(
            faces,
            region.effectiveSide,
            region.effectivePhysical,
            region.effectiveProfile,
            sideIndex,
            [&launchRegion](auto const& prescription) {
                launchRegion(detail::make_state_resolver(prescription));
            });
    }
};

} // namespace lbm::boundary::initialization

#endif
