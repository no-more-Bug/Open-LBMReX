#ifndef LBM_BOUNDARY_DOMAIN_POST_STREAM_GUOBOUNDARYDISPATCH_H
#define LBM_BOUNDARY_DOMAIN_POST_STREAM_GUOBOUNDARYDISPATCH_H

#include "lbm/boundary/domain/DomainBoundaryPrescription.H"
#include "lbm/boundary/domain/DomainFaceMetadata.H"
#include "lbm/boundary/domain/LbmPostStreamingBoundary.H"
#include "lbm/boundary/schemes/GuoNonEquilibriumExtrapolationScheme.H"

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

namespace lbm::boundary::post_stream {

namespace detail {

template <class Descriptor>
struct GuoNoSlipReconstruction {
    template <class DonorCell, class BoundaryCell>
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    void operator()(
        DonorCell const& donor,
        BoundaryCell& boundary,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const&,
        amrex::Real) const noexcept
    {
        GuoNonEquilibriumExtrapolationScheme<Descriptor>::no_slip(
            donor, boundary);
    }
};

template <class Descriptor, class VelocityProfile>
struct GuoVelocityReconstruction {
    VelocityProfile profile;

    template <class DonorCell, class BoundaryCell>
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    void operator()(
        DonorCell const& donor,
        BoundaryCell& boundary,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& position,
        amrex::Real physicalTime) const noexcept
    {
        const amrex::GpuArray<amrex::Real, 3> velocity =
            profile(position, physicalTime);
        GuoNonEquilibriumExtrapolationScheme<Descriptor>::velocity(
            donor, boundary, velocity);
    }
};

template <class Descriptor>
struct GuoPressureReconstruction {
    ConstantDensityProfile profile;

    template <class DonorCell, class BoundaryCell>
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    void operator()(
        DonorCell const& donor,
        BoundaryCell& boundary,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& position,
        amrex::Real physicalTime) const noexcept
    {
        const amrex::Real rho = profile(position, physicalTime);
        GuoNonEquilibriumExtrapolationScheme<Descriptor>::pressure(
            donor, boundary, rho);
    }
};

template <class Descriptor, class VelocityProfile>
struct GuoPrescribedStateReconstruction {
    ConstantDensityProfile densityProfile;
    VelocityProfile velocityProfile;

    template <class DonorCell, class BoundaryCell>
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    void operator()(
        DonorCell const& donor,
        BoundaryCell& boundary,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& position,
        amrex::Real physicalTime) const noexcept
    {
        const amrex::Real rho = densityProfile(position, physicalTime);
        const amrex::GpuArray<amrex::Real, 3> velocity =
            velocityProfile(position, physicalTime);
        GuoNonEquilibriumExtrapolationScheme<Descriptor>::prescribed_state(
            donor, boundary, rho, velocity);
    }
};

template <class Descriptor>
struct GuoFreeOutflowReconstruction {
    template <class DonorCell, class BoundaryCell>
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    void operator()(
        DonorCell const& donor,
        BoundaryCell& boundary,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const&,
        amrex::Real) const noexcept
    {
        GuoNonEquilibriumExtrapolationScheme<Descriptor>::free_outflow_copy(
            donor, boundary);
    }
};

template <class Descriptor>
AMREX_FORCE_INLINE
GuoNoSlipReconstruction<Descriptor> make_reconstruction(
    NoSlipPrescription) noexcept
{
    return {};
}

template <class Descriptor, class VelocityProfile>
AMREX_FORCE_INLINE
GuoVelocityReconstruction<Descriptor, VelocityProfile> make_reconstruction(
    VelocityPrescription<VelocityProfile> const& prescription) noexcept
{
    return GuoVelocityReconstruction<Descriptor, VelocityProfile>{
        prescription.velocityProfile};
}

template <class Descriptor>
AMREX_FORCE_INLINE
GuoPressureReconstruction<Descriptor> make_reconstruction(
    PressurePrescription const& prescription) noexcept
{
    return GuoPressureReconstruction<Descriptor>{
        prescription.densityProfile};
}

template <class Descriptor, class VelocityProfile>
AMREX_FORCE_INLINE
GuoPrescribedStateReconstruction<Descriptor, VelocityProfile>
make_reconstruction(
    PrescribedStatePrescription<VelocityProfile> const& prescription) noexcept
{
    return GuoPrescribedStateReconstruction<Descriptor, VelocityProfile>{
        prescription.densityProfile,
        prescription.velocityProfile};
}

template <class Descriptor>
AMREX_FORCE_INLINE
GuoFreeOutflowReconstruction<Descriptor> make_reconstruction(
    FreeOutflowPrescription) noexcept
{
    return {};
}

} // namespace detail

/**
 * @brief Maps Guo domain metadata to one device-safe reconstruction functor.
 *
 * A scheme-neutral domain-boundary prescription dispatcher owns the shared
 * physical/profile matrix. This Guo adapter maps the selected prescription to
 * a non-equilibrium reconstruction (or exact donor copy). The supplied
 * launcher owns AMReX region traversal and provides the valid-node physical
 * position and target physical time, so adding another post-stream numerical scheme
 * does not add its reconstruction types here or to the generic executor.
 */
template <class Descriptor>
class GuoBoundaryDispatch {
public:
    /**
     * @brief Select and launch the Guo reconstruction for one compiled region.
     *
     * @tparam RegionLauncher Callable accepting one reconstruction functor.
     * @param faces Device-safe metadata for all domain faces.
     * @param region Disjoint boundary region being reconstructed.
     * @param sideIndex Index of `region.effectiveSide` in the face table.
     * @param launchRegion Synchronous host callback that launches region work.
     */
    template <class RegionLauncher>
    static void dispatch(
        DomainFaceTableView const& faces,
        PostStreamBoundaryRegion const& region,
        int sideIndex,
        RegionLauncher const& launchRegion)
    {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            faces.scheme[sideIndex] == BoundaryScheme::GuoExtrapolation,
            "Guo dispatch requires Guo face metadata");

        dispatch_domain_boundary_prescription(
            faces,
            region.effectiveSide,
            region.effectivePhysical,
            region.effectiveProfile,
            sideIndex,
            [&launchRegion](auto const& prescription) {
                launchRegion(
                    detail::make_reconstruction<Descriptor>(prescription));
            });
    }
};

} // namespace lbm::boundary::post_stream

#endif
