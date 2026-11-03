#ifndef LBM_BOUNDARY_DOMAIN_DOMAINBOUNDARYPRESCRIPTION_H
#define LBM_BOUNDARY_DOMAIN_DOMAINBOUNDARYPRESCRIPTION_H

#include "lbm/boundary/domain/DomainFaceMetadata.H"
#include "lbm/boundary/profiles/BuiltinProfiles.H"

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

namespace lbm::boundary {

/** @brief Boundary state with donor density and prescribed zero velocity. */
struct NoSlipPrescription {};

/** @brief Boundary state with donor density and a prescribed velocity profile. */
template <class VelocityProfile>
struct VelocityPrescription {
    VelocityProfile velocityProfile;
};

/** @brief Boundary state with prescribed density and donor velocity. */
struct PressurePrescription {
    ConstantDensityProfile densityProfile;
};

/** @brief Boundary state with prescribed density and velocity profiles. */
template <class VelocityProfile>
struct PrescribedStatePrescription {
    ConstantDensityProfile densityProfile;
    VelocityProfile velocityProfile;
};

/** @brief Exact donor-state/copy outflow prescription. */
struct FreeOutflowPrescription {};

/**
 * @brief Return the sign of a prescribed normal inlet/outlet velocity.
 *
 * Positive lattice velocity follows the domain-axis orientation. An inlet
 * points inward and an outlet points outward; the magnitude is resolved
 * separately and remains positive in face metadata.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int prescribed_normal_velocity_sign(
    DomainSide side,
    BoundaryPhysical physical) noexcept
{
    const int inwardSign = domain_side_is_low(side) ? 1 : -1;
    return physical == BoundaryPhysical::VelocityInlet ?
        inwardSign : -inwardSign;
}

/**
 * @brief Build the standard immutable Poiseuille velocity profile for one face.
 *
 * The face table owns the setup-resolved transverse interval: HBB samples
 * between problem faces, while Guo samples between the two outer valid-node
 * centers. This helper only assembles the immutable profile value object.
 *
 * @pre physical is a Poiseuille velocity inlet/outlet or prescribed-state
 *      boundary and profileAxis is valid for the active descriptor.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
PoiseuilleVelocityProfile make_poiseuille_velocity_profile(
    DomainFaceTableView const& faces,
    int sideIndex,
    DomainSide side,
    BoundaryPhysical physical) noexcept
{
    amrex::GpuArray<amrex::Real, 3> peakVelocity{0.0, 0.0, 0.0};
    if (physical == BoundaryPhysical::PrescribedState) {
        peakVelocity = faces.prescribedULb[sideIndex];
    } else {
        peakVelocity[domain_side_axis(side)] =
            static_cast<amrex::Real>(
                prescribed_normal_velocity_sign(side, physical)) *
            faces.uMaxLb[sideIndex];
    }
    return PoiseuilleVelocityProfile{
        faces.profileAxis[sideIndex],
        peakVelocity,
        faces.profileOrigin[sideIndex],
        faces.profileHeight[sideIndex]};
}

/**
 * @brief Dispatch one resolved face to a typed macroscopic prescription.
 *
 * This is the single host-side physical/profile matrix shared by Guo
 * valid-boundary initialization and normal post-stream reconstruction. The
 * visitor receives a small device-safe value object; each numerical lifecycle
 * remains responsible for adapting that prescription to its own Operator or
 * reconstruction functor.
 *
 * HBB keeps its device-side profile switch because physical ghost cells are
 * selected inside the PhysBC kernel. The profile constructors and prescribed
 * velocity semantics remain shared.
 */
template <class Visitor>
void dispatch_domain_boundary_prescription(
    DomainFaceTableView const& faces,
    DomainSide side,
    BoundaryPhysical physical,
    BoundaryProfileKind profile,
    int sideIndex,
    Visitor const& visitor)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        sideIndex == domain_side_index(side) &&
            faces.enabled[sideIndex] != 0 &&
            faces.physical[sideIndex] == physical &&
            faces.profile[sideIndex] == profile,
        "domain boundary prescription does not match compiled face metadata");

    switch (physical) {
    case BoundaryPhysical::NoSlipWall:
        visitor(NoSlipPrescription{});
        return;

    case BoundaryPhysical::MovingWall:
        visitor(VelocityPrescription<UniformVelocityProfile>{
            UniformVelocityProfile{faces.prescribedULb[sideIndex]}});
        return;

    case BoundaryPhysical::VelocityInlet:
    case BoundaryPhysical::VelocityOutlet:
        if (profile == BoundaryProfileKind::Uniform) {
            visitor(VelocityPrescription<UniformVelocityProfile>{
                UniformVelocityProfile{faces.prescribedULb[sideIndex]}});
            return;
        }
        if (profile == BoundaryProfileKind::Poiseuille) {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                faces.profileAxis[sideIndex] >= 0 &&
                    faces.profileAxis[sideIndex] < AMREX_SPACEDIM,
                "Poiseuille velocity prescription has invalid profile axis");
            visitor(VelocityPrescription<PoiseuilleVelocityProfile>{
                make_poiseuille_velocity_profile(
                    faces, sideIndex, side, physical)});
            return;
        }
        break;

    case BoundaryPhysical::PressureInlet:
    case BoundaryPhysical::PressureOutlet:
        visitor(PressurePrescription{
            ConstantDensityProfile{faces.rhoLb[sideIndex]}});
        return;

    case BoundaryPhysical::PrescribedState:
        if (profile == BoundaryProfileKind::Uniform) {
            visitor(PrescribedStatePrescription<UniformVelocityProfile>{
                ConstantDensityProfile{faces.rhoLb[sideIndex]},
                UniformVelocityProfile{faces.prescribedULb[sideIndex]}});
            return;
        }
        if (profile == BoundaryProfileKind::Poiseuille) {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                faces.profileAxis[sideIndex] >= 0 &&
                    faces.profileAxis[sideIndex] < AMREX_SPACEDIM,
                "Poiseuille prescribed-state prescription has invalid profile axis");
            visitor(PrescribedStatePrescription<PoiseuilleVelocityProfile>{
                ConstantDensityProfile{faces.rhoLb[sideIndex]},
                make_poiseuille_velocity_profile(
                    faces, sideIndex, side, physical)});
            return;
        }
        break;

    case BoundaryPhysical::FreeOutflow:
        visitor(FreeOutflowPrescription{});
        return;

    default:
        break;
    }

    amrex::Abort(
        "domain boundary prescription received unsupported physical/profile metadata");
}

} // namespace lbm::boundary

#endif
