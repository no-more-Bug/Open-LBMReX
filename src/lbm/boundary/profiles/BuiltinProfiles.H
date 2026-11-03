#ifndef LBM_BOUNDARY_PROFILES_BUILTINPROFILES_H
#define LBM_BOUNDARY_PROFILES_BUILTINPROFILES_H

#include <AMReX_Array.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

namespace lbm::boundary {

/**
 * @brief Device-safe constant lattice-velocity boundary profile.
 *
 * The physical position/time arguments are intentionally part of the common boundary
 * profile contract even though a uniform profile does not use them.
 */
struct UniformVelocityProfile {
    amrex::GpuArray<amrex::Real, 3> velocity{0.0, 0.0, 0.0};

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::GpuArray<amrex::Real, 3> operator()(
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const&,
        amrex::Real /* physicalTime */) const noexcept
    {
        return velocity;
    }
};

/**
 * @brief Device-safe constant lattice-density boundary profile.
 *
 * The physical position/time arguments are intentionally part of the common boundary
 * profile contract even though a constant profile does not use them.
 */
struct ConstantDensityProfile {
    amrex::Real density = 0.0;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real operator()(
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const&,
        amrex::Real /* physicalTime */) const noexcept
    {
        return density;
    }
};

/**
 * @brief Device-safe parabolic lattice-velocity boundary profile.
 *
 * The caller supplies the physical sampling position appropriate to its
 * numerical scheme: an HBB halfway-link point or a Guo outer valid-cell
 * center. profileAxis selects its coordinate, peakVelocity fixes direction
 * and magnitude, and transverseOrigin/channelHeight hold the physical wall
 * interval resolved once by the boundary execution path. No integer index,
 * Geometry, Box, boundary plan, or scheme data is retained by the profile.
 *
 * @pre profileAxis is a valid AMREX_SPACEDIM axis and channelHeight is
 *      positive.
 */
struct PoiseuilleVelocityProfile {
    int profileAxis = -1;
    amrex::GpuArray<amrex::Real, 3> peakVelocity{0.0, 0.0, 0.0};
    amrex::Real transverseOrigin = 0.0;
    amrex::Real channelHeight = 1.0;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real normalized_coordinate(
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& position) const
        noexcept
    {
        amrex::Real eta = channelHeight > 0.0 ?
            (position[profileAxis] - transverseOrigin) / channelHeight : 0.0;
        eta = eta < 0.0 ? 0.0 : eta;
        eta = eta > 1.0 ? 1.0 : eta;
        return eta;
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::GpuArray<amrex::Real, 3> operator()(
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& position,
        amrex::Real /* physicalTime */) const noexcept
    {
        const amrex::Real eta = normalized_coordinate(position);
        const amrex::Real scale = 4.0 * eta * (1.0 - eta);
        return amrex::GpuArray<amrex::Real, 3>{
            scale * peakVelocity[0],
            scale * peakVelocity[1],
            scale * peakVelocity[2]};
    }
};

} // namespace lbm::boundary

#endif
