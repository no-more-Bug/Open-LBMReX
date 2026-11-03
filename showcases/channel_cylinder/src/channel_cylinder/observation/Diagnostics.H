#ifndef LBM_CASES_CHANNELCYLINDER_DIAGNOSTICS_H
#define LBM_CASES_CHANNELCYLINDER_DIAGNOSTICS_H

#include "channel_cylinder/config/ObservationConfig.H"
#include "lbm/boundary/internal/force/InternalSolidForce.H"
#include "lbm/config/LbmConfig.H"
#include "lbm/dynamics/Moments.H"
#include "lbm/fields/LatticeField.H"

#include <AMReX_BLassert.H>
#include <AMReX_Box.H>
#include <AMReX_Gpu.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>
#include <AMReX_REAL.H>

#include <algorithm>
#include <array>
#include <cmath>

namespace lbm::cases::channel_cylinder {

struct ChannelCylinderCoefficients {
    amrex::Real cd = 0.0;
    amrex::Real cl = 0.0;
};

struct ChannelCylinderPressureSample {
    amrex::Real rhoFrontLb = 0.0;
    amrex::Real rhoRearLb = 0.0;
    amrex::Real deltaPLb = 0.0;
    amrex::Real cDeltaP = 0.0;
};

/** @brief Normalize one global object-force value with the configured D and span. */
inline ChannelCylinderCoefficients channel_cylinder_coefficients(
    ChannelCylinderRuntimeConfig const& config,
    amrex::Real rhoReferenceLb,
    boundary::InternalObjectForce const& force) noexcept
{
    const amrex::Real denominator =
        rhoReferenceLb * config.uMeanLb * config.uMeanLb *
        config.diameterLb * config.spanLb;
    return {
        amrex::Real(2.0) * force.forceLb[0] / denominator,
        amrex::Real(2.0) * force.forceLb[1] / denominator};
}

/**
 * @brief Sample the two aligned DFG pressure points from current populations.
 *
 * Local reduction may execute on CPU/OpenMP or GPU. `ReduceData::value()`
 * completes device result retrieval before one four-Real MPI allreduce is
 * entered outside the OpenMP region.
 */
template <class Descriptor>
ChannelCylinderPressureSample sample_channel_cylinder_pressure(
    LatticeField<Descriptor> const& current,
    ChannelCylinderRuntimeConfig const& config,
    config::ResolvedLbmConfig const& lbm)
{
    using ReduceOps = amrex::ReduceOps<
        amrex::ReduceOpSum,
        amrex::ReduceOpSum,
        amrex::ReduceOpSum,
        amrex::ReduceOpSum>;
    using ReduceData =
        amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real, amrex::Real>;
    using ReduceTuple = typename ReduceData::Type;

    amrex::MultiFab const& populations = current.multifab();
    const amrex::Box domain = populations.boxArray().minimalBox();
    const int firstI = std::min(
        config.pressureFrontCell[0], config.pressureRearCell[0]);
    const int lastI = std::max(
        config.pressureFrontCell[0], config.pressureRearCell[0]);
    const amrex::Box probeBox(
        amrex::IntVect(AMREX_D_DECL(
            firstI,
            config.pressureFrontCell[1],
            domain.smallEnd(2))),
        amrex::IntVect(AMREX_D_DECL(
            lastI,
            config.pressureFrontCell[1],
            domain.bigEnd(2))));

    ReduceOps reduceOps;
    ReduceData reduceData(reduceOps);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(populations, false); mfi.isValid(); ++mfi) {
        const amrex::Box bx = probeBox & mfi.validbox();
        if (!bx.ok()) {
            continue;
        }
        auto const f = populations.const_array(mfi);
        const int frontI = config.pressureFrontCell[0];
        const int rearI = config.pressureRearCell[0];
        reduceOps.eval(
            bx,
            reduceData,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) -> ReduceTuple {
                if (i != frontI && i != rearI) {
                    return {0.0, 0.0, 0.0, 0.0};
                }
                const amrex::Real rho = Moments<Descriptor>::rho(
                    ConstPopulationCell{f, i, j, k});
                return i == frontI ?
                    ReduceTuple{rho, 0.0, 1.0, 0.0} :
                    ReduceTuple{0.0, rho, 0.0, 1.0};
            });
    }

    ReduceTuple const local = reduceData.value();
    std::array<amrex::Real, 4> global{
        amrex::get<0>(local),
        amrex::get<1>(local),
        amrex::get<2>(local),
        amrex::get<3>(local)};
    amrex::ParallelDescriptor::ReduceRealSum(
        global.data(), static_cast<int>(global.size()));

    const amrex::Real expected =
        static_cast<amrex::Real>(config.pressureSamplesPerPoint);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        global[2] == expected && global[3] == expected,
        "Channel-cylinder pressure probes do not have the resolved global span count");

    ChannelCylinderPressureSample sample;
    sample.rhoFrontLb = global[0] / global[2];
    sample.rhoRearLb = global[1] / global[3];
    sample.deltaPLb =
        Descriptor::cs2() * (sample.rhoFrontLb - sample.rhoRearLb);
    sample.cDeltaP = sample.deltaPLb /
        (lbm.rhoRefLb * config.uMeanLb * config.uMeanLb);
    return sample;
}

} // namespace lbm::cases::channel_cylinder

#endif
