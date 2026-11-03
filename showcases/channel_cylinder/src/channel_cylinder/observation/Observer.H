#ifndef LBM_CASES_CHANNELCYLINDER_OBSERVER_H
#define LBM_CASES_CHANNELCYLINDER_OBSERVER_H

#include "channel_cylinder/observation/Diagnostics.H"
#include "channel_cylinder/observation/ObservationResult.H"
#include "channel_cylinder/observation/Output.H"
#include "lbm/boundary/internal/force/InternalSolidForce.H"
#include "runtime/SingleLevelRunSession.H"

#include <AMReX_BLassert.H>
#include <AMReX_REAL.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace lbm::cases::channel_cylinder {

inline void add_channel_cylinder_observation_requests(
    ChannelCylinderRuntimeConfig const& config,
    runtime::SingleLevelObservationPlan& plan) noexcept
{
    if (config.enabled) {
        plan.forceCalculationInterval = config.forceInterval;
        plan.forceDeliveryInterval = config.outputInterval;
    }
}

/**
 * @brief Channel-cylinder cadence, per-object statistics, and output owner.
 *
 * The observer consumes only globally aggregated ObjectId samples and the
 * current completed population at finalize. It owns neither boundary metadata
 * nor force pending storage.
 */
template <class Descriptor>
class ChannelCylinderObserver {
public:
    ChannelCylinderObserver(
        ChannelCylinderRuntimeConfig const& config,
        config::ResolvedLbmConfig const& lbm,
        std::string const& runOutputDirectory)
        : config_(config),
          lbm_(lbm),
          output_(config.enabled, config.writeHistory != 0)
    {
        statistics_.resize(config_.observedObjects.size());
        outputError_ = output_.prepare(runOutputDirectory);
    }

    std::string const& outputError() const noexcept { return outputError_; }


    void consume(
        boundary::InternalForceBatch const& batch)
    {
        if (!config_.enabled) {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                batch.empty(),
                "Disabled channel-cylinder observation received a force batch");
            return;
        }
        std::vector<ChannelCylinderHistoryRow> rows;
        rows.reserve(batch.objectSteps.size() * config_.observedObjects.size());
        for (boundary::InternalObjectForceSample const& sample : batch.objectSteps) {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                sample.step.populationStep > lastPopulationStep_,
                "Channel-cylinder force samples must be consumed in increasing step order");
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                sample.objectForces.size() == config_.observedObjects.size(),
                "Channel-cylinder global batch must contain every observed object");
            const amrex::Real forceTime =
                static_cast<amrex::Real>(sample.step.populationStep) -
                amrex::Real(0.5);
            for (std::size_t index = 0;
                 index < config_.observedObjects.size(); ++index) {
                auto const& object = config_.observedObjects[index];
                auto const& force = sample.objectForces[index];
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                    force.objectId.value == object.objectId,
                    "Channel-cylinder global batch ObjectId order changed");
                ChannelCylinderCoefficients const coefficients =
                    channel_cylinder_coefficients(
                        config_, lbm_.rhoRefLb, force);
                rows.push_back(ChannelCylinderHistoryRow{
                    sample.step.populationStep,
                    forceTime,
                    forceTime * config_.uMeanLb / config_.diameterLb,
                    config_.referenceLabel,
                    object.name,
                    object.objectId,
                    force.forceLb[0],
                    force.forceLb[1],
                    force.forceLb[2],
                    coefficients.cd,
                    coefficients.cl});

                ObjectStatistics& stats = statistics_[index];
                stats.finite = stats.finite &&
                    std::isfinite(force.forceLb[0]) &&
                    std::isfinite(force.forceLb[1]) &&
                    std::isfinite(force.forceLb[2]) &&
                    std::isfinite(coefficients.cd) &&
                    std::isfinite(coefficients.cl);
                if (sampleCount_ == 0) {
                    stats.cdMin = stats.cdMax = coefficients.cd;
                    stats.clMin = stats.clMax = coefficients.cl;
                } else {
                    stats.cdMin = std::min(stats.cdMin, coefficients.cd);
                    stats.cdMax = std::max(stats.cdMax, coefficients.cd);
                    stats.clMin = std::min(stats.clMin, coefficients.cl);
                    stats.clMax = std::max(stats.clMax, coefficients.cl);
                }
                stats.latestCd = coefficients.cd;
                stats.latestCl = coefficients.cl;
            }
            lastPopulationStep_ = sample.step.populationStep;
            ++sampleCount_;
        }
        output_.append(rows);
    }

    BenchmarkObservationResult finalize(
        LatticeField<Descriptor> const& current,
        bool steadyReached)
    {
        if (!config_.enabled) {
            return {};
        }
        if (!outputError_.empty()) {
            return {{}, outputError_, true};
        }
        ChannelCylinderPressureSample const pressure =
            sample_channel_cylinder_pressure(
                current, config_, lbm_);
        const bool pressureFinite =
            std::isfinite(pressure.rhoFrontLb) &&
            std::isfinite(pressure.rhoRearLb) &&
            std::isfinite(pressure.deltaPLb) &&
            std::isfinite(pressure.cDeltaP);

        const amrex::Real nan =
            std::numeric_limits<amrex::Real>::quiet_NaN();
        std::vector<ChannelCylinderMetrics> metricsRows;
        metricsRows.reserve(statistics_.size());
        std::ostringstream summary;
        bool allGatesPassed = true;
        for (std::size_t index = 0; index < statistics_.size(); ++index) {
            auto const& object = config_.observedObjects[index];
            auto const& stats = statistics_[index];
            ChannelCylinderMetrics metrics;
            metrics.referenceLabel = config_.referenceLabel;
            metrics.objectName = object.name;
            metrics.objectId = object.objectId;
            metrics.sampleCount = sampleCount_;
            metrics.finalPopulationStep = lastPopulationStep_;
            metrics.cdFinal = sampleCount_ > 0 ? stats.latestCd : nan;
            metrics.clFinal = sampleCount_ > 0 ? stats.latestCl : nan;
            metrics.cdMin = sampleCount_ > 0 ? stats.cdMin : nan;
            metrics.cdMax = sampleCount_ > 0 ? stats.cdMax : nan;
            metrics.cdMidrange = sampleCount_ > 0 ?
                amrex::Real(0.5) * (stats.cdMin + stats.cdMax) : nan;
            metrics.cdPeakToPeak = sampleCount_ > 0 ?
                stats.cdMax - stats.cdMin : nan;
            metrics.clMin = sampleCount_ > 0 ? stats.clMin : nan;
            metrics.clMax = sampleCount_ > 0 ? stats.clMax : nan;
            metrics.clMidrange = sampleCount_ > 0 ?
                amrex::Real(0.5) * (stats.clMin + stats.clMax) : nan;
            metrics.clPeakToPeak = sampleCount_ > 0 ?
                stats.clMax - stats.clMin : nan;
            metrics.rhoFrontLb = pressure.rhoFrontLb;
            metrics.rhoRearLb = pressure.rhoRearLb;
            metrics.deltaPLb = pressure.deltaPLb;
            metrics.cDeltaP = pressure.cDeltaP;

            const bool gatePassed = sampleCount_ > 0 && stats.finite &&
                pressureFinite && steadyReached &&
                stats.latestCd > 0.0 && stats.latestCl > 0.0;
            allGatesPassed = allGatesPassed && gatePassed;
            metrics.gateStatus = config_.assertEnabled == 0 ?
                "assert_disabled" : (gatePassed ? "pass" : "fail");
            summary << "channel_cylinder reference=" << config_.referenceLabel
                    << " object=" << object.name
                    << " samples=" << sampleCount_
                    << " final_Cd=" << metrics.cdFinal
                    << " final_Cl=" << metrics.clFinal
                    << " C_delta_p=" << pressure.cDeltaP
                    << " gate_status=" << metrics.gateStatus << '\n';
            metricsRows.push_back(std::move(metrics));
        }
        std::string outputError = output_.finalize(metricsRows);
        if (!outputError.empty()) {
            return {summary.str(), std::move(outputError), true};
        }
        summary << "channel_cylinder history_csv="
                << (config_.writeHistory != 0 ?
                        output_.historyPath() : "disabled")
                << " metrics_csv=" << output_.metricsPath() << '\n';
        BenchmarkObservationResult result;
        result.summary = summary.str();
        if (config_.assertEnabled != 0 && !allGatesPassed) {
            result.error =
                "Channel-cylinder steady acceptance requires samples, "
                "finite values, steady convergence, positive drag, and "
                "positive lift";
        }
        return result;
    }

private:
    struct ObjectStatistics {
        bool finite = true;
        amrex::Real cdMin = 0.0;
        amrex::Real cdMax = 0.0;
        amrex::Real clMin = 0.0;
        amrex::Real clMax = 0.0;
        amrex::Real latestCd = 0.0;
        amrex::Real latestCl = 0.0;
    };

    ChannelCylinderRuntimeConfig const& config_;
    config::ResolvedLbmConfig const& lbm_;
    ChannelCylinderOutput output_;
    std::vector<ObjectStatistics> statistics_;
    int sampleCount_ = 0;
    int lastPopulationStep_ = -1;
    std::string outputError_;
};

} // namespace lbm::cases::channel_cylinder

#endif
