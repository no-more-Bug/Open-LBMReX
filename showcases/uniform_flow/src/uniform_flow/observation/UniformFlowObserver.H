#ifndef LBM_CASES_UNIFORMFLOW_OBSERVER_H
#define LBM_CASES_UNIFORMFLOW_OBSERVER_H

#include "uniform_flow/config/Controls.H"
#include "uniform_flow/observation/UniformFlowDiagnostics.H"
#include "lbm/boundary/domain/LbmPostStreamingBoundary.H"
#include "lbm/config/LbmConfig.H"
#include "lbm/fields/MacroField.H"

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_REAL.H>
#include <AMReX_Utility.H>

#include <cmath>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>

namespace lbm::cases::uniform_flow {

struct ObservationResult {
    std::string error;
    bool outputFailure = false;

    bool ok() const noexcept { return error.empty(); }
};

/** Own the exact-uniform metrics, CSV schema and acceptance policy. */
template <class Descriptor, class Output>
class UniformFlowObserver {
public:
    UniformFlowObserver(
        ObservationConfig config,
        config::ResolvedLbmConfig const& lbm,
        amrex::Geometry const& geometry,
        amrex::BoxArray const& boxArray,
        amrex::DistributionMapping const& distributionMapping,
        boundary::PostStreamingBoundaryPlan const& boundaryPlan,
        std::string outputDirectory,
        Output& output)
        : cfg_(std::move(config)),
          lbm_(lbm),
          geom_(geometry),
          ba_(boxArray),
          dm_(distributionMapping),
          boundaryPlan_(boundaryPlan),
          metricsPath_(std::move(outputDirectory) + "/uniform_flow_metrics.csv"),
          output_(output)
    {}

    std::string prepare() const
    {
        std::string error;
        if (amrex::ParallelDescriptor::IOProcessor()) {
            try {
                if (cfg_.writeMetrics != 0 || cfg_.assertEnabled != 0) {
                    std::filesystem::remove(metricsPath_);
                }
            } catch (std::filesystem::filesystem_error const& exception) {
                error = "Failed to prepare uniform-flow artifacts: ";
                error += exception.what();
            }
        }
        amrex::BroadcastString(
            error,
            amrex::ParallelDescriptor::MyProc(),
            amrex::ParallelDescriptor::IOProcessorNumber(),
            amrex::ParallelDescriptor::Communicator());
        return error;
    }

    std::string runSummary() const
    {
        std::ostringstream os;
        os << "\nuniform_flow diagnostics:"
           << " write_uniform_flow_metrics = " << cfg_.writeMetrics
           << ", uniform_flow_assert = " << cfg_.assertEnabled
           << ", min_flow_throughs = " << cfg_.minFlowThroughs
           << ", E_2_u_tol = " << cfg_.e2UTolerance
           << ", E_inf_u_tol = " << cfg_.eInfUTolerance
           << ", std_ux/u0_tol = " << cfg_.stdUxTolerance
           << ", transverse_u/u0_tol = " << cfg_.transverseTolerance
           << ", relative_mass_tol(raw_lattice) = " << cfg_.massTolerance
           << ", metrics_csv = " << metricsPath_;
        return os.str();
    }

    ObservationResult writeFinal(MacroField const& macro, int step) const
    {
        if (!cfg_.enabled ||
            (cfg_.writeMetrics == 0 && cfg_.assertEnabled == 0)) {
            return {};
        }

        UniformFlowIntegrationGeometry integration;
        integration.flowLengthLb = lbm_.lengthRefLb;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            integration.trapezoidalAxes[d] =
                boundaryPlan_.axisGeometry[d].active ? 1 : 0;
        }
        UniformFlowMetrics const metrics = compute_uniform_flow_metrics(
            macro, geom_, ba_, dm_, step, lbm_.rhoRefLb, lbm_.uRefLb,
            integration);
        std::string error = write_uniform_flow_metrics_csv(metricsPath_, metrics);
        if (!error.empty()) {
            return {std::move(error), true};
        }
        output_.printAndLog(
            uniform_flow_metrics_summary(metrics) +
            "\nuniform_flow_metrics_csv = " + metricsPath_ + "\n");

        if (cfg_.assertEnabled == 0) {
            return {};
        }
        bool ok = true;
        std::ostringstream errors;
        errors << std::setprecision(12);
        auto check_at_least = [&](
            char const* label, amrex::Real value, amrex::Real limit) {
            if (!std::isfinite(value)) {
                ok = false;
                errors << "\n  - " << label << '=' << value
                       << " is not finite";
            } else if (value < limit) {
                ok = false;
                errors << "\n  - " << label << '=' << value
                       << " < required " << limit;
            }
        };
        auto check_at_most = [&](
            char const* label, amrex::Real value, amrex::Real limit) {
            if (!std::isfinite(value)) {
                ok = false;
                errors << "\n  - " << label << '=' << value
                       << " is not finite";
            } else if (value > limit) {
                ok = false;
                errors << "\n  - " << label << '=' << value
                       << " > " << limit;
            }
        };
        check_at_least(
            "flow_throughs", metrics.flowThroughs,
            static_cast<amrex::Real>(cfg_.minFlowThroughs));
        check_at_most("E_2_u", metrics.e2U, cfg_.e2UTolerance);
        check_at_most("E_inf_u", metrics.eInfU, cfg_.eInfUTolerance);
        check_at_most("std_ux/u0", metrics.stdUxOverU0, cfg_.stdUxTolerance);
        check_at_most(
            "max_abs_uy/u0", metrics.maxAbsUyOverU0,
            cfg_.transverseTolerance);
        check_at_most(
            "max_abs_uz/u0", metrics.maxAbsUzOverU0,
            cfg_.transverseTolerance);
        check_at_most(
            "relative_mass_error(raw_lattice)", metrics.relativeMassError,
            cfg_.massTolerance);
        check_at_most(
            "physical_weighted_relative_mass_error",
            metrics.physicalWeightedRelativeMassError,
            cfg_.massTolerance);
        if (!ok) {
            return {"uniform_flow check failed:" + errors.str(), false};
        }
        output_.printAndLog("uniform_flow check passed\n");
        return {};
    }

private:
    ObservationConfig cfg_;
    config::ResolvedLbmConfig const& lbm_;
    amrex::Geometry const& geom_;
    amrex::BoxArray const& ba_;
    amrex::DistributionMapping const& dm_;
    boundary::PostStreamingBoundaryPlan const& boundaryPlan_;
    std::string metricsPath_;
    Output& output_;
};

} // namespace lbm::cases::uniform_flow

#endif
