#ifndef LBM_CASES_LIDDRIVENCAVITY_CAVITYOBSERVER_H
#define LBM_CASES_LIDDRIVENCAVITY_CAVITYOBSERVER_H

#include "lid_driven_cavity/observation/CavityOutput.H"
#include "lid_driven_cavity/observation/CavityValidationDiagnostics.H"
#include "lbm/config/LbmConfig.H"
#include "lbm/fields/MacroField.H"

#include <AMReX_BLassert.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_REAL.H>

#include <sstream>
#include <string>
#include <utility>

namespace lbm::cases::lid_driven_cavity {

/** Validated cavity observation, acceptance and artifact policy. */
struct CavityObservationConfig {
    bool enabled = false;
    int writeCenterlines = 0;
    int writeMetrics = 0;
    amrex::Real lidSpeedLb = 0.0;
    amrex::Real lengthLb = 0.0;
    CavityDiagnosticGeometry geometry;
};

/**
 * Own final cavity sampling, physical geometry and output schema.
 *
 * The generic runtime supplies a completed read-only macro field at the fixed
 * final observation moment. This policy neither reconstructs macros nor owns
 * population, boundary or MPI lifecycle.
 */
class CavityObserver {
public:
    CavityObserver(
        CavityObservationConfig config,
        config::ResolvedLbmConfig const& lbm,
        amrex::Geometry const& geometry,
        amrex::BoxArray const& boxArray,
        amrex::DistributionMapping const& distributionMapping)
        : config_(std::move(config)),
          lbm_(lbm),
          geom_(geometry),
          ba_(boxArray),
          dm_(distributionMapping)
    {}

    std::string prepare(std::string const& runOutputDirectory)
    {
        if (!config_.enabled) {
            return {};
        }
        return output_.prepare(
            runOutputDirectory,
            config_.writeCenterlines,
            config_.writeMetrics);
    }

    std::string runSummary(
        int steadyCheckInterval,
        amrex::Real steadyTolerance,
        int stopOnSteady) const
    {
        if (!config_.enabled) {
            return {};
        }
        const amrex::Real cavityRe =
            config_.lidSpeedLb * config_.lengthLb / lbm_.nuLb;
        std::ostringstream os;
        os << "\nlid_driven_cavity viscosity source: "
           << "lbm.parameterization.kind="
           << config::to_string(lbm_.parameterizationKind)
           << "; moving-lid boundary u_lb is a read-only consistency check\n"
           << "cavity U_lid_lb = " << config_.lidSpeedLb
           << ", L_lb = " << config_.lengthLb
           << ", nu_lb = " << lbm_.nuLb
           << ", tau = " << lbm_.tau
           << ", omega = " << lbm_.omega
           << ", Re_cavity = " << cavityRe << "\n"
           << "cavity steady_check_int = " << steadyCheckInterval
           << ", steady_l2_tol = " << steadyTolerance
           << ", stop_on_steady = " << stopOnSteady
           << ", write_cavity_centerlines = "
           << config_.writeCenterlines
           << ", write_cavity_metrics = " << config_.writeMetrics
           << "\n"
           << "cavity boundary_geometry = "
           << (config_.geometry.onLatticeBoundaryNodes != 0 ?
                   "on_lattice_valid_node" : "domain_face_halfway")
           << ", sampling = "
           << (config_.geometry.onLatticeBoundaryNodes != 0 ?
                   "direct_node" : "interpolated_cell_center")
           << ", valid_nodes_xy = ("
           << config_.geometry.validNodes[0] << ','
           << config_.geometry.validNodes[1] << ')'
           << ", lattice_intervals_xy = ("
           << config_.geometry.latticeIntervals[0] << ','
           << config_.geometry.latticeIntervals[1] << ')'
           << ", physical_bounds_xy = (["
           << config_.geometry.physicalLo[0] << ','
           << config_.geometry.physicalHi[0] << "],["
           << config_.geometry.physicalLo[1] << ','
           << config_.geometry.physicalHi[1] << "])"
           << "\n"
           << "cavity steady residual compares current velocity to the "
              "previous time step;"
           << " steady_check_int only controls reduction/log cadence";
        return os.str();
    }

    std::string writeFinal(
        MacroField const& macro,
        int step,
        amrex::Real lastSteadyResidual,
        amrex::Real steadyTolerance,
        bool steadyReached)
    {
        if (!config_.enabled) {
            return {};
        }
        std::string summary;
        if (config_.writeCenterlines != 0) {
            output_.prepareCenterlineWrite();
            CavityCenterlineSamples const samples = sample_cavity_centerlines(
                macro, geom_, config_.lidSpeedLb, config_.geometry);
            summary += output_.writeCenterlines(
                samples,
                geom_,
                config_.lidSpeedLb,
                step,
                lastSteadyResidual,
                steadyTolerance,
                steadyReached,
                config_.geometry);
        }

        if (config_.writeMetrics != 0) {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                config_.geometry.onLatticeBoundaryNodes != 0,
                "cavity validation metrics currently require all-Guo "
                "on-lattice cavity walls");
            CavityValidationMetrics const metrics =
                compute_cavity_validation_metrics(
                    macro,
                    geom_,
                    ba_,
                    dm_,
                    step,
                    lbm_.rhoRefLb,
                    config_.lidSpeedLb,
                    config_.geometry);
            summary += output_.writeValidationMetrics(metrics);
        }
        return summary;
    }

private:
    CavityObservationConfig config_;
    config::ResolvedLbmConfig const& lbm_;
    amrex::Geometry const& geom_;
    amrex::BoxArray const& ba_;
    amrex::DistributionMapping const& dm_;
    CavityOutput output_;
};

} // namespace lbm::cases::lid_driven_cavity

#endif
