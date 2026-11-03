#ifndef LBM_CASES_POISEUILLEFLOW_CASEOBSERVER_H
#define LBM_CASES_POISEUILLEFLOW_CASEOBSERVER_H

#include "poiseuille_flow/config/Controls.H"
#include "poiseuille_flow/observation/PoiseuilleDiagnostics.H"
#include "lbm/config/LbmConfig.H"
#include "lbm/fields/MacroField.H"
#include "runtime/SingleLevelDiagnosticsConfig.H"

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_REAL.H>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace lbm::cases::poiseuille_flow {

struct ObservationResult {
    std::string error;
    bool outputFailure = false;

    bool ok() const noexcept { return error.empty(); }
};

/** Own the Poiseuille reductions, artifact schema and acceptance policy. */
template <class Descriptor, class Output>
class PoiseuilleCaseObserver {
public:
    PoiseuilleCaseObserver(
        ObservationConfig config,
        config::ResolvedLbmConfig const& lbm,
        amrex::Geometry const& geometry,
        amrex::BoxArray const& boxArray,
        amrex::DistributionMapping const& distributionMapping,
        std::string outputDirectory,
        Output& output)
        : cfg_(std::move(config)),
          lbm_(lbm),
          geom_(geometry),
          ba_(boxArray),
          dm_(distributionMapping),
          profilePath_(outputDirectory + "/poiseuille_profile.csv"),
          metricsPath_(outputDirectory + "/poiseuille_profile_metrics.csv"),
          densityPath_(outputDirectory + "/poiseuille_density_pressure.csv"),
          output_(output)
    {}

    std::string prepare() const
    {
        std::string error;
        if (amrex::ParallelDescriptor::IOProcessor()) {
            try {
                if (cfg_.enabled &&
                    (cfg_.writeProfile != 0 || cfg_.profileAssert != 0)) {
                    std::filesystem::remove(profilePath_);
                    std::filesystem::remove(metricsPath_);
                    std::filesystem::remove(densityPath_);
                }
            } catch (std::filesystem::filesystem_error const& exception) {
                error = "Failed to prepare Poiseuille artifacts: ";
                error += exception.what();
            }
        }
        broadcast(error);
        return error;
    }

    std::string runSummary() const
    {
        std::ostringstream os;
        os << std::setprecision(12)
           << "\npoiseuille_flow diagnostics:"
           << " U_max_lb = " << lbm_.uRefLb
           << ", U_mean_lb = " << (2.0 * lbm_.uRefLb / 3.0)
           << ", profile_axis = y"
           << ", profile_geometry = "
           << (cfg_.geometry.onLatticeBoundaryNodes != 0 ?
                   "valid_node_center" : "cell_center_inside_domain_face")
           << ", profile_valid_nodes = " << cfg_.geometry.profileValidNodes
           << ", profile_lattice_intervals = "
           << cfg_.geometry.profileLatticeIntervals
           << ", profile_physical_bounds = ["
           << cfg_.geometry.profilePhysicalLo << ','
           << cfg_.geometry.profilePhysicalHi << ']'
           << ", write_profile = " << cfg_.writeProfile
           << ", profile_assert = " << cfg_.profileAssert
           << ", E_2_profile_tol = " << cfg_.e2ProfileTolerance
           << ", E_inf_profile_tol = " << cfg_.eInfProfileTolerance
           << ", transverse_u/Umax_tol = " << cfg_.transverseTolerance
           << ", mass_assert = " << cfg_.massAssert
           << ", relative_mass_tol = " << cfg_.massTolerance
           << ", pressure_boundary_rho_tol = "
           << cfg_.pressureBoundaryRhoTolerance
           << ", profile_csv = " << profilePath_
           << ", metrics_csv = " << metricsPath_
           << ", density_pressure_csv = " << densityPath_;
        if (cfg_.pressure.enabled != 0) {
            os << (cfg_.pressure.prescribedState != 0 ?
                       "\nprescribed-state Poiseuille: prescribed_state = true" :
                       "\npressure-driven Poiseuille: pressure_driven = true")
               << ", rho_in_lb = " << cfg_.pressure.rhoInLb
               << ", rho_out_lb = " << cfg_.pressure.rhoOutLb
               << ", density_delta_target_lb = "
               << cfg_.pressure.densityDeltaTargetLb
               << ", density_delta_expected_lb = "
               << cfg_.pressure.densityDeltaExpectedLb
               << ", density_delta/rho_ref = "
               << cfg_.pressure.densityDeltaOverRhoRef
               << ", Lx_lb(intervals) = "
               << cfg_.pressure.streamwiseLengthLb
               << ", H_lb(intervals) = "
               << cfg_.pressure.channelHeightLb
               << ", Mach = " << cfg_.pressure.mach
               << ", Re_effective = " << cfg_.pressure.reEffective
               << ", convergence_scaling = " << cfg_.pressureScaling;
        }
        return os.str();
    }

    ObservationResult writeFinal(
        MacroField const& macro,
        int step,
        runtime::SingleLevelFinalSummaryView const& steady,
        bool steadyReached) const
    {
        if (!cfg_.enabled ||
            (cfg_.writeProfile == 0 && cfg_.profileAssert == 0)) {
            return {};
        }

        PoiseuilleProfileData const data = compute_poiseuille_profile_data(
            macro,
            geom_,
            ba_,
            dm_,
            step,
            lbm_.rhoRefLb,
            lbm_.uRefLb,
            lbm_.nuLb,
            Descriptor::cs2(),
            cfg_.geometry,
            cfg_.pressure);
        std::string error = writeArtifacts(data);
        if (!error.empty()) {
            return {std::move(error), true};
        }
        writeSummary(data.output, steady);

        if (cfg_.profileAssert == 0) {
            return {};
        }
        PoiseuilleProfileOutput const& profile = data.output;
        bool ok = true;
        std::ostringstream errors;
        errors << std::setprecision(12);
        if (!steadyReached) {
            ok = false;
            errors << "\n  - steady gate not reached";
        }
        auto check_at_most = [&](char const* label, amrex::Real value,
                                 amrex::Real limit) {
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
        auto check_finite = [&](char const* label, amrex::Real value) {
            if (!std::isfinite(value)) {
                ok = false;
                errors << "\n  - " << label << '=' << value
                       << " is not finite";
            }
        };

        check_at_most("E_2_profile(interior)", profile.e2Profile,
                      cfg_.e2ProfileTolerance);
        check_at_most("E_inf_profile(interior)", profile.eInfProfile,
                      cfg_.eInfProfileTolerance);
        check_at_most("E_2_profile_full", profile.e2ProfileFull,
                      cfg_.e2ProfileTolerance);
        check_at_most("E_inf_profile_full", profile.eInfProfileFull,
                      cfg_.eInfProfileTolerance);
        check_at_most("max_abs_uy/Umax", profile.maxAbsUyOverUmax,
                      cfg_.transverseTolerance);
        check_at_most("max_abs_uz/Umax", profile.maxAbsUzOverUmax,
                      cfg_.transverseTolerance);
        check_finite("relative_mass_error(raw_lattice)",
                     profile.relativeMassError);
        check_finite("physical_weighted_relative_mass_error",
                     profile.physicalWeightedRelativeMassError);
        check_finite("rho_streamwise_linear_r2",
                     profile.rhoStreamwiseLinearR2);
        check_finite("max_abs_profile_error_all_x/Umax",
                     profile.maxAbsProfileErrorAllXOverUmax);
        check_finite("max_streamwise_ux_spread/Umax",
                     profile.maxStreamwiseUxSpreadOverUmax);
        if (cfg_.pressure.enabled != 0) {
            check_at_most("max_abs_pressure_boundary_rho_error",
                          profile.maxAbsPressureBoundaryRhoError,
                          cfg_.pressureBoundaryRhoTolerance);
            check_finite("rho_streamwise_delta_relative_error",
                         profile.rhoStreamwiseDeltaRelativeError);
        }
        if (cfg_.massAssert != 0) {
            check_at_most("relative_mass_error(raw_lattice)",
                          profile.relativeMassError, cfg_.massTolerance);
            check_at_most("physical_weighted_relative_mass_error",
                          profile.physicalWeightedRelativeMassError,
                          cfg_.massTolerance);
        }
        if (!ok) {
            return {"poiseuille_flow check failed:" + errors.str(), false};
        }
        output_.printAndLog("poiseuille_flow check passed\n");
        return {};
    }

private:
    static void broadcast(std::string& value)
    {
        amrex::BroadcastString(
            value,
            amrex::ParallelDescriptor::MyProc(),
            amrex::ParallelDescriptor::IOProcessorNumber(),
            amrex::ParallelDescriptor::Communicator());
    }

    std::string writeArtifacts(PoiseuilleProfileData const& data) const
    {
        std::string error;
        if (amrex::ParallelDescriptor::IOProcessor()) {
            try {
                writeProfile(data);
                writeDensity(data);
                writeMetrics(data.output);
            } catch (std::exception const& exception) {
                error = "Failed to write Poiseuille artifacts: ";
                error += exception.what();
            }
        }
        broadcast(error);
        return error;
    }

    static std::ofstream checkedStream(std::string const& path)
    {
        std::ofstream stream(path, std::ios::out);
        if (!stream.is_open()) {
            throw std::runtime_error("cannot open " + path);
        }
        stream.exceptions(std::ios::badbit | std::ios::failbit);
        stream << std::setprecision(17);
        return stream;
    }

    void writeProfile(PoiseuilleProfileData const& data) const
    {
        std::ofstream os = checkedStream(profilePath_);
        os << "j,y_phys,eta,ux_exact,ux_center_zmean,uy_abs_max,"
              "uz_abs_max,sample_i,is_physical_boundary_node\n";
        for (int jy = 0; jy < data.output.ny; ++jy) {
            os << jy << ',' << data.profilePhysicalCoordinate[jy] << ','
               << data.profileEta[jy] << ','
               << data.profileExactVelocity[jy] << ',' << data.uxMean[jy]
               << ',' << data.uyAbsMax[jy] << ',' << data.uzAbsMax[jy]
               << ',' << data.centerI << ',' << data.profileBoundaryNode[jy]
               << '\n';
        }
    }

    void writeDensity(PoiseuilleProfileData const& data) const
    {
        std::ofstream os = checkedStream(densityPath_);
        os << "i,x_phys,x_over_L,rho_cross_section_weighted_mean,"
              "pressure_lb_cross_section_weighted_mean\n";
        for (int ix = 0; ix < data.output.nx; ++ix) {
            os << ix << ',' << data.streamwisePhysicalCoordinate[ix] << ','
               << data.streamwiseXi[ix] << ','
               << data.rhoCrossSectionMean[ix] << ','
               << data.pressureCrossSectionMean[ix] << '\n';
        }
    }

    void writeMetrics(PoiseuilleProfileOutput const& value) const
    {
        std::ofstream os = checkedStream(metricsPath_);
        os << "step,nx,ny,nz,n_intervals_y,on_lattice_boundary_nodes,"
              "u_max_lb,E_2_profile,E_inf_profile,E_2_profile_full,"
              "E_inf_profile_full,max_abs_uy_over_umax,max_abs_uz_over_umax,"
              "relative_mass_error,raw_lattice_mass,raw_lattice_reference_mass,"
              "physical_weighted_mass,physical_weighted_reference_mass,"
              "physical_weighted_relative_mass_error,rho_mean_minus_rho0,"
              "rho_std_over_rho0,rho_min,rho_max,rho_streamwise_xlo,"
              "rho_streamwise_xhi,pressure_streamwise_xlo,"
              "pressure_streamwise_xhi,rho_streamwise_slope,"
              "pressure_streamwise_slope,rho_streamwise_linear_r2,"
              "pressure_driven,prescribed_state,pressure_scaling_code,"
              "re_effective,rho_in_target_lb,rho_out_target_lb,"
              "density_delta_target_lb,density_delta_expected_lb,"
              "density_delta_over_rho_ref,mach,"
              "max_abs_pressure_boundary_rho_error,"
              "rho_streamwise_delta_measured,"
              "rho_streamwise_delta_relative_error,"
              "max_abs_profile_error_all_x_over_umax,"
              "max_streamwise_ux_spread_over_umax\n";
        os << value.step << ',' << value.nx << ',' << value.ny << ','
           << value.nz << ',' << value.nIntervalsY << ','
           << value.onLatticeBoundaryNodes << ',' << value.uMaxLb << ','
           << value.e2Profile << ',' << value.eInfProfile << ','
           << value.e2ProfileFull << ',' << value.eInfProfileFull << ','
           << value.maxAbsUyOverUmax << ',' << value.maxAbsUzOverUmax << ','
           << value.relativeMassError << ',' << value.rawLatticeMass << ','
           << value.rawLatticeReferenceMass << ','
           << value.physicalWeightedMass << ','
           << value.physicalWeightedReferenceMass << ','
           << value.physicalWeightedRelativeMassError << ','
           << value.rhoMeanMinusRho0 << ',' << value.rhoStdOverRho0 << ','
           << value.rhoMin << ',' << value.rhoMax << ','
           << value.rhoStreamwiseXLo << ',' << value.rhoStreamwiseXHi << ','
           << value.pressureStreamwiseXLo << ','
           << value.pressureStreamwiseXHi << ','
           << value.rhoStreamwiseSlope << ','
           << value.pressureStreamwiseSlope << ','
           << value.rhoStreamwiseLinearR2 << ',' << value.pressureDriven << ','
           << value.prescribedState << ',' << value.pressureScalingCode << ','
           << value.reEffective << ',' << value.rhoInTargetLb << ','
           << value.rhoOutTargetLb << ',' << value.densityDeltaTargetLb << ','
           << value.densityDeltaExpectedLb << ','
           << value.densityDeltaOverRhoRef << ',' << value.mach << ','
           << value.maxAbsPressureBoundaryRhoError << ','
           << value.rhoStreamwiseDeltaMeasured << ','
           << value.rhoStreamwiseDeltaRelativeError << ','
           << value.maxAbsProfileErrorAllXOverUmax << ','
           << value.maxStreamwiseUxSpreadOverUmax << '\n';
    }

    void writeSummary(
        PoiseuilleProfileOutput const& value,
        runtime::SingleLevelFinalSummaryView const& steady) const
    {
        std::ostringstream os;
        os << poiseuille_profile_summary(value)
           << "\npoiseuille_profile_csv = " << profilePath_
           << "\npoiseuille_profile_metrics_csv = " << metricsPath_
           << "\npoiseuille_density_pressure_csv = " << densityPath_
           << "\npoiseuille_steady_reached = "
           << (steady.steadyReached ? "true" : "false")
           << "\npoiseuille_steady_reached_step = "
           << steady.steadyReachedStep
           << "\npoiseuille_final_relative_l2_delta_u = "
           << steady.lastSteadyResidual
           << "\npoiseuille_mass_gate = "
           << (cfg_.massAssert != 0 ? "assert" : "record_only") << '\n';
        output_.printAndLog(os.str());
    }

    ObservationConfig cfg_;
    config::ResolvedLbmConfig const& lbm_;
    amrex::Geometry const& geom_;
    amrex::BoxArray const& ba_;
    amrex::DistributionMapping const& dm_;
    std::string profilePath_;
    std::string metricsPath_;
    std::string densityPath_;
    Output& output_;
};

} // namespace lbm::cases::poiseuille_flow

#endif
