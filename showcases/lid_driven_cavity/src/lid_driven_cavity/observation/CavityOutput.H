#ifndef LBM_CASES_LIDDRIVENCAVITY_CAVITYOUTPUT_H
#define LBM_CASES_LIDDRIVENCAVITY_CAVITYOUTPUT_H

#include "lid_driven_cavity/observation/CavityCenterlineDiagnostics.H"
#include "lid_driven_cavity/observation/CavityValidationDiagnostics.H"

#include <AMReX_Geometry.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_REAL.H>
#include <AMReX_Utility.H>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>

namespace lbm::cases::lid_driven_cavity {

/** Concrete owner of cavity artifact paths and serialization schema. */
class CavityOutput {
public:
    std::string prepare(
        std::string runOutputDirectory,
        int writeCenterlines,
        int writeMetrics)
    {
        runOutputDirectory_ = std::move(runOutputDirectory);
        std::string error;
        if (amrex::ParallelDescriptor::IOProcessor()) {
            try {
                if (writeCenterlines != 0) {
                    std::filesystem::remove(centerlineUxPath());
                    std::filesystem::remove(centerlineUyPath());
                    std::filesystem::remove(centerlineMetadataPath());
                }
                if (writeMetrics != 0) {
                    std::filesystem::remove(validationMetricsPath());
                }
            } catch (std::filesystem::filesystem_error const& exception) {
                error = "Failed to prepare cavity output: ";
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

    void prepareCenterlineWrite() const
    {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            std::filesystem::create_directories(runOutputDirectory_);
        }
        amrex::ParallelDescriptor::Barrier();
    }

    std::string writeCenterlines(
        CavityCenterlineSamples const& samples,
        amrex::Geometry const& geometry,
        amrex::Real lidSpeedLb,
        int step,
        amrex::Real relativeL2DeltaU,
        amrex::Real steadyL2Tol,
        bool steadyReached,
        CavityDiagnosticGeometry const& diagnosticGeometry) const
    {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            const amrex::Real invLidSpeed = 1.0 / lidSpeedLb;
            {
                std::ofstream ux(centerlineUxPath(), std::ios::out);
                ux << std::setprecision(17);
                ux << "y_over_L,ux_lb,ux_over_Ulid\n";
                for (std::size_t n = 0; n < samples.uxLb.size(); ++n) {
                    ux << samples.yCoordinates[n] << ','
                       << samples.uxLb[n] << ','
                       << samples.uxLb[n] * invLidSpeed << '\n';
                }
            }
            {
                std::ofstream uy(centerlineUyPath(), std::ios::out);
                uy << std::setprecision(17);
                uy << "x_over_L,uy_lb,uy_over_Ulid\n";
                for (std::size_t n = 0; n < samples.uyLb.size(); ++n) {
                    uy << samples.xCoordinates[n] << ','
                       << samples.uyLb[n] << ','
                       << samples.uyLb[n] * invLidSpeed << '\n';
                }
            }
            {
                std::ofstream metadata(centerlineMetadataPath(), std::ios::out);
                metadata
                    << std::setprecision(17)
                    << "step = " << step << '\n'
                    << "relative_l2_delta_u = " << relativeL2DeltaU << '\n'
                    << "steady_l2_tol = " << steadyL2Tol << '\n'
                    << "steady_delta_step = 1\n"
                    << "steady_reached = "
                    << (steadyReached ? "true" : "false") << '\n'
                    << "x_center_index = " << samples.centerI << '\n'
                    << "y_center_index = " << samples.centerJ << '\n'
                    << "x_center_coordinate = "
                    << samples.centerXCoordinate << '\n'
                    << "y_center_coordinate = "
                    << samples.centerYCoordinate << '\n'
                    << "boundary_geometry = "
                    << (diagnosticGeometry.onLatticeBoundaryNodes != 0 ?
                            "on_lattice_valid_node" : "domain_face_halfway")
                    << '\n'
                    << "sampling = "
                    << (diagnosticGeometry.onLatticeBoundaryNodes != 0 ?
                            "direct_node" : "interpolated_cell_center")
                    << '\n'
                    << "interpolation = "
                    << (diagnosticGeometry.onLatticeBoundaryNodes != 0 ?
                            "false" : "true") << '\n'
                    << "endpoint_injection = "
                    << (diagnosticGeometry.onLatticeBoundaryNodes != 0 ?
                            "false" : "true") << '\n'
                    << "valid_nodes_x = "
                    << diagnosticGeometry.validNodes[0] << '\n'
                    << "valid_nodes_y = "
                    << diagnosticGeometry.validNodes[1] << '\n'
                    << "lattice_intervals_x = "
                    << diagnosticGeometry.latticeIntervals[0] << '\n'
                    << "lattice_intervals_y = "
                    << diagnosticGeometry.latticeIntervals[1] << '\n'
                    << "physical_lo_x = "
                    << diagnosticGeometry.physicalLo[0] << '\n'
                    << "physical_hi_x = "
                    << diagnosticGeometry.physicalHi[0] << '\n'
                    << "physical_lo_y = "
                    << diagnosticGeometry.physicalLo[1] << '\n'
                    << "physical_hi_y = "
                    << diagnosticGeometry.physicalHi[1] << '\n'
                    << "envelope_lo_x = " << geometry.ProbLo(0) << '\n'
                    << "envelope_hi_x = " << geometry.ProbHi(0) << '\n'
                    << "envelope_lo_y = " << geometry.ProbLo(1) << '\n'
                    << "envelope_hi_y = " << geometry.ProbHi(1) << '\n'
                    << "z_policy = average\n"
                    << "lid_speed_lb = " << lidSpeedLb << '\n';
            }
        }
        amrex::ParallelDescriptor::Barrier();

        std::ostringstream os;
        os << "cavity_centerline_x0p5_ux_csv = "
           << centerlineUxPath() << "\n"
           << "cavity_centerline_y0p5_uy_csv = "
           << centerlineUyPath() << "\n"
           << "cavity_centerline_metadata = "
           << centerlineMetadataPath() << "\n";
        return os.str();
    }

    std::string writeValidationMetrics(
        CavityValidationMetrics const& metrics) const
    {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            std::ofstream os(validationMetricsPath(), std::ios::out);
            os << std::setprecision(17);
            os << "step,raw_lattice_mass,raw_lattice_reference_mass,"
                  "relative_mass_error,physical_weighted_mass,"
                  "physical_weighted_reference_mass,"
                  "physical_weighted_relative_mass_error,rho_min,rho_max,"
                  "max_static_wall_velocity_error_over_Ulid,"
                  "max_moving_lid_velocity_error_over_Ulid,"
                  "max_moving_lid_tangential_error_over_Ulid,"
                  "max_moving_lid_normal_velocity_over_Ulid,"
                  "max_abs_uz_over_Ulid\n";
            os << metrics.step << ','
               << metrics.rawLatticeMass << ','
               << metrics.rawLatticeReferenceMass << ','
               << metrics.relativeMassError << ','
               << metrics.physicalWeightedMass << ','
               << metrics.physicalWeightedReferenceMass << ','
               << metrics.physicalWeightedRelativeMassError << ','
               << metrics.rhoMin << ','
               << metrics.rhoMax << ','
               << metrics.maxStaticWallVelocityErrorOverUlid << ','
               << metrics.maxMovingLidVelocityErrorOverUlid << ','
               << metrics.maxMovingLidTangentialErrorOverUlid << ','
               << metrics.maxMovingLidNormalVelocityOverUlid << ','
               << metrics.maxAbsUzOverUlid << '\n';
        }
        std::ostringstream os;
        os << cavity_validation_metrics_summary(metrics) << "\n"
           << "cavity_validation_metrics_csv = "
           << validationMetricsPath() << "\n";
        return os.str();
    }

private:
    std::string centerlineUxPath() const
    {
        return runOutputDirectory_ + "/cavity_centerline_x0p5_ux.csv";
    }
    std::string centerlineUyPath() const
    {
        return runOutputDirectory_ + "/cavity_centerline_y0p5_uy.csv";
    }
    std::string centerlineMetadataPath() const
    {
        return runOutputDirectory_ + "/cavity_centerline_metadata.txt";
    }
    std::string validationMetricsPath() const
    {
        return runOutputDirectory_ + "/cavity_validation_metrics.csv";
    }

    std::string runOutputDirectory_;
};

} // namespace lbm::cases::lid_driven_cavity

#endif
