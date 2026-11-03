#ifndef LBM_RUNTIME_SINGLELEVELDIAGNOSTICOUTPUT_H
#define LBM_RUNTIME_SINGLELEVELDIAGNOSTICOUTPUT_H

#include "runtime/SingleLevelDiagnosticRow.H"

#include <AMReX_ParallelDescriptor.H>

#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

namespace lbm::runtime {

inline std::string diagnostic_console_text(DiagnosticRow const& row)
{
    std::ostringstream os;
    os << "step=" << row.step
       << " time_lb=" << row.timeLb
       << " time_phys=" << row.timePhys
       << " l2_u=" << row.l2U
       << " relative_l2_u=" << row.relativeL2U
       << " linf_u=" << row.linfU
       << " l2_uz=" << row.l2Uz
       << " linf_uz=" << row.linfUz
       << " max_u=" << row.maxU
       << " rel_mass=" << row.relativeMassError << '\n';
    return os.str();
}

inline std::string diagnostic_log_text(DiagnosticRow const& row)
{
    std::ostringstream os;
    os << std::setprecision(12)
       << "step=" << row.step
       << " time_lb=" << row.timeLb
       << " time_phys=" << row.timePhys
       << " l2_u=" << row.l2U
       << " l2_u_exact=" << row.l2UExact
       << " relative_l2_u=" << row.relativeL2U
       << " linf_u=" << row.linfU
       << " l2_uz=" << row.l2Uz
       << " linf_uz=" << row.linfUz
       << " rel_mass=" << row.relativeMassError
       << " max_u=" << row.maxU
       << " max_rho_error=" << row.maxRhoError << '\n';
    return os.str();
}

inline void write_diagnostics_row(
    std::string const& path,
    DiagnosticRow const& row,
    bool resetFile)
{
    if (!amrex::ParallelDescriptor::IOProcessor()) {
        return;
    }
    std::ofstream os(path, resetFile ? std::ios::out : std::ios::app);
    os << std::setprecision(17);
    if (resetFile) {
        os << "step,time_lb,time_phys,mass,relative_mass_error,l2_ux,"
              "l2_uy,l2_uz,l2_u,l2_u_exact,relative_l2_u,linf_u,"
              "linf_uz,max_u,max_rho_error\n";
    }
    os << row.step << ','
       << row.timeLb << ','
       << row.timePhys << ','
       << row.mass << ','
       << row.relativeMassError << ','
       << row.l2Ux << ','
       << row.l2Uy << ','
       << row.l2Uz << ','
       << row.l2U << ','
       << row.l2UExact << ','
       << row.relativeL2U << ','
       << row.linfU << ','
       << row.linfUz << ','
       << row.maxU << ','
       << row.maxRhoError << '\n';
}

} // namespace lbm::runtime

#endif
