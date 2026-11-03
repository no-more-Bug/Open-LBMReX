#ifndef LBM_BOUNDARY_CORE_BOUNDARYDIAGNOSTICS_H
#define LBM_BOUNDARY_CORE_BOUNDARYDIAGNOSTICS_H

#include "lbm/boundary/core/ResolvedBoundary.H"
#include "lbm/boundary/internal/InternalBoundarySchemeTraits.H"

#include <AMReX_Geometry.H>
#include <AMReX_Print.H>

#include <iomanip>
#include <sstream>
#include <string>

namespace lbm::boundary {

inline std::string format_resolved_boundary_table(
    ResolvedBoundarySet const& resolved,
    amrex::Geometry const& geom)
{
    std::ostringstream os;
    os << "Resolved boundaries:\n";
    os << "  " << std::left
       << std::setw(4) << "id"
       << std::setw(10) << "source"
       << std::setw(28) << "name"
       << std::setw(20) << "location/geometry"
       << std::setw(18) << "physical"
       << std::setw(31) << "scheme"
       << std::setw(28) << "stage"
       << std::setw(12) << "profile"
       << "tag\n";

    for (ResolvedDomainBoundary const& b : resolved.domain) {
        const std::string scheme = std::string(to_string(b.scheme)) +
            "(" + std::string(to_string(b.schemeOrigin)) + ")";
        os << "  " << std::left
           << std::setw(4) << b.boundaryId.value
           << std::setw(10) << "domain"
           << std::setw(28) << b.name
           << std::setw(20) << to_string(b.side)
           << std::setw(18) << to_string(b.physical)
           << std::setw(31) << scheme
           << std::setw(28) << to_string(b.stage)
           << std::setw(12) << to_string(b.profile)
           << b.tag << "\n";
        const int axis = domain_side_axis(b.side);
        if (b.stage == BoundaryExecutionStage::PostStreamingValidNode) {
            const amrex::Real dx = geom.CellSize(axis);
            amrex::Box const& domain = geom.Domain();
            const amrex::Real physicalLo =
                geom.ProbLo(axis) +
                (static_cast<amrex::Real>(domain.smallEnd(axis)) + 0.5) *
                    dx;
            const amrex::Real physicalHi =
                geom.ProbLo(axis) +
                (static_cast<amrex::Real>(domain.bigEnd(axis)) + 0.5) *
                    dx;
            os << "      geometry=valid_node_center"
               << " valid_nodes=" << geom.Domain().length(axis)
               << " lattice_intervals=" << geom.Domain().length(axis) - 1
               << " physical_bounds=[" << physicalLo << ',' << physicalHi << ']'
               << " envelope_bounds=[" << geom.ProbLo(axis) << ','
               << geom.ProbHi(axis) << "]\n";
        } else {
            os << "      geometry=domain_face_halfway"
               << " envelope_bounds=[" << geom.ProbLo(axis) << ','
               << geom.ProbHi(axis) << "]\n";
        }
        if (b.profile == BoundaryProfileKind::Poiseuille) {
            os << "      poiseuille profile_axis=" << axis_to_string(b.profileAxis)
               << " u_max_lb=" << b.uMaxLb << "\n";
        }
        if (is_pressure_boundary(b.physical) ||
            is_prescribed_state_boundary(b.physical)) {
            os << "      constant density rho_lb=" << b.rhoLb << "\n";
        }
        if (is_prescribed_state_boundary(b.physical)) {
            os << "      prescribed velocity u_lb=["
               << b.prescribedULb[0] << ',' << b.prescribedULb[1] << ','
               << b.prescribedULb[2] << "]\n";
        }
    }

    for (ResolvedInternalBoundary const& b : resolved.internal) {
        const std::string scheme = std::string(to_string(b.scheme)) +
            "(" + std::string(to_string(b.schemeOrigin)) + ")";
        os << "  " << std::left
           << std::setw(4) << b.boundaryId.value
           << std::setw(10) << "internal"
           << std::setw(28) << b.name
           << std::setw(20) << b.geometry
           << std::setw(18) << to_string(b.physical)
           << std::setw(31) << scheme
           << std::setw(28) << "post_stream_valid_node"
           << std::setw(12) << to_string(b.profile)
           << b.tag << "\n";
        auto const traits = internal_boundary_scheme_traits(b.scheme);
        os << "      boundary_id=" << b.boundaryId.value
           << " object_id=" << b.objectId.value
           << " object_name=" << b.objectName
           << " anchor=fluid_node";
        if (traits.has_value()) {
            os << " half_link_policy=" << to_string(traits->halfLinkPolicy);
        }
        os << "\n";
    }

    if (!resolved.internalObjects.empty()) {
        os << "Internal object registry (disjoint/separated only):\n";
        for (ResolvedInternalObject const& object : resolved.internalObjects) {
            os << "  object_id=" << object.objectId.value
               << " name=" << object.name
               << " boundary_ids=[";
            for (int n = 0; n < object.boundaryCount; ++n) {
                if (n != 0) {
                    os << ',';
                }
                os << resolved.internalObjectBoundaries[
                    static_cast<std::size_t>(object.boundaryOffset + n)].value;
            }
            os << "]\n";
        }
    }

    return os.str();
}

inline void print_boundary_diagnostics(
    ResolvedBoundarySet const& resolved,
    amrex::Geometry const& geom)
{
    amrex::Print() << format_resolved_boundary_table(resolved, geom);
    for (std::string const& warning : resolved.warnings) {
        amrex::Print() << "Boundary warning: " << warning << "\n";
    }
}

} // namespace lbm::boundary

#endif
