#ifndef LBM_BOUNDARY_DOMAIN_LBMPOSTSTREAMINGBOUNDARY_H
#define LBM_BOUNDARY_DOMAIN_LBMPOSTSTREAMINGBOUNDARY_H

#include "lbm/boundary/core/ResolvedBoundary.H"

#include <AMReX_Array.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_REAL.H>

#include <array>
#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace lbm::boundary {

/**
 * @brief Cell-centered geometry of one axis with on-lattice boundary nodes.
 *
 * envelopeLo/envelopeHi retain the Geometry ProbLo/ProbHi metadata.
 * physicalLo/physicalHi are the centers of the actual outer valid indices,
 * including for a Geometry whose domain small end is nonzero. Thus
 * validNodes = latticeIntervals + 1 for an active Guo axis.
 */
struct OnLatticeAxisGeometry {
    bool active = false;
    int validNodes = 0;
    int latticeIntervals = 0;
    amrex::Real dx = 0.0;
    amrex::Real envelopeLo = 0.0;
    amrex::Real envelopeHi = 0.0;
    amrex::Real physicalLo = 0.0;
    amrex::Real physicalHi = 0.0;
};

/**
 * @brief One disjoint face, edge, or corner region of the outer valid shell.
 *
 * inward contains one signed cell step for every touched physical axis; adding
 * it to a boundary node yields the first strictly interior donor, including a
 * diagonal donor at edges/corners. effectivePhysical/profile implement the
 * deterministic wall-wins rule compiled during setup.
 */
struct PostStreamBoundaryRegion {
    amrex::Box box;
    amrex::IntVect inward{AMREX_D_DECL(0, 0, 0)};
    std::array<DomainSide, 3> touchedSides{
        DomainSide::XLo, DomainSide::XLo, DomainSide::XLo};
    int touchedSideCount = 0;
    DomainSide effectiveSide = DomainSide::XLo;
    BoundaryPhysical effectivePhysical = BoundaryPhysical::NoSlipWall;
    BoundaryProfileKind effectiveProfile = BoundaryProfileKind::None;
};

/**
 * @brief Immutable setup-time execution plan for domain boundary stages.
 *
 * regions partition only post-stream valid nodes; pre-stream faces are tracked
 * by the stage flags and executed by PhysBC rather than represented as regions.
 * The plan contains global index-space geometry and is identical on all ranks.
 */
struct PostStreamingBoundaryPlan {
    std::vector<PostStreamBoundaryRegion> regions;
    amrex::Array<OnLatticeAxisGeometry, AMREX_SPACEDIM> axisGeometry{};
    bool hasPreStreamingFaces = false;
    bool hasPostStreamingFaces = false;
};

/**
 * @brief Return whether any post-stream donor crosses its boundary-owner FAB.
 *
 * The decision uses the replicated global BoxArray, so every MPI rank reaches
 * the same collective/no-collective branch.  Tile boundaries do not matter:
 * an Array4 obtained from an MFIter spans the complete owning FAB allocation.
 */
inline bool post_stream_donor_fill_boundary_required(
    PostStreamingBoundaryPlan const& plan,
    amrex::BoxArray const& boxArray) noexcept
{
    for (PostStreamBoundaryRegion const& region : plan.regions) {
        for (int boxIndex = 0; boxIndex < boxArray.size(); ++boxIndex) {
            amrex::Box donorBox = region.box & boxArray[boxIndex];
            if (!donorBox.ok()) {
                continue;
            }
            donorBox.shift(region.inward);
            if (!boxArray[boxIndex].contains(donorBox)) {
                return true;
            }
        }
    }
    return false;
}

/** @brief Compiled plan plus accumulated geometry validation failures. */
struct PostStreamingBoundaryPlanResult {
    PostStreamingBoundaryPlan plan;
    std::vector<std::string> errors;

    bool ok() const noexcept { return errors.empty(); }

    std::string error_message() const
    {
        std::ostringstream os;
        os << "Post-stream domain-boundary geometry validation failed:";
        for (std::string const& error : errors) {
            os << "\n  - " << error;
        }
        return os.str();
    }
};

/**
 * @brief Test whether one active plan axis represents a supplied reference length.
 *
 * The lattice comparison uses the compiled number of physical intervals, not
 * the number of valid cells; the physical comparison uses the compiled
 * boundary coordinates, not the AMReX envelope. This query is host-only,
 * read-only, and enters no collective.
 */
inline bool on_lattice_reference_matches_axis(
    PostStreamingBoundaryPlan const& plan,
    int axis,
    amrex::Real lengthRefLb,
    amrex::Real lengthRefPhys) noexcept
{
    if (axis < 0 || axis >= AMREX_SPACEDIM || !plan.axisGeometry[axis].active) {
        return false;
    }
    OnLatticeAxisGeometry const& geometry = plan.axisGeometry[axis];
    const amrex::Real physicalLength = geometry.physicalHi - geometry.physicalLo;
    const amrex::Real lbScale = std::max(
        std::max(std::abs(lengthRefLb),
                 static_cast<amrex::Real>(geometry.latticeIntervals)),
        amrex::Real(1.0));
    const amrex::Real physicalScale = std::max(
        std::max(std::abs(lengthRefPhys), std::abs(physicalLength)),
        amrex::Real(1.0));
    const amrex::Real tolerance =
        64.0 * std::numeric_limits<amrex::Real>::epsilon();
    return std::abs(lengthRefLb - geometry.latticeIntervals) <= tolerance * lbScale &&
           std::abs(lengthRefPhys - physicalLength) <= tolerance * physicalScale;
}

/**
 * @brief Return the first active axis matching both lattice and physical lengths.
 *
 * Axis order is deterministic on every rank because `axisGeometry` is compiled
 * from the same global domain plan. A negative result means that no active
 * on-lattice axis matches; callers own any application-level validation policy.
 */
inline int matching_on_lattice_reference_axis(
    PostStreamingBoundaryPlan const& plan,
    amrex::Real lengthRefLb,
    amrex::Real lengthRefPhys) noexcept
{
    for (int axis = 0; axis < AMREX_SPACEDIM; ++axis) {
        if (on_lattice_reference_matches_axis(
                plan, axis, lengthRefLb, lengthRefPhys)) {
            return axis;
        }
    }
    return -1;
}

inline ResolvedDomainBoundary const* find_domain_boundary(
    std::vector<ResolvedDomainBoundary> const& boundaries,
    DomainSide side) noexcept
{
    for (ResolvedDomainBoundary const& boundary : boundaries) {
        if (boundary.side == side) {
            return &boundary;
        }
    }
    return nullptr;
}

namespace detail {

struct BoundaryAxisPart {
    int lo = 0;
    int hi = -1;
    int sideIndex = -1;
};

inline std::vector<BoundaryAxisPart> make_axis_parts(
    amrex::Box const& domain,
    int axis,
    bool lowActive,
    bool highActive)
{
    std::vector<BoundaryAxisPart> parts;
    const int lo = domain.smallEnd(axis);
    const int hi = domain.bigEnd(axis);
    if (lowActive) {
        parts.push_back(BoundaryAxisPart{lo, lo, 2 * axis});
    }
    const int interiorLo = lo + static_cast<int>(lowActive);
    const int interiorHi = hi - static_cast<int>(highActive);
    if (interiorLo <= interiorHi) {
        parts.push_back(BoundaryAxisPart{interiorLo, interiorHi, -1});
    }
    if (highActive) {
        parts.push_back(BoundaryAxisPart{hi, hi, 2 * axis + 1});
    }
    return parts;
}

} // namespace detail

/**
 * @brief Compile resolved domain faces into disjoint on-lattice execution regions.
 *
 * Active Guo axes must have both opposite faces, at least three valid nodes, and
 * descriptor support. The Cartesian face/interior partition produces each
 * face, edge, and corner exactly once. At intersections a no-slip wall wins;
 * otherwise only an unambiguous single non-wall condition is accepted. Every
 * shifted donor box is checked to lie strictly inside the physical shell.
 *
 * @note This is host-only setup work over global Geometry index space. No
 *       MFIter traversal, communication, or population write occurs here.
 */
template <class Descriptor>
PostStreamingBoundaryPlanResult compile_post_streaming_boundary_plan(
    amrex::Geometry const& geom,
    std::vector<ResolvedDomainBoundary> const& boundaries)
{
    PostStreamingBoundaryPlanResult result;
    amrex::Box const& domain = geom.Domain();

    amrex::Array<bool, DomainSideCount> postStreamSide{};
    for (ResolvedDomainBoundary const& boundary : boundaries) {
        const int sideIndex = domain_side_index(boundary.side);
        if (boundary.stage == BoundaryExecutionStage::PreStreamingPhysicalGhost) {
            result.plan.hasPreStreamingFaces = true;
        } else {
            result.plan.hasPostStreamingFaces = true;
            postStreamSide[sideIndex] = true;
        }
    }

    for (int axis = 0; axis < AMREX_SPACEDIM; ++axis) {
        const bool lowActive = postStreamSide[2 * axis];
        const bool highActive = postStreamSide[2 * axis + 1];
        OnLatticeAxisGeometry& axisGeometry = result.plan.axisGeometry[axis];
        axisGeometry.active = lowActive || highActive;
        axisGeometry.validNodes = domain.length(axis);
        axisGeometry.latticeIntervals = domain.length(axis) - 1;
        axisGeometry.dx = geom.CellSize(axis);
        axisGeometry.envelopeLo = geom.ProbLo(axis);
        axisGeometry.envelopeHi = geom.ProbHi(axis);
        axisGeometry.physicalLo =
            geom.ProbLo(axis) +
            (static_cast<amrex::Real>(domain.smallEnd(axis)) + 0.5) *
                axisGeometry.dx;
        axisGeometry.physicalHi =
            geom.ProbLo(axis) +
            (static_cast<amrex::Real>(domain.bigEnd(axis)) + 0.5) *
                axisGeometry.dx;

        if (!axisGeometry.active) {
            continue;
        }
        if (lowActive != highActive) {
            result.errors.push_back(
                "post-stream Guo geometry on axis '" +
                std::string(axis_to_string(axis)) +
                "' requires both opposite physical faces to use the same "
                "valid-node execution stage");
            continue;
        }
        if (axis >= Descriptor::d) {
            result.errors.push_back(
                std::string(Descriptor::name()) +
                " cannot compile a post-stream boundary on axis " +
                std::string(axis_to_string(axis)));
            continue;
        }
        if (domain.length(axis) < 3) {
            result.errors.push_back(
                "post-stream Guo axis '" + std::string(axis_to_string(axis)) +
                "' requires at least 3 valid nodes");
            continue;
        }
    }

    if (!result.errors.empty() || !result.plan.hasPostStreamingFaces) {
        return result;
    }

    amrex::Array<std::vector<detail::BoundaryAxisPart>, AMREX_SPACEDIM> parts;
    for (int axis = 0; axis < AMREX_SPACEDIM; ++axis) {
        parts[axis] = detail::make_axis_parts(
            domain, axis, postStreamSide[2 * axis], postStreamSide[2 * axis + 1]);
    }

    for (detail::BoundaryAxisPart const& x : parts[0]) {
        for (detail::BoundaryAxisPart const& y : parts[1]) {
            for (detail::BoundaryAxisPart const& z : parts[2]) {
                std::array<detail::BoundaryAxisPart, 3> const selected{x, y, z};
                PostStreamBoundaryRegion region;
                amrex::IntVect small(AMREX_D_DECL(x.lo, y.lo, z.lo));
                amrex::IntVect big(AMREX_D_DECL(x.hi, y.hi, z.hi));
                region.box = amrex::Box(small, big);

                ResolvedDomainBoundary const* effective = nullptr;
                for (int axis = 0; axis < AMREX_SPACEDIM; ++axis) {
                    if (selected[axis].sideIndex < 0) {
                        continue;
                    }
                    DomainSide const side = AllDomainSides[selected[axis].sideIndex];
                    region.touchedSides[region.touchedSideCount++] = side;
                    region.inward[axis] = domain_side_is_low(side) ? 1 : -1;
                    ResolvedDomainBoundary const* candidate =
                        find_domain_boundary(boundaries, side);
                    if (candidate != nullptr &&
                        candidate->physical == BoundaryPhysical::NoSlipWall) {
                        effective = candidate;
                    } else if (effective == nullptr) {
                        effective = candidate;
                    }
                }

                if (region.touchedSideCount == 0) {
                    continue;
                }
                if (effective == nullptr) {
                    result.errors.push_back(
                        "compiled region has no resolved effective domain side");
                    continue;
                }
                if (region.touchedSideCount > 1 &&
                    effective->physical != BoundaryPhysical::NoSlipWall) {
                    result.errors.push_back(
                        "post-stream edge/corner without a no-slip wall has no "
                        "unambiguous effective physical boundary");
                    continue;
                }

                region.effectiveSide = effective->side;
                region.effectivePhysical = effective->physical;
                region.effectiveProfile = effective->profile;

                amrex::Box donorBox = region.box;
                donorBox.shift(region.inward);
                bool donorIsStrictlyInterior = true;
                for (int axis = 0; axis < AMREX_SPACEDIM; ++axis) {
                    if (result.plan.axisGeometry[axis].active &&
                        (donorBox.smallEnd(axis) <= domain.smallEnd(axis) ||
                         donorBox.bigEnd(axis) >= domain.bigEnd(axis))) {
                        donorIsStrictlyInterior = false;
                        break;
                    }
                }
                if (!donorIsStrictlyInterior) {
                    result.errors.push_back(
                        "first-interior donor box for a post-stream region is not "
                        "strictly inside the physical boundary shell");
                    continue;
                }
                result.plan.regions.push_back(region);
            }
        }
    }

    return result;
}

} // namespace lbm::boundary

#endif
