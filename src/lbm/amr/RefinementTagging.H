#ifndef LBM_AMR_REFINEMENTTAGGING_H
#define LBM_AMR_REFINEMENTTAGGING_H

#include "lbm/fields/LatticeField.H"
#include "lbm/amr/RefinementConstraints.H"
#include <AMReX_TagBox.H>
#include <AMReX_iMultiFab.H>

namespace lbm::amr {

struct StaticTagging {};

/** Borrowed synchronized valid-cell state, valid only during the tagging call.
 * A custom rule composes its matching equilibrium evaluator and reads populations
 * and uLb from this same cell. velocityLb is present only for neighborhood rules.
 */
template<class Descriptor>
struct TaggingCell {
    using DescriptorType = Descriptor;
    ConstPopulationCell populations;
    amrex::Real rho;
    amrex::GpuArray<amrex::Real, 3> uLb;
    amrex::GpuArray<amrex::Real, 3> positionPhys;
    amrex::GpuArray<amrex::Real, 3> inverseCellSizePhys;
    amrex::Array4<amrex::Real const> velocityLb;
    int level;
    amrex::Real physicalTime;
    bool coveredByFine;
};

template<class Descriptor, class Rule>
void tag_valid_cells(amrex::MultiFab const& populations, amrex::MultiFab const* velocity,
                     amrex::iMultiFab const& covered, amrex::Geometry const& geometry,
                     int level, amrex::Real time, Rule rule, amrex::TagBoxArray& tags,
                     RefinementConstraints const* constraints = nullptr,
                     amrex::TagBoxArray* optionalTags = nullptr);

// A dynamic strategy supplies needsVelocity (compile-time bool), validate(maxLevel)
// and rule(level). The returned device-safe rule maps TaggingCell to a raw bool tag.
// Buffering and hierarchy changes remain the owner's responsibility.

} // namespace lbm::amr
#include "lbm/amr/RefinementTagging.HH"
#endif
