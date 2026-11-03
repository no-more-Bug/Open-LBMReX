#ifndef LBM_BOUNDARY_METADATA_INTERNALCUTLINKCOMPILER_HH
#define LBM_BOUNDARY_METADATA_INTERNALCUTLINKCOMPILER_HH

#include <AMReX_Arena.H>
#include <AMReX_BoxIterator.H>
#include <AMReX_Gpu.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>

#include <algorithm>
#include <cmath>
#include <sstream>

namespace lbm::boundary {

template <class Descriptor>
bool InternalCutLinkCompiler<Descriptor>::build(
    InternalGeometryQueryRegistry const& query,
    CenterPhaseField<Descriptor> const& phaseMetadata,
    amrex::Geometry const& geom,
    amrex::BoxArray const& coveredCells,
    std::string& error)
{
    records_.clear();
    error.clear();
    auto const& phase = phaseMetadata.phase().multifab();
    auto const& provenance = phaseMetadata.surfaceProvenance().multifab();
    if (!phaseMetadata.built() || query.empty() ||
        phase.nGrow() < Descriptor::vicinity ||
        provenance.nGrow() < Descriptor::vicinity ||
        phase.boxArray() != provenance.boxArray() ||
        phase.DistributionMap() != provenance.DistributionMap()) {
        error = "cut-link compiler received an invalid phase/layout/query";
        return false;
    }

    amrex::MFInfo hostInfo;
    hostInfo.SetArena(amrex::The_Pinned_Arena());
    amrex::iMultiFab hostPhase(
        phase.boxArray(), phase.DistributionMap(), 1,
        Descriptor::vicinity, hostInfo);
    amrex::iMultiFab hostProvenance(
        provenance.boxArray(), provenance.DistributionMap(), 1,
        Descriptor::vicinity, hostInfo);
    amrex::iMultiFab::Copy(
        hostPhase, phase, 0, 0, 1, Descriptor::vicinity);
    amrex::iMultiFab::Copy(
        hostProvenance, provenance, 0, 0, 1, Descriptor::vicinity);
    amrex::Gpu::streamSynchronize();

    amrex::Long localErrors = 0;
    int ownershipFailure[7]{-1, 0, 0, 0, 0, 0, 0};
    auto const probLo = geom.ProbLoArray();
    auto const cellSize = geom.CellSizeArray();
    amrex::Box const& domain = geom.Domain();
    auto physicalPoint = [&](amrex::IntVect const& iv) {
        return amrex::RealArray{
            probLo[0] + (static_cast<amrex::Real>(iv[0]) + 0.5) * cellSize[0],
            probLo[1] + (static_cast<amrex::Real>(iv[1]) + 0.5) * cellSize[1],
            probLo[2] + (static_cast<amrex::Real>(iv[2]) + 0.5) * cellSize[2]};
    };
    auto outsideNonPeriodic = [&](amrex::IntVect const& iv) {
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            if (!geom.isPeriodic(d) &&
                (iv[d] < domain.smallEnd(d) || iv[d] > domain.bigEnd(d))) {
                return true;
            }
        }
        return false;
    };
    auto effectiveCell = [&](amrex::IntVect cell) {
        for (int axis = 0; axis < AMREX_SPACEDIM; ++axis) {
            if (geom.isPeriodic(axis)) {
                if (cell[axis] < domain.smallEnd(axis)) {
                    cell[axis] += domain.length(axis);
                } else if (cell[axis] > domain.bigEnd(axis)) {
                    cell[axis] -= domain.length(axis);
                }
            }
        }
        return phase.boxArray().contains(cell) && !coveredCells.contains(cell);
    };

    for (amrex::MFIter mfi(hostPhase); mfi.isValid(); ++mfi) {
        const int ownerGrid = mfi.index();
        auto const phaseArray = hostPhase.const_array(mfi);
        auto const tokenArray = hostProvenance.const_array(mfi);
        for (amrex::BoxIterator bit(mfi.validbox()); bit.ok(); ++bit) {
            amrex::IntVect const& fluidCell = bit();
            if (coveredCells.contains(fluidCell)) {
                continue;
            }
            const int fluidPhase = phaseArray(fluidCell);
            const int fluidToken = tokenArray(fluidCell);
            if (!is_center_phase_code(fluidPhase) || fluidToken < 0 ||
                (fluidToken > 0 &&
                 fluidPhase != center_phase_code(CenterPhase::Fluid))) {
                ++localErrors;
                continue;
            }
            if (fluidPhase != center_phase_code(CenterPhase::Fluid)) {
                continue;
            }
            GeometrySurfaceProvenance ownerProvenance;
            if (fluidToken > 0 &&
                query.decodeSurfaceProvenance(
                    fluidToken, ownerProvenance) !=
                    GeometryQueryStatus::Success) {
                ++localErrors;
                continue;
            }

            for (int population = 1; population < Descriptor::q; ++population) {
                amrex::IntVect solidCell = fluidCell;
                solidCell[0] += Descriptor::cx(population);
                solidCell[1] += Descriptor::cy(population);
                solidCell[2] += Descriptor::cz(population);
                const int neighborPhase = phaseArray(solidCell);
                const int neighborToken = tokenArray(solidCell);
                if (!is_center_phase_code(neighborPhase)) {
                    if (!outsideNonPeriodic(solidCell)) {
                        ++localErrors;
                    }
                    continue;
                }
                if (neighborToken < 0 ||
                    (neighborToken > 0 &&
                     neighborPhase != center_phase_code(CenterPhase::Fluid))) {
                    ++localErrors;
                    continue;
                }
                if (neighborPhase == center_phase_code(CenterPhase::Fluid)) {
                    if (population < Descriptor::opposite(population) &&
                        query.segmentTopology(
                            physicalPoint(fluidCell),
                            physicalPoint(solidCell),
                            fluidToken > 0,
                            neighborToken > 0,
                            geom) !=
                            GeometryIntersectionStatus::NoHit) {
                        ++localErrors;
                    }
                    continue;
                }

                LinkIntersectionRequest request;
                request.fluidPoint = physicalPoint(fluidCell);
                request.solidPoint = physicalPoint(solidCell);
                request.surfaceCoincidentFluid = fluidToken > 0;
                if (request.surfaceCoincidentFluid) {
                    request.boundaryId = ownerProvenance.boundaryId;
                    request.objectId = ownerProvenance.objectId;
                }
                LinkIntersectionResult const hit =
                    query.intersectLink(request, geom);
                if (hit.status != GeometryIntersectionStatus::UniqueHit) {
                    ++localErrors;
                    continue;
                }
                if (!effectiveCell(solidCell)) {
                    ++localErrors;
                    if (ownershipFailure[0] < 0) {
                        ownershipFailure[0] = hit.provenance.boundaryId.value;
                        for (int axis = 0; axis < 3; ++axis) {
                            ownershipFailure[1 + axis] = fluidCell[axis];
                            ownershipFailure[4 + axis] = solidCell[axis];
                        }
                    }
                    continue;
                }

                amrex::IntVect nextFluidCell = fluidCell;
                nextFluidCell[0] -= Descriptor::cx(population);
                nextFluidCell[1] -= Descriptor::cy(population);
                nextFluidCell[2] -= Descriptor::cz(population);
                if (outsideNonPeriodic(nextFluidCell) ||
                    !phaseArray.contains(nextFluidCell) ||
                    !tokenArray.contains(nextFluidCell) ||
                    phaseArray(nextFluidCell) !=
                        center_phase_code(CenterPhase::Fluid) ||
                    tokenArray(nextFluidCell) != 0) {
                    ++localErrors;
                    continue;
                }

                CanonicalCutLinkRecord record;
                record.ownerGrid = ownerGrid;
                record.fluidCell = {
                    fluidCell[0], fluidCell[1], fluidCell[2]};
                record.solidCell = {
                    solidCell[0], solidCell[1], solidCell[2]};
                record.population = population;
                record.oppositePopulation = Descriptor::opposite(population);
                record.boundaryId = hit.provenance.boundaryId.value;
                record.objectId = hit.provenance.objectId.value;
                record.qFromFluid = hit.q;
                for (int d = 0; d < 3; ++d) {
                    record.intersection[d] = hit.intersection[d];
                    record.normalSolidToFluid[d] =
                        hit.normalSolidToFluid[d];
                }
                record.surfaceCoincidentFluid = fluidToken > 0;
                records_.push_back(record);
            }
        }
    }

    std::sort(records_.begin(), records_.end(), canonical_cut_link_key_less);
    amrex::ParallelDescriptor::ReduceLongSum(localErrors);
    if (localErrors != 0) {
        error = "internal cut-link compilation failed exact geometry/ownership checks";
        int owner = ownershipFailure[0] >= 0 ? amrex::ParallelDescriptor::MyProc() : amrex::ParallelDescriptor::NProcs();
        amrex::ParallelDescriptor::ReduceIntMin(owner);
        if (owner < amrex::ParallelDescriptor::NProcs()) {
            amrex::ParallelDescriptor::Bcast(ownershipFailure, 7, owner);
            std::ostringstream detail;
            detail << ": BoundaryId=" << ownershipFailure[0] << " crosses effective levels; fluid=("
                   << ownershipFailure[1] << ',' << ownershipFailure[2] << ',' << ownershipFailure[3]
                   << "), solid=(" << ownershipFailure[4] << ',' << ownershipFailure[5] << ','
                   << ownershipFailure[6] << ')';
            error += detail.str();
        }
        records_.clear();
        return false;
    }
    return true;
}

} // namespace lbm::boundary

#endif
