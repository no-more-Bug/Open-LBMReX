#ifndef LBM_TESTS_HOSTFIELDSNAPSHOT_H
#define LBM_TESTS_HOSTFIELDSNAPSHOT_H

#include <AMReX_Arena.H>
#include <AMReX_Gpu.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>

namespace lbm::tests {

// Copy observed values, including ghosts, without changing the host oracle.
inline amrex::MultiFab host_field_snapshot(amrex::MultiFab const& source)
{
    amrex::MultiFab result(
        source.boxArray(), source.DistributionMap(), source.nComp(),
        source.nGrowVect(), amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
    amrex::Gpu::streamSynchronize();
    for (amrex::MFIter mfi(source, false); mfi.isValid(); ++mfi) {
        amrex::Gpu::dtoh_memcpy(
            result[mfi].dataPtr(), source[mfi].dataPtr(),
            source[mfi].size() * sizeof(amrex::Real));
    }
    return result;
}

} // namespace lbm::tests

#endif
