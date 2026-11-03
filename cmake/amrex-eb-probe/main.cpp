#include <AMReX_EB2.H>
#include <AMReX_EBFabFactory.H>

#ifndef AMREX_USE_EB
#error "AMReX package headers do not define AMREX_USE_EB"
#endif

int main()
{
    auto const* indexSpace = amrex::EB2::TopIndexSpaceIfPresent();
    return indexSpace == nullptr ? 0 : 1;
}
