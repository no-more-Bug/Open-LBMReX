#ifndef LBM_SOLVERS_BOUNDARYSTEPCONTEXT_H
#define LBM_SOLVERS_BOUNDARYSTEPCONTEXT_H

#include <AMReX_REAL.H>

#include <utility>

namespace lbm::solvers {

/**
 * @brief Select physical boundary times for one already scheduled advance.
 *
 * HBB pre-stream samples the source physical time; Guo/internal post-stream
 * receives the target physical time. The caller supplies its prepared physical
 * timestep. This context neither advances a clock nor performs boundary work.
 */
class BoundaryStepContext {
public:
    BoundaryStepContext(amrex::Real sourcePhysicalTime, amrex::Real dtPhys)
        : sourcePhysicalTime_(sourcePhysicalTime),
          targetPhysicalTime_(sourcePhysicalTime + dtPhys)
    {}

    /** @brief Invoke pre-stream at the source physical time. */
    template <class F>
    decltype(auto) runPreStreaming(F&& operation) const
    {
        return std::forward<F>(operation)(sourcePhysicalTime_);
    }

    /** @brief Invoke post-stream at the target physical time. */
    template <class F>
    decltype(auto) runPostStreaming(F&& operation) const
    {
        return std::forward<F>(operation)(targetPhysicalTime_);
    }

private:
    amrex::Real sourcePhysicalTime_;
    amrex::Real targetPhysicalTime_;
};

} // namespace lbm::solvers

#endif
