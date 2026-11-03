#ifndef LBM_BOUNDARY_CORE_BOUNDARYRESOLVERESULT_H
#define LBM_BOUNDARY_CORE_BOUNDARYRESOLVERESULT_H

#include "lbm/boundary/core/ResolvedBoundary.H"

#include <sstream>
#include <string>
#include <vector>

namespace lbm::boundary {

/** Canonical resolution result with accumulated semantic errors. */
struct BoundaryResolveResult {
    ResolvedBoundarySet resolved;
    std::vector<std::string> errors;

    bool ok() const noexcept { return errors.empty(); }

    std::string error_message() const
    {
        std::ostringstream os;
        os << "Boundary specification validation failed:";
        for (std::string const& error : errors) {
            os << "\n  - " << error;
        }
        return os.str();
    }
};

} // namespace lbm::boundary

#endif
