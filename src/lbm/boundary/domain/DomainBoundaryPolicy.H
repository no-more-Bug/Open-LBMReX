#ifndef LBM_BOUNDARY_DOMAIN_DOMAINBOUNDARYPOLICY_H
#define LBM_BOUNDARY_DOMAIN_DOMAINBOUNDARYPOLICY_H

#include "lbm/boundary/core/BoundarySpec.H"

#include <optional>
#include <string>
#include <vector>

namespace lbm::boundary {

void append_boundary_resolve_error(
    std::vector<std::string>& errors,
    std::string const& context,
    std::string const& message);

std::optional<BoundaryScheme> default_domain_scheme(BoundaryPhysical physical);

std::optional<BoundaryProfileKind> resolve_domain_profile(
    DomainBoundarySpec const& spec,
    std::string const& context,
    std::vector<std::string>& errors);

void validate_domain_profile_values(
    DomainBoundarySpec const& spec,
    BoundaryProfileKind profile,
    std::string const& context,
    std::vector<std::string>& errors);

} // namespace lbm::boundary

#endif
