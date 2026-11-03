#ifndef LBM_BOUNDARY_INTERNAL_EBGEOMETRYBACKEND_H
#define LBM_BOUNDARY_INTERNAL_EBGEOMETRYBACKEND_H

#include "lbm/boundary/internal/InternalGeometryQuery.H"

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_EBFabFactory.H>
#include <AMReX_Geometry.H>

#include <memory>
#include <string>

namespace lbm::boundary {

/** @brief Exact owner of one setup-time EB2 IndexSpace and basic factory. */
class EbGeometryBackend {
public:
    EbGeometryBackend() = default;
    ~EbGeometryBackend();

    EbGeometryBackend(EbGeometryBackend const&) = delete;
    EbGeometryBackend(EbGeometryBackend&&) = delete;
    EbGeometryBackend& operator=(EbGeometryBackend const&) = delete;
    EbGeometryBackend& operator=(EbGeometryBackend&&) = delete;

    bool define(
        InternalGeometryQueryRegistry const& query,
        amrex::Geometry const& geom,
        amrex::BoxArray const& ba,
        amrex::DistributionMapping const& dm,
        std::string& error);
    void reset() noexcept;

    [[nodiscard]] amrex::EBFArrayBoxFactory const* factory() const noexcept
    {
        return factory_.get();
    }
private:
    void resetBuiltState() noexcept;

    amrex::EB2::IndexSpace const* indexSpace_ = nullptr;
    std::unique_ptr<amrex::EBFArrayBoxFactory> factory_;
};

} // namespace lbm::boundary

#endif
