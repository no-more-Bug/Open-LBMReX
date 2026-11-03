#include "lbm/boundary/internal/EbGeometryBackend.H"

#include "lbm/boundary/internal/InternalGeometryLayout.H"

#include <AMReX_EB2.H>
#include <AMReX_EB2_GeometryShop.H>
#include <AMReX_EBSupport.H>
#include <AMReX_GpuContainers.H>

#include <memory>
#include <type_traits>
#include <vector>

namespace lbm::boundary {
namespace {

struct RuntimeCylinderStorage {
    explicit RuntimeCylinderStorage(
        std::vector<amrex::EB2::CylinderIF> const& source)
    {
        cylinders.reserve(source.size());
        for (auto const& cylinder : source) {
            cylinders.push_back(cylinder);
        }
    }

    amrex::Gpu::ManagedVector<amrex::EB2::CylinderIF> cylinders;
};

class RuntimeCylinderIF : public amrex::GPUable {
public:
    RuntimeCylinderIF(
        amrex::EB2::CylinderIF const* cylinders,
        int count) noexcept
        : cylinders_(cylinders), count_(count)
    {}

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real operator()(
        AMREX_D_DECL(amrex::Real x, amrex::Real y, amrex::Real z)) const noexcept
    {
        amrex::Real value = cylinders_[0](AMREX_D_DECL(x, y, z));
        for (int n = 1; n < count_; ++n) {
            value = amrex::max(
                value, cylinders_[n](AMREX_D_DECL(x, y, z)));
        }
        return value;
    }

    AMREX_FORCE_INLINE amrex::Real operator()(
        amrex::RealArray const& point) const noexcept
    {
        return (*this)(AMREX_D_DECL(point[0], point[1], point[2]));
    }

private:
    amrex::EB2::CylinderIF const* cylinders_ = nullptr;
    int count_ = 0;
};

static_assert(std::is_trivially_copyable_v<RuntimeCylinderIF>);
static_assert(amrex::EB2::IsGPUable<RuntimeCylinderIF>::value);

EbGeometryBackend* activeOwner = nullptr;

} // namespace

EbGeometryBackend::~EbGeometryBackend()
{
    reset();
}

bool EbGeometryBackend::define(
    InternalGeometryQueryRegistry const& query,
    amrex::Geometry const& geom,
    amrex::BoxArray const& ba,
    amrex::DistributionMapping const& dm,
    std::string& error)
{
    error.clear();
    if (!is_canonical_internal_cell_layout(ba, geom) ||
        ba.size() != dm.size()) {
        error = "internal EB backend requires the canonical cell-centered BA/DM";
        return false;
    }
    if (query.empty()) {
        return true;
    }
    if (activeOwner != nullptr) {
        error = "only one internal EB aggregate may be active";
        return false;
    }
    std::vector<amrex::EB2::CylinderIF> functions;
    if (!query.makeEbCylinderFunctions(functions) || functions.empty()) {
        error = "internal geometry has no complete AMReX CylinderIF representation";
        return false;
    }

    auto storage = std::make_shared<RuntimeCylinderStorage>(functions);
    RuntimeCylinderIF implicit(
        storage->cylinders.data(),
        static_cast<int>(storage->cylinders.size()));
    using Shop = amrex::EB2::GeometryShop<
        RuntimeCylinderIF,
        std::shared_ptr<RuntimeCylinderStorage>>;
    Shop shop(implicit, storage);

    const int stackSize = amrex::EB2::IndexSpace::size();
    auto const* previous = amrex::EB2::TopIndexSpaceIfPresent();
    amrex::EB2::Build(shop, geom, 0, 0, 1, true, true, 0);
    indexSpace_ = amrex::EB2::TopIndexSpaceIfPresent();
    if (indexSpace_ == nullptr || indexSpace_ == previous ||
        amrex::EB2::IndexSpace::size() != stackSize + 1) {
        if (indexSpace_ != nullptr && indexSpace_ != previous) {
            amrex::EB2::IndexSpace::erase(
                const_cast<amrex::EB2::IndexSpace*>(indexSpace_));
        }
        indexSpace_ = nullptr;
        error = "EB2::Build did not install one identifiable IndexSpace";
        return false;
    }

    factory_ = amrex::makeEBFabFactory(
        indexSpace_,
        geom,
        ba,
        dm,
        amrex::Vector<int>{1},
        amrex::EBSupport::basic);
    if (factory_ == nullptr || factory_->boxArray() != ba ||
        factory_->DistributionMap() != dm ||
        factory_->getEBIndexSpace() != indexSpace_) {
        resetBuiltState();
        error = "basic EB factory does not match the canonical layout";
        return false;
    }
    activeOwner = this;
    return true;
}

void EbGeometryBackend::resetBuiltState() noexcept
{
    factory_.reset();
    if (indexSpace_ != nullptr) {
        amrex::EB2::IndexSpace::erase(
            const_cast<amrex::EB2::IndexSpace*>(indexSpace_));
        indexSpace_ = nullptr;
    }
    if (activeOwner == this) {
        activeOwner = nullptr;
    }
}

void EbGeometryBackend::reset() noexcept
{
    resetBuiltState();
}

} // namespace lbm::boundary
