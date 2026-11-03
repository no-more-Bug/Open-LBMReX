#ifndef LBM_TESTS_AMRBOUNDARYEVOLUTIONTESTS_HH
#define LBM_TESTS_AMRBOUNDARYEVOLUTIONTESTS_HH

#include "integration/lbm/AmrBoundaryIntegrationTests.H"
#include "integration/runtime/StaticHierarchyTestSupport.H"
#include "lbm/amr/LbmAmrSimulation.H"
#include "lbm/dynamics/BGKDynamicsSelection.H"
#include "lbm/initialization/MacroscopicState.H"
#include "runtime/LbmAmrPreparation.H"

namespace lbm::tests::integration::amr_boundary_test {

using Real = amrex::Real;
using Velocity = amrex::GpuArray<Real, 3>;
using Snapshot = static_hierarchy::ValidSnapshot;

template<class Descriptor>
struct BoundaryEvolutionProfile {
    AMREX_GPU_HOST_DEVICE
    initialization::MacroscopicState operator()(amrex::GpuArray<Real, 3> const& p, Real) const noexcept
    {
        return {1 + p[0]/4096 + p[1]/8192 + p[2]/16384,
                0.01, 0.007, Descriptor::d == 3 ? Real(0.003) : Real(0)};
    }
};

template<class Descriptor>
amr::StaticHierarchySpec evolution_spec(int levels, bool split)
{
    amr::StaticHierarchySpec spec;
    spec.nCell = {65, 65, Descriptor::d == 3 ? 5 : 1};
    spec.probHi = {16.25, 16.25, Descriptor::d == 3 ? 1.25 : 0.25};
    spec.isPeriodic = {0, 0, 1};
    spec.refRatio = amrex::IntVect(2, 2, Descriptor::d == 3 ? 2 : 1);
    const int box = split ? 16 : 128;
    spec.maxGridSize = {amrex::IntVect(box, box, Descriptor::d == 3 ? box : 1)};
    spec.gapFillRadiusCells = spec.coarseNestingMarginCells = 1;
    for (int level = 1; level < levels; ++level) {
        const Real lo = level == 1 ? 3 : 6, hi = level == 1 ? 13 : 10;
        spec.regions.push_back({level, {{lo, lo, 0}, {hi, hi, spec.probHi[2]}}});
    }
    return spec;
}

inline amr::AmrBoundarySpec evolution_boundaries(int levels, bool guo)
{
    amr::AmrBoundarySpec spec;
    for (auto side : {boundary::DomainSide::XLo, boundary::DomainSide::XHi,
                      boundary::DomainSide::YLo, boundary::DomainSide::YHi}) {
        boundary::DomainBoundarySpec wall;
        wall.name = "wall_" + std::to_string(static_cast<int>(side));
        wall.location = side;
        wall.scheme = guo ? boundary::BoundaryScheme::GuoExtrapolation : boundary::BoundaryScheme::HalfwayBounceBack;
        if (guo && side == boundary::DomainSide::YHi) {
            wall.physical = boundary::BoundaryPhysical::MovingWall;
            wall.profile = boundary::BoundaryProfileKind::Uniform;
            wall.prescribedULb = {0.03, 0, 0};
            wall.hasPrescribedULb = true;
        }
        spec.domain.push_back(std::move(wall));
    }
    if (levels == 3) {
        spec.internal = {cylinder("L1", 4.5, 8, 0.625), cylinder("L2", 8.15625, 8.15625, 0.25)};
        spec.internal[1].scheme = boundary::BoundaryScheme::Bouzidi;
    } else {
        const Real h = Real(0.25) / (1 << (levels - 1));
        spec.internal = {cylinder("owned", 8 + h/2, 8 + h/2, 4*h)};
    }
    return spec;
}

inline amrex::IntVect wrapped(amrex::Geometry const& geom, amrex::IntVect cell)
{
    for (int d = 0; d < 3; ++d) if (geom.isPeriodic(d)) {
        const int lo = geom.Domain().smallEnd(d), n = geom.Domain().length(d);
        cell[d] = lo + ((cell[d] - lo) % n + n) % n;
    }
    return cell;
}

template<class Descriptor>
Real equilibrium_value(int q, Real rho, Velocity const& u)
{
    const Real cu = Descriptor::cx(q)*u[0] + Descriptor::cy(q)*u[1] + Descriptor::cz(q)*u[2];
    return Descriptor::w(q)*rho*(1 + 3*cu + 4.5*cu*cu - 1.5*(u[0]*u[0]+u[1]*u[1]+u[2]*u[2]));
}

template<class Descriptor>
Real initial_value(amrex::Geometry const& geom, amrex::IntVect cell, int q, bool guo)
{
    cell = wrapped(geom, cell);
    const auto domain = geom.Domain();
    Velocity velocity{0.01, 0.007, Descriptor::d == 3 ? Real(0.003) : Real(0)};
    if (guo && (cell[0] == domain.smallEnd(0) || cell[0] == domain.bigEnd(0) ||
                cell[1] == domain.smallEnd(1) || cell[1] == domain.bigEnd(1))) {
        const bool topInterior = cell[1] == domain.bigEnd(1) &&
            cell[0] > domain.smallEnd(0) && cell[0] < domain.bigEnd(0);
        velocity = {topInterior ? Real(0.03) : Real(0), 0, 0};
        for (int d = 0; d < 2; ++d) {
            cell[d] += cell[d] == domain.smallEnd(d) ? 1 : (cell[d] == domain.bigEnd(d) ? -1 : 0);
        }
    }
    const auto h = geom.CellSizeArray(), lo = geom.ProbLoArray();
    const Velocity position{lo[0]+(cell[0]+0.5)*h[0], lo[1]+(cell[1]+0.5)*h[1], lo[2]+(cell[2]+0.5)*h[2]};
    const auto state = BoundaryEvolutionProfile<Descriptor>{}(position, 0);
    return equilibrium_value<Descriptor>(q, state.rho, velocity);
}

template<class Descriptor>
std::vector<Snapshot> capture_boundary_view(amr::LbmAmrView<Descriptor> const& view)
{
    std::vector<Snapshot> result;
    for (auto const& level : view.levels)
        result.push_back(static_hierarchy::snapshot(level.populations.multifab(), level.geometry.Domain()));
    return result;
}

template<class Descriptor>
void check_boundary_initial(amr::LbmAmrView<Descriptor> const& view, bool guo)
{
    require(view.step == 0 && view.physicalTime == 0, "initial boundary view clock");
    for (std::size_t level = 0; level < view.levels.size(); ++level) {
        auto const& data = view.levels[level];
        auto host = host_field_snapshot(data.populations.multifab());
        for (amrex::MFIter mfi(host); mfi.isValid(); ++mfi) {
            auto const values = host.const_array(mfi);
            for (amrex::BoxIterator it(mfi.validbox()); it.ok(); ++it) {
                for (int q = 0; q < Descriptor::q; ++q)
                    static_hierarchy::close(values(it(), q), initial_value<Descriptor>(data.geometry, it(), q, guo && level == 0),
                                            "initial Guo shell / untouched HBB and internal valid");
            }
        }
    }
}

template<class Descriptor>
amrex::Long check_internal_first(amr::LbmAmrLevelView<Descriptor> const& level,
                                std::vector<boundary::InternalBoundarySpec> const& specs)
{
    const auto& geom = level.geometry;
    const Real h = geom.CellSize(0);
    auto host = host_field_snapshot(level.populations.multifab());
    amrex::Long checked = 0, remote = 0;
    for (amrex::MFIter mfi(host); mfi.isValid(); ++mfi) {
        const auto values = host.const_array(mfi);
        for (amrex::BoxIterator it(mfi.validbox()); it.ok(); ++it) for (auto const& spec : specs) {
            auto const cell = it();
            const Real x = (cell[0]+0.5)*h-spec.centerPhys[0], y = (cell[1]+0.5)*h-spec.centerPhys[1];
            const Real radius2 = spec.radiusPhys*spec.radiusPhys;
            if (x*x+y*y < radius2) continue;
            for (int q = 1; q < Descriptor::q; ++q) {
                const Real vx = h*Descriptor::cx(q), vy = h*Descriptor::cy(q);
                if ((x+vx)*(x+vx)+(y+vy)*(y+vy) >= radius2) continue;
                const Real a = vx*vx+vy*vy, b = 2*(x*vx+y*vy), c = x*x+y*y-radius2;
                const Real fraction = (-b-std::sqrt(b*b-4*a*c))/(2*a);
                const auto direction = amrex::IntVect(Descriptor::cx(q),Descriptor::cy(q),Descriptor::cz(q));
                const auto behind = wrapped(geom, cell-direction);
                const int opposite = Descriptor::opposite(q);
                const Real outgoing = initial_value<Descriptor>(geom, cell, q, false);
                const Real incoming = initial_value<Descriptor>(geom, cell, opposite, false);
                const Real donor = initial_value<Descriptor>(geom, behind, q, false);
                const Real expected = spec.scheme == boundary::BoundaryScheme::Yms ?
                    (fraction*incoming+fraction*outgoing+(1-fraction)*donor)/(1+fraction) :
                    (fraction < 0.5 ? 2*fraction*outgoing+(1-2*fraction)*donor :
                     (outgoing+(2*fraction-1)*incoming)/(2*fraction));
                static_hierarchy::close(values(cell, opposite), expected, "first-step analytic internal reconstruction");
                ++checked;
                for (int grid = 0; grid < level.boxes.size(); ++grid) {
                    if (level.boxes[grid].contains(behind)) {
                        remote += level.mapping[grid] != amrex::ParallelDescriptor::MyProc();
                        break;
                    }
                }
            }
        }
    }
    amrex::ParallelDescriptor::ReduceLongSum(checked);
    amrex::ParallelDescriptor::ReduceLongSum(remote);
    require(checked > 0, "internal first-step oracle must visit real links");
    return remote;
}

template<class Descriptor>
void check_domain_first(amr::LbmAmrLevelView<Descriptor> const& level, bool guo)
{
    auto const& geom = level.geometry;
    const auto domain = geom.Domain();
    auto host = host_field_snapshot(level.populations.multifab());
    for (amrex::MFIter mfi(host); mfi.isValid(); ++mfi) {
        auto const values = host.const_array(mfi);
        for (int z = domain.smallEnd(2); z <= domain.bigEnd(2); ++z) {
            for (int face = 0; face < 5; ++face) {
                const int hi = domain.bigEnd(0), middle = hi/2;
                const amrex::IntVect cell = face == 0 ? amrex::IntVect(0,middle,z) :
                    face == 1 ? amrex::IntVect(hi,middle,z) : face == 2 ? amrex::IntVect(middle,0,z) :
                    face == 3 ? amrex::IntVect(middle,hi,z) : amrex::IntVect(2,3,z);
                if (!mfi.validbox().contains(cell)) continue;
                std::array<Real, Descriptor::q> donor{};
                Real rho = 0;
                Velocity u{};
                if (guo && face < 4) {
                    auto interior = cell;
                    interior[face/2] += face%2 == 0 ? 1 : -1;
                    for (int q = 0; q < Descriptor::q; ++q) {
                        auto source = interior-amrex::IntVect(Descriptor::cx(q),Descriptor::cy(q),Descriptor::cz(q));
                        donor[q] = initial_value<Descriptor>(geom, source, q, true);
                        rho += donor[q];
                        for (int d = 0; d < 3; ++d) u[d] += Descriptor::c(q,d)*donor[q];
                    }
                    for (auto& value : u) value /= rho;
                }
                for (int q = 0; q < Descriptor::q; ++q) {
                    auto source = wrapped(geom, cell-amrex::IntVect(Descriptor::cx(q),Descriptor::cy(q),Descriptor::cz(q)));
                    Real expected;
                    if (guo && face < 4) {
                        const Velocity wallVelocity{face == 3 ? Real(0.03) : Real(0), 0, 0};
                        expected = equilibrium_value<Descriptor>(q,rho,wallVelocity)+donor[q]-equilibrium_value<Descriptor>(q,rho,u);
                    } else {
                        expected = domain.contains(source) ? initial_value<Descriptor>(geom,source,q,guo) :
                            initial_value<Descriptor>(geom,cell,Descriptor::opposite(q),false);
                    }
                    static_hierarchy::close(values(cell,q),expected,"first-step domain / unaffected interior oracle");
                }
            }
        }
    }
}

template<class Descriptor>
std::vector<Snapshot> run_boundary_case(int levels, bool guo, bool split, bool segmented)
{
    auto spec = evolution_spec<Descriptor>(levels,split);
    auto boundaries = evolution_boundaries(levels,guo);
    const int baseline = amrex::EB2::IndexSpace::size();
    auto prepared = amr::prepare_static_hierarchy<Descriptor>(spec,BoundaryEvolutionProfile<Descriptor>{},boundaries);
    for (auto const& error : prepared.errors) amrex::Print() << error << '\n';
    require(prepared.ok(), "boundary simulation preparation");
    auto simulation = prepared.take();
    require(simulation->numLevels() == levels && amrex::EB2::IndexSpace::size() == baseline,
            "final hierarchy must own only runtime boundary metadata");
    auto const& identities = simulation->resolvedBoundaries();
    require(identities.internal.size() == boundaries.internal.size() && identities.internalObjects.size() == 1,
            "simulation must publish the resolved boundary and Object identities");
    for (std::size_t index = 0; index < identities.internal.size(); ++index)
        require(simulation->internalBoundaryLevel(identities.internal[index].boundaryId) ==
                    (levels == 3 ? int(index)+1 : levels-1), "simulation must publish actual internal ownership");
    std::vector<Snapshot> initial;
    simulation->observe(true,[&](auto const& view) {
        check_boundary_initial(view,guo);
        initial = capture_boundary_view(view);
        if (!split && levels > 1 && amrex::ParallelDescriptor::NProcs() > 1) {
            auto const& owners = view.levels.back().mapping.ProcessorMap();
            int empty = std::find(owners.begin(),owners.end(),amrex::ParallelDescriptor::MyProc()) == owners.end();
            amrex::ParallelDescriptor::ReduceIntSum(empty);
            require(empty == amrex::ParallelDescriptor::NProcs()-1,"one-box internal boundary must leave empty ranks");
        }
    });
    config::LbmConfigIntent intent;
    intent.parameterizationKind = config::LbmParameterizationKind::FromTau;
    intent.tau = 0.8; intent.uRefLb = 0.04; intent.uRefPhys = 2.0;
    auto parameters = runtime::prepare_amr_evolution(*simulation,intent,BGKDynamicsSelection{});
    require(parameters.ok(), "boundary evolution preparation");
    auto evolution = std::move(parameters).take();
    for (int level = 0; level < levels; ++level) {
        static_hierarchy::close(evolution.level(level).dtPhys,0.005/(1 << level),"actual per-level physical dt");
        static_hierarchy::close(evolution.level(level).lbm.tau,0.5+0.3/(1 << (levels-1-level)),"actual per-level tau");
    }
    int calls = 0;
    std::vector<int> fineCalls(levels,0);
    amrex::Long remote = 0;
    auto callback = [&](auto const& view) {
        ++calls;
        require(view.step == calls,"boundary synchronized callback count");
        static_hierarchy::close(view.physicalTime,calls*0.005,"boundary synchronized physical time");
        for (int level = 0; level < levels; ++level) {
            require(view.levels[level].step == calls*(1 << level),"boundary local target step");
            static_hierarchy::close(view.levels[level].physicalTime,view.physicalTime,"boundary common endpoint");
        }
        if (view.step == 1) {
            check_domain_first(view.levels[0],guo);
            if (levels == 1) remote += check_internal_first(view.levels[0],boundaries.internal);
        }
    };
    auto fineCallback = [&](auto const& view) {
        ++fineCalls[view.level];
        if (view.fine.step != 1) return;
        std::vector<boundary::InternalBoundarySpec> owned;
        for (std::size_t index = 0; index < boundaries.internal.size(); ++index) {
            const int owner = levels == 3 ? int(index)+1 : levels-1;
            if (owner == view.level) owned.push_back(boundaries.internal[index]);
        }
        if (!owned.empty()) remote += check_internal_first(view.fine,owned);
    };
    require(simulation->evolve(evolution,0,true,callback,fineCallback).ok() && calls == 0,"boundary zero-step no-op");
    simulation->observe(false,[&](auto const& view) {
        auto after = capture_boundary_view(view);
        for (int level = 0; level < levels; ++level)
            static_hierarchy::compare(after[level],initial[level],"observe and zero-step preserve initialized fields",true);
    });
    for (int part = 0; part < (segmented ? 2 : 1); ++part) {
        auto result = simulation->evolve(evolution,segmented ? 1 : 2,true,callback,fineCallback);
        require(result.ok(),result.error);
    }
    require(calls == 2,"two complete boundary steps");
    for (int level = 1; level < levels; ++level)
        require(fineCalls[level] == 2*(1 << level),"boundary fine-step callback count");
    if (split && levels == 3 && amrex::ParallelDescriptor::NProcs() > 1)
        require(remote > 0,"boundary evolution must consume actual remote same-level donors");
    require(amrex::EB2::IndexSpace::size() == baseline,"boundary evolution must not rebuild EB");
    std::vector<Snapshot> final;
    simulation->observe(false,[&](auto const& view) { final = capture_boundary_view(view); });
    amrex::Print() << Descriptor::name() << " boundary evolution passed: levels=" << levels << " guo=" << guo
                   << " split=" << split << " segmented=" << segmented << " remote_boundary_donors=" << remote << '\n';
    return final;
}

template<class Descriptor>
void check_mixed_boundary_layout()
{
    auto spec = evolution_spec<Descriptor>(2,true);
    auto missing = amr::prepare_static_hierarchy<Descriptor>(spec,BoundaryEvolutionProfile<Descriptor>{});
    require(!missing.ok() && !missing.simulation && !missing.errors.empty() &&
            missing.errors.front().find("boundary_preparation") != std::string::npos,"nonperiodic entry must check missing domain specs");
    spec.isPeriodic[1] = 1;
    spec.regions = {{1,{{4,-0.5,0},{8,0.5,spec.probHi[2]}}}};
    auto layout = amr::prepare_static_layout(spec,Descriptor::d);
    require(layout.ok(),"mixed-periodic region must wrap only its periodic axis");
    for (auto const& blocking : layout.layout->mesh.blocking_factor)
        require(blocking == amrex::IntVect(1),"all hierarchy blocking factors must be one");
    auto const& domain = layout.layout->geometry.Domain();
    auto grown = amr::detail::periodic_grow(amrex::BoxArray(amrex::Box(amrex::IntVect(0),amrex::IntVect(0))),
        domain,layout.layout->geometry.periodicity(),Descriptor::d,1);
    require(grown.contains(amrex::IntVect(0,domain.bigEnd(1),0)) &&
            !grown.contains(amrex::IntVect(domain.bigEnd(0),0,0)),"mixed growth must wrap y and clip x");
    spec.regions[0].box.lo[0] = -0.5; spec.regions[0].box.hi[0] = 0.5;
    auto outside = amr::prepare_static_layout(spec,Descriptor::d);
    require(!outside.ok() && outside.errors.front().find("invalid_region") != std::string::npos,
            "nonperiodic requests must not silently wrap outside the domain");
}

template<class Descriptor>
void run_evolution_checks()
{
    check_mixed_boundary_layout<Descriptor>();
    const auto prior = amrex::DistributionMapping::strategy();
    amrex::DistributionMapping::strategy(amrex::DistributionMapping::ROUNDROBIN);
    for (int sample = 0; sample < 5; ++sample) {
        const int levels = sample < 3 ? sample+1 : sample == 3 ? 3 : 2;
        const bool guo = sample >= 3, split = sample != 4;
        auto continuous = run_boundary_case<Descriptor>(levels,guo,split,false);
        auto segmented = run_boundary_case<Descriptor>(levels,guo,split,true);
        for (int level = 0; level < levels; ++level)
            static_hierarchy::compare(continuous[level],segmented[level],"boundary continuous versus segmented");
    }
    amrex::DistributionMapping::strategy(prior);
    amrex::Print() << Descriptor::name() << " AMR boundary evolution integration passed.\n";
}

} // namespace lbm::tests::integration::amr_boundary_test
#endif
