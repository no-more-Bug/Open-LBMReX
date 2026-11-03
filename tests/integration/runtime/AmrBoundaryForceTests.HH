#ifndef LBM_TESTS_AMRBOUNDARYFORCETESTS_HH
#define LBM_TESTS_AMRBOUNDARYFORCETESTS_HH

#include "integration/runtime/AmrBoundaryEvolutionTests.H"
#include "lbm/dynamics/DynamicsModel.H"

#include <limits>

namespace lbm::tests::integration::amr_boundary_test {

inline void force_close(Real actual, Real expected, char const* message)
{
    const Real tolerance = 4096*std::numeric_limits<Real>::epsilon()*(1+std::abs(expected));
    if (!std::isfinite(actual) || std::abs(actual-expected) > tolerance) {
        std::ostringstream detail;
        detail << std::setprecision(17) << message << ": actual=" << actual << " expected=" << expected
               << " tolerance=" << tolerance;
        amrex::Abort(detail.str());
    }
}

struct ForceRestProfile {
    AMREX_GPU_HOST_DEVICE
    initialization::MacroscopicState operator()(Velocity const&, Real) const noexcept { return {1,0,0,0}; }
};

/** The rest population is a local step counter, unaffected by cut-link writes. */
template<class Descriptor>
struct ForceProbeDynamics : LocalDynamicsModel<Descriptor> {
    template<class Cell>
    AMREX_GPU_HOST_DEVICE
    void collide(Cell&& cell) const noexcept
    {
        const Real amplitude = cell[0] + Real(1)/128;
        cell[0] = amplitude;
        for (int q = 1; q < Descriptor::q; ++q) cell[q] = Real(q+1)*amplitude/64;
    }
};

struct ForceProbeSelection {
    template<class Descriptor> using Dynamics = ForceProbeDynamics<Descriptor>;
    template<class Descriptor>
    DynamicsPreparationResult<Dynamics<Descriptor>> prepare(config::ResolvedLbmConfig const&) const
    {
        DynamicsPreparationResult<Dynamics<Descriptor>> result;
        result.dynamics.emplace();
        return result;
    }
};

inline amr::AmrBoundarySpec force_boundaries(int grouping, bool enabled)
{
    auto spec = evolution_boundaries(2,false);
    spec.collectForces = enabled;
    spec.internal = {cylinder("coarse_a",3.125,4.125,1), cylinder("coarse_b",6.125,4.125,1),
                     cylinder("fine",10.0625,10.0625,0.5)};
    spec.internal[1].scheme = spec.internal[2].scheme = boundary::BoundaryScheme::Bouzidi;
    for (int b = 0; b < 3; ++b) {
        spec.internal[b].objectName = grouping == 0 ? "object_"+std::to_string(b) :
            grouping == 1 ? (b < 2 ? "coarse_pair" : "fine") : (b == 1 ? "coarse_only" : "across");
    }
    return spec;
}

/** Independent circle intersections and synthetic momentum exchange at unit amplitude. */
template<class Descriptor>
Velocity force_reference(amrex::Geometry const& geom, boundary::InternalBoundarySpec const& spec)
{
    const Real h = geom.CellSize(0), radius2 = spec.radiusPhys*spec.radiusPhys;
    const int xlo = static_cast<int>(std::floor((spec.centerPhys[0]-spec.radiusPhys)/h))-1;
    const int xhi = static_cast<int>(std::ceil((spec.centerPhys[0]+spec.radiusPhys)/h));
    const int ylo = static_cast<int>(std::floor((spec.centerPhys[1]-spec.radiusPhys)/h))-1;
    const int yhi = static_cast<int>(std::ceil((spec.centerPhys[1]+spec.radiusPhys)/h));
    Velocity force{};
    for (int k = 0; k < geom.Domain().length(2); ++k)
        for (int j = ylo; j <= yhi; ++j) for (int i = xlo; i <= xhi; ++i) {
            const Real x = (i+0.5)*h-spec.centerPhys[0], y = (j+0.5)*h-spec.centerPhys[1];
            if (x*x+y*y < radius2) continue;
            for (int q = 1; q < Descriptor::q; ++q) {
                const Real vx = h*Descriptor::cx(q), vy = h*Descriptor::cy(q);
                if ((x+vx)*(x+vx)+(y+vy)*(y+vy) >= radius2) continue;
                const Real a = vx*vx+vy*vy, b = 2*(x*vx+y*vy), c = x*x+y*y-radius2;
                const Real fraction = (-b-std::sqrt(b*b-4*a*c))/(2*a);
                const Real outgoing = Real(q+1)/64, incoming = Real(Descriptor::opposite(q)+1)/64;
                const Real returned = spec.scheme == boundary::BoundaryScheme::Yms ?
                    (fraction*incoming+outgoing)/(1+fraction) :
                    (fraction < 0.5 ? outgoing : (outgoing+(2*fraction-1)*incoming)/(2*fraction));
                for (int d = 0; d < 3; ++d) force[d] += (outgoing+returned)*Descriptor::c(q,d);
            }
        }
    require(std::abs(force[0])+std::abs(force[1])+std::abs(force[2]) > 1.0e-6,
            "force unit oracle must have a nonzero signal");
    return force;
}

template<class Descriptor>
Velocity expected_force(std::array<Velocity,3> const& reference, int boundary, int localStep)
{
    // Known two-level fixture: s=2 at L0, s=1 at L1. Do not reuse the
    // production scale conversion or its exponent to manufacture the oracle.
    const Real factor = boundary < 2 ? (Descriptor::d == 2 ? Real(2) : Real(4)) : Real(1);
    const Real amplitude = Descriptor::w(0)+Real(localStep)/128;
    auto force = reference[boundary];
    for (auto& component : force) component *= amplitude*factor;
    return force;
}

inline void check_force_label(boundary::InternalForceStep const& step, int level, int localStep, int coarseStep)
{
    require(step.level == level && step.populationStep == localStep && step.coarsestStep == coarseStep,
            "force step identity must survive collector and publication");
    const Real dt = level == 0 ? 0.005 : 0.0025;
    static_hierarchy::close(step.beginPhysicalTime,(localStep-1)*dt,"force begin time");
    static_hierarchy::close(step.endPhysicalTime,localStep*dt,"force end time");
    static_hierarchy::close(step.samplePhysicalTime(),(localStep-0.5)*dt,"force midpoint label");
}

template<class Descriptor>
void check_force_batch(boundary::InternalForceBatch const& batch, std::array<Velocity,3> const& reference,
                       int grouping, int coarseStep)
{
    require(batch.boundarySteps.size() == 3,"one L0 and two L1 raw sample sets per coarse step");
    const std::array<int,3> objectForBoundary = grouping == 0 ? std::array<int,3>{0,1,2} :
        grouping == 1 ? std::array<int,3>{0,0,1} : std::array<int,3>{0,1,0};
    for (int sample = 0; sample < 3; ++sample) {
        const int level = sample == 0 ? 0 : 1, localStep = sample == 0 ? coarseStep : 2*coarseStep-2+sample;
        auto const& row = batch.boundarySteps[sample];
        check_force_label(row.step,level,localStep,coarseStep);
        require(row.boundaryForces.size() == (sample == 0 ? 2 : 1),"owned boundary sample shape");
        for (std::size_t entry = 0; entry < row.boundaryForces.size(); ++entry) {
            const int boundary = sample == 0 ? static_cast<int>(entry) : 2;
            require(row.boundaryForces[entry].boundaryId.value == 4+boundary,
                    "sparse global BoundaryIds must not become local array indices");
            auto expected = expected_force<Descriptor>(reference,boundary,localStep);
            for (int d = 0; d < 3; ++d)
                force_close(row.boundaryForces[entry].forceLb[d],expected[d],"published boundary force / scale");
        }
    }
    require(batch.objectSteps.size() == (grouping == 2 ? 1 : 3),"Object step publication shape");
    for (std::size_t sample = 0; sample < batch.objectSteps.size(); ++sample) {
        const int level = sample == 0 ? 0 : 1, localStep = sample == 0 ? coarseStep : 2*coarseStep-2+int(sample);
        auto const& row = batch.objectSteps[sample];
        check_force_label(row.step,level,localStep,coarseStep);
        std::vector<int> objects = grouping == 2 ? std::vector<int>{1} :
            level == 1 ? std::vector<int>{objectForBoundary[2]} : grouping == 0 ? std::vector<int>{0,1} : std::vector<int>{0};
        require(row.objectForces.size() == objects.size(),"only complete same-level Objects have step forces");
        for (std::size_t object = 0; object < objects.size(); ++object) {
            Velocity expected{};
            for (int boundary = 0; boundary < 3; ++boundary) if (objectForBoundary[boundary] == objects[object]) {
                auto contribution = expected_force<Descriptor>(reference,boundary,localStep);
                for (int d = 0; d < 3; ++d) expected[d] += contribution[d];
            }
            require(row.objectForces[object].objectId.value == objects[object],"ObjectId publication order");
            for (int d = 0; d < 3; ++d)
                force_close(row.objectForces[object].forceLb[d],expected[d],"same-level Object sum");
        }
    }
    require(batch.objectIntervals.size() == (grouping == 2 ? 1 : 0),"only cross-level Objects have interval forces");
    if (grouping == 2) {
        auto const& interval = batch.objectIntervals[0];
        require(interval.coarsestStep == coarseStep && interval.objectForces.size() == 1 &&
                interval.objectForces[0].objectId.value == 0,"cross-level interval identity");
        static_hierarchy::close(interval.beginPhysicalTime,(coarseStep-1)*0.005,"coarse interval begin");
        static_hierarchy::close(interval.endPhysicalTime,coarseStep*0.005,"coarse interval end");
        auto expected = expected_force<Descriptor>(reference,0,coarseStep);
        auto first = expected_force<Descriptor>(reference,2,2*coarseStep-1);
        auto second = expected_force<Descriptor>(reference,2,2*coarseStep);
        for (int d = 0; d < 3; ++d)
            force_close(interval.objectForces[0].forceLb[d],expected[d]+(first[d]+second[d])/2,
                                    "cross-level Object time-weighted mean");
    }
}

template<class Descriptor>
void run_force_case(int grouping, bool segmented, bool split, bool enabled)
{
    auto spec = evolution_spec<Descriptor>(2,split);
    spec.regions = {{1,{{8,8,0},{12,12,spec.probHi[2]}}}};
    auto boundaries = force_boundaries(grouping,enabled);
    auto prepared = amr::prepare_static_hierarchy<Descriptor>(spec,ForceRestProfile{},boundaries);
    for (auto const& error : prepared.errors) amrex::Print() << error << '\n';
    require(prepared.ok(),"force hierarchy preparation");
    auto simulation = prepared.take();
    config::LbmConfigIntent intent;
    intent.parameterizationKind = config::LbmParameterizationKind::FromTau;
    intent.tau = 0.8; intent.uRefLb = 0.04; intent.uRefPhys = 2;
    auto parameters = runtime::prepare_amr_evolution(*simulation,intent,ForceProbeSelection{});
    require(parameters.ok(),"synthetic force evolution preparation");
    auto evolution = std::move(parameters).take();
    require(evolution.level(0).dxLb() == 2 && evolution.level(1).dxLb() == 1,"force fixture scales");
    std::array<Velocity,3> reference;
    for (int b = 0; b < 3; ++b) {
        const int owner = b < 2 ? 0 : 1;
        require(simulation->internalBoundaryLevel({4+b}) == owner,"force boundary effective ownership");
        reference[b] = force_reference<Descriptor>(simulation->geometry(owner),boundaries.internal[b]);
    }
    int completed = 0;
    auto emptyObservation = [&](auto const& view) {
        require(view.forceBatch.empty(),"initialization and standalone observe must not publish or replay force");
        if (!split && amrex::ParallelDescriptor::NProcs() > 1) {
            auto const& owners = view.levels[1].mapping.ProcessorMap();
            int empty = std::find(owners.begin(),owners.end(),amrex::ParallelDescriptor::MyProc()) == owners.end();
            amrex::ParallelDescriptor::ReduceIntSum(empty);
            require(empty == amrex::ParallelDescriptor::NProcs()-1,"force reduction requires an empty local rank");
        }
    };
    auto observer = [&](auto const& view) {
        require(view.step == ++completed,"force publication must have no missing or repeated coarse step");
        if (enabled) check_force_batch<Descriptor>(view.forceBatch,reference,grouping,view.step);
        else require(view.forceBatch.empty(),"disabled force sampling must publish no rows");
    };
    simulation->observe(false,emptyObservation);
    require(simulation->evolve(evolution,0,false,observer).ok() && completed == 0,"zero steps must not sample or deliver force");
    for (int part = 0; part < (segmented ? 2 : 1); ++part) {
        auto result = simulation->evolve(evolution,segmented ? 1 : 2,false,observer);
        require(result.ok(),result.error);
        simulation->observe(false,emptyObservation);
    }
    require(completed == 2,"force run complete delivery");
    amrex::Print() << Descriptor::name() << " force publication passed: grouping=" << grouping
                   << " segmented=" << segmented << " split=" << split << " enabled=" << enabled << '\n';
}

template<class Descriptor>
void run_force_checks()
{
    const auto prior = amrex::DistributionMapping::strategy();
    amrex::DistributionMapping::strategy(amrex::DistributionMapping::ROUNDROBIN);
    for (int grouping = 0; grouping < 3; ++grouping)
        for (bool segmented : {false,true}) run_force_case<Descriptor>(grouping,segmented,true,true);
    run_force_case<Descriptor>(2,true,false,true);
    run_force_case<Descriptor>(0,false,true,false);
    amrex::DistributionMapping::strategy(prior);
    amrex::Print() << Descriptor::name() << " AMR boundary force integration passed.\n";
}

} // namespace lbm::tests::integration::amr_boundary_test
#endif
