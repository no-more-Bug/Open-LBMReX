#include "integration/runtime/StaticHierarchy3DTests.H"
#include "integration/runtime/StaticHierarchy3DFixtures.H"

namespace lbm::tests::static_hierarchy::d3 {
namespace {

class LifecycleProbe final : public Simulation {
public:
    LifecycleProbe(amr::StaticHierarchyLayout layout, Profile profile, bool controlled)
        : Simulation(std::move(layout),profile), controlled_(controlled) {}
    std::vector<std::string> prepare() { return initialize({}); }
    int scratches = 0, clears = 0, tags = 0;
protected:
    void ErrorEst(int lev, amrex::TagBoxArray& target, Real time, int ngrow) override
    { ++tags; Simulation::ErrorEst(lev,target,time,ngrow); }
    void MakeNewLevelFromScratch(int lev, Real time, amrex::BoxArray const& ba,
                                amrex::DistributionMapping const& dm) override
    {
        ++scratches;
        if (controlled_ && lev == 1) {
            amrex::BoxArray alternate(amrex::Box(amrex::IntVect(0),amrex::IntVect(3)));
            alternate.maxSize(2);
            amrex::Vector<int> ranks(alternate.size());
            for (int i = 0; i < alternate.size(); ++i) ranks[i] = (i+1)%amrex::ParallelDescriptor::NProcs();
            amrex::DistributionMapping remapped(ranks);
            Simulation::MakeNewLevelFromScratch(lev,time,alternate,remapped);
            Simulation::ClearLevel(lev); Simulation::ClearLevel(lev);
            clears += 2;
            Simulation::MakeNewLevelFromScratch(lev,time,alternate,remapped);
        }
        // Rebuild using the real callback BA/DM, without Clear after the last scratch.
        Simulation::MakeNewLevelFromScratch(lev,time,ba,dm);
    }
private:
    bool controlled_;
};

} // namespace

void lifecycle_checks()
{
    auto spec = central_fixture(false,2,8).spec;
    std::vector<Frame> native;
    for (bool controlled : {false,true}) {
        auto layout = amr::prepare_static_layout(spec,3);
        require(layout.ok(),"D3 lifecycle layout");
        LifecycleProbe simulation(std::move(*layout.layout),make_profile(spec),controlled);
        require(simulation.prepare().empty(),"D3 lifecycle initialization");
        require(simulation.scratches >= 2 && simulation.tags > 0,"D3 actual initialization callbacks");
        if (controlled) require(simulation.clears >= 2,"D3 controlled clear callbacks");
        auto frames = observe(simulation);
        if (!controlled) native = frames;
        else for (std::size_t lev = 0; lev < frames.size(); ++lev)
            compare(frames[lev],native[lev],"D3 scratch/clear replacement");
        auto repeated = simulation.prepare();
        require(!repeated.empty() && repeated.front().find("already_initialized:") == 0,
            "D3 repeated initialization refusal");
        auto evolutionIntent = intent();
        evolutionIntent.lengthRefLb.reset(); evolutionIntent.lengthRefPhys.reset();
        auto parameters = runtime::prepare_amr_evolution(simulation,evolutionIntent,BGKDynamicsSelection{});
        require(parameters.ok(),"D3 lifecycle evolution preparation");
        auto evolution = std::move(parameters).take();
        simulation.observe(false,[&](View const& view) {
            for (auto const& level : view.levels) require(!level.macro,"D3 opt-out macro view");
            auto reentrant = simulation.evolve(evolution,1,false,[](View const&) {});
            require(reentrant.error.find("invalid_run_state:") == 0,"D3 reentrant evolve refusal");
        });
        auto unchanged = observe(simulation);
        for (std::size_t lev = 0; lev < frames.size(); ++lev)
            compare(frames[lev],unchanged[lev],"D3 failed state operations");
        amrex::Print() << "d3_lifecycle_callbacks controlled=" << controlled
            << " scratch=" << simulation.scratches << " clear=" << simulation.clears
            << " tag=" << simulation.tags << '\n';

        // Test-local mutation models in-place population updates at unchanged time.
        simulation.observe(false,[](View const& view) {
            for (auto const& level : view.levels)
                const_cast<amrex::MultiFab&>(level.populations.multifab()).mult(1.02,0,27,0);
        });
        simulation.observe(true,[&](View const& view) {
            require(view.step == 0 && view.physicalTime == 0,"D3 macro refresh time");
            for (std::size_t lev = 0; lev < view.levels.size(); ++lev) {
                const auto refreshed = snapshot(view.levels[lev].macro->multifab(),view.levels[lev].geometry.Domain());
                auto const& previous = frames[lev].macro;
                for (std::size_t cell = 0; cell < previous.owners.size(); ++cell) {
                    if (!previous.owners[cell]) continue;
                    for (int n = 0; n < 4; ++n)
                        close(refreshed.values[cell*4+n],previous.values[cell*4+n]*(n == 0 ? 1.02 : 1.0),
                            "D3 macro refreshed after population change");
                }
            }
        });
    }
    auto profile = make_profile(spec); profile.density = 1.02;
    auto recreated = amr::prepare_static_hierarchy<Descriptor>(spec,profile);
    require(recreated.ok(),"D3 independent object creation");
    observe(*recreated.simulation,false,1.02);
    amrex::Print() << "d3_lifecycle_checks: PASS\n";
}

} // namespace lbm::tests::static_hierarchy::d3
