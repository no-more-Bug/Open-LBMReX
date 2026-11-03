#include "integration/runtime/StaticHierarchyLifecycleTests.H"
#include "integration/runtime/StaticHierarchyTestSupport.H"
#include "lbm/amr/LbmAmrSimulation.H"
#include "lbm/initialization/MacroscopicState.H"

namespace lbm::tests::static_hierarchy {
namespace {

struct Profile {
    Real density = 1.0;
    AMREX_GPU_HOST_DEVICE
    initialization::MacroscopicState operator()(amrex::GpuArray<Real,3> const& x, Real) const noexcept
    {
        constexpr Real twoPi = 6.283185307179586476925286766559;
        return {density + 0.001*std::sin(twoPi*x[0])*std::cos(twoPi*x[1]),
                0.01*std::cos(twoPi*x[0]), 0.005*std::sin(twoPi*x[1]), 0.0};
    }
};

// This adapter observes actual callbacks. Only the controlled variant injects
// extra production scratch/Clear calls, before AmrCore registers the final BA/DM.
class LifecycleProbe final : public amr::LbmAmrSimulation<D2Q9Descriptor, Profile> {
    using Base = amr::LbmAmrSimulation<D2Q9Descriptor, Profile>;
public:
    LifecycleProbe(amr::StaticHierarchyLayout layout, Profile profile, bool controlled)
        : Base(std::move(layout),profile), controlled_(controlled) {}
    std::vector<std::string> prepare() { return Base::initialize({}); }
    int scratchCalls = 0, clearCalls = 0, errorCalls = 0;
protected:
    void ErrorEst(int lev, amrex::TagBoxArray& tags, Real time, int ngrow) override
    {
        ++errorCalls;
        amrex::Print() << "native_callback ErrorEst level=" << lev << " time=" << time << '\n';
        Base::ErrorEst(lev,tags,time,ngrow);
    }
    void MakeNewLevelFromScratch(int lev, Real time, amrex::BoxArray const& ba,
                                amrex::DistributionMapping const& dm) override
    {
        ++scratchCalls;
        amrex::Print() << "native_callback MakeNewLevelFromScratch level=" << lev
                       << " boxes=" << ba.size() << " cells=" << ba.numPts() << '\n';
        if (controlled_ && lev == 1) {
            amrex::BoxArray alternate(amrex::Box(amrex::IntVect(0),amrex::IntVect(7,7,0)));
            alternate.maxSize(amrex::IntVect(4,4,1));
            amrex::Vector<int> ranks(alternate.size());
            for (int i = 0; i < alternate.size(); ++i) ranks[i] = (i+1)%amrex::ParallelDescriptor::NProcs();
            amrex::DistributionMapping remapped(ranks);
            Base::MakeNewLevelFromScratch(lev,time,alternate,remapped);
            Base::ClearLevel(lev);
            Base::ClearLevel(lev); // Clearing an already empty slot must remain valid.
            clearCalls += 2;
            Base::MakeNewLevelFromScratch(lev,time,alternate,remapped);
            amrex::Print() << "controlled_callback scratch/clear/clear/scratch level=" << lev
                           << " alternate_boxes=" << alternate.size() << " followed_by_scratch_without_clear\n";
        }
        Base::MakeNewLevelFromScratch(lev,time,ba,dm);
    }
private:
    bool controlled_;
};

std::vector<Frame> inspect(amr::LbmAmrSimulation<D2Q9Descriptor, Profile>& simulation, Real density)
{
    std::vector<Frame> result;
    simulation.observe(true,[&](amr::LbmAmrView<D2Q9Descriptor> const& view) {
        require(view.step == 0 && view.physicalTime == 0 && view.levels.size() == 2,
                "lifecycle preparation changed time or level count");
        for (auto const& level : view.levels) {
            require(level.macro && level.populations.multifab().boxArray() == level.boxes &&
                    level.populations.multifab().DistributionMap() == level.mapping,
                    "lifecycle did not restore final fields and layout");
            auto frame = capture(0,0,level.geometry,level.populations.multifab(),&level.macro->multifab());
            check_moments(frame);
            const int n = frame.populations.domain.length(0);
            const Real pi = std::acos(Real(-1));
            for (std::size_t c = 0; c < frame.populations.owners.size(); ++c) {
                const auto cell = frame.populations.cell_at(c);
                const int i = cell[0], j = cell[1];
                const Real x = 2*pi*(i+0.5)/n, y = 2*pi*(j+0.5)/n;
                const Real rho = density+0.001*std::sin(x)*std::cos(y);
                const Real ux = 0.01*std::cos(x), uy = 0.005*std::sin(y);
                for (int q = 0; q < 9; ++q) {
                    const Real cu = cx[q]*ux+cy[q]*uy;
                    close(frame.populations.values[c*9+q],weights[q]*rho*(1+3*cu+4.5*cu*cu-1.5*(ux*ux+uy*uy)),
                          "recreated equilibrium");
                }
            }
            result.push_back(std::move(frame));
        }
    });
    return result;
}

void check_ghosts(amr::StaticHierarchySpec const& spec, bool periodicFine, bool periodicCoarse)
{
    auto prepared = amr::prepare_static_hierarchy<D2Q9Descriptor>(spec,Profile{});
    require(prepared.ok(),"ghost fixture preparation failed");
    auto simulation = prepared.take();
    auto frames = inspect(*simulation,1.0);
    int sameLevel = 0, periodicZ = 0, periodicXYFine = 0, periodicXYCoarse = 0, crossRank = 0, periodicFineCrossRank = 0, coarseFine = 0;
    simulation->observe(false,[&](amr::LbmAmrView<D2Q9Descriptor> const& view) {
        for (std::size_t lev = 0; lev < view.levels.size(); ++lev) {
            auto const& level = view.levels[lev];
            auto const& reference = frames[lev].populations;
            const int n = reference.domain.length(0);
            require(level.populations.nGrowVect() ==
                    (lev ? amrex::IntVect(2,2,0) : amrex::IntVect(1)),
                    "D2 level population ghost width");
            auto host = host_field_snapshot(level.populations.multifab());
            for (amrex::MFIter mfi(host); mfi.isValid(); ++mfi) {
                auto const data = host.const_array(mfi);
                for (amrex::BoxIterator it(amrex::grow(mfi.validbox(),host.nGrowVect())); it.ok(); ++it) {
                    auto const cell = it();
                    if (mfi.validbox().contains(cell)) continue;
                    amrex::IntVect donor((cell[0]%n+n)%n,(cell[1]%n+n)%n,0);
                    const auto index = reference.offset(donor);
                    const bool xyWrapped = cell[0] != donor[0] || cell[1] != donor[1];
                    if (index < reference.owners.size()) {
                        ++sameLevel;
                        if (cell[2] != 0) ++periodicZ;
                        if (lev && xyWrapped) ++periodicXYFine;
                        for (int b = 0; b < level.boxes.size(); ++b) {
                            if (level.boxes[b].contains(donor) && level.mapping[b] != amrex::ParallelDescriptor::MyProc()) {
                                ++crossRank;
                                if (lev && xyWrapped) ++periodicFineCrossRank;
                            }
                        }
                        for (int q = 0; q < 9; ++q)
                            close(data(cell,q),reference.values[index*9+q],"same-level/periodic ghost copy");
                    } else {
                        require(lev > 0,"L0 ghost has no periodic valid owner");
                        auto const& coarse = frames[lev-1].populations;
                        require(coarse.offset(amrex::IntVect(donor[0]/2,donor[1]/2,0)) < coarse.owners.size(),
                                "coarse/fine ghost has no coarse owner");
                        ++coarseFine;
                        if (xyWrapped) ++periodicXYCoarse;
                        // No coarse/fine value is certified; only its donor topology is checked.
                    }
                }
            }
        }
    });
    for (int* count : {&sameLevel,&periodicZ,&periodicXYFine,&periodicXYCoarse,&crossRank,&periodicFineCrossRank,&coarseFine})
        amrex::ParallelDescriptor::ReduceIntSum(*count);
    require(sameLevel > 0 && periodicZ > 0 && coarseFine > 0,"ghost fixture missed required categories");
    if (periodicFine) require(periodicXYFine > 0,"no periodic same-level fine donors were exercised");
    if (periodicCoarse) require(periodicXYCoarse > 0,"no periodic coarse donors were exercised");
    if (amrex::ParallelDescriptor::NProcs() > 1) require(crossRank > 0,"no actual cross-rank ghost donor was exercised");
    if (periodicFine && amrex::ParallelDescriptor::NProcs() > 1)
        require(periodicFineCrossRank > 0,"no actual periodic fine donor crossed ranks");
    amrex::Print() << "ghost_sources same_level=" << sameLevel << " periodic_z=" << periodicZ
                   << " periodic_xy_fine=" << periodicXYFine << " periodic_xy_coarse=" << periodicXYCoarse
                   << " cross_rank=" << crossRank << " coarse_fine=" << coarseFine << '\n';
    amrex::Print() << "ghost_periodic_fine_cross_rank=" << periodicFineCrossRank << '\n';
}

} // namespace

void run_static_lifecycle_checks()
{
    amr::StaticHierarchySpec spec;
    spec.gapFillRadiusCells = 1;
    spec.coarseNestingMarginCells = 1;
    spec.regions = {{1,{{0.25,0.25,0.0},{0.75,0.75,1.0/16}}}};
    std::vector<Frame> reference;
    for (bool controlled : {false,true}) {
        auto layout = amr::prepare_static_layout(spec, D2Q9Descriptor::d);
        require(layout.ok(),"callback fixture configuration failed");
        LifecycleProbe simulation(std::move(*layout.layout),Profile{},controlled);
        require(simulation.prepare().empty(),"callback fixture initialization failed");
        auto current = inspect(simulation,1.0);
        require(simulation.scratchCalls >= 2 && simulation.errorCalls > 0,
                "real AmrCore callbacks were not observed");
        require(simulation.clearCalls == (controlled ? 2 : 0),"controlled clear sequence mismatch");
        if (!controlled) reference = current;
        else for (std::size_t lev = 0; lev < current.size(); ++lev)
            compare(reference[lev],current[lev],"controlled repeated scratch final state");
    }
    auto recreated = amr::prepare_static_hierarchy<D2Q9Descriptor>(spec,Profile{1.02});
    require(recreated.ok(),"independent object recreation failed");
    inspect(*recreated.simulation,1.02);
    recreated.simulation.reset();
    spec.maxGridSize = {amrex::IntVect(8,8,1)};
    spec.regions = {{1,{{7.0/8,15.0/16,0.0},{1.25,9.0/8,1.0/16}}}};
    check_ghosts(spec,true,false);
    spec.regions = {{1,{{0,0.5,0.0},{0.5,1,1.0/16}}}};
    check_ghosts(spec,false,true);
    amrex::Print() << "Static hierarchy lifecycle checks: PASS\n";
}

} // namespace lbm::tests::static_hierarchy
