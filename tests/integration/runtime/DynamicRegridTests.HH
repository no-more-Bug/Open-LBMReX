#ifndef LBM_TESTS_DYNAMICREGRIDTESTS_HH
#define LBM_TESTS_DYNAMICREGRIDTESTS_HH

#include "integration/runtime/DynamicAmrTests.HH"
#include <array>

namespace lbm::tests::dynamic_amr {

inline config::LbmConfigIntent evolution_intent()
{
    config::LbmConfigIntent intent;
    intent.parameterizationKind = config::LbmParameterizationKind::FromTau;
    intent.tau = 0.8; intent.uRefLb = 0.1; intent.uRefPhys = 1;
    return intent;
}

struct TimedTagging {
    static constexpr bool needsVelocity = false;
    Real dt;
    std::shared_ptr<std::array<int,3>> calls;
    struct Rule {
        Real dt;
        template<class Descriptor>
        AMREX_GPU_HOST_DEVICE bool operator()(amr::TaggingCell<Descriptor> const& cell) const
        {
            int const event = static_cast<int>(cell.physicalTime/dt+0.5)/2;
            if (event == 0 || event == 5) return false;
            Real const x = cell.positionPhys[0];
            if (event == 3) return x >= 0.25 && x < 0.75;
            if (event == 4) return x >= 0.375 && x < 0.625;
            return x < 0.5;
        }
    };
    std::vector<std::string> validate(int) const { return {}; }
    Rule rule(int lev) const { ++(*calls)[lev]; return {dt}; }
};

struct FullTagging {
    static constexpr bool needsVelocity = false;
    struct Rule {
        template<class Descriptor>
        AMREX_GPU_HOST_DEVICE bool operator()(amr::TaggingCell<Descriptor> const&) const { return true; }
    };
    std::vector<std::string> validate(int) const { return {}; }
    Rule rule(int) const { return {}; }
};

template<class Left, class Right>
void compare_owners(Left& left, Right& right, bool sameBoxes, std::string const& label)
{
    left.observe(true,[&](auto const& a) {
        right.observe(true,[&](auto const& b) {
            require(a.step == b.step && a.levels.size() == b.levels.size(),"comparison clocks/levels");
            close(a.physicalTime,b.physicalTime,"comparison physical time");
            for (int lev = 0; lev < static_cast<int>(a.levels.size()); ++lev) {
                auto const& x = a.levels[lev]; auto const& y = b.levels[lev];
                require(x.boxes.numPts() == y.boxes.numPts(),"comparison coverage size");
                if (sameBoxes && x.boxes != y.boxes) {
                    amrex::Print() << label << " level=" << lev << " box_counts="
                                   << x.boxes.size() << ',' << y.boxes.size() << '\n';
                    for (int box = 0; box < std::min(x.boxes.size(),y.boxes.size()); ++box) {
                        if (x.boxes[box] == y.boxes[box]) continue;
                        amrex::Print() << "first_different_box=" << box << " left=" << x.boxes[box]
                                       << " right=" << y.boxes[box] << " owners="
                                       << x.mapping[box] << ',' << y.mapping[box] << '\n';
                        break;
                    }
                    amrex::Abort(label+": comparison changed layout");
                }
                for (bool macro : {false,true}) {
                    auto const& leftField = macro ? x.macro->multifab() : x.populations.multifab();
                    auto const& rightField = macro ? y.macro->multifab() : y.populations.multifab();
                    amrex::MultiFab aligned(x.boxes,x.mapping,rightField.nComp(),0);
                    aligned.setVal(-777);
                    aligned.ParallelCopy(rightField,0,0,rightField.nComp(),0,0);
                    auto hostA = host_field_snapshot(leftField);
                    auto hostB = host_field_snapshot(aligned);
                    for (amrex::MFIter mfi(hostA); mfi.isValid(); ++mfi) {
                        auto const af = hostA.const_array(mfi), bf = hostB.const_array(mfi);
                        for (amrex::BoxIterator cell(mfi.validbox()); cell.ok(); ++cell) {
                            auto const& p = cell();
                            for (int q = 0; q < leftField.nComp(); ++q)
                                close(af(p[0],p[1],p[2],q),bf(p[0],p[1],p[2],q),
                                      "completed population/macro comparison");
                        }
                    }
                }
            }
        });
    });
}

template<class Descriptor>
auto timed_run(int allowed, std::vector<int> const& segments)
{
    auto spec = base_spec<Descriptor>(allowed);
    spec.regridInterval = 2;
    spec.tagBufferCells = {amrex::IntVect(0)};
    spec.gapFillRadiusCells = 1;
    spec.coarseNestingMarginCells = 2;
    Real const dt = 0.1/spec.nCell[0];
    auto calls = std::make_shared<std::array<int,3>>();
    auto result = amr::prepare_dynamic_hierarchy<Descriptor>(
        spec,WaveProfile{Descriptor::d == 3},TimedTagging{dt,calls});
    require(result.ok() && result.simulation->numLevels() == 1,"timeline must start on L0");
    auto owner = result.take();
    auto prepared = runtime::prepare_amr_evolution(*owner,evolution_intent(),BGKDynamicsSelection{});
    require(prepared.ok(),"timeline preparation");
    auto evolution = std::move(prepared).take();
    profiling::AmrTimestepProfiler profiler(profiling::TimestepProfilingMode::StageBreakdown,0,16,allowed+1);
    *calls = {};
    int previousLevels = 1, observations = 0;
    std::array<int,3> expectedSteps{}, fineSteps{};
    std::vector<amrex::BoxArray> previousBoxes;
    auto observe = [&](auto const& view) {
        ++observations;
        require(view.step == observations,"coarsest observations must be consecutive");
        close(view.physicalTime,view.step*dt,"synchronized endpoint");
        for (int lev = 0; lev <= allowed; ++lev) {
            int const expectedCalls = view.step%2 == 0 && lev < std::min(previousLevels,allowed) ? 1 : 0;
            require((*calls)[lev] == expectedCalls,"candidate tagging must run once per eligible level/event");
            if (lev < previousLevels) expectedSteps[lev] += 1<<lev;
            if (lev > 0) require(fineSteps[lev] == expectedSteps[lev],"regrid changed hierarchy during substeps");
        }
        *calls = {};
        require(view.levels.size() <= static_cast<std::size_t>(previousLevels+1),"more than one new level");
        if (view.step%2 != 0) require(view.levels.size() == static_cast<std::size_t>(previousLevels),"off-cadence regrid");
        if (view.step == 2 || view.step == 12) require(view.levels.size() == 2,"first added level");
        if (view.step == 4 || view.step == 14) require(view.levels.size() == static_cast<std::size_t>(allowed+1),"allowed finest not reached");
        if (view.step == 10) require(view.levels.size() == 1,"all fine levels must be removed");
        if (view.step == 6) require(view.levels[1].boxes != previousBoxes[1],"fine patch did not move");
        if (view.step == 8) require(view.levels[1].boxes.numPts() < previousBoxes[1].numPts(),"fine patch did not shrink");
        previousBoxes.clear();
        for (int lev = 0; lev < static_cast<int>(view.levels.size()); ++lev) {
            auto const& level = view.levels[lev];
            require(level.step == view.step*(1<<lev) && level.macro != nullptr,"new level clock/macro");
            close(level.physicalTime,view.physicalTime,"level synchronization");
            close(evolution.level(lev).dtPhys,dt/(1<<lev),"fixed prepared dt");
            require(evolution.level(lev).dtLb == (1<<(allowed-lev)),"allowed finest lattice units changed");
            previousBoxes.push_back(level.boxes);
        }
        previousLevels = static_cast<int>(view.levels.size());
    };
    auto fineObserve = [&](auto const& view) { ++fineSteps[view.level]; };
    for (int count : segments)
        require(owner->evolve(evolution,count,true,observe,fineObserve,&profiler).ok(),"dynamic evolution failed");
    require(observations == 16,"timeline did not reach its endpoint");
    for (int lev = 0; lev <= allowed; ++lev)
        require(profiler.levelSteps()[lev] == expectedSteps[lev],"inactive profiling levels advanced");
    owner->observe(true,[&](auto const& view) {
        require(view.step == 16 && view.levels.size() == previousBoxes.size(),"return observe is not post-regrid");
        for (int lev = 0; lev < previousLevels; ++lev)
            require(view.levels[lev].boxes == previousBoxes[lev],"return observer layout differs");
    });
    return owner;
}

template<class Descriptor>
void run_regrid_checks()
{
    for (int allowed : {1,2}) {
        auto continuous = timed_run<Descriptor>(allowed,{16});
        auto segmented = timed_run<Descriptor>(allowed,{7,9});
        compare_owners(*continuous,*segmented,true,"continuous/segmented");

        auto spec = base_spec<Descriptor>(allowed);
        spec.regridInterval = 0;
        spec.tagBufferCells = {amrex::IntVect(0)};
        auto frozen = amr::prepare_dynamic_hierarchy<Descriptor>(spec,WaveProfile{Descriptor::d == 3},FullTagging{});
        spec.regridInterval = 2;
        auto unchanged = amr::prepare_dynamic_hierarchy<Descriptor>(spec,WaveProfile{Descriptor::d == 3},FullTagging{});
        amr::StaticHierarchySpec staticSpec;
        static_cast<amr::HierarchyGeometrySpec&>(staticSpec) = spec;
        for (int lev = 1; lev <= allowed; ++lev)
            staticSpec.regions.push_back({lev,{spec.probLo,spec.probHi}});
        auto fixed = amr::prepare_static_hierarchy<Descriptor>(staticSpec,WaveProfile{Descriptor::d == 3});
        require(frozen.ok() && unchanged.ok() && fixed.ok(),"frozen/no-op/static preparation");
        compare_owners(*frozen.simulation,*fixed.simulation,true,"frozen/static initial");
        auto advance = [](auto& owner) {
            auto prepared = runtime::prepare_amr_evolution(owner,evolution_intent(),BGKDynamicsSelection{});
            require(prepared.ok(),"frozen/no-op evolution preparation");
            auto evolution = std::move(prepared).take();
            require(owner.evolve(evolution,4,true,[](auto const&) {}).ok(),"frozen/no-op evolution");
        };
        advance(*frozen.simulation); advance(*unchanged.simulation); advance(*fixed.simulation);
        compare_owners(*frozen.simulation,*unchanged.simulation,true,"frozen/no-op final");
        compare_owners(*frozen.simulation,*fixed.simulation,true,"frozen/static final");
    }
    amrex::Print() << "dynamic_regrid_pass d=" << Descriptor::d << '\n';
}

// The same production installation helper accepts AMReX-generated candidates in regrid.
// This test subclass supplies controlled equivalent layouts to exercise cache dependencies.
template<class Descriptor>
class LayoutTestOwner : public amr::LbmAmrSimulation<Descriptor,WaveProfile,RegionTagging> {
    using Base = amr::LbmAmrSimulation<Descriptor,WaveProfile,RegionTagging>;
public:
    explicit LayoutTestOwner(amr::DynamicHierarchyLayout layout)
        : Base(std::move(layout),WaveProfile{Descriptor::d == 3},RegionTagging{2})
    { require(this->initialize({}).empty(),"layout test initialization"); }

    void changeLayout(bool reblock)
    {
        std::vector<amrex::BoxArray> boxes;
        std::vector<amrex::DistributionMapping> mappings;
        Real time = 0;
        this->observe(false,[&](auto const& view) {
            time = view.physicalTime;
            for (int lev = 0; lev < static_cast<int>(view.levels.size()); ++lev) {
                auto const& level = view.levels[lev];
                boxes.push_back(level.boxes);
                mappings.push_back(level.mapping);
                if (lev == 0) continue;
                if (reblock) {
                    boxes.back().maxSize(amrex::IntVect(4,4,Descriptor::d == 3 ? 4 : 1));
                    require(boxes.back() != level.boxes,"reblock fixture did not change BoxArray");
                    amrex::Vector<int> owners(boxes.back().size(),amrex::ParallelDescriptor::NProcs()-1);
                    mappings.back() = amrex::DistributionMapping(owners);
                } else if (amrex::ParallelDescriptor::NProcs() > 1) {
                    auto owners = level.mapping.ProcessorMap();
                    for (auto& rank : owners) rank = (rank+1)%amrex::ParallelDescriptor::NProcs();
                    mappings.back() = amrex::DistributionMapping(owners);
                    require(mappings.back() != level.mapping,"owner fixture did not change DM");
                }
            }
        });
        this->installHierarchy(boxes,mappings,time);
    }
};

template<class Descriptor>
void run_relayout_checks()
{
    auto spec = base_spec<Descriptor>(2);
    spec.regridInterval = 0;
    auto layout = amr::prepare_dynamic_layout(spec,Descriptor::d);
    require(layout.ok(),"relayout geometry");
    LayoutTestOwner<Descriptor> changed(*layout.layout), reference(*layout.layout);
    auto preparedA = runtime::prepare_amr_evolution(changed,evolution_intent(),BGKDynamicsSelection{});
    auto preparedB = runtime::prepare_amr_evolution(reference,evolution_intent(),BGKDynamicsSelection{});
    require(preparedA.ok() && preparedB.ok(),"relayout parameters");
    auto a = std::move(preparedA).take(), b = std::move(preparedB).take();
    for (bool reblock : {true,false}) {
        require(changed.evolve(a,1,true,[](auto const&) {}).ok(),"relayout advance before change");
        require(reference.evolve(b,1,true,[](auto const&) {}).ok(),"reference advance");
        changed.changeLayout(reblock);
        compare_owners(changed,reference,false,"relayout");
        require(changed.evolve(a,2,true,[](auto const&) {}).ok(),"advance after layout change");
        require(reference.evolve(b,2,true,[](auto const&) {}).ok(),"reference continuation");
        compare_owners(changed,reference,false,"relayout");
    }
    amrex::Print() << "dynamic_relayout_pass d=" << Descriptor::d << '\n';
}
} // namespace lbm::tests::dynamic_amr
#endif
