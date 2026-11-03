#include "integration/runtime/AmrEvolutionTestCase.H"
#include "integration/runtime/StaticHierarchyTopologyOracle.H"
#include <AMReX_BoxIterator.H>
#include <AMReX_Print.H>
#include <set>

namespace lbm::tests::amr_evolution {

void require(bool condition, std::string_view message)
{ if (!condition) amrex::Abort(std::string(message)); }

namespace {
void check_close(Real actual, Real expected, std::string_view subject, Real scale,
                 amrex::IntVect const* cell = nullptr)
{
    Real const tolerance=4096*std::numeric_limits<Real>::epsilon()*
        std::max({scale,std::abs(actual),std::abs(expected)});
    if (!std::isfinite(actual) || !std::isfinite(expected) || std::abs(actual-expected)>tolerance) {
        std::ostringstream out;
        out << std::setprecision(std::numeric_limits<Real>::max_digits10) << subject
            << ": actual=" << actual << " expected=" << expected << " tolerance=" << tolerance;
        if (cell) out << " cell=" << *cell;
        amrex::Abort(out.str());
    }
}
} // namespace

void close(Real actual, Real expected, std::string_view subject, Real scale)
{ check_close(actual,expected,subject,scale); }

Fixture central_fixture(int levels, int box)
{
    Fixture f;
    f.name="D"+std::to_string(Descriptor::d)+"-E"+std::to_string(levels-1)+"-box"+std::to_string(box);
    int const n=Descriptor::d == 2 ? 32 : 16;
    bool const d3=Descriptor::d == 3;
    auto& s=f.spec;
    s.nCell={n,n,d3 ? n : 1}; s.probHi={1,1,d3 ? 1 : Real(1)/16};
    s.refRatio=amrex::IntVect(2,2,d3 ? 2 : 1);
    s.maxGridSize={amrex::IntVect(box,box,d3 ? box : 1)};
    s.gapFillRadiusCells=s.coarseNestingMarginCells=1;
    f.expectedCoverage.push_back({{amrex::Box(amrex::IntVect(0),amrex::IntVect(n-1,n-1,d3 ? n-1 : 0))},1});
    for (int lev=1; lev<levels; ++lev) {
        Real const lo=lev == 1 ? Real(1)/4 : Real(3)/8, hi=1-lo;
        s.regions.push_back({lev,{{lo,lo,d3 ? lo : 0},{hi,hi,d3 ? hi : s.probHi[2]}}});
        int const cells=n*(1 << lev), a=int(lo*cells), b=int(hi*cells)-1;
        f.expectedCoverage.push_back({{amrex::Box(amrex::IntVect(a,a,d3 ? a : 0),
            amrex::IntVect(b,b,d3 ? b : 0))},1});
    }
    return f;
}

Fixture four_level_fixture(bool periodicShift)
{
    auto const name="D"+std::to_string(Descriptor::d)+"-Topo-L3Separated-L2Overlap-AutoL1"+
        (periodicShift ? "-half-x" : "");
    for (auto const& f : static_hierarchy::topology::supplement_fixtures(Descriptor::d,false))
        if (f.name == name) return f;
    amrex::Abort("missing four-level topology fixture");
    return {};
}

std::unique_ptr<EvolutionSimulation> make_simulation(Fixture const& fixture, EvolutionProfile profile)
{
    auto prepared=amr::prepare_static_hierarchy<Descriptor>(fixture.spec,profile);
    for (auto const& error : prepared.errors) amrex::Print() << error << '\n';
    require(prepared.ok(),fixture.name+": static preparation failed");
    auto result=prepared.take();
    result->observe(false,[&](EvolutionView const& view) {
        if (fixture.expectedCoverage.empty()) {
            static_hierarchy::topology::validate_coverage(fixture.spec,capture(view),Descriptor::d);
            return;
        }
        require(view.levels.size()==fixture.expectedCoverage.size(),fixture.name+": actual level count");
        for (std::size_t lev=0; lev<view.levels.size(); ++lev) {
            auto const& level=view.levels[lev];
            auto const& domain=level.geometry.Domain();
            amr_reference::SparseVolumeGrid coordinates(
                {domain.smallEnd(0),domain.smallEnd(1),domain.smallEnd(2)},
                {domain.length(0),domain.length(1),domain.length(2)},1);
            std::set<amr_reference::Cell> expected, actual;
            for (auto const& box : fixture.expectedCoverage[lev].boxes)
                for (amrex::BoxIterator it(box); it.ok(); ++it)
                    expected.insert(coordinates.wrap({it()[0],it()[1],it()[2]}));
            for (int b=0; b<level.boxes.size(); ++b)
                for (amrex::BoxIterator it(level.boxes[b]); it.ok(); ++it)
                    actual.insert({it()[0],it()[1],it()[2]});
            require(actual==expected,fixture.name+": actual valid coverage changed at level "+std::to_string(lev));
        }
    });
    return result;
}

config::LbmConfigIntent evolution_intent(Real tau)
{
    config::LbmConfigIntent result;
    result.parameterizationKind=config::LbmParameterizationKind::FromTau;
    result.tau=tau; result.rhoRefLb=1; result.uRefLb=0.02; result.uRefPhys=1;
    return result;
}

amr_reference::HierarchyReference initial_reference(
    EvolutionView const& view, Real finestTau, EvolutionProfile profile)
{
    auto const stencil=amr_reference::stencil(Descriptor::d);
    std::vector<amr_reference::ReferenceLevel> levels;
    int const finest=int(view.levels.size())-1;
    for (int lev=0; lev<=finest; ++lev) {
        auto const& actual=view.levels[lev];
        auto const& geometry=actual.geometry;
        auto const& domain=geometry.Domain();
        amr_reference::Cell const low{domain.smallEnd(0),domain.smallEnd(1),domain.smallEnd(2)};
        amr_reference::Cell const size{domain.length(0),domain.length(1),domain.length(2)};
        amr_reference::SparseVolumeGrid values(low,size,Descriptor::q), support(low,size,1);
        auto const dx=geometry.CellSizeArray(), lo=geometry.ProbLoArray();
        for (int b=0; b<actual.boxes.size(); ++b) {
            for (amrex::BoxIterator it(actual.boxes[b]); it.ok(); ++it) {
                auto const p=it();
                auto const state=profile({lo[0]+(p[0]-low[0]+Real(0.5))*dx[0],
                    lo[1]+(p[1]-low[1]+Real(0.5))*dx[1],lo[2]+(p[2]-low[2]+Real(0.5))*dx[2]},0);
                auto* target=values.insert({p[0],p[1],p[2]});
                for (int q=0; q<Descriptor::q; ++q)
                    target[q]={amr_reference::equilibrium(stencil,q,state.rho,{state.ux,state.uy,state.uz}),0};
            }
            auto const grown=amrex::grow(actual.boxes[b],amrex::IntVect(4,4,Descriptor::d == 3 ? 4 : 0));
            for (amrex::BoxIterator it(grown); it.ok(); ++it) support.insert({it()[0],it()[1],it()[2]});
        }
        std::vector<amr_reference::Cell> cells;
        cells.reserve(support.cells());
        for (std::size_t n=0; n<support.cells(); ++n) cells.push_back(support.cell(n));
        Real const tau=Real(0.5)+(finestTau-Real(0.5))/Real(1 << (finest-lev));
        levels.push_back({std::move(values),std::move(cells),1/tau,Real(0.02)*dx[0]});
    }
    return {std::move(levels),Descriptor::d};
}

std::vector<Frame> capture(EvolutionView const& view)
{
    std::vector<Frame> result;
    for (std::size_t lev=0; lev<view.levels.size(); ++lev) {
        auto const& l=view.levels[lev];
        auto f=static_hierarchy::capture(l.step,l.physicalTime,l.geometry,l.populations.multifab(),
                                        l.macro ? &l.macro->multifab() : nullptr);
        f.level=int(lev); result.push_back(std::move(f));
    }
    return result;
}

void check_populations(Snapshot const& actual, amr_reference::SparseVolumeGrid const& expected,
                       std::string const& subject)
{
    require(actual.cellIndices.size()==expected.cells() && actual.components==expected.populations(),subject+": coverage");
    auto const missing=subject+": missing reference cell";
    std::vector<std::string> componentSubjects;
    for (int q=0; q<actual.components; ++q) componentSubjects.push_back(subject+": q="+std::to_string(q));
    for (std::size_t n=0; n<actual.cellIndices.size(); ++n) {
        auto const p=actual.cell_at(n);
        auto const* reference=expected.find({p[0],p[1],p[2]});
        require(reference!=nullptr,missing);
        for (int q=0; q<actual.components; ++q)
            check_close(actual.values[n*actual.components+q],reference[q].value,componentSubjects[q],1,&p);
    }
}

void check_macros(Frame const& frame, amr_reference::SparseVolumeGrid const& reference)
{
    auto const& f=frame.populations;
    auto const& m=frame.macro;
    require(m.components==4 && m.cellIndices==f.cellIndices,"evolution macro coverage");
    auto const stencil=amr_reference::stencil(Descriptor::d);
    for (std::size_t n=0; n<f.cellIndices.size(); ++n) {
        Real rho=0, expectedRho=0;
        std::array<Real,3> momentum{}, expectedMomentum{};
        auto const cell=f.cell_at(n);
        auto const* expected=reference.find({cell[0],cell[1],cell[2]});
        for (int q=0; q<f.components; ++q) {
            Real const value=f.values[n*f.components+q];
            rho+=value;
            expectedRho+=expected[q].value;
            for (int d=0; d<3; ++d) {
                momentum[d]+=stencil.directions[q][d]*value;
                expectedMomentum[d]+=stencil.directions[q][d]*expected[q].value;
            }
        }
        close(m.values[n*4],rho,"evolution density moment");
        close(m.values[n*4],expectedRho,"evolution/reference density");
        for (int d=0; d<3; ++d) {
            close(m.values[n*4+1+d],momentum[d]/rho,"evolution velocity moment",0.02);
            close(m.values[n*4+1+d],expectedMomentum[d]/expectedRho,"evolution/reference velocity",0.02);
        }
    }
}

void compare_frames(std::vector<Frame> const& actual, std::vector<Frame> const& expected,
                    std::string const& subject)
{
    require(actual.size()==expected.size(),subject+": levels");
    for (std::size_t lev=0; lev<actual.size(); ++lev) {
        auto const& a=actual[lev]; auto const& b=expected[lev];
        require(a.level==b.level && a.step==b.step && a.physicalTime==b.physicalTime,subject+": clocks");
        for (bool macro : {false,true}) {
            auto const& x=macro ? a.macro : a.populations;
            auto const& y=macro ? b.macro : b.populations;
            require(x.components==y.components && x.cellIndices==y.cellIndices,subject+": coordinates");
            for (std::size_t n=0; n<x.values.size(); ++n)
                close(x.values[n],y.values[n],subject,macro && n%4 ? Real(0.02) : Real(1));
        }
    }
}

} // namespace lbm::tests::amr_evolution
