#include "integration/runtime/AmrEvolutionSnapshot.H"
#include "integration/runtime/AmrCompositeBalance.H"
#include "lbm/dynamics/BGKDynamicsSelection.H"
#include "runtime/LbmAmrPlotfile.H"
#include <AMReX_PlotFileDataImpl.H>
#include <AMReX_Print.H>
#include <optional>

namespace lbm::tests::amr_evolution {
namespace {
constexpr Real rhoPlot=2, velocityPlot=0.5;

std::vector<Snapshot> read_plot(std::string const& path, EvolutionView const& view,
                                std::vector<Frame> const& source)
{
    amrex::PlotFileData plot(path);
    require(plot.spaceDim()==3 && plot.finestLevel()+1==int(view.levels.size()) &&
        plot.nComp()==4 && plot.coordSys()==0,"evolution native dimensions");
    require(plot.varNames()==amrex::Vector<std::string>{"rho_norm","ux_norm","uy_norm","uz_norm"},
            "evolution native variable names");
    close(plot.time(),view.physicalTime,"evolution native time",view.physicalTime);
    std::vector<Snapshot> result;
    for (std::size_t lev=0; lev<view.levels.size(); ++lev) {
        auto const& l=view.levels[lev];
        require(plot.probDomain(int(lev))==l.geometry.Domain() && plot.levelStep(int(lev))==l.step &&
                plot.nGrowVect(int(lev))==amrex::IntVect(0),"evolution native domain/step/ghosts");
        if (lev+1<view.levels.size())
            require(plot.refRatioVect(int(lev))==view.refRatios[lev],"evolution native vector ratio");
        for (int d=0; d<3; ++d) {
            close(plot.probLo()[d],l.geometry.ProbLo(d),"native physical lower bound");
            close(plot.probHi()[d],l.geometry.ProbHi(d),"native physical upper bound");
            close(plot.cellSize(int(lev))[d],l.geometry.CellSize(d),"native directional spacing");
        }
        for (int b=0; b<plot.boxArray(int(lev)).size(); ++b)
            require(plot.boxArray(int(lev))[b].ixType().cellCentered(),"native cell centering");
        result.push_back(static_hierarchy::snapshot(plot.get(int(lev)),l.geometry.Domain()));
        auto const& native=result.back(); auto const& expected=source[lev].macro;
        require(native.cellIndices==expected.cellIndices,"native actual coverage");
        for (std::size_t n=0; n<native.values.size(); ++n)
            close(native.values[n]*(n%4 ? velocityPlot : rhoPlot),expected.values[n],
                  "native inverse normalization",n%4 ? Real(0.02) : Real(1));
    }
    return result;
}

} // namespace

EvolutionSnapshot::EvolutionSnapshot(std::string directory, Fixture const& fixture, int coarseSteps)
    : directory_(std::move(directory)), fixture_(fixture), coarseSteps_(coarseSteps)
{
    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::filesystem::create_directories(directory_);
        states_.open(directory_+"/states.csv");
        states_ << std::setprecision(std::numeric_limits<Real>::max_digits10)
                << "event,phase,coarsest_step,level,local_step,time_phys,i,j,k";
        for (std::string_view kind : {"actual","reference"})
            for (int q=0; q<Descriptor::q; ++q) states_ << ',' << kind << "_f" << q;
        states_ << ",rho,ux,uy,uz,native_rho_norm,native_ux_norm,native_uy_norm,native_uz_norm\n";
        require(states_.good(),"cannot open evolution states");
    }
    amrex::ParallelDescriptor::Barrier();
}

void EvolutionSnapshot::identity(EvolutionView const& view)
{
    if (!amrex::ParallelDescriptor::IOProcessor()) return;
    std::ofstream identity(directory_+"/identity.csv"), layout(directory_+"/layout.csv");
    identity << std::setprecision(std::numeric_limits<Real>::max_digits10)
        << "schema,sample,descriptor,backend,real_digits,ranks,levels,coarse_steps,nx,ny,nz,lx,ly,lz,dt0,tau_finest,rho_plot,u_plot,reference\n"
        << "amr-evolution-v1," << fixture_.name << ',' << Descriptor::name() << ',';
#ifdef AMREX_USE_CUDA
    identity << "cuda";
#else
    identity << "cpu";
#endif
    auto const& g=view.levels[0].geometry;
    identity << ',' << std::numeric_limits<Real>::digits << ',' << amrex::ParallelDescriptor::NProcs()
        << ',' << view.levels.size() << ',' << coarseSteps_;
    for (int d=0; d<3; ++d) identity << ',' << g.Domain().length(d);
    for (int d=0; d<3; ++d) identity << ',' << g.ProbHi(d)-g.ProbLo(d);
    identity << ',' << Real(0.02)*g.CellSize(0) << ",0.65," << rhoPlot << ',' << velocityPlot
             << ",independent-level-bgk-volume-v1\n";
    layout << "level,box,owner,ilo,jlo,klo,ihi,jhi,khi\n";
    for (std::size_t lev=0; lev<view.levels.size(); ++lev) {
        auto const& l=view.levels[lev];
        for (int b=0; b<l.boxes.size(); ++b) {
            layout << lev << ',' << b << ',' << l.mapping[b];
            for (int d=0; d<3; ++d) layout << ',' << l.boxes[b].smallEnd(d);
            for (int d=0; d<3; ++d) layout << ',' << l.boxes[b].bigEnd(d);
            layout << '\n';
        }
    }
    require(identity.good() && layout.good(),"cannot write evolution identity/layout");
}

void EvolutionSnapshot::rows(std::string_view phase, int coarsestStep, Frame const& actual,
    amr_reference::SparseVolumeGrid const& reference, Snapshot const* native)
{
    int const event=event_++;
    if (!amrex::ParallelDescriptor::IOProcessor()) return;
    auto const& f=actual.populations;
    for (std::size_t row=0; row<f.cellIndices.size(); ++row) {
        auto const p=f.cell_at(row);
        auto const* expected=reference.find({p[0],p[1],p[2]});
        require(expected!=nullptr,"artifact reference cell missing");
        states_ << event << ',' << phase << ',' << coarsestStep << ',' << actual.level << ','
                << actual.step << ',' << actual.physicalTime << ',' << p[0] << ',' << p[1] << ',' << p[2];
        for (int q=0; q<Descriptor::q; ++q) states_ << ',' << f.values[row*Descriptor::q+q];
        for (int q=0; q<Descriptor::q; ++q) states_ << ',' << expected[q].value;
        for (int n=0; n<4; ++n) {
            states_ << ',';
            if (native) states_ << actual.macro.values[row*4+n];
        }
        for (int n=0; n<4; ++n) {
            states_ << ',';
            if (native) states_ << native->values[row*4+n];
        }
        states_ << '\n';
    }
    states_.flush();
    require(states_.good(),"cannot write evolution states");
}

void EvolutionSnapshot::synchronized(
    EvolutionView const& view, std::vector<amr_reference::ReferenceLevel> const& reference)
{
    if (view.step==0) identity(view);
    auto const before=capture(view);
    std::string const plotPath=directory_+"/plot"+std::to_string(view.step);
    auto written=runtime::write_amr_plotfile(plotPath,view,rhoPlot,velocityPlot);
    require(written.ok(),written.error);
    auto const after=capture(view);
    for (std::size_t lev=0; lev<before.size(); ++lev) {
        require(before[lev].populations.values==after[lev].populations.values &&
                before[lev].macro.values==after[lev].macro.values && before[lev].step==after[lev].step &&
                before[lev].physicalTime==after[lev].physicalTime,"writer modified its borrowed source");
        check_populations(before[lev].populations,reference[lev].current,"output/reference population");
        check_macros(before[lev],reference[lev].current);
    }
    auto const native=read_plot(plotPath,view,before);
    auto const totals=composite(view,before,reference);
    Real domainVolume=1;
    for (int d=0; d<3; ++d)
        domainVolume*=view.levels[0].geometry.ProbHi(d)-view.levels[0].geometry.ProbLo(d);
    close(Real(totals.volume),domainVolume,"composite measure",domainVolume);
    if (view.step==0) {
        initialMass_=Real(totals.mass);
        for (int d=0; d<3; ++d) initialMomentum_[d]=Real(totals.momentum[d]);
    }
    std::array<Real,3> expectedMomentum;
    for (int d=0; d<3; ++d) expectedMomentum[d]=Real(totals.referenceMomentum[d]);
    compare_totals(totals,Real(totals.referenceMass),expectedMomentum,initialMass_,"composite independent reference");
    compare_totals(totals,initialMass_,initialMomentum_,initialMass_,"composite conservation");
    for (std::size_t lev=0; lev<before.size(); ++lev)
        rows("sync",view.step,before[lev],reference[lev].current,&native[lev]);
    amrex::Print() << std::setprecision(17) << "evolution_output sample=" << fixture_.name
        << " step=" << view.step << " composite_cells=" << totals.cells
        << " volume=" << totals.volume << " mass=" << totals.mass << " momentum="
        << totals.momentum[0] << ',' << totals.momentum[1] << ',' << totals.momentum[2] << '\n';
}

void EvolutionSnapshot::fine(
    amr::FineSubstepView<Descriptor> const& view, amr_reference::ReferenceSample const& reference)
{
    require(!view.fine.macro,"fine artifact must not publish a synchronized macro field");
    auto frame=static_hierarchy::capture(view.fine.step,view.fine.physicalTime,view.fine.geometry,
                                        view.fine.populations.multifab(),nullptr);
    frame.level=view.level;
    check_populations(frame.populations,reference.populations,"fine artifact/reference");
    rows(view.substep==amr::FineSubstep::First ? "First" : "Second",view.coarsestStep,
         frame,reference.populations,nullptr);
}

void output_checks()
{
    std::string const root="artifacts/evolution-output";
    if (amrex::ParallelDescriptor::IOProcessor()) std::filesystem::remove_all(root);
    amrex::ParallelDescriptor::Barrier();
    for (int levels : {2,3}) {
        auto const fixture=central_fixture(levels,levels==2 ? (Descriptor::d==2 ? 32 : 16) : 8);
        auto owner=make_simulation(fixture,EvolutionProfile{});
        auto preparation=runtime::prepare_amr_evolution(*owner,evolution_intent(0.65),BGKDynamicsSelection{});
        require(preparation.ok(),"output evolution preparation");
        auto evolution=std::move(preparation).take();
        std::optional<amr_reference::HierarchyReference> reference;
        EvolutionSnapshot writer(root+'/'+fixture.name,fixture,2);
        owner->observe(true,[&](EvolutionView const& view) {
            reference.emplace(initial_reference(view,0.65,EvolutionProfile{}));
            writer.synchronized(view,reference->levels());
        });
        for (int step=1; step<=2; ++step) {
            auto const samples=reference->advance();
            std::size_t cursor=0;
            auto result=owner->evolve(evolution,1,true,
                [&](EvolutionView const& view) {
                    require(cursor==samples.size(),"output missing fine sample");
                    writer.synchronized(view,reference->levels());
                },[&](amr::FineSubstepView<Descriptor> const& view) {
                    require(cursor<samples.size(),"output extra fine sample");
                    writer.fine(view,samples[cursor++]);
                });
            require(result.ok(),result.error);
        }
    }
    amrex::Print() << Descriptor::name() << " C5 synchronized/fine artifacts and native readback: PASS\n";
}

} // namespace lbm::tests::amr_evolution
