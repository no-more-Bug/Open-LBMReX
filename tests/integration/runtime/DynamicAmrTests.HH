#ifndef LBM_TESTS_DYNAMICAMRTESTS_HH
#define LBM_TESTS_DYNAMICAMRTESTS_HH

#include "integration/runtime/DynamicAmrTestSupport.H"
#include "common/HostFieldSnapshot.H"
#include "lbm/amr/LbmAmrSimulation.H"
#include "lbm/amr/VorticityTagging.H"
#include "lbm/dynamics/BGKDynamicsSelection.H"
#include "runtime/LbmAmrPreparation.H"

#include <AMReX_BoxIterator.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>

namespace lbm::tests::dynamic_amr {

struct WaveProfile {
    bool thirdDirection = false;
    AMREX_GPU_HOST_DEVICE
    initialization::MacroscopicState operator()(amrex::GpuArray<Real,3> const& p, Real) const
    {
        Real const phase = Real(6.2831853071795864769)*(thirdDirection ? p[2] : p[0]);
        return {1,0,Real(0.04)*std::sin(phase),0};
    }
};

struct RegionTagging {
    static constexpr bool needsVelocity = false;
    int levels = 2;
    struct Rule {
        int levels;
        template<class Descriptor>
        AMREX_GPU_HOST_DEVICE bool operator()(amr::TaggingCell<Descriptor> const& cell) const
        {
            return cell.level < levels && cell.positionPhys[0] < 0.5;
        }
    };
    std::vector<std::string> validate(int) const { return {}; }
    Rule rule(int) const { return {levels}; }
};

struct EquilibriumEvaluator {
    template<class Descriptor>
    AMREX_GPU_HOST_DEVICE Real operator()(int q, amr::TaggingCell<Descriptor> const& cell) const
    {
        return Equilibrium<Descriptor>::compute(q,cell.rho,cell.uLb[0],cell.uLb[1],cell.uLb[2]);
    }
};

struct NonequilibriumTagging {
    static constexpr bool needsVelocity = false;
    struct Rule {
        EquilibriumEvaluator equilibrium;
        template<class Descriptor>
        AMREX_GPU_HOST_DEVICE bool operator()(amr::TaggingCell<Descriptor> const& cell) const
        {
            Real ratio = 0;
            for (int q = 0; q < Descriptor::q; ++q) {
                Real const eq = equilibrium(q,cell);
                ratio += std::abs(cell.populations[q]-eq)/std::abs(eq);
            }
            return ratio > 1.e-8;
        }
    };
    std::vector<std::string> validate(int) const { return {}; }
    Rule rule(int) const { return {}; }
};

// Test-owned counters observe real ErrorEst context without changing solver fields.
struct CenterConsistencyTagging {
    static constexpr bool needsVelocity = true;
    int* failures;
    int* witnesses;
    struct Rule {
        int* failures;
        int* witnesses;
        template<class Descriptor>
        AMREX_GPU_HOST_DEVICE bool operator()(amr::TaggingCell<Descriptor> const& cell) const
        {
            Real const expectedUy = 0.04*std::sin(Real(6.2831853071795864769)*cell.positionPhys[0]);
            bool mismatch = std::abs(cell.rho-1) > 2.e-12 ||
                std::abs(cell.uLb[0]) > 2.e-12 ||
                std::abs(cell.uLb[1]-expectedUy) > 2.e-12 ||
                std::abs(cell.uLb[2]) > 2.e-12;
            for (int q = 0; q < Descriptor::q; ++q)
                mismatch = mismatch ||
                    std::abs(cell.populations[q]-EquilibriumEvaluator{}(q,cell)) > 2.e-12;
            if (mismatch) amrex::Gpu::Atomic::Add(failures,1);
            auto const p = cell.populations;
            if (cell.coveredByFine &&
                std::abs(cell.velocityLb(p.i,p.j,p.k,1)-expectedUy) > 1.e-6)
                amrex::Gpu::Atomic::Add(witnesses,1);
            return cell.level == 0;
        }
    };
    std::vector<std::string> validate(int) const { return {}; }
    Rule rule(int) const { return {failures,witnesses}; }
};

template<class Descriptor>
void run_initialization_checks()
{
    for (int allowed : {1,2}) {
        for (int active : {0,allowed}) {
            auto spec = base_spec<Descriptor>(allowed);
            spec.regridInterval = 0;
            auto result = amr::prepare_dynamic_hierarchy<Descriptor>(
                spec,WaveProfile{Descriptor::d == 3},RegionTagging{active});
            for (auto const& error : result.errors) amrex::Print() << error << '\n';
            require(result.ok(),"dynamic initialization rejected");
            auto simulation = result.take();
            require(simulation->numLevels() == active+1 &&
                    simulation->numAllowedLevels() == allowed+1,"active versus allowed levels");
            simulation->observe(true,[&](auto const& view) {
                require(view.step == 0 && view.physicalTime == 0,"initial clocks");
                for (auto const& level : view.levels) {
                    require(level.macro != nullptr && level.populations.nComp() == Descriptor::q,
                            "initialized field publication");
                    if constexpr (Descriptor::d == 2)
                        require(level.geometry.Domain().length(2) == 1,"D2 fixed Nz");
                }
            });
            config::LbmConfigIntent intent;
            intent.parameterizationKind = config::LbmParameterizationKind::FromTau;
            intent.tau = 0.8; intent.uRefLb = 0.1; intent.uRefPhys = 1;
            auto prepared = runtime::prepare_amr_evolution(*simulation,intent,BGKDynamicsSelection{});
            require(prepared.ok(),"allowed-level parameter preparation");
            auto evolution = std::move(prepared).take();
            require(evolution.numLevels() == allowed+1,"inactive levels need prepared parameters");
            Real const coarseDt = 0.1/spec.nCell[0];
            for (int lev = 0; lev <= allowed; ++lev) {
                int const scale = 1 << (allowed-lev);
                require(evolution.level(lev).dtLb == scale,"fixed allowed lattice units");
                close(evolution.level(lev).dtPhys,coarseDt/(1<<lev),"fixed physical dt");
                close(evolution.level(lev).lbm.tau,0.5+0.3/scale,"fixed finest relaxation");
            }
            int observations = 0;
            auto run = simulation->evolve(evolution,2,false,[&](auto const& view) {
                ++observations;
                close(view.physicalTime,view.step*coarseDt,"frozen dynamic hierarchy time");
            });
            require(run.ok() && observations == 2,"N=0 frozen hierarchy evolution");
        }
    }

    // Single-level strip: buffering must wrap through the periodic x boundary.
    for (int width : {0,2}) {
        auto spec = base_spec<Descriptor>(1);
        spec.gapFillRadiusCells = 1;
        spec.tagBufferCells = {amrex::IntVect(width,width,Descriptor::d == 3 ? width : 0)};
        auto result = amr::prepare_dynamic_hierarchy<Descriptor>(spec,WaveProfile{},RegionTagging{1});
        require(result.ok() && result.simulation->numLevels() == 2,"buffered initial strip");
        result.simulation->observe(false,[&](auto const& view) {
            auto const& boxes = view.levels[1].boxes;
            int const n = spec.nCell[0];
            amrex::Long expected = n+4*width;
            for (int d = 1; d < Descriptor::d; ++d) expected *= 2*n;
            require(boxes.numPts() == expected,"independent periodic tag buffer coverage");
            require(boxes.contains(amrex::IntVect(2*n-1,0,0)) == (width > 0),
                    "buffer did not wrap across periodic boundary");
        });
    }
    {
        amrex::Gpu::DeviceScalar<int> failures(0), witnesses(0);
        auto spec = base_spec<Descriptor>(1);
        auto checked = amr::prepare_dynamic_hierarchy<Descriptor>(
            spec,WaveProfile{},CenterConsistencyTagging{failures.dataPtr(),witnesses.dataPtr()});
        require(checked.ok() && checked.simulation->numLevels() == 2,"center-state context case");
        int failed = failures.dataValue(), witnessed = witnesses.dataValue();
        amrex::ParallelDescriptor::ReduceIntSum(failed);
        amrex::ParallelDescriptor::ReduceIntSum(witnessed);
        require(failed == 0 && witnessed > 0,
                "raw center state mixed with averaged velocity, or covered-state check not exercised");
    }
    auto spec = base_spec<Descriptor>();
    auto kinetic = amr::prepare_dynamic_hierarchy<Descriptor>(spec,WaveProfile{},NonequilibriumTagging{});
    require(kinetic.ok() && kinetic.simulation->numLevels() == 1,
            "equilibrium case-local ratio rule should leave L0 only");
    amr::VorticityTagging vort(1,1,10);
    vort.thresholds = {{0.8,0.4}};
    auto initial = amr::prepare_dynamic_hierarchy<Descriptor>(spec,WaveProfile{Descriptor::d == 3},vort);
    require(initial.ok() && initial.simulation->numLevels() == 3,"vorticity initial multilevel hierarchy");
    vort.thresholds = {{0.8,0.4},{1000,500}};
    auto oneFine = amr::prepare_dynamic_hierarchy<Descriptor>(spec,WaveProfile{Descriptor::d == 3},vort);
    require(oneFine.ok() && oneFine.simulation->numLevels() == 2,
            "per-level vorticity threshold override");
    vort.thresholds = {{1000,500}};
    auto quiet = amr::prepare_dynamic_hierarchy<Descriptor>(spec,WaveProfile{},vort);
    require(quiet.ok() && quiet.simulation->numLevels() == 1,"case override must suppress refinement");
    vort.thresholds = {{2,2}};
    require(!amr::prepare_dynamic_hierarchy<Descriptor>(spec,WaveProfile{},vort).ok(),
            "invalid hysteresis thresholds accepted");
    amrex::Print() << "dynamic_initialization_pass d=" << Descriptor::d << '\n';
}

template<class Descriptor>
void run_tagging_checks()
{
    auto spec = base_spec<Descriptor>(1);
    auto layout = amr::prepare_dynamic_layout(spec,Descriptor::d);
    require(layout.ok(),"tagging geometry");
    auto const& geom = layout.layout->geometry;
    auto boxes = amrex::BoxArray(geom.Domain());
    boxes.maxSize(spec.maxGridSize[0]);
    amrex::DistributionMapping mapping(boxes);
    LatticeField<Descriptor> populations(boxes,mapping);
    WaveProfile profile{Descriptor::d == 3};
    operators::EquilibriumPopulationInitializationOperator<Descriptor,WaveProfile>(profile,geom)
        .apply(populations,0);
    amr::SynchronizedVelocity<Descriptor> velocity({{geom,populations.multifab()}},spec.refRatio);
    auto host = host_field_snapshot(velocity.level(0));
    for (amrex::MFIter mfi(host); mfi.isValid(); ++mfi) {
        auto const u = host.const_array(mfi);
        for (amrex::BoxIterator cell(mfi.fabbox()); cell.ok(); ++cell) {
            auto const& p = cell();
            int const direction = Descriptor::d == 3 ? 2 : 0;
            Real const x = (p[direction]+0.5)*geom.CellSize(direction);
            close(u(p[0],p[1],p[2],1),0.04*std::sin(6.2831853071795864769*x),
                  "independent periodic velocity halo");
        }
    }

    // Exact binary linear velocity makes threshold equality independent of roundoff.
    amrex::MultiFab linear(boxes,mapping,3,stencil_extent<Descriptor>());
    amrex::MultiFab flags(boxes,mapping,5,0);
    auto const dx = geom.CellSizeArray();
    auto const inverseDx = geom.InvCellSizeArray();
    for (amrex::MFIter mfi(linear); mfi.isValid(); ++mfi) {
        auto const u = linear.array(mfi);
        amrex::ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
            u(i,j,k,0)=0; u(i,j,k,1)=(i+0.5)*dx[0]/32; u(i,j,k,2)=0;
        });
    }
    for (amrex::MFIter mfi(populations.multifab()); mfi.isValid(); ++mfi) {
        auto const f = populations.multifab().array(mfi);
        auto const u = linear.const_array(mfi);
        auto const out = flags.array(mfi);
        amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
            ConstPopulationCell pop{f,i,j,k};
            Real rho,ux,uy,uz;
            kernels::MacroscopicKernel<Descriptor>::compute(pop,rho,ux,uy,uz);
            amr::TaggingCell<Descriptor> cell{pop,rho,{ux,uy,uz},{0,0,0},inverseDx,u,0,0,false};
            out(i,j,k,0) = amr::VorticityTaggingRule{{0.5,0.25},16}(cell);
            cell.coveredByFine = true;
            out(i,j,k,1) = amr::VorticityTaggingRule{{0.5,0.25},16}(cell);
            out(i,j,k,2) = amr::VorticityTaggingRule{{1,0.5},16}(cell);
            out(i,j,k,3) = NonequilibriumTagging::Rule{}(cell);
            f(i,j,k,1) += 1.e-4;
            f(i,j,k,Descriptor::opposite(1)) += 1.e-4;
            f(i,j,k,0) -= 2.e-4;
            out(i,j,k,4) = NonequilibriumTagging::Rule{}(cell);
        });
    }
    auto hostFlags = host_field_snapshot(flags);
    for (amrex::MFIter mfi(hostFlags); mfi.isValid(); ++mfi) {
        auto const result = hostFlags.const_array(mfi);
        for (amrex::BoxIterator cell(mfi.validbox()); cell.ok(); ++cell) {
            auto const& p = cell();
            for (int component = 0; component < 5; ++component)
                require(result(p[0],p[1],p[2],component) == (component == 1 || component == 4 ? 1 : 0),
                        "strict threshold/hysteresis or custom nonequilibrium rule");
        }
    }
    amrex::Print() << "dynamic_tagging_pass d=" << Descriptor::d << '\n';
}

template<class Descriptor>
void run_velocity_interface_checks()
{
    auto spec = base_spec<Descriptor>(1);
    auto layout = amr::prepare_dynamic_layout(spec,Descriptor::d);
    auto const coarseGeom = layout.layout->geometry;
    auto const fineGeom = amrex::refine(coarseGeom,spec.refRatio);
    amrex::BoxArray coarseBoxes(coarseGeom.Domain());
    coarseBoxes.maxSize(spec.maxGridSize[0]);
    amrex::IntVect lo(4,4,Descriptor::d == 3 ? 4 : 0);
    amrex::IntVect hi(11,11,Descriptor::d == 3 ? 11 : 0);
    amrex::BoxArray fineBoxes(amrex::refine(amrex::Box(lo,hi),spec.refRatio));
    fineBoxes.maxSize(spec.maxGridSize[0]);
    LatticeField<Descriptor> coarse(coarseBoxes,amrex::DistributionMapping(coarseBoxes));
    LatticeField<Descriptor> fine(fineBoxes,amrex::DistributionMapping(fineBoxes));
    for (int lev = 0; lev < 2; ++lev) {
        auto& field = lev == 0 ? coarse : fine;
        auto const dx = (lev == 0 ? coarseGeom : fineGeom).CellSizeArray();
        for (amrex::MFIter mfi(field.multifab()); mfi.isValid(); ++mfi) {
            auto const f = field.multifab().array(mfi);
            amrex::ParallelFor(mfi.validbox(),Descriptor::q,
                [=] AMREX_GPU_DEVICE(int i,int j,int k,int q) {
                    Real const uy = (i+0.5)*dx[0]/32;
                    Real const uz = Descriptor::d == 3 ? (j+0.5)*dx[1]/64 : 0;
                    Real const cu = Descriptor::cy(q)*uy+Descriptor::cz(q)*uz;
                    f(i,j,k,q)=Descriptor::w(q)*(1+3*cu+4.5*cu*cu-1.5*(uy*uy+uz*uz));
                });
        }
    }
    amr::SynchronizedVelocity<Descriptor> velocity(
        {{coarseGeom,coarse.multifab()},{fineGeom,fine.multifab()}},spec.refRatio);
    auto host = host_field_snapshot(velocity.level(1));
    for (amrex::MFIter mfi(host); mfi.isValid(); ++mfi) {
        auto const u = host.const_array(mfi);
        for (amrex::BoxIterator cell(mfi.fabbox()); cell.ok(); ++cell) {
            auto const& p = cell();
            close(u(p[0],p[1],p[2],1),(p[0]+0.5)*fineGeom.CellSize(0)/32,
                  "coarse-fine linear velocity halo");
            close(u(p[0],p[1],p[2],2),Descriptor::d == 3 ? (p[1]+0.5)*fineGeom.CellSize(1)/64 : 0,
                  "coarse-fine full vector halo");
        }
    }
    amrex::Print() << "dynamic_velocity_interface_pass d=" << Descriptor::d << '\n';
}
} // namespace lbm::tests::dynamic_amr
#endif
