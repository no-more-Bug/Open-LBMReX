#include "integration/runtime/FixedThicknessCylinderTests.H"
#include "integration/runtime/FixedThicknessTestSupport.H"
#include "lbm/boundary/core/BoundaryResolver.H"
#include "lbm/boundary/internal/AnalyticCylinderGeometry.H"
#include "lbm/boundary/internal/EbGeometryBackend.H"
#include "lbm/boundary/internal/InternalSolidPopulationExecutor.H"
#include "lbm/boundary/internal/force/InternalSolidForceCollector.H"
#include "lbm/boundary/metadata/InternalCutLinkCompiler.H"

#include <AMReX_RealBox.H>

#include <limits>

namespace lbm::tests::fixed_thickness {
namespace {

constexpr int nx = 24, ny = 16;
constexpr Real h = 0.05, centerX = 0.6, centerY = 0.4, radius = 0.2;
// Independent velocity/opposite table for the tests' D2Q9 ordering.
constexpr std::array<int, 9> cx{0, -1, -1, -1, 0, 1, 1, 1, 0};
constexpr std::array<int, 9> cy{0, 1, 0, -1, -1, -1, 0, 1, 1};
constexpr std::array<int, 9> opposite{0, 5, 6, 7, 8, 1, 2, 3, 4};

void geometry_close(Real actual, Real expected)
{
    require(std::isfinite(actual) && std::abs(actual-expected) <=
        64.0*std::numeric_limits<Real>::epsilon()*(1.0+std::abs(expected)),
        "cylinder analytic geometry/reconstruction mismatch");
}

void force_close(Real actual, Real expected)
{
    require(std::isfinite(actual) && std::abs(actual-expected) <=
        4096.0*std::numeric_limits<Real>::epsilon()*(1.0+std::abs(expected)),
        "cylinder independent momentum-exchange mismatch");
}

bool solid(int i, int j)
{
    return std::hypot((i+0.5)*h-centerX, (j+0.5)*h-centerY) < radius;
}

Real circle_q(int i, int j, int direction)
{
    const Real x = (i+0.5)*h-centerX, y = (j+0.5)*h-centerY;
    const Real dx = h*cx[direction], dy = h*cy[direction];
    const Real a = dx*dx+dy*dy, b = 2.0*(x*dx+y*dy);
    const Real c = x*x+y*y-radius*radius;
    return (-b-std::sqrt(b*b-4.0*a*c))/(2.0*a);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real synthetic(int i, int j, int q) noexcept
{
    return 0.3 + 0.001*i + 0.002*j + 0.005*q;
}

Real reconstructed(int i, int j, int d, Real q, boundary::BoundaryScheme scheme)
{
    const Real forward = synthetic(i,j,d), backward = synthetic(i,j,opposite[d]);
    const Real donor = synthetic(i-cx[d],j-cy[d],d);
    // Same published scalar oracles as unit/lbm/BoundaryFormulaTests.H.
    if (scheme == boundary::BoundaryScheme::Bouzidi) {
        return q < 0.5 ? 2.0*q*forward+(1.0-2.0*q)*donor :
            (forward+(2.0*q-1.0)*backward)/(2.0*q);
    }
    return (q*backward+q*forward+(1.0-q)*donor)/(1.0+q);
}

boundary::InternalBoundarySpec cylinder(boundary::BoundaryScheme scheme)
{
    boundary::InternalBoundarySpec spec;
    spec.name = spec.objectName = "cylinder";
    spec.geometry = "cylinder";
    spec.hasPhysical = true;
    spec.scheme = scheme;
    spec.profile = boundary::BoundaryProfileKind::None;
    spec.centerPhys = {centerX, centerY, h/2.0};
    spec.hasCenterPhys = true;
    spec.radiusPhys = radius;
    spec.hasRadiusPhys = true;
    spec.axis = "z";
    spec.hasAxis = true;
    return spec;
}

Program cylinder_program(boundary::BoundaryScheme scheme)
{
    Program program;
    program.mesh.nCell = {nx,ny,1};
    program.mesh.envelopeLo = {0.0,0.0,0.0};
    program.mesh.envelopeHi = {1.2,0.8,h};
    program.mesh.isPeriodic = {0,0,1};
    program.lbm.parameterizationKind = config::LbmParameterizationKind::FromNuLb;
    program.lbm.rhoRefLb = 1.0;
    program.lbm.uRefLb = speed;
    program.lbm.uRefPhys = 1.0;
    program.lbm.nuLb = viscosity;
    program.lbm.lengthRefPhys = 2.0*radius;
    program.lbm.lengthRefLb = 8.0;
    program.run.maxStep = steps;
    program.run.finalTimePhys = 1.0;
    program.run.name = "fixed_thickness_cylinder";
    program.initialization.intent.kind = initialization::InitializationKind::Uniform;
    program.initialization.intent.rhoLb = 1.0;
    program.initialization.intent.uniformULb = amrex::Array<Real,3>{speed,0.0,0.0};
    for (auto side : {boundary::DomainSide::XLo,boundary::DomainSide::XHi,
                      boundary::DomainSide::YLo,boundary::DomainSide::YHi}) {
        boundary::DomainBoundarySpec wall;
        wall.name = "wall_" + std::to_string(static_cast<int>(side));
        wall.location = side;
        wall.scheme = boundary::BoundaryScheme::HalfwayBounceBack;
        program.domainBoundaries.push_back(wall);
    }
    program.internalBoundaries.push_back(cylinder(scheme));
    return program;
}

// Recreate the real setup pipeline only after the prior runtime released its EB owner.
void check_geometry_and_force(Trace const& trace, boundary::BoundaryScheme scheme)
{
    auto const& geom = trace.frames.front().geometry;
    amrex::BoxArray ba(geom.Domain());
    ba.maxSize(trace.maxGridSize);
    amrex::DistributionMapping dm(ba);
    auto resolved = boundary::resolve_boundary_specs<Descriptor>({}, {cylinder(scheme)}, geom);
    require(resolved.ok(), "cylinder boundary resolution failed");
    auto query = boundary::build_internal_geometry_query(resolved.resolved, geom);
    require(query.ok(), query.errorMessage());
    boundary::EbGeometryBackend backend;
    std::string error;
    require(backend.define(query.query, geom, ba, dm, error), error);
    boundary::CenterPhaseField<Descriptor> phase(ba,dm);
    require(phase.build(query.query, geom, error), error);
    boundary::InternalCutLinkCompiler<Descriptor> compiler;
    require(compiler.build(query.query, phase, geom, {}, error), error);
    require(boundary::check_internal_links_against_eb(*backend.factory(), compiler.records(), error), error);
    boundary::InternalBoundaryMetadataStorage storage;
    require(storage.define(compiler.records(), resolved.resolved, geom, ba, dm, error), error);

    // Integer field snapshot uses an explicit device-to-host copy as the Real helper does.
    auto const& devicePhase = phase.phase().multifab();
    amrex::iMultiFab hostPhase(ba,dm,1,devicePhase.nGrowVect(),
        amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
    amrex::Gpu::streamSynchronize();
    amrex::Long solids = 0;
    for (amrex::MFIter mfi(devicePhase); mfi.isValid(); ++mfi) {
        amrex::Gpu::dtoh_memcpy(hostPhase[mfi].dataPtr(), devicePhase[mfi].dataPtr(),
            devicePhase[mfi].size()*sizeof(int));
        auto const values = hostPhase.const_array(mfi);
        for (amrex::BoxIterator it(mfi.validbox()); it.ok(); ++it) {
            const bool inside = solid(it()[0],it()[1]);
            require(values(it()) == static_cast<int>(inside), "analytic cell-center phase mismatch");
            solids += inside;
        }
    }
    amrex::ParallelDescriptor::ReduceLongSum(solids);
    require(solids == phase.report().solidCount && solids > 0 &&
        phase.report().fluidCount+solids == nx*ny && phase.report().onSurfaceCount == 0,
        "cylinder phase count mismatch");
    boundary::AnalyticCylinderGeometry geometry(resolved.resolved.internal.front());
    auto const tolerance = geometry.surfaceTolerance(geom);
    require(tolerance.valid, "cylinder geometry tolerance invalid");
    geometry_close(tolerance.physicalLength,64.0*std::numeric_limits<Real>::epsilon()*1.2);

    std::vector<int> counts(nx*ny*9,0);
    std::vector<Real> fractions(nx*ny*9,0.0);
    auto view = storage.hostView();
    for (amrex::Long n=0; n<view.count; ++n) {
        auto const& link = view.canonical[n];
        const int i=link.fluidCell[0], j=link.fluidCell[1], d=link.population;
        require(i>=0 && i<nx && j>=0 && j<ny && d>0 && d<9 &&
            link.fluidCell[2]==0 && link.solidCell[2]==0 && link.objectId==0 && link.boundaryId==0 &&
            link.surfaceCoincidentFluid==0 && link.oppositePopulation==opposite[d],
            "cylinder link identity mismatch");
        require(!solid(i,j) && solid(i+cx[d],j+cy[d]) &&
            link.solidCell[0]==i+cx[d] && link.solidCell[1]==j+cy[d],
            "cylinder analytic link endpoints mismatch");
        const Real q=circle_q(i,j,d), x=(i+0.5)*h+q*h*cx[d], y=(j+0.5)*h+q*h*cy[d];
        geometry_close(link.qFromFluid,q);
        geometry_close(link.intersection[0],x); geometry_close(link.intersection[1],y);
        geometry_close(link.intersection[2],geom.CellSize(2)/2.0);
        geometry_close(link.normalSolidToFluid[0],(x-centerX)/radius);
        geometry_close(link.normalSolidToFluid[1],(y-centerY)/radius);
        geometry_close(link.normalSolidToFluid[2],0.0);
        ++counts[(j*nx+i)*9+d]; fractions[(j*nx+i)*9+d]=link.qFromFluid;
    }
    amrex::ParallelDescriptor::ReduceIntSum(counts.data(),static_cast<int>(counts.size()));
    amrex::ParallelDescriptor::ReduceRealSum(fractions.data(),static_cast<int>(fractions.size()));

    LatticeField<Descriptor> collided(ba,dm), returned(ba,dm);
    returned.multifab().setVal(-13.0);
    for (amrex::MFIter mfi(collided.multifab()); mfi.isValid(); ++mfi) {
        auto const f=collided.multifab().array(mfi);
        amrex::ParallelFor(mfi.validbox(),9,
            [=] AMREX_GPU_DEVICE(int i,int j,int k,int q) noexcept { f(i,j,k,q)=synthetic(i,j,q); });
    }
    collided.fillBoundary(geom);
    boundary::InternalSolidPopulationExecutor<Descriptor>{}.apply(storage,collided,returned);
    auto observed=gather_xy(returned.multifab(),geom.Domain());
    std::vector<Real> expected(nx*ny*9,-13.0);
    std::array<Real,3> force{};
    int links=0, lower=0, upper=0;
    std::ofstream output;
    if (amrex::ParallelDescriptor::IOProcessor()) {
        output.open(trace.directory+"/cut-links.csv");
        output << std::setprecision(17) << "i,j,direction,object_id,q,returned,link_fx,link_fy\n";
    }
    for (int j=0; j<ny; ++j) {
        for (int i=0; i<nx; ++i) {
            for (int d=0; d<9; ++d) {
                const bool cut=d>0 && !solid(i,j) && solid(i+cx[d],j+cy[d]);
                require(counts[(j*nx+i)*9+d]==static_cast<int>(cut),
                    "missing/duplicate/unexpected cut-link owner");
                if (!cut) { continue; }
                ++links;
                const Real q=circle_q(i,j,d), value=reconstructed(i,j,d,q,scheme);
                geometry_close(fractions[(j*nx+i)*9+d],q);
                expected[(j*nx+i)*9+opposite[d]]=value;
                const Real momentum=synthetic(i,j,d)+value;
                force[0]+=momentum*cx[d]; force[1]+=momentum*cy[d];
                lower+=q<0.5; upper+=q>=0.5;
                if (amrex::ParallelDescriptor::IOProcessor()) {
                    output << i << ',' << j << ',' << d << ",0," << q << ',' << value
                           << ',' << momentum*cx[d] << ',' << momentum*cy[d] << '\n';
                }
            }
        }
    }
    require(lower>0 && upper>0 && links==storage.report().globalRecordCount,
        "cylinder fixture does not cover both reconstruction branches");
    for (std::size_t n=0; n<expected.size(); ++n) { geometry_close(observed[n],expected[n]); }
    boundary::InternalSolidForceCollector<Descriptor> collector;
    require(collector.define(storage,ba,dm,{resolved.resolved.internal[0].boundaryId},error),error);
    collector.recordInternalForceSample(collided,returned,{0,1,1,0,speed*h});
    auto batch=collector.takeGlobalBoundaryForceBatch();
    require(batch.size()==1 && batch[0].step.populationStep==1 && batch[0].boundaryForces.size()==1 &&
        batch[0].boundaryForces[0].boundaryId.value==0, "synthetic force batch identity mismatch");
    for (int d=0; d<3; ++d) { force_close(batch[0].boundaryForces[0].forceLb[d],force[d]); }
    require(!amrex::ParallelDescriptor::IOProcessor() || output.good(),"cannot write cylinder cut links");
    amrex::Print() << "cylinder_oracle_pass solids=" << solids << " links=" << links
        << " q_lt_half=" << lower << " q_ge_half=" << upper
        << " Fx=" << force[0] << " Fy=" << force[1] << '\n';
}

void check_cylinder(Trace const& trace, boundary::BoundaryScheme scheme)
{
    require(trace.forces.size()==steps,"missing or duplicate cylinder force sample");
    std::ofstream output;
    if (amrex::ParallelDescriptor::IOProcessor()) {
        output.open(trace.directory+"/forces.csv");
        output << std::setprecision(17) << "population_step,force_time_lb,force_time_phys,object_id,Fx_lb,Fy_lb,Fz_lb,span_lb,Cd,Cl\n";
    }
    constexpr Real spanLb=1.0, diameterLb=8.0;
    const Real denominator=1.0*speed*speed*diameterLb*spanLb;
    for (int n=0; n<steps; ++n) {
        auto const& sample=trace.forces[n];
        require(sample.step.populationStep==n+1 && sample.objectForces.size()==1 &&
            sample.objectForces[0].objectId.value==0,"cylinder runtime force identity mismatch");
        auto const& force=sample.objectForces[0].forceLb;
        require(std::isfinite(force[0]) && std::isfinite(force[1]) && force[2]==0.0,
            "invalid cylinder runtime force");
        const Real forceTime=sample.step.populationStep-0.5;
        if (amrex::ParallelDescriptor::IOProcessor()) {
            output << sample.step.populationStep << ',' << forceTime << ',' << forceTime*speed*h
                << ",0," << force[0] << ',' << force[1] << ',' << force[2] << ',' << spanLb
                << ',' << 2.0*force[0]/denominator << ',' << 2.0*force[1]/denominator << '\n';
        }
    }
    for (auto const& frame : trace.frames) {
        for (std::size_t cell=0; cell<frame.macro.size()/4; ++cell) {
            require(frame.macro[cell*4+3]==0.0,"nonzero inactive cylinder velocity");
        }
    }
    require(!amrex::ParallelDescriptor::IOProcessor() || output.good(),"cannot write cylinder forces");
    check_geometry_and_force(trace,scheme);
}

} // namespace

void run_cylinder_checks()
{
    for (auto scheme : {boundary::BoundaryScheme::Yms,boundary::BoundaryScheme::Bouzidi}) {
        const std::string name=scheme==boundary::BoundaryScheme::Yms ? "cylinder_yms" : "cylinder_bouzidi";
        check_case(name,cylinder_program(scheme),[scheme](Trace const& trace) { check_cylinder(trace,scheme); });
    }
}

} // namespace lbm::tests::fixed_thickness
