#include "integration/runtime/FixedThicknessTestSupport.H"
#include "integration/runtime/FixedThicknessCylinderTests.H"

#include <AMReX.H>

namespace lbm::tests::fixed_thickness {
namespace {

// Tests-owned D2Q9 algebra: the oracle does not call production moments/feq.
constexpr std::array<int, 9> cx{0, -1, -1, -1, 0, 1, 1, 1, 0};
constexpr std::array<int, 9> cy{0, 1, 0, -1, -1, -1, 0, 1, 1};
constexpr std::array<Real, 9> weights{4.0/9.0, 1.0/36.0, 1.0/9.0,
    1.0/36.0, 1.0/9.0, 1.0/36.0, 1.0/9.0, 1.0/36.0, 1.0/9.0};

void equilibrium(Frame const& frame, std::size_t cell, Real rho, Real ux, Real uy)
{
    close(frame.macro[cell * 4], rho, "analytic density");
    close(frame.macro[cell * 4 + 1], ux, "analytic ux");
    close(frame.macro[cell * 4 + 2], uy, "analytic uy");
    for (int q = 0; q < 9; ++q) {
        const Real cu = cx[q] * ux + cy[q] * uy;
        close(frame.populations[cell * 9 + q],
            weights[q] * rho * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*(ux*ux + uy*uy)),
            "independent D2Q9 equilibrium");
    }
}

void moments_and_integrals(Trace const& trace, bool weightX = false, bool weightY = false)
{
    std::ofstream output;
    if (amrex::ParallelDescriptor::IOProcessor()) {
        output.open(trace.directory + "/integrals.csv");
        output << std::setprecision(17) << "step,mass_per_span_lb,mx_per_span_lb,my_per_span_lb,mass_slab_lb,H_lb\n";
    }
    for (auto const& frame : trace.frames) {
        auto const& geom = frame.geometry;
        const int nx = geom.Domain().length(0), ny = geom.Domain().length(1);
        const Real area = geom.CellSize(0) * geom.CellSize(1) / (frame.referenceDx * frame.referenceDx);
        const Real volume = geom.CellSize(0) * geom.CellSize(1) * geom.CellSize(2) /
            (frame.referenceDx * frame.referenceDx * frame.referenceDx);
        const Real height = geom.ProbLength(2) / frame.referenceDx;
        Real mass = 0.0, mx = 0.0, my = 0.0, slabMass = 0.0;
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const int cell = j * nx + i;
                Real rho = 0.0, jx = 0.0, jy = 0.0;
                for (int q = 0; q < 9; ++q) {
                    const Real f = frame.populations[cell * 9 + q];
                    rho += f; jx += f * cx[q]; jy += f * cy[q];
                }
                require(std::isfinite(rho) && rho > 0.0, "nonpositive density in fluid fixture");
                close(frame.macro[cell * 4], rho, "independent density moment");
                close(frame.macro[cell * 4 + 1], jx/rho, "independent x momentum");
                close(frame.macro[cell * 4 + 2], jy/rho, "independent y momentum");
                require(frame.macro[cell * 4 + 3] == 0.0, "nonzero inactive velocity");
                Real weight = 1.0;
                if (weightX && (i == 0 || i == nx-1)) { weight *= 0.5; }
                if (weightY && (j == 0 || j == ny-1)) { weight *= 0.5; }
                const Real density = frame.macro[cell * 4];
                mass += weight * area * density;
                mx += weight * area * density * frame.macro[cell * 4 + 1];
                my += weight * area * density * frame.macro[cell * 4 + 2];
                slabMass += weight * volume * rho;
            }
        }
        close(slabMass, mass * height, "slab mass versus unit-span mass");
        if (amrex::ParallelDescriptor::IOProcessor()) {
            output << frame.step << ',' << mass << ',' << mx << ',' << my << ',' << slabMass << ',' << height << '\n';
        }
    }
    require(!amrex::ParallelDescriptor::IOProcessor() || output.good(), "cannot write integral observations");
}

Program uniform_program()
{
    Program program;
    program.mesh.nCell = {32, 32, 1};
    program.mesh.envelopeLo = {0.0, 0.0, 0.0};
    program.mesh.envelopeHi = {1.0, 1.0, 1.0/32.0};
    program.mesh.isPeriodic = {1, 1, 1};
    program.lbm.parameterizationKind = config::LbmParameterizationKind::FromNuLb;
    program.lbm.rhoRefLb = 1.0;
    program.lbm.uRefLb = speed;
    program.lbm.uRefPhys = 1.0;
    program.lbm.nuLb = viscosity;
    program.lbm.lengthRefLb = 32.0;
    program.lbm.lengthRefPhys = 1.0;
    program.run.maxStep = steps;
    program.run.finalTimePhys = 1.0;
    program.run.name = "fixed_thickness";
    program.initialization.intent.kind = initialization::InitializationKind::Uniform;
    program.initialization.intent.rhoLb = 1.0;
    program.initialization.intent.uniformULb = amrex::Array<Real,3>{speed, 0.0, 0.0};
    return program;
}

Program uniform_io(boundary::BoundaryScheme scheme)
{
    auto program = uniform_program();
    program.mesh.isPeriodic[0] = 0;
    if (scheme == boundary::BoundaryScheme::GuoExtrapolation) {
        program.mesh.nCell[0] = 33;
        program.mesh.envelopeLo[0] = -1.0/64.0;
        program.mesh.envelopeHi[0] = 1.0 + 1.0/64.0;
    }
    for (auto side : {boundary::DomainSide::XLo, boundary::DomainSide::XHi}) {
        boundary::DomainBoundarySpec spec;
        const bool inlet = side == boundary::DomainSide::XLo;
        spec.name = inlet ? "inlet" : "outlet";
        spec.location = side;
        spec.physical = inlet ? boundary::BoundaryPhysical::VelocityInlet : boundary::BoundaryPhysical::VelocityOutlet;
        spec.scheme = scheme;
        spec.profile = boundary::BoundaryProfileKind::Uniform;
        spec.prescribedULb = {speed, 0.0, 0.0};
        spec.hasPrescribedULb = true;
        program.domainBoundaries.push_back(spec);
    }
    return program;
}

Program poiseuille_program()
{
    auto program = uniform_program();
    program.mesh.nCell = {65, 17, 1};
    program.mesh.envelopeLo = {-1.0/32.0, -1.0/32.0, 0.0};
    program.mesh.envelopeHi = {4.0+1.0/32.0, 1.0+1.0/32.0, 1.0/16.0};
    program.mesh.isPeriodic = {0, 0, 1};
    program.lbm.lengthRefLb = 16.0;
    program.initialization.intent = {};
    program.initialization.intent.kind = initialization::InitializationKind::Poiseuille;
    program.initialization.intent.rhoLb = 1.0;
    program.initialization.intent.poiseuilleUMaxLb = speed;
    program.initialization.intent.poiseuilleRhoXLoLb = 1.006;
    program.initialization.intent.poiseuilleRhoXHiLb = 1.0;
    for (auto side : {boundary::DomainSide::XLo, boundary::DomainSide::XHi,
                      boundary::DomainSide::YLo, boundary::DomainSide::YHi}) {
        boundary::DomainBoundarySpec spec;
        spec.name = "poiseuille_" + std::to_string(static_cast<int>(side));
        spec.location = side;
        spec.scheme = boundary::BoundaryScheme::GuoExtrapolation;
        if (side == boundary::DomainSide::XLo || side == boundary::DomainSide::XHi) {
            const bool inlet = side == boundary::DomainSide::XLo;
            spec.physical = inlet ? boundary::BoundaryPhysical::PressureInlet : boundary::BoundaryPhysical::PressureOutlet;
            spec.profile = boundary::BoundaryProfileKind::Constant;
            spec.rhoLb = inlet ? 1.006 : 1.0;
            spec.hasRhoLb = true;
        }
        program.domainBoundaries.push_back(spec);
    }
    return program;
}

void check_uniform(Trace const& trace)
{
    for (auto const& frame : trace.frames) {
        for (std::size_t cell = 0; cell < frame.macro.size()/4; ++cell) {
            equilibrium(frame, cell, 1.0, speed, 0.0);
        }
    }
    moments_and_integrals(trace, trace.frames.front().geometry.Domain().length(0) == 33);
}

void check_poiseuille(Trace const& trace)
{
    auto const& frame = trace.frames.front();
    const int nx = frame.geometry.Domain().length(0), ny = frame.geometry.Domain().length(1);
    // The initial bulk is equilibrium; the Guo boundary cells have their own reconstruction.
    for (int j = 1; j < ny-1; ++j) {
        for (int i = 1; i < nx-1; ++i) {
            const Real x = i/16.0, y = j/16.0;
            equilibrium(frame, j*nx+i, 1.006 - 0.006*x/4.0, 4.0*speed*y*(1.0-y), 0.0);
        }
    }
    moments_and_integrals(trace, true, true);
}

void check_tgv(Trace const& trace)
{
    const Real pi = std::acos(Real(-1.0));
    auto const& initial = trace.frames.front();
    auto const& final = trace.frames.back();
    Real initialEnergy = 0.0, finalEnergy = 0.0, error = 0.0, exactNorm = 0.0;
    const Real decay = std::exp(-viscosity * 2.0 * std::pow(2.0*pi/32.0, 2) * steps);
    for (int j = 0; j < 32; ++j) {
        for (int i = 0; i < 32; ++i) {
            const int cell = j*32+i;
            const Real x = 2.0*pi*(i+0.5)/32.0, y = 2.0*pi*(j+0.5)/32.0;
            const Real ux = -speed*std::cos(x)*std::sin(y), uy = speed*std::sin(x)*std::cos(y);
            equilibrium(initial, cell, 1.0 - 0.75*speed*speed*(std::cos(2*x)+std::cos(2*y)), ux, uy);
            const Real actualX = final.macro[cell*4+1], actualY = final.macro[cell*4+2];
            initialEnergy += ux*ux + uy*uy;
            finalEnergy += actualX*actualX + actualY*actualY;
            error += std::pow(actualX-ux*decay,2) + std::pow(actualY-uy*decay,2);
            exactNorm += (ux*ux+uy*uy)*decay*decay;
        }
    }
    require(finalEnergy < initialEnergy, "positive-viscosity TGV did not decay");
    amrex::Print() << "tgv_analytic_observation relative_l2_u=" << std::sqrt(error/exactNorm)
                   << " energy_ratio=" << finalEnergy/initialEnergy << '\n';
    moments_and_integrals(trace);
}

} // namespace
} // namespace lbm::tests::fixed_thickness

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        using namespace lbm::tests::fixed_thickness;
        check_case("uniform_periodic", uniform_program(), check_uniform);
        auto tgv = uniform_program();
        tgv.initialization.intent = {};
        tgv.initialization.intent.kind = lbm::initialization::InitializationKind::TaylorGreen;
        tgv.initialization.intent.rhoLb = 1.0;
        tgv.initialization.intent.taylorGreenVelocityAmplitudeLb = speed;
        check_case("tgv", tgv, check_tgv);
        check_case("uniform_hbb", uniform_io(lbm::boundary::BoundaryScheme::HalfwayBounceBack), check_uniform);
        check_case("uniform_guo", uniform_io(lbm::boundary::BoundaryScheme::GuoExtrapolation), check_uniform);
        check_case("poiseuille", poiseuille_program(), check_poiseuille);
        run_cylinder_checks();
        amrex::Print() << "Fixed-thickness field/boundary/output checks: PASS\n";
    }
    amrex::Finalize();
}
