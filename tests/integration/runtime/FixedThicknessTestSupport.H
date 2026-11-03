#ifndef LBM_TESTS_FIXEDTHICKNESSTESTSUPPORT_H
#define LBM_TESTS_FIXEDTHICKNESSTESTSUPPORT_H

#include "common/HostFieldSnapshot.H"
#include "lbm/descriptors/D2Q9Descriptor.H"
#include "lbm/dynamics/BGKDynamicsSelection.H"
#include "runtime/SingleLevelPlotfile.H"
#include "runtime/SingleLevelPreparation.H"
#include "runtime/SingleLevelRun.H"

#include <AMReX_BoxIterator.H>
#include <AMReX_PlotFileUtil.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace lbm::tests::fixed_thickness {

using Real = amrex::Real;
using Descriptor = D2Q9Descriptor;
constexpr int steps = 20;
constexpr Real speed = 0.02;
constexpr Real viscosity = 0.05;

inline void require(bool condition, std::string const& message)
{
    if (!condition) {
        amrex::Abort(message);
    }
}

inline void close(Real actual, Real expected, char const* subject)
{
    const Real tolerance = std::max(Real(1.0e-12),
        Real(1.0e-12) * std::max(std::abs(actual), std::abs(expected)));
    if (!std::isfinite(actual) || !std::isfinite(expected) ||
        std::abs(actual - expected) > tolerance) {
        std::ostringstream message;
        message << std::setprecision(17) << subject << ": actual=" << actual
                << " expected=" << expected << " tolerance=" << tolerance;
        amrex::Abort(message.str());
    }
}

// One valid owner per cell; retain a common order across Box/rank layouts.
inline std::vector<Real> gather_xy(amrex::MultiFab const& source, amrex::Box const& domain)
{
    require(domain.length(2) == 1, "thickness snapshot requires one z plane");
    const int nx = domain.length(0);
    const int cells = nx * domain.length(1);
    const int components = source.nComp();
    std::vector<Real> values(static_cast<std::size_t>(cells * components), 0.0);
    std::vector<int> owners(static_cast<std::size_t>(cells), 0);
    auto const host = host_field_snapshot(source);
    for (amrex::MFIter mfi(host); mfi.isValid(); ++mfi) {
        auto const data = host.const_array(mfi);
        for (amrex::BoxIterator it(mfi.validbox()); it.ok(); ++it) {
            auto const cell = it();
            const int index = (cell[1] - domain.smallEnd(1)) * nx + cell[0] - domain.smallEnd(0);
            ++owners[index];
            for (int n = 0; n < components; ++n) {
                values[index * components + n] = data(cell, n);
            }
        }
    }
    amrex::ParallelDescriptor::ReduceIntSum(owners.data(), cells);
    amrex::ParallelDescriptor::ReduceRealSum(values.data(), static_cast<int>(values.size()));
    for (int count : owners) {
        require(count == 1, "snapshot has missing or duplicate valid owners");
    }
    return values;
}

struct Frame {
    int step;
    Real physicalTime;
    Real referenceDx;
    amrex::Geometry geometry;
    std::vector<Real> populations;
    std::vector<Real> macro;
};

struct Trace {
    std::string directory;
    bool withForce = false;
    int starts = 0;
    int maxGridSize = 0;
    std::vector<Frame> frames;
    std::vector<boundary::InternalObjectForceSample> forces;
};

class Session : public runtime::NoCaseRunSession<Descriptor> {
public:
    explicit Session(std::shared_ptr<Trace> trace) : trace_(std::move(trace)) {}

    runtime::SingleLevelRunSessionResult start(runtime::SingleLevelRunContext<Descriptor> const& context)
    {
        ++trace_->starts;
        close(context.lbm.nuLb, viscosity, "viscosity");
        close(context.lbm.tau, 0.65, "tau");
        close(context.lbm.uRefLb, speed, "reference speed");
        require(context.geometry.isPeriodic(2), "z must remain periodic");
        if (amrex::ParallelDescriptor::IOProcessor()) {
            std::filesystem::create_directories(trace_->directory);
            std::ofstream identity(trace_->directory + "/identity.txt");
            char const* source = std::getenv("LBM_EXPECTED_GIT_COMMIT");
            identity << std::setprecision(17) << "descriptor=D2Q9\namrex_spacedim=3\n"
                     << "source_id=" << (source ? source : "unspecified") << '\n'
                     << "ranks=" << amrex::ParallelDescriptor::NProcs() << '\n'
                     << "domain=" << context.geometry.Domain() << '\n'
                     << "height_phys=" << context.geometry.ProbLength(2) << '\n'
                     << "dx_phys=" << context.geometry.CellSize(0) << '\n'
                     << "steps=" << steps << '\n';
#ifdef AMREX_USE_GPU
            identity << "backend=gpu\n";
#else
            identity << "backend=cpu\n";
#endif
            require(identity.good(), "cannot write thickness identity");
        }
        return {};
    }

    runtime::SingleLevelObservationPlan observationPlan() const noexcept
    {
        runtime::SingleLevelObservationPlan plan;
        plan.observeInitialized = plan.observeFinal = true;
        plan.macroAtInitialized = plan.macroAtFinal = true;
        plan.completed.everyStep = plan.completedMacro.everyStep = true;
        if (trace_->withForce) {
            plan.forceCalculationInterval = plan.forceDeliveryInterval = 1;
        }
        return plan;
    }

    runtime::SingleLevelRunSessionResult observeInitialized(runtime::SingleLevelPublishedState<Descriptor> const& state)
    {
        capture(state);
        return {};
    }

    runtime::SingleLevelCompletedObservationResult observeCompleted(runtime::SingleLevelPublishedState<Descriptor> const& state)
    {
        if (state.step() <= 2) {
            capture(state);
        }
        trace_->forces.insert(trace_->forces.end(), state.forceBatch().objectSteps.begin(), state.forceBatch().objectSteps.end());
        return {};
    }

    runtime::SingleLevelRunSessionResult observeFinal(runtime::SingleLevelPublishedState<Descriptor> const& state)
    {
        capture(state);
        trace_->forces.insert(trace_->forces.end(), state.forceBatch().objectSteps.begin(), state.forceBatch().objectSteps.end());
        auto const& context = state.context();
        auto const& geometry = context.geometry;
        const std::string path = trace_->directory + "/plot";
        runtime::write_single_level_plotfile(path, state.macro(), geometry,
            context.boxArray, context.distributionMapping, context.lbm.rhoRefLb,
            context.lbm.uRefLb, state.step(), state.physicalTime());
        amrex::PlotFileData plot(path);
        require(plot.spaceDim() == 3 && plot.finestLevel() == 0 &&
            plot.probDomain(0) == geometry.Domain(), "plotfile domain identity changed");
        require(plot.varNames() == amrex::Vector<std::string>{
            "rho_norm", "ux_norm", "uy_norm", "uz_norm"}, "plotfile fields changed");
        close(plot.time(), state.physicalTime(), "plotfile physical time");
        for (int d = 0; d < 3; ++d) {
            close(plot.probLo()[d], geometry.ProbLo(d), "plotfile lower bound");
            close(plot.probHi()[d], geometry.ProbHi(d), "plotfile upper bound");
            close(plot.cellSize(0)[d], geometry.CellSize(d), "plotfile cell size");
        }
        auto values = gather_xy(plot.get(0), geometry.Domain());
        auto const& expected = trace_->frames.back().macro;
        require(values.size() == expected.size(), "plotfile component count changed");
        for (std::size_t n = 0; n < values.size(); ++n) {
            close(values[n], expected[n] / (n % 4 == 0 ? context.lbm.rhoRefLb : context.lbm.uRefLb),
                  "plotfile normalized value");
        }
        if (amrex::ParallelDescriptor::IOProcessor()) {
            std::ofstream output(trace_->directory + "/states.csv");
            output << std::setprecision(17) << "step,i,j,f0,f1,f2,f3,f4,f5,f6,f7,f8,rho,ux,uy,uz\n";
            for (auto const& frame : trace_->frames) {
                const int nx = frame.geometry.Domain().length(0);
                for (std::size_t cell = 0; cell < frame.macro.size() / 4; ++cell) {
                    output << frame.step << ',' << cell % nx << ',' << cell / nx;
                    for (int q = 0; q < 9; ++q) { output << ',' << frame.populations[cell * 9 + q]; }
                    for (int n = 0; n < 4; ++n) { output << ',' << frame.macro[cell * 4 + n]; }
                    output << '\n';
                }
            }
            require(output.good(), "cannot write thickness states");
        }
        return {};
    }

private:
    void capture(runtime::SingleLevelPublishedState<Descriptor> const& state)
    {
        auto const& context = state.context();
        const Real referenceDx = context.lbm.lengthRefPhys / context.lbm.lengthRefLb;
        close(state.latticeTime(), state.step(), "published lattice time");
        close(state.physicalTime(), state.step() * speed * referenceDx, "published physical time");
        trace_->frames.push_back({state.step(), state.physicalTime(), referenceDx,
            context.geometry, gather_xy(state.populations().multifab(), context.geometry.Domain()),
            gather_xy(state.macro().multifab(), context.geometry.Domain())});
    }
    std::shared_ptr<Trace> trace_;
};

struct Preparation {
    std::shared_ptr<Trace> trace;
    template <class> using RunSession = Session;
    template <class View>
    runtime::SingleLevelCasePreparationResult<Session> resolve(View const&) const
    {
        return {Session(trace), {}};
    }
};

using Program = runtime::SingleLevelProgram<Descriptor, BGKDynamicsSelection,
    initialization::ConfiguredInitializationSelection, Preparation>;

inline void compare(Trace const& reference, Trace const& candidate)
{
    require(reference.starts == 1 && candidate.starts == 1, "run session did not start once");
    require(reference.frames.size() == 4 && candidate.frames.size() == 4, "missing state observations");
    require(reference.forces.size() == candidate.forces.size(), "force comparison shape");
    for (std::size_t n = 0; n < reference.forces.size(); ++n) {
        auto const& a = reference.forces[n];
        auto const& b = candidate.forces[n];
        require(a.step.populationStep == b.step.populationStep && a.objectForces.size() == 1 &&
            b.objectForces.size() == 1 && a.objectForces[0].objectId.value == 0 &&
            b.objectForces[0].objectId.value == 0, "force comparison identity");
        for (int d = 0; d < 3; ++d) {
            close(a.objectForces[0].forceLb[d], b.objectForces[0].forceLb[d],
                "unit-span force thickness/layout invariance");
        }
    }
    const std::array<int, 4> expectedSteps{0, 1, 2, steps};
    for (std::size_t n = 0; n < expectedSteps.size(); ++n) {
        auto const& a = reference.frames[n];
        auto const& b = candidate.frames[n];
        require(a.step == expectedSteps[n] && b.step == expectedSteps[n], "wrong observation step");
        close(a.physicalTime, b.physicalTime, "thickness comparison time");
        require(a.populations.size() == b.populations.size() && a.macro.size() == b.macro.size(),
                "thickness comparison shape");
        for (std::size_t q = 0; q < a.populations.size(); ++q) {
            close(a.populations[q], b.populations[q], "population thickness/layout invariance");
        }
        for (std::size_t c = 0; c < a.macro.size(); ++c) {
            close(a.macro[c], b.macro[c], "macro thickness/layout invariance");
        }
    }
}

template <class Check>
void check_case(std::string const& name, Program base, Check check)
{
    std::shared_ptr<Trace> reference;
    const Real h = (base.mesh.envelopeHi[0] - base.mesh.envelopeLo[0]) / base.mesh.nCell[0];
    for (int boxSize : {32, 8}) {
        for (int scale : {1, 2}) {
            auto program = base;
            auto trace = std::make_shared<Trace>();
            trace->directory = "artifacts/" + name + "/h" + std::to_string(scale) + "-box" + std::to_string(boxSize);
            trace->withForce = !base.internalBoundaries.empty();
            trace->maxGridSize = boxSize;
            program.mesh.envelopeHi[2] = program.mesh.envelopeLo[2] + scale * h;
            program.mesh.maxGridSize = boxSize;
            program.casePreparation.trace = trace;
            auto prepared = runtime::prepare_single_level(std::move(program));
            for (auto const& diagnostic : prepared.diagnostics()) {
                amrex::Print() << diagnostic.code << ": " << diagnostic.message << '\n';
            }
            require(prepared.ok(), "fixed-thickness program preparation failed");
            auto report = runtime::run_single_level(std::move(prepared).take());
            require(report.ok() && report.finalStep == steps, "fixed-thickness run did not complete");
            require(!trace->frames.empty(), "missing fixed-thickness frame");
            close(trace->frames.front().geometry.ProbLength(2), scale * h, "requested physical thickness");
            close(trace->frames.front().referenceDx, h, "requested active reference step");
            if (!reference) { reference = trace; }
            compare(*reference, *trace);
            check(*trace);
            amrex::Print() << "thickness_case_pass name=" << name << " H_over_h=" << scale
                           << " max_grid_size=" << boxSize << " steps=" << report.finalStep << '\n';
        }
    }
}

} // namespace lbm::tests::fixed_thickness

#endif
