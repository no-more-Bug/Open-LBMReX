#include "AmrPerformanceCases.H"
#include "AmrPerformanceCsv.H"
#include "runtime/LbmAmrPreparation.H"
#include "lbm/descriptors/D2Q9Descriptor.H"
#include "lbm/descriptors/D3Q27Descriptor.H"
#include "lbm/dynamics/BGKDynamicsSelection.H"
#include <AMReX_Print.H>
#include <cctype>
#include <cmath>
#include <limits>

namespace {
#ifdef LBM_DESCRIPTOR_D3Q27
using Descriptor = lbm::D3Q27Descriptor;
#else
using Descriptor = lbm::D2Q9Descriptor;
#endif

int run()
{
    using namespace lbm::tests::performance;
    using Mode = lbm::profiling::TimestepProfilingMode;
    static_assert(std::numeric_limits<amrex::Real>::digits == 53, "AMR performance requires FP64");
    auto controls = read_performance_controls();
    std::string layout;
    amrex::ParmParse("case").get("layout", layout);
    std::string descriptor = Descriptor::name();
    for (auto& letter : descriptor) letter = static_cast<char>(std::tolower(letter));
    if (controls.workload != std::string(amrCaseName)+"_"+descriptor ||
        (layout != "amr" && layout != "uniform") ||
        (controls.mode != "smoke" && controls.mode != "calibration" && controls.mode != "baseline") ||
        (controls.measurementMode != "throughput" && controls.measurementMode != "stage_breakdown") ||
        controls.warmupSteps < 0 || controls.measurementSteps <= 0 ||
        controls.maxStep != controls.warmupSteps+controls.measurementSteps ||
        controls.repeat <= 0 || controls.outputCsv.empty()) {
        amrex::Print()<<"invalid AMR performance controls\n";
        return 2;
    }
    amrex::Gpu::streamSynchronize();
    amrex::ParallelDescriptor::Barrier();
    auto const begin = amrex::second();
    auto prepared = lbm::amr::prepare_static_hierarchy<Descriptor>(
        amr_geometry(Descriptor::d, layout == "uniform"), amr_initial(), amr_boundaries());
    if (!prepared.ok()) {
        for (auto const& error : prepared.errors) amrex::Print()<<error<<'\n';
        return 2;
    }
    auto owner = prepared.take();
    auto preparation = lbm::runtime::prepare_amr_evolution(
        *owner, amr_intent(owner->geometry(owner->numLevels()-1).CellSize(0)), lbm::BGKDynamicsSelection{});
    if (!preparation.ok()) {
        for (auto const& error : preparation.errors) amrex::Print()<<error<<'\n';
        return 2;
    }
    auto evolution = std::move(preparation).take();
    amrex::Gpu::streamSynchronize();
    amrex::Real initializationSeconds = amrex::second()-begin;
    amrex::ParallelDescriptor::ReduceRealMax(initializationSeconds);
    auto const initializedUsedBytes = amr_device_used_bytes();
    std::vector<AmrPerformanceLevel> levels;
    owner->observe(false, [&](auto const& view) {
        for (int lev=0; lev<owner->numLevels(); ++lev) {
            auto const& level = view.levels[lev];
            auto const lengths = level.geometry.Domain().length();
            levels.push_back({{lengths[0],lengths[1],lengths[2]}, level.boxes, level.mapping,
                              evolution.level(lev).dtPhys, evolution.level(lev).lbm.tau});
        }
    });
    auto const mode = controls.measurementMode == "throughput" ? Mode::Throughput : Mode::StageBreakdown;
    lbm::profiling::AmrTimestepProfiler profiler(mode, controls.warmupSteps, controls.measurementSteps, owner->numLevels());
    auto result = owner->evolve(evolution, controls.maxStep, false, [](auto const&) {}, nullptr, &profiler);
    if (!result.ok()) { amrex::Print()<<result.error<<'\n'; return 2; }
    profiler.finalize();
    auto const record = profiler.performanceRecord();
    if (!record || record->window.nSteps != controls.measurementSteps ||
        record->window.sampleStart != controls.warmupSteps+1 || record->window.sampleEnd != controls.maxStep ||
        !std::isfinite(record->window.synchronizedWindow) || record->window.synchronizedWindow <= 0) return 2;
    // Only check finiteness outside the measured window; no scientific sampling.
    bool finite = true;
    owner->observe(true, [&](auto const& view) {
        for (auto const& level : view.levels) {
            auto const& populations = level.populations.multifab();
            auto const& macro = level.macro->multifab();
            bool const populationsFinite = populations.is_finite(0, Descriptor::q, 0);
            bool const macroFinite = macro.is_finite(0, lbm::MacroField::NumComp, 0);
            finite = finite && populationsFinite && macroFinite;
        }
    });
    if (!finite) { amrex::Print()<<"nonfinite final state\n"; return 1; }
    write_amr_resource_csv(controls.outputCsv, initializedUsedBytes);
    write_amr_performance_csv(controls, layout, Descriptor::name(), levels, *record, initializationSeconds);
    return 0;
}
} // namespace

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    int status = 2;
    try { status = run(); }
    catch (std::exception const& error) { amrex::Abort(error.what()); }
    amrex::Finalize();
    return status;
}
