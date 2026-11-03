#ifndef LBM_CASES_TAYLORGREENVORTEX_PREPARATION_H
#define LBM_CASES_TAYLORGREENVORTEX_PREPARATION_H

#include "taylor_green_vortex/composition/TaylorGreenCaseValidation.H"
#include "taylor_green_vortex/config/Controls.H"
#include "taylor_green_vortex/runtime/RunSession.H"
#include "lbm/initialization/InitializationTypes.H"

#include <iterator>
#include <string>
#include <utility>

namespace lbm::cases::taylor_green_vortex {

/** Case-owned validation and observation assembly for the TGV Program. */
struct Preparation {
    Controls controls;

    template <class Descriptor>
    using RunSession = taylor_green_vortex::RunSession<Descriptor>;

    template <class PreparationView>
    runtime::SingleLevelCasePreparationResult<
        RunSession<typename PreparationView::DescriptorType>>
    resolve(PreparationView const& view)
    {
        using Descriptor = typename PreparationView::DescriptorType;
        using Result = runtime::SingleLevelCasePreparationResult<
            RunSession<Descriptor>>;
        Result result;
        auto add = [&](char const* code, char const* subject, std::string message) {
            result.diagnostics.push_back({code, subject, std::move(message)});
        };

        auto const& resolvedInitialization = view.initialization.resolved();
        if (resolvedInitialization.kind !=
            initialization::InitializationKind::TaylorGreen) {
            add(
                "invalid_taylor_green_initialization",
                "initialization",
                "direct Taylor-Green requires the Taylor-Green initialization profile");
        }
        auto meshDiagnostics = validate_case_mesh<Descriptor>(view.mesh);
        result.diagnostics.insert(
            result.diagnostics.end(),
            std::make_move_iterator(meshDiagnostics.begin()),
            std::make_move_iterator(meshDiagnostics.end()));
        if (!result.diagnostics.empty()) {
            return result;
        }

        controls.nCell = view.mesh.nCell;
        controls.maxStep = view.run.maxStep;
        controls.finalTimePhys = view.run.finalTimePhys;
        controls.runKind = view.run.kind;
        controls.runVariant = view.run.variant;
        controls.convergenceEnabled = view.run.convergenceEnabled;
        controls.convergenceReset = view.run.convergenceReset;
        if (controls.convergenceEnabled != 0) {
            controls.diagnosticsInterval = view.run.maxStep;
            controls.plotInterval = 0;
        }
        result.session = RunSession<Descriptor>(
            std::move(controls), resolvedInitialization.taylorGreenParams);
        return result;
    }
};

} // namespace lbm::cases::taylor_green_vortex

#endif
