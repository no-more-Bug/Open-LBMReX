#ifndef LBM_CASES_POISEUILLEFLOW_PREPARATION_H
#define LBM_CASES_POISEUILLEFLOW_PREPARATION_H

#include "poiseuille_flow/composition/PoiseuilleCaseValidation.H"
#include "lbm/initialization/InitializationSelection.H"
#include "poiseuille_flow/config/Controls.H"
#include "poiseuille_flow/runtime/RunSession.H"

#include <string>
#include <utility>

namespace lbm::cases::poiseuille_flow {

struct Preparation {
    Controls controls;

    template <class Descriptor>
    using RunSession = poiseuille_flow::RunSession<Descriptor>;

    template <class PreparationView>
    runtime::SingleLevelCasePreparationResult<
        RunSession<typename PreparationView::DescriptorType>>
    resolve(PreparationView const& view)
    {
        using Descriptor = typename PreparationView::DescriptorType;
        runtime::SingleLevelCasePreparationResult<RunSession<Descriptor>> result;
        if constexpr (Descriptor::d == 3) {
            if (view.mesh.nCell[2] < 2) {
                result.diagnostics.push_back({
                    "invalid_poiseuille_depth",
                    "mesh.n_cell.z",
                    "D3Q27 Poiseuille requires at least two periodic z cells"});
            }
        }
        if (view.mesh.isPeriodic[0] != 0 ||
            view.mesh.isPeriodic[1] != 0 ||
            view.mesh.isPeriodic[2] == 0) {
            result.diagnostics.push_back({
                "invalid_poiseuille_periodicity",
                "mesh.periodicity",
                "Poiseuille requires non-periodic x/y and periodic z"});
        }
        if (!initialization::initialization_kind_matches(
                view.initialization,
                initialization::InitializationKind::Poiseuille)) {
            result.diagnostics.push_back({
                "invalid_poiseuille_case",
                "initialization",
                "Poiseuille case requires Poiseuille initialization"});
        }
        CaseValidationResult validation = validate_case<Descriptor>(
            view.lbm,
            initialization::poiseuille_initialization_params(
                view.initialization),
            view.geometry,
            view.boundaries,
            view.postStreamingPlan,
            controls.pressureScaling);
        for (std::string& error : validation.errors) {
            result.diagnostics.push_back({
                "invalid_poiseuille_case",
                controls.runVariant,
                std::move(error)});
        }
        if (!result.diagnostics.empty()) {
            return result;
        }

        controls.maxStep = view.run.maxStep;
        controls.finalTimePhys = view.run.finalTimePhys;
        controls.runKind = view.run.kind;
        controls.runVariant = view.run.variant;
        ObservationConfig observation = controls.observation;
        observation.enabled = true;
        observation.geometry = validation.data.diagnosticGeometry;
        observation.pressure = validation.data.pressureDiagnostic;
        observation.pressureScaling =
            std::move(validation.data.pressureScaling);
        for (int axis = 0; axis < 2; ++axis) {
            auto const& axisGeometry =
                view.postStreamingPlan.axisGeometry[axis];
            controls.intervals[axis] = axisGeometry.active ?
                axisGeometry.latticeIntervals : view.mesh.nCell[axis];
        }
        result.session = RunSession<Descriptor>(
            std::move(controls), std::move(observation));
        return result;
    }
};

} // namespace lbm::cases::poiseuille_flow

#endif
