#ifndef LBM_CASES_LIDDRIVENCAVITY_PREPARATION_H
#define LBM_CASES_LIDDRIVENCAVITY_PREPARATION_H

#include "lid_driven_cavity/composition/CavityCaseValidation.H"
#include "lbm/initialization/InitializationSelection.H"
#include "lid_driven_cavity/config/Controls.H"
#include "lid_driven_cavity/runtime/RunSession.H"
#include "runtime/SingleLevelSetupDiagnostic.H"

#include <string>
#include <utility>

namespace lbm::cases::lid_driven_cavity {

/** Case-owned validation and observation assembly for the cavity Program. */
struct Preparation {
    Controls controls;

    template <class Descriptor>
    using RunSession = lid_driven_cavity::RunSession<Descriptor>;

    template <class PreparationView>
    runtime::SingleLevelCasePreparationResult<
        RunSession<typename PreparationView::DescriptorType>>
    resolve(PreparationView const& view)
    {
        using Descriptor = typename PreparationView::DescriptorType;
        using Result = runtime::SingleLevelCasePreparationResult<
            RunSession<Descriptor>>;
        Result result;

        if (!initialization::initialization_kind_matches(
                view.initialization,
                initialization::InitializationKind::Rest)) {
            result.diagnostics.push_back({
                "invalid_canonical_cavity",
                "initialization",
                "canonical lid-driven cavity requires rest initialization"});
        }
        auto const& initialState =
            initialization::constant_initialization_state(
                view.initialization);
        initialization::InitializationConfigIntent initializationIntent;
        initializationIntent.kind = initialization::InitializationKind::Rest;
        initializationIntent.rhoLb = initialState.rho;
        if (initialState.ux != 0.0 || initialState.uy != 0.0 ||
            initialState.uz != 0.0) {
            result.diagnostics.push_back({
                "non_quiescent_initialization",
                "initialization",
                "canonical lid-driven cavity requires a quiescent "
                "constant initialization profile"});
        }

        CavityCaseValidationResult cavity = validate_cavity_case(
            view.lbm,
            initializationIntent,
            view.geometry,
            view.boundaries,
            view.postStreamingPlan);
        for (std::string const& error : cavity.errors) {
            result.diagnostics.push_back({
                "invalid_canonical_cavity",
                "lid_driven_cavity",
                error});
        }
        if (!result.diagnostics.empty()) {
            return result;
        }

        CavityObservationConfig observation;
        observation.enabled = true;
        observation.writeCenterlines = controls.writeCenterlines;
        observation.writeMetrics = controls.writeMetrics;
        observation.lidSpeedLb = cavity.data.lidSpeedLb;
        observation.lengthLb = cavity.data.lengthLb;
        observation.geometry = cavity.data.diagnosticGeometry;
        result.session = RunSession<Descriptor>(
            std::move(controls), std::move(observation));
        return result;
    }
};

} // namespace lbm::cases::lid_driven_cavity

#endif
