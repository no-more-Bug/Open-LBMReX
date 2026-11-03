#ifndef LBM_CASES_CHANNELCYLINDER_PREPARATION_H
#define LBM_CASES_CHANNELCYLINDER_PREPARATION_H

#include "channel_cylinder/composition/CaseValidation.H"
#include "lbm/initialization/InitializationSelection.H"
#include "channel_cylinder/config/Controls.H"
#include "channel_cylinder/runtime/RunSession.H"

#include <string>
#include <utility>

namespace lbm::cases::channel_cylinder {

struct Preparation {
    Controls controls;

    template <class Descriptor>
    using RunSession = channel_cylinder::RunSession<Descriptor>;

    template <class PreparationView>
    runtime::SingleLevelCasePreparationResult<
        RunSession<typename PreparationView::DescriptorType>>
    resolve(PreparationView const& view)
    {
        using Descriptor = typename PreparationView::DescriptorType;
        using Result = runtime::SingleLevelCasePreparationResult<
            RunSession<Descriptor>>;
        Result result;

        if constexpr (Descriptor::d == 2) {
            if (view.mesh.nCell[2] != 1) {
                result.diagnostics.push_back({
                    "invalid_channel_cylinder_depth",
                    "mesh.n_cell.z",
                    "D2Q9 channel-cylinder requires one periodic z cell"});
            }
        } else {
            if (view.mesh.nCell[2] < 2) {
                result.diagnostics.push_back({
                    "invalid_channel_cylinder_depth",
                    "mesh.n_cell.z",
                    "D3Q27 channel-cylinder requires at least two periodic z cells"});
            }
        }
        if (controls.variant == Variant::Re20Yms) {
            if constexpr (Descriptor::d == 3) {
                if (controls.diameterIntervals != 20 ||
                    view.mesh.nCell[2] != 4) {
                    result.diagnostics.push_back({
                        "invalid_channel_cylinder_re20_d3_identity",
                        "channel_cylinder",
                        "D3Q27 Re20 supports N_D=20,nz=4 only"});
                }
            }
        }
        if (controls.variant == Variant::Re100Yms) {
            if constexpr (Descriptor::d != 2) {
                result.diagnostics.push_back({
                    "invalid_channel_cylinder_re100_descriptor",
                    "channel_cylinder",
                    "Re100 supports D2Q9 only"});
            }
        }
        if (view.mesh.isPeriodic[0] != 0 ||
            view.mesh.isPeriodic[1] != 0 ||
            view.mesh.isPeriodic[2] == 0) {
            result.diagnostics.push_back({
                "invalid_channel_cylinder_periodicity",
                "mesh.periodicity",
                "channel-cylinder requires non-periodic x/y and periodic z"});
        }

        const bool tandem = is_tandem(controls.variant);
        const amrex::Real expectedUx =
            uses_uniform_initialization(controls.variant) ?
                view.lbm.uRefLb : amrex::Real(0.0);
        const initialization::InitializationKind expectedInitialization =
            uses_uniform_initialization(controls.variant) ?
                initialization::InitializationKind::Uniform :
                initialization::InitializationKind::Rest;
        auto const& initialState =
            initialization::constant_initialization_state(
                view.initialization);
        if (!initialization::initialization_kind_matches(
                view.initialization, expectedInitialization) ||
            initialState.rho != view.lbm.rhoRefLb ||
            initialState.ux != expectedUx ||
            initialState.uy != 0.0 ||
            initialState.uz != 0.0) {
            result.diagnostics.push_back({
                "invalid_channel_cylinder_initialization",
                "initialization",
                "channel-cylinder initialization does not match its variant"});
        }

        char const* referenceLabel = "dfg_2d_1_re20";
        if (tandem) {
            referenceLabel = "tandem_two_cylinder";
        } else if (controls.variant == Variant::Re100Yms) {
            referenceLabel = "dfg_2d_2_re100";
        }
        ChannelCylinderBenchmarkValidationResult validation =
            validate_channel_cylinder_benchmark<Descriptor>(
                view.lbm,
                view.geometry,
                view.boundaries,
                view.postStreamingPlan,
                referenceLabel,
                controls.forceInterval,
                controls.outputInterval,
                controls.writeHistory,
                controls.assertEnabled);
        for (std::string& error : validation.errors) {
            result.diagnostics.push_back({
                "invalid_channel_cylinder_case",
                controls.runVariant,
                std::move(error)});
        }
        if (!result.diagnostics.empty()) {
            return result;
        }

        controls.rhoLb = initialState.rho;
        controls.uMeanLb = view.lbm.uRefLb;
        controls.maxStep = view.run.maxStep;
        controls.finalTimePhys = view.run.finalTimePhys;
        controls.runKind = view.run.kind;
        controls.runVariant = view.run.variant;
        result.session = RunSession<Descriptor>(
            std::move(controls), std::move(validation.data));
        return result;
    }
};

} // namespace lbm::cases::channel_cylinder

#endif
