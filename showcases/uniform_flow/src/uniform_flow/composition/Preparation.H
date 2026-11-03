#ifndef LBM_CASES_UNIFORMFLOW_PREPARATION_H
#define LBM_CASES_UNIFORMFLOW_PREPARATION_H

#include "lbm/initialization/InitializationSelection.H"
#include "uniform_flow/config/Controls.H"
#include "uniform_flow/runtime/RunSession.H"

#include <string>
#include <utility>

namespace lbm::cases::uniform_flow {

struct Preparation {
    Controls controls;

    template <class Descriptor>
    using RunSession = uniform_flow::RunSession<Descriptor>;

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
                    "invalid_uniform_flow_depth",
                    "mesh.n_cell.z",
                    "D3Q27 uniform-flow requires at least two periodic z cells"});
            }
        }
        if (view.mesh.isPeriodic[0] != 0 ||
            view.mesh.isPeriodic[1] == 0 ||
            view.mesh.isPeriodic[2] == 0) {
            result.diagnostics.push_back({
                "invalid_uniform_flow_periodicity",
                "mesh.periodicity",
                "uniform-flow requires non-periodic x and periodic y/z"});
        }
        auto const& initialState =
            initialization::constant_initialization_state(
                view.initialization);
        if (!initialization::initialization_kind_matches(
                view.initialization,
                initialization::InitializationKind::Uniform) ||
            initialState.rho != view.lbm.rhoRefLb ||
            initialState.ux != view.lbm.uRefLb ||
            initialState.uy != 0.0 ||
            initialState.uz != 0.0) {
            result.diagnostics.push_back({
                "invalid_uniform_flow_initialization",
                "initialization",
                "uniform-flow initialization must match resolved density "
                "and streamwise velocity"});
        }
        if (!result.diagnostics.empty()) {
            return result;
        }
        auto const& xGeometry = view.postStreamingPlan.axisGeometry[0];
        controls.intervals[0] = xGeometry.active ?
            xGeometry.latticeIntervals : view.mesh.nCell[0];
        controls.rhoLb = initialState.rho;
        controls.velocityLb = initialState.ux;
        controls.maxStep = view.run.maxStep;
        controls.finalTimePhys = view.run.finalTimePhys;
        controls.runKind = view.run.kind;
        controls.runVariant = view.run.variant;
        result.session = RunSession<Descriptor>(std::move(controls));
        return result;
    }
};

} // namespace lbm::cases::uniform_flow

#endif
