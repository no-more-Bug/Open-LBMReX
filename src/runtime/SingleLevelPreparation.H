#ifndef LBM_RUNTIME_SINGLELEVELPREPARATION_H
#define LBM_RUNTIME_SINGLELEVELPREPARATION_H

#include "runtime/SingleLevelProgram.H"
#include "runtime/SingleLevelRunReport.H"
#include "runtime/SingleLevelSetupDiagnostic.H"

#include <memory>
#include <optional>
#include <vector>

namespace lbm::runtime {

template <class Descriptor, class Dynamics, class Initialization, class RunSession>
class SingleLevelPreparationResult;

template <class Descriptor, class Dynamics, class Initialization, class RunSession>
class PreparedSingleLevelRun;

template <
    class Descriptor,
    class DynamicsSelection,
    class InitializationSelection,
    class CasePreparation>
SingleLevelPreparationResult<
    Descriptor,
    typename DynamicsSelection::template Dynamics<Descriptor>,
    initialization::PreparedInitialization<
        Descriptor,
        InitializationSelection>,
    typename CasePreparation::template RunSession<Descriptor>>
prepare_single_level(
    SingleLevelProgram<
        Descriptor,
        DynamicsSelection,
        InitializationSelection,
        CasePreparation> program);

/** Opaque, move-only ownership token produced only by successful prepare. */
template <class Descriptor, class Dynamics, class Initialization, class RunSession>
class PreparedSingleLevelRun {
public:
    PreparedSingleLevelRun(PreparedSingleLevelRun const&) = delete;
    PreparedSingleLevelRun& operator=(PreparedSingleLevelRun const&) = delete;
    PreparedSingleLevelRun(PreparedSingleLevelRun&&) noexcept;
    PreparedSingleLevelRun& operator=(PreparedSingleLevelRun&&) noexcept;
    ~PreparedSingleLevelRun();

private:
    struct State;

    explicit PreparedSingleLevelRun(std::unique_ptr<State> state) noexcept;

    std::unique_ptr<State> state_;

    template <class D, class DS, class IS, class C>
    friend SingleLevelPreparationResult<
        D,
        typename DS::template Dynamics<D>,
        initialization::PreparedInitialization<D, IS>,
        typename C::template RunSession<D>> prepare_single_level(
        SingleLevelProgram<D, DS, IS, C> program);
    template <class D, class DY, class I, class S>
    friend SingleLevelRunReport run_single_level(
        PreparedSingleLevelRun<D, DY, I, S> prepared);
};

/** Structured result: either one prepared token or ordered diagnostics. */
template <class Descriptor, class Dynamics, class Initialization, class RunSession>
class SingleLevelPreparationResult {
public:
    SingleLevelPreparationResult(SingleLevelPreparationResult const&) = delete;
    SingleLevelPreparationResult& operator=(SingleLevelPreparationResult const&) = delete;
    SingleLevelPreparationResult(SingleLevelPreparationResult&&) noexcept = default;
    SingleLevelPreparationResult& operator=(SingleLevelPreparationResult&&) noexcept = default;

    bool ok() const noexcept;
    std::vector<SingleLevelSetupDiagnostic> const& diagnostics() const noexcept;
    PreparedSingleLevelRun<Descriptor, Dynamics, Initialization, RunSession>
    take() &&;

private:
    explicit SingleLevelPreparationResult(
        std::vector<SingleLevelSetupDiagnostic> diagnostics);
    explicit SingleLevelPreparationResult(
        PreparedSingleLevelRun<
            Descriptor,
            Dynamics,
            Initialization,
            RunSession> prepared);

    std::optional<PreparedSingleLevelRun<
        Descriptor,
        Dynamics,
        Initialization,
        RunSession>> prepared_;
    std::vector<SingleLevelSetupDiagnostic> diagnostics_;

    template <class D, class DS, class IS, class C>
    friend SingleLevelPreparationResult<
        D,
        typename DS::template Dynamics<D>,
        initialization::PreparedInitialization<D, IS>,
        typename C::template RunSession<D>> prepare_single_level(
        SingleLevelProgram<D, DS, IS, C> program);
};

} // namespace lbm::runtime

#include "runtime/SingleLevelPreparation.HH"

#endif
