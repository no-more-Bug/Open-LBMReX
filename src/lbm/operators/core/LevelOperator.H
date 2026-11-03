#ifndef LBM_OPERATORS_CORE_LEVELOPERATOR_H
#define LBM_OPERATORS_CORE_LEVELOPERATOR_H

#include "lbm/operators/core/Operator.H"

namespace lbm::operators {

class LevelOperator : public Operator {
public:
    OperatorScope scope() const noexcept override
    {
        return OperatorScope::Level;
    }
};

} // namespace lbm::operators

#endif
