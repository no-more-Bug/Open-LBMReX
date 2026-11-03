#ifndef LBM_OPERATORS_CORE_OPERATOR_H
#define LBM_OPERATORS_CORE_OPERATOR_H

#include <string_view>

namespace lbm::operators {

enum class OperatorScope {
    Level
};

class Operator {
public:
    virtual ~Operator() = default;

    virtual std::string_view name() const noexcept = 0;
    virtual OperatorScope scope() const noexcept = 0;
};

} // namespace lbm::operators

#endif
