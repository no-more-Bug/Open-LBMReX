#ifndef LBM_INPUT_PARSEDINPUTVALUE_H
#define LBM_INPUT_PARSEDINPUTVALUE_H

namespace lbm::input {

/**
 * @brief Describes which input spelling supplied a decoded value.
 *
 * AMReX has already combined the inputs file and command-line definitions when
 * project parsers run. Repeated definitions of the same canonical key have
 * already been reduced to AMReX's last-definition value.
 */
enum class InputValueSource {
    Default,
    Canonical
};

/**
 * @brief Host-side raw value with explicit canonical presence.
 *
 * A default-initialized value is deliberately distinct from an explicitly
 * supplied value. Semantic resolvers use provided() rather than comparing
 * against the stored default when enforcing required/forbidden inputs.
 */
template <class T>
struct ParsedInputValue {
    T value{};
    InputValueSource source = InputValueSource::Default;

    bool provided() const noexcept
    {
        return source != InputValueSource::Default;
    }

};

} // namespace lbm::input

#endif
