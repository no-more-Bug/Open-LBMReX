#ifndef LBM_TESTS_REGRESSION_COMMON_MACROSAMPLING_H
#define LBM_TESTS_REGRESSION_COMMON_MACROSAMPLING_H

#include "common/HostFieldSnapshot.H"
#include "lbm/fields/MacroField.H"

#include <AMReX_BLassert.H>
#include <AMReX_Box.H>
#include <AMReX_BoxIterator.H>
#include <AMReX_IntVect.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>

#include <algorithm>
#include <cmath>
#include <limits>

namespace lbm::tests::regression {

struct ComponentStatistics {
    amrex::Real sum = 0.0;
    amrex::Real squaredDeviationSum = 0.0;
    amrex::Real minimum = std::numeric_limits<amrex::Real>::max();
    amrex::Real maximum = std::numeric_limits<amrex::Real>::lowest();
    amrex::Real count = 0.0;
};

inline ComponentStatistics component_statistics(
    MacroField const& macro,
    int component,
    amrex::Real reference = 0.0)
{
    ComponentStatistics result;
    auto const field = lbm::tests::host_field_snapshot(macro.multifab());
    for (amrex::MFIter mfi(field); mfi.isValid(); ++mfi) {
        amrex::Box const& box = mfi.validbox();
        auto const values = field.const_array(mfi);
        for (amrex::BoxIterator index(box); index.ok(); ++index) {
            auto const& cell = index();
            const amrex::Real value = values(cell, component);
            const amrex::Real deviation = value - reference;
            result.sum += value;
            result.squaredDeviationSum += deviation * deviation;
            result.minimum = std::min(result.minimum, value);
            result.maximum = std::max(result.maximum, value);
            result.count += 1.0;
        }
    }
    amrex::ParallelDescriptor::ReduceRealSum(result.sum);
    amrex::ParallelDescriptor::ReduceRealSum(result.squaredDeviationSum);
    amrex::ParallelDescriptor::ReduceRealMin(result.minimum);
    amrex::ParallelDescriptor::ReduceRealMax(result.maximum);
    amrex::ParallelDescriptor::ReduceRealSum(result.count);
    AMREX_ALWAYS_ASSERT(result.count > 0.0);
    return result;
}

inline amrex::Real trapezoidal_weighted_sum_xy(
    MacroField const& macro,
    int component,
    amrex::Box const& domain,
    bool weightX,
    bool weightY)
{
    amrex::Real sum = 0.0;
    auto const field = lbm::tests::host_field_snapshot(macro.multifab());
    for (amrex::MFIter mfi(field); mfi.isValid(); ++mfi) {
        amrex::Box const& box = mfi.validbox();
        auto const values = field.const_array(mfi);
        for (amrex::BoxIterator index(box); index.ok(); ++index) {
            auto const& cell = index();
            amrex::Real weight = 1.0;
            if (weightX &&
                (cell[0] == domain.smallEnd(0) ||
                 cell[0] == domain.bigEnd(0))) {
                weight *= 0.5;
            }
            if (weightY &&
                (cell[1] == domain.smallEnd(1) ||
                 cell[1] == domain.bigEnd(1))) {
                weight *= 0.5;
            }
            sum += weight * values(cell, component);
        }
    }
    amrex::ParallelDescriptor::ReduceRealSum(sum);
    return sum;
}

inline amrex::Real sample_z_average(
    MacroField const& macro,
    int i,
    int j,
    int component,
    amrex::Box const& domain)
{
    amrex::Real sum = 0.0;
    amrex::Real count = 0.0;
    auto const field = lbm::tests::host_field_snapshot(macro.multifab());
    for (amrex::MFIter mfi(field); mfi.isValid(); ++mfi) {
        amrex::Box const& box = mfi.validbox();
        auto const values = field.const_array(mfi);
        for (int k = domain.smallEnd(2); k <= domain.bigEnd(2); ++k) {
            amrex::IntVect const index(AMREX_D_DECL(i, j, k));
            if (box.contains(index)) {
                sum += values(i, j, k, component);
                count += 1.0;
            }
        }
    }
    amrex::ParallelDescriptor::ReduceRealSum(sum);
    amrex::ParallelDescriptor::ReduceRealSum(count);
    AMREX_ALWAYS_ASSERT(count > 0.0);
    return sum / count;
}

inline amrex::Real sample_z_max_abs(
    MacroField const& macro,
    int i,
    int j,
    int component,
    amrex::Box const& domain)
{
    amrex::Real value = 0.0;
    auto const field = lbm::tests::host_field_snapshot(macro.multifab());
    for (amrex::MFIter mfi(field); mfi.isValid(); ++mfi) {
        amrex::Box const& box = mfi.validbox();
        auto const values = field.const_array(mfi);
        for (int k = domain.smallEnd(2); k <= domain.bigEnd(2); ++k) {
            amrex::IntVect const index(AMREX_D_DECL(i, j, k));
            if (box.contains(index)) {
                value = std::max(value, std::abs(values(i, j, k, component)));
            }
        }
    }
    amrex::ParallelDescriptor::ReduceRealMax(value);
    return value;
}

inline amrex::Real sample_yz_trapezoidal_average(
    MacroField const& macro,
    int i,
    int component,
    amrex::Box const& domain,
    bool weightY)
{
    amrex::Real sum = 0.0;
    amrex::Real weightSum = 0.0;
    auto const field = lbm::tests::host_field_snapshot(macro.multifab());
    for (amrex::MFIter mfi(field); mfi.isValid(); ++mfi) {
        amrex::Box const& box = mfi.validbox();
        auto const values = field.const_array(mfi);
        for (int j = domain.smallEnd(1); j <= domain.bigEnd(1); ++j) {
            const amrex::Real weight = weightY &&
                    (j == domain.smallEnd(1) || j == domain.bigEnd(1))
                ? 0.5 : 1.0;
            for (int k = domain.smallEnd(2); k <= domain.bigEnd(2); ++k) {
                amrex::IntVect const index(AMREX_D_DECL(i, j, k));
                if (box.contains(index)) {
                    sum += weight * values(i, j, k, component);
                    weightSum += weight;
                }
            }
        }
    }
    amrex::ParallelDescriptor::ReduceRealSum(sum);
    amrex::ParallelDescriptor::ReduceRealSum(weightSum);
    AMREX_ALWAYS_ASSERT(weightSum > 0.0);
    return sum / weightSum;
}

} // namespace lbm::tests::regression

#endif
