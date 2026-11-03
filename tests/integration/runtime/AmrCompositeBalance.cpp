#include "integration/runtime/AmrCompositeBalance.H"
#include <set>

namespace lbm::tests::amr_evolution {
using Cell=amr_reference::Cell;

CompositeTotals composite(EvolutionView const& view, std::vector<Frame> const& frames,
                 std::vector<amr_reference::ReferenceLevel> const& reference)
{
    CompositeTotals result;
    auto const stencil=amr_reference::stencil(Descriptor::d);
    for (std::size_t lev=0; lev<frames.size(); ++lev) {
        auto const& f=frames[lev].populations;
        std::set<Cell> covered;
        if (lev+1<frames.size()) {
            auto const& finer=frames[lev+1].populations;
            auto const ratio=view.refRatios[lev];
            for (std::size_t row=0; row<finer.cellIndices.size(); ++row) {
                auto const p=finer.cell_at(row);
                covered.insert({p[0]/ratio[0],p[1]/ratio[1],p[2]/ratio[2]});
            }
        }
        auto const& geometry=view.levels[lev].geometry;
        long double const area=static_cast<long double>(geometry.CellSize(0))*geometry.CellSize(1);
        long double const measure=area*(Descriptor::d==2 ? geometry.ProbHi(2)-geometry.ProbLo(2)
                                                         : geometry.CellSize(2));
        for (std::size_t row=0; row<f.cellIndices.size(); ++row) {
            auto const p=f.cell_at(row);
            Cell const cell{p[0],p[1],p[2]};
            if (covered.count(cell)) continue;
            ++result.cells; result.volume+=measure;
            auto const* expected=reference[lev].current.find(cell);
            require(expected!=nullptr,"composite reference cell missing");
            for (int q=0; q<Descriptor::q; ++q) {
                long double const actual=f.values[row*Descriptor::q+q]*measure;
                long double const desired=expected[q].value*measure;
                result.mass+=actual; result.referenceMass+=desired;
                for (int d=0; d<3; ++d) {
                    result.momentum[d]+=stencil.directions[q][d]*actual;
                    result.referenceMomentum[d]+=stencil.directions[q][d]*desired;
                }
            }
        }
    }
    // Each snapshot is already globally gathered. Do not reduce these totals again.
    return result;
}

void compare_totals(CompositeTotals const& actual, Real referenceMass, std::array<Real,3> const& referenceMomentum,
                    Real initialMass, std::string_view subject)
{
    long double const eta=4096*std::numeric_limits<Real>::epsilon()*
        (1+std::ceil(std::log2(std::max(std::size_t(2),actual.cells))));
    long double square=0;
    for (int d=0; d<3; ++d) square+=std::pow(actual.momentum[d]-referenceMomentum[d],2);
    require(std::abs(actual.mass-referenceMass)/initialMass<=eta &&
            std::sqrt(square)/(initialMass*Real(0.02))<=eta,subject);
}
} // namespace lbm::tests::amr_evolution
