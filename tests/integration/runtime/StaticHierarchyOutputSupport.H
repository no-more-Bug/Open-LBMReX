#ifndef LBM_TESTS_STATICHIERARCHYOUTPUTSUPPORT_H
#define LBM_TESTS_STATICHIERARCHYOUTPUTSUPPORT_H

#include "integration/runtime/StaticHierarchySnapshot.H"
#include "runtime/LbmAmrPlotfile.H"

#include <set>

namespace lbm::tests::static_hierarchy::output_detail {

inline amrex::Box level_domain(amr::StaticHierarchySpec const& spec, int level)
{
    amrex::IntVect cells(spec.nCell[0],spec.nCell[1],spec.nCell[2]);
    for (int lev = 0; lev < level; ++lev) cells *= spec.refRatio;
    return amrex::Box(amrex::IntVect(0),cells-amrex::IntVect(1));
}

template <typename Expected>
std::array<Real,2> composite(amr::StaticHierarchySpec const& spec,
    std::vector<ValidSnapshot> const& fields, Real rhoReference, Expected const& expected)
{
    long double volume = 0, mass = 0, expectedMass = 0, domainVolume = 1;
    for (int d = 0; d < 3; ++d)
        domainVolume *= static_cast<long double>(spec.probHi[d])-spec.probLo[d];
    for (std::size_t lev = 0; lev < fields.size(); ++lev) {
        auto const& field = fields[lev];
        std::set<std::array<int,3>> covered;
        if (lev+1 < fields.size()) {
            auto const& fine = fields[lev+1];
            for (std::size_t row = 0; row < fine.owners.size(); ++row) {
                auto cell = fine.cell_at(row);
                for (int d = 0; d < 3; ++d) cell[d] /= spec.refRatio[d];
                covered.insert({cell[0],cell[1],cell[2]});
            }
        }
        const long double cellVolume = domainVolume/field.domain.numPts();
        std::size_t uncovered = 0;
        for (std::size_t row = 0; row < field.owners.size(); ++row) {
            const auto cell = field.cell_at(row);
            if (covered.count({cell[0],cell[1],cell[2]})) continue;
            ++uncovered;
            mass += static_cast<long double>(field.values[row*4])*rhoReference*cellVolume;
            expectedMass += static_cast<long double>(expected(static_cast<int>(lev),cell)[0])*cellVolume;
        }
        volume += uncovered*cellVolume;
    }
    close(static_cast<Real>(volume),static_cast<Real>(domainVolume),"native composite volume");
    close(static_cast<Real>(mass),static_cast<Real>(expectedMass),"native composite independent mass");
    return {static_cast<Real>(volume),static_cast<Real>(mass)};
}

template <typename Expected>
void write_expectations(std::string const& directory, std::string const& sample,
    amr::StaticHierarchySpec const& spec, std::vector<Frame> const& frames,
    std::string const& profile, std::array<Real,2> references, Expected const& expected)
{
    write_static_identity(directory,sample,spec,frames,profile,references);
    if (!amrex::ParallelDescriptor::IOProcessor()) return;
    std::ofstream out(directory+"/expected.csv");
    out << std::setprecision(17) << "level,i,j,k,rho_norm,ux_norm,uy_norm,uz_norm\n";
    for (auto const& frame : frames) {
        auto const& field = frame.populations;
        for (std::size_t row = 0; row < field.owners.size(); ++row) {
            auto const cell = field.cell_at(row);
            auto const values = expected(frame.level,cell);
            out << frame.level << ',' << cell[0] << ',' << cell[1] << ',' << cell[2];
            for (int n = 0; n < 4; ++n) out << ',' << values[n]/references[n ? 1 : 0];
            out << '\n';
        }
    }
    require(out.good(),"cannot write independent native output expectations");
}

template <typename Expected>
std::array<Real,2> readback(std::string const& path, amr::StaticHierarchySpec const& spec,
    std::vector<Frame> const& source, std::array<Real,2> references, Expected const& expected)
{
    amrex::PlotFileData plot(path);
    const int levels = static_cast<int>(source.size());
    require(plot.spaceDim() == 3 && plot.finestLevel()+1 == levels && plot.nComp() == 4 &&
        plot.coordSys() == 0,"native plotfile dimension/levels/components/coordinates");
    require(plot.varNames() == amrex::Vector<std::string>{"rho_norm","ux_norm","uy_norm","uz_norm"},
        "native plotfile variable names");
    close(plot.time(),source.front().physicalTime,"native plotfile physical time");
    for (int d = 0; d < 3; ++d) {
        close(plot.probLo()[d],spec.probLo[d],"native plotfile lower physical bound");
        close(plot.probHi()[d],spec.probHi[d],"native plotfile upper physical bound");
    }
    std::vector<ValidSnapshot> fields;
    for (int lev = 0; lev < levels; ++lev) {
        const auto domain = level_domain(spec,lev);
        require(plot.probDomain(lev) == domain && plot.levelStep(lev) == source[lev].step &&
            plot.nGrowVect(lev) == amrex::IntVect(0),"native level domain/step/ghosts");
        if (lev+1 < levels) require(plot.refRatioVect(lev) == spec.refRatio,"native vector refinement ratio");
        for (int d = 0; d < 3; ++d)
            close(plot.cellSize(lev)[d],(spec.probHi[d]-spec.probLo[d])/domain.length(d),"native directional spacing");
        for (int b = 0; b < plot.boxArray(lev).size(); ++b)
            require(plot.boxArray(lev)[b].ixType().cellCentered() && domain.contains(plot.boxArray(lev)[b]),
                "native Box centering/domain");
        fields.push_back(snapshot(plot.get(lev),domain));
        auto const& field = fields.back();
        require(field.cellIndices == source[lev].macro.cellIndices && field.owners == source[lev].macro.owners,
            "native output lost actual valid cells");
        for (std::size_t row = 0; row < field.owners.size(); ++row) {
            auto const values = expected(lev,field.cell_at(row));
            for (int n = 0; n < 4; ++n) {
                const Real reference = references[n ? 1 : 0];
                close(field.values[row*4+n],values[n]/reference,"native independent normalized field");
                close(field.values[row*4+n],source[lev].macro.values[row*4+n]/reference,"native observed normalized field");
            }
        }
    }
    return composite(spec,fields,references[0],expected);
}

} // namespace lbm::tests::static_hierarchy::output_detail
#endif
