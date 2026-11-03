#include "integration/runtime/StaticHierarchySnapshot.H"

#include <cstdlib>
#include <map>

namespace lbm::tests::static_hierarchy {

void write_static_identity(std::string const& directory, std::string const& sample,
    amr::StaticHierarchySpec const& spec, std::vector<Frame> const& frames,
    std::string const& profile, std::array<Real,2> references)
{
    if (!amrex::ParallelDescriptor::IOProcessor()) return;
    require(!frames.empty(), "snapshot requires at least one frame");
    std::filesystem::create_directories(directory);
    std::ofstream out(directory + "/identity.txt");
    char const* source = std::getenv("LBM_EXPECTED_GIT_COMMIT");
    const int q = frames.front().populations.components;
    require(q == 9 || q == 27,"static snapshot requires D2Q9 or D3Q27");
    std::map<int,Frame const*> levels;
    std::map<int,Real> times;
    for (auto const& frame : frames) {
        auto inserted = times.emplace(frame.step,frame.physicalTime);
        require(inserted.first->second == frame.physicalTime, "snapshot frames disagree on time");
        auto level = levels.emplace(frame.level,&frame).first->second;
        require(level->populations.domain == frame.populations.domain &&
            level->populations.boxes == frame.populations.boxes,"static snapshot layout changed between steps");
    }
    out << std::setprecision(17) << "schema=static-hierarchy-v3\nsample=" << sample
        << "\ndescriptor=" << (q == 9 ? "D2Q9" : "D3Q27") << "\nd=" << (q == 9 ? 2 : 3) << "\nq=" << q
        << "\nrho_reference=" << references[0] << "\nvelocity_reference=" << references[1]
        << "\nsource_id=" << (source ? source : "local-unversioned")
        << "\nprofile=" << profile << "\nlevels=" << levels.size()
        << "\nranks=" << amrex::ParallelDescriptor::NProcs();
#ifdef AMREX_USE_GPU
    out << "\nbackend=gpu";
#else
    out << "\nbackend=cpu";
#endif
    auto triple = [&](char const* key, auto const& values) {
        out << '\n' << key << '=' << values[0] << ',' << values[1] << ',' << values[2];
    };
    triple("base_cells",spec.nCell);
    triple("prob_lo",spec.probLo);
    triple("prob_hi",spec.probHi);
    triple("periodic",spec.isPeriodic);
    triple("ref_ratio",spec.refRatio);
    out << "\ngap_fill_radius_cells=" << spec.gapFillRadiusCells
        << "\ncoarse_nesting_margin_cells=" << spec.coarseNestingMarginCells
        << "\ngrid_eff=" << spec.gridEfficiency << "\nmax_grid_size=";
    for (std::size_t i = 0; i < spec.maxGridSize.size(); ++i) {
        if (i) out << ';';
        auto const& size = spec.maxGridSize[i];
        out << size[0] << ',' << size[1] << ',' << size[2];
    }
    out << "\ncoverage_contract=" << (spec.gridEfficiency == 1.0 ? "exact-repair" : "minimum-repair")
        << "\nregions=";
    if (spec.regions.empty()) out << "none";
    for (std::size_t i = 0; i < spec.regions.size(); ++i) {
        if (i) out << ';';
        auto const& request = spec.regions[i];
        out << request.level;
        for (int d = 0; d < 3; ++d) out << ',' << request.box.lo[d] << ',' << request.box.hi[d];
    }
    out << "\nsteps=";
    bool first = true;
    for (auto const& [step,time] : times) {
        if (!first) out << ',';
        out << step;
        first = false;
    }
    for (auto const& [step,time] : times) out << "\ntime_" << step << '=' << time;
    auto box = [&](amrex::Box const& b) {
        for (int d = 0; d < 3; ++d) {
            if (d) out << ',';
            out << b.smallEnd(d) << ',' << b.bigEnd(d)+1;
        }
    };
    for (auto const& [lev,frame] : levels) {
        out << "\ndomain_" << lev << '=';
        box(frame->populations.domain);
        out << "\nboxes_" << lev << '=';
        auto const& boxes = frame->populations.boxes;
        for (int i = 0; i < boxes.size(); ++i) {
            if (i) out << ';';
            box(boxes[i]);
        }
    }
    out << '\n';
    require(out.good(), "cannot write static snapshot identity");
}

void write_static_snapshot(std::string const& directory, std::string const& sample,
    amr::StaticHierarchySpec const& spec, std::vector<Frame> const& frames,
    std::string const& profile)
{
    write_frames(directory,frames);
    write_static_identity(directory,sample,spec,frames,profile);
}

} // namespace lbm::tests::static_hierarchy
