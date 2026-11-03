#ifndef LBM_TESTS_NUMERICAL_AMRSAMPLEOUTPUT_HH
#define LBM_TESTS_NUMERICAL_AMRSAMPLEOUTPUT_HH

#include "lbm/operators/level/MacroscopicOperator.H"
#include "runtime/SingleLevelPlotfile.H"
#include <filesystem>
#include <iomanip>
#include <limits>

namespace lbm::tests::numerical {

template<class D>
AmrSampleOutput<D>::AmrSampleOutput(std::string directory, amrex::Real velocityReference)
    : directory_(std::move(directory)), velocityReference_(velocityReference)
{
    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::filesystem::create_directories(directory_);
        samples_.open(directory_+"/samples.csv");
        samples_ << std::setprecision(std::numeric_limits<amrex::Real>::max_digits10)
                 << "kind,level,step,time,coarsest_step,phase,path\n";
        if (!samples_) amrex::Abort("cannot create numerical sample index");
    }
    amrex::ParallelDescriptor::Barrier();
}

template<class D>
void AmrSampleOutput<D>::layout(amr::LbmAmrView<D> const& view, std::string const& filename)
{
    if (!amrex::ParallelDescriptor::IOProcessor()) return;
    std::ofstream out(directory_+"/"+filename);
    out << "level,box,rank,ilo,jlo,klo,ihi,jhi,khi\n";
    for (int lev=0; lev<int(view.levels.size()); ++lev) {
        auto const& l=view.levels[lev];
        for (int b=0; b<l.boxes.size(); ++b) {
            out << lev << ',' << b << ',' << l.mapping[b];
            for (int d=0; d<3; ++d) out << ',' << l.boxes[b].smallEnd(d);
            for (int d=0; d<3; ++d) out << ',' << l.boxes[b].bigEnd(d)+1;
            out << '\n';
        }
    }
    if (!out) amrex::Abort("cannot write numerical layout");
}

template<class D>
void AmrSampleOutput<D>::record(std::string const& kind, int level, int step,
    amrex::Real time, int coarsestStep, std::string const& phase, std::string const& path)
{
    if (!amrex::ParallelDescriptor::IOProcessor()) return;
    samples_ << kind << ',' << level << ',' << step << ',' << time << ','
             << coarsestStep << ',' << phase << ',' << path << '\n';
    samples_.flush();
    if (!samples_) amrex::Abort("cannot write numerical sample index");
}

template<class D>
void AmrSampleOutput<D>::synchronized(amr::LbmAmrView<D> const& view)
{
    std::string const path="sync-"+std::to_string(view.step);
    auto result=runtime::write_amr_plotfile(directory_+'/'+path,view,1,velocityReference_);
    if (!result.ok()) amrex::Abort(result.error);
    for (int lev=0; lev<int(view.levels.size()); ++lev) {
        auto const& l=view.levels[lev];
        record("sync",lev,l.step,l.physicalTime,view.step,"Sync",path);
    }
}

template<class D>
void AmrSampleOutput<D>::fine(amr::FineSubstepView<D> const& view)
{
    auto const& l=view.fine;
    MacroField macro("fine-sample",l.boxes,l.mapping);
    operators::MacroscopicOperator<D>{}.apply(l.populations,macro);
    std::string const path="fine-L"+std::to_string(view.level)+'-'+std::to_string(l.step);
    runtime::write_single_level_plotfile(directory_+'/'+path,macro,l.geometry,l.boxes,l.mapping,
                                        1,velocityReference_,l.step,l.physicalTime);
    record("fine",view.level,l.step,l.physicalTime,view.coarsestStep,
           view.substep==amr::FineSubstep::First ? "First" : "Second",path);
}

} // namespace lbm::tests::numerical
#endif
