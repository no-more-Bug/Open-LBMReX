#include "integration/runtime/AmrAdvanceEvidence.H"

namespace lbm::tests::amr_evolution {
AdvanceEvidence::AdvanceEvidence(std::string directory, std::string sample, int steps, Real tau, bool uniform)
    : directory_(std::move(directory)), sample_(std::move(sample)), steps_(steps), tau_(tau), uniform_(uniform)
{}

void AdvanceEvidence::identity(EvolutionView const& view)
{
    if (!amrex::ParallelDescriptor::IOProcessor()) return;
    std::filesystem::create_directories(directory_);
    std::ofstream identity(directory_+"/identity.csv"), layout(directory_+"/layout.csv");
    identity << std::setprecision(std::numeric_limits<Real>::max_digits10)
        << "schema,sample,descriptor,backend,real_digits,ranks,levels,coarse_steps,tau_finest,uniform,nx,ny,nz,lx,ly,lz,dt0\n"
        << "amr-advance-v1," << sample_ << ',' << Descriptor::name() << ',';
#ifdef AMREX_USE_CUDA
    identity << "cuda";
#else
    identity << "cpu";
#endif
    identity << ',' << std::numeric_limits<Real>::digits << ',' << amrex::ParallelDescriptor::NProcs()
        << ',' << view.levels.size() << ',' << steps_ << ',' << tau_ << ',' << uniform_;
    auto const& g=view.levels[0].geometry;
    for (int d=0; d<3; ++d) identity << ',' << g.Domain().length(d);
    for (int d=0; d<3; ++d) identity << ',' << g.ProbHi(d)-g.ProbLo(d);
    identity << ',' << Real(0.02)*g.CellSize(0) << '\n';
    layout << "level,box,owner,ilo,jlo,klo,ihi,jhi,khi\n";
    for (std::size_t lev=0; lev<view.levels.size(); ++lev) {
        auto const& l=view.levels[lev];
        for (int b=0; b<l.boxes.size(); ++b) {
            layout << lev << ',' << b << ',' << l.mapping[b];
            for (int d=0; d<3; ++d) layout << ',' << l.boxes[b].smallEnd(d);
            for (int d=0; d<3; ++d) layout << ',' << l.boxes[b].bigEnd(d);
            layout << '\n';
        }
    }
    require(identity.good() && layout.good(),"cannot write advance identity/layout");
}

void AdvanceEvidence::record(EvolutionView const& view, std::vector<Frame> const& frames,
                             std::vector<amr_reference::ReferenceLevel> const& reference)
{
    auto const totals=composite(view,frames,reference);
    Real domainVolume=1;
    for (int d=0; d<3; ++d)
        domainVolume*=view.levels[0].geometry.ProbHi(d)-view.levels[0].geometry.ProbLo(d);
    close(Real(totals.volume),domainVolume,"advance composite measure",domainVolume);
    if (view.step==0) {
        initialMass_=Real(totals.mass);
        for (int d=0; d<3; ++d) initialMomentum_[d]=Real(totals.momentum[d]);
        identity(view);
    }
    std::array<Real,3> referenceMomentum;
    for (int d=0; d<3; ++d) referenceMomentum[d]=Real(totals.referenceMomentum[d]);
    compare_totals(totals,Real(totals.referenceMass),referenceMomentum,initialMass_,"advance composite/reference");
    compare_totals(totals,initialMass_,initialMomentum_,initialMass_,"advance composite conservation");
    static_hierarchy::write_frames(directory_+"/step-"+std::to_string(view.step),frames);
    if (!amrex::ParallelDescriptor::IOProcessor()) return;
    std::ofstream balance(directory_+"/balance.csv",view.step==0 ? std::ios::out : std::ios::app);
    if (view.step==0) balance << "step,time_phys,cells,volume,mass,px,py,pz,reference_mass,reference_px,reference_py,reference_pz\n";
    balance << std::setprecision(std::numeric_limits<Real>::max_digits10) << view.step << ','
            << view.physicalTime << ',' << totals.cells << ',' << totals.volume << ',' << totals.mass;
    for (auto p : totals.momentum) balance << ',' << p;
    balance << ',' << totals.referenceMass;
    for (auto p : totals.referenceMomentum) balance << ',' << p;
    balance << '\n';
    require(balance.good(),"cannot write advance balance");
}
} // namespace lbm::tests::amr_evolution
