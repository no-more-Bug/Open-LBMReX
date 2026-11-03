#ifndef LBM_BOUNDARY_DOMAIN_DOMAINBOUNDARYSYSTEM_H
#define LBM_BOUNDARY_DOMAIN_DOMAINBOUNDARYSYSTEM_H

#include "lbm/boundary/domain/DomainFaceMetadata.H"
#include "lbm/boundary/domain/LbmPostStreamingBoundary.H"
#include "lbm/boundary/domain/LbmPostStreamingBoundaryExecutor.H"
#include "lbm/boundary/domain/LbmPhysBCFunct.H"
#include "lbm/boundary/domain/LbmValidBoundaryInitializationOperator.H"
#include "lbm/fields/LatticeField.H"

#include <AMReX.H>
#include <AMReX_BC_TYPES.H>
#include <AMReX_BCRec.H>
#include <AMReX_BoxArray.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_PhysBCFunct.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <sstream>

namespace lbm::boundary {

/**
 * @brief Compiled runtime for physical faces of one immutable single-level Geometry.
 *
 * tryDefine() converts canonical resolved metadata into a device-safe face
 * table and disjoint post-stream region plan. Runtime methods never parse input or
 * select defaults. The object may cache whether Guo donors cross owner FABs,
 * but that decision depends only on the replicated global BoxArray and is
 * therefore identical on every MPI rank.
 */
template <class Descriptor>
class DomainBoundarySystem {
public:
    DomainBoundarySystem() = default;

    /**
     * @brief Compile face metadata, coverage, geometry regions, and AMReX BCRec data.
     *
     * @pre boundaries have passed descriptor-aware semantic resolution.
     * @post The object is bound to geom; donor-access caching may subsequently
     *       be initialized for exactly one population BoxArray.
     */
    /** Result-based compilation used by deep preparation and focused tests. */
    bool tryDefine(
        amrex::Geometry const& geom,
        std::vector<ResolvedDomainBoundary> const& boundaries,
        std::string& error)
    {
        DomainFaceTable table = build_domain_face_table(boundaries, geom);
        std::vector<std::string> coverageErrors =
            validate_domain_face_coverage<Descriptor>(table.view(), geom);
        if (!coverageErrors.empty()) {
            std::ostringstream os;
            os << "DomainBoundarySystem validation failed:";
            for (std::string const& error : coverageErrors) {
                os << "\n  - " << error;
            }
            error = os.str();
            return false;
        }
        PostStreamingBoundaryPlanResult planResult =
            compile_post_streaming_boundary_plan<Descriptor>(geom, boundaries);
        if (!planResult.ok()) {
            error = planResult.error_message();
            return false;
        }

        geom_ = geom;
        table_ = std::move(table);
        postStreamingPlan_ = std::move(planResult.plan);
        bcs_.assign(Descriptor::q, make_lbm_bc_rec());
        donorAccessConfigured_ = false;
        donorFillBoundaryRequired_ = true;
        donorAccessBoxArray_ = amrex::BoxArray{};
        defined_ = true;
        error.clear();
        return true;
    }

    /**
     * @brief Fill pre-stream physical population ghosts through AMReX PhysBCFunct.
     *
     * The population ghost width must equal the descriptor vicinity (currently
     * one). AMReX invokes the device functor only outside the valid+periodic
     * grown domain, so valid cells and Guo on-lattice boundary nodes are not
     * touched. The supplied source physical time is forwarded unchanged to
     * PhysBCFunct.
     */
    void fillPhysicalGhosts(LatticeField<Descriptor>& populations, amrex::Real physicalTime) const
    {
        if (!defined_ || !postStreamingPlan_.hasPreStreamingFaces) {
            return;
        }
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            populations.nGrow() == Descriptor::vicinity,
            "Stage 4.2 domain boundary fill expects population ghost width equal to descriptor vicinity");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            populations.nGrow() == 1,
            "Stage 4.2 domain boundary fill only supports population ghost width 1");

        LbmDomainPopulationFill<Descriptor> fill{table_.view()};
        amrex::GpuBndryFuncFab<LbmDomainPopulationFill<Descriptor>> gpuBndryFunc(fill);
        amrex::PhysBCFunct<amrex::GpuBndryFuncFab<LbmDomainPopulationFill<Descriptor>>>
            physbc(geom_, bcs_, gpuBndryFunc);
        const amrex::IntVect nghost(populations.nGrow());
        physbc(populations.multifab(), 0, Descriptor::q, nghost, physicalTime, 0);
    }

    /**
     * @brief Initialize valid-node boundaries with equilibrium populations.
     *
     * Bulk valid populations must already be initialized. HBB-only plans return
     * without writing valid cells; Guo plans synchronize inward donors if needed
     * and overwrite every population on the compiled boundary shell at t_0.
     * Runtime donor non-equilibrium reconstruction is deliberately not used.
     */
    void initializeValidBoundaries(
        LatticeField<Descriptor>& populations,
        amrex::Real physicalTime) const
    {
        if (!prepare_post_stream_donors(populations)) {
            return;
        }
        validBoundaryInitialization_.apply(
            populations, table_.view(), postStreamingPlan_, geom_, physicalTime);
    }

    /**
     * @brief Reconstruct the post-pull destination layer on compiled valid nodes.
     *
     * @pre Pull streaming has written the complete valid domain, including the
     *      boundary shell, at the destination population layer.
     * @param physicalTime Target physical time of this advance.
     */
    void applyPostStreamingValidNodes(
        LatticeField<Descriptor>& populations,
        amrex::Real physicalTime) const
    {
        if (!prepare_post_stream_donors(populations)) {
            return;
        }
        postStreamingExecutor_.apply(
            populations, table_.view(), postStreamingPlan_, geom_, physicalTime);
    }

    DomainFaceTable const& table() const noexcept { return table_; }
    PostStreamingBoundaryPlan const& postStreamingPlan() const noexcept
    {
        return postStreamingPlan_;
    }
    bool hasPreStreamingBoundaries() const noexcept
    {
        return postStreamingPlan_.hasPreStreamingFaces;
    }
    bool hasPostStreamingBoundaries() const noexcept
    {
        return postStreamingPlan_.hasPostStreamingFaces;
    }
    bool defined() const noexcept { return defined_; }

private:
    /**
     * @brief Establish donor access before either valid-node execution path.
     *
     * cross=false is required because diagonal edge/corner donors may be stored
     * in corner ghosts. When every shifted donor stays in the same owner FAB,
     * Array4 access spans that full FAB allocation and communication is skipped.
     * The returned decision depends only on replicated plan/BoxArray metadata,
     * so every rank enters or skips FillBoundary symmetrically.
     */
    bool prepare_post_stream_donors(
        LatticeField<Descriptor>& populations) const
    {
        if (!defined_ || !postStreamingPlan_.hasPostStreamingFaces) {
            return false;
        }
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            populations.nGrow() >= Descriptor::vicinity,
            "post-stream Guo donor synchronization requires descriptor ghost width");
        configure_donor_access(populations.multifab().boxArray());

        if (donorFillBoundaryRequired_) {
            // cross=false is intentional: diagonal edge/corner donors may live
            // in corner ghosts when the owner and x_1 are different FABs.
            populations.fillBoundary(geom_, false);
        }
        return true;
    }

    /**
     * @brief Cache the global BoxArray donor-locality policy.
     *
     * Every rank sees the same BoxArray and compiled plan, so all ranks either
     * call or skip FillBoundary together. Reusing this runtime with a different
     * decomposition is rejected rather than risking asymmetric collectives.
     */
    void configure_donor_access(amrex::BoxArray const& boxArray) const
    {
        if (donorAccessConfigured_) {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                donorAccessBoxArray_ == boxArray,
                "post-stream donor access policy cannot be reused with a different BoxArray");
            return;
        }
        donorAccessBoxArray_ = boxArray;
        donorFillBoundaryRequired_ =
            post_stream_donor_fill_boundary_required(
                postStreamingPlan_, donorAccessBoxArray_);
        donorAccessConfigured_ = true;
    }

    static amrex::BCRec make_lbm_bc_rec()
    {
        amrex::BCRec rec;
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            rec.setLo(dir, amrex::BCType::user_1);
            rec.setHi(dir, amrex::BCType::user_1);
        }
        return rec;
    }

    bool defined_ = false;
    amrex::Geometry geom_;
    DomainFaceTable table_;
    PostStreamingBoundaryPlan postStreamingPlan_;
    LbmValidBoundaryInitializationOperator<Descriptor>
        validBoundaryInitialization_;
    LbmPostStreamingBoundaryExecutor<Descriptor> postStreamingExecutor_;
    amrex::Vector<amrex::BCRec> bcs_;
    mutable bool donorAccessConfigured_ = false;
    mutable bool donorFillBoundaryRequired_ = true;
    mutable amrex::BoxArray donorAccessBoxArray_;
};

} // namespace lbm::boundary

#endif
