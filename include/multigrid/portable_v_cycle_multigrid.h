#ifndef portable_v_cycle_multigrid_h
#define portable_v_cycle_multigrid_h

#include <deal.II/base/enable_observer_pointer.h>
#include <deal.II/base/mg_level_object.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/matrix_free/portable_matrix_free.h>

#include <deal.II/multigrid/mg_smoother.h>

#include <Kokkos_Core.hpp>

#include "base/portable_laplace_operator_base.h"

DEAL_II_NAMESPACE_OPEN

namespace Portable
{

  template <typename VectorType, typename SmootherType>
  class MGCoarseFromSmoother : public MGCoarseGridBase<VectorType>
  {
  public:
    MGCoarseFromSmoother(const SmootherType &mg_smoother, const bool is_empty)
      : smoother(mg_smoother)
      , is_empty(is_empty)
    {}

    virtual void
    operator()(const unsigned int level,
               VectorType        &dst,
               const VectorType  &src) const override
    {
      if (is_empty)
        return;
      smoother[level].vmult(dst, src);
    }

    const SmootherType &smoother;
    const bool          is_empty;
  };


  template <int dim, typename number, typename TransferType>
  class VCycleMultigrid : public EnableObserverPointer
  {
  public:
    using VectorType =
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default>;
    using LevelMatrixType = LaplaceOperatorBase<dim, number>;
    using SmootherType    = PreconditionChebyshev<LevelMatrixType, VectorType>;

    VCycleMultigrid(
      const MGLevelObject<std::unique_ptr<LevelMatrixType>> &mg_matrices,
      const MGLevelObject<std::unique_ptr<TransferType>>    &mg_transfers,
      const MGLevelObject<SmootherType>                     &mg_smoothers,
      const VectoreType                                     &right_hand_side,
      const unsigned int pre_smoothing_steps,
      const unsigned int post_smoothing_steps);

    void
    vmult(VectorType &dst, const VectorType &src) const;

    std::pair<unsigned int, double>
    solve_cg();

  private:
    void
    smooth(VectorType        &u,
           const VectorType  &rhs,
           const unsigned int level) const;

    void
    v_cycle(const unsigned int level) const;

    /**
     * Lowest level of cells.
     */
    unsigned int minlevel;

    /**
     * Highest level of cells.
     */
    unsigned int maxlevel;

    const MGLevelObject<std::unique_ptr<LevelMatrixType>> &mg_matrices;
    const MGLevelObject<std::unique_ptr<TransferType>>    &mg_transfers;

    const MGLevelObject<SmootherType> &mg_smoothers;

    /**
     * The coarse solver
     */
    MGCoarseFromSmoother<VectorTypeDevice, MGLevelObject<SmootherType>> coarse;

    const VectorType &rhs;

    /**
     * The solution update after the multigrid step.
     */
    mutable MGLevelObject<VectorTypeDevice> solution;

    /**
     * Input vector for the cycle. Contains the defect of the outer method
     * projected to the multilevel vectors.
     */
    mutable MGLevelObject<VectorTypeDevice> defect;

    /**
     * Auxiliary vector.
     */
    mutable MGLevelObject<VectorTypeDevice> t;


    const unsigned int pre_smoothing_steps;
    const unsigned int post_smoothing_steps;
  };

  template <int dim, typename number, typename TransferType>
  VCycleMultigrid<dim, number, TransferType>::VCycleMultigrid(
    const MGLevelObject<std::unique_ptr<LevelMatrixType>> &mg_matrices,
    const MGLevelObject<std::unique_ptr<TransferType>>    &mg_transfers,
    const MGLevelObject<SmootherType>                     &mg_smoothers,
    const VectoreType                                     &right_hand_side,
    const unsigned int                                     pre_smoothing_steps,
    const unsigned int                                     post_smoothing_steps)
    : minlevel(mg_matrices.min_level())
    , maxlevel(mg_matrices.max_level())
    , mg_matrices(mg_matrices)
    , mg_transfers(mg_transfers)
    , mg_smoothers(mg_smoothers)
    , coarse(mg_smoothers, false)
    , rhs(right_hand_side)
    , solution(minlevel, maxlevel)
    , defect(minlevel, maxlevel)
    , t(minlevel, maxlevel)
    , pre_smoothing_steps(pre_smoothing_steps)
    , post_smoothing_steps(post_smoothing_steps)
  {
    Assert(pre_smoothing_steps == post_smoothing_steps,
           ExcNotImplemented("Change of pre- and post-smoother degree "
                             "currently not possible with deal.II"));
    for (unsigned int level = minlevel; level <= maxlevel; ++level)
      {
        mg_matrices[level]->initialize_dof_vector(solution[level]);
        defect[level] = solution[level];
        t[level]      = solution[level];
      }
  }

  std::pair<unsigned int, double>
  solve_cg()
  {
    ReductionControl solver_control(100, 1e-16, 1e-9);

    SolverCG<VectorType> solver_cg(solver_control);

    VectorType solution_update = solution[maxlevel];
    solution_update            = 0;

    solver_cg.solve(*mg_matrices[maxlevel], solution_update, rhs, *this);

    solution[maxlevel] = solution_update;

    return std::make_pair(solver_control.last_step(),
                          std::pow(solver_control.last_value() /
                                     solver_control.initial_value(),
                                   1. / solver_control.last_step()));
  }

  template <int dim, typename number, typename TransferType>
  void
  VCycleMultigrid<dim, number, TransferType>::vmult(VectorType       &dst,
                                                    const VectorType &src) const
  {
    for (unsigned int level = minlevel; level < maxlevel; ++level)
      {
        defect[level] = 0;
      }

    defect[maxlevel] = src;

    v_cycle(maxlevel);

    dst = solution[maxlevel];
  }

  template <int dim, typename number, typename TransferType>
  void
  VCycleMultigrid<dim, number, TransferType>::v_cycle(
    const unsigned int level) const
  {
    if (level == minlevel)
      {
        // Accuracy on coarsest level should be comparable to overall level
        // accuracy (~1e-3)
        (coarse)(level, solution[level], defect[level]);

        return;
      }

    // Pre-smoothing
    mg_smoothers[level].vmult(solution[level], defect[level]);

    // Compute residual
    mg_matrices[level]->vmult(t[level], solution[level]);
    t[level].sadd(-1.0, 1.0, defect[level]);

    // Restrict residual to the next coarser level
    defect[level - 1] = 0;
    mg_transfers[level]->restrict_and_add(defect[level - 1], t[level]);

    // Recursive call to v_cycle on the coarser level
    v_cycle(level - 1);

    // Prolongate coarse correction and add to current solution
    mg_transfers[level]->prolongate_and_add(solution[level], solution[level - 1]);

    // Post-smoothing
        mg_smoothers[level].step(solution[level], defect[level]);

  }


} // namespace Portable

DEAL_II_NAMESPACE_CLOSE

#endif
