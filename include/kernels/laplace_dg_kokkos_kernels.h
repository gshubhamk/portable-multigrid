#ifndef laplace_dg_kokkos_kernels_h
#define laplace_dg_kokkos_kernels_h

#include <deal.II/base/memory_space.h>
#include <deal.II/base/utilities.h>

#include <Kokkos_Core.hpp>

#include <vector>

DEAL_II_NAMESPACE_OPEN

namespace BK3
{
  namespace Parallel
  {

    template <typename Number>
    using DeviceView = Kokkos::View<Number *, MemorySpace::Default::kokkos_space>;

    using DoFIndicesView = Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>;

    template <int dim, int nm, int nq, typename Number>
    void
    compute_cell(const DeviceView<Number>                                    d_shape_values,
                 const DeviceView<Number>                                    d_co_shape_gradients,
                 const DeviceView<Number>                                    d_G,
                 const DeviceView<Number>                                    d_in,
                 DeviceView<Number>                                          d_out,
                 Kokkos::View<Number **, MemorySpace::Default::kokkos_space> quad_values,
                 const DoFIndicesView                                        dof_indices,
                 const unsigned int                                          n_cells,
                 const unsigned int n_cells_per_batch = numbers::invalid_unsigned_int,
                 const unsigned int n_blocks          = numbers::invalid_unsigned_int,
                 const unsigned int threads_per_block = numbers::invalid_unsigned_int)
    {
      constexpr int nq_total = Utilities::pow(nq, dim);
      constexpr int nm_total = Utilities::pow(nm, dim);


      // finding the batch size
      constexpr int shmemPerBlock = 10800; // total shared memory used per block (KB)

      constexpr int n_scratch_arrays = 1 + dim; // values and gradients in each direction

      const int nelmt = n_cells;

      const int nelmtPerBatch = std::max(1,
                                         ((n_cells_per_batch == numbers::invalid_unsigned_int) ?
                                            (shmemPerBlock / (n_scratch_arrays * nq_total) /
                                             static_cast<int>(sizeof(Number))) :
                                            static_cast<int>(n_cells_per_batch)));

      const int numBlocks = std::max(1,
                                     ((n_blocks == numbers::invalid_unsigned_int) ?
                                        ((nelmt + nelmtPerBatch - 1) / nelmtPerBatch / 2) :
                                        static_cast<int>(n_blocks)));


      const int threadsPerBlock = std::max(1,
                                           ((threads_per_block == numbers::invalid_unsigned_int) ?
                                              (Utilities::pow(nq, dim - 1) * nelmtPerBatch) :
                                              static_cast<int>(threads_per_block)));

      // if (n_blocks == numbers::invalid_unsigned_int)
      // n_blocks = (nelmt + nelmtPerBatch - 1) / nelmtPerBatch / 2;

      // if (n_blocks == 0)
      // n_blocks = 1;

      // if (threadsPerBlock == numbers::invalid_unsigned_int)
      //   threadsPerBlock = nq * nq * std::max(1u, nelmtPerBatch);


      {
        const int ssize = nm * nq + // shape values
                          nq * nq + // co-shape gradients
                          n_scratch_arrays * nelmtPerBatch *
                            nq_total; // working scratch arrays: scratch_values, scratch_grads_0,
                                      // scratch_grads_1, scratch_grads_2


        const int shmem_size = ssize * sizeof(Number);

        typedef Kokkos::TeamPolicy<>::member_type member_type;
        Kokkos::TeamPolicy<>                      policy(numBlocks, threadsPerBlock);
        policy.set_scratch_size(0, Kokkos::PerTeam(shmem_size));

        Kokkos::parallel_for(
          policy, KOKKOS_LAMBDA(member_type team_member) {
            Number r_p[nq];
            Number r_q[nq];
            Number r_r[nq];

            Number *scratch = (Number *)team_member.team_shmem().get_shmem(shmem_size);

            Number *s_shape_values       = scratch;
            Number *s_co_shape_gradients = s_shape_values + nq * nm;

            Number *scratch_values  = s_co_shape_gradients + nq * nq;
            Number *scratch_grads_0 = scratch_values + nelmtPerBatch * nq_total;
            Number *scratch_grads_1 = scratch_grads_0 + nelmtPerBatch * nq_total;

            Number *scratch_grads_2;
            if (dim == 3)
              scratch_grads_2 = scratch_grads_1 + nelmtPerBatch * nq_total;


            const int threadIdx = team_member.team_rank();
            const int blockSize = team_member.team_size();

            // copy to shared memory
            for (int tid = threadIdx; tid < nm * nq; tid += blockSize)
              {
                s_shape_values[tid] = d_shape_values[tid];
              }

            for (int tid = threadIdx; tid < nq * nq; tid += blockSize)
              {
                s_co_shape_gradients[tid] = d_co_shape_gradients[tid];
              }
            team_member.team_barrier();

            /*
            Interpolate to GL nodes
            */

            // element batch iteration
            int eb = team_member.league_rank();
            while (eb < (nelmt + nelmtPerBatch - 1) / nelmtPerBatch)
              {
                // current nelmtPerBatch (edge case, last batch size can be
                // less)
                int c_nelmtPerBatch = (eb * nelmtPerBatch + nelmtPerBatch > nelmt) ?
                                        (nelmt - eb * nelmtPerBatch) :
                                        nelmtPerBatch;

                // step-1 : Copy from in to the scratch values
                {
                  for (int tid = threadIdx; tid < c_nelmtPerBatch * nm_total; tid += blockSize)
                    {
                      const int e         = tid / nm_total;
                      const int local_idx = tid % nm_total;

                      const int global_cell_index = eb * nelmtPerBatch + e;

                      // Fetch the global DoF index
                      const int dof_index = dof_indices(local_idx, global_cell_index);

                      if (dof_index == numbers::invalid_unsigned_int)
                        scratch_values[tid] = 0;
                      else
                        scratch_values[tid] = d_in[dof_index];
                    }
                  team_member.team_barrier();
                }

                // interpolate dof values to quadrature points in each direction
                {
                  // direction 0
                  {
                    constexpr int co_dimension_size = Utilities::pow(nm, dim - 1);

                    for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                         tid += blockSize)
                      {
                        const int e = tid / co_dimension_size;

                        if (dim == 2)
                          {
                            const int j = tid % co_dimension_size;

                            for (int i = 0; i < nm; ++i)
                              {
                                r_p[i] = scratch_values[e * nm * nm + j * nm + i];
                              }

                            for (int p = 0; p < nq; ++p)
                              {
                                Number tmp = 0.0;

                                for (int i = 0; i < nm; ++i)
                                  {
                                    tmp += s_shape_values[i * nq + p] * r_p[i];
                                  }

                                scratch_grads_0[e * nq * nm + j * nq + p] = tmp;
                              }
                          }
                        else if (dim == 3)
                          {
                            const int k = (tid % co_dimension_size) / nm;
                            const int j = tid % nm;

                            for (int i = 0; i < nm; ++i)
                              {
                                r_p[i] =
                                  scratch_values[e * nm * nm * nm + k * nm * nm + j * nm + i];
                              }

                            for (int p = 0; p < nq; ++p)
                              {
                                Number tmp = 0.0;

                                for (int i = 0; i < nm; ++i)
                                  {
                                    tmp += s_shape_values[i * nq + p] * r_p[i];
                                  }

                                scratch_grads_0[e * nq * nm * nm + k * nq * nm + j * nq + p] = tmp;
                              }
                          }
                      }
                    team_member.team_barrier();
                  }

                  // direction 1
                  {
                    constexpr int co_dimension_size = nq * Utilities::pow(nm, dim - 2);

                    for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                         tid += blockSize)
                      {
                        const int e = tid / co_dimension_size;

                        if (dim == 2)
                          {
                            const int p = tid % co_dimension_size;

                            for (int j = 0; j < nm; ++j)
                              {
                                r_p[j] = scratch_grads_0[e * nq * nm + j * nq + p];
                              }

                            for (int q = 0; q < nq; ++q)
                              {
                                Number tmp = 0.0;

                                for (int j = 0; j < nm; ++j)
                                  {
                                    tmp += s_shape_values[j * nq + q] * r_p[j];
                                  }

                                scratch_values[e * nq * nq + q * nq + p] = tmp;
                              }
                          }
                        else if (dim == 3)
                          {
                            const int k = (tid % co_dimension_size) / nq;
                            const int p = tid % nq;

                            for (int j = 0; j < nm; ++j)
                              {
                                r_q[j] =
                                  scratch_grads_0[e * nq * nm * nm + k * nq * nm + j * nq + p];
                              }

                            for (int q = 0; q < nq; ++q)
                              {
                                Number tmp = 0.0;

                                for (int j = 0; j < nm; ++j)
                                  {
                                    tmp += s_shape_values[j * nq + q] * r_q[j];
                                  }

                                scratch_grads_1[e * nq * nq * nm + k * nq * nq + q * nq + p] = tmp;
                              }
                          }
                      }
                    team_member.team_barrier();
                  }

                  // direction 2
                  if (dim == 3)
                    {
                      constexpr int co_dimension_size = nq * nq;

                      for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                           tid += blockSize)
                        {
                          const int e = tid / co_dimension_size;

                          const int q = (tid % (nq * nq)) / nq;
                          const int p = tid % nq;

                          for (int k = 0; k < nm; ++k)
                            {
                              r_r[k] = scratch_grads_1[e * nq * nq * nm + k * nq * nq + q * nq + p];
                            }
                          for (int r = 0; r < nq; ++r)
                            {
                              Number tmp = 0.0;

                              for (int k = 0; k < nm; ++k)
                                {
                                  tmp += s_shape_values[k * nq + r] * r_r[k];
                                }

                              scratch_values[e * nq * nq * nq + r * nq * nq + q * nq + p] = tmp;
                            }
                        }
                      team_member.team_barrier();
                    }
                }

                // copy quad values to global memory for face integrals
                {
                  for (int tid = threadIdx; tid < c_nelmtPerBatch * nq_total; tid += blockSize)
                    {
                      const int e       = tid / nq_total;
                      const int q_index = tid % nq_total;

                      const int global_cell_index = eb * nelmtPerBatch + e;

                      quad_values(q_index, global_cell_index) =
                        scratch_values[e * nq_total + q_index];
                    }
                  team_member.team_barrier();
                }

                // apply geometric factors and compute stiffness contributions at quadrature points
                {
                  constexpr int co_dimension_size = Utilities::pow(nq, dim - 1);

                  for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                       tid += blockSize)
                    {
                      const int e = tid / co_dimension_size;

                      if (dim == 2)
                        {
                          const int q = tid % co_dimension_size;

                          // copy to register
                          for (int n = 0; n < nq; n++)
                            {
                              r_p[n] = scratch_values[e * nq * nq + q * nq + n];
                              r_q[n] = s_co_shape_gradients[n * nq + q];
                            }

                          Number Grr, Grs, Gss;
                          Number qr, qs;

                          for (int p = 0; p < nq; ++p)
                            {
                              qr = 0;
                              qs = 0;

                              // Load Geometric Factors, coalesced access
                              Grr = d_G[eb * nelmtPerBatch * 3 * nq_total + e * 3 * nq_total +
                                        0 * nq_total + q * nq + p];
                              Grs = d_G[eb * nelmtPerBatch * 3 * nq_total + e * 3 * nq_total +
                                        1 * nq_total + q * nq + p];
                              Gss = d_G[eb * nelmtPerBatch * 3 * nq_total + e * 3 * nq_total +
                                        2 * nq_total + q * nq + p];

                              // Multiply by D
                              for (int n = 0; n < nq; ++n)
                                {
                                  qr += s_co_shape_gradients[n * nq + p] * r_p[n];
                                  qs += r_q[n] * scratch_values[e * nq * nq + n * nq + p];
                                }
                              // Apply chain rule
                              scratch_grads_0[e * nq * nq + q * nq + p] = Grr * qr + Grs * qs;
                              scratch_grads_1[e * nq * nq + q * nq + p] = Grs * qr + Gss * qs;
                            }
                        }
                      else if (dim == 3)
                        {
                          int r = (tid % co_dimension_size) / nq;
                          int q = tid % nq;

                          // copy to register
                          for (int n = 0; n < nq; n++)
                            {
                              r_p[n] = scratch_values[e * nq * nq * nq + r * nq * nq + q * nq + n];
                              r_q[n] = s_co_shape_gradients[n * nq + q];
                              r_r[n] = s_co_shape_gradients[n * nq + r];
                            }

                          Number Grr, Grs, Grt, Gss, Gst, Gtt;
                          Number qr, qs, qt;

                          for (int p = 0; p < nq; ++p)
                            {
                              qr = 0;
                              qs = 0;
                              qt = 0;

                              // Load Geometric Factors, coalesced access
                              Grr = d_G[eb * nelmtPerBatch * 6 * nq_total + e * 6 * nq_total +
                                        0 * nq_total + r * nq * nq + q * nq + p];

                              Grs = d_G[eb * nelmtPerBatch * 6 * nq_total + e * 6 * nq_total +
                                        1 * nq_total + r * nq * nq + q * nq + p];

                              Grt = d_G[eb * nelmtPerBatch * 6 * nq_total + e * 6 * nq_total +
                                        2 * nq_total + r * nq * nq + q * nq + p];

                              Gss = d_G[eb * nelmtPerBatch * 6 * nq_total + e * 6 * nq_total +
                                        3 * nq_total + r * nq * nq + q * nq + p];

                              Gst = d_G[eb * nelmtPerBatch * 6 * nq_total + e * 6 * nq_total +
                                        4 * nq_total + r * nq * nq + q * nq + p];

                              Gtt = d_G[eb * nelmtPerBatch * 6 * nq_total + e * 6 * nq_total +
                                        5 * nq_total + r * nq * nq + q * nq + p];

                              // Multiply by D
                              for (int n = 0; n < nq; n++)
                                {
                                  qr += s_co_shape_gradients[n * nq + p] * r_p[n];
                                  qs += r_q[n] *
                                        scratch_values[e * nq * nq * nq + r * nq * nq + n * nq + p];
                                  qt += r_r[n] *
                                        scratch_values[e * nq * nq * nq + n * nq * nq + q * nq + p];
                                }

                              // Apply chain rule
                              scratch_grads_0[e * nq * nq * nq + r * nq * nq + q * nq + p] =
                                Grr * qr + Grs * qs + Grt * qt;

                              scratch_grads_1[e * nq * nq * nq + r * nq * nq + q * nq + p] =
                                Grs * qr + Gss * qs + Gst * qt;

                              scratch_grads_2[e * nq * nq * nq + r * nq * nq + q * nq + p] =
                                Grt * qr + Gst * qs + Gtt * qt;
                            }
                        }
                    }
                  team_member.team_barrier();
                }

                // apply D^T
                {
                  constexpr int co_dimension_size = Utilities::pow(nq, dim - 1);

                  for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                       tid += blockSize)
                    {
                      const int e = tid / co_dimension_size;

                      if (dim == 2)
                        {
                          const int q = tid % co_dimension_size;

                          // copy to register
                          for (int n = 0; n < nq; n++)
                            {
                              r_p[n] = scratch_grads_0[e * nq * nq + q * nq + n];
                              r_q[n] = s_co_shape_gradients[q * nq + n];
                            }

                          for (int p = 0; p < nq; ++p)
                            {
                              Number tmp = 0;

                              for (int n = 0; n < nq; ++n)
                                tmp += s_co_shape_gradients[p * nq + n] * r_p[n];

                              for (int n = 0; n < nq; ++n)
                                tmp += scratch_grads_1[e * nq * nq + q * nq + n] * r_q[n];

                              scratch_values[e * nq * nq + q * nq + p] = tmp;
                            }
                        }
                      else if (dim == 3)
                        {
                          const int r = tid % (nq * nq) / nq;
                          const int q = tid % nq;

                          // copy to register
                          for (int n = 0; n < nq; n++)
                            {
                              r_p[n] = scratch_grads_0[e * nq * nq * nq + r * nq * nq + q * nq + n];
                              r_q[n] = s_co_shape_gradients[q * nq + n];
                              r_r[n] = s_co_shape_gradients[r * nq + n];
                            }

                          for (int p = 0; p < nq; ++p)
                            {
                              Number tmp = 0;
                              for (int n = 0; n < nq; ++n)
                                tmp += r_p[n] * s_co_shape_gradients[p * nq + n];

                              for (int n = 0; n < nq; ++n)
                                tmp +=
                                  scratch_grads_1[e * nq * nq * nq + r * nq * nq + n * nq + p] *
                                  r_q[n];

                              for (int n = 0; n < nq; ++n)
                                tmp +=
                                  scratch_grads_2[e * nq * nq * nq + n * nq * nq + q * nq + p] *
                                  r_r[n];

                              scratch_values[e * nq * nq * nq + r * nq * nq + q * nq + p] = tmp;
                            }
                        }
                      team_member.team_barrier();
                    }
                }

                /*
                Interpolate to GLL nodes
                */
                {
                  // direction 2
                  if (dim == 3)
                    {
                      constexpr int co_dimension_size = Utilities::pow(nq, dim - 1);

                      for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                           tid += blockSize)
                        {
                          const int e = tid / co_dimension_size;

                          const int q = (tid % co_dimension_size) / nq;
                          const int p = tid % nq;

                          for (int r = 0; r < nq; ++r)
                            {
                              r_r[r] = scratch_values[e * nq * nq * nq + r * nq * nq + q * nq + p];
                            }

                          for (int k = 0; k < nm; ++k)
                            {
                              Number tmp = 0.0;

                              for (int r = 0; r < nq; ++r)
                                {
                                  tmp += scratch_grads_0[k * nq + r] * r_r[r];
                                }

                              scratch_grads_0[e * nq * nq * nm + k * nq * nq + q * nq + p] = tmp;
                            }
                        }
                      team_member.team_barrier();
                    }

                  //  direction 1
                  {
                    constexpr int co_dimension_size = nq * Utilities::pow(nm, dim - 2);

                    for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                         tid += blockSize)
                      {
                        const int e = tid / co_dimension_size;

                        if (dim == 2)
                          {
                            const int p = tid % co_dimension_size;

                            for (int q = 0; q < nq; ++q)
                              {
                                r_q[q] = scratch_values[e * nq * nq + q * nq + p];
                              }

                            for (int j = 0; j < nm; ++j)
                              {
                                Number tmp = 0.0;

                                for (int q = 0; q < nq; ++q)
                                  {
                                    tmp += s_shape_values[j * nq + q] * r_q[q];
                                  }
                                scratch_grads_1[e * nq * nm + j * nq + p] = tmp;
                              }
                          }
                        else if (dim == 3)
                          {
                            int k = (tid % co_dimension_size) / nq;
                            int p = tid % nq;

                            for (int j = 0; j < nq; ++j)
                              {
                                r_q[j] =
                                  scratch_grads_0[e * nq * nq * nm + k * nq * nq + j * nq + p];
                              }

                            for (int j = 0; j < nm; ++j)
                              {
                                Number tmp = 0.0;

                                for (int q = 0; q < nq; ++q)
                                  {
                                    tmp += s_shape_values[j * nq + q] * r_q[q];
                                  }
                                scratch_grads_1[e * nq * nm * nm + k * nq * nm + j * nq + p] = tmp;
                              }
                          }
                      }
                    team_member.team_barrier();
                  }

                  // direction 0
                  {
                    const int co_dimension_size = Utilities::pow(nm, dim - 1);

                    for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                         tid += blockSize)
                      {
                        const int e = tid / co_dimension_size;

                        if (dim == 2)
                          {
                            const int j = tid % co_dimension_size;
                            for (int p = 0; p < nq; ++p)
                              {
                                r_p[p] = scratch_grads_1[e * nq * nq + j * nq + p];
                              }

                            for (int i = 0; i < nm; ++i)
                              {
                                Number tmp = 0.0;
                                for (int p = 0; p < nq; ++p)
                                  {
                                    tmp += s_shape_values[i * nq + p] * r_p[p];
                                  }
                                scratch_values[e * nm * nm + j * nm + i] = tmp;
                              }
                          }
                        else if (dim == 3)
                          {
                            int k = (tid % co_dimension_size) / nm;
                            int j = tid % nm;

                            for (int i = 0; i < nq; ++i)
                              {
                                r_p[i] =
                                  scratch_grads_1[e * nq * nm * nm + k * nq * nm + j * nq + i];
                              }

                            for (int p = 0; p < nm; ++p)
                              {
                                Number tmp = 0.0;
                                for (int i = 0; i < nq; ++i)
                                  {
                                    tmp += s_shape_values[p * nq + i] * r_p[i];
                                  }
                                scratch_values[e * nm * nm * nm + k * nm * nm + j * nm + p] = tmp;
                              }
                          }
                      }
                    team_member.team_barrier();
                  }
                }

                // step-12 : Copy wsp0 (result) back to global out vector
                for (int tid = threadIdx; tid < c_nelmtPerBatch * nm_total; tid += blockSize)
                  {
                    const int e = tid / nm_total;

                    const int local_idx = tid % nm_total;

                    const int global_cell_index = eb * nelmtPerBatch + e;

                    // Find where this node lives in the global 'd_out'
                    // vector
                    const int dof_index = dof_indices(local_idx, global_cell_index);

                    if (dof_index != numbers::invalid_unsigned_int)
                      {
                        // CRITICAL: Use atomic_add because elements share
                        // nodes!
                        Kokkos::atomic_add(&d_out[dof_index], scratch_values[tid]);
                      }
                  }

                team_member.team_barrier();

                eb += team_member.league_size();
              }
          });
        Kokkos::fence();
      }
    }

  } // namespace Parallel
} // namespace BK3

DEAL_II_NAMESPACE_CLOSE

#endif