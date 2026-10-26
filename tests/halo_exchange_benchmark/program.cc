// Benchmark of the ghost value exchange (update_ghost_values()) on the
// partitioners of the fine levels of tests/poisson_cube_bk3_dg: FE_Q with the
// degrees of the p-multigrid levels and FE_DGQ on the doubling mesh of the
// given cycle. The partitioners are built in the same way as in
// Portable::MatrixFree (locally owned and locally relevant DoFs).
//
// Variants:
//   deal.II ghost      LinearAlgebra::distributed::Vector::update_ghost_values()
//   deal.II compress   compress(VectorOperation::add), the second exchange of
//                      the continuous operators (no correctness check)
//   local only         pack kernel and copy into the ghost range without any
//                      communication, i.e. the kernel launch and fence overhead
//   MPI device         pack kernel, fence, MPI_Isend/Irecv on device buffers
//   MPI host           as above, but staged through pinned host buffers
//   HaloExchange       Portable::HaloExchange (include/base), the fused
//                      exchange used by the operators; its compress_add() is
//                      also compared bitwise with compress(VectorOperation::add)
//   NVSHMEM stream     host-enqueued NVSHMEM operations on the Kokkos stream
//   NVSHMEM device     device-initiated puts from two kernels
// The NVSHMEM variants are timed once with a fence after every exchange
// ("synced", comparable to the MPI variants) and once without ("pipelined",
// the exchanges are only ordered on the stream as in a solver).
//
// Usage: ./program [cycle (17)] [degree (4)] [n_iterations (200)] [nvshmem (1)]
// where nvshmem = 0 skips the initialization of NVSHMEM and its variants.

#include <deal.II/base/mpi.h>
#include <deal.II/base/partitioner.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <sched.h>

#include <array>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "base/portable_halo_exchange.h"

#ifdef HALO_WITH_NVSHMEM
#  include "nvshmem_halo.h"
#endif

namespace halo
{
  using namespace dealii;

  using Number      = float;
  using VectorType  = LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>;
  using DeviceSpace = MemorySpace::Default::kokkos_space;

  template <typename T>
  using DeviceView = Kokkos::View<T *, DeviceSpace>;
  template <typename T>
  using UnmanagedDeviceView =
    Kokkos::View<T *, DeviceSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  template <typename T>
  using PinnedView = Kokkos::View<T *, Kokkos::SharedHostPinnedSpace>;

  // value stored at a global index, exactly representable in float
  inline Number
  test_value(const types::global_dof_index index)
  {
    return static_cast<Number>(index % 1048576);
  }



  // The doubling mesh of tests/poisson_cube_bk3_dg for the given cycle.
  void
  create_mesh(parallel::distributed::Triangulation<3> &triangulation, const unsigned int cycle)
  {
    constexpr int      dim       = 3;
    const unsigned int n_refine  = cycle / 3;
    const unsigned int remainder = cycle % 3;
    Point<dim>         p1;
    for (unsigned int d = 0; d < dim; ++d)
      p1[d] = -1;
    Point<dim> p2;
    for (unsigned int d = 0; d < remainder; ++d)
      p2[d] = 2.8;
    for (unsigned int d = remainder; d < dim; ++d)
      p2[d] = 0.9;
    std::vector<unsigned int> subdivisions(dim, 1);
    for (unsigned int d = 0; d < remainder; ++d)
      subdivisions[d] = 2;
    GridGenerator::subdivided_hyper_rectangle(triangulation, subdivisions, p1, p2);
    triangulation.refine_global(n_refine);
  }



  void
  pack(const UnmanagedDeviceView<Number> &values,
       const DeviceView<unsigned int>    &import_indices,
       const DeviceView<Number>          &send_buffer)
  {
    Kokkos::parallel_for(
      "halo::pack",
      Kokkos::RangePolicy<MemorySpace::Default::kokkos_space::execution_space>(
        0, import_indices.extent(0)),
      KOKKOS_LAMBDA(const int i) { send_buffer(i) = values(import_indices(i)); });
  }



  class LevelBenchmark
  {
  public:
    LevelBenchmark(const std::shared_ptr<const Utilities::MPI::Partitioner> &partitioner,
                   const unsigned int                                        n_iterations);

    void
    run(const bool use_nvshmem);

  private:
    void
    print_pattern_statistics() const;

    // zero the ghost range, run one exchange and compare the ghost range with
    // the expected values; returns the global number of wrong entries
    types::global_dof_index
    check(const std::function<void()> &exchange);

    // compare HaloExchange::compress_add() with compress(VectorOperation::add)
    // on the same data; returns the global number of entries that are not
    // bitwise identical
    types::global_dof_index
    check_compress_add(const Portable::HaloExchange<Number> &halo_exchange) const;

    void
    time(const std::string           &name,
         const std::function<void()> &exchange,
         const bool                   check_result,
         const bool                   fence_every_iteration);

    void
    exchange_mpi(const bool stage_on_host);

    void
    exchange_local_only();

    std::shared_ptr<const Utilities::MPI::Partitioner> partitioner;
    const MPI_Comm                                     comm;
    const unsigned int                                 n_iterations;

    VectorType                  vector;
    UnmanagedDeviceView<Number> values;
    UnmanagedDeviceView<Number> ghost_values;
    std::vector<Number>         expected_ghost_values;

    DeviceView<unsigned int> import_indices;
    DeviceView<Number>       send_buffer;
    DeviceView<Number>       local_recv_buffer;
    PinnedView<Number>       send_buffer_host;
    PinnedView<Number>       recv_buffer_host;
  };



  LevelBenchmark::LevelBenchmark(
    const std::shared_ptr<const Utilities::MPI::Partitioner> &partitioner,
    const unsigned int                                        n_iterations)
    : partitioner(partitioner)
    , comm(partitioner->get_mpi_communicator())
    , n_iterations(n_iterations)
  {
    vector.reinit(partitioner);
    const unsigned int n_owned = partitioner->locally_owned_size();
    const unsigned int n_ghost = partitioner->n_ghost_indices();
    values       = UnmanagedDeviceView<Number>(vector.get_values(), n_owned + n_ghost);
    ghost_values = UnmanagedDeviceView<Number>(vector.get_values() + n_owned, n_ghost);

    std::vector<Number> owned_values(n_owned);
    for (unsigned int i = 0; i < n_owned; ++i)
      owned_values[i] = test_value(partitioner->local_range().first + i);
    Kokkos::deep_copy(
      Kokkos::subview(values, Kokkos::make_pair(0u, n_owned)),
      Kokkos::View<Number *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
        owned_values.data(), n_owned));

    for (const auto index : partitioner->ghost_indices())
      expected_ghost_values.push_back(test_value(index));

    std::vector<unsigned int> import_indices_host;
    for (const auto &[begin, end] : partitioner->import_indices())
      for (unsigned int i = begin; i < end; ++i)
        import_indices_host.push_back(i);
    AssertDimension(import_indices_host.size(), partitioner->n_import_indices());

    import_indices = DeviceView<unsigned int>("import_indices", import_indices_host.size());
    Kokkos::deep_copy(
      import_indices,
      Kokkos::View<unsigned int *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
        import_indices_host.data(), import_indices_host.size()));
    send_buffer       = DeviceView<Number>("send_buffer", import_indices_host.size());
    local_recv_buffer = DeviceView<Number>("local_recv_buffer", n_ghost);
    send_buffer_host  = PinnedView<Number>("send_buffer_host", import_indices_host.size());
    recv_buffer_host  = PinnedView<Number>("recv_buffer_host", n_ghost);
    Kokkos::fence();
  }



  void
  LevelBenchmark::print_pattern_statistics() const
  {
    ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(comm) == 0);

    const auto print = [&](const std::string &name, const double value)
    {
      const auto stats = Utilities::MPI::min_max_avg(value, comm);
      pcout << "  " << std::left << std::setw(24) << name << std::right << std::setw(12)
            << stats.min << std::setw(12) << stats.avg << std::setw(12) << stats.max << std::endl;
    };

    pcout << "  " << std::left << std::setw(24) << "per rank" << std::right << std::setw(12)
          << "min" << std::setw(12) << "avg" << std::setw(12) << "max" << std::endl;
    print("owned DoFs", partitioner->locally_owned_size());
    print("ghost DoFs (receive)", partitioner->n_ghost_indices());
    print("import DoFs (send)", partitioner->n_import_indices());
    print("ranks to receive from", partitioner->ghost_targets().size());
    print("ranks to send to", partitioner->import_targets().size());
    print("receive volume [kB]", partitioner->n_ghost_indices() * sizeof(Number) / 1024.);

    double max_message = 0;
    for (const auto &[rank, count] : partitioner->ghost_targets())
      max_message = std::max(max_message, count * sizeof(Number) / 1024.);
    print("largest message [kB]", max_message);
  }



  types::global_dof_index
  LevelBenchmark::check(const std::function<void()> &exchange)
  {
    Kokkos::deep_copy(ghost_values, Number(0));
    Kokkos::fence();
    exchange();
    Kokkos::fence();

    std::vector<Number> ghosts(ghost_values.extent(0));
    Kokkos::deep_copy(
      Kokkos::View<Number *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
        ghosts.data(), ghosts.size()),
      ghost_values);

    types::global_dof_index n_wrong = 0;
    for (unsigned int i = 0; i < ghosts.size(); ++i)
      if (ghosts[i] != expected_ghost_values[i])
        ++n_wrong;
    return Utilities::MPI::sum(n_wrong, comm);
  }



  void
  LevelBenchmark::time(const std::string           &name,
                       const std::function<void()> &exchange,
                       const bool                   check_result,
                       const bool                   fence_every_iteration)
  {
    ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(comm) == 0);

    std::string check_text = "-";
    if (check_result)
      {
        const auto n_wrong = check(exchange);
        check_text = n_wrong == 0 ? "ok" : ("FAILED: " + std::to_string(n_wrong) + " wrong");
      }

    for (unsigned int i = 0; i < 10; ++i)
      exchange();
    Kokkos::fence();
    MPI_Barrier(comm);

    Timer timer;
    for (unsigned int i = 0; i < n_iterations; ++i)
      {
        exchange();
        if (fence_every_iteration)
          Kokkos::fence();
      }
    Kokkos::fence();
    const double time_us = timer.wall_time() / n_iterations * 1e6;

    const auto stats = Utilities::MPI::min_max_avg(time_us, comm);
    pcout << "  " << std::left << std::setw(28) << name << std::right << std::fixed
          << std::setprecision(1) << std::setw(10) << stats.min << std::setw(10) << stats.avg
          << std::setw(10) << stats.max << "   " << check_text << std::defaultfloat
          << std::setprecision(6) << std::endl;
  }



  void
  LevelBenchmark::exchange_mpi(const bool stage_on_host)
  {
    pack(values, import_indices, send_buffer);
    if (stage_on_host)
      Kokkos::deep_copy(send_buffer_host, send_buffer);
    else
      Kokkos::fence();

    Number *send = stage_on_host ? send_buffer_host.data() : send_buffer.data();
    Number *recv = stage_on_host ? recv_buffer_host.data() : ghost_values.data();

    std::vector<MPI_Request> requests;
    requests.reserve(partitioner->ghost_targets().size() + partitioner->import_targets().size());

    unsigned int offset = 0;
    for (const auto &[rank, count] : partitioner->ghost_targets())
      {
        requests.emplace_back();
        const int ierr =
          MPI_Irecv(recv + offset, count, MPI_FLOAT, rank, 201, comm, &requests.back());
        AssertThrowMPI(ierr);
        offset += count;
      }
    offset = 0;
    for (const auto &[rank, count] : partitioner->import_targets())
      {
        requests.emplace_back();
        const int ierr =
          MPI_Isend(send + offset, count, MPI_FLOAT, rank, 201, comm, &requests.back());
        AssertThrowMPI(ierr);
        offset += count;
      }
    const int ierr = MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    AssertThrowMPI(ierr);

    if (stage_on_host)
      Kokkos::deep_copy(ghost_values, recv_buffer_host);
  }



  types::global_dof_index
  LevelBenchmark::check_compress_add(const Portable::HaloExchange<Number> &halo_exchange) const
  {
    const unsigned int n_owned = partitioner->locally_owned_size();
    const unsigned int n_ghost = partitioner->n_ghost_indices();
    const unsigned int rank    = Utilities::MPI::this_mpi_process(comm);

    // values that are not exactly representable, so that the summation order
    // matters
    std::vector<Number> initial(n_owned + n_ghost);
    for (unsigned int i = 0; i < initial.size(); ++i)
      initial[i] = Number(1) / Number(3 + (i * 7 + rank * 13) % 101);

    std::array<VectorType, 2>          vectors;
    std::array<std::vector<Number>, 2> results;
    for (unsigned int v = 0; v < 2; ++v)
      {
        vectors[v].reinit(partitioner);
        Kokkos::deep_copy(
          UnmanagedDeviceView<Number>(vectors[v].get_values(), n_owned + n_ghost),
          Kokkos::View<Number *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
            initial.data(), initial.size()));
      }
    Kokkos::fence();

    vectors[0].compress(VectorOperation::add);
    halo_exchange.compress_add(vectors[1]);
    Kokkos::fence();

    for (unsigned int v = 0; v < 2; ++v)
      {
        results[v].resize(n_owned + n_ghost);
        Kokkos::deep_copy(
          Kokkos::View<Number *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
            results[v].data(), results[v].size()),
          UnmanagedDeviceView<Number>(vectors[v].get_values(), n_owned + n_ghost));
      }

    types::global_dof_index n_wrong = 0;
    for (unsigned int i = 0; i < n_owned + n_ghost; ++i)
      if (std::memcmp(&results[0][i], &results[1][i], sizeof(Number)) != 0)
        ++n_wrong;
    return Utilities::MPI::sum(n_wrong, comm);
  }



  void
  LevelBenchmark::exchange_local_only()
  {
    pack(values, import_indices, send_buffer);
    Kokkos::deep_copy(ghost_values, local_recv_buffer);
  }



  void
  LevelBenchmark::run(const bool use_nvshmem)
  {
    ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(comm) == 0);

    print_pattern_statistics();

    pcout << std::endl
          << "  " << std::left << std::setw(28) << "time per exchange [us]" << std::right
          << std::setw(10) << "min" << std::setw(10) << "avg" << std::setw(10) << "max"
          << "   check" << std::endl;

    time(
      "deal.II ghost", [&]() { vector.update_ghost_values(); }, true, false);
    time(
      "local only", [&]() { exchange_local_only(); }, false, false);
    time(
      "MPI device", [&]() { exchange_mpi(false); }, true, false);
    time(
      "MPI host", [&]() { exchange_mpi(true); }, true, false);

    const Portable::HaloExchange<Number> halo_exchange(partitioner);
    time(
      "HaloExchange ghost", [&]() { halo_exchange.update_ghost_values(vector); }, true, false);

#ifdef HALO_WITH_NVSHMEM
    if (use_nvshmem)
      {
        // the offset of the data of this rank in the ghost range of each
        // receiver and the index of this rank in the lists of the neighbors
        // are only known by the neighbors, so exchange them once with MPI
        const auto &ghost_targets  = partitioner->ghost_targets();
        const auto &import_targets = partitioner->import_targets();

        std::vector<std::array<unsigned int, 2>> to_senders(ghost_targets.size());
        std::vector<std::array<unsigned int, 2>> from_receivers(import_targets.size());
        std::vector<unsigned int>                to_receivers(import_targets.size());
        std::vector<unsigned int>                from_senders(ghost_targets.size());
        std::vector<MPI_Request>                 requests;

        unsigned int offset = 0;
        for (unsigned int i = 0; i < ghost_targets.size(); ++i)
          {
            to_senders[i] = {offset, i};
            offset += ghost_targets[i].second;
            requests.emplace_back();
            MPI_Isend(to_senders[i].data(),
                      2,
                      MPI_UNSIGNED,
                      ghost_targets[i].first,
                      301,
                      comm,
                      &requests.back());
            requests.emplace_back();
            MPI_Irecv(&from_senders[i],
                      1,
                      MPI_UNSIGNED,
                      ghost_targets[i].first,
                      302,
                      comm,
                      &requests.back());
          }
        for (unsigned int j = 0; j < import_targets.size(); ++j)
          {
            to_receivers[j] = j;
            requests.emplace_back();
            MPI_Isend(&to_receivers[j],
                      1,
                      MPI_UNSIGNED,
                      import_targets[j].first,
                      302,
                      comm,
                      &requests.back());
            requests.emplace_back();
            MPI_Irecv(from_receivers[j].data(),
                      2,
                      MPI_UNSIGNED,
                      import_targets[j].first,
                      301,
                      comm,
                      &requests.back());
          }
        const int ierr = MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
        AssertThrowMPI(ierr);

        nvshmem_halo::Pattern pattern;
        for (unsigned int j = 0; j < import_targets.size(); ++j)
          {
            pattern.send_ranks.push_back(import_targets[j].first);
            pattern.send_counts.push_back(import_targets[j].second);
            pattern.send_remote_offsets.push_back(from_receivers[j][0]);
            pattern.send_remote_slots.push_back(from_receivers[j][1]);
          }
        for (unsigned int i = 0; i < ghost_targets.size(); ++i)
          {
            pattern.recv_ranks.push_back(ghost_targets[i].first);
            pattern.recv_counts.push_back(ghost_targets[i].second);
            pattern.recv_ack_slots.push_back(from_senders[i]);
          }
        pattern.import_indices_device = import_indices.data();
        pattern.n_import_indices      = partitioner->n_import_indices();
        pattern.n_ghost_indices       = partitioner->n_ghost_indices();
        pattern.max_n_import_indices  = Utilities::MPI::max(pattern.n_import_indices, comm);
        pattern.max_n_ghost_indices   = Utilities::MPI::max(pattern.n_ghost_indices, comm);
        pattern.max_n_neighbors       = Utilities::MPI::max(
          static_cast<unsigned int>(std::max(ghost_targets.size(), import_targets.size())), comm);

        nvshmem_halo::Exchanger exchanger(pattern);

        const unsigned int n_owned = partitioner->locally_owned_size();
        void              *stream  = Kokkos::DefaultExecutionSpace().cuda_stream();

        const auto on_stream = [&]()
        { exchanger.update_ghost_values_on_stream(values.data(), n_owned, stream); };
        const auto device = [&]()
        { exchanger.update_ghost_values_device(values.data(), n_owned, stream); };
        time("NVSHMEM stream (synced)", on_stream, true, true);
        time("NVSHMEM stream (pipelined)", on_stream, true, false);
        time("NVSHMEM device (synced)", device, true, true);
        time("NVSHMEM device (pipelined)", device, true, false);
      }
#endif

    // compress() changes the owned values, so it comes last
    vector.zero_out_ghost_values();
    time(
      "deal.II compress", [&]() { vector.compress(VectorOperation::add); }, false, false);
    time(
      "HaloExchange compress", [&]() { halo_exchange.compress_add(vector); }, false, false);

    const auto n_wrong = check_compress_add(halo_exchange);
    pcout << "  HaloExchange compress vs deal.II compress: "
          << (n_wrong == 0 ? "bitwise identical" :
                             "DIFFERENT in " + std::to_string(n_wrong) + " entries")
          << std::endl;
  }
} // namespace halo



int
main(int argc, char *argv[])
{
  using namespace dealii;
  using namespace halo;

  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
      const MPI_Comm                   comm = MPI_COMM_WORLD;
      ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(comm) == 0);

      const unsigned int cycle        = argc > 1 ? std::atoi(argv[1]) : 17;
      const unsigned int degree       = argc > 2 ? std::atoi(argv[2]) : 4;
      const unsigned int n_iterations = argc > 3 ? std::atoi(argv[3]) : 200;
#ifdef HALO_WITH_NVSHMEM
      const bool use_nvshmem = argc > 4 ? std::atoi(argv[4]) == 1 : true;
#else
      const bool use_nvshmem = false;
#endif
      AssertThrow(degree > 0 && n_iterations > 0, ExcMessage("Invalid arguments."));

      // initialize Kokkos (and select the device) before NVSHMEM
      Kokkos::fence();

#ifdef HALO_WITH_NVSHMEM
      if (use_nvshmem)
        {
          std::vector<char> unique_id;
          if (Utilities::MPI::this_mpi_process(comm) == 0)
            unique_id = nvshmem_halo::get_unique_id();
          unique_id = Utilities::MPI::broadcast(comm, unique_id, 0);
          nvshmem_halo::initialize(Utilities::MPI::this_mpi_process(comm),
                                   Utilities::MPI::n_mpi_processes(comm),
                                   unique_id);
        }
#endif

      parallel::distributed::Triangulation<3> triangulation(comm);
      create_mesh(triangulation, cycle);

      pcout << "Halo exchange benchmark" << std::endl
            << "Number of MPI ranks:   " << Utilities::MPI::n_mpi_processes(comm) << std::endl
            << "Cycle:                 " << cycle << " (" << triangulation.n_global_active_cells()
            << " cells)" << std::endl
            << "Number of iterations:  " << n_iterations << std::endl
            << "NVSHMEM:               " << (use_nvshmem ? "on" : "off") << std::endl;

      // CPU cores each rank may run on and number of threads of the process
      // (MPI, UCX and NVSHMEM start their own threads); progress threads that
      // share cores with the main thread slow down all host-driven exchanges
      {
        cpu_set_t cpu_set;
        CPU_ZERO(&cpu_set);
        sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
        const auto cores = Utilities::MPI::min_max_avg(CPU_COUNT(&cpu_set), comm);

        unsigned int  n_threads = 0;
        std::ifstream status("/proc/self/status");
        for (std::string line; std::getline(status, line);)
          if (line.rfind("Threads:", 0) == 0)
            n_threads = std::stoi(line.substr(8));
        const auto threads = Utilities::MPI::min_max_avg(n_threads, comm);

        pcout << "CPU cores per rank:    " << cores.min << " - " << cores.max << std::endl
              << "Threads per rank:      " << threads.min << " - " << threads.max << std::endl;
      }
      pcout << std::endl;

      // the levels of the p-multigrid on the fine mesh, as in setup_dofs()
      // of tests/poisson_cube_bk3_dg
      std::vector<unsigned int> p_levels({degree});
      while (p_levels.back() > 1)
        p_levels.push_back(std::max(p_levels.back() - 2, 1u));

      std::vector<std::unique_ptr<FiniteElement<3>>> fes;
      for (auto p = p_levels.rbegin(); p != p_levels.rend(); ++p)
        fes.push_back(std::make_unique<FE_Q<3>>(*p));
      fes.push_back(std::make_unique<FE_DGQ<3>>(degree));

      for (const auto &fe : fes)
        {
          DoFHandler<3> dof_handler(triangulation);
          dof_handler.distribute_dofs(*fe);

          const auto partitioner =
            std::make_shared<Utilities::MPI::Partitioner>(dof_handler.locally_owned_dofs(),
                                                          DoFTools::extract_locally_relevant_dofs(
                                                            dof_handler),
                                                          comm);

          pcout << fe->get_name() << ", " << dof_handler.n_dofs() << " DoFs" << std::endl;
          LevelBenchmark benchmark(partitioner, n_iterations);
          benchmark.run(use_nvshmem);
          pcout << std::endl;
        }

#ifdef HALO_WITH_NVSHMEM
      if (use_nvshmem)
        nvshmem_halo::finalize();
#endif
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << "----------------------------------------------------" << std::endl
                << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------" << std::endl;
      return 1;
    }

  return 0;
}
