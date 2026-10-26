#ifndef portable_halo_exchange_h
#define portable_halo_exchange_h

#include <deal.II/base/memory_space.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/partitioner.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

DEAL_II_NAMESPACE_OPEN

namespace Portable
{
  // Ghost value exchange for LinearAlgebra::distributed::Vector in the default
  // memory space. It sends the same messages as update_ghost_values() and
  // compress(VectorOperation::add) of the vector, but packs (and unpacks) the
  // data of all neighbor ranks with a single kernel and a single fence,
  // whereas deal.II launches one kernel and fences once per neighbor rank. The
  // contributions in compress_add() are summed in the same order as in
  // deal.II, so the results are bitwise identical.
  template <typename Number>
  class HaloExchange
  {
  public:
    using VectorType = LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>;

    explicit HaloExchange(const std::shared_ptr<const Utilities::MPI::Partitioner> &partitioner);

    // Equivalent of vec.update_ghost_values().
    void
    update_ghost_values(const VectorType &vec) const;

    // Equivalent of vec.compress(VectorOperation::add).
    void
    compress_add(VectorType &vec) const;

  private:
    using IndexView  = Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space>;
    using NumberView = Kokkos::View<Number *, MemorySpace::Default::kokkos_space>;

    // Without device-aware MPI, the messages are staged through host buffers.
#ifdef DEAL_II_MPI_WITH_DEVICE_SUPPORT
    static constexpr bool stage_on_host = false;
#else
    static constexpr bool stage_on_host =
      !Kokkos::SpaceAccessibility<Kokkos::HostSpace,
                                  MemorySpace::Default::kokkos_space>::accessible;
#endif

    static constexpr int mpi_tag = 4711;

    std::shared_ptr<const Utilities::MPI::Partitioner> partitioner;

    // locally owned entries sent to other ranks in update_ghost_values(), in
    // the order of the import targets
    IndexView import_indices;

    // for compress_add(): the distinct locally owned entries that receive
    // contributions and, in CSR format, the positions of their contributions
    // in the receive buffer, in increasing order (the order of deal.II)
    IndexView compress_indices;
    IndexView compress_offsets;
    IndexView compress_positions;

    // send buffer of update_ghost_values() and receive buffer of
    // compress_add(), and their host counterparts for staging
    NumberView                                buffer;
    Kokkos::View<Number *, Kokkos::HostSpace> buffer_host;
    Kokkos::View<Number *, Kokkos::HostSpace> ghost_buffer_host;

    mutable std::vector<MPI_Request> requests;
  };



  template <typename Number>
  HaloExchange<Number>::HaloExchange(
    const std::shared_ptr<const Utilities::MPI::Partitioner> &partitioner)
    : partitioner(partitioner)
  {
    std::vector<unsigned int> import_indices_host;
    import_indices_host.reserve(partitioner->n_import_indices());
    for (const auto &[begin, end] : partitioner->import_indices())
      for (unsigned int i = begin; i < end; ++i)
        import_indices_host.push_back(i);
    AssertDimension(import_indices_host.size(), partitioner->n_import_indices());

    // an entry can be a ghost on several ranks, so sort the contributions by
    // entry, keeping the order of the ranks for each entry
    std::vector<std::pair<unsigned int, unsigned int>> entry_and_position;
    entry_and_position.reserve(import_indices_host.size());
    for (unsigned int p = 0; p < import_indices_host.size(); ++p)
      entry_and_position.emplace_back(import_indices_host[p], p);
    std::stable_sort(entry_and_position.begin(),
                     entry_and_position.end(),
                     [](const auto &a, const auto &b) { return a.first < b.first; });

    std::vector<unsigned int> compress_indices_host;
    std::vector<unsigned int> compress_offsets_host;
    std::vector<unsigned int> compress_positions_host;
    for (const auto &[entry, position] : entry_and_position)
      {
        if (compress_indices_host.empty() || compress_indices_host.back() != entry)
          {
            compress_indices_host.push_back(entry);
            compress_offsets_host.push_back(compress_positions_host.size());
          }
        compress_positions_host.push_back(position);
      }
    compress_offsets_host.push_back(compress_positions_host.size());

    const auto copy_to_device = [](const std::vector<unsigned int> &host, const std::string &name)
    {
      IndexView device(Kokkos::view_alloc(name, Kokkos::WithoutInitializing), host.size());
      Kokkos::deep_copy(device,
                        Kokkos::View<const unsigned int *,
                                     Kokkos::HostSpace,
                                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>(host.data(),
                                                                              host.size()));
      return device;
    };
    import_indices     = copy_to_device(import_indices_host, "halo_import_indices");
    compress_indices   = copy_to_device(compress_indices_host, "halo_compress_indices");
    compress_offsets   = copy_to_device(compress_offsets_host, "halo_compress_offsets");
    compress_positions = copy_to_device(compress_positions_host, "halo_compress_positions");

    buffer = NumberView(Kokkos::view_alloc("halo_buffer", Kokkos::WithoutInitializing),
                        partitioner->n_import_indices());
    if constexpr (stage_on_host)
      {
        buffer_host = Kokkos::View<Number *, Kokkos::HostSpace>("halo_buffer_host",
                                                                partitioner->n_import_indices());
        ghost_buffer_host =
          Kokkos::View<Number *, Kokkos::HostSpace>("halo_ghost_buffer_host",
                                                    partitioner->n_ghost_indices());
      }

    requests.resize(partitioner->ghost_targets().size() + partitioner->import_targets().size());
  }



  template <typename Number>
  void
  HaloExchange<Number>::update_ghost_values(const VectorType &vec) const
  {
    Assert(vec.get_partitioner()->is_compatible(*partitioner),
           ExcMessage("The vector has a different parallel layout than the HaloExchange."));

    const unsigned int n_owned = partitioner->locally_owned_size();
    const unsigned int n_ghost = partitioner->n_ghost_indices();

    // like update_ghost_values(), this writes into the ghost range of a const
    // vector
    Number *values = const_cast<Number *>(vec.get_values());

    if (!requests.empty())
      {
        const unsigned int n_import = import_indices.extent(0);
        if (n_import > 0)
          {
            const auto indices = import_indices;
            const auto send    = buffer;
            Kokkos::parallel_for(
              "Portable::HaloExchange::pack",
              Kokkos::RangePolicy<MemorySpace::Default::kokkos_space::execution_space>(0, n_import),
              KOKKOS_LAMBDA(const unsigned int i) { send(i) = values[indices(i)]; });
          }

        Number *send_ptr = buffer.data();
        Number *recv_ptr = values + n_owned;
        if constexpr (stage_on_host)
          {
            Kokkos::deep_copy(buffer_host, buffer);
            send_ptr = buffer_host.data();
            recv_ptr = ghost_buffer_host.data();
          }
        else
          Kokkos::fence();

        const MPI_Comm comm    = partitioner->get_mpi_communicator();
        unsigned int   request = 0;
        for (const auto &[rank, count] : partitioner->ghost_targets())
          {
            const int ierr = MPI_Irecv(recv_ptr,
                                       count * sizeof(Number),
                                       MPI_BYTE,
                                       rank,
                                       mpi_tag,
                                       comm,
                                       &requests[request++]);
            AssertThrowMPI(ierr);
            recv_ptr += count;
          }
        for (const auto &[rank, count] : partitioner->import_targets())
          {
            const int ierr = MPI_Isend(send_ptr,
                                       count * sizeof(Number),
                                       MPI_BYTE,
                                       rank,
                                       mpi_tag,
                                       comm,
                                       &requests[request++]);
            AssertThrowMPI(ierr);
            send_ptr += count;
          }
        const int ierr = MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
        AssertThrowMPI(ierr);

        if constexpr (stage_on_host)
          Kokkos::deep_copy(Kokkos::View<Number *,
                                         MemorySpace::Default::kokkos_space,
                                         Kokkos::MemoryTraits<Kokkos::Unmanaged>>(values + n_owned,
                                                                                  n_ghost),
                            ghost_buffer_host);
      }
    (void)n_ghost;

    vec.set_ghost_state(true);
  }



  template <typename Number>
  void
  HaloExchange<Number>::compress_add(VectorType &vec) const
  {
    Assert(vec.get_partitioner()->is_compatible(*partitioner),
           ExcMessage("The vector has a different parallel layout than the HaloExchange."));

    const unsigned int n_owned = partitioner->locally_owned_size();
    const unsigned int n_ghost = partitioner->n_ghost_indices();
    Number *const      values  = vec.get_values();

    if (!requests.empty())
      {
        // the ghost range was written by kernels
        Number *send_ptr = values + n_owned;
        Number *recv_ptr = buffer.data();
        if constexpr (stage_on_host)
          {
            Kokkos::deep_copy(
              ghost_buffer_host,
              Kokkos::View<const Number *,
                           MemorySpace::Default::kokkos_space,
                           Kokkos::MemoryTraits<Kokkos::Unmanaged>>(values + n_owned, n_ghost));
            send_ptr = ghost_buffer_host.data();
            recv_ptr = buffer_host.data();
          }
        else
          Kokkos::fence();

        const MPI_Comm comm    = partitioner->get_mpi_communicator();
        unsigned int   request = 0;
        for (const auto &[rank, count] : partitioner->import_targets())
          {
            const int ierr = MPI_Irecv(recv_ptr,
                                       count * sizeof(Number),
                                       MPI_BYTE,
                                       rank,
                                       mpi_tag,
                                       comm,
                                       &requests[request++]);
            AssertThrowMPI(ierr);
            recv_ptr += count;
          }
        for (const auto &[rank, count] : partitioner->ghost_targets())
          {
            const int ierr = MPI_Isend(send_ptr,
                                       count * sizeof(Number),
                                       MPI_BYTE,
                                       rank,
                                       mpi_tag,
                                       comm,
                                       &requests[request++]);
            AssertThrowMPI(ierr);
            send_ptr += count;
          }
        const int ierr = MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
        AssertThrowMPI(ierr);

        if constexpr (stage_on_host)
          Kokkos::deep_copy(buffer, buffer_host);

        // add the contributions to the owned entries and zero the ghost range
        // in one kernel
        const unsigned int n_entries = compress_indices.extent(0);
        if (n_entries + n_ghost > 0)
          {
            const auto indices   = compress_indices;
            const auto offsets   = compress_offsets;
            const auto positions = compress_positions;
            const auto recv      = buffer;
            Kokkos::parallel_for(
              "Portable::HaloExchange::unpack_add",
              Kokkos::RangePolicy<MemorySpace::Default::kokkos_space::execution_space>(0,
                                                                                       n_entries +
                                                                                         n_ghost),
              KOKKOS_LAMBDA(const unsigned int i) {
                if (i < n_entries)
                  {
                    Number value = values[indices(i)];
                    for (unsigned int p = offsets(i); p < offsets(i + 1); ++p)
                      value += recv(positions(p));
                    values[indices(i)] = value;
                  }
                else
                  values[n_owned + i - n_entries] = Number();
              });
          }
      }

    vec.set_ghost_state(false);
  }
} // namespace Portable

DEAL_II_NAMESPACE_CLOSE

#endif
