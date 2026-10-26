#pragma once

// Ghost value exchange (the equivalent of update_ghost_values()) for the
// float vectors of LinearAlgebra::distributed::Vector with NVSHMEM. This
// header deliberately does not include any CUDA or NVSHMEM header, so that
// it can be used from the deal.II/Kokkos translation units. The streams are
// passed as void * and are interpreted as cudaStream_t.

#include <cstdint>
#include <vector>

namespace nvshmem_halo
{
  // NVSHMEM is bootstrapped with a unique id that rank 0 creates and the
  // caller broadcasts with MPI. This avoids the dependency on the NVSHMEM
  // MPI bootstrap plugin, which must match the MPI library.
  std::vector<char>
  get_unique_id();

  void
  initialize(const int rank, const int n_ranks, const std::vector<char> &unique_id);

  void
  finalize();

  // The communication pattern of one partitioner, all offsets in units of
  // vector entries.
  struct Pattern
  {
    // Ranks this rank sends to (import_targets() of the partitioner) and the
    // number of entries per rank.
    std::vector<int>          send_ranks;
    std::vector<unsigned int> send_counts;
    // Offset of the data of this rank in the ghost range of the receiver and
    // the index of this rank in the list of senders of the receiver.
    std::vector<unsigned int> send_remote_offsets;
    std::vector<unsigned int> send_remote_slots;

    // Ranks this rank receives from (ghost_targets() of the partitioner) and
    // the number of entries per rank.
    std::vector<int>          recv_ranks;
    std::vector<unsigned int> recv_counts;
    // Index of this rank in the list of receivers of the sender, used to
    // tell the sender that the receive buffer can be reused.
    std::vector<unsigned int> recv_ack_slots;

    // Local indices of the owned entries to send, in the order of
    // send_ranks, as a device pointer.
    const unsigned int *import_indices_device = nullptr;
    unsigned int        n_import_indices      = 0;
    unsigned int        n_ghost_indices       = 0;

    // Maxima over all ranks, needed for the symmetric allocations.
    unsigned int max_n_import_indices = 0;
    unsigned int max_n_ghost_indices  = 0;
    unsigned int max_n_neighbors      = 0;
  };

  // Double-buffered exchange. Every exchange writes into the receive buffer
  // of the current parity and waits for the acknowledgment of the receiver
  // from the exchange two steps back before reusing a buffer, so exchanges
  // can be issued back-to-back without host synchronization.
  class Exchanger
  {
  public:
    // Collective over all ranks (symmetric allocation).
    explicit Exchanger(const Pattern &pattern);

    // Collective over all ranks (symmetric deallocation).
    ~Exchanger();

    Exchanger(const Exchanger &) = delete;
    Exchanger &
    operator=(const Exchanger &) = delete;

    // Stream-triggered variant: the host enqueues one NVSHMEM operation per
    // neighbor and step on the stream (pack kernel, waits, puts with signal,
    // copy into the ghost range, acknowledgments).
    void
    update_ghost_values_on_stream(float *values, const unsigned int n_owned, void *stream);

    // Device-initiated variant: two kernels, one block per neighbor. The
    // first packs the data of its neighbor and puts it with a signal, the
    // second waits for the signal of its neighbor, copies the data into the
    // ghost range and sends the acknowledgment.
    void
    update_ghost_values_device(float *values, const unsigned int n_owned, void *stream);

  private:
    struct Impl;
    Impl *impl;
  };
} // namespace nvshmem_halo
