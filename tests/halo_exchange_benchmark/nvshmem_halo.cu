#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>

#include "nvshmem_halo.h"

#define CHECK_CUDA(call)                                                                      \
  do                                                                                          \
    {                                                                                         \
      const cudaError_t err = (call);                                                         \
      if (err != cudaSuccess)                                                                 \
        {                                                                                     \
          std::fprintf(                                                                       \
            stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
          std::abort();                                                                       \
        }                                                                                     \
    }                                                                                         \
  while (false)

namespace nvshmem_halo
{
  namespace
  {
    // One communication partner. For a send neighbor, local_offset is the
    // offset into the list of import indices (and the send buffer) and slot
    // is the index of this rank among the senders of the receiver. For a
    // receive neighbor, local_offset is the offset into the ghost range (and
    // the receive buffer) and slot is the index of this rank among the
    // receivers of the sender.
    struct Neighbor
    {
      int          rank;
      unsigned int count;
      unsigned int local_offset;
      unsigned int remote_offset;
      unsigned int slot;
    };

    constexpr unsigned int threads_per_block = 256;

    __global__ void
    pack(const float        *values,
         const unsigned int *import_indices,
         const unsigned int  n_import_indices,
         float              *send_buffer)
    {
      const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < n_import_indices)
        send_buffer[i] = values[import_indices[i]];
    }

    // One block per send neighbor: wait until the receiver has consumed the
    // buffer of this parity, pack, put with signal.
    __global__ void
    pack_and_put(const Neighbor     *send_neighbors,
                 const float        *values,
                 const unsigned int *import_indices,
                 float              *send_buffer,
                 float              *recv_buffer,
                 uint64_t           *recv_signals,
                 uint64_t           *ack_signals,
                 const uint64_t      step)
    {
      const Neighbor neighbor = send_neighbors[blockIdx.x];

      if (step >= 2 && threadIdx.x == 0)
        nvshmem_signal_wait_until(ack_signals + blockIdx.x, NVSHMEM_CMP_GE, step - 1);
      __syncthreads();

      for (unsigned int i = threadIdx.x; i < neighbor.count; i += blockDim.x)
        send_buffer[neighbor.local_offset + i] = values[import_indices[neighbor.local_offset + i]];
      __syncthreads();

      nvshmemx_putmem_signal_nbi_block(recv_buffer + neighbor.remote_offset,
                                       send_buffer + neighbor.local_offset,
                                       neighbor.count * sizeof(float),
                                       recv_signals + neighbor.slot,
                                       step + 1,
                                       NVSHMEM_SIGNAL_SET,
                                       neighbor.rank);
    }

    // One block per receive neighbor: wait for the data, copy it into the
    // ghost range, acknowledge.
    __global__ void
    wait_and_unpack(const Neighbor *recv_neighbors,
                    float          *ghost_values,
                    const float    *recv_buffer,
                    uint64_t       *recv_signals,
                    uint64_t       *ack_signals,
                    const uint64_t  step)
    {
      const Neighbor neighbor = recv_neighbors[blockIdx.x];

      if (threadIdx.x == 0)
        nvshmem_signal_wait_until(recv_signals + blockIdx.x, NVSHMEM_CMP_GE, step + 1);
      __syncthreads();

      // the data was written by another device or the network, so bypass
      // the L1 cache
      for (unsigned int i = threadIdx.x; i < neighbor.count; i += blockDim.x)
        ghost_values[neighbor.local_offset + i] = __ldcv(recv_buffer + neighbor.local_offset + i);
      __syncthreads();

      if (threadIdx.x == 0)
        nvshmemx_signal_op(ack_signals + neighbor.slot,
                           step + 1,
                           NVSHMEM_SIGNAL_SET,
                           neighbor.rank);
    }
  } // namespace



  std::vector<char>
  get_unique_id()
  {
    nvshmemx_uniqueid_t id = NVSHMEMX_UNIQUEID_INITIALIZER;
    if (nvshmemx_get_uniqueid(&id) != 0)
      {
        std::fprintf(stderr, "nvshmemx_get_uniqueid failed\n");
        std::abort();
      }
    std::vector<char> bytes(sizeof(id));
    std::memcpy(bytes.data(), &id, sizeof(id));
    return bytes;
  }



  void
  initialize(const int rank, const int n_ranks, const std::vector<char> &unique_id)
  {
    nvshmemx_uniqueid_t id = NVSHMEMX_UNIQUEID_INITIALIZER;
    if (unique_id.size() != sizeof(id))
      {
        std::fprintf(stderr, "Wrong size of the NVSHMEM unique id\n");
        std::abort();
      }
    std::memcpy(&id, unique_id.data(), sizeof(id));

    nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
    nvshmemx_set_attr_uniqueid_args(rank, n_ranks, &id, &attr);
    if (nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr) != 0)
      {
        std::fprintf(stderr, "nvshmemx_init_attr failed\n");
        std::abort();
      }
    if (nvshmem_my_pe() != rank)
      {
        std::fprintf(stderr, "NVSHMEM PE %d does not match MPI rank %d\n", nvshmem_my_pe(), rank);
        std::abort();
      }
  }



  void
  finalize()
  {
    nvshmem_finalize();
  }



  struct Exchanger::Impl
  {
    std::vector<Neighbor> send_neighbors;
    std::vector<Neighbor> recv_neighbors;
    Neighbor             *send_neighbors_device = nullptr;
    Neighbor             *recv_neighbors_device = nullptr;

    const unsigned int *import_indices   = nullptr;
    unsigned int        n_import_indices = 0;
    unsigned int        n_ghost_indices  = 0;
    unsigned int        max_n_import     = 0;
    unsigned int        max_n_ghost      = 0;

    // symmetric memory: two send and two receive buffers (one per parity),
    // the signals of the received data and of the acknowledgments, and a
    // dummy word used as payload of the acknowledgments on the stream
    float    *send_buffer  = nullptr;
    float    *recv_buffer  = nullptr;
    uint64_t *recv_signals = nullptr;
    uint64_t *ack_signals  = nullptr;
    uint64_t *dummy        = nullptr;

    uint64_t step = 0;
  };



  Exchanger::Exchanger(const Pattern &pattern)
    : impl(new Impl)
  {
    impl->import_indices   = pattern.import_indices_device;
    impl->n_import_indices = pattern.n_import_indices;
    impl->n_ghost_indices  = pattern.n_ghost_indices;
    impl->max_n_import     = pattern.max_n_import_indices;
    impl->max_n_ghost      = pattern.max_n_ghost_indices;

    unsigned int offset = 0;
    for (unsigned int i = 0; i < pattern.send_ranks.size(); ++i)
      {
        impl->send_neighbors.push_back({pattern.send_ranks[i],
                                        pattern.send_counts[i],
                                        offset,
                                        pattern.send_remote_offsets[i],
                                        pattern.send_remote_slots[i]});
        offset += pattern.send_counts[i];
      }
    offset = 0;
    for (unsigned int i = 0; i < pattern.recv_ranks.size(); ++i)
      {
        impl->recv_neighbors.push_back(
          {pattern.recv_ranks[i], pattern.recv_counts[i], offset, 0, pattern.recv_ack_slots[i]});
        offset += pattern.recv_counts[i];
      }

    for (auto [neighbors, neighbors_device] :
         {std::make_pair(&impl->send_neighbors, &impl->send_neighbors_device),
          std::make_pair(&impl->recv_neighbors, &impl->recv_neighbors_device)})
      if (!neighbors->empty())
        {
          CHECK_CUDA(cudaMalloc(neighbors_device, neighbors->size() * sizeof(Neighbor)));
          CHECK_CUDA(cudaMemcpy(*neighbors_device,
                                neighbors->data(),
                                neighbors->size() * sizeof(Neighbor),
                                cudaMemcpyHostToDevice));
        }

    // nvshmem_malloc is collective and needs the same size on all ranks
    const std::size_t n_signals = std::max(pattern.max_n_neighbors, 1u);
    impl->send_buffer =
      static_cast<float *>(nvshmem_malloc(2 * std::max(impl->max_n_import, 1u) * sizeof(float)));
    impl->recv_buffer =
      static_cast<float *>(nvshmem_malloc(2 * std::max(impl->max_n_ghost, 1u) * sizeof(float)));
    impl->recv_signals = static_cast<uint64_t *>(nvshmem_malloc(n_signals * sizeof(uint64_t)));
    impl->ack_signals  = static_cast<uint64_t *>(nvshmem_malloc(n_signals * sizeof(uint64_t)));
    impl->dummy        = static_cast<uint64_t *>(nvshmem_malloc(2 * sizeof(uint64_t)));
    if (impl->send_buffer == nullptr || impl->recv_buffer == nullptr ||
        impl->recv_signals == nullptr || impl->ack_signals == nullptr || impl->dummy == nullptr)
      {
        std::fprintf(stderr, "nvshmem_malloc failed, increase NVSHMEM_SYMMETRIC_SIZE\n");
        std::abort();
      }

    CHECK_CUDA(cudaMemset(impl->recv_signals, 0, n_signals * sizeof(uint64_t)));
    CHECK_CUDA(cudaMemset(impl->ack_signals, 0, n_signals * sizeof(uint64_t)));
    CHECK_CUDA(cudaMemset(impl->dummy, 0, 2 * sizeof(uint64_t)));
    CHECK_CUDA(cudaDeviceSynchronize());
    nvshmem_barrier_all();
  }



  Exchanger::~Exchanger()
  {
    CHECK_CUDA(cudaDeviceSynchronize());
    nvshmem_barrier_all();
    nvshmem_free(impl->dummy);
    nvshmem_free(impl->ack_signals);
    nvshmem_free(impl->recv_signals);
    nvshmem_free(impl->recv_buffer);
    nvshmem_free(impl->send_buffer);
    cudaFree(impl->send_neighbors_device);
    cudaFree(impl->recv_neighbors_device);
    delete impl;
  }



  void
  Exchanger::update_ghost_values_on_stream(float             *values,
                                           const unsigned int n_owned,
                                           void              *stream_ptr)
  {
    const cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    const uint64_t     step   = impl->step++;
    float *const       send   = impl->send_buffer + (step % 2) * std::max(impl->max_n_import, 1u);
    float *const       recv   = impl->recv_buffer + (step % 2) * std::max(impl->max_n_ghost, 1u);

    // the receivers must have consumed the data sent two steps ago before
    // the buffer of this parity is overwritten
    if (step >= 2)
      for (unsigned int t = 0; t < impl->send_neighbors.size(); ++t)
        nvshmemx_signal_wait_until_on_stream(impl->ack_signals + t,
                                             NVSHMEM_CMP_GE,
                                             step - 1,
                                             stream);

    if (impl->n_import_indices > 0)
      pack<<<(impl->n_import_indices + threads_per_block - 1) / threads_per_block,
             threads_per_block,
             0,
             stream>>>(values, impl->import_indices, impl->n_import_indices, send);

    for (const Neighbor &neighbor : impl->send_neighbors)
      nvshmemx_putmem_signal_on_stream(recv + neighbor.remote_offset,
                                       send + neighbor.local_offset,
                                       neighbor.count * sizeof(float),
                                       impl->recv_signals + neighbor.slot,
                                       step + 1,
                                       NVSHMEM_SIGNAL_SET,
                                       neighbor.rank,
                                       stream);

    for (unsigned int r = 0; r < impl->recv_neighbors.size(); ++r)
      nvshmemx_signal_wait_until_on_stream(impl->recv_signals + r,
                                           NVSHMEM_CMP_GE,
                                           step + 1,
                                           stream);

    if (impl->n_ghost_indices > 0)
      CHECK_CUDA(cudaMemcpyAsync(values + n_owned,
                                 recv,
                                 impl->n_ghost_indices * sizeof(float),
                                 cudaMemcpyDeviceToDevice,
                                 stream));

    for (const Neighbor &neighbor : impl->recv_neighbors)
      nvshmemx_putmem_signal_on_stream(impl->dummy,
                                       impl->dummy + 1,
                                       sizeof(uint64_t),
                                       impl->ack_signals + neighbor.slot,
                                       step + 1,
                                       NVSHMEM_SIGNAL_SET,
                                       neighbor.rank,
                                       stream);
  }



  void
  Exchanger::update_ghost_values_device(float *values, const unsigned int n_owned, void *stream_ptr)
  {
    const cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    const uint64_t     step   = impl->step++;
    float *const       send   = impl->send_buffer + (step % 2) * std::max(impl->max_n_import, 1u);
    float *const       recv   = impl->recv_buffer + (step % 2) * std::max(impl->max_n_ghost, 1u);

    if (!impl->send_neighbors.empty())
      pack_and_put<<<impl->send_neighbors.size(), threads_per_block, 0, stream>>>(
        impl->send_neighbors_device,
        values,
        impl->import_indices,
        send,
        recv,
        impl->recv_signals,
        impl->ack_signals,
        step);

    if (!impl->recv_neighbors.empty())
      wait_and_unpack<<<impl->recv_neighbors.size(), threads_per_block, 0, stream>>>(
        impl->recv_neighbors_device,
        values + n_owned,
        recv,
        impl->recv_signals,
        impl->ack_signals,
        step);

    CHECK_CUDA(cudaGetLastError());
  }
} // namespace nvshmem_halo
