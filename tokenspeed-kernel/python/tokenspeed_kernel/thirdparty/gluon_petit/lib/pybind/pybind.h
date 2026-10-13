#pragma once
#include <torch/all.h>
#include <torch/python.h>
#include <cstdint>
#include <memory>
namespace causalflow::petit::pybind {
// Each rank reserves one virtual region with the layout:
// [per-rank barriers][per-rank token slots][optional padding][private workspace].
// Each rank owns its physical barrier and token-slot allocations and shares HIP
// handles over local sockets so every rank can map them in the same rank order.
// Symmetry means matching offsets from each rank's base, not identical virtual
// base addresses across processes. Padding and workspace are private to each rank.
// LocalTensor() borrows the entire local virtual mapping, including peer regions;
// it and any derived tensor views must not outlive the heap.
class VmmSymmetricHeap {
  public:
    struct Layout {
        std::uint32_t barrier_record_bytes;
        std::uint32_t rank_sym_buffer_base;
        std::uint32_t rank_slot_bytes;
        std::uint32_t local_offset;
        std::uint32_t local_bytes;
    };

    explicit VmmSymmetricHeap(int world_size);
    VmmSymmetricHeap(const VmmSymmetricHeap &) = delete;
    VmmSymmetricHeap &operator=(const VmmSymmetricHeap &) = delete;
    ~VmmSymmetricHeap();
    bool Allocate(const Layout &layout);
    torch::Tensor LocalTensor() const;
    int world_size() const;
    int rank() const;
    int device_index() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    void Cleanup();
};

}
