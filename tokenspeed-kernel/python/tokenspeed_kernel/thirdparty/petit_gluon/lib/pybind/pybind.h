#pragma once
#include <torch/all.h>
#include <torch/python.h>
#include <cstdint>
#include <memory>
namespace causalflow::petit::pybind {
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
