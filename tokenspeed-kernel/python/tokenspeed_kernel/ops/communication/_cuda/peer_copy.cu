// Copyright (c) 2026 LightSeek Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Copy-engine peer fetch. A worker thread per store submits each layer's
// copies as one cudaMemcpyBatchAsync; the call blocks once the copy stream's
// queue is full, so it runs off the forward thread. Fetch and reader order is
// kept on the GPU with 64-bit sequence flags (stream wait/write value). A flag
// wait is enqueued only after its write: a wait ahead of its producer blocks
// the GPU, and lazy module loading on any thread then deadlocks.
#include "tvm_ffi_utils.h"
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using StreamValue64 = CUresult (*)(CUstream, CUdeviceptr, cuuint64_t, unsigned int);

struct DriverOps {
  StreamValue64 wait;
  StreamValue64 write;
};

StreamValue64 entry_point(const char *symbol) {
  void *function = nullptr;
  cudaDriverEntryPointQueryResult status;
  TVM_FFI_ICHECK(cudaGetDriverEntryPointByVersion(symbol, &function, 12000, cudaEnableDefault,
                                                  &status) == cudaSuccess &&
                 status == cudaDriverEntryPointSuccess)
      << "missing driver entry point " << symbol;
  return reinterpret_cast<StreamValue64>(function);
}

const DriverOps &driver() {
  static const DriverOps ops{entry_point("cuStreamWaitValue64"),
                             entry_point("cuStreamWriteValue64")};
  return ops;
}

[[noreturn]] void fail(const char *what, int code) {
  std::fprintf(stderr, "peer copy worker: %s failed with error %d\n", what, code);
  std::abort();
}

struct Layer {
  std::vector<void *> dst;
  std::vector<void *> src;
  std::vector<size_t> size;
};

struct Job {
  const Layer *layer;
  CUdeviceptr wait_flag;
  uint64_t wait_value;
  CUdeviceptr ready_flag;
  uint64_t ready_value;
};

class Worker {
public:
  Worker(int device, cudaStream_t stream)
      : device_(device), stream_(stream), thread_([this] { run(); }) {}

  ~Worker() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    cv_.notify_one();
    thread_.join();
  }

  void set_layer(int64_t index, Layer layer) {
    std::lock_guard<std::mutex> lock(mutex_);
    layers_[index] = std::make_unique<Layer>(std::move(layer));
  }

  int64_t push(int64_t index, CUdeviceptr wait_flag, uint64_t wait_value, CUdeviceptr ready_flag,
               uint64_t ready_value) {
    int64_t job;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto layer = layers_.find(index);
      TVM_FFI_ICHECK(layer != layers_.end()) << "no copy schedule for layer " << index;
      jobs_.push_back(Job{layer->second.get(), wait_flag, wait_value, ready_flag, ready_value});
      job = ++pushed_;
    }
    cv_.notify_one();
    return job;
  }

  // Blocks until job `job` (numbered from 1 by push) has its ready write enqueued.
  void wait_submitted(int64_t job) {
    std::unique_lock<std::mutex> lock(mutex_);
    TVM_FFI_ICHECK(job <= pushed_) << "peer copy job " << job << " was never queued";
    submitted_cv_.wait(lock, [&] { return submitted_ >= job; });
  }

private:
  void run() {
    cudaError_t error = cudaSetDevice(device_);
    if (error != cudaSuccess)
      fail("cudaSetDevice", error);
    cudaMemcpyAttributes attr = {};
    attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    size_t attr_index = 0;
    auto stream = reinterpret_cast<CUstream>(stream_);
    for (;;) {
      Job job;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return stop_ || !jobs_.empty(); });
        if (jobs_.empty())
          return;
        job = jobs_.front();
        jobs_.pop_front();
      }
      CUresult result =
          driver().wait(stream, job.wait_flag, job.wait_value, CU_STREAM_WAIT_VALUE_GEQ);
      if (result != CUDA_SUCCESS)
        fail("cuStreamWaitValue64", result);
      const Layer &layer = *job.layer;
      if (!layer.size.empty()) {
        error = cudaMemcpyBatchAsync(const_cast<void **>(layer.dst.data()),
                                     const_cast<void **>(layer.src.data()),
                                     const_cast<size_t *>(layer.size.data()), layer.size.size(),
                                     &attr, &attr_index, 1, stream_);
        if (error != cudaSuccess)
          fail("cudaMemcpyBatchAsync", error);
      }
      result = driver().write(stream, job.ready_flag, job.ready_value, CU_STREAM_WRITE_VALUE_DEFAULT);
      if (result != CUDA_SUCCESS)
        fail("cuStreamWriteValue64", result);
      {
        std::lock_guard<std::mutex> lock(mutex_);
        ++submitted_;
      }
      submitted_cv_.notify_all();
    }
  }

  int device_;
  cudaStream_t stream_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::condition_variable submitted_cv_;
  std::deque<Job> jobs_;
  int64_t pushed_ = 0;
  int64_t submitted_ = 0;
  std::unordered_map<int64_t, std::unique_ptr<Layer>> layers_;
  bool stop_ = false;
  std::thread thread_;
};

std::mutex registry_mutex;
std::unordered_map<int64_t, std::unique_ptr<Worker>> registry;
int64_t next_handle = 1;

Worker &worker(int64_t handle) {
  std::lock_guard<std::mutex> lock(registry_mutex);
  auto found = registry.find(handle);
  TVM_FFI_ICHECK(found != registry.end()) << "unknown peer copy worker " << handle;
  return *found->second;
}

CUdeviceptr flag_address(TensorView flags, int64_t index) {
  TVM_FFI_ICHECK(flags.dtype().code == kDLInt && flags.dtype().bits == 64);
  TVM_FFI_ICHECK(index >= 0 && index < flags.numel());
  return reinterpret_cast<CUdeviceptr>(static_cast<int64_t *>(flags.data_ptr()) + index);
}

void check_int64(TensorView t) {
  TVM_FFI_ICHECK(t.dtype().code == kDLInt && t.dtype().bits == 64);
  TVM_FFI_ICHECK(t.device().device_type == kDLCPU);
}

} // namespace

int64_t create_worker(int64_t device, int64_t stream) {
  driver();
  std::lock_guard<std::mutex> lock(registry_mutex);
  int64_t handle = next_handle++;
  registry[handle] =
      std::make_unique<Worker>(static_cast<int>(device), reinterpret_cast<cudaStream_t>(stream));
  return handle;
}

void destroy_worker(int64_t handle) {
  std::unique_ptr<Worker> owned;
  {
    std::lock_guard<std::mutex> lock(registry_mutex);
    auto found = registry.find(handle);
    TVM_FFI_ICHECK(found != registry.end()) << "unknown peer copy worker " << handle;
    owned = std::move(found->second);
    registry.erase(found);
  }
}

void set_layer(int64_t handle, int64_t layer, TensorView dst, TensorView src, TensorView size) {
  check_int64(dst);
  check_int64(src);
  check_int64(size);
  const int64_t count = dst.numel();
  TVM_FFI_ICHECK(src.numel() == count && size.numel() == count);
  Layer value;
  auto *d = static_cast<const int64_t *>(dst.data_ptr());
  auto *s = static_cast<const int64_t *>(src.data_ptr());
  auto *n = static_cast<const int64_t *>(size.data_ptr());
  value.dst.reserve(count);
  value.src.reserve(count);
  value.size.reserve(count);
  for (int64_t i = 0; i < count; ++i) {
    TVM_FFI_ICHECK(n[i] > 0);
    value.dst.push_back(reinterpret_cast<void *>(d[i]));
    value.src.push_back(reinterpret_cast<void *>(s[i]));
    value.size.push_back(static_cast<size_t>(n[i]));
  }
  worker(handle).set_layer(layer, std::move(value));
}

int64_t fetch(int64_t handle, int64_t layer, TensorView flags, int64_t wait_index,
              int64_t wait_value, int64_t ready_index, int64_t ready_value) {
  return worker(handle).push(layer, flag_address(flags, wait_index),
                             static_cast<uint64_t>(wait_value), flag_address(flags, ready_index),
                             static_cast<uint64_t>(ready_value));
}

void wait_submitted(int64_t handle, int64_t job) { worker(handle).wait_submitted(job); }

void wait_flag(TensorView flags, int64_t index, int64_t value, int64_t stream) {
  CUresult result = driver().wait(reinterpret_cast<CUstream>(stream), flag_address(flags, index),
                                  static_cast<uint64_t>(value), CU_STREAM_WAIT_VALUE_GEQ);
  TVM_FFI_ICHECK(result == CUDA_SUCCESS) << "cuStreamWaitValue64 failed with error " << result;
}

void write_flag(TensorView flags, int64_t index, int64_t value, int64_t stream) {
  CUresult result = driver().write(reinterpret_cast<CUstream>(stream), flag_address(flags, index),
                                   static_cast<uint64_t>(value), CU_STREAM_WRITE_VALUE_DEFAULT);
  TVM_FFI_ICHECK(result == CUDA_SUCCESS) << "cuStreamWriteValue64 failed with error " << result;
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(create_worker, create_worker);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(destroy_worker, destroy_worker);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(set_layer, set_layer);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fetch, fetch);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(wait_submitted, wait_submitted);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(wait_flag, wait_flag);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(write_flag, write_flag);
