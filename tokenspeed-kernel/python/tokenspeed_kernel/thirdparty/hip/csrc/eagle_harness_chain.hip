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

// EAGLE-4 chained rounds, HIP build of thirdparty/cuda/csrc/eagle_harness_chain.cu.
//
// HIP has no device-side graph launch, so the kernel inside each depth's graph does the chain
// kernel's choice on the GPU and posts the chosen exec to a mailbox in host memory; a native
// launcher thread (one per device, outside the GIL) waits on the mailbox and launches that exec
// on the stream the round's depth 1 was launched on. The launch is queued behind the depth that
// posted it, so it starts once that graph has completed, as a tail launch does. A depth's
// decision runs at its start, so the next depth is queued while the posting depth still runs.
//
// A graph's first nodes can run, and post, before the call launching it has queued its last
// ones: the thread launches nothing behind a graph until that call has returned. So the round
// is handed to it (begin) once the host has queued depth 1, and it launches each later graph
// itself.

#include <hip/hip_runtime.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

#include "hip_ffi_utils.h"

namespace {

constexpr DLDataType kInt64 = DLDataType{kDLInt, 64, 1};
constexpr DLDataType kInt32 = DLDataType{kDLInt, 32, 1};
constexpr DLDataType kBool = DLDataType{kDLBool, 8, 1};

// Written by the GPU, read by the launcher thread: `ticket` (post count << 1 | finalize) is
// stored last, with release, so `exec` is visible once the ticket is.
struct Mailbox {
  uint64_t exec;
  uint64_t ticket;
};

// execs: [d] depth d's exec (0: uncaptured), [last_depth + 1 + sampled] finalize; go[d]: d drafts.
__global__ void chain_post_kernel(const int64_t* __restrict__ execs, const int32_t* __restrict__ go,
                                  const bool* __restrict__ sampled, int depth, int last_depth,
                                  Mailbox* box, uint64_t* posts) {
  const bool deeper = depth < last_depth && go[depth + 1] != 0;
  const int64_t next = execs[deeper ? depth + 1 : last_depth + 1 + (*sampled ? 1 : 0)];
  // An uncaptured graph would leave the round's tree unwritten: trap instead.
  if (next == 0) {
    __builtin_trap();
  }
  // Posts are serial: the next one comes from the graph this one launches.
  const uint64_t post = *posts + 1;
  *posts = post;
  __hip_atomic_store(&box->exec, static_cast<uint64_t>(next), __ATOMIC_RELAXED,
                     __HIP_MEMORY_SCOPE_SYSTEM);
  __hip_atomic_store(&box->ticket, (post << 1) | (deeper ? 0u : 1u), __ATOMIC_RELEASE,
                     __HIP_MEMORY_SCOPE_SYSTEM);
}

void check_hip(hipError_t status, const char* what) {
  TVM_FFI_ICHECK(status == hipSuccess)
      << "eagle_harness_chain: " << what << ": " << hipGetErrorString(status);
}

// A round without a post for this long has hung (a depth or decision never ran).
constexpr auto kPostTimeout = std::chrono::seconds(60);

// One device's mailbox and launcher thread. Created once and never freed: the detached thread
// may outlive any owner.
class Launcher {
 public:
  explicit Launcher(int device) : device_(device) {
    check_hip(hipHostMalloc(reinterpret_cast<void**>(&host_box_), sizeof(Mailbox),
                            hipHostMallocMapped | hipHostMallocCoherent | hipHostMallocPortable),
              "mailbox allocation");
    host_box_->exec = 0;
    host_box_->ticket = 0;
    check_hip(hipHostGetDevicePointer(reinterpret_cast<void**>(&device_box_), host_box_, 0),
              "mailbox device pointer");
    check_hip(hipMalloc(reinterpret_cast<void**>(&posts_), sizeof(uint64_t)), "post counter");
    check_hip(hipMemset(posts_, 0, sizeof(uint64_t)), "post counter reset");
    check_hip(hipDeviceSynchronize(), "post counter reset");
    std::thread([this] { run(); }).detach();
  }

  Mailbox* device_box() const { return device_box_; }
  uint64_t* posts() const { return posts_; }

  // Hands the thread the round whose depth 1 the caller has just queued on `stream`.
  void begin(hipStream_t stream) {
    std::unique_lock<std::mutex> lock(mutex_);
    TVM_FFI_ICHECK(!active_) << "eagle_harness_chain: a round began before the last was joined";
    wait_idle(lock);  // reports the last round's error, if not joined yet
    stream_ = stream;
    active_ = true;
    cv_.notify_all();
  }

  // Blocks until the begun round has launched its finalize (at once if none runs).
  void join() {
    std::unique_lock<std::mutex> lock(mutex_);
    wait_idle(lock);
  }

 private:
  uint64_t ticket() const {
    return __atomic_load_n(&host_box_->ticket, __ATOMIC_ACQUIRE);
  }

  void wait_idle(std::unique_lock<std::mutex>& lock) {
    cv_.wait(lock, [this] { return !active_; });
    if (!error_.empty()) {
      std::string error;
      error.swap(error_);
      TVM_FFI_THROW(RuntimeError) << "eagle_harness_chain: " << error;
    }
  }

  void finish(std::string error) {
    std::lock_guard<std::mutex> lock(mutex_);
    error_ = std::move(error);
    active_ = false;
    cv_.notify_all();
  }

  void run() {
    if (hipSetDevice(device_) != hipSuccess) {
      return;
    }
    while (true) {
      hipStream_t stream;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return active_; });
        stream = stream_;
      }
      // Every post is one round's: the round's first is the one after the last.
      uint64_t expected = posts_seen_ + 1;
      // Launch each posted graph until the round's finalize.
      std::string error;
      while (true) {
        uint64_t posted = ticket();
        if ((posted >> 1) < expected) {
          const auto start = std::chrono::steady_clock::now();
          do {
            __builtin_ia32_pause();
            posted = ticket();
          } while ((posted >> 1) < expected &&
                   std::chrono::steady_clock::now() - start < kPostTimeout);
        }
        if ((posted >> 1) != expected) {
          error = (posted >> 1) < expected ? "no graph posted within 60 s (a hung round)"
                                           : "a graph posted outside a begun round";
          posts_seen_ = posted >> 1;
          break;
        }
        const auto exec = reinterpret_cast<hipGraphExec_t>(host_box_->exec);
        const hipError_t status = hipGraphLaunch(exec, stream);
        if (status != hipSuccess) {
          error = std::string("graph launch: ") + hipGetErrorString(status);
          break;
        }
        posts_seen_ = expected;
        if (posted & 1) {
          break;
        }
        ++expected;
      }
      finish(std::move(error));
    }
  }

  const int device_;
  Mailbox* host_box_ = nullptr;
  Mailbox* device_box_ = nullptr;
  uint64_t* posts_ = nullptr;
  // The posts the thread has launched (its own).
  uint64_t posts_seen_ = 0;
  std::mutex mutex_;
  std::condition_variable cv_;
  // Guarded by mutex_.
  hipStream_t stream_ = nullptr;
  bool active_ = false;
  std::string error_;
};

std::mutex launchers_mutex;
std::unordered_map<int, Launcher*> launchers;

Launcher* launcher_for(int device, bool create) {
  std::lock_guard<std::mutex> lock(launchers_mutex);
  auto it = launchers.find(device);
  if (it != launchers.end()) {
    return it->second;
  }
  TVM_FFI_ICHECK(create) << "eagle_harness_chain: device " << device
                         << " not initialized (eagle_harness_chain_init)";
  Launcher* launcher = new Launcher(device);
  launchers.emplace(device, launcher);
  return launcher;
}

int stream_device(int64_t stream) {
  hipDevice_t device = 0;
  check_hip(hipStreamGetDevice(reinterpret_cast<hipStream_t>(stream), &device), "stream device");
  return static_cast<int>(device);
}

}  // namespace

void eagle_harness_chain_init(int64_t device) {
  launcher_for(static_cast<int>(device), true);
}

int64_t eagle_harness_chain_instantiate(int64_t graph, int64_t stream) {
  hipGraphExec_t exec = nullptr;
  check_hip(hipGraphInstantiateWithFlags(&exec, reinterpret_cast<hipGraph_t>(graph), 0),
            "instantiate");
  check_hip(hipGraphUpload(exec, reinterpret_cast<hipStream_t>(stream)), "upload");
  return reinterpret_cast<int64_t>(exec);
}

void eagle_harness_chain_destroy(int64_t exec) {
  check_hip(hipGraphExecDestroy(reinterpret_cast<hipGraphExec_t>(exec)), "destroy");
}

void eagle_harness_chain_next(TensorView execs, TensorView go, TensorView sampled, int64_t depth,
                              int64_t last_depth) {
  TVM_FFI_ICHECK(execs.dtype() == kInt64 && execs.ndim() == 1 && execs.stride(0) == 1 &&
                 execs.size(0) == last_depth + 3)
      << "eagle_harness_chain: exec table";
  TVM_FFI_ICHECK(go.dtype() == kInt32 && go.ndim() == 1 && go.stride(0) == 1 &&
                 go.size(0) > last_depth)
      << "eagle_harness_chain: go flags";
  TVM_FFI_ICHECK(sampled.dtype() == kBool && sampled.numel() >= 1)
      << "eagle_harness_chain: sampled flag";
  TVM_FFI_ICHECK(depth >= 1 && depth <= last_depth) << "eagle_harness_chain: depth";
  Launcher* launcher = launcher_for(execs.device().device_id, false);
  chain_post_kernel<<<1, 1, 0, get_stream(execs.device())>>>(
      static_cast<const int64_t*>(execs.data_ptr()), static_cast<const int32_t*>(go.data_ptr()),
      static_cast<const bool*>(sampled.data_ptr()), static_cast<int>(depth),
      static_cast<int>(last_depth), launcher->device_box(), launcher->posts());
  check_hip(hipGetLastError(), "chain kernel launch");
}

void eagle_harness_chain_begin(int64_t stream) {
  launcher_for(stream_device(stream), false)->begin(reinterpret_cast<hipStream_t>(stream));
}

void eagle_harness_chain_join(int64_t stream) {
  launcher_for(stream_device(stream), false)->join();
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(eagle_harness_chain_init, eagle_harness_chain_init);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(eagle_harness_chain_instantiate, eagle_harness_chain_instantiate);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(eagle_harness_chain_destroy, eagle_harness_chain_destroy);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(eagle_harness_chain_next, eagle_harness_chain_next);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(eagle_harness_chain_begin, eagle_harness_chain_begin);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(eagle_harness_chain_join, eagle_harness_chain_join);
