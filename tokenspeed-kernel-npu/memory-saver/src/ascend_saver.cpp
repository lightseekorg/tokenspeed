#include "ascend_saver.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <iterator>
#include <limits>
#include <list>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {
enum class Tag { Weights, KV, Graph };
enum class Policy { Backup, Discard };
enum class Phase { Active, Pausing, Paused, Resuming, Releasing };
constexpr uint64_t kHugePageReservation = 1;
constexpr int kSaverAbiVersion = 2;
thread_local char last_error[2048] = {};

const char *tag_name(Tag t) {
  switch (t) {
  case Tag::Weights: return "weights";
  case Tag::KV: return "kv_cache";
  case Tag::Graph: return "acl_graph";
  }
  return "invalid";
}
const char *phase_name(Phase p) {
  switch (p) {
  case Phase::Active: return "active";
  case Phase::Pausing: return "pausing";
  case Phase::Paused: return "paused";
  case Phase::Resuming: return "resuming";
  case Phase::Releasing: return "releasing";
  }
  return "invalid";
}
int parse_tag(const char *tag) {
  if (!tag) return -1;
  if (std::strcmp(tag, "weights") == 0) return static_cast<int>(Tag::Weights);
  if (std::strcmp(tag, "kv_cache") == 0) return static_cast<int>(Tag::KV);
  if (std::strcmp(tag, "acl_graph") == 0)
    return static_cast<int>(Tag::Graph);
  throw std::runtime_error("unknown tag; use weights, kv_cache, acl_graph, or NULL");
}
void checked(aclError status, const char *operation) {
  if (status != 0)
    throw std::runtime_error(std::string(operation) + " failed, aclError=" +
                             std::to_string(status));
}
void require_device(int expected) {
  int32_t current = -1;
  checked(aclrtGetDevice(&current), "aclrtGetDevice");
  if (current != expected)
    throw std::runtime_error("current NPU device does not match allocation device; "
                             "bind the worker device before this call");
}
std::string json_string(const char *text) {
  std::ostringstream out;
  out << '"';
  for (const unsigned char *p = reinterpret_cast<const unsigned char *>(text); *p; ++p) {
    switch (*p) {
    case '"': out << "\\\""; break;
    case '\\': out << "\\\\"; break;
    case '\n': out << "\\n"; break;
    case '\r': out << "\\r"; break;
    case '\t': out << "\\t"; break;
    default:
      if (*p < 0x20) {
        char buf[7];
        std::snprintf(buf, sizeof(buf), "\\u%04x", *p);
        out << buf;
      } else {
        out << static_cast<char>(*p);
      }
    }
  }
  out << '"';
  return out.str();
}

struct Block {
  void *va = nullptr;
  size_t requested = 0;
  size_t size = 0;
  size_t granularity = 0;
  int device = -1;
  Tag tag = Tag::Weights;
  Policy policy = Policy::Backup;
  Phase phase = Phase::Resuming;
  aclrtPhysicalMemProp prop{};
  aclrtDrvMemHandle handle{};
  bool reserved = false;
  bool has_handle = false;
  bool mapped = false;
  // Whether existing physical contents may be used without a backup restore.
  // A new DISCARD allocation becomes usable with intentionally undefined bytes.
  bool contents_valid = false;
  void *shadow = nullptr;
  bool shadow_valid = false;
};

class Backend {
public:
  std::mutex mutex;
  char latest_error[2048] = {};
  std::atomic<bool> callback_fault{false};
  char callback_error[2048] = {};
  bool context_bound = false;
  aclrtContext bound_context = nullptr;
  int32_t bound_device = -1;
  std::list<Block> blocks;
  // Control state outlives individual segments: allocation into a paused tag
  // must fail even when the tag currently has no live allocations.
  std::array<bool, 3> paused_tags{{false, false, false}};

  void record_error(const char *message) noexcept {
    std::snprintf(last_error, sizeof(last_error), "%s", message);
    std::snprintf(latest_error, sizeof(latest_error), "%s", message);
  }
  void latch_callback_fault(const char *message) noexcept {
    // Called with mutex held; publish the first error, never clear it in-process.
    if (!callback_fault.load(std::memory_order_relaxed)) {
      std::snprintf(callback_error, sizeof(callback_error), "%s", message);
      callback_fault.store(true, std::memory_order_release);
    }
  }
  void require_healthy() const {
    if (callback_fault.load(std::memory_order_acquire))
      throw std::runtime_error("allocator callback fault is latched; cleanup is allowed, but worker restart is required");
  }
  void verify_context() {
    int32_t current_device = -1;
    aclrtContext current_context = nullptr;
    checked(aclrtGetDevice(&current_device), "aclrtGetDevice for context binding");
    checked(aclrtGetCurrentContext(&current_context), "aclrtGetCurrentContext");
    if (!context_bound) {
      // A successful query returning nullptr is a valid value to pin/compare.
      bound_context = current_context;
      bound_device = current_device;
      context_bound = true;
    } else if (bound_context != current_context || bound_device != current_device) {
      throw std::runtime_error("current NPU device/context differs from backend binding; context switching is unsupported");
    }
  }

  void create_physical(Block &b) {
    checked(aclrtMallocPhysical(&b.handle, b.size, &b.prop, 0), "aclrtMallocPhysical");
    b.has_handle = true;
    b.contents_valid = false;
  }
  void map(Block &b) {
    if (!b.mapped) {
      // Local HBM mappings have read/write access on the target A2 runtime.
      checked(aclrtMapMem(b.va, b.size, 0, b.handle, 0), "aclrtMapMem");
      b.mapped = true;
    }
  }
  void copy_complete(void *dst, const void *src, size_t size, aclrtMemcpyKind kind) {
    checked(aclrtMemcpy(dst, size, src, size, kind), "aclrtMemcpy");
    // The caller must already have quiesced writers. This post-copy fence makes
    // completion an explicit backend invariant and does not replace that gate.
    checked(aclrtSynchronizeDevice(), "aclrtSynchronizeDevice after copy");
  }
  void release_block(Block &b) {
    b.phase = Phase::Releasing;
    if (b.mapped) {
      checked(aclrtUnmapMem(b.va), "aclrtUnmapMem");
      b.mapped = false;
    }
    if (b.has_handle) {
      checked(aclrtFreePhysical(b.handle), "aclrtFreePhysical");
      b.has_handle = false;
      b.handle = {};
      b.contents_valid = false;
    }
    if (b.shadow) {
      checked(aclrtFreeHost(b.shadow), "aclrtFreeHost");
      b.shadow = nullptr;
      b.shadow_valid = false;
    }
    // Keep reservation until every other resource is released, so a cleanup
    // failure cannot allow VA reuse while this record still owns that address.
    if (b.reserved) {
      checked(aclrtReleaseMemAddress(b.va), "aclrtReleaseMemAddress");
      b.reserved = false;
    }
  }

  void *allocate(size_t requested, int device, Tag tag, Policy policy) {
    require_healthy();
    if (requested == 0) return nullptr;
    if (device < 0) throw std::runtime_error("negative device id");
    require_device(device);
    verify_context();
    if (paused_tags[static_cast<size_t>(tag)])
      throw std::runtime_error("allocation rejected: requested tag is paused or recovering");
    // Allocate host-side bookkeeping before acquiring any ACL resource.
    blocks.emplace_back();
    auto it = std::prev(blocks.end());
    Block &b = *it;
    b.requested = requested;
    b.device = device;
    b.tag = tag;
    b.policy = policy;
    b.prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
    b.prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
    b.prop.memAttr = ACL_HBM_MEM_HUGE;
    b.prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
    b.prop.location.id = static_cast<uint32_t>(device);
    b.prop.reserve = 0;
    try {
      checked(aclrtMemGetAllocationGranularity(&b.prop, ACL_RT_MEM_ALLOC_GRANULARITY_MINIMUM,
                                               &b.granularity),
              "aclrtMemGetAllocationGranularity");
      if (b.granularity == 0 || requested > std::numeric_limits<size_t>::max() - (b.granularity - 1))
        throw std::runtime_error("invalid allocation granularity or rounded size overflow");
      b.size = ((requested + b.granularity - 1) / b.granularity) * b.granularity;
      // Reserve the page type selected by the physical allocation properties.
      checked(aclrtReserveMemAddress(&b.va, b.size, 0, nullptr, kHugePageReservation),
              "aclrtReserveMemAddress");
      b.reserved = true;
      create_physical(b);
      map(b);
      b.contents_valid = true; // malloc semantics: uninitialized, but usable.
      b.phase = Phase::Active;
      return b.va;
    } catch (const std::exception &allocation) {
      try {
        release_block(b);
        blocks.erase(it);
      } catch (const std::exception &cleanup) {
        char error[sizeof(callback_error)];
        std::snprintf(error, sizeof(error),
                      "%s; allocation rollback failed: %s", allocation.what(), cleanup.what());
        latch_callback_fault(error);
        throw std::runtime_error(error);
      }
      throw;
    }
  }

  void pause_block(Block &b) {
    if (b.phase == Phase::Releasing)
      throw std::runtime_error("cannot pause a block pending release; collect failed cleanup first");
    if (b.phase == Phase::Paused) return;
    b.phase = Phase::Pausing;
    if (b.policy == Policy::Backup && !b.shadow_valid) {
      if (!b.mapped || !b.contents_valid)
        throw std::runtime_error("cannot back up block: no complete readable device contents");
      if (!b.shadow) checked(aclrtMallocHost(&b.shadow, b.size), "aclrtMallocHost");
      copy_complete(b.shadow, b.va, b.size, ACL_MEMCPY_DEVICE_TO_HOST);
      b.shadow_valid = true;
    }
    if (b.mapped) {
      checked(aclrtUnmapMem(b.va), "aclrtUnmapMem");
      b.mapped = false;
    }
    if (b.has_handle) {
      checked(aclrtFreePhysical(b.handle), "aclrtFreePhysical");
      b.has_handle = false;
      b.handle = {};
      b.contents_valid = false;
    }
    b.phase = Phase::Paused;
  }

  void resume_block(Block &b) {
    if (b.phase == Phase::Releasing)
      throw std::runtime_error("cannot resume a block pending release; collect failed cleanup first");
    if (b.phase == Phase::Active) return;
    b.phase = Phase::Resuming;
    if (b.policy == Policy::Backup && !b.shadow_valid && !b.contents_valid)
      throw std::runtime_error("resume refused: no valid Host backup or surviving physical contents");
    if (!b.has_handle) create_physical(b);
    map(b);
    if (b.policy == Policy::Backup && b.shadow_valid) {
      // H2D failure leaves shadow and shadow_valid untouched for retry.
      b.contents_valid = false;
      copy_complete(b.va, b.shadow, b.size, ACL_MEMCPY_HOST_TO_DEVICE);
      b.contents_valid = true;
    } else if (b.policy == Policy::Discard) {
      b.contents_valid = true; // Caller must rebuild discarded data before use.
    }
    if (!b.contents_valid) throw std::runtime_error("resume refused: contents not restored");
    if (b.shadow) {
      // A failed free leaves the snapshot retained and phase=resuming. Retrying
      // resume may repeat H2D safely while the external launch gate stays shut.
      checked(aclrtFreeHost(b.shadow), "aclrtFreeHost");
      b.shadow = nullptr;
      b.shadow_valid = false;
    }
    b.phase = Phase::Active;
  }

  void control(const char *tag_text, bool pause) {
    require_healthy();
    const int selected = parse_tag(tag_text);
    verify_context();
    // Validate every target's context before changing any target's state.
    for (const auto &b : blocks)
      if (selected == -1 || selected == static_cast<int>(b.tag)) require_device(b.device);
    for (size_t i = 0; i < paused_tags.size(); ++i)
      if (selected == -1 || selected == static_cast<int>(i)) paused_tags[i] = true;
    for (auto &b : blocks) {
      if (selected == -1 || selected == static_cast<int>(b.tag)) {
        if (pause) pause_block(b); else resume_block(b);
      }
    }
    if (!pause)
      for (size_t i = 0; i < paused_tags.size(); ++i)
        if (selected == -1 || selected == static_cast<int>(i)) paused_tags[i] = false;
  }

  void release(void *ptr, std::optional<int> callback_device) {
    if (!ptr) return;
    auto it = std::find_if(blocks.begin(), blocks.end(),
                           [ptr](const Block &b) { return b.va == ptr; });
    if (it == blocks.end())
      throw std::runtime_error("unknown or already freed pointer");

    // PTA has dropped ownership before calling its void free callback.
    // Keep the segment eligible for cleanup even if validation fails.
    if (callback_device) it->phase = Phase::Releasing;
    verify_context();
    if (callback_device && *callback_device != it->device)
      throw std::runtime_error("free callback device does not match allocation device");
    require_device(it->device);
    it->phase = Phase::Releasing;
    // Raw allocator free may be invoked during framework pool destruction.
    // Do not rely on that caller having performed its own device fence.
    checked(aclrtSynchronizeDevice(), "aclrtSynchronizeDevice before final free");
    release_block(*it);
    blocks.erase(it);
  }
  void collect_failed() {
    verify_context();
    for (auto it = blocks.begin(); it != blocks.end();) {
      if (it->phase != Phase::Releasing) { ++it; continue; }
      require_device(it->device);
      checked(aclrtSynchronizeDevice(), "aclrtSynchronizeDevice before cleanup retry");
      release_block(*it);
      it = blocks.erase(it);
    }
  }

  std::string stats() const {
    size_t requested = 0, reserved = 0, physical = 0, mapped = 0, host = 0, backup = 0;
    size_t active = 0, paused = 0, transitional = 0, releasing = 0;
    for (const auto &b : blocks) {
      requested += b.requested;
      if (b.reserved) reserved += b.size;
      if (b.has_handle) physical += b.size;
      if (b.mapped) mapped += b.size;
      if (b.shadow) host += b.size;
      if (b.shadow_valid) backup += b.size;
      if (b.phase == Phase::Active) ++active;
      else if (b.phase == Phase::Paused) ++paused;
      else if (b.phase == Phase::Releasing) ++releasing;
      else ++transitional;
    }
    std::ostringstream out;
    out << "{\"abi_version\":" << kSaverAbiVersion << ",\"segments\":" << blocks.size()
        << ",\"context_bound\":" << (context_bound ? "true" : "false")
        << ",\"bound_device\":" << bound_device << ",\"bound_context\":\"" << bound_context << '"'
        << ",\"requested_bytes\":" << requested << ",\"reserved_bytes\":" << reserved
        << ",\"physical_bytes\":" << physical << ",\"mapped_bytes\":" << mapped
        << ",\"host_bytes\":" << host << ",\"valid_backup_bytes\":" << backup
        << ",\"active_segments\":" << active << ",\"paused_segments\":" << paused
        << ",\"transitional_segments\":" << transitional
        << ",\"releasing_segments\":" << releasing
        << ",\"callback_fault\":" << (callback_fault.load(std::memory_order_acquire) ? "true" : "false")
        << ",\"callback_error\":" << json_string(callback_error)
        << ",\"last_backend_error\":" << json_string(latest_error) << ",\"paused_tags\":[";
    bool first = true;
    for (size_t i = 0; i < paused_tags.size(); ++i) {
      if (!paused_tags[i]) continue;
      if (!first) out << ',';
      first = false;
      out << json_string(tag_name(static_cast<Tag>(i)));
    }
    out << "],\"blocks\":[";
    first = true;
    for (const auto &b : blocks) {
      if (!first) out << ',';
      first = false;
      out << "{\"ptr\":\"" << b.va << "\",\"tag\":" << json_string(tag_name(b.tag))
          << ",\"policy\":\"" << (b.policy == Policy::Backup ? "backup" : "discard")
          << "\",\"phase\":" << json_string(phase_name(b.phase))
          << ",\"device\":" << b.device << ",\"requested_bytes\":" << b.requested
          << ",\"aligned_bytes\":" << b.size << ",\"granularity\":" << b.granularity
          << ",\"reserved\":" << (b.reserved ? "true" : "false")
          << ",\"physical_live\":" << (b.has_handle ? "true" : "false")
          << ",\"mapped\":" << (b.mapped ? "true" : "false")
          << ",\"contents_valid\":" << (b.contents_valid ? "true" : "false")
          << ",\"backup_valid\":" << (b.shadow_valid ? "true" : "false") << '}';
    }
    out << "]}";
    return out.str();
  }
};

Backend &backend() {
  // Framework allocators can invoke the callbacks during process teardown.
  // Keep bookkeeping alive until exit and leave ACL cleanup to checked calls.
  static Backend *const value = new Backend;
  return *value;
}
template <class Fn> int invoke(Fn &&fn) noexcept {
  try {
    Backend &b = backend();
    std::lock_guard<std::mutex> lock(b.mutex);
    last_error[0] = '\0';
    try { fn(b); return 0; }
    catch (const std::exception &e) { b.record_error(e.what()); }
    catch (...) { b.record_error("unknown backend exception"); }
  } catch (...) {
    std::snprintf(last_error, sizeof(last_error), "%s", "backend lock/initialization failure");
  }
  return 1;
}
void *allocate_callback(size_t size, int device, Tag tag, Policy policy) noexcept {
  void *result = nullptr;
  invoke([&](Backend &b) { result = b.allocate(size, device, tag, policy); });
  return result;
}
} // namespace

extern "C" {
int saver_abi_version(void) { return kSaverAbiVersion; }
int saver_has_callback_fault(void) {
  try { return backend().callback_fault.load(std::memory_order_acquire) ? 1 : 0; }
  catch (...) { return 1; }
}
void *weights_malloc(size_t n, int d, aclrtStream) { return allocate_callback(n, d, Tag::Weights, Policy::Backup); }
void *weights_backup_malloc(size_t n, int d, aclrtStream) { return allocate_callback(n, d, Tag::Weights, Policy::Backup); }
void *weights_discard_malloc(size_t n, int d, aclrtStream) { return allocate_callback(n, d, Tag::Weights, Policy::Discard); }
void *kv_malloc(size_t n, int d, aclrtStream) { return allocate_callback(n, d, Tag::KV, Policy::Discard); }
void *graph_malloc(size_t n, int d, aclrtStream) { return allocate_callback(n, d, Tag::Graph, Policy::Backup); }
void *graph_backup_malloc(size_t n, int d, aclrtStream) { return allocate_callback(n, d, Tag::Graph, Policy::Backup); }
void *graph_discard_malloc(size_t n, int d, aclrtStream) { return allocate_callback(n, d, Tag::Graph, Policy::Discard); }
void saver_free(void *ptr, size_t, int device, aclrtStream) {
  invoke([&](Backend &b) {
    try { b.release(ptr, device); }
    catch (const std::exception &e) { b.latch_callback_fault(e.what()); throw; }
    catch (...) { b.latch_callback_fault("unknown exception in raw free callback"); throw; }
  });
}
int saver_pause(const char *tag) { return invoke([&](Backend &b) { b.control(tag, true); }); }
int saver_resume(const char *tag) { return invoke([&](Backend &b) { b.control(tag, false); }); }
int saver_release(void *ptr) { return invoke([&](Backend &b) { b.release(ptr, std::nullopt); }); }
int saver_collect_failed(void) { return invoke([&](Backend &b) { b.collect_failed(); }); }
const char *saver_last_error(void) { return last_error; }
const char *saver_stats(void) {
  static thread_local std::string result;
  try {
    Backend &b = backend();
    std::lock_guard<std::mutex> lock(b.mutex);
    result = b.stats();
    return result.c_str();
  } catch (...) {
    return "{\"abi_version\":2,\"stats_error\":\"cannot build statistics\"}";
  }
}
} // extern C
