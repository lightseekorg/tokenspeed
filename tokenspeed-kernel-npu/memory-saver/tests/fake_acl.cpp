#include "fake_acl.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <vector>

namespace {
constexpr aclError failure = 700001;
constexpr size_t page = 4096; // Small fake granule to exercise query/rounding.
struct Physical { std::vector<unsigned char> bytes; bool mapped = false; };
struct Reservation { size_t size; Physical *physical = nullptr; };
std::map<void *, Reservation> reservations;
std::map<void *, Physical *> physicals;
std::map<void *, size_t> hosts;
std::map<std::string, size_t> counts;
std::map<std::string, unsigned> failures;
std::vector<std::string> operations;
int current_device = 0;
aclrtContext current_context = nullptr;
bool enter(const char *name) {
  ++counts[name];
  operations.emplace_back(name);
  auto it = failures.find(name);
  if (it != failures.end() && it->second) { --it->second; return false; }
  return true;
}
bool props_ok(const aclrtPhysicalMemProp *p) {
  return p && p->handleType == ACL_MEM_HANDLE_TYPE_NONE &&
         p->allocationType == ACL_MEM_ALLOCATION_TYPE_PINNED &&
         p->memAttr == ACL_HBM_MEM_HUGE && p->reserve == 0 &&
         p->location.type == ACL_MEM_LOCATION_TYPE_DEVICE &&
         p->location.id == static_cast<uint32_t>(current_device);
}
bool device_range(const void *ptr, size_t count) {
  auto it = reservations.find(const_cast<void *>(ptr));
  return it != reservations.end() && it->second.physical && count <= it->second.size;
}
bool host_range(const void *ptr, size_t count) {
  auto it = hosts.find(const_cast<void *>(ptr));
  return it != hosts.end() && count <= it->second;
}
} // namespace

namespace fake_acl {
void reset() {
  if (!reservations.empty() || !physicals.empty() || !hosts.empty())
    throw std::runtime_error("fake reset attempted with live resources");
  counts.clear(); failures.clear(); operations.clear(); current_device = 0; current_context = nullptr;
}
void fail_next(const std::string &api) { ++failures[api]; }
void clear_failures() { failures.clear(); }
void set_device(int device) { current_device = device; }
void set_context(aclrtContext context) { current_context = context; }
size_t live_reservations() { return reservations.size(); }
size_t live_physical() { return physicals.size(); }
size_t live_host() { return hosts.size(); }
size_t calls(const std::string &api) { return counts[api]; }
std::vector<std::string> trace() { return operations; }
size_t granularity() { return page; }
} // namespace fake_acl

extern "C" {
aclError aclrtGetDevice(int32_t *device) {
  if (!enter("aclrtGetDevice") || !device) return failure;
  *device = current_device; return 0;
}
aclError aclrtGetCurrentContext(aclrtContext *context) {
  if (!enter("aclrtGetCurrentContext") || !context) return failure;
  *context = current_context; return 0;
}
aclError aclrtMemGetAllocationGranularity(aclrtPhysicalMemProp *prop, aclrtMemGranularityOptions option, size_t *granularity) {
  if (!enter("aclrtMemGetAllocationGranularity") || !props_ok(prop) ||
      option != ACL_RT_MEM_ALLOC_GRANULARITY_MINIMUM || !granularity) return failure;
  *granularity = page; return 0;
}
aclError aclrtReserveMemAddress(void **ptr, size_t size, size_t alignment, void *expected, uint64_t flags) {
  if (!enter("aclrtReserveMemAddress") || !ptr || !size || size % page || alignment || expected || flags != 1) return failure;
  void *memory = nullptr;
  if (posix_memalign(&memory, page, size) != 0) return failure;
  reservations.emplace(memory, Reservation{size, nullptr});
  *ptr = memory; return 0;
}
aclError aclrtMallocPhysical(aclrtDrvMemHandle *handle, size_t size, const aclrtPhysicalMemProp *prop, uint64_t flags) {
  if (!enter("aclrtMallocPhysical") || !handle || !size || size % page || !props_ok(prop) || flags) return failure;
  auto *p = new Physical{std::vector<unsigned char>(size, 0xcd), false};
  physicals.emplace(p, p); *handle = p; return 0;
}
aclError aclrtMapMem(void *ptr, size_t size, size_t offset, aclrtDrvMemHandle handle, uint64_t flags) {
  if (!enter("aclrtMapMem")) return failure;
  auto r = reservations.find(ptr); auto p = physicals.find(handle);
  if (r == reservations.end() || p == physicals.end() || r->second.physical || p->second->mapped ||
      size != r->second.size || size != p->second->bytes.size() || offset || flags) return failure;
  r->second.physical = p->second; p->second->mapped = true;
  std::memcpy(ptr, p->second->bytes.data(), size); return 0;
}
aclError aclrtUnmapMem(void *ptr) {
  if (!enter("aclrtUnmapMem")) return failure;
  auto r = reservations.find(ptr);
  if (r == reservations.end() || !r->second.physical) return failure;
  auto *p = r->second.physical;
  std::memcpy(p->bytes.data(), ptr, r->second.size);
  p->mapped = false; r->second.physical = nullptr;
  std::memset(ptr, 0xdd, r->second.size); return 0;
}
aclError aclrtFreePhysical(aclrtDrvMemHandle handle) {
  if (!enter("aclrtFreePhysical")) return failure;
  auto it = physicals.find(handle);
  if (it == physicals.end() || it->second->mapped) return failure;
  delete it->second; physicals.erase(it); return 0;
}
aclError aclrtReleaseMemAddress(void *ptr) {
  if (!enter("aclrtReleaseMemAddress")) return failure;
  auto it = reservations.find(ptr);
  if (it == reservations.end() || it->second.physical) return failure;
  std::free(ptr); reservations.erase(it); return 0;
}
aclError aclrtMallocHost(void **ptr, size_t size) {
  if (!enter("aclrtMallocHost") || !ptr || !size) return failure;
  void *p = std::malloc(size); if (!p) return failure;
  hosts.emplace(p, size); *ptr = p; return 0;
}
aclError aclrtFreeHost(void *ptr) {
  if (!enter("aclrtFreeHost")) return failure;
  auto it = hosts.find(ptr); if (it == hosts.end()) return failure;
  std::free(ptr); hosts.erase(it); return 0;
}
aclError aclrtMemcpy(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind) {
  if (!enter("aclrtMemcpy") || destMax < count) return failure;
  const bool valid = kind == ACL_MEMCPY_DEVICE_TO_HOST
      ? host_range(dst, destMax) && device_range(src, count)
      : kind == ACL_MEMCPY_HOST_TO_DEVICE && device_range(dst, destMax) && host_range(src, count);
  if (!valid) return failure;
  std::memcpy(dst, src, count); return 0;
}
aclError aclrtSynchronizeDevice(void) { return enter("aclrtSynchronizeDevice") ? 0 : failure; }
} // extern C
