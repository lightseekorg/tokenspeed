#pragma once
/* Minimal test double, NOT a CANN header and NOT proof of binary compatibility.
 * Signatures and fields mirror the API subset. Numeric enums are test-only.
 * Real builds include the installed acl/acl_rt.h.
 */
#include <stddef.h>
#include <stdint.h>

typedef int32_t aclError;
typedef void *aclrtStream;
typedef void *aclrtContext;
typedef void *aclrtDrvMemHandle;
typedef enum { ACL_MEM_HANDLE_TYPE_NONE } aclrtMemHandleType;
typedef enum { ACL_MEM_ALLOCATION_TYPE_PINNED } aclrtMemAllocationType;
typedef enum { ACL_HBM_MEM_HUGE } aclrtMemAttr;
typedef enum { ACL_MEM_LOCATION_TYPE_HOST, ACL_MEM_LOCATION_TYPE_DEVICE } aclrtMemLocationType;
typedef struct { uint32_t id; aclrtMemLocationType type; } aclrtMemLocation;
typedef struct {
  aclrtMemHandleType handleType;
  aclrtMemAllocationType allocationType;
  aclrtMemAttr memAttr;
  aclrtMemLocation location;
  uint64_t reserve;
} aclrtPhysicalMemProp;
typedef enum {
  ACL_RT_MEM_ALLOC_GRANULARITY_MINIMUM,
  ACL_RT_MEM_ALLOC_GRANULARITY_RECOMMENDED,
  ACL_RT_MEM_ALLOC_GRANULARITY_UNDEF = 0xFFFF
} aclrtMemGranularityOptions;
typedef enum {
  ACL_MEMCPY_HOST_TO_HOST,
  ACL_MEMCPY_HOST_TO_DEVICE,
  ACL_MEMCPY_DEVICE_TO_HOST,
  ACL_MEMCPY_DEVICE_TO_DEVICE,
  ACL_MEMCPY_DEFAULT
} aclrtMemcpyKind;

#ifdef __cplusplus
extern "C" {
#endif
aclError aclrtGetDevice(int32_t *device);
aclError aclrtGetCurrentContext(aclrtContext *context);
aclError aclrtMemGetAllocationGranularity(aclrtPhysicalMemProp *prop, aclrtMemGranularityOptions option, size_t *granularity);
aclError aclrtReserveMemAddress(void **ptr, size_t size, size_t alignment, void *expected, uint64_t flags);
aclError aclrtMallocPhysical(aclrtDrvMemHandle *handle, size_t size, const aclrtPhysicalMemProp *prop, uint64_t flags);
aclError aclrtMapMem(void *ptr, size_t size, size_t offset, aclrtDrvMemHandle handle, uint64_t flags);
aclError aclrtUnmapMem(void *ptr);
aclError aclrtFreePhysical(aclrtDrvMemHandle handle);
aclError aclrtReleaseMemAddress(void *ptr);
aclError aclrtMallocHost(void **ptr, size_t size);
aclError aclrtFreeHost(void *ptr);
aclError aclrtMemcpy(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind);
aclError aclrtSynchronizeDevice(void);
#ifdef __cplusplus
}

#include <string>
#include <vector>
namespace fake_acl {
void reset(); // Requires all fake allocations already freed.
void fail_next(const std::string &api);
void clear_failures();
void set_device(int device);
void set_context(aclrtContext context);
size_t live_reservations();
size_t live_physical();
size_t live_host();
size_t calls(const std::string &api);
std::vector<std::string> trace();
size_t granularity();
}
#endif
