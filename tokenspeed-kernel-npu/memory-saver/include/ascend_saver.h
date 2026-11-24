#pragma once

#include <stddef.h>
#include <stdint.h>
#ifdef SAVER_FAKE_ACL
#include "fake_acl.h"
#else
#include <acl/acl_rt.h>
#endif

#if defined(_WIN32)
#define SAVER_API __declspec(dllexport)
#else
#define SAVER_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ABI v3. Bind the worker's NPU device before calling any mutating API.
 * The caller must stop new launches/allocations and wait for all prior users
 * before pause/resume. Final free synchronizes the current device before
 * unmapping; the owner must still prevent new uses of the released storage.
 * Stream arguments identify the torch_npu ABI; this backend uses whole-device
 * fences rather than inserting stream-order dependencies.
 */
SAVER_API int saver_abi_version(void);
/* Sticky first callback fault from a void free or failed allocation rollback.
 * A return value of 1 means restart the worker before reuse.
 * Allocation/pause/resume fail closed. Explicit release/cleanup remain allowed.
 */
SAVER_API int saver_has_callback_fault(void);
/* Register VMM before any managed allocation. The supplied local HCCL
 * communicator and functions must stay alive until saver_close_hccl succeeds.
 * configure is collective across the communicator once; allocation callbacks
 * only use the already-established memory domain afterwards.
 */
typedef int (*saver_hccl_set_range_fn)(void *, void *, size_t, size_t, uint64_t);
typedef int (*saver_hccl_unset_range_fn)(void *, void *);
typedef int (*saver_hccl_activate_fn)(void *, void *, size_t, size_t, aclrtDrvMemHandle, uint64_t);
typedef int (*saver_hccl_deactivate_fn)(void *, void *);
SAVER_API int saver_configure_hccl(void *comm, saver_hccl_set_range_fn set_range,
    saver_hccl_unset_range_fn unset_range, saver_hccl_activate_fn activate,
    saver_hccl_deactivate_fn deactivate);
/* Drain users first. Must finish before the communicator is destroyed. */
SAVER_API int saver_close_hccl(void);

SAVER_API void *weights_malloc(size_t size, int device, aclrtStream stream);
SAVER_API void *weights_backup_malloc(size_t size, int device, aclrtStream stream);
SAVER_API void *weights_discard_malloc(size_t size, int device, aclrtStream stream);
SAVER_API void *kv_malloc(size_t size, int device, aclrtStream stream);
SAVER_API void *graph_malloc(size_t size, int device, aclrtStream stream);
SAVER_API void *graph_backup_malloc(size_t size, int device, aclrtStream stream);
SAVER_API void *graph_discard_malloc(size_t size, int device, aclrtStream stream);
/* Retires a known allocation even if context or device validation fails. */
SAVER_API void saver_free(void *ptr, size_t size, int device, aclrtStream stream);

/* tag: weights, kv_cache, acl_graph, or NULL for all.
 * Return 0 on success, 1 on failure. A failed batch can be partially applied;
 * keep the external launch gate closed and retry pause/resume/release.
 */
SAVER_API int saver_pause(const char *tag);
SAVER_API int saver_resume(const char *tag);
/* A context or device validation failure leaves the segment phase unchanged. */
SAVER_API int saver_release(void *ptr);
/* Retry cleanup only for blocks already marked releasing by a failed alloc/free. */
SAVER_API int saver_collect_failed(void);

/* Thread-local strings, valid until the next same-thread call to the same
 * function (stats), or the next mutating operation (last_error). Copy promptly.
 * stats also contains the latest backend error, including callback errors
 * produced on another thread. No ACL calls are made by these read-only APIs.
 */
SAVER_API const char *saver_last_error(void);
SAVER_API const char *saver_stats(void);

#ifdef __cplusplus
}
#endif
