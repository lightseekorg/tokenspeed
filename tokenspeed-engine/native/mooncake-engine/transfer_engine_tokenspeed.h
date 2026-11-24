// Copyright 2026 Mooncake Authors
// SPDX-License-Identifier: Apache-2.0
#ifndef MOONCAKE_TRANSFER_ENGINE_TOKENSPEED_H
#define MOONCAKE_TRANSFER_ENGINE_TOKENSPEED_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#define TS_MOONCAKE_NOEXCEPT noexcept
extern "C" {
#else
#define TS_MOONCAKE_NOEXCEPT
#endif

typedef struct ts_mooncake_engine ts_mooncake_engine;

// ABI 1, consumed by TokenSpeed's Rust ts-mooncake crate through dlopen.
int ts_mooncake_abi_version(void) TS_MOONCAKE_NOEXCEPT;

// The caller must select an ACL device before create. Only ascend_direct is
// supported; device_name must be NULL or empty. metadata accepts the native
// TransferEngine metadata endpoint, including P2PHANDSHAKE.
// Device operations bind the calling thread to the device captured at create.
// Keep ACL initialized and that device alive until destroy returns. A failure
// to rebind during destroy is logged and retains the engine rather than tearing
// down its transport on the wrong device. MC_TRANSFER_TIMEOUT is an integer
// number of seconds (default 60, minimum 5), plus one second per GB in a batch;
// this bounds polling, not blocking native setup/submission/teardown calls.
// Returns NULL on error. Other status functions return 0 on success and a
// negative value on error, except rpc_port (positive port on success).
ts_mooncake_engine* ts_mooncake_create(
    const char* host, const char* metadata, const char* protocol,
    const char* device_name) TS_MOONCAKE_NOEXCEPT;
void ts_mooncake_destroy(ts_mooncake_engine* engine) TS_MOONCAKE_NOEXCEPT;
int ts_mooncake_rpc_port(ts_mooncake_engine* engine) TS_MOONCAKE_NOEXCEPT;
int ts_mooncake_register(ts_mooncake_engine* engine, void* address,
                         size_t length) TS_MOONCAKE_NOEXCEPT;
int ts_mooncake_unregister(ts_mooncake_engine* engine,
                           void* address) TS_MOONCAKE_NOEXCEPT;

// Writes from registered local device buffers to a peer's registered addresses.
// target is the peer's native segment name (host:rpc_port for P2PHANDSHAKE).
// Each array has count elements; zero count is a no-op, zero lengths are
// skipped. Arrays must remain valid and unchanged for the call. Nonzero spans
// must fit uint64 addresses; the total length must fit int64. Registered device
// buffers must cover every source and destination span; the adapter does not
// own them. Success means every transfer completed. Failure can follow partial
// writes. On transport failure/timeout, keep registered buffers alive until
// transport teardown; an error is not a cancellation guarantee.
int ts_mooncake_write(ts_mooncake_engine* engine, const char* target,
                      const uint64_t* sources, const uint64_t* destinations,
                      const uint64_t* lengths,
                      size_t count) TS_MOONCAKE_NOEXCEPT;

// fields is a contiguous [num_fields, 5] int64 array:
//   {source_base, source_stride_bytes, destination_base,
//    destination_stride_bytes, length_bytes}.
// For each field and page pair, copy source_base + source_page * source_stride
// to destination_base + destination_page * destination_stride.
// Validates all spans before submitting any; max_batch_size=0 submits all spans
// together. Zero pages or fields is a no-op; nonempty spans need positive
// length. Bases must be positive, indices and strides nonnegative, and span
// ends and total length must fit int64. Arrays follow the same lifetime rules
// as write.
int ts_mooncake_write_pages(ts_mooncake_engine* engine, const char* target,
                            const int64_t* source_pages,
                            const int64_t* destination_pages, size_t pages,
                            const int64_t* fields, size_t num_fields,
                            size_t max_batch_size) TS_MOONCAKE_NOEXCEPT;

// READ uses peer source addresses/pages and local destinations. On return,
// submitted reads have retired, including when the transfer failed.
int ts_mooncake_read(ts_mooncake_engine* engine, const char* target,
                      const uint64_t* sources, const uint64_t* destinations,
                      const uint64_t* lengths,
                      size_t count) TS_MOONCAKE_NOEXCEPT;

int ts_mooncake_read_pages(ts_mooncake_engine* engine, const char* target,
                            const int64_t* source_pages,
                            const int64_t* destination_pages, size_t pages,
                            const int64_t* fields, size_t num_fields,
                            size_t max_batch_size) TS_MOONCAKE_NOEXCEPT;

// Concurrent transfers are allowed. The caller must serialize registration,
// unregistration and destruction against transfers using those buffers/engine.
#ifdef __cplusplus
}
#endif
#undef TS_MOONCAKE_NOEXCEPT
#endif
