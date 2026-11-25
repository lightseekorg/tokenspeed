// Copyright 2026 Mooncake Authors
// SPDX-License-Identifier: Apache-2.0
#include "transfer_engine_tokenspeed.h"
#include "transfer_engine.h"
#include <acl/acl.h>
#include <algorithm>
#include <chrono>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <string>
#include <vector>

using mooncake::INVALID_BATCH_ID;
using mooncake::SegmentHandle;
using mooncake::TransferEngine;
using mooncake::TransferRequest;
using mooncake::TransferStatus;
using mooncake::TransferStatusEnum;

namespace {
// freeBatchID may transfer ownership to deferred cleanup. Never retry freeing
// that handle, and never interpret releasing it as cancellation of the writes.
struct BatchGuard {
    TransferEngine& engine;
    mooncake::BatchID id;
    ~BatchGuard() {
        try {
            auto status = engine.freeBatchID(id);
            if (!status.ok() && !status.IsBatchCleanupDeferred())
                std::fprintf(stderr, "Mooncake batch cleanup: %s\n",
                             status.ToString().c_str());
        } catch (...) {
            std::fprintf(stderr, "Mooncake batch cleanup threw an exception\n");
        }
    }
};
}  // namespace

struct ts_mooncake_engine {
    TransferEngine native{false};
    int32_t device = -1;
    std::mutex mutex;
    std::unordered_map<std::string, SegmentHandle> peers;
    int timeout_seconds = 60;
    void context() {
        if (aclrtSetDevice(device) != ACL_SUCCESS)
            throw std::runtime_error("aclrtSetDevice failed");
    }
    SegmentHandle peer(const char* target) {
        std::lock_guard<std::mutex> guard(mutex);
        auto found = peers.find(target);
        if (found != peers.end()) return found->second;
        auto handle = native.openSegment(target);
        if (handle == static_cast<SegmentHandle>(-1))
            throw std::runtime_error("openSegment failed");
        peers.emplace(target, handle);
        return handle;
    }
    int transfer(TransferRequest::OpCode opcode, const char* target, const uint64_t* src, const uint64_t* dst,
              const uint64_t* lengths, size_t count) {
        if (!count) return 0;
        if (!target || !*target || !src || !dst || !lengths)
            throw std::invalid_argument("missing transfer arguments");
        std::vector<TransferRequest> requests;
        requests.reserve(count);
        uint64_t total = 0;
        for (size_t i = 0; i < count; ++i) {
            if (!lengths[i]) continue;
            if (!src[i] || !dst[i] || lengths[i] > UINT64_MAX - src[i] ||
                lengths[i] > UINT64_MAX - dst[i] ||
                lengths[i] > static_cast<uint64_t>(INT64_MAX) - total)
                throw std::invalid_argument("invalid transfer span");
            TransferRequest request{};
            request.opcode = opcode;
            // Mooncake calls the local address "source" for both directions.
            const bool read = opcode == TransferRequest::READ;
            request.source = reinterpret_cast<void*>(read ? dst[i] : src[i]);
            request.target_offset = read ? src[i] : dst[i];
            request.length = lengths[i];
            requests.push_back(request);
            total += lengths[i];
        }
        if (requests.empty()) return 0;
        context();
        auto timeout = std::chrono::duration<long double>(timeout_seconds) +
                       std::chrono::duration<long double>(total / 1e9L);
        if (opcode == TransferRequest::READ) {
            // The public scatter operation owns retirement, including submit
            // failures and wait timeouts. Its destructor drains physical I/O
            // before returning these destination pages to the Rust caller.
            const size_t zero = 0;
            std::vector<size_t> sizes;
            sizes.reserve(requests.size());
            for (const auto& request : requests) sizes.push_back(request.length);
            std::vector<TransferEngine::ScatterTransferRange> ranges;
            ranges.reserve(requests.size());
            for (size_t i = 0; i < requests.size(); ++i) {
                const auto& request = requests[i];
                ranges.push_back({opcode, target, request.target_offset,
                                  sizes[i], request.source, sizes[i],
                                  std::span<const size_t>(&zero, 1),
                                  std::span<const size_t>(&zero, 1),
                                  std::span<const size_t>(&sizes[i], 1), {}});
            }
            auto operation = native.submitScatter(ranges);
            const auto ns = std::chrono::duration<long double, std::nano>(timeout).count();
            const auto limit = std::chrono::nanoseconds::max().count();
            auto status = operation.waitFor(std::chrono::nanoseconds(
                ns >= limit ? limit : static_cast<int64_t>(ns)));
            if (!status.ok())
                std::fprintf(stderr, "Mooncake READ failed: %s\n", status.ToString().c_str());
            return status.ok() ? 0 : -1;
        }
        auto handle = peer(target);
        for (auto& request : requests) request.target_id = handle;
        auto start = std::chrono::steady_clock::now();
        auto batch = native.allocateBatchID(requests.size());
        if (batch == INVALID_BATCH_ID) return -1;
        BatchGuard cleanup{native, batch};
        auto submitted = native.submitTransfer(batch, requests);
        if (!submitted.ok()) {
            std::fprintf(stderr, "Mooncake submit failed: %s\n",
                         submitted.ToString().c_str());
            return -1;
        }
        for (;;) {
            TransferStatus status{};
            auto result = native.getBatchTransferStatus(batch, status);
            if (!result.ok()) return -1;
            if (status.s == TransferStatusEnum::COMPLETED) return 0;
            // A failed batch can still contain pending writes. Retrying here
            // could report success while those earlier writes remain in flight.
            if (status.s == TransferStatusEnum::FAILED ||
                status.s == TransferStatusEnum::TIMEOUT)
                return -1;
            if (std::chrono::steady_clock::now() - start > timeout) {
                std::fprintf(stderr, "Mooncake transfer timed out\n");
                return -1;
            }
        }
    }
};

template <class F>
int guarded(F&& f) noexcept {
    try {
        return f();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Mooncake NPU C ABI: %s\n", e.what());
    } catch (...) {
        std::fprintf(stderr, "Mooncake NPU C ABI: unknown exception\n");
    }
    return -1;
}

extern "C" {
int ts_mooncake_abi_version() noexcept { return 1; }
ts_mooncake_engine* ts_mooncake_create(const char* host, const char* metadata,
                                       const char* protocol,
                                       const char* device_name) noexcept {
    ts_mooncake_engine* result = nullptr;
    guarded([&] {
        if (!host || !*host || !metadata || !*metadata || !protocol ||
            std::strcmp(protocol, "ascend_direct") != 0)
            throw std::invalid_argument("this adapter requires ascend_direct");
        if (device_name && *device_name)
            throw std::invalid_argument(
                "Ascend Direct does not use an IB device filter");
        auto e = std::make_unique<ts_mooncake_engine>();
        if (aclrtGetDevice(&e->device) != ACL_SUCCESS)
            throw std::runtime_error(
                "set NPU device before creating the engine");
        if (const char* timeout = std::getenv("MC_TRANSFER_TIMEOUT")) {
            int seconds = 0;
            auto end = timeout + std::strlen(timeout);
            auto parsed = std::from_chars(timeout, end, seconds);
            if (parsed.ec != std::errc{} || parsed.ptr != end)
                throw std::invalid_argument(
                    "MC_TRANSFER_TIMEOUT must be an integer");
            e->timeout_seconds = std::max(5, seconds);
        }
        e->native.setLocalProtocolHint(protocol);
        if (e->native.init(metadata, host, "", 0, protocol) != 0)
            throw std::runtime_error("TransferEngine::init failed");
        result = e.release();
        return 0;
    });
    return result;
}
void ts_mooncake_destroy(ts_mooncake_engine* e) noexcept {
    if (e)
        guarded([&] {
            e->context();
            delete e;
            return 0;
        });
}
int ts_mooncake_rpc_port(ts_mooncake_engine* e) noexcept {
    return guarded([&] { return e ? e->native.getRpcPort() : -1; });
}
int ts_mooncake_register(ts_mooncake_engine* e, void* addr,
                         size_t length) noexcept {
    return guarded([&] {
        if (!e || !addr || !length ||
            length > UINTPTR_MAX - reinterpret_cast<uintptr_t>(addr) ||
            length > static_cast<size_t>(INT64_MAX))
            return -1;
        e->context();
        aclrtPtrAttributes attrs{};
        if (aclrtPointerGetAttributes(addr, &attrs) != ACL_SUCCESS) return -1;
        switch (attrs.location.type) {
            case ACL_MEM_LOCATION_TYPE_DEVICE:
                if (attrs.location.id != e->device) return -1;
                return e->native.registerLocalMemory(
                    addr, length, "npu:" + std::to_string(e->device));
            case ACL_MEM_LOCATION_TYPE_HOST:
            case ACL_MEM_LOCATION_TYPE_HOST_NUMA:
            case ACL_MEM_LOCATION_TYPE_UNREGISTERED:
                return e->native.registerLocalMemory(addr, length, "cpu:0");
            default:
                return -1;
        }
    });
}
int ts_mooncake_unregister(ts_mooncake_engine* e, void* addr) noexcept {
    return guarded([&] {
        if (!e || !addr) return -1;
        e->context();
        return e->native.unregisterLocalMemory(addr);
    });
}
int ts_mooncake_write(ts_mooncake_engine* e, const char* target,
                      const uint64_t* src, const uint64_t* dst,
                      const uint64_t* lengths, size_t count) noexcept {
    return guarded(
        [&] { return e ? e->transfer(TransferRequest::WRITE, target, src, dst, lengths, count) : -1; });
}
int ts_mooncake_read(ts_mooncake_engine* e, const char* target,
                     const uint64_t* src, const uint64_t* dst,
                     const uint64_t* lengths, size_t count) noexcept {
    return guarded([&] {
        return e ? e->transfer(TransferRequest::READ, target, src, dst, lengths, count) : -1;
    });
}
static int transfer_pages(TransferRequest::OpCode opcode, ts_mooncake_engine* e, const char* target,
                            const int64_t* src_pages, const int64_t* dst_pages,
                            size_t pages, const int64_t* fields,
                            size_t num_fields, size_t max_batch_size) noexcept {
    return guarded([&] {
        if (!e) return -1;
        if (!pages || !num_fields) return 0;
        if (!target || !*target || !src_pages || !dst_pages || !fields)
            throw std::invalid_argument("missing page arguments");
        constexpr auto max_elements = PTRDIFF_MAX / sizeof(int64_t);
        if (num_fields > max_elements / 5 || pages > max_elements)
            throw std::invalid_argument("field count overflow");
        if (num_fields > std::numeric_limits<size_t>::max() / pages)
            throw std::invalid_argument("page count overflow");
        auto count = pages * num_fields;
        auto step = max_batch_size ? std::min(max_batch_size, count) : count;
        std::vector<uint64_t> src, dst, lengths;
        auto address = [](int64_t base, int64_t page, int64_t stride,
                          int64_t length) {
            if (base <= 0 || page < 0 || stride < 0 || length <= 0)
                throw std::invalid_argument("invalid page span");
            auto end = static_cast<__int128>(base) +
                       static_cast<__int128>(page) * stride + length;
            if (end > std::numeric_limits<int64_t>::max())
                throw std::invalid_argument("page address overflow");
            return static_cast<uint64_t>(end - length);
        };
        // Validate all spans before submitting any part of a malformed request.
        __int128 total = 0;
        for (size_t f = 0; f < num_fields; ++f) {
            const auto* row = fields + 5 * f;
            if (row[4] <= 0) throw std::invalid_argument("invalid page length");
            total += static_cast<__int128>(row[4]) * pages;
            if (total > INT64_MAX)
                throw std::invalid_argument("page total length overflow");
            for (size_t p = 0; p < pages; ++p) {
                address(row[0], src_pages[p], row[1], row[4]);
                address(row[2], dst_pages[p], row[3], row[4]);
            }
        }
        src.reserve(step);
        dst.reserve(step);
        lengths.reserve(step);
        for (size_t f = 0; f < num_fields; ++f)
            for (size_t p = 0; p < pages; ++p) {
                const auto* row = fields + 5 * f;
                src.push_back(address(row[0], src_pages[p], row[1], row[4]));
                dst.push_back(address(row[2], dst_pages[p], row[3], row[4]));
                lengths.push_back(row[4]);
                if (src.size() == step ||
                    (f + 1 == num_fields && p + 1 == pages)) {
                    int rc = e->transfer(opcode, target, src.data(), dst.data(),
                                      lengths.data(), src.size());
                    if (rc) return rc;
                    src.clear();
                    dst.clear();
                    lengths.clear();
                }
            }
        return 0;
    });
}
int ts_mooncake_write_pages(ts_mooncake_engine* e, const char* target,
                            const int64_t* src_pages, const int64_t* dst_pages,
                            size_t pages, const int64_t* fields,
                            size_t num_fields, size_t max_batch_size) noexcept {
    return transfer_pages(TransferRequest::WRITE, e, target, src_pages, dst_pages,
                          pages, fields, num_fields, max_batch_size);
}
int ts_mooncake_read_pages(ts_mooncake_engine* e, const char* target,
                            const int64_t* src_pages, const int64_t* dst_pages,
                            size_t pages, const int64_t* fields,
                            size_t num_fields, size_t max_batch_size) noexcept {
    return transfer_pages(TransferRequest::READ, e, target, src_pages, dst_pages,
                          pages, fields, num_fields, max_batch_size);
}
}
