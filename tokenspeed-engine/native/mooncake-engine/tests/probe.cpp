// Copyright 2026 Mooncake Authors
// SPDX-License-Identifier: Apache-2.0
// Compiled only by test_tokenspeed_cabi.py with isolated backend/ACL stubs.
#include "../transfer_engine_tokenspeed.cpp"
#include <cassert>

int main() {
    using namespace mooncake;
    assert(ts_mooncake_abi_version() == 1);
    assert(!ts_mooncake_create(nullptr, "P2PHANDSHAKE", "ascend_direct", ""));
    assert(!ts_mooncake_create("host", "P2PHANDSHAKE", "rdma", ""));
    assert(!ts_mooncake_create("host", "P2PHANDSHAKE", "ascend_direct", "ib0"));
    setenv("MC_TRANSFER_TIMEOUT", "999999999999999999999", 1);
    assert(!ts_mooncake_create("host", "P2PHANDSHAKE", "ascend_direct", ""));
    unsetenv("MC_TRANSFER_TIMEOUT");
    auto* e = ts_mooncake_create("host", "P2PHANDSHAKE", "ascend_direct", "");
    assert(e && ts_mooncake_rpc_port(e) == 1234);
    assert(ts_mooncake_rpc_port(nullptr) < 0);
    assert(ts_mooncake_register(e, reinterpret_cast<void*>(UINTPTR_MAX - 4),
                                8) < 0);
    assert(registered == 0);
    uint64_t src[] = {4096, UINT64_MAX}, dst[] = {8192, 8192},
             lengths[] = {8, 1};
    assert(ts_mooncake_write(e, "peer", src, dst, lengths, 2) < 0);
    assert(opened == 0 && submitted == 0);
    lengths[1] = 0;
    assert(ts_mooncake_write(e, "peer", src, dst, lengths, 2) == 0);
    assert(batch_sizes.back() == 1 && submitted == 1 && released == 1);
    assert(ts_mooncake_write(e, nullptr, nullptr, nullptr, nullptr, 0) == 0);
    assert(ts_mooncake_write(nullptr, "peer", src, dst, lengths, 1) < 0);
    assert(ts_mooncake_write(e, "peer", nullptr, dst, lengths, 1) < 0);
    int64_t pages[] = {0, 1, 2};
    int64_t fields[] = {4096, 32, 8192, 32, 16, 16384, 64, 32768, 64, 16};
    assert(ts_mooncake_write_pages(e, "peer", pages, pages, 3, fields, 2, 2) ==
           0);
    assert((batch_sizes == std::vector<size_t>{1, 2, 2, 2}));
    int before = submitted;
    fields[5] = INT64_MAX - 16;
    assert(ts_mooncake_write_pages(e, "peer", pages, pages, 3, fields, 2, 1) <
           0);
    assert(submitted == before);
    fields[5] = 16384;
    fields[9] = INT64_MAX / 2;
    assert(ts_mooncake_write_pages(e, "peer", pages, pages, 3, fields, 2, 1) <
           0);
    assert(submitted == before);
    fields[9] = 16;
    pages[2] = -1;
    assert(ts_mooncake_write_pages(e, "peer", pages, pages, 3, fields, 2, 1) <
           0);
    assert(submitted == before);
    assert(ts_mooncake_write_pages(e, "peer", pages, pages, SIZE_MAX, fields, 2,
                                   1) < 0);
    assert(ts_mooncake_write_pages(e, "peer", pages, pages, 3, fields, SIZE_MAX,
                                   1) < 0);
    assert(ts_mooncake_write_pages(e, nullptr, nullptr, nullptr, 0, nullptr, 1,
                                   0) == 0);
    // Submit failure/exception, terminal failure/timeout and query failure must
    // release each batch exactly once and must never retry partial writes.
    for (fail_mode = 1; fail_mode <= 5; ++fail_mode) {
        before = submitted;
        int frees = released;
        assert(ts_mooncake_write(e, "peer", src, dst, lengths, 1) < 0);
        assert(submitted == before + 1 && released == frees + 1);
    }
    fail_mode = 0;
    uint64_t remote = 1024, local = 4096, length = 16;
    assert(ts_mooncake_read(e, "peer", &remote, &local, &length, 1) == 0);
    assert(last_requests[0].opcode == TransferRequest::READ);
    assert(last_requests[0].source == reinterpret_cast<void*>(local));
    assert(last_requests[0].target_offset == remote);
    int64_t source_page = 2, dest_page = 3, grid[] = {1024, 32, 4096, 48, 8};
    assert(ts_mooncake_read_pages(e, "peer", &source_page, &dest_page, 1,
                                  grid, 1, 1) == 0);
    assert(last_requests[0].source == reinterpret_cast<void*>(4096+3*48));
    assert(last_requests[0].target_offset == 1024+2*32);
    for (fail_mode = 1; fail_mode <= 5; ++fail_mode) {
        int drains = blocking_releases;
        assert(ts_mooncake_read(e, "peer", &remote, &local, &length, 1) < 0);
        assert(blocking_releases == drains + 1);
    }
    fixture_location = ACL_MEM_LOCATION_TYPE_HOST;
    assert(ts_mooncake_register(e, reinterpret_cast<void*>(local), 16) == 0);
    assert(last_location == "cpu:0");
    assert(device_binds > 0);
    ts_mooncake_destroy(e);
    ts_mooncake_destroy(nullptr);
}
