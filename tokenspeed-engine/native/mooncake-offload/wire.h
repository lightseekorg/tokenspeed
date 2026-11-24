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

#pragma once

#include <async_simple/coro/SyncAwait.h>
#include <chrono>
#include <iostream>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <variant>
#include <vector>
#include <ylt/coro_rpc/coro_rpc_client.hpp>
#include <ylt/util/tl/expected.hpp>
// A wire-only client stub: never linked to Mooncake's RealClient
// implementation. Message layout follows mooncake-store/include/rpc_types.h.
// The method's qualified name determines coro_rpc's service id.
namespace mooncake {
enum class ErrorCode : int32_t;

using UUID = std::pair<uint64_t, uint64_t>;
class AllocatedBuffer {
public:
  struct Descriptor {
    uint64_t size_;
    uintptr_t buffer_address_;
    std::string protocol_;
    std::string transport_endpoint_;
    YLT_REFL(Descriptor, size_, buffer_address_, protocol_,
             transport_endpoint_);
  };
};
struct MemoryDescriptor {
  AllocatedBuffer::Descriptor buffer_descriptor;
  YLT_REFL(MemoryDescriptor, buffer_descriptor);
};
struct NoFDescriptor {
  AllocatedBuffer::Descriptor buffer_descriptor;
  YLT_REFL(NoFDescriptor, buffer_descriptor);
};
struct DiskDescriptor {
  std::string file_path;
  uint64_t object_size;
  YLT_REFL(DiskDescriptor, file_path, object_size);
};
struct LocalDiskDescriptor {
  UUID client_id;
  uint64_t object_size;
  std::string transport_endpoint;
  struct_pack::compatible<UUID, 1> storage_generation{};
  YLT_REFL(LocalDiskDescriptor, client_id, object_size, transport_endpoint,
           storage_generation);
};
struct DistributedFSDescriptor {
  std::string file_path;
  uint64_t offset, object_size, aligned_size;
  int shard_idx;
  YLT_REFL(DistributedFSDescriptor, file_path, offset, object_size,
           aligned_size, shard_idx);
};
enum class ReplicaStatus {
  UNDEFINED,
  INITIALIZED,
  PROCESSING,
  COMPLETE,
  REMOVED,
  FAILED
};
class Replica {
public:
  struct Descriptor {
    uint64_t id;
    std::variant<MemoryDescriptor, NoFDescriptor, DiskDescriptor,
                 LocalDiskDescriptor, DistributedFSDescriptor>
        descriptor_variant;
    ReplicaStatus status;
    YLT_REFL(Descriptor, id, descriptor_variant, status);
  };
};
struct GetReplicaListResponse {
  std::vector<Replica::Descriptor> replicas;
  uint64_t lease_ttl_ms;
  std::optional<uint64_t> object_checksum;
  YLT_REFL(GetReplicaListResponse, replicas, lease_ttl_ms, object_checksum);
};
class WrappedMasterService {
public:
  std::vector<tl::expected<GetReplicaListResponse, ErrorCode>>
  BatchGetReplicaList(const std::vector<std::string> &, const std::string &);
};

struct BatchGetOffloadObjectResponse {
  uint64_t batch_id;
  std::vector<uint64_t> pointers;
  std::string transfer_engine_addr;
  uint64_t gc_ttl_ms;
};
YLT_REFL(BatchGetOffloadObjectResponse, batch_id, pointers,
         transfer_engine_addr, gc_ttl_ms);
class RealClient {
public:
  async_simple::coro::Lazy<
      tl::expected<BatchGetOffloadObjectResponse, ErrorCode>>
  batch_get_offload_object(const std::vector<std::string> &,
                           const std::vector<int64_t> &);
  bool release_offload_buffer(uint64_t);
};
} // namespace mooncake
