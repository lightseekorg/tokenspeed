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

#include "wire.h"
#include <cstring>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>

namespace {
using Json = nlohmann::json;
using async_simple::coro::syncAwait;

// A provider restarts its batch counter. Never reconnect and release an old
// numeric id against a new process at the same endpoint. Each local handle owns
// exactly the TCP connection on which that source was prepared.
using Clock = std::chrono::steady_clock;
struct Lease {
  std::shared_ptr<coro_rpc::coro_rpc_client> client;
  std::string host, port;
  uint64_t remote_id = 0;
  Clock::time_point expires = Clock::time_point::max();
};
std::mutex leases_mutex;
std::map<uint64_t, std::shared_ptr<Lease>> leases;
uint64_t next_lease = 1;
constexpr size_t kMaxLeases = 1024;

void Prune() {
  const auto now = Clock::now();
  for (auto it = leases.begin(); it != leases.end();) {
    if (it->second->expires <= now)
      it = leases.erase(it);
    else
      ++it;
  }
}

// Own the registry entry before issuing preparation, including exception paths.
struct PendingLease {
  uint64_t id;
  bool committed = false;
  explicit PendingLease(const std::shared_ptr<Lease> &lease) {
    std::lock_guard lock(leases_mutex);
    Prune();
    if (leases.size() >= kMaxLeases ||
        next_lease == std::numeric_limits<uint64_t>::max())
      throw std::runtime_error("prepared connection budget exceeded");
    id = next_lease++;
    leases.emplace(id, lease);
  }
  ~PendingLease() {
    if (!committed) {
      std::lock_guard lock(leases_mutex);
      leases.erase(id);
    }
  }
};

// Discovery lives in a separate SDK-linked library: the RPC-only bridge keeps
// its protocol declarations isolated from Mooncake implementation classes.
std::string ResolveMaster(const char *entry, int timeout_ms) {
  using Resolve = char *(*)(const char *, int);
  using Free = void (*)(char *);
  struct Discovery {
    void *library;
    Resolve resolve;
    Free free;
    Discovery() {
      const char *override = std::getenv("TOKENSPEED_MOONCAKE_DISCOVERY_LIBRARY");
      Dl_info location{};
      if (!dladdr(reinterpret_cast<void *>(&ResolveMaster), &location))
        throw std::runtime_error("cannot locate discovery adapter");
      const auto path = override ? std::filesystem::path(override)
          : std::filesystem::path(location.dli_fname).parent_path() / "libtokenspeed_mooncake_discovery.so";
      library = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
      if (!library) throw std::runtime_error(dlerror());
      resolve = reinterpret_cast<Resolve>(dlsym(library, "ts_mooncake_discovery_resolve"));
      free = reinterpret_cast<Free>(dlsym(library, "ts_mooncake_discovery_free"));
      if (!resolve || !free) {
        dlclose(library);
        throw std::runtime_error("invalid Mooncake discovery adapter ABI");
      }
    }
    ~Discovery() { dlclose(library); }
  };
  static Discovery discovery;
  char *raw = discovery.resolve(entry, timeout_ms);
  if (!raw) throw std::runtime_error("discovery response allocation failed");
  std::unique_ptr<char, Free> response(raw, discovery.free);
  auto value = Json::parse(raw);
  if (value.contains("error")) throw std::runtime_error(value.at("error").get<std::string>());
  return value.at("endpoint").get<std::string>();
}

Json Call(const char *host, const char *port, const Json &q) {
  const auto timeout_ms = q.at("timeout_ms").get<int>();
  if (timeout_ms < 1 || timeout_ms > 10000)
    throw std::invalid_argument("invalid RPC timeout");
  const auto timeout = std::chrono::milliseconds(timeout_ms);
  const auto mode = q.at("mode").get<std::string>();
  if (mode == "release") {
    const auto local_id = q.at("batch_id").get<uint64_t>();
    std::shared_ptr<Lease> lease;
    {
      std::lock_guard lock(leases_mutex);
      Prune();
      const auto it = leases.find(local_id);
      if (it == leases.end())
        return Json{{"released", false}};
      if (it->second->host != host || it->second->port != port)
        throw std::invalid_argument("lease endpoint mismatch");
      lease = std::move(it->second);
      leases.erase(it);
    }
    // No connect(), retry or alternate endpoint here. A broken original
    // connection cannot vouch for this source's generation.
    auto reply = syncAwait(
        lease->client->call_for<&mooncake::RealClient::release_offload_buffer>(
            timeout, lease->remote_id));
    if (!reply)
      throw std::runtime_error(std::string(reply.error().msg));
    return Json{{"released", reply.value()}};
  }
  std::string resolved_host, resolved_port;
  if (mode == "query" && std::string_view(host).starts_with("etcd://")) {
    const auto endpoint = ResolveMaster(host, timeout_ms);
    const auto split = endpoint.rfind(':');
    if (split == std::string::npos) throw std::runtime_error("invalid discovered Master endpoint");
    resolved_host = endpoint.substr(0, split);
    if (resolved_host.starts_with('[') && resolved_host.ends_with(']'))
      resolved_host = resolved_host.substr(1, resolved_host.size() - 2);
    resolved_port = endpoint.substr(split + 1);
    host = resolved_host.c_str();
    port = resolved_port.c_str();
  }
  const auto started = Clock::now();
  auto client = std::make_shared<coro_rpc::coro_rpc_client>();
  if (auto error = syncAwait(client->connect(host, port, timeout)); error)
    throw std::runtime_error(std::string(error.message()));
  const auto remaining = [&] {
    const auto left =
        timeout - std::chrono::duration_cast<std::chrono::milliseconds>(
                      Clock::now() - started);
    if (left.count() <= 0)
      throw std::runtime_error("RPC deadline elapsed during connect");
    return left;
  };
  auto keys = q.at("keys").get<std::vector<std::string>>();
  auto tenant = q.at("tenant").get<std::string>();
  if (keys.empty() || keys.size() > 1024 || tenant.empty() ||
      tenant.size() > 4096 || tenant.find('\0') != std::string::npos)
    throw std::invalid_argument("invalid bounded offload query");
  for (const auto &key : keys)
    if (key.empty() || key.size() > 4096 || key.find('\0') != std::string::npos)
      throw std::invalid_argument("invalid object key");
  if (mode == "query") {
    auto reply = syncAwait(
        client->call_for<&mooncake::WrappedMasterService::BatchGetReplicaList>(
            remaining(), keys, tenant));
    if (!reply)
      throw std::runtime_error(std::string(reply.error().msg));
    if (reply.value().size() != keys.size())
      throw std::runtime_error("query result count mismatch");
    auto rows = Json::array();
    for (const auto &item : reply.value()) {
      if (!item) {
        rows.push_back({{"error", int(item.error())}});
        continue;
      }
      auto replicas = Json::array();
      for (const auto &replica : item.value().replicas) {
        if (replica.status != mooncake::ReplicaStatus::COMPLETE)
          continue;
        if (const auto *disk = std::get_if<mooncake::LocalDiskDescriptor>(
                &replica.descriptor_variant))
          replicas.push_back({{"endpoint", disk->transport_endpoint},
                              {"bytes", disk->object_size},
                              {"generation", disk->storage_generation.value_or(
                                                 mooncake::UUID{})}});
      }
      rows.push_back({{"ssd", replicas}});
    }
    return Json{{"objects", rows}};
  }
  if (mode == "prepare") {
    const auto sizes = q.at("sizes").get<std::vector<int64_t>>();
    const auto generations =
        q.at("generations").get<std::vector<mooncake::UUID>>();
    if (sizes.size() != keys.size() || generations.size() != keys.size())
      throw std::invalid_argument("prepare columns disagree");
    uint64_t total = 0;
    for (const auto size : sizes) {
      if (size <= 0 || uint64_t(size) > (1ULL << 31) ||
          total > (1ULL << 31) - uint64_t(size))
        throw std::invalid_argument("prepare byte budget exceeded");
      total += uint64_t(size);
    }
    for (size_t i = 0; i < keys.size(); ++i) {
      auto logical_key = tenant + std::string(1, '\0') + keys[i];
      // Match Mooncake's GenerationStorageKey: generation zero is the legacy
      // layout; a nonzero incarnation addresses only that exact SSD object.
      const auto generation = generations[i];
      if (generation == mooncake::UUID{}) {
        keys[i] = std::move(logical_key);
      } else {
        std::string physical_key("\0MCGEN1", 7);
        for (auto part : {generation.first, generation.second})
          for (unsigned byte = 0; byte < 8; ++byte)
            physical_key += static_cast<char>(part >> (byte * 8));
        physical_key += logical_key;
        keys[i] = std::move(physical_key);
      }
    }
    auto lease = std::make_shared<Lease>();
    lease->client = client;
    lease->host = host;
    lease->port = port;
    PendingLease pending(lease);
    auto reply = syncAwait(
        client->call_for<&mooncake::RealClient::batch_get_offload_object>(
            remaining(), keys, sizes));
    if (!reply)
      throw std::runtime_error(std::string(reply.error().msg));
    if (!reply.value())
      return Json{{"remote_error", int(reply.value().error())}};
    const auto &ready = reply.value().value();
    if (ready.gc_ttl_ms == 0 || ready.gc_ttl_ms > 3'600'000)
      throw std::runtime_error("invalid provider source TTL");
    {
      std::lock_guard lock(leases_mutex);
      lease->remote_id = ready.batch_id;
      lease->expires = started + std::chrono::milliseconds(ready.gc_ttl_ms);
      pending.committed = true;
    }
    // batch_id is opaque to callers: the SDK's process-local id never
    // leaves this connection guard.
    return Json{{"batch_id", pending.id},
                {"addresses", ready.pointers},
                {"session", ready.transfer_engine_addr},
                {"ttl_ms", ready.gc_ttl_ms}};
  }
  throw std::invalid_argument("unknown offload operation");
}
} // namespace

// A small, separately loaded client ABI. It links no Mooncake server or store
// implementation and does not replace or upgrade the deployed Mooncake binary.
extern "C" int ts_mooncake_offload_abi() { return 4; }
extern "C" char *ts_mooncake_offload_call(const char *host, const char *port,
                                          const char *request) noexcept {
  try {
    Json result;
    try {
      result = Call(host, port, Json::parse(request));
    } catch (const std::exception &error) {
      result = Json{{"error", error.what()}};
    }
    const auto encoded = result.dump();
    auto *bytes = new char[encoded.size() + 1];
    std::memcpy(bytes, encoded.c_str(), encoded.size() + 1);
    return bytes;
  } catch (...) {
    return nullptr;
  }
}
extern "C" void ts_mooncake_offload_free(char *reply) noexcept {
  delete[] reply;
}
