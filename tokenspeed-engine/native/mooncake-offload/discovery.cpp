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

#include "ha/leadership/leader_coordinator_factory.h"
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <nlohmann/json.hpp>

namespace {
std::mutex mutex;
std::map<std::string, std::unique_ptr<mooncake::ha::LeaderCoordinator>> coordinators;
nlohmann::json Resolve(const char *entry, int timeout_ms) {
  if (!entry || timeout_ms <= 0 || timeout_ms > 10000)
    throw std::invalid_argument("invalid discovery request");
  std::string endpoint(entry);
  if (!endpoint.starts_with("etcd://") || endpoint.size() == 7)
    throw std::invalid_argument("expected an etcd discovery entry");
  std::lock_guard lock(mutex);
  auto &coordinator = coordinators[endpoint];
  if (!coordinator) {
    mooncake::ha::HABackendSpec spec{.type = mooncake::ha::HABackendType::ETCD,
                                   .connstring = endpoint.substr(7), .cluster_namespace = ""};
    auto result = mooncake::ha::CreateLeaderCoordinator(spec);
    if (!result) throw std::runtime_error("Mooncake etcd coordinator creation failed: " +
                                         std::to_string(static_cast<int>(result.error())));
    coordinator = std::move(result.value());
  }
  // Use the SDK's cluster namespace, certificate authentication and warming
  // semantics. Re-read the view on each query; never retain an old leader IP.
  auto result = mooncake::ha::ReadCurrentViewOrWaitForReady(
      *coordinator, std::chrono::milliseconds(timeout_ms));
  if (!result) throw std::runtime_error("Mooncake etcd discovery failed: " +
                                       std::to_string(static_cast<int>(result.error())));
  if (!result->has_value()) throw std::runtime_error("Mooncake Master is not ready");
  return {{"endpoint", result->value().leader_address}, {"view_version", result->value().view_version}};
}
}
extern "C" char *ts_mooncake_discovery_resolve(const char *entry, int timeout_ms) noexcept {
  try {
    nlohmann::json result;
    try { result = Resolve(entry, timeout_ms); }
    catch (const std::exception &e) { result = {{"error", e.what()}}; }
    return ::strdup(result.dump().c_str());
  } catch (...) { return nullptr; }
}
extern "C" void ts_mooncake_discovery_free(char *value) noexcept { std::free(value); }
