#include "ascend_saver.h"
#include "fake_acl.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#define REQUIRE(condition) do { if (!(condition)) throw std::runtime_error( \
  std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": " #condition \
  + "; backend=" + saver_last_error() + "; stats=" + saver_stats()); } while (false)

namespace {
void ok(int result) { REQUIRE(result == 0); }
bool stat_has(const std::string &text) { return std::string(saver_stats()).find(text) != std::string::npos; }
void reset() {
  REQUIRE(fake_acl::live_reservations() == 0);
  REQUIRE(fake_acl::live_physical() == 0);
  REQUIRE(fake_acl::live_host() == 0);
  ok(saver_resume(nullptr));
  fake_acl::reset();
}
void pattern(void *ptr, size_t size, unsigned char value) { std::memset(ptr, value, size); }
void expect_pattern(void *ptr, size_t size, unsigned char value) {
  auto *bytes = static_cast<unsigned char *>(ptr);
  for (size_t i = 0; i < size; ++i) REQUIRE(bytes[i] == value);
}

void backup_roundtrip() {
  reset();
  REQUIRE(saver_abi_version() == 2);
  constexpr size_t n = 7000;
  void *ptr = weights_malloc(n, 0, nullptr);
  REQUIRE(ptr != nullptr);
  REQUIRE(stat_has("\"aligned_bytes\":8192"));
  pattern(ptr, n, 0x6b);
  for (int i = 0; i < 5; ++i) {
    ok(saver_pause("weights"));
    REQUIRE(fake_acl::live_reservations() == 1);
    REQUIRE(fake_acl::live_physical() == 0);
    REQUIRE(fake_acl::live_host() == 1);
    REQUIRE(stat_has("\"paused_segments\":1"));
    REQUIRE(stat_has("\"physical_bytes\":0"));
    ok(saver_pause("weights"));
    ok(saver_resume("weights"));
    expect_pattern(ptr, n, 0x6b); // Same original pointer, no re-reservation.
    REQUIRE(fake_acl::live_host() == 0);
    ok(saver_resume("weights"));
  }
  REQUIRE(fake_acl::calls("aclrtReserveMemAddress") == 1);
  REQUIRE(fake_acl::calls("aclrtMapMem") == 6);
  REQUIRE(fake_acl::calls("aclrtSynchronizeDevice") == 10);
  ok(saver_release(ptr));
}

void default_policies_tags_and_paused_free() {
  reset();
  void *w = weights_malloc(100, 0, nullptr);
  void *k = kv_malloc(100, 0, nullptr);
  void *g = graph_malloc(100, 0, nullptr);
  REQUIRE(w && k && g);
  pattern(w, 100, 0x11); pattern(g, 100, 0x22);
  REQUIRE(stat_has("\"tag\":\"acl_graph\""));
  REQUIRE(saver_pause("cuda_graph") != 0);
  REQUIRE(saver_pause("graph") != 0);
  REQUIRE(fake_acl::live_physical() == 3);
  ok(saver_pause("acl_graph"));
  REQUIRE(fake_acl::live_physical() == 2);
  REQUIRE(fake_acl::live_host() == 1);
  ok(saver_resume("acl_graph")); expect_pattern(g, 100, 0x22);
  ok(saver_pause(nullptr));
  REQUIRE(fake_acl::live_physical() == 0);
  REQUIRE(fake_acl::live_host() == 2); // Weights+graph backup, KV discard.
  const auto before_copies = fake_acl::calls("aclrtMemcpy");
  ok(saver_release(g)); // Free a paused backup block without copying it back.
  REQUIRE(fake_acl::calls("aclrtMemcpy") == before_copies);
  ok(saver_resume(nullptr));
  expect_pattern(w, 100, 0x11);
  expect_pattern(k, 100, 0xcd); // New fake physical memory, old KV data discarded.
  saver_free(w, 100, 0, nullptr); REQUIRE(std::strlen(saver_last_error()) == 0);
  ok(saver_release(k));
}

void pause_failure_matrix() {
  for (const char *api : {"aclrtMallocHost", "aclrtMemcpy", "aclrtSynchronizeDevice",
                          "aclrtUnmapMem", "aclrtFreePhysical"}) {
    reset();
    void *p = graph_backup_malloc(321, 0, nullptr); REQUIRE(p);
    pattern(p, 321, 0x37);
    fake_acl::fail_next(api);
    REQUIRE(saver_pause("acl_graph") != 0);
    REQUIRE(std::strlen(saver_last_error()) != 0);
    REQUIRE(graph_backup_malloc(16, 0, nullptr) == nullptr); // Tag stays closed.
    REQUIRE(fake_acl::live_reservations() == 1);
    ok(saver_pause("acl_graph"));
    REQUIRE(fake_acl::live_host() == 1 && fake_acl::live_physical() == 0);
    ok(saver_resume("acl_graph")); expect_pattern(p, 321, 0x37);
    ok(saver_release(p));
  }
}

void resume_failure_matrix() {
  for (const char *api : {"aclrtMallocPhysical", "aclrtMapMem",
                          "aclrtMemcpy", "aclrtSynchronizeDevice", "aclrtFreeHost"}) {
    reset();
    void *p = weights_backup_malloc(77, 0, nullptr); REQUIRE(p);
    pattern(p, 77, 0xa7); ok(saver_pause("weights"));
    fake_acl::fail_next(api);
    REQUIRE(saver_resume("weights") != 0);
    REQUIRE(fake_acl::live_host() == 1);
    REQUIRE(stat_has("\"backup_valid\":true"));
    REQUIRE(stat_has("\"active_segments\":0"));
    REQUIRE(weights_malloc(1, 0, nullptr) == nullptr);
    ok(saver_resume("weights"));
    expect_pattern(p, 77, 0xa7);
    REQUIRE(fake_acl::live_host() == 0);
    REQUIRE(fake_acl::calls("aclrtReserveMemAddress") == 1);
    ok(saver_release(p));
  }
}

void abort_failed_pause_and_repause_failed_resume() {
  reset();
  void *p = weights_malloc(19, 0, nullptr); REQUIRE(p); pattern(p, 19, 0x49);
  fake_acl::fail_next("aclrtMemcpy");
  REQUIRE(saver_pause("weights") != 0);
  ok(saver_resume("weights")); // Failed backup left original physical data valid.
  expect_pattern(p, 19, 0x49);
  ok(saver_pause("weights"));
  fake_acl::fail_next("aclrtMemcpy");
  REQUIRE(saver_resume("weights") != 0);
  ok(saver_pause("weights")); // Keep original valid shadow after failed H2D.
  REQUIRE(fake_acl::live_physical() == 0 && fake_acl::live_host() == 1);
  ok(saver_resume("weights")); expect_pattern(p, 19, 0x49);
  ok(saver_release(p));
}

void allocation_failure_cleanup() {
  for (const char *api : {"aclrtGetDevice", "aclrtGetCurrentContext", "aclrtMemGetAllocationGranularity", "aclrtReserveMemAddress",
                          "aclrtMallocPhysical", "aclrtMapMem"}) {
    reset(); fake_acl::fail_next(api);
    REQUIRE(weights_malloc(42, 0, nullptr) == nullptr);
    REQUIRE(std::strlen(saver_last_error()) != 0);
    REQUIRE(fake_acl::live_reservations() == 0 && fake_acl::live_physical() == 0);
    REQUIRE(saver_has_callback_fault() == 0);
    void *retry = weights_malloc(42, 0, nullptr); REQUIRE(retry);
    ok(saver_release(retry));
  }
}

void allocation_rollback_fault_is_terminal() {
  reset();
  void *kv = kv_malloc(42, 0, nullptr); REQUIRE(kv);
  fake_acl::fail_next("aclrtMapMem");
  fake_acl::fail_next("aclrtFreePhysical");
  REQUIRE(weights_malloc(42, 0, nullptr) == nullptr);
  const std::string error = saver_last_error();
  REQUIRE(error.find("aclrtMapMem failed") != std::string::npos);
  REQUIRE(error.find("allocation rollback failed: aclrtFreePhysical failed") != std::string::npos);
  REQUIRE(saver_has_callback_fault() == 1);
  REQUIRE(stat_has("\"releasing_segments\":1"));
  REQUIRE(fake_acl::live_reservations() == 2 && fake_acl::live_physical() == 2);
  const auto reserves = fake_acl::calls("aclrtReserveMemAddress");
  REQUIRE(weights_malloc(42, 0, nullptr) == nullptr);
  REQUIRE(fake_acl::calls("aclrtReserveMemAddress") == reserves);
  REQUIRE(saver_pause("kv_cache") != 0);
  REQUIRE(saver_resume("kv_cache") != 0);
  REQUIRE(stat_has("\"active_segments\":1"));
  ok(saver_release(kv));
  ok(saver_collect_failed());
  REQUIRE(saver_has_callback_fault() == 1);
  REQUIRE(stat_has("\"callback_error\":\"" + error + "\""));
  REQUIRE(fake_acl::live_reservations() == 0 && fake_acl::live_physical() == 0);
}

void free_failure_matrix() {
  for (const char *api : {"aclrtSynchronizeDevice", "aclrtUnmapMem", "aclrtFreePhysical", "aclrtReleaseMemAddress"}) {
    reset(); void *p = kv_malloc(12, 0, nullptr); REQUIRE(p);
    fake_acl::fail_next(api);
    REQUIRE(saver_release(p) != 0); // Checked API is recoverable, unlike void callback.
    REQUIRE(std::strlen(saver_last_error()) != 0);
    REQUIRE(stat_has("\"releasing_segments\":1"));
    REQUIRE(fake_acl::live_reservations() == 1);
    if (std::strcmp(api, "aclrtSynchronizeDevice") == 0)
      REQUIRE(fake_acl::calls("aclrtUnmapMem") == 0);
    ok(saver_collect_failed());
  }
  reset(); void *p = graph_malloc(12, 0, nullptr); REQUIRE(p);
  ok(saver_pause("acl_graph")); fake_acl::fail_next("aclrtFreeHost");
  REQUIRE(saver_release(p) != 0);
  REQUIRE(fake_acl::live_host() == 1 && fake_acl::live_reservations() == 1);
  ok(saver_release(p));
}

void guards_and_partial_batch() {
  reset();
  REQUIRE(weights_malloc(std::numeric_limits<size_t>::max(), 0, nullptr) == nullptr);
  REQUIRE(fake_acl::live_reservations() == 0);
  REQUIRE(saver_pause("unknown") != 0);
  ok(saver_pause("kv_cache"));
  REQUIRE(kv_malloc(10, 0, nullptr) == nullptr); // Empty paused tag also rejects alloc.
  ok(saver_resume("kv_cache"));
  void *w = weights_discard_malloc(12, 0, nullptr); REQUIRE(w);
  void *g = graph_malloc(12, 0, nullptr); REQUIRE(g); pattern(g, 12, 0x31);
  fake_acl::set_device(1);
  REQUIRE(saver_pause(nullptr) != 0);
  REQUIRE(stat_has("\"active_segments\":2"));
  REQUIRE(weights_malloc(12, 1, nullptr) == nullptr);
  fake_acl::set_device(0);
  int alternate_context = 0;
  fake_acl::set_context(&alternate_context); // Same device, different context.
  const size_t unmaps = fake_acl::calls("aclrtUnmapMem");
  REQUIRE(saver_pause(nullptr) != 0);
  REQUIRE(saver_resume(nullptr) != 0);
  REQUIRE(saver_release(w) != 0);
  REQUIRE(saver_collect_failed() != 0);
  REQUIRE(weights_malloc(12, 0, nullptr) == nullptr);
  REQUIRE(fake_acl::calls("aclrtUnmapMem") == unmaps);
  REQUIRE(stat_has("\"active_segments\":2"));
  fake_acl::set_context(nullptr);
  ok(saver_collect_failed());
  REQUIRE(saver_has_callback_fault() == 0);
  REQUIRE(stat_has("\"active_segments\":2"));
  fake_acl::fail_next("aclrtMallocHost");
  REQUIRE(saver_pause(nullptr) != 0); // First discard block paused; next backup failed.
  REQUIRE(stat_has("\"paused_segments\":1"));
  REQUIRE(stat_has("\"transitional_segments\":1"));
  ok(saver_resume(nullptr)); expect_pattern(g, 12, 0x31);
  ok(saver_release(w)); ok(saver_release(g));
}

void callback_fault_is_terminal() {
  reset();
  REQUIRE(saver_has_callback_fault() == 0);
  void *p = kv_malloc(17, 0, nullptr); REQUIRE(p);
  fake_acl::fail_next("aclrtSynchronizeDevice");
  saver_free(p, 17, 0, nullptr);
  REQUIRE(saver_has_callback_fault() == 1);
  REQUIRE(stat_has("\"callback_fault\":true"));
  REQUIRE(stat_has("\"callback_error\":\"aclrtSynchronizeDevice before final free failed"));
  REQUIRE(fake_acl::calls("aclrtUnmapMem") == 0);
  REQUIRE(weights_malloc(12, 0, nullptr) == nullptr);
  REQUIRE(saver_pause(nullptr) != 0);
  REQUIRE(saver_resume(nullptr) != 0);
  ok(saver_collect_failed()); // Cleanup is allowed, but cannot reopen a worker.
  REQUIRE(saver_has_callback_fault() == 1);
  REQUIRE(fake_acl::live_reservations() == 0);
  saver_free(p, 17, 0, nullptr); // A later unknown-pointer error cannot erase first fault.
  REQUIRE(stat_has("\"callback_error\":\"aclrtSynchronizeDevice before final free failed"));
  REQUIRE(stat_has("\"last_backend_error\":\"unknown or already freed pointer\""));
}

void callback_retires_before_validation(bool wrong_context) {
  reset();
  constexpr size_t size = 17;
  void *p = graph_malloc(size, 0, nullptr);
  REQUIRE(p);
  if (!wrong_context) {
    ok(saver_pause("acl_graph"));
    REQUIRE(fake_acl::live_host() == 1);
  }

  int alternate_context = 0;
  if (wrong_context) fake_acl::set_context(&alternate_context);
  const size_t syncs = fake_acl::calls("aclrtSynchronizeDevice");
  saver_free(p, size, wrong_context ? 0 : 1, nullptr);
  const std::string first_error = saver_last_error();
  REQUIRE(first_error.find(wrong_context ? "current NPU device/context differs"
                                         : "free callback device does not match") != std::string::npos);
  REQUIRE(saver_has_callback_fault() == 1);
  REQUIRE(fake_acl::calls("aclrtSynchronizeDevice") == syncs);
  REQUIRE(fake_acl::live_reservations() == 1);
  if (wrong_context) REQUIRE(saver_collect_failed() != 0);
  fake_acl::set_context(nullptr);

  ok(saver_collect_failed());
  REQUIRE(fake_acl::live_reservations() == 0);
  REQUIRE(fake_acl::live_physical() == 0);
  REQUIRE(fake_acl::live_host() == 0);
  REQUIRE(stat_has("\"segments\":0"));
  REQUIRE(stat_has("\"mapped_bytes\":0"));
  REQUIRE(stat_has("\"reserved_bytes\":0"));
  REQUIRE(stat_has("\"physical_bytes\":0"));
  REQUIRE(stat_has("\"host_bytes\":0"));

  int unknown_pointer = 0;
  saver_free(&unknown_pointer, size, 0, nullptr);
  REQUIRE(std::string(saver_last_error()) == "unknown or already freed pointer");
  REQUIRE(stat_has("\"callback_error\":\"" + first_error + "\""));
  REQUIRE(saver_has_callback_fault() == 1);
  REQUIRE(weights_malloc(size, 0, nullptr) == nullptr);
  REQUIRE(saver_pause(nullptr) != 0);
  REQUIRE(saver_resume(nullptr) != 0);
}
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc == 2) {
      const std::string scenario = argv[1];
      if (scenario == "malloc-rollback") {
        allocation_rollback_fault_is_terminal();
      } else {
        REQUIRE(scenario == "wrong-context" || scenario == "wrong-device");
        callback_retires_before_validation(scenario == "wrong-context");
      }
      std::cout << "PASS: " << scenario << " fault is terminal and cleanup released all resources\n";
      return 0;
    }
    REQUIRE(argc == 1);
    backup_roundtrip();
    default_policies_tags_and_paused_free();
    pause_failure_matrix();
    resume_failure_matrix();
    abort_failed_pause_and_repause_failed_resume();
    allocation_failure_cleanup();
    free_failure_matrix();
    guards_and_partial_batch();
    reset();
    callback_fault_is_terminal(); // Must run last: intentionally no in-process reset.
    REQUIRE(fake_acl::live_reservations() == 0);
    REQUIRE(fake_acl::live_physical() == 0);
    REQUIRE(fake_acl::live_host() == 0);
    std::cout << "PASS: 9 state-machine scenarios with API failure matrices; no fake resources leaked\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "FAIL: " << e.what() << '\n'; return 1;
  }
}
