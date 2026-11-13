# CPU benchmarks

## Admission with a full prefix cache

`tokenspeed_scheduler_full_cache_benchmark` measures a batch of long requests
against a full cache using the real C++ scheduler, prefix matcher, boundary
tree, eviction planner, and allocator. It allocates CPU metadata only: no GPU,
model weights, or KV payload is needed.

Build in Release mode, with the scheduler build dependencies installed in your
Python environment (`nanobind` and `tokenspeed-spdlog`):

```bash
cmake -S tokenspeed-scheduler -B tokenspeed-scheduler/build/full-cache \
  -DCMAKE_BUILD_TYPE=Release \
  -DTOKENSPEED_SCHEDULER_BUILD_TESTS=ON \
  -DTOKENSPEED_SCHEDULER_BUILD_PYTHON=OFF
cmake --build tokenspeed-scheduler/build/full-cache \
  --target tokenspeed_scheduler_full_cache_benchmark -j 8
```

All workload arguments are explicit, in this order:

```text
POOL_PAGES PAGE_SIZE REQUEST_TOKENS BATCH_SIZE HIT_PERCENT WARMUPS REPEATS
```

For 200 GiB at 890 bytes per token, 64-token pages give
`floor(200 * 1024^3 / (890 * 64)) = 3,770,160` usable pages, representing
241,290,240 cached tokens. The scheduler's separate null page is not counted
as usable capacity. To measure 16 requests of 128 Ki tokens at four prefix hit
fractions, with five warmups and 34 measured batches per fraction:

```bash
for hit in 0 30 60 90; do
  taskset -c 2 tokenspeed-scheduler/build/full-cache/tokenspeed_scheduler_full_cache_benchmark \
    3770160 64 131072 16 "$hit" 5 34
done > full-cache-results.jsonl
```

Choose an available, otherwise idle CPU for `taskset`. Each invocation emits
one JSON object containing all samples, median, p95, minimum, and maximum,
for both elapsed time and thread CPU time. Timings are **milliseconds per
whole batch**, not per request. Setup progress goes to stderr. Report the
CPU, compiler, Release build, source revision, affinity, and repeat count with
results. Several GiB of RAM are needed for millions of real cache entries;
the reported 200 GiB describes the represented KV payload, not benchmark RSS.

The workload fixes the following geometry and policies:

- One full-history cache group, one physical page per prefix page, no L2/L3,
  no snapshots, no speculation, no overlap, FIFO request order, and the fused
  engine role.
- The cache contains independent histories of `REQUEST_TOKENS` tokens, plus
  one shorter history if needed. Every page has its real chained SHA-256
  digest and boundary-tree node, published through
  `CacheCoordinator::CacheCompletedBlocks`. Token IDs are deterministic
  synthetic integers; histories share no prefix.
- Each query shares the requested fraction with a different recent history,
  then diverges. The fraction is a token-prefix fraction **within every
  request**, rounded down to a full page, rather than the percentage of
  requests hitting. Actual matched token counts are included in the output.
- Every sample begins with zero empty pages, every usable page in the prefix
  index, and no active requests. All batch requests retain their pages until
  the final admission. The token budget allows every unmatched prompt to
  complete in the same plan. Each request declares one output token, and
  admission also reserves one page for its first decode token.
- Between samples, query resources are released without publishing query KV,
  and evicted historical suffixes are restored outside the timed region.
  Real cache contents and chain lengths are restored; access epochs advance.
  Warmups exercise the same workload. This is a repeated workload, not a
  measurement with flushed CPU caches.

Two measurements run separately against this workload:

| JSON field | Measured work |
| --- | --- |
| `hash` | Hash all prompt pages, as the scheduler's lazy prompt hashing does |
| `prepare_keys` | Expand candidate hashes into cache-group keys |
| `prefix_probe` | `ProbePromptPrefix` with prepared hashes and keys |
| `admit_including_eviction` | `Admit`: victim selection, eviction, hit acquisition, and fresh-page allocation |
| `submit_requests` | Submit the batch to the real scheduler |
| `next_execution_plan` | One real plan build, including hashing, probing, admission, and forward-batch construction |

The phase timings sum each request's call over the batch. `Admit` is one
transaction, so its timing includes eviction and allocation; it is not a
measurement of deletion alone. `next_execution_plan` is independently timed,
not reconstructed by adding the phase medians. Request submission is reported
separately. Filling, validation, query cleanup, and restoring the full cache
are excluded. Checks enforce the expected hits, fresh pages, evictions, and
all requests completing prefill in one plan.

The small-scale `FullCacheBenchmark.*` unit tests also check that every
historical prefix remains fully matchable after restoration and that the
boundary tree agrees with the prefix index.


The optional final argument `lifecycle` measures a real two-token generation:
submission, first plan, prompt result, publication/first decode plan, final
result plus finish, and cleanup plan. `events` runs the same lifecycle with
KV events enabled and times draining them separately. Query suffixes (including
any final tail) are evicted outside timing, then all missing historical prefixes
are restored. The number of newly indexed prompt pages excludes the decode
reserve and already cached hits. Both variants validate full restoration before
every sample. Original seven-argument invocations retain the first-plan workload.


An optional argument after the mode selects `sequential` (the default) or
`shuffled` physical layout. Shuffling acquires all physical slots and returns
them in a deterministic shuffled order before filling the cache. Publication
order, prompt tokens, cache capacity and target hits are unchanged; only
physical placement changes. Setup and this permutation are outside timing.

For acceptance, alternate baseline/candidate processes at least three times
per hit rate and mode (three groups of 34 samples give 102 per revision).
Aggregate raw samples for wall and thread-CPU P50/P95; sum corresponding
lifecycle sample rows to compute total API cost. Do not sum phase medians.
Record process maximum RSS separately (for example `/usr/bin/time -v`) and
retain commands, binary hashes and machine load. Measure startup with zero
warmups separately; do not mix it into the steady-state distribution. Run
`first`, `lifecycle`, and `events` on the same CPU sequentially.

`../record_lifecycle_trace.py` records public plans, token arrays, page tables,
cache statistics and events through deterministic request/result sequences,
including aborts, delayed cache acknowledgements, Host stores, State, packing,
sharding, replay and forced retraction. Build an isolated native extension for
each revision with its matching headers/core, run the driver with `--module-dir`
and `--output`, and compare the parsed JSON. This complements the C++ and
Python behavioral suites; it does not expose or depend on hash-slot order.

## Decode rounds with long contexts

`tokenspeed_scheduler_decode_rounds_benchmark` measures the steady state of a
decode server: a batch of long-context requests that all decode in every
round, each accepting `DECODE_TOKENS` tokens per round (the verify width) so
that the block tables keep growing. Build it like the full-cache benchmark
(target `tokenspeed_scheduler_decode_rounds_benchmark`); its arguments are

```text
PAGE_SIZE CONTEXT_TOKENS REQUESTS DECODE_TOKENS WARMUP_ROUNDS ROUNDS
```

Request `i`'s prompt is `CONTEXT_TOKENS + i % PAGE_SIZE` tokens, so requests
cross page boundaries in different rounds. Admission reserves part of each
request's generation budget up front, and a table grows only once decode has
used that reserve: setup prefills every prompt in one plan and decodes,
unmeasured, until the tables start to grow. The output reports thread-CPU and
wall time of each round's `NextExecutionPlan` (`plan`) and of applying its
results (`advance`). The cost of a round's admission should not depend on the
context length; building the forward batch copies every request's page table
and does.

# Sanitizer builds

`TOKENSPEED_SCHEDULER_SANITIZE` lists sanitizers to build the core library
with, and with it every test, benchmark and binding that links the core. It is
empty, meaning no sanitizer, by default. To run the C++ tests under
AddressSanitizer (with LeakSanitizer) and UndefinedBehaviorSanitizer:

```bash
cmake -S tokenspeed-scheduler -B tokenspeed-scheduler/build/sanitize \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DTOKENSPEED_SCHEDULER_BUILD_TESTS=ON \
  -DTOKENSPEED_SCHEDULER_BUILD_PYTHON=OFF \
  "-DTOKENSPEED_SCHEDULER_SANITIZE=address;undefined"
cmake --build tokenspeed-scheduler/build/sanitize \
  --target tokenspeed_scheduler_tests -j 8
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  tokenspeed-scheduler/build/sanitize/tokenspeed_scheduler_tests
```

ASan stops at its first report and LeakSanitizer reports leaks at exit; UBSan
only prints its reports unless `halt_on_error=1` is set. The compiler's
sanitizer runtime libraries must be installed. When the environment sets
`LD_PRELOAD`, ASan refuses to start because its runtime does not come first:
run the tests with the compiler's `libasan` first in `LD_PRELOAD`. The build
runs the test binary too, to list its tests for `gtest_discover_tests`; there,
set `ASAN_OPTIONS=verify_asan_link_order=0` instead, since preloading ASan
into CMake and Ninja would make LeakSanitizer fail them.

`FixedSlotResource.FreeSlotsArePoisonedUntilHandedOut` runs only in an ASan
build and is skipped otherwise. `cache/core/fixed_slot_resource.h` detects ASan
from the compiler (`__SANITIZE_ADDRESS__`, or Clang's
`__has_feature(address_sanitizer)`) and then poisons every free pooled slot
with `__asan_poison_memory_region`: the slot's memory stays allocated, yet ASan
reports a read through a stale pointer to the freed node. The test checks with
`__asan_address_is_poisoned` that freed slots are poisoned and handed-out slots
are not. Make sure the run reports it as passed, not skipped.
