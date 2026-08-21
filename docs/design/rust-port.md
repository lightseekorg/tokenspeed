# Rust Port: TokenSpeed Scheduler

Status: **draft** · Target: replace `tokenspeed-scheduler` (C++ control plane) with a Rust implementation behind an identical Python API.

## 1. Goal and scope

Port the C++ scheduler control plane (`tokenspeed-scheduler/`) to Rust so the
Python runtime and the kernel stack keep running unchanged. The port must be
**behaviorally identical at the Python boundary**: same `tokenspeed_scheduler`
package API, same scheduling semantics, same cache/prefix/hash contracts, and
the existing pytest suite must pass unmodified.

### In scope

- `tokenspeed-scheduler/csrc/` — scheduler FSM, cache coordinator, block pool,
  prefix matching/hashing, tier transfers, operations, Python bindings.
- `tokenspeed-scheduler/bindings/python_module.cpp` — rewritten with pyo3.
- `tokenspeed-scheduler/tests/cpp/*` — not translated; superseded by the pytest
  suite and Rust unit tests derived from the same scenarios.

### Out of scope (stays as-is)

- `tokenspeed-kernel/.../thirdparty/{cuda,msa}/` — GPU kernels (CUDA/Cutlass/
  FMHA/topk). Rust would only add an FFI layer; keep loading them as today.
- `python/tokenspeed/runtime/` — Python execution plane (431k lines).
- `tokenspeed-mla/`, `tokenspeed-kernel-amd/`.

## 2. Current C++ footprint (measured on main @ 739c0194)

| Area | Files | Lines |
|---|---|---|
| `tokenspeed-scheduler/csrc/` | 54 | 6,748 |
| `tokenspeed-scheduler/tests/cpp/` | 20 | 11,095 |
| `tokenspeed-scheduler/bindings/python_module.cpp` | 1 | 291 |
| `tokenspeed-scheduler/python/tests/` (pytest, contract) | 8 | 1,517 |
| `tokenspeed-kernel/.../thirdparty/` (CUDA, not ported) | 113 | 45,879 |

## 3. Architecture recap

```
SubmitRequests ──► Request[]  (FSM state = Bootstrapping | Submitted |
                               Prefilling | PrefillDone | Decoding |
                               Retracted | Finished)
       │
NextExecutionPlan() ──► CacheCoordinator admission (prefix match + acquire)
       │                   └─► buildForwardOperations → ForwardBatch
       │                   └─► retractForCapacity → WriteBack/LoadBack
       ▼
ExecutionPlan ──► Python executes model forward / cache transfers
       │
Advance(ExecutionEvent) ──► handleEvent(): 10 concrete events drive FSM
       │                        transitions + cache mutations
       ▼
DrainKvEvents() ──► KV cache event stream (PD cross-node sync)
```

Key design properties to preserve:

1. **Type-safe FSM**: `std::variant<State...>` + `std::visit`; illegal
   transitions throw `std::logic_error`. → Rust `enum State` + `match`, which
   is strictly safer (exhaustive).
2. **RAII resources**: `CacheBlockRef` is a shared handle whose last owner
   releases the block slot back to `BlockPool`; `ReqPoolIndex` returns its slot
   on drop.
3. **Single token buffer**: `TokenContainer` stores one `tokens_` vector plus
   `num_prefill_tokens_`; windows are offsets into it.
4. **Prefix hash chain**: SHA-256 over prefix-framed pages
   (`[prior_len][prior][count][tokens][extra_count][extra...]`), hex-encoded.
5. **LCM physical placement**: `BlockPool` binds group → LCM parents; children
   pack into slots; page 0 is reserved null.
6. **Ordered maps**: `std::map<group_id, ...>` ordering for block tables must be
   preserved (`BTreeMap`).

## 4. Module inventory and dependency chain (port order)

```
1  utils (assert helpers)                    — Rust: panic!/assert! + Result<String>
2  core/token_container                       ✅
3  resource/allocator/req_pool_allocator      ✅
4  cache/core/cache_types                     ✅
5  cache/core/block_table                     ✅
6  cache/core/cache_block_ref                 ✅
7  cache/core/block_pool                      ✅
8  cache/prefix/prefix_hasher                 ✅
9  cache/core/cache_config                    ✅
10 cache/prefix/prefix_index                  ✅
11 cache/prefix/prefix_matcher                ✅
12 cache/allocator/group_allocator            ✅
13 cache/coordinator/group_geometry           ✅
14 cache/cache_group                          ✅
15 cache/coordinator/cache_admission          ✅
16 cache/coordinator/cache_coordinator        ✅
17 cache/tier/{transfer,transfer_manager}     ✅
18 fsm/{forward_states,forward_events,
        pd_states,pd_events,states,base_event}✅
19 scheduler/{request,types,request_spec,
             kv_cache_events,outside_event_handler,
             scheduler,operations/*,outside_events/*} ✅
20 bindings/python_module.cpp (pyo3)          ✅
```

`✅` = ported in the initial scaffold (this batch). `⏳` = next batches.

## 5. C++ → Rust mapping

| C++ | Rust |
|---|---|
| `std::variant<State...>` + `std::visit(Overloaded{...})` | `enum State` + `match` |
| `std::unique_ptr<T>` | `Box<T>` / owned field |
| `std::shared_ptr<T>` | `Arc<T>` |
| `std::deque` | `VecDeque` |
| `std::map<K,V>` (ordered) | `BTreeMap<K,V>` |
| `std::unordered_map` | `HashMap` |
| `std::vector<bool>` occupancy | `Vec<bool>` (same bit-packing not needed; port uses `Vec<bool>`) |
| `std::span<const int32_t>` | `&[i32]` |
| `std::optional<T>` | `Option<T>` |
| raw `T*` cross-handles (`TokenContainer*`, `CacheCoordinator*`, `ReqPoolAllocator*`) | `Rc<RefCell<T>>` shared ownership (unsafe-free core) |
| `_assert` → `throw runtime_error` | `assert!`/`panic!` with message; config validation returns `Result<String, _>` |
| `FatalCheck` → `abort()` | `panic!` in core; binding layer converts to Python exception where appropriate |
| `std::logic_error` invalid FSM transition | `unreachable!`/`panic!` + exhaustive match; tests assert on panic |
| OpenSSL `SHA256_*` | `sha2` crate (byte-for-byte identical framing) |
| spdlog | `tracing` (binding logs to Python logging) |
| nanobind | pyo3 + maturin |

## 6. Phase 0 contract (must not drift)

### 6.1 Python API surface

The pyo3 module must expose exactly the symbols below, with identical names,
field names, defaults, and validation behavior (see
`bindings/python_module.cpp` as the authoritative list; pytest is the gate).

```
tokenspeed_scheduler (package)
├── SchedulerConfig  (Role: P/D/Fused; all writable props incl.
│                     num_device_pages, num_host_pages, max_scheduled_tokens,
│                     max_batch_size, prefix_granularity, decode_input_tokens,
│                     overlap_schedule_depth, disable_l2_cache,
│                     enable_l3_storage, enable_kv_cache_events,
│                     enable_mixed_prefill_decode, disable_prefix_cache,
│                     prefix_replay_tokens, cache_groups)
├── CacheGroupConfig (group_id, rows_per_page, entry_stride_tokens, total_pages,
│                     cache_blocks_per_lcm_block, retention, sliding_window_tokens,
│                     family, transfer_policy; validate())
├── RequestSpec      (request_id, tokens, max_new_tokens)
├── Scheduler        (submit_requests, next_execution_plan, advance,
│                     drain_kv_events, waiting_size, decoding_size,
│                     prefilling_size, pd_transfer_pinned, available_kv_pages,
│                     active_kv_pages, request_token_size,
│                     max_single_request_tokens, clear_l1_cache, clear_cache,
│                     cache_group_total_pages, cache_group_available_pages)
├── ExecutionPlan    (forward: [Forward.Batch], cache: [Cache.LoadBackOp|WriteBackOp],
│                     pages_to_zero)
├── ForwardEvent     (ExtendResult, Finish, Abort, UpdateReserveNumTokens)
├── Cache            (WriteBackDoneEvent, LoadBackDoneEvent, LoadBackOp, WriteBackOp)
├── PD               (BootstrappedEvent, FailedEvent, SucceededEvent,
│                     RemotePrefillDoneEvent)
└── KVEvent          (BlockStored, BlockRemoved)
```

### 6.2 Data-layout contracts (byte/semantics compatible)

- `ForwardBatch.block_tables`: `BTreeMap<group_id, Vec<Vec<i32>>>`, rows padded
  with `-1`, `num_reqs × max_pages_in_batch`, absolute logical-page indexing
  (null hole = 0, no compaction).
- `block_tables_arrays()`: zero-copy contiguous row-major `int32[rows, cols]`
  ndarray views per group.
- `ExecutionPlan.pages_to_zero`: `BTreeMap<group_id, Vec<i32>>`.
- KV event `block_hashes`: `u64` produced by `HashKvBlock` — must be reproduced
  byte-for-byte if consumed cross-process (check `runtime/pd/kv_events.py`
  wire format before changing).
- Request pool slots: rank-local, **1-based** (`1..=max_batch_size`), row 0
  reserved; CUDA-graph sink row at `max_batch_size + 1` is outside scheduler
  ownership.

### 6.3 Acceptance gates

1. `tokenspeed-scheduler/python/tests/` pytest suite passes unmodified against
   the Rust extension (drop-in wheel).
2. Rust unit tests per module (ported scenario tests, see §7).
3. `cargo test` + `cargo clippy -- -D warnings` + `cargo fmt --check` green.
4. Python runtime integration (`event_loop.py` driving a real model) parity:
   same execution plans for identical event streams (A/B harness).
5. No `unsafe` in the core crate (`#![forbid(unsafe_code)]`); binding crate is
   the only white-listed exception (FFI only, each block with `// SAFETY:`).

## 7. Test strategy

- **Do not translate GTest.** `tests/cpp/*` (11k lines) is preserved as the C++
  oracle for A/B checks during development, then retired at cutover.
- Port the **scenario essence** of `test_cache_coordinator.cpp` and
  `test_cache_scenarios.cpp` as Rust integration tests over the coordinator once
  modules 10–16 land.
- Rust unit tests for leaf modules now: token window slicing, pool allocate/
  release invariants, block-ref shared ownership, SHA-256 chain framing
  (golden vectors from the C++ prefix hasher tests), config validation
  messages.
- `proptest` for pool/placement invariants in a later batch.

## 8. Known risks

1. **Shared mutable state refactor** — FSM events touch `Request`,
   `CacheCoordinator`, `ReqPoolAllocator` simultaneously. Rust forces
   collect-borrow-then-mutate or `Rc<RefCell>`; highest-risk area is
   `scheduler/operations/forward.cpp` (1.2k lines).
2. **Error semantics** — C++ distinguishes `throw runtime_error` (assert),
   `abort()` (FatalCheck), `throw logic_error` (invalid transition), and
   `throw invalid_argument` (config validation, messages asserted by tests).
   Rust must map these deliberately; config messages must match character for
   character.
3. **KV event hash** — byte-for-byte `HashKvBlock` reproduction, or a
   coordinated wire change.
4. **Build/toolchain swap** — scikit-build-core + CMake + nanobind →
   maturin + pyo3; update CI (`.github/workflows`) and Dockerfiles
   (`Dockerfile.nvidia/amd/release`) install steps.
5. **Perf parity** — scheduler is CPU hot path for agentic workloads; keep an
   A/B latency bench for `next_execution_plan` from `test/` scenarios.

## 9. Roadmap

```
Phase 0  Contract freeze   ✅ this doc + API inventory + pytest green baseline
Phase 1  Core port          ✅ scaffold: workspace + modules 1–9
Phase 2  Cache coordinator  ✅ modules 10–16 (87 tests green: debug + release,
                            clippy -D warnings, fmt)
Phase 3  FSM + scheduler    ✅ modules 17–19 complete: the full scheduler core
                            (csrc/) is now ported; 110 tests green (debug +
                            release, clippy -D warnings, fmt)
Phase 4  Bindings           ✅ ts-scheduler-pyo3 (pyo3 + numpy), maturin wheel;
                            full pytest contract passes: 58/58
Phase 5  Cutover            🔶 plan below; NOT yet applied (needs explicit
                            authorization + commit gate, see §11)
```

## 10. Repository layout

```
tokenspeed-scheduler-rs/          (new root package, sibling of tokenspeed-scheduler/)
├── Cargo.toml                    (workspace)
├── README.md
└── crates/
    ├── ts-scheduler-core/        (#![forbid(unsafe_code)])
    │   └── src/                  (modules per §4)
    └── ts-scheduler-pyo3/        (white-listed unsafe for FFI)
```






## 11. Phase 5 cutover plan (drafted, not applied)

Gate: the Rust scheduler must pass the full runtime integration
(`python/tokenspeed/runtime/engine/event_loop.py` driving a real model) and the
commit gate (`pre-commit run --all-files`, currently blocked by the missing
python3.12 toolchain on this machine) before any destructive step below.

### 11.1 CI workflows (`.github/workflows/`)

| Workflow | Change |
|---|---|
| `scheduler-python-test.yml` | Replace `pip install -e 'tokenspeed-scheduler[test]'` with: install `maturin`, `maturin build`/`pip install` the wheel from `tokenspeed-scheduler-rs/crates/ts-scheduler-pyo3`, then run the same pytest command. |
| `scheduler-cpp-test.yml` | Retire: C++ GTest is superseded by `cargo test --workspace` (110 tests) + the pytest contract (58 tests). |
| `lint.yml` | Replace C++ clang-format/clang-tidy steps with `cargo fmt --check` + `cargo clippy --workspace --all-targets -- -D warnings`; keep the Python ruff/absolute-import checks. |
| `release-tokenspeed-scheduler.yml` | Replace cibuildwheel (scikit-build) with `maturin build` per arch; `package-dir` -> `tokenspeed-scheduler-rs/crates/ts-scheduler-pyo3`; drop OpenSSL `dnf install` (Rust `sha2` replaces OpenSSL). Keep sdist via maturin `sdist`. |
| `pr-test-*.yml`, `gb300-slurm-per-commit.yml` | Add `tokenspeed-scheduler-rs/**` to path filters. |

### 11.2 Docker

`Dockerfile.nvidia/amd/release` install `tokenspeed-scheduler` from PyPI — the
package name is unchanged, so no Docker change is required unless the image
builds the scheduler from source (then switch to maturin as in §11.1).

### 11.3 C++ retirement

Remove (in order, after the gate passes and the commit gate is runnable):

```
tokenspeed-scheduler/csrc/
tokenspeed-scheduler/bindings/
tokenspeed-scheduler/tests/cpp/
tokenspeed-scheduler/CMakeLists.txt
tokenspeed-scheduler/pyproject.toml      (scikit-build config)
.github/workflows/scheduler-cpp-test.yml
```

Keep `tokenspeed-scheduler/python/tokenspeed_scheduler/__init__.py` as the
canonical package (the pyo3 crate currently bundles a synced copy at
`tokenspeed-scheduler-rs/crates/ts-scheduler-pyo3/python/tokenspeed_scheduler/`).

### 11.4 Rollback

Until cutover, the C++ tree is untouched and the PyPI package can be rebuilt
from it. Rollback = restore the previous release workflow; the Rust port is
purely additive (`tokenspeed-scheduler-rs/` + `docs/design/rust-port.md`).

### 11.5 Verified packaging (this batch)

- `maturin build --release` produces a standalone wheel
  (`target/wheels/tokenspeed_scheduler-0.1.8-*.whl`) that bundles
  `tokenspeed_scheduler/__init__.py` + the compiled extension.
- A fresh venv installing only that wheel + pytest/numpy passes the full
  scheduler pytest contract: **58/58**.
- Note: maturin resolved the *remote* `python-source` path inconsistently on
  this machine, so the crate uses a *local* `python/` source dir that mirrors
  the package `__init__.py` (single file, kept in sync until cutover).
