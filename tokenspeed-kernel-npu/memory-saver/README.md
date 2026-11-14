# Ascend memory saver backend

This shared library supplies allocator callbacks and checked pause/resume
operations for TokenSpeed's NPU Memory Saver. Build it against the CANN runtime
used by the worker. The hardware baseline is Linux aarch64, Atlas A2 (910B3),
CANN 9.0.0, and PyTorch 2.9.0 with a PTA 2.9.0-730 backport wheel.
[Ascend setup and usage](../../docs/serving/ascend-memory-saver.md)

## PTA options

The primary baseline is PTA 2.9.0-730 plus the six changes below, with
PyTorch `2.9.0+cpu` and CANN/HCCL 9.0.0. Build the backport wheel against the
deployment's Python, PyTorch C++ ABI, and CANN.
The tested PTA build is `ee1742b06d2c0e59cb0b357288ec29ee190cfd2d`. The public
snapshot `74e6508b896f630e5d5cacd39a3877047e91c19a` has the same source tree
`546ac928ce6199c1e2f221d7f5279b4eef6713a8`. The build can report either
commit, depending on the source checkout.
[Public PTA branch](https://gitcode.com/Huyanlong/pytorch/tree/backport/pta-730-acl-graph-offload)
· [Pinned snapshot](https://gitcode.com/Huyanlong/pytorch/commit/74e6508b896f630e5d5cacd39a3877047e91c19a)

Stock TorchNPU 2.13.0 contains these changes and passed an earlier
single-host A2 compatibility run with CANN 9.0.0. Its official CANN pairing
is 9.2.0. That historical run is separate from the current 2.9.0 revalidation.
[Installation and coverage](../../docs/serving/ascend-memory-saver.md#before-you-start)

The snapshot applies exactly six official upstream changes to the July 30
PTA 2.9.0-730 base
[`9d40ac3`](https://gitcode.com/Ascend/pytorch/commit/9d40ac3a2b3714df7e5d164bda28a42eff3282cd).
No additional PTA patches or system-library replacements are required:

| Backport | Required behavior |
| --- | --- |
| [!39221](https://gitcode.com/Ascend/pytorch/merge_requests/39221) | MemPool allocation and lifetime handling. |
| [!45270](https://gitcode.com/Ascend/pytorch/merge_requests/45270) | Keep the correct allocator owner when a MemPool is used for capture. |
| [!30625](https://gitcode.com/Ascend/pytorch/merge_requests/30625) | Support normal ProcessGroupHCCL shutdown. |
| [!32154](https://gitcode.com/Ascend/pytorch/merge_requests/32154) | Synchronize streams during HCCL shutdown. |
| [!32564](https://gitcode.com/Ascend/pytorch/merge_requests/32564) | Preserve task queue ordering during multi-stream ACL Graph capture. |
| [!38431](https://gitcode.com/Ascend/pytorch/merge_requests/38431) | Remove the HCCL pending-event wait from ACL Graph capture. |

PTA builds must provide `NPUPluggableAllocator`, `MemPool`, `use_mem_pool`,
and ACL Graph capture into a user pool with allocator ownership preserved.
The worker checks the required APIs and native library ABI at startup.
After model loading and capture, it verifies NPU parameter storage, device KV
regions, and captured graph-pool segments against live native VMM allocations.
Tied parameters and views can share backing segments. Eager startup requires
weight and KV backing without a graph pool. A captured graph may have an
allocation-free pool. Keep
`PYTORCH_NO_NPU_MEMORY_CACHING` unset or `0` so PTA routes these allocations
through their managed MemPools.
Qualify other PTA or CANN builds with the real-device probes, which check
how PTA routes allocations and frees.

## Build

Load the CANN toolkit environment, then build with CMake 3.18 or later and a
C++17 compiler. The library links CANN's `acl_rt` and uses saver ABI 2.
Run the TokenSpeed Python runtime with a matching `ts-host` that supports
protocol 3.

```bash
cmake -S tokenspeed-kernel-npu/memory-saver -B /tmp/ascend-saver-build \
  -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/ascend-saver-build
export TOKENSPEED_NPU_MEMORY_SAVER_LIBRARY=/tmp/ascend-saver-build/libascend_memory_saver.so
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
```

The callback selects a tag and content policy at allocation. The checked
pause and resume calls use `weights`, `kv_cache`, and `acl_graph` to select
regions.
The backend queries the minimum huge-page allocation granularity, reserves
a virtual address, and maps a physical allocation. On the target A2 and CANN 9.0.0,
the minimum huge-page granularity measured 2 MiB and same-device
`aclrtMapMem` provided read/write access. An explicit `aclrtMemSetAccess`
returned `507899` in the device probe, so this same-device path relies on
the access granted by the map.
`pause` snapshots backup segments into pinned host memory before releasing
their physical pages. Discard segments release physical pages directly.
`resume` maps new physical pages at the reserved addresses and restores any
snapshots. The caller reinitializes discarded contents before reading them.

## Pool and storage policy

Each managed pool has a release tag, a retention policy, and an allocation
scope. Pools with the same release tag can have separate scopes or policies.
Release and resume act on all managed pools for that tag. `cpu_backup` copies
contents to pinned host memory. `discard` restores the virtual mapping with
undefined contents. `remain` uses ordinary PTA allocation and creates no
saver pool.

A joint capture uses one pool and one retention policy. Separately captured
graphs can explicitly reuse that pool or select independent scopes. A request
to share a pool must match the registered scope and policy. The graph handle
does not by itself define which storage is shared. Input buffers and graph
metadata allocated outside capture follow their own allocation scopes.

The decode graph private pool uses discard for audited BF16 MHA model and
drafter combinations with the greedy sampler. The target and any draft
model must declare that persistent state lives outside capture. The serving
runtime uses Host backup for other captures. It discards the KV arena.
The dense BF16 Qwen3 target can use weight discard with a complete in-place
disk update. Greedy chain drafting with `retain` backs up draft-owned weights.
The storage policies require the following conditions.

| Storage | Target | Frozen draft | Condition |
| --- | --- | --- | --- |
| Weights | `discard` | `remain` or `cpu_backup` | Reload all discarded weights in place before serving resumes. |
| RoPE tables | `cpu_backup` | `remain` or `cpu_backup` | Preserve these non-checkpoint buffers explicitly. Aliased storage has one policy. |
| Activation storage | `discard` | `discard` | Verify replay fully writes each allocation before its first read. Preserve joint pool contents while any of its allocations need backup. |
| KV arena | `discard` | `discard` | Reinitialize the arena before serving resumes. |

The latest-source serving regression overwrites every remapped graph-pool block
before each batch bucket's first replay after wake. It checks tokens against a fresh engine,
original graph IDs and addresses, and zero graph snapshot bytes.

Use one device and one ACL context per process. Before pause or resume,
the caller stops launches and new allocations and drains outstanding
operations. Keep forward execution stopped if a checked transition fails.
An earlier segment in the batch may already have changed state. Retrying a
checked transition reuses the resources recorded for each segment. A fault
from the void free callback is terminal for the worker. PTA relinquishes
ownership before invoking this callback, so a known segment enters the
releasing state even if context or device validation fails.
An allocation failure remains retryable when its rollback releases every
resource. A rollback failure retains the segment for cleanup and latches the
same terminal callback fault, even when other managed pools remain healthy.
`saver_collect_failed()` can finish releasing the segment on the original
device and context. Cleanup preserves the terminal fault and requires worker
restart before reuse. An explicit `saver_release()` that fails context or
device validation preserves the segment's phase for the caller.

`saver_stats()` reports physical, mapped, reserved and host bytes and the
state of each segment. Paused allocations retain their virtual reservations.
Final release removes the physical pages or host snapshot still owned by a
segment and then frees its reservation.

For host-only state and fault-injection tests, use a separate build directory.

```bash
cmake -S tokenspeed-kernel-npu/memory-saver -B /tmp/ascend-saver-test \
  -DSAVER_FAKE_ACL=ON
cmake --build /tmp/ascend-saver-test
ctest --test-dir /tmp/ascend-saver-test --output-on-failure
```

See the [hardware test instructions](../../test/runtime/ascend_memory_saver/README.md)
for real-device VMM and GSM8K round trips. The
[validation report](../../docs/serving/ascend-memory-saver-test-report.md)
records root-accepted latest functional qualification separately from earlier
N18 timing and historical coverage. Host-only
fake-ACL tests do not establish device correctness. ACL Graph executables and
HCCL stay resident; only managed physical backing is released.
