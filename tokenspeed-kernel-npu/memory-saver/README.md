# Ascend memory saver native backend

`libascend_memory_saver.so` provides allocator callbacks and checked
pause/resume for TokenSpeed's NPU adapter. For the required PTA backports,
stack qualification, environment and serving sequence, use the
[Ascend user guide](../../docs/serving/ascend-memory-saver.md).
[Pool/lifecycle invariants](../../docs/design/ascend-acl-graph-memory-saver.md)
are owned by the design document.

## Build and ABI

Load the deployment's CANN toolkit environment. Build with CMake 3.18+ and a
C++17 compiler against its `acl_rt`. The adapter requires saver ABI 2;
Ascend serving also needs a matching runtime/`ts-host` with protocol 3.

```bash
cmake -S tokenspeed-kernel-npu/memory-saver -B /tmp/ascend-saver-build \
  -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/ascend-saver-build
export TOKENSPEED_NPU_MEMORY_SAVER_LIBRARY=/tmp/ascend-saver-build/libascend_memory_saver.so
```

The backend queries minimum huge-page granularity, reserves a virtual range
with `aclrtReserveMemAddress`, then uses `aclrtMallocPhysical`/`aclrtMapMem`.
On the measured A2/CANN 9.0.0 stack, granularity is 2 MiB and same-device mapping
provides read/write access. Explicit `aclrtMemSetAccess` returned `507899`;
this same-device path relies on the map's access. Qualify other stacks separately.
`cpu_backup` snapshots to pinned host memory before unmap/free; `discard`
remaps with undefined contents. Callers must initialize discarded data before
reading it. `remain` uses ordinary PTA allocation and creates no saver pool.

The module-host shim runs the selected Python with
`-m tokenspeed_engine.host`. Install `host.py`, ABI3 `_host`/`_native`, shim and
scheduler from the same checkout on Python 3.12+. Remove stale shadowing native
modules; override `TOKENSPEED_HOST_BINARY` only with the matching shim.

## Checked-operation failure contract

Use one device and ACL context per process. Before pause/resume, stop launches
and allocations and drain outstanding operations. A checked transition can
partially change a batch; keep execution stopped on failure. Retrying uses the
resources recorded per segment.

A void free-callback failure is terminal: PTA already relinquished ownership,
so a known segment enters releasing even if device/context validation fails.
An allocation failure is retryable only if rollback releases every resource.
Failed rollback retains the segment for cleanup and latches the terminal fault,
even if other pools remain healthy. `saver_collect_failed()` can finish cleanup
on the original device/context; it preserves the fault and requires restart
before reuse. An explicit `saver_release()` validation failure instead preserves
the segment's phase for its caller.

`saver_stats()` reports physical, mapped, reserved and host bytes plus segment
state. Paused allocations retain virtual reservations. Final release removes
remaining physical pages/snapshots, then the reservation. These counters are
not driver-free device-memory measurements.

## Host-only verification

Use a separate fake-ACL build; these tests do not establish device correctness.

```bash
cmake -S tokenspeed-kernel-npu/memory-saver -B /tmp/ascend-saver-test \
  -DSAVER_FAKE_ACL=ON
cmake --build /tmp/ascend-saver-test
ctest --test-dir /tmp/ascend-saver-test --output-on-failure
```
