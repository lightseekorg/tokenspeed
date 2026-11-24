# Ascend ACL Graph Memory Saver design

A colocated trainer needs physical device memory between rollout windows.
Serving must drain, release selected backing pages, and restore in the same
workers with the original captured executable and virtual addresses. Admission
reopens only after every rank has restored storage and committed any required
complete weight update. See the [user guide](../serving/ascend-memory-saver.md)
for dependencies, installation, supported configurations and control requests.

## Ownership and retention

The NPU adapter installs PTA pluggable allocators only inside explicit `MemPool`
scopes and passes the owned pool to ACL Graph capture. Ordinary allocations
continue to use PTA's allocator. The [native backend](https://github.com/lightseekorg/tokenspeed/blob/main/tokenspeed-kernel-npu/memory-saver/README.md)
reserves virtual addresses and maps physical pages with ACL VMM. Pause unmaps
and releases physical backing, optionally retaining a pinned-host snapshot;
resume maps new backing at the original addresses. Final free also releases the
reservation. Managed frees must work outside the original allocation scope.

| Storage | Invariant |
| --- | --- |
| Target parameters | `cpu_backup` preserves bytes; explicit `discard` requires a model-owned complete in-place checkpoint contract and an immutable-buffer inventory. |
| Independent draft parameters | `retain` uses CPU backup; `refresh` requires complete target and draft streams in one version. The target alone owns shared parameters. |
| Persistent buffers | `executor_inputs`, `runtime_states`, `attention_state` and `workspace` alias `PoolKey("weights", CPU_BACKUP, "persistent_buffers")`, distinct from parameter-discard pools. Contents and addresses survive either parameter policy. |
| Device KV | Discard and clear on wake; synchronize zeroing before admission. Target and draft share the arena. Scheduler cache references are cleared after release readiness agreement. |
| Decode graph private pool | Default `discard` releases temporary storage; `cpu_backup` preserves its contents in host memory. Executables and reserved virtual addresses survive. |
| Communication and unmanaged storage | HCCL groups, communication workspaces and allocations outside managed regions stay resident. |

Pool identity includes tag, retention policy and allocation scope. Several pools
may share a release tag. Joint capture uses one compatible policy; separately
captured graphs may explicitly share a compatible pool or own independent
scopes. A graph handle alone does not define shared storage. Pool ownership
survives capture and out-of-scope frees.

On Atlas A3, each managed VMM segment is registered with the server-local HCCL
group before a collective can use it. This also covers slices reused by other
operators. Pause deactivates the segment before unmapping. Resume activates the
new physical handle at the original address before replay. The range stays
registered until final free or shutdown, while the communicator is still alive.

Named persistent-region aliases, including aliases from different adapter
instances, reuse the active pool under the shared worker lock. Other nested
allocation scopes are invalid. Communication initialization is deferred until
the outer managed allocation scope exits. Shared workspace growth follows the
same persistent owner as its initial allocation.

The graph-pool policy is independent of the parameter policy. With `discard`, remapped storage has
undefined contents; it is not zero-initialized. Values initialized outside
replay must live outside this pool, in resident storage or a persistent backup
pool. Replay must overwrite every value it reads, including padded batches and
draft verification. Unread allocator padding need not be initialized. A weight
change also invalidates KV and requires rebuilding weight-derived state; a
backup of that old state cannot substitute for rebuilding it.
See the [decode data contract](unified_path.md).

## Transition and checkpoint invariants

1. The shared protocol drains work and coordinates release across the
   replica's memory domain. Releasing `acl_graph` also releases resident
   `kv_cache`. Graph-only wake leaves KV asleep.
2. Resume `weights` and `acl_graph` while KV stays released. Remapping
   discarded parameters does not make them valid.
3. A changed-weight update requires `flush_cache=true`: clear L1/L2 and delete
   the old L3 namespace before writes. Discard requires complete source coverage
   before rebuilding derived state. `cpu_backup` reuses the existing disk/SDK
   loaders and permits partial replacement; omitted weights
   retain their restored bytes. Loaders shard/fuse into existing storage;
   addresses, shapes, strides and dtypes stay unchanged. Every rank agrees on
   admission and final commit, including cache namespace publication, before KV
   can wake. Complete discard loads also converge errors before post-load hooks.
4. Resume KV, clear its arena, complete all-rank wake, then admit requests
   using the original ACL Graph.

Disk updates use the existing checkpoint reader and model loading hooks.
Checkpoint directories must remain immutable during loading: admission does
not snapshot files or protect against external writes. A target-only disk
update cannot restore an independent draft whose weights were also discarded.

Partial `cpu_backup` disk updates retain the
[common loader limits](../serving/ascend-memory-saver.md#partial-disk-update-limits).

An admission refusal leaves the operation unapplied and permits a corrected
retry. Once a weight transaction begins, failure keeps the whole replica
unavailable and requires restart. The launcher supervises fatal rank failures. [Control-plane ownership](event-loop.md) ·
[Shared weight and failure contract](../serving/weight-memory.md)

The SDK adapter retains its existing update and return-value contract. NPU
copies finish before post-load agreement and weight publication. `cpu_backup`
does not add a complete-stream requirement to partial SDK updates.

## Replay and host boundaries

Ascend MHA target replay updates FIA host sequence lengths from live decode
metadata, including verify-width padding. Speculative attention is causal and
aligned to the cache end. Captured native draft decode uses device lengths,
including the accepted frontier computed inside the graph; eager decode uses
normal kernel selection. Replay writes Q, workspace and outputs.
FIA descriptors use live host lengths while tasks consume device data in graph
order. Shared masks, sampler inputs and attention metadata stay resident.

The launcher passes the rank device in the child environment so older hosts
can use the same command line. On NPU the matching host restores the original
main thread-state after Python log cleanup and before native exit handlers
run, including when Memory Saver is disabled.
Before native exit, NPU workers explicitly close process groups: the host skips
Python atexit, where PTA normally performs this cleanup.
Build and ABI details belong to the [native README](https://github.com/lightseekorg/tokenspeed/blob/main/tokenspeed-kernel-npu/memory-saver/README.md).

The runtime and native host in this change provide the complete protocol 3
transition interface. Ascend uses `acl_graph`; its HCCL allocations remain resident.
CUDA and ROCm continue to use protocol 2.

Future HCCL work must investigate releasing communicator-owned buffer backing
while retaining the captured graph, then restoring the buffers at the captured
virtual addresses before replay. This is not implemented by the current
release path; destroying and rebuilding communicators is a separate design.
