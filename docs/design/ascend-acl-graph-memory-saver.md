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
continue to use PTA's allocator. The [native backend](../../tokenspeed-kernel-npu/memory-saver/README.md)
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
| Decode graph private pool | Audited BF16 MHA/greedy target and draft paths discard; other supported captures use CPU backup. Executables survive. |
| Communication and unmanaged storage | HCCL groups, communication workspaces and allocations outside managed regions stay resident. |

Pool identity includes tag, retention policy and allocation scope. Several pools
may share a release tag. Joint capture uses one compatible policy; separately
captured graphs may explicitly share a compatible pool or own independent
scopes. A graph handle alone does not define shared storage. Pool ownership
survives capture and out-of-scope frees.

Named persistent-region aliases, including aliases from different adapter
instances, reuse the active pool under the shared worker lock. Other nested
allocation scopes are invalid. Communication initialization is deferred until
the outer managed allocation scope exits. Shared workspace growth follows the
same persistent owner as its initial allocation.

Graph discard requires every replay path to write each temporary byte before
reading it, including padded batches and draft verification. Persistent state
belongs outside capture. Address/graph identity alone cannot prove this:
acceptance poisons remapped pages before each bucket's first replay, compares
outputs against a fresh worker and checks zero graph snapshot bytes.
[Decode data contract](unified_path.md) ·
[Direct smoke procedures](../../test/runtime/ascend_memory_saver/README.md)

## Transition and checkpoint invariants

1. The shared protocol drains work and coordinates release across the
   single-node TP replica. Releasing `acl_graph` also releases resident
   `kv_cache`, under either graph policy. Graph-only wake leaves KV asleep.
2. Resume `weights` and `acl_graph` while KV stays released. Remapping
   discarded parameters does not make them valid.
3. A changed-weight update requires `flush_cache=true`: clear L1/L2 and delete
   the old L3 namespace before writes. Discard requires complete source coverage
   and source-shape validation before rebuilding derived state. `cpu_backup`
   reuses the CUDA disk loader and permits partial replacement; omitted weights
   retain their restored bytes. Loaders shard/fuse into existing storage;
   addresses, shapes, strides and dtypes stay unchanged. Every rank agrees on
   admission and final commit, including cache namespace publication, before KV
   can wake. Complete discard loads also converge errors before post-load hooks.
4. Resume KV, clear its arena, complete all-rank wake, then admit requests
   using the original ACL Graph.

Disk preflight selects one supported direct-reader file family before cache
invalidation or a weight transaction. The admitted `CheckpointFileSelection`
is the required fourth `weight_files` argument to `_prepare_weights`; reload
must not rescan into a different family. Checkpoint directories must remain
immutable: admission does not snapshot files or protect against external writes.

Partial `cpu_backup` disk updates retain the
[common loader limits](../serving/ascend-memory-saver.md#partial-disk-update-limits).

An admission refusal leaves the operation unapplied and permits a corrected
retry. Once a weight transaction begins, failure keeps the whole replica
unavailable and requires restart. The launcher supervises fatal rank failures. [Control-plane ownership](event-loop.md) ·
[Shared weight and failure contract](../serving/weight-memory.md)

## Replay and host boundaries

Ascend MHA target replay updates FIA host sequence lengths from live decode
metadata, including verify-width padding. Speculative attention is causal and
aligned to the cache end. Captured native draft decode uses device lengths,
including the accepted frontier computed inside the graph; eager decode uses
normal kernel selection. Discard replay rewrites Q, workspace and outputs.
FIA descriptors use live host lengths while tasks consume device data in graph
order. Shared masks, sampler inputs and attention metadata stay resident.

The NPU launcher opts in to restoring the original main thread-state after
Python log cleanup, before native exit handlers run. This also applies with
Memory Saver disabled. Other launchers retain the default-false detached exit.
Build and ABI details belong to the [native README](../../tokenspeed-kernel-npu/memory-saver/README.md).

The common transition interface is owned by
[PR #3397](https://github.com/lightseekorg/tokenspeed/pull/3397).
Ascend uses protocol 3 and `acl_graph`; its HCCL allocations remain resident.
The [validation report](../serving/ascend-memory-saver-test-report.md) binds
results to their executed source and keeps later-source qualification pending.
