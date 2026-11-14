# Ascend ACL Graph Memory Saver design

## Requirements

A colocated trainer needs to reclaim device memory between rollout windows.
TokenSpeed must drain serving work, release selected physical allocations, and
restore serving in the same workers. The next forward must reuse the captured
executable and the device virtual addresses it recorded. The trainer must be
able to load a complete new weight version before any rank resumes generation.

- Keep the release and resume calls, tag selection, and checkpoint version
  contract shared with CUDA. Each backend owns its allocator, graph API, and
  supported model and communication paths.
- Give weights, KV cache, and decode graph pools independent lifetimes. Back up
  bytes that must survive and discard bytes the runtime can safely recreate.
- Use the shared memory transition protocol to validate release and resume
  across the serving ranks. Admit requests again after every rank finishes
  restoring storage and commits any required weight update.
- Account for resident memory. The managed graph region covers the decode
  graph's private allocation pool. Graph handles, persistent input and metadata
  buffers, and ordinary communication workspaces have separate lifetimes.

See the [Ascend user guide](../serving/ascend-memory-saver.md) for setup and the
control requests an orchestrator sends.

The integration uses the already-upstream launcher failure notification and the
backend-specific packaging of `torch_memory_saver`. CUDA/ROCm retain their
existing `0.0.9.post1` dependency and allocator paths. The feature selects the
Ascend allocator only for NPU workers that enable the saver.
[Rank failure notification #3306](https://github.com/lightseekorg/tokenspeed/pull/3306)
· [GPU dependency isolation #3310](https://github.com/lightseekorg/tokenspeed/pull/3310)

The shared protocol dependency is
[PR #3397](https://github.com/lightseekorg/tokenspeed/pull/3397), pinned at
`2e0975c31d88661b2fe110e0fac860594ed62479` for this integration. The PR is
open and Ready for Review, not merged. Its request matching, cache publication, and weight commit contracts remain
shared. Checked Ascend Saver serving selects protocol 3 and rejects an
older local host; CUDA and ROCm retain the protocol 2 default.

## CUDA reference flow

TokenSpeed configures the `torch_memory_saver` preload hook before starting
CUDA workers. Weight and KV construction enter tagged allocator regions.
Decode graph capture goes through `torch_memory_saver.cuda_graph` so the
private pool receives the `cuda_graph` tag with discard semantics.
The allocator selects pools by tag and backup policy. It also recognizes
managed pointers freed outside the original allocation region.
[TokenSpeed allocator adapter](../../python/tokenspeed/runtime/utils/torch_memory_saver_adapter.py)
· [Reference allocator implementation](https://github.com/fzyzcjy/torch_memory_saver/tree/7078bc36ebc926ced1cd6d314b869c0a0d3d1f18)

1. A managed allocation reserves a virtual address and maps device pages with
   CUDA virtual memory management (VMM).
2. Once requests and device work have drained, pause copies backup allocations
   to host memory, then unmaps and releases physical pages for the selected
   tags. The address reservations stay live.
3. Resume maps new physical pages at those same addresses. It restores saved
   contents for backup allocations. Replay recreates discardable graph
   temporaries. TokenSpeed clears discarded KV before use.
4. After a weight discard, the configured Model Updater SDK must replace the
   complete checkpoint in place. The control plane agrees on admission, load,
   and commit before it allows KV wake. A compatible NCCL build can separately
   suspend and resume communicator memory under the `nccl` tag.
   [Weight update contract](../serving/weight-memory.md)
   · [TokenSpeed NCCL integration](../../python/tokenspeed/runtime/distributed/nccl_suspend.py)
   · [NCCL communicator API](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2297/user-guide/docs/api/comms.html#ncclcommsuspend)

## Ascend allocation and capture

Each worker selects one NPU and its ACL context. The adapter constructs
`torch_npu` pluggable allocators and installs them only in explicit PTA
`MemPool` scopes. At capture, it passes the owned pool to ACL Graph. Ordinary
allocations use PTA's default allocator. The Ascend path uses the native
`libascend_memory_saver.so` backend. Its checked interface is ABI 2.
[Python adapter](../../python/tokenspeed/runtime/utils/ascend_memory_saver_adapter.py)
· [Native backend](../../tokenspeed-kernel-npu/memory-saver/README.md)

| Storage | Policy and lifetime |
| --- | --- |
| Target weights | The chosen `cpu_backup` policy copies the managed region to host memory. With `discard`, the model must enumerate immutable buffers to preserve, such as rotary tables, and provide a complete in-place weight update contract. |
| Independent draft weights | `retain` preserves them through CPU backup. `refresh` puts them in the same update version as the target. The target owns shared parameters, such as tied embeddings or output weights. |
| KV arena | Discard physical pages. Clear resumed KV and synchronize the zeroing before admission reopens. Release clears scheduler-side device cache references after all ranks agree they are ready. |
| Decode graph private pool | Discard for audited BF16 MHA paths with greedy sampling and explicit model and draft capability. Other supported capture paths back up the pool to host memory. A shared pool has one compatible retention policy. |
| Persistent graph inputs, metadata, and communication workspaces | Keep their owning allocations resident, or restore them through their own explicit memory region. Captured HCCL operations continue to use the live communicator and remapped addresses. |

A managed pool is identified by its release tag, retention policy, and
allocation scope. Several pools can share a tag. Joint capture uses one pool.
Separate graphs can share a compatible pool or use independent scopes. Pool
identity and allocator ownership remain valid even when capture ends and frees
occur outside the original allocation scope.

The native backend reserves an address range with `aclrtReserveMemAddress`,
obtains physical pages with `aclrtMallocPhysical`, and maps them with
`aclrtMapMem`. Pause optionally copies contents to pinned host memory and
calls `aclrtUnmapMem` and `aclrtFreePhysical`. Resume creates physical backing
at the original address and restores saved bytes. Final free also releases the
address reservation. [Ascend VMM and allocator details](../../tokenspeed-kernel-npu/memory-saver/README.md)

Graph discard depends on a write-before-read invariant for graph-private
temporary bytes on every replay path, including padded batches and draft
verification. The execution graph, owning pool, and recorded pointers survive
release and resume. This releases private tensor storage, not the ACL Graph
executable. The latest-source real-device regression overwrites restored graph pages
before first replay, checks the original graph and addresses, compares tokens
against a fresh worker, and checks that the graph pool used zero backup bytes.
[Decode graph data contract](unified_path.md)
· [Real-device acceptance procedures](../../test/runtime/ascend_memory_saver/README.md)

## Ascend pause, update, and resume

1. Release the managed regions through the shared memory transition protocol.
   On Ascend the coordination group is the single-node TP replica. The
   native saver releases the local allocations for each requested tag and
   preserves their virtual addresses. The release of `kv_cache` discards
   device pages after the scheduler clears its L1 index. Releasing `acl_graph`
   also selects resident `kv_cache`, for either graph-pool policy. Target and
   draft share this device arena. Graph-only wake leaves KV asleep.
2. The trainer publishes the next checkpoint. Resume `weights` and
   `acl_graph`, while `kv_cache` remains released. Device addresses are valid
   again. Discarded weights still require a checkpoint.
3. Load a complete target checkpoint from disk or through the Model Updater
   SDK. Ascend updates require `flush_cache=true`: the host deletes the old L3
   namespace and clears L1/L2 before dispatching parameter writes. The SDK also
   supports an eligible independent draft in the same
   version. Every rank verifies required source names and source tensor
   shapes. The loaders shard and fuse weights into their current storage.
   All ranks agree that coverage passed before any model rebuilds
   weight-derived state. The replica commits after all ranks finish and
   any requested cache namespace change succeeds.

   Disk admission rejects cached or non-direct readers and mixed AUTO file
   families before cache invalidation or a weight transaction. A format
   refusal preserves the existing state for a corrected request. The admitted
   `CheckpointFileSelection` is passed as the required fourth `weight_files`
   argument to `_prepare_weights`; the NPU reload does not rescan a different
   file family. The caller publishes immutable checkpoint directories; this check
   does not snapshot
   files or prevent external writers from changing them during a load.
4. Resume `kv_cache`. Each rank clears its arena and completes wake. The
   scheduler can then dispatch new work using the original ACL Graph.

An invalid request rejected during preflight leaves the transition unapplied.
A device transition failure or a checkpoint failure after the update begins
keeps the replica unavailable. The launcher supervises fatal rank failures.
[Control and data plane ownership](event-loop.md)
· [Weight update and failure semantics](../serving/weight-memory.md)

## CUDA and Ascend user interface

| Operation | NVIDIA CUDA | Ascend ACL Graph |
| --- | --- | --- |
| Enable | `--enable-memory-saver` and `--weight-memory-policy` with `cpu_backup` or `discard` | Same flags. Use a matching `ts-host`, supported PTA build, and the Ascend saver library. |
| Release or resume | `release_memory_occupation` and `resume_memory_occupation` | Same calls and optional `tags`. The SMG direct gateway exposes them under `/v1/rl` with `--enable-rl`. |
| Select managed memory | `kv_cache`, `weights`, `cuda_graph`, and optional `nccl` | `kv_cache`, `weights`, and `acl_graph`. HCCL and its communication workspaces stay resident. |
| KV on graph sleep | Follows the requested release tags. | `acl_graph` release also discards device KV. Graph-only wake keeps KV asleep until its explicit wake. |
| Recover captured execution | Remap at the original addresses and replay the original CUDA Graph. | Remap at the original addresses and replay the original ACL Graph. Audited pools discard and recreate temporary bytes. Other supported pools restore them from host. |
| Replace discarded target weights | `update_weights_from_mooncake` via a configured Model Updater SDK. | Same SDK request, or `update_weights_from_disk` for a complete eligible target checkpoint. |
| Receive trainer broadcasts | The NCCL update group can stream compatible updates when weights use CPU backup. | Use disk or the Model Updater SDK with Ascend Memory Saver. |
| Handle independent draft weights | `--model-update-draft-weights` with `retain` or `refresh` | Same SDK policies. Disk reload uses `retain` when a draft model is present. |
| Coordinate memory and weights | Protocol 2 keeps the existing memory-transition path and coordinates weight-load admission and commit. | Protocol 3 coordinates release and resume across the replica. Ascend Memory Saver requires a matching host and runtime. |
| Handle a fatal rank | Launcher receives the native failure notification and reaps local ranks. | Same launcher mechanism. PTA handles normal HCCL group shutdown. |

## Validation scope

The hardware baseline is Linux aarch64, Atlas A2 (Ascend 910B3), CANN 9.0.0,
PyTorch `2.9.0+cpu`, and PTA 2.9.0-730 (`9d40ac3`, July 30) plus
the [six official backports](../../tokenspeed-kernel-npu/memory-saver/README.md#pta-options).
This requires no additional PTA patches or system-library replacements. The primary
deployment target is single-node aggregate serving with TP1, TP2, TP4, or
TP8. Startup requires the attention, dense, MoE, and vision TP groups to
span the whole worker group. Serving collectives stay inside that replica.
Dense BF16 Qwen3 has the complete target discard contract. Qwen3 with
fixed-width chain MTP and default paged MHA checks supported verify widths
before initializing distributed groups. The A2 kernel registrations support
widths up to five for this route.

Graph-pool support and weight-update support depend on different model
capabilities. TP1 Llama EAGLE3, DFLASH, and vanilla DSPARK have audited graph
discard combinations with CPU-backed weights. The block drafters use full
attention and verify widths from two to four. Complete target and draft refresh
through a local Mooncake TCP store was exercised with Qwen3 and a synthetic MTP
draft. Production SDK providers, other Ascend devices or CANN releases, and
full native draft weight refresh require their own validation. A TP replica
spanning nodes requires separate Ascend enablement and hardware validation,
including deployment supervision across hosts.

TP4 and TP8 no-pause controls also showed token variation. Those baseline
counterexamples do not exclude every possible Saver contribution. Numerical
attribution and HCCL failure behavior remain separately tracked. Latest-source
functional qualification is root accepted; earlier N18 timing components and
historical results remain separately identified in the report.
[Hardware validation report](../serving/ascend-memory-saver-test-report.md)
· [Hardware acceptance procedure](../../test/runtime/ascend_memory_saver/README.md)

HCCL TP2 subgroup destruction and graph recapture were measured in a separate
prototype. The supported path keeps HCCL resident and replays the existing graph.
Faulted communicator destruction and generic HCCL watchdog failures remain
PTA issues. Measurements from graph-only recapture and garbage-collection
experiments describe those configurations.
