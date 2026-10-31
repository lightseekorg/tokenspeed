# Weight memory policies

Enable memory saving with an explicit `--weight-memory-policy cpu_backup` or
`--weight-memory-policy discard`. Both use the existing memory tags and control
routes. Rebuild the Python runtime, native host and kernel package together;
the memory-saving startup handshake rejects native hosts without the
replica-wide update-commit protocol (version 2: load admission and commit).

## Policy and model contract

`cpu_backup` copies weight-region allocations to host memory and restores their
contents on wake. `discard` releases target weight pages without copying old
parameters to CPU. Remapping their original addresses does not make the model
usable. A complete new version must arrive through the configured Model Updater
SDK (`update_weights_from_mooncake`). Disk, tensor and distributed updates are
not supported under discard.

Discard requires an explicitly opted-in model that provides:

- every canonical checkpoint source name required on this rank;
- an explicit inventory of immutable tensors that must survive lost pages;
- in-place reconstruction of all state derived from the new weights.

Only immutable tensors, such as rotary caches, receive small CPU snapshots.
The inventory cannot alias parameter storage. Parameter addresses, shapes,
strides and dtypes must remain unchanged by the update. Models without this
contract, quantized weights and online expert rebalancing are rejected. An
opting-in model must also reject padded or transformed expert layouts unless
it can reconstruct all padding and transformations after an update.

`--model-update-draft-weights retain` preserves frozen draft allocations using
CPU backup. Target/draft shared embeddings or output weights keep the target's
discard policy and receive the new target version. `refresh` instead requires
complete target **and** draft streams before either model derives new state.
A trainer that does not publish draft weights must use `retain`.

DeepEP and attention head-TP exchange allocations are deferred until the
weight allocation region has exited. Head-TP peer address tables and Lamport
counters share the communication lifetime; preserving only model tensors does
not preserve this backend-owned state. This avoids nested CUDA memory pools and
keeps communication storage outside the weight region. They remain resident,
including across discard; their memory must be included in the training headroom. NCCL release is independent and
optional. No NCCL suspend feature is required when its tag is omitted.

## Allocator runtime requirements

The allocator must recognize managed pointers when they are freed outside their
allocation region, and must separate pools by both tag and CPU-backup policy.
Some `0.0.9.post1` binaries lack the former fix. Validate clean process shutdown
as well as sleep/wake; a successful graph replay alone does not establish this.
The upstream implementation at
[`7078bc36`](https://github.com/fzyzcjy/torch_memory_saver/commit/7078bc36ebc926ced1cd6d314b869c0a0d3d1f18)
contains both behaviors and can be built against the deployment's CUDA toolkit.
Record the exact source and wheel hashes when using a local build.

When CUDA graphs operate on remapped storage, set `NCCL_GRAPH_REGISTER=0` before
launch so NCCL does not retain graph registrations across remapping. This does
not suspend NCCL or release its communication buffers.

## One update cycle

1. Stop admissions and drain running work on every serving replica.
2. Release `kv_cache`, then `weights` and optionally `cuda_graph`. Keep NCCL
   resident unless the deployment explicitly supports suspending it.
3. Train and publish the next complete checkpoint version.
4. Resume the weight and graph tags, leaving `kv_cache` released. This restores
   storage and immutable constants; target parameters remain invalid.
5. Apply the complete SDK update to every prefill/decode engine. Each model
   checks for missing, duplicate, unexpected and unconsumed checkpoint inputs.
   All models validate coverage and all ranks agree on receiving success before
   any derived state is rebuilt. A stream or coverage failure aborts the session
   on every rank, so peers do not enter post-load collectives without it.
6. Every rank confirms the final result, including namespace publication. Only
   this commit makes locally loaded weights usable. A failed load poisons every
   rank, under either policy, including ranks whose own load succeeded;
   restart the deployment rather than attempting a partial wake. Before the
   update's cache flush and before any rank loads, all ranks agree that the
   update can start. If any rank refuses (weights still released, KV still
   resident under discard, a non-SDK update under discard), no rank flushes or
   loads; if the flush is refused, no rank loads. Either way the weights stay
   untouched and the update may be retried.
7. Resume `kv_cache`, then allow new rollouts. Remapped KV pages are cleared
   and re-registered for transfer; old request history is not CPU-backed up.

A request to resume all regions immediately after discard fails: the new
weights have not been committed yet. A weight update while KV is resident is
also rejected under discard. Keep the orchestrator paused on any failed
response, including a failure in a different serving replica.

## Validation before enabling in a training job

CPU tests cover state transitions, failure handling, exact checkpoint coverage,
constant restoration and communication-allocation scoping. They do not measure
GPU memory reclamation, graph replay correctness or cross-node transport.

First compare the same small, complete update under `cpu_backup` and `discard`.
Measure GPU usage before release, after release and after resume, and host-memory
usage during release. Account for frozen draft backup and resident communication
buffers separately. Confirm changed target weights and preserved frozen drafts,
finite aligned log probabilities, stable graph addresses, and no growth across
at least two update cycles. Inject a missing chunk and a failed rank: neither
KV wake nor dispatch may succeed. Finally validate the full colocated training
and rollout job at its intended parallelism and context length.
