# Communication kernels

The runtime chooses process groups, prepares capacity before cache sizing and
capture, and orders producers and consumers on streams. The kernel package owns
transport selection, peer addresses, symmetric storage, synchronization, and
fused device work. Model code supplies tensors and weights through communication
operations; it does not own Iris heaps or import a hardware kernel package.
Selection depends on row count and consumer capability across prefill, decode,
and mixed batches; the workspace boundaries follow the communication protocols.

The public operations in `__init__.py` and the existing `triton.py` adapter
keep optional implementations behind the kernel boundary. An optional fusion
returns `None` before launching any collective when it cannot serve a call.
The caller then uses its ordinary reduction and compute operations. Every rank
must agree on that choice. Kernel or transport failures propagate.

## Iris layout

All Iris implementation code lives in this directory:

| Module | Responsibility |
| --- | --- |
| `iris.py` | Shared heap lifetime, group peer maps, capacity reuse, ordinary all-reduce dispatch, and RS/AG and RMSNorm adapters. |
| `_iris/__init__.py` | Optional Iris imports under the TokenSpeed Triton redirect. |
| `_iris/all_reduce.py` | Staged, packed pull, two-stage, and Lamport protocols; shared device synchronization and RMSNorm kernels. |
| `_iris/attnres.py` | AttnRes tensor eligibility, inbox ownership, generations, and fused reduction/epilogue using precomputed history partials. |
| `_iris/row_sharded.py` | Row-sharded attention/MoE pipelines, their shared result workspace, and their matching reduce/gather kernels. |

`IrisAllReduce` composes optional `IrisAttnResWorkspace` and
`IrisRowShardedWorkspace` instances. Each fusion owns its allocation requirements
and launch contract. The row-sharded workspace borrows the producer input, reduction
scratch, and entry flags, and owns the result and gather-completion flags.
Attention and MoE deliberately share this workspace because the attention mix
hands its local residual rows to the MoE tail.

Prepare the largest required state first. The shared symmetric heap cannot grow
after peer mappings or graphs reference it. Reuse requires the same group,
rank, device, dtype, sufficient capacities, and compatible Lamport policy.
AttnRes reuses any prepared workspace that fits the call; standalone lazy setup
reserves its full row window. Group-local peer maps are distinct from Iris's
world heap map.

## Output ownership

| Operation | Result lifetime |
| --- | --- |
| Ordinary Iris all-reduce | Writes the required `out=` tensor. Passing the input reduces in place; disjoint output storage preserves the input. Unaligned destinations use element stores in the same collective, with no temporary or copy-back. |
| Producer-direct all-reduce | Producers borrow consecutive symmetric input views. Reduction preserves those views and returns newly allocated local outputs, which survive input reuse. |
| Fused AttnRes | Returns owned normalized activations and accumulated residuals; the inbox remains private. |
| `attention_reduce_mix` | Returns owned local residual rows and a borrowed replicated activation. |
| `moe_reduce_project` | Returns a borrowed replicated result in the same row-sharded workspace. |
| Iris residual RMSNorm | Writes supplied output tensors, allocating only omitted outputs. |
| Iris CCL RS/AG adapter | Returns a clone by default; `safe=False` borrows its symmetric result until reuse. |

Borrowed row-sharded results must be consumed on the calling stream before the next
attention mix or MoE tail reuses them; clone at the call site to retain them.
A replicated residual may exactly alias the shared result: each rank consumes
its rows before publishing replacements. Shifted overlaps, aliased history or
weights, and sharded prefixes overlapping the result are rejected before launch.

Calls sharing a workspace run in order on one stream. Join side-stream producers
before reduction, and finish consumers before reuse. Eager execution, capture,
and replay use the same buffers and device-owned generations. No host-side slot
rotation is frozen into a graph. These rules follow the
[device execution contract](../../../../../docs/design/event-loop.md) and
[unified graph contract](../../../../../docs/design/unified_path.md).

## Iris kernels and variants

The table describes device behavior; runtime policy may select a narrower token
window than a kernel supports. Batch-varying counts and derived grid sizes stay
runtime values. Lamport's payload size determines only its grid of complete tiles.

| Kernel or adapter | Behavior |
| --- | --- |
| `iris_stage_one_shot_allreduce_kernel` | Stages each tile into rotating symmetric slots, waits for peer publication, sums in FP32, and writes the selected destination. Per-tile epochs protect slot reuse. |
| `iris_reduce_symmetric_gluon_kernel` | Pulls packed producer outputs from every peer, sums in FP32, and writes local results. Entry and exit synchronization protect the reusable input. |
| `iris_reduce_symmetric_two_stage_gluon_kernel` | Each rank reduces one packed partition into symmetric scratch, then gathers all reduced partitions. Ordinary staged calls also wait at exit before their next staging copy. |
| `lamport_all_reduce_bf16` | Pushes BF16 tiles into three mailbox generations and polls peer payloads directly. It normalizes negative zero to reserve the readiness sentinel, sums in ascending rank order, and clears consumed tiles. |
| `iris_push_one_shot_allreduce_residual_attnres_gluon_kernel` | Pushes attention rows into two-slot inboxes, reduces in rank order, adds the residual, combines the historical AttnRes partials, and applies output RMSNorm. |
| `iris_attention_reduce_scatter_gluon_kernel` | Reduces each rank's consecutive attention rows, rounds to BF16, and optionally adds the replicated residual to form local prefix rows. |
| `iris_attention_mix_push_gluon_kernel` | Mixes local prefixes with history, applies output RMSNorm, and pushes each rank's normalized rows to every peer. |
| `iris_attention_push_gather_gluon_kernel` | Gathers already mixed rows when a separate AttnRes mixer is selected. |
| `iris_moe_reduce_scatter_gluon_kernel` | Reduces routed and shared expert partials into consecutive local-row scratch. |
| `iris_moe_add_push_gather_gluon_kernel` | Adds the local routed projection, shared reduction, and residual in FP32, rounds to BF16, and pushes complete rows to every peer. |
| `iris_allreduce_residual_rmsnorm_kernel` | Pulls peer contributions per row, adds the residual and normalizes in FP32, and writes residual and normalized outputs. |
| `iris_allreduce_residual_rmsnorm_kernel_persistent` | Performs the same RMSNorm computation with a bounded grid that strides over rows. |
| `IrisRSAG` | Stages uniform BF16 row shards and invokes Iris CCL reduce-scatter or all-gather. This adapter requires the full world group; production Triton RS/AG has its own implementation. |

The ordinary staged path supports tails and uses the two-stage gfx950 path for
partitionable TP4/TP8 payloads, with the existing tuned TP4 one-shot override.
Producer-direct reductions require gfx950, TP2/TP4/TP8, BF16/FP16/FP32, and packed
alignment of the payload. Their size thresholds choose one- or two-stage pull.
Lamport is an explicit opt-in for TP8 BF16 paired widths 3584 and 7168 at 1–6 rows;
other producer-direct calls retain pull.

Fused AttnRes supports node-local TP8 BF16 width 7168 through 16 rows. It consumes
FP32 historical max, exponential sum, and weighted-sum partials. Reduction,
residual addition, mixture, and normalization retain their existing BF16 rounding
boundaries. Row-sharded fusions use TP8, positive row counts divisible by eight,
hidden width 7168, and MoE latent width 3584. Attention supports 0–11 history
blocks and selects either fused mix/gather or separate mix and gather. MoE
optionally normalizes the routed rows before the local projection.

The row reduce and its matching gather form one protocol. Gather completion
orders every rank's producer reads before reuse; the reduce must not be used
alone as a complete collective. Barrier flags and inbox generations are scoped
to their protocol so another operation cannot satisfy a pending wait.

## Other communication solutions

| Solution | Behavior |
| --- | --- |
| `nccl.py` | NCCL/RCCL bindings for ordinary collectives and backend fallbacks. |
| `triton.py` RS/AG | Reduces contiguous token shards and gathers them through PyTorch symmetric memory, including uneven token counts. NVIDIA also has a multimem variant and an inner-dimension gather. |
| `triton.py` DP sampling | Swaps logits to their sampling owners and gathers predicted tokens, acceptance indices, and lengths back into persistent result buffers. |
| `flashinfer.py` | Adapts custom all-reduce and its buffer/graph registration lifecycle. |
| `trtllm.py` | Fuses all-reduce or reduce-scatter with residual/RMSNorm, latent-lane normalization, or AttnRes; all-gather can normalize Q/KV projections. |
| `multimem.py` | Prepares multicast storage, stages projection lanes, and reduces them in place. |
| `allreduce_fusion.py` | Selects routed reduction/RMSNorm and expert-finalization patterns with prepared workspaces. |
| `deep_ep.py` | Owns prepared expert dispatch/combine buffers and the normal and low-latency transfer variants. |
| `cuda_lamport.py` | Exchanges BF16 payloads using packet or chunk protocols; see its [contract](cuda_lamport.md). |

## Validation

The existing AMD suites cover numerical results, prepared-buffer ownership,
unaligned destinations, subgroup mapping, slot/epoch reuse under rank skew,
changing row counts, compilation reuse, and graph capture/replay:

```bash
python -m pytest -q tokenspeed-kernel/test/amd/ops/test_iris_*.py
```

The tests spawn their own workers. The complete matrix needs eight gfx950 GPUs
and the optional Iris and AMD kernel dependencies.
