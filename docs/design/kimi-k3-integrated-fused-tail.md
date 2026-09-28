# K3 MoE tail

The NVIDIA K3 TP8/EP1 path uses deferred expert finalize, all-reduce and
RMSNorm for routed latents, followed by the standard sharded up-projection
and shared all-reduce. Attention DP/CP configurations remain separate.

## Routing

| Kernel M | Routed stage | Shared output and up-projection |
| --- | --- | --- |
| 0 | No MoE work | Return the residual |
| 1..32 in graph phase | Existing multicast latent tail, when available | Existing small-tail implementation |
| 33..1024 | FlashInfer BT finalize/all-reduce/RMSNorm | Sharded projection and shared all-reduce |
| 1025..8192 | Vendored HT finalize/all-reduce/RMSNorm | Sharded projection and shared all-reduce |
| Other forwards | Existing separate reduction | Existing sharded or replicated projection path |

BT/HT require the K3 H7168/latent3584/top-k16 layout, attention DP1/CP1,
routed RMSNorm, sharded up-projection and deferred-finalize expert output.
These are static model/backend requirements. Ranks agree on communication
capabilities before allocating either routed workspace. A missing capability
or in-range workspace fails explicitly. PDL remains required by BT/HT.

The token ranges use the actual kernel M, including CUDA-graph padding when
enabled. The graph capture limit does not set the BT/HT workspace capacity.
The small path retains its existing split-shared-RS behavior. Other layouts
and token counts retain the separate-reduce path.

## Second stage

BT/HT produce replicated normalized BF16 routed latents `[M,3584]`. Each rank
owns one `[896,3584]` up-projection weight shard and a full-width rank-local
shared-expert partial `[M,7168]`.

The second stage uses the sharded sequence from main:

1. Add the residual to this rank's owning columns of the shared partial.
2. Apply `addmm_` to those columns using the routed latent and local weight.
3. All-reduce the full shared partial over the MoE TP/EP group.

The projection shards occupy disjoint columns, so this all-reduce also
assembles the full up-projected output and includes the residual once per
column. Preserve the BF16 addition and `addmm_` boundaries of that sequence.

Shared producers use ordinary output tensors. This stage has no fused
shared-RS/GEMM kernels, dedicated symmetric output pool or per-layer compiled
GEMM plans. The small multicast tail keeps its own existing resources.

## Implementation boundaries

`python/tokenspeed/runtime/models/kimi_k3_comm.py` owns routing and workspace
lifetimes. Runtime accesses third-party kernels through `tokenspeed-kernel`.
The BT adapter imports FlashInfer; HT uses the vendored native-H3584 kernel
until upstream support is released. Compatibility uses available APIs and
hardware capabilities, without an exact FlashInfer version gate.

The HT device source derives from FlashInfer v0.6.18, commit
`69ff11fc4954396d98326656dc85debd2223f637`, under its original Apache-2.0
license. Original notices remain in source. Its predicated loads/stores
support the partial 56-pack TP8 reduction shard.
