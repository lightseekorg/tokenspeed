# Distributed immutable expert weights

Distributed Weight Data Parallelism (DWDP) separates expert-weight ownership
from execution parallelism. Each worker permanently stores a contiguous expert
shard, but runs its own token batch through every logical expert. Attention,
dense layers, and MoE execute locally. Weight owners do not have to execute the
same layer or have requests at the same time.

`Mapping.moe_weight_group` identifies storage peers. It does not replace
attention DP ranks, request routing, P/D rank pins, or an execution group.
`MoELayer._storage_spec` controls allocation and checkpoint slicing;
`MoELayer._spec` describes compute-visible experts. Kernel selection and routing
use the compute specification.

## Loading and memory ownership

The selected kernel preprocessor returns a `PreparedExpertWeights` manifest:
a layout identity, processed expert-major tensors, and an explicit distinction
between streamed weights and small per-expert constants. Non-expert state stays
on the layer. In particular, the loader must preserve the existing layer-wide
input-scale maximum across all logical experts.

A model-owned `MoEPeerWeights` publishes each processed layer immediately and
rebinds its parameters to immutable owned storage. Old processed allocations do
not accumulate beside the full model. Finalization exchanges peer handles and
manifests, reconstructs complete per-expert constants, and binds stable
full-expert tensor views before tuning and cache sizing.

The kernel-package factory dispatches the storage implementation. CUDA driver,
fabric and virtual-memory details remain under `tokenspeed-kernel/thirdparty`.
The CUDA implementation maps each layer's local expert region to permanent
source storage and remote regions to one of two reusable staging pools. Source
pages that straddle expert boundaries receive their missing immutable edge bytes
once at setup. All foreign allocations for one owner share a virtual-address
arena, reducing fabric access-region use.

Physical source storage, both staging pools, full per-expert metadata, kernel
workspace, and maximum-batch activations must be accounted for before sizing KV
cache. Reserved virtual addresses are not physical cache consumption.

## Forward ordering

Weight operations belong on the data plane. The ordinary model forward primes
the first required MoE. Before each MoE, the compute stream waits for its weights;
after all readers are enqueued, the layer records that its slot can be reused
and starts the next layer's prefetch. Copies into a reused slot wait for the
previous reader's release event. This includes the final readers of a forward:
events are not reset at forward boundaries.

Any side-stream weight reader must join the compute stream before release.
A new reader on a different stream waits for the preceding release, including
when the same layer remains resident, so a later slot overwrite covers all readers.
Host run-ahead is allowed. Only startup and collective teardown coordinate with
other weight owners, over a dedicated named process group; ordinary prefetch,
acquire and release do not.

A prefetch is either copy-engine copies or one SM copy kernel on the store's
copy stream; both follow the same slot events. The kernel's tile table is built
at finalize, interleaves peers so every owner is read at once, and lets CTAs
claim tiles so late-starting CTAs do not delay completion. Its CTA count trades
fetch bandwidth against SMs taken from concurrent compute: persistent kernels
sized to every SM stall while copy CTAs occupy SMs.

The event loop receives only `DeviceSpecs.requires_forward_coordination`,
derived from execution groups. It continues using the same scheduler and
submission path. When execution is fully local, it does not gather per-round
DP metadata or dispatch an idle forward for another worker. The executor uses
the same execution-group requirement when accepting forward metadata. Startup cache
capacity agreement remains collective; local cache-copy completion stays local.
Cache-control handling also omits the per-round DP rendezvous when execution
is local and there is no shared L3 namespace. The initial DWDP configuration
rejects shared L3 and online expert rebalancing; Host L2 remains supported.
Paused workers use the same execution-coordination requirement as normal rounds.

Weights are immutable while published. Live weight replacement is rejected.
Teardown drains local reads/copies and establishes peer quiescence before
unmapping source storage. A model exposing peer weights implements
`PeerWeightOwner`; the device handle performs teardown on its forward thread.

## Initial supported configuration

The initial integration supports K3 NVFP4 SiTU on a fabric-connected NVIDIA
prefill instance with local execution groups and a fixed token budget. Model
CUDA graphs are disabled; kernel tuning and graph consumers can use stable
weight views with acquisition/release outside capture. Unsupported model,
layout, transport and graph combinations fail explicitly.

Validation includes cross-node reconstruction, page-boundary handling, complete
expert metadata, idle owners, odd/even queued forwards, bounded staging,
full-model mapping counts, and numerical comparison against resident weights.

Checkpoint loading currently supports auto and safetensors. Dummy/sharded-state
loading and memory-saver sleep/wake are rejected before model allocation.
All owners must remain available for the instance lifetime; deployment must
terminate the instance on worker failure. Independent forwarding does not
provide owner replacement or recovery after an owner fails.
