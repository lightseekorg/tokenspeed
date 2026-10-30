# TokenSpeed-Kernel-NPU

TokenSpeed-Kernel-NPU contains the Ascend-specific operators used by
TokenSpeed. Keeping these implementations in a standalone package mirrors the
AMD package layout and keeps the TokenSpeed runtime vendor-neutral.

The initial Ascend path provides paged MHA, RMSNorm, Q/K RMSNorm, rotary
embedding, and the Triton-Ascend import adapter required by CANN 9.0.0.

For development from this repository:

```bash
test/ci_system/install_triton_ascend.sh
```

The validated stack is CANN 9.0.0, PyTorch 2.9.0, `torch_npu` 2.9.0.post2,
Transformers 5.12.0, Triton 3.2.0, and Triton-Ascend 3.2.1. The setup script
records every Python package mutation needed by this Ascend path, including the
`apache-tvm-ffi==0.1.13` build dependency and editable install of this package.

TokenSpeed applications should continue to import operators from
`tokenspeed-kernel`; it owns registration and dispatch to this package.

The BF16 local-expert chain keeps FP32 router coefficients through
FinalizeRouting. Expert GEMM inputs, weights and outputs remain BF16; routing
coefficients are not rounded to BF16 before the weighted sum.

The runtime's unclamped `SiluAndMul` uses the fused NPU SwiGLU operator through
`tokenspeed-kernel`, including shared-expert FFNs. Clamped activations keep
the native FP32 calculation; FP8 output is unsupported on NPU.

The Ascend plugin's explicit prefill composition can prepare a
`W4A4GroupedPlan` from the existing fused W4A4 weight plan. It shares NZ
weights, encodes official GMM scales once, and executes Cube FlatQuant before
official InitRouting, GMM13, SwiGLU, Cube FlatQuant, GMM2, and FinalizeRouting.
The caller supplies token-major expert IDs, route weights, and its local
expert interval. InitRouting copies packed INT4 bytes and FP32 scales with
one permutation; group-aware consumers skip unused capacity rows. No token
padding is added. Both NZ orientations are supported. This is an explicit
composed implementation, not
a fallback when the fused decode kernel rejects a shape. Official GMM's BF16
epilogue has different intermediate rounding from fused vector dequantization.

Ascend BF16 DSA Prefill uses `mlp_lightning_indexer` (FP32 score weights)
and `sparse_mla_attention_prefill`. Build both OPP operators alongside the
existing Decode operators. Prefill keeps token-local queries and all heads;
there is no query/output AllToAll or TopK/LSE exchange. Persistent KV remains
owner-sharded.

Index-K RMS normalization uses the checkpoint's `rms_norm_eps` in both the
composed path and the fused Decode prologue.

The CANN 9.0/A2 CP runtime additionally loads ABI 2
`libdsa_cp_prefill.so`, built from the operator repository's standalone
`cann-flash-ops/src/dsa_cp_prefill` CMake project. Set
`TOKENSPEED_DSA_CP_LIBRARY` to its absolute library path before cache binding.
One AICPU communicator and stream gather directly into preallocated receive
planes. Index-K is submitted after its producer; current KV communication
runs while Indexer computes. Sparse attention waits for KV. The preceding KDA
layers can overlap historical gathers; cold Prefill skips those transfers.

One device metadata launch prepares the compact page table shared by Indexer
and sparse attention, cumulative lengths, causal scans and current-page aliases.
Complete aligned pages are read directly at their receive address; only
boundary pages are stitched. The DSA copy kernel also commits persistent owner
rows. The unbounded configuration gathers the full cache without a CPU page
plan. With `--chunk-prefill-kv-size`, the scheduler limits the sum of per-request
KV page counts, including each cached prefix. Native Prefill reuses the host
page table to compact referenced owner pages once per batch and gathers unequal
contributions with HCCL AllGatherV; no device count readback is needed. This
requires the library's `DsaCpAllGatherV` and `DsaCpPatchMapped` exports. Reserve
the bounded history send/receive planes, a null page and current-row padding
before sizing the cache arena. Decode and the shared cache writer keep their
existing contracts.
The scheduler wheel and `ts-host` must both be rebuilt when adding this
scheduler option, since the host embeds its own copy of the scheduler.
Native commits borrow int64 write-slot scratch from the runtime's main-stream
`WorkspacePool`. Startup warms its peak before KV memory profiling. The model
recipe only declares `AscendPrefillWorkspace`; TS derives and deducts the
communication budget before agreeing on cache capacity. Each attention backend
allocates its communication planes together through a frozen `WorkspacePool`
and retains that owner across layers. Those side-stream buffers stay separate
from main-stream scratch. The native transport borrows the supplied views.
Every commit obtains a fresh scratch view, so another main-stream user cannot
invalidate its slot allocation.

Ascend dense MLA Prefill reuses this ABI with only the 576-wide KV plane;
it allocates no Index-K plane and retains the original head-sharded weights.
Historical KV gathers can start in preceding KDA layers. After the local
QKV-A projection and norms, current KV gathers on the same AICPU stream while
the normalized Q latent gathers over TP and QB projects this rank's heads.
The receive table aliases aligned current pages and stitches boundary pages
using the existing copy kernel. Each rank expands all visible KV with its own
KV-B weight, calls the existing dense MLA attention once per request, and
projects its local-head output with the original O shard. CommManager retains
the output ReduceScatter. No additional LSE merge or complete-weight copy is
needed. Causal attention with a cached prefix uses bottom-right alignment;
CPU scheduler lengths are passed directly, without a device length readback.

MLA decode partial packing reuses its compiled kernel across changing batch
sizes and preserves empty-shard masks during Graph replay.

Run the operator correctness suite on a visible NPU with:

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
PYTHONPATH="${PWD}/python:${PWD}/tokenspeed-kernel/python:${PWD}/tokenspeed-kernel-npu/python:${PYTHONPATH:-}" \
    pytest -q tokenspeed-kernel-npu/test
```

For the complete Qwen3-0.6B launch command, ACL Graph capture sizes, serving
limits, and a request example, see the
[Ascend model recipe](../docs/recipes/models.md#qwen3-06b-on-ascend-npu).

## Serving dependencies in a source checkout

When running TokenSpeed directly from a checkout through `PYTHONPATH`, install
the SMG gateway explicitly, at the version `python/pyproject.toml` pins. Run
these commands from the repository root with the same Python interpreter used
to launch TokenSpeed:

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh

python -m pip install \
    "tokenspeed-smg==1.13.0.post20261019" \
    "viztracer"
```

If the host requires an outbound HTTP proxy, configure it for the installation
without committing credentials:

```bash
export PROXY_URL="http://<username>:<password>@<proxy-host>:<port>"
export HTTP_PROXY="${PROXY_URL}"
export HTTPS_PROXY="${PROXY_URL}"
export http_proxy="${PROXY_URL}"
export https_proxy="${PROXY_URL}"
```

Verify the gateway, then the engine from the source paths:

```bash
python -m smg launch --help
PYTHONPATH="${PWD}/python:${PWD}/tokenspeed-kernel/python:${PWD}/tokenspeed-kernel-npu/python:${PYTHONPATH:-}" \
    python -m tokenspeed.runtime.entrypoints.engine --help
```
