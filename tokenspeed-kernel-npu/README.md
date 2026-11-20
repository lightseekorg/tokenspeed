# TokenSpeed-Kernel-NPU

TokenSpeed-Kernel-NPU contains the Ascend-specific operators used by
TokenSpeed. Keeping these implementations in a standalone package mirrors the
AMD package layout and keeps the TokenSpeed runtime vendor-neutral.

The initial Ascend path provides paged MHA, RMSNorm, Q/K RMSNorm, rotary
embedding, and the Triton-Ascend import adapter required by CANN 9.0.0.

Per-head Q/K normalization uses native RMSNorm. RoPE supports K-only updates
and casts only the cache rows selected by the current positions. Block-draft
MHA dispatch exposes a `block_draft` trait so providers can register noncausal
block implementations. Decode and schedule calls require an explicit
`block_draft` value: `False` for ordinary decode and `True` for draft blocks.

For development from this repository:

```bash
test/ci_system/install_triton_ascend.sh
```

The validated stack is CANN 9.0.0, PyTorch 2.9.0, `torch_npu` 2.9.0.post2,
Transformers 5.12.0, Triton 3.2.0, and Triton-Ascend 3.2.1. The setup script
records every Python package mutation needed by this Ascend path, including the
`apache-tvm-ffi==0.1.13` build dependency and editable install of this package.

For the optional Ascend Memory Saver package, PTA requirements and startup
instructions, see the [serving guide](../docs/serving/ascend-memory-saver.md).

TokenSpeed applications should continue to import operators from
`tokenspeed-kernel`; it owns registration and dispatch to this package.

LongCat OE host-table registration and packed lookup use the `custom` Torch
namespace supplied by the unified `flash_ops` `.run` installation. Source that
installation's environment script before starting the engine. Missing current
operators fail during model preparation; retired namespace aliases are not used.

The BF16 local-expert chain keeps FP32 router coefficients through
FinalizeRouting. Expert GEMM inputs, weights and outputs remain BF16; routing
coefficients are not rounded to BF16 before the weighted sum.

The runtime's unclamped `SiluAndMul` uses the fused NPU SwiGLU operator through
`tokenspeed-kernel`, including shared-expert FFNs. Clamped activations keep
the native FP32 calculation; FP8 output is unsupported on NPU.

RMSNorm accepts FP32 checkpoint scales with BF16 activations. It promotes mixed
inputs for the CANN operation and returns the activation dtype, preserving the
caller's output buffer and residual storage.

The Ascend plugin prepares a `W4A4GroupedPlan` for Prefill. Fused Decode and
Prefill share the weights for NN checkpoints. For an NT checkpoint the loader
keeps Decode's NZ layout and also packs the output channels into INT4 words
for Prefill GMM.
Prefill encodes official GMM scales once and executes Cube FlatQuant before
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

The CANN 9.0/A2 CP runtime uses an isolated official HCCL process group.
The packaged `custom::npu_dsa_cp_metadata.out` operator prepares compact page
tables, cumulative lengths, causal scans and current-page aliases once per batch.
Triton kernels gather cache rows and commit current rows without changing bits;
see [DSA/MLA Prefill transport](#dsamla-prefill-transport) for ownership and sizing.

Ascend dense MLA Prefill reuses this ABI with only the 576-wide KV plane;
it allocates no Index-K plane and retains the original head-sharded weights.
Historical KV gathers can start in preceding KDA layers. After the local
QKV-A projection and norms, current KV is gathered, followed by the normalized
Q latent gather over TP and QB projection for this rank's heads.
The receive table aliases aligned current pages and stitches boundary pages
in the fused current-row store. Each rank expands all visible KV with its own
KV-B weight, calls the existing dense MLA attention once per request, and
projects its local-head output with the original O shard. CommManager retains
the output ReduceScatter. No additional LSE merge or complete-weight copy is
needed. Causal attention with a cached prefix uses bottom-right alignment;
CPU scheduler lengths are passed directly, without a device length readback.

MLA decode partial packing reuses its compiled kernel across changing batch
sizes and preserves empty-shard masks during Graph replay.

Ascend paged MHA supports up to five query tokens per request for speculative
verification. The fused attention and Triton decode registrations cover
64- and 128-token pages.
Qwen3 with fixed-width chain MTP and the default paged MHA route checks the
registered verify widths before initializing the distributed process group.

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

### DSA/MLA Prefill transport

The transport borrows an isolated HCCL group for cache owners and allocates one
persistent Torch tensor. Startup checks its bytes against the cache recipe's
reservation. All layers of one backend reuse it; history length changes only
views, with no allocation on the forward path. PyTorch owns allocation and page
policy, and the workspace appears in its memory statistics.
Synchronous HCCL submissions order their internal stream before subsequent work
on the executor stream. The transport adds no user-managed communication stream,
per-layer synchronization or permanent stream identity restriction.

History includes only referenced pages, deduplicated by owner. One gather kernel
copies cache rows into separate KV576 and optional Indexer-K128 send planes.
HCCL receives each plane directly at its final attention address, so there is
no whole-history packet to unpack. For degree one, the gather kernel writes the
attention planes directly. Cold Prefill skips history work.
The kernel checks each 32-row tile for contiguous source slots and copies a
matching run as one span. Arbitrary slot order keeps scalar row addresses;
page order and tail lengths do not change the copied values.

Current DSA projections pack Indexer-K128 and KV576 into 704 BF16 values per row;
MLA carries only KV576. One uneven collective carries actual rows, then one
kernel writes the padded current attention region, compact partial pages and
persistent owner caches. Complete current pages alias their attention region.
The byte copies use integer views to preserve arbitrary BF16 bit patterns.
Cache inputs accept contiguous `[rows,width]` and `[rows,1,width]`.
Packing and current writes also use 32-row tiles. Direct rows that share one
contiguous ownership block reuse the unpacked values for a bulk cache write.
Partial pages and irregular slots retain per-row writes, including valid rows
in a tile that crosses the cache extent. Both planes stay in the same kernel.

For history capacity H, current capacity C, group degree D > 1 and null rows N,
let A = ceil(C / (128 D)) * 128 D and S = max(H, ceil(C / D)). The allocation is
`2 * width * (N + H + A + S + C)` bytes. S covers an entire batch's pages on one
owner; page placement need not balance one batch. Degree one only needs C send
rows and no receive space. The cache recipe page-rounds
`chunk_prefill_kv_size`, or the model context length if omitted, to obtain H.
The scheduler enforces this summed KV-page bound, including cached prefixes.
There is no change to Decode history distribution.

Each attention layer gathers history at entry, then gathers current rows and
commits them to the cache. Execution-stream order protects the shared Torch
buffers across successive layers. The executor synchronizes before workspace
handoff or teardown. The process-group manager owns the communicator.

Run `test/test_mla_cp_transport.py` with `torchrun` and an explicit `TEST_OUTPUT`
directory. It checks shared history, empty/uneven owners, both planes, natural
singleton-head shapes, partial pages and repeated layer reuse. CPU buffer tests
use real Torch storage and substitute unavailable kernels and collectives; they do
not establish hardware correctness or full-model serving capacity.

Prepared fused Decode retains the current `npu_dcp_merge_topk` and
`npu_dsa_merge_lse_direct` communication operators. Errors propagate without
retrying through another implementation.

Variable-length MLA converts FIA's empty-KV LSE sentinel to the partial-state
identity before returning it through the registry: the request's query rows
have zero output and natural-log LSE `-inf`. Cumulative Q/KV endpoints determine
those rows, including mixed batches and history loops where only one request
has keys. Nonempty rows keep their original output and LSE. Repeated prefix
merges therefore preserve a zero-history request's current causal result.

### Sampling dispatcher

The Ascend sampler calls `torch.ops.custom` for filtering, categorical sampling,
greedy/speculative verification and scratch allocation. Install the matching
`flash_ops` package; the adapter uses its registered operators directly.
Scratch tensors remain caller-owned across NPU Graph replay. Caller-supplied
coins and the existing sampling arithmetic are preserved.
