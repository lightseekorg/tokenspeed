# Parallelism

TokenSpeed exposes familiar `--tensor-parallel-size` and `--tp` entry points
plus additional split parallelism controls for attention, dense, and MoE layers.

## Quick Start

Use this form when the same tensor-parallel group is acceptable for the model:

```bash
tokenspeed serve <model> \
  --tensor-parallel-size 8
```

`--tensor-parallel-size` maps to TokenSpeed attention tensor parallelism and
cannot be used together with `--attn-tp-size`.

## Split Parallelism

Use split knobs when different layer families need different process groups:

```bash
tokenspeed serve <model> \
  --world-size 8 \
  --attn-tp-size 4 \
  --dense-tp-size 4 \
  --moe-tp-size 4
```

| Parameter | Use |
| --- | --- |
| `--world-size` | Total worker processes across all nodes. |
| `--nprocs-per-node` | Worker processes launched on each node. |
| `--attn-tp-size` | Attention tensor parallel size. |
| `--dense-tp-size` | Dense layer tensor parallel size. Defaults to the attention replica width (attn TP x CP): the full world without DP attention, one replica with it. |
| `--moe-tp-size` | MoE layer tensor parallel size. |
| `--data-parallel-size` | Replicated data-parallel groups. |
| `--mm-encoder-tp-mode` | `weights` (default), or TP1 whole-item DP within each attention TP group (`data`). |
| `--enable-expert-parallel` | Expert parallelism across the selected world size. |
| `--expert-parallel-size` | Explicit expert parallel size. |

Kimi-K3 TP8 deployments must combine `--tensor-parallel-size 8` with
`--mm-encoder-tp-mode data`. This keeps the text model at TP8 while running the
wide-QKV MoonViT encoder at TP1 with whole-item DP8.

### Pinning a request to an attention-DP rank

Each attention-DP rank owns a private prefix cache, so multi-turn requests
only reuse their cache when every turn lands on the same rank. The
`data_parallel_rank` request field (`Engine.generate` / `async_generate`, or
the gateway's gRPC protocol) pins a request to one rank: it dispatches
straight there, bypassing load balancing — overload spill is the router's
job. An invalid pin fails that request; the engine keeps serving. Engines
without attention DP drop the pin.

Disaggregation engines ignore the pin: the `bootstrap_room` residue
(`room % dp_size`) governs prefill placement (and decode placement under
`round_robin`), so steer placement by minting the room instead. A
conflicting pin is logged and ignored, never rejected.

With the bundled gateway, pass `--policy cache_aware --dp-aware` to
`ts serve` to enable per-rank affinity routing. This requires bundled smg
releases that carry the TokenSpeed dp-affinity support; see the lockstep
note in `serve_smg.py`.

## Decode Context Parallelism

`--decode-context-parallel-size D` cyclically shards dense MLA history across
`D` ranks inside the attention TP group. Use `--dcp-comm-backend a2a` for the
packed output/LSE exchange. DCP currently rejects speculative decoding and MTP:
target and draft cache placement do not yet share a validated layout.
DeepSeek V4 keeps compressed attention history cyclically sharded, but
replicates its much smaller ratio-4 MXFP4 sparse-indexer cache in a dedicated
64-row group. Every rank therefore runs the unchanged TP indexer over complete
history: there is no candidate-score collective, score bucketing, or
local-to-global top-k merge. Because the native persistent top-k order is
unspecified, rank zero broadcasts the small ordered index tensor. Each
owner then packs only its selected attention-history bytes into their final
global positions, and an integer NCCL sum reconstructs one compact page-planar
cache on every rank. The native paged selected-attention kernel then runs with
local query heads. Page zero, invalid graph tokens, and page tails remain zero.

The reserved null page is read-only. Cyclic cache writers receive an explicit
per-token ownership mask, including fused RoPE/NoPE MLA writes; a location of
zero is not itself the writer contract. Chunked prefill reads only rank-local
cached-prefix rows, all-gathers the compact rows, and restores global
request-major order before the latent projection.

For TP-versus-DCP correctness, run
`test/manual/dcp_activation_parity.py` against the TP server to write a
reference and then against every DCP degree. It requires the repeated request
to report a real prefix-cache hit and compares exact generated IDs plus
sampled-token log probabilities only while the generated history is identical.
The HTTP API does not expose forced-token decode or full logits, so this harness
does not claim teacher-forced logit parity. DeepSeek V4's native TP persistent
top-k also has an unspecified output order; record repeated TP runs before
setting a numerical tolerance, and use the sparse-indexer unit tests to require
exact selected-index sets independently of that reduction-order noise. Final
hidden states are compared when the response path provides that optional field.
Only after that TP control demonstrates generated-ID nondeterminism, pass
`--allow-output-id-divergence` with an explicit `--logprob-atol` bounded by the
measured TP-versus-TP envelope; the common generated history remains checked.
For full-logit integration testing, run eager servers with
`TOKENSPEED_TEST_LOGIT_DUMP_DIR` set to separate empty TP and DCP directories,
issue the same short request, then compare them with
`test/manual/dcp_logit_parity.py`. The dump occurs before sampling and includes
input IDs and positions; the comparison rejects decode steps whose contexts
have already diverged.

Enable `--enable-nvtx` when profiling decode. Nsight Systems then reports these
per-layer DCP ranges separately:

- `dcp_query_pack`
- `dcp_query_all_gather`
- `dcp_attention_kernel`
- `dcp_output_lse_pack`
- `dcp_output_lse_all_to_all`
- `dcp_output_lse_unpack_merge`
- `dcp_prefix_pack`, `dcp_prefix_all_gather`, and `dcp_prefix_reorder`

DeepSeek V4 should not report an `indexer_global_topk_gather` range: its
dedicated replicated indexer deliberately removes that synchronization.

This breakdown distinguishes packing and launch latency from the attention
kernel and NCCL payload time; peak link bandwidth alone is not a useful
explanation for small decode collectives.

## MoE Deployments

Large MoE models usually choose one of these shapes:

- TP only: simplest startup path, often best for smaller MoE checkpoints.
- TP + EP: tensor parallelism within a replica, expert parallelism across ranks.
- DP + EP: multiple replicated decode groups with experts distributed inside each group.

Start with the recipe closest to your model family, then tune:

- `--tensor-parallel-size` or split TP values
- `--enable-expert-parallel`
- `--moe-backend`
- `--all2all-backend`
- `--deepep-mode`

### DeepEP all-to-all

`--all2all-backend deepep` moves expert routing off all-gather and onto DeepEP
dispatch/combine. It requires a MoE backend whose kernels own those legs:
`--moe-backend deep_gemm` (block-scale FP8) or `--moe-backend flashinfer_cutedsl`
(nvfp4, decode-shaped batches only).

DeepEP has two sets of legs, and `--deepep-mode` picks between them:

| Mode | Legs | Fits |
| --- | --- | --- |
| `low_latency` | IBGDA dispatch into a preallocated per-expert buffer | Decode-shaped batches up to `--low-latency-max-num-tokens-per-gpu` |
| `normal` | High-throughput dispatch, tokens permuted into per-expert row blocks | Extend-shaped batches of any size |
| `auto` (default) | Both are allocated; each forward picks | Aggregated serving, which mixes both shapes |

For block-scale FP8 with `deep_gemm`, keep `auto` unless the instance only ever
sees one shape -- for example a decode-only worker in a PD split, which can pin
`low_latency` and skip the normal-mode buffers. The nvfp4
`flashinfer_cutedsl` kernel implements only the low-latency legs, so it requires
an explicit `--deepep-mode low_latency`; `auto` and `normal` are rejected while
the execution plan is built. Every forward on such an instance, including any
prefill, must fit `--low-latency-max-num-tokens-per-gpu`.

A batch above the low-latency capacity is rejected rather than truncated, so
raise `--low-latency-max-num-tokens-per-gpu` if decode plus speculative draft
tokens exceed it. Both current DeepEP MoE backends require BF16 activations;
`--dtype float16` is not supported.

The mode is chosen per forward from a value every rank agrees on, because the two
modes are different collectives. With DP attention that value is "every DP rank
is decoding", so one extending rank moves the whole group to the normal legs.

The prefill CUDA graph is disabled whenever an all-to-all backend is selected:
normal-mode dispatch reports its per-expert receive counts to the host, and a
host sync cannot be captured. Decode graphs are unaffected.

For block-scale FP8 decode on NVIDIA, the low-latency path keeps routing
metadata in DeepEP's required contiguous int64/float32 formats across both
collective legs. Ordinary softmax routing selects experts and normalizes their
weights in one Triton launch before dispatch, instead of materializing the
full softmax and launching separate ATen top-k, reduction, and division kernels.
Its fused SwiGLU quantizer writes packed UE8M0 scales directly
in DeepGEMM's MN-major TMA layout, so padded rows need no zero-fill and the
second expert GEMM needs no separate activation-scale transpose/pack pass.
For sparse decode it launches a bounded number of row splits per expert and
walks only rows below the device-side expert count; full-capacity workloads
retain one-CTA-per-row parallelism. The host-side expected-row estimate selects
between these mappings without synchronizing the expert counts to the CPU.
Expert weight scales are expanded and packed once when weights are loaded,
instead of ahead of both expert GEMMs in every layer forward. Shared-expert work
is queued between the dispatch send and receive legs to overlap the collective
whenever the model has a shared expert. Low-latency dispatch asks DeepEP to
produce packed UE8M0 scales directly in its column-major TMA layout. Normal-mode
dispatch still transports FP32 power-of-two scales, but the existing expert
scatter packs them while permuting tokens, so neither mode needs a separate
sequence of elementwise shifts, fills, copies, and a transpose before GEMM1.

Dense `(128, 128)` FP8 projections have two scale contracts against
FlashInfer's FP8 block-scale GEMM, and both are copy-free for the layout they
own. The canonical K-major contract takes the quant kernel's `[M, K/128]`
activation scales and the checkpoint's `[N/128, K/128]` weight scales with no
layout conversion (strided scale views are normalized to contiguous first).
The prepared MN-major contract, selected at load time on every Blackwell
datacenter part, transposes the weight scales once and then consumes the
TRT-LLM quantizer's native `[K/128, M]` activation scales directly, which is
what the canonical path would otherwise have to transpose on every call. Both
produce bitwise identical output.

MN-major requires `M` to be a multiple of four, so a prepared layer falls back
to the canonical contract once padding would cost more than the transpose it
saves — the fused padding quantizer grows with `M` while the transpose does
not. Decode row counts stay on the prepared path.

## Multi-Node

Set these explicitly:

```bash
tokenspeed serve <model> \
  --nnodes 2 \
  --node-rank 0 \
  --nprocs-per-node 8 \
  --world-size 16 \
  --dist-init-addr <rank0-host>:25000
```

Each node must use the same model, backend, precision, and scheduler settings.
Only `--node-rank` should differ between nodes.

Run one `tokenspeed serve` per node. Node rank 0 serves the HTTP API; higher
ranks run the engine only and expose no endpoint.

### Under a launcher

Inside a multi-node Slurm step, `--nnodes`, `--node-rank` and
`--dist-init-addr` are all derived from the step environment when they are not
given, so the same command line runs on every node:

```bash
srun --nodes=2 --ntasks-per-node=1 tokenspeed serve <model> --attn-tp-size 16
```

| Argument | Derived from |
| --- | --- |
| `--nnodes` | `SLURM_STEP_NUM_NODES` |
| `--node-rank` | `SLURM_NODEID` |
| `--dist-init-addr` | first host of `SLURM_STEP_NODELIST`, port 23456 |

Rules:

- An explicit `--nnodes` or `--node-rank` that contradicts the environment is
  an error, not an override. Omit the flag to accept the launcher's value.
- An explicit `--dist-init-addr` is always used as given.
- Derivation only engages inside an `srun` step of more than one node. Outside
  a step — including the batch script of a multi-node `sbatch` — or in a
  single-node step, behaviour is unchanged: launch the ranks yourself and pass
  `--nnodes`/`--node-rank`/`--dist-init-addr`.
- If a multi-node step is detected but the topology cannot be resolved,
  startup fails with the reason rather than falling back to a single node.
- The derived address is the one the head node's hostname resolves to. Where
  that is not the interface you want carrying bootstrap traffic, set
  `--dist-init-addr` explicitly.
- `GLOO_SOCKET_IFNAME` and `NCCL_SOCKET_IFNAME` are set from the interface that
  routes to the head node, unless already present in the environment. Gloo
  needs this: it has no peer-address heuristic and otherwise binds whatever the
  local hostname resolves to, which is a loopback entry on many hosts. NCCL
  normally selects correctly on its own; it is set for consistency.
- `NCCL_IB_HCA` is not set. NCCL's own device selection prefers the
  higher-bandwidth InfiniBand devices and skips Ethernet-link ones.
- The rendezvous port is a fixed constant, not a function of `--port`. Every
  node has to arrive at the same port without talking to any other node, and
  under `tokenspeed serve` the engine's own port is allocated per node. The
  constant also stays clear of the kernel's ephemeral range, which is checked
  at startup. Pass `--dist-init-addr` to use a different port.

Apply the same NCCL transport and channel settings on every node as well. In
particular, do not mix IB and Socket selection or different
`NCCL_MIN_NCHANNELS` / `NCCL_MAX_NCHANNELS` values across ranks.

## Runtime Notes

Overlap scheduling can prepare the next forward on the CPU while the previous
forward's non-blocking host-to-device copies are still in flight. Any pinned CPU
staging buffer used for per-step model inputs must therefore have per-step
lifetime, or use an explicit synchronization before reuse. This applies to
MTP/GDN mamba state indices as well as token, length, and request-pool inputs.

CUDA IPC collectives are node-local; the mnnvl fabric workspace spans nodes.
`AutoBackend` serves single-tensor SUM all-reduces (16-bit, or fp32 where a
single-node workspace was armed for it; mnnvl serves 16-bit only) on groups
whose fan-in is 2, 4, 8, or 16 through the armed workspace -- one-shot
inside its traffic window, two-shot up to the workspace token capacity.
Cross-node groups arm the full two-shot capacity at startup; single-node
groups start at the one-shot window and serve larger shapes once
model-level preparation widens the shared workspace. Other
fan-ins and dtypes, and any shape the workspace rejects, fall back to NCCL
(inside the trtllm backend for armed groups; the Triton all-reduce tier
serves AMD only, where no trtllm workspace exists). Single-node token
all-gather/reduce-scatter runs on the Triton RSAG backend and uses NCCL
across nodes. Logits all-gather and distributed argmax use the same
cross-node fallback. This is required for layouts such as attention DP
with dense TP or MoE EP spanning nodes.

On ARM systems, [NCCL 2.29.3](https://github.com/NVIDIA/nccl/releases/tag/v2.29.3-1)
fixes a weak compare-and-swap failure that can hang NCCL when it was compiled
with GCC older than 10. Affected NCCL builds older than 2.29.3 can exhaust proxy
operations during repeated multi-node CUDA graph replay. Use NCCL 2.29.3 or
newer for this configuration. Disabling CUDA graphs avoids the affected path,
but is not required with the fixed NCCL runtime.

## Validation

Before benchmarking:

- verify every rank starts and joins the distributed group
- verify the API responds before sending load
- confirm GPU visibility and process placement
- compare output correctness before tuning throughput
- keep the full launch command with benchmark results
