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

With `--moe-backend flashinfer_trtllm`, NVFP4 expert shards are automatically
padded to a multiple of 64 along their per-rank intermediate dimension. The
packed weights and block scales in the padded tail are zero-filled, so the
extra dimensions do not change the MoE result. For example, a 640-wide expert
under MoE TP4 is padded from 160 to 192 values per rank.

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

DeepEP prefill breakable CUDA graphs have a scoped explicit opt-in described
below. Other all-to-all configurations keep prefill graphs disabled by default.
Decode graphs are unaffected.

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

### DeepEP prefill breakable CUDA graphs

DeepEP normal dispatch and expert compute stay eager because expert receive
counts depend on routing. Breakable CUDA graphs capture the surrounding model
work and replay it around the complete eager MoE operation. The initial opt-in
targets `Qwen/Qwen3.5-35B-A3B-FP8` in text-only serving, with block-scale FP8
weights, BF16 activations, `deep_gemm`, DeepEP `auto`, and no speculative decoder.
It requires one node, MoE TP1, and a single EP group spanning all serving ranks.
Actual multimodal execution, other MoE kernels, multi-node EP,
pipeline/context parallelism, disaggregated serving, and speculative decoding
are outside this support scope. Startup checks the loaded model blocks and
resolved kernels, in addition to the requested configuration.

The four-GPU attention-TP2/DP2, dense-TP4, MoE-TP1/EP4 configuration is:

```bash
tokenspeed serve Qwen/Qwen3.5-35B-A3B-FP8 \
  --attn-tp-size 2 --data-parallel-size 2 --dense-tp-size 4 \
  --moe-tp-size 1 --ep-size 4 \
  --quantization fp8 --dtype bfloat16 --moe-backend deep_gemm \
  --all2all-backend deepep --deepep-mode auto \
  --language-model-only \
  --max-model-len 16384 --max-num-seqs 16 \
  --max-prefill-tokens 8192 --chunked-prefill-size 8192 \
  --prefill-graph-max-tokens 8192 \
  --prefill-graph-capture-sizes 32 128 512 2048 8192
```

Set an explicit positive `--prefill-graph-max-tokens` to enable the feature.
Leaving the cap unset keeps DeepEP prefill graphs off; set the cap to `0` or use
`--disable-prefill-graph` to disable them. Unsupported static configurations with
an explicit positive cap fail with a configuration error. The capture ladder
must fit the prefill, chunk, and KV budgets; smaller caps reduce startup and
graph-memory costs. Keep `auto` so prefill uses normal dispatch and decode uses
the existing low-latency path.

A replay requires pure prefill work on every attention-DP replica. All ranks
select the same bucket from the largest replicated token count. Mixed
prefill/decode, idle replicas, and batches above the capture cap use the existing
dispatcher; eligible decode still uses its decode graph. Unequal prompt lengths
can replay in a shared bucket. Padded rows never enter expert routing, including
when attention TP leaves a rank with no real source rows; that rank still joins
the collective.

Successful startup is not proof that a workload replayed a prefill graph. Check
the `DeepEP prefill BCG replay: rank=... bucket=... real_tokens=...` message on
every participating rank while sending concurrent prefill requests to both DP
replicas. This once-per-rank message is emitted after serving replay, not during
startup smoke replay. A serialized request stream can leave one replica idle
and legitimately use eager prefill throughout. Capture failures terminate
startup instead of silently turning the feature off.
See [the execution invariants](../design/unified_path.md#deepep-prefill-in-the-same-forward)
for the capture-only stub, output lifetime, padding, and startup protocol.

Measure against the same DeepEP server with only the prefill-graph flags changed;
keep decode graphs enabled in both runs. Report TTFT and input throughput along
with TPOT, startup time, peak/retained GPU memory, and available KV capacity.
The complete MoE operation remains eager, so the benefit depends on how much
launch overhead is saved relative to padding and graph-memory costs.

For paired trials, copy the
[qualification example](../../test/random_benchmark/tokenspeed/deepep_bcg_qualification.example.json)
and fill in the actual source/model revisions, dependency versions, hardware,
and workload. The common server command must omit prefill-graph flags and
`--enforce-eager`; the tool adds only the selected arm's prefill-graph flags.
Keep streaming enabled for the EvalScope client.

```bash
python3 test/random_benchmark/tokenspeed/deepep_bcg_qualification.py plan \
  --config qualification.json --output results/deepep-bcg
# Run each entry in results/deepep-bcg/commands.txt in order, with a fresh server.
python3 test/random_benchmark/tokenspeed/deepep_bcg_qualification.py report \
  results/deepep-bcg
```

`plan` writes a seeded schedule of at least eight eager/BCG pairs and does not
launch any processes. Use `commands.txt` as a manual checklist: start a fresh
server, wait for readiness, then run that entry's client. Save all ranks' server
output to each run's `evidence/server.log` and retain memory/startup measurements
alongside it. `report` reads the completed EvalScope outputs, checks per-rank
replay markers, and writes `report.json` with paired latency reductions and
bootstrap confidence intervals. Client TTFT
includes queueing and transport; optional server-prefill measurements require
separate `evidence/server_prefill.csv` samples with a `latency_ms` column and a
`server_prefill_source.txt` description of the measured boundary and raw source.
Missing metrics remain null and receive no comparison estimate. The report
preserves evidence hashes and makes no automatic performance-promotion decision.

Run the model correctness qualification independently of the timing trials:

```bash
TOKENSPEED_DEEPEP_BCG_E2E=1 python3 -m pytest -q \
  test/runtime/models/test_deepep_prefill_bcg_e2e.py
```

It requires four Blackwell GPUs and compares two fresh eager-prefill engines
with a BCG engine. The separate
[GSM8K CI recipe](../../test/ci/eval/qwen3.5-35b-a3b-fp8-deepep-bcg-tp2dp2ep4-evalscope-gsm8k.yaml)
retains the existing `0.90` quality threshold. Successful graph replay,
correctness, and a measured performance benefit are separate checks.

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
