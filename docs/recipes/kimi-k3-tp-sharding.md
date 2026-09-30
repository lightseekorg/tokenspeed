# Kimi-K3 TP sharding under DEP16

Kimi-K3 can shard its QKV projections, output projections and BF16 shared-expert
MLPs independently. Attention and caches stay TP1/DP16; routed experts stay
TP1/EP16. Each sharded component restores outputs to the original token owners
before the next consumer. This changes weight placement and communication, not
the scheduler, cache layout or checkpoint parameter names.

Use real checkpoint weights and the full model by default. A reduced-layer run
is a separate capacity experiment, not a full-model performance or accuracy result.

## Choose what to shard

All three environment variables default to `1` (replicated):

| Component | Variable | Execution within each TP subgroup |
| --- | --- | --- |
| QKV projections | `TOKENSPEED_KIMI_K3_QKV_PROJ_TP_SIZE` | AllGather → column-parallel GEMM → A2A |
| Output projections | `TOKENSPEED_KIMI_K3_O_PROJ_TP_SIZE` | A2A → row-parallel GEMM → ReduceScatter |
| Shared experts | `TOKENSPEED_KIMI_K3_SHARED_EXPERT_TP_SIZE` | AllGather → gate/up → activation → down → ReduceScatter |

For a combined TP4 run:

```bash
export TOKENSPEED_KIMI_K3_QKV_PROJ_TP_SIZE=4
export TOKENSPEED_KIMI_K3_O_PROJ_TP_SIZE=4
export TOKENSPEED_KIMI_K3_SHARED_EXPERT_TP_SIZE=4
```

Set all three to `1` for DEP16. To measure one component, enable only its
variable and leave the other two at `1`. Label combined results separately:
a projection-only speedup is not a shared-expert or full-model speedup.

Projection TP must divide world size and respect the checkpoint's quantization
alignment. QKV currently supports TP2/4/8/16 with block-scaled FP8 weights and
BF16 activations. TP4 is the validated model configuration here; other sizes need
validation on the target topology and compatible collective backends.
Shared-expert TP must be a positive divisor strictly smaller than world size,
and must divide the MLP intermediate width. DEP16 therefore supports shared
TP2, TP4 and TP8.

These paths require attention and linear-attention TP1, attention DP and routed
MoE EP equal to world size, and pipeline parallelism 1. Settings must agree
across ranks before groups are created.

### Weight placement and token ownership

KDA shards the fused Q/K/V/output-gate/decay-down/beta projection; MLA shards
fused QKV-A/output-gate and Q-B. MLA's absorbed KV-B weights and KDA's decay-up
projection stay unchanged. QKV weights and 128×128 scales load as contiguous
output-channel shards, with zero tail padding to keep scale blocks aligned.
Output projections load input-channel shards and their scales directly.
Checkpoint codes are not requantized.

`DPColumnParallelLinear` and `DPRowParallelLinear` live in `layers/linear.py`
and inherit their weight creation, shard loading and quantization from the
ordinary column/row parallel layers. Parameter paths stay unchanged, including
`qkvgb_proj.weight` and `o_proj.weight`; there is no nested `linear` module.
KDA retains its segment-aware checkpoint loader in `KimiKDAColumnProj`.

Both DP layers take explicit physical token counts and return `(output, bias)`.
Column projection keeps the parent's `output_size` as the padded weight width
and returns only `logical_output_size` channels. It disables the parent's output
gather because A2A restores complete outputs. Row projection similarly disables
the parent's all-reduce because ReduceScatter performs the required reduction.
The ordinary GEMM paths use the parent methods; compatible FP8 plans retain
the fused quantization paths described below.

Communication buffers are separate from the linear modules. Model setup binds
one `DPColumnParallelCommunication` per stored shape and one shared
`DPRowParallelCommunication` for sequential O projections before memory
budgeting and graph capture. Forward never creates communicators or grows
their buffers; the auxiliary-stream MLP buffers remain independent.

Inspect tensors rather than inferring every layer's precision from the model
name. The NVFP4 checkpoint used for this recipe has block-FP8 attention
projections and BF16 shared experts; its routed experts use NVFP4.

Uneven and empty owners participate using padding to the subgroup's maximum
physical row count. Only wholly empty subgroups skip communication. CUDA-graph
padding counts, not just live requests, determine collective sizes. There is
one execution path for prefill, decode, eager execution and graph replay.

## Communication backends

The default projection settings are:

```bash
export TOKENSPEED_PROJ_A2A_BACKEND=tokenspeed_a2a_lamport
export TOKENSPEED_O_PROJ_RS_BACKEND=triton_peer
```

`TOKENSPEED_PROJ_A2A_BACKEND` selects the A2A implementation for both QKV and
output projections. Use `tokenspeed_a2a_lamport` for the optimized TP4 path on
a supported single-host NVLink topology. Set it to `nccl` to use the reference
collective path or when the optimized path does not support the deployment.

| Operation | Fast path | NCCL fallback |
| --- | --- | --- |
| QKV AllGather | TRT-LLM, up to 128 physical rows/rank | Larger batches |
| QKV and O-projection A2A | TokenSpeed Lamport, TP4 BF16 on one NVLink-connected host, up to 512 rows/rank | Other group sizes/topologies, incompatible shapes or larger batches |
| O-projection ReduceScatter | `triton_peer`, TP4 BF16, up to 8192 rows/rank | Other group sizes/dtypes or larger batches |
| Optional O-projection ReduceScatter | `trtllm_lamport`, TP4 BF16, up to 128 rows/rank | Other group sizes/dtypes or larger batches |
| Shared-expert AllGather/ReduceScatter | TRT-LLM, TP2/4/8/16, positive 128-aligned hidden width, up to 128 rows/rank | Other sizes, unaligned widths or larger batches |

Explicit `nccl` selects the reference projection A2A or reduction path.
The one-shot paths require CUDA IPC peer access; initialization failures
within a supported contract are fatal, not a reason to retry a collective
inside forward. Shared AllGather subdivides aligned widths within the native
limit; native address-space limits still apply.

TokenSpeed A2A fuses exchange and channel-layout conversion in both directions.
Messages through 8 MiB use tagged packets; larger messages use vectorized chunk
exchange when channels are divisible by 32. The threshold is the whole per-rank
BF16 payload, including padding, not one peer's share. It is independent of the
512-row limit. See the [kernel protocol and lifetime contract](https://github.com/lightseekorg/tokenspeed/blob/main/tokenspeed-kernel/python/tokenspeed_kernel/ops/communication/tokenspeed_a2a_lamport.md).

`--all2all-backend flashinfer` in the launch command below selects **routed MoE
transport**, not projection A2A. That transport and other FlashInfer kernels
remain supported.

### Fused QKV AllGather and quantization

QKV uses fused BF16 AllGather plus 128-element FP8 activation quantization by
default for TP2/TP4, 1..128 physical rows/rank, and a compatible prepared
FlashInfer block-FP8 GEMM plan with FP32 scales. No extra switch is needed.
Communication stays BF16: the fusion removes the gathered-output write/read
and a separate quantization launch, not network bytes.

Larger batches, TP8/TP16, explicit NCCL AllGather, BF16 linears and incompatible
GEMM plans retain separate gather and linear execution. This fusion applies
only to QKV, not shared experts. It is not RMSNorm fusion or PDL; GEMM
waits for the fused kernel to complete normally.

The kernel APIs are `TrtllmAllGatherQuantState` and
`trtllm_allgather_fp8_quantize`. Create state collectively before graph capture,
with an explicit positive CTA count no larger than the device's SM count.
Runtime uses one CTA per SM. Inputs are contiguous BF16, width divisible by
128, with equal physical counts after padding. Quantization matches the
prepared GEMM's scale/clamp and padding rules; dependency changes require
checking exact FP8 bytes and scales again.

### Fused O-projection A2A and quantization

Compatible output projections fuse Lamport A2A with the same 128-element FP8
activation quantizer by default. The receiver quantizes each complete group
only after its BF16 payload is ready, writing FP8 values and MN-major FP32
scales directly for the prepared GEMM. This removes a BF16 output write/read
and the separate quantization launch; it does not reduce network traffic.

The fusion requires TP4, 1..512 physical rows/rank, input width divisible by
512, no bias/redundant Linear reduction, and a compatible prepared FlashInfer
block-FP8 GEMM plan. Other linears, NCCL A2A and larger batches keep separate
exchange and Linear execution. GEMM still writes directly into the selected
ReduceScatter input buffer. Quantization does not change its reduction backend.

Prepare the FP8 outputs and scales before memory profiling or graph capture.
They are borrowed across serialized calls and add approximately 0.516 times
the maximum BF16 payload to A2A scratch. The ordinary BF16 output is retained
for fallback and reference use. QKV's inverse A2A remains unquantized.

### Workspace and stream lifetime

Allocate communicators and persistent scratch before cache budgeting and graph
capture. Same-shape sequential layers reuse scratch on one serialized stream.
Projection scratch is separate from attention buffers and auxiliary-stream
shared-expert communication.

TokenSpeed A2A's packet/chunk rings and local output cost about ten times the
maximum payload, in addition to NCCL staging. QKV gives A2A an owned output tensor
to write directly, avoiding a post-exchange copy while keeping earlier results
valid across later projection calls. O projection consumes borrowed output
directly in GEMM.
Lamport protects its own rings, not the separate GEMM buffer used by
`triton_peer`. Peer reduction keeps an explicit completion fence before that
buffer can be written again.

The fused AllGather state shares its IPC ring with ordinary BF16 AllGather.
Its outputs are borrowed until the next call. The ordinary BF16 output buffer
is retained for fallback/reference use, so the fusion does not reduce allocated
scratch. Plain TRT-LLM collectives use independent explicit
`TrtllmAllGatherState`/`TrtllmReduceScatterState` instances; gather output is
borrowed and reduction output is owned.

Shared-expert AllGather runs on the auxiliary stream and completes before
routed dispatch. Shared GEMMs finish before routed BMM. Shared ReduceScatter
starts after dispatch and can overlap routed BMM; it completes before combine.
Shared collectives do not overlap routed dispatch/combine.

## Reserve GPUs and use the shared environment

For four GPUs per node, reserve a persistent allocation:

```bash
salloc --no-shell --account ACCOUNT --partition PARTITION --nodes 4 --exclusive \
  --ntasks-per-node 1 --gpus-per-node 4 --time 02:00:00
```

Use the site's fabric-placement constraints. FlashInfer MoE transport requires
all 16 GPUs on a compatible NVLink fabric, not only each contiguous TP4 subgroup.
Run through the site's `submit` wrapper or `srun --jobid JOB_ID`, with one
serving launcher per node and four workers per launcher. Keep the allocation
for both baseline and sharded runs. `HEAD_NODE` must be an allocated node,
not localhost.

Mount the shared checkout and environment at the same absolute paths on every
node, activate its `.venv`, and select worktree sources using `PYTHONPATH`.
Repeat explicit mounts even when reusing named containers. Do not create an
environment per experiment or restore obsolete dependency overlays. Record
the container, package versions, source commit and native-extension build
identity; a source rebase may require rebuilding extensions.

## Correctness checks before model benchmarks

The shared-expert model orchestration and stream-ordering tests are registered
in the per-commit `runtime-1gpu` CI suite. The distributed real-weight validators
below are separate and are not automatically run by that suite. A green
unit-test job does not establish real collective or real-weight validation.
Multi-node Slurm CI skips fork PRs; running that validation requires a reviewed
dispatch with suitable GPUs and checkpoint access.

Run orchestration and real stream-dependency checks:

```bash
python -m pytest -q \
  test/runtime/test_kimi_k3_moe_attn_dp.py \
  test/runtime/test_cuda_stream.py test/runtime/test_kimi_k3_config.py
```

The distributed validators check projection numerics with real weights, output
ownership, graph replay, and backend boundaries. They also exercise the output
buffer fallback and empty KDA/MLA model forwards with independent QKV/O switches
and real collectives, using small deterministic matrices for the latter.
Within the allocation, launch each validator on all nodes:

```bash
python -m torch.distributed.run --nnodes=4 --nproc-per-node=4 \
  --node-rank=NODE_RANK --master-addr=HEAD_NODE --master-port=PORT \
  -m test.runtime.distributed.validate_kimi_k3_o_proj \
  --model MODEL_DIR --large-tokens

python -m torch.distributed.run --nnodes=4 --nproc-per-node=4 \
  --node-rank=NODE_RANK --master-addr=HEAD_NODE --master-port=PORT \
  -m test.runtime.distributed.validate_kimi_k3_qkv_proj --model MODEL_DIR --rows 128

python -m torch.distributed.run --nnodes=4 --nproc-per-node=4 \
  --node-rank=NODE_RANK --master-addr=HEAD_NODE --master-port=PORT \
  -m test.runtime.distributed.validate_kimi_k3_shared_expert_tp \
  --model MODEL_DIR --layer 1 --tp-size 4
```

The validators cover uneven/empty owners, exact shards, retained-output lifetime
and changing-input graph replay. O projection compares replicated and sharded
projections against a dequantized FP32 reference, including the 128/129,
512/513 and 8192/8193 collective boundaries. QKV checks fused versus unfused
execution, packet/chunk/NCCL transitions and delayed peers. Shared-expert
validation covers rank agreement, real collectives, row-count boundaries and
runtime widths; repeat with TP2 and TP8 on the intended topology.

On four GPUs, run kernel correctness separately from runtime tests:

```bash
python -m pytest -q \
  test/runtime/distributed/test_comm_ops.py::TestCommOps::test_all_to_all_single
python -m pytest -q \
  tokenspeed-kernel/test/nvidia/ops/communication/test_trtllm_allgather_quant.py
python -m pytest -q \
  tokenspeed-kernel/test/nvidia/ops/communication/test_tokenspeed_a2a_quant.py
```

These tests spawn their own workers; do not wrap pytest in torchrun. A2A is
bit-compared with NCCL, including special payloads, ring reuse and graph replay.
Fused gather/quantization checks FP8 bytes and scales for TP2/TP4, padded/empty
owners and interchange with ordinary AllGather on the same workspace.
Fused A2A checks exact FP8 values/scales, zero and extreme-value groups,
delayed peers, packet/chunk transitions and interchange with unquantized A2A
in captured graphs. Real-weight O-projection validation additionally compares
fused and separate quantization with identical TP4 GEMM and reduction.

For performance, warm up both variants and time the complete operation with
CUDA graphs, identical weights/counts and repeated unprofiled measurements.
Include packing, quantization, communication, GEMM and output restoration.
Separate baseline quantization error from sharding's accumulation-order error.
Neither a unit benchmark nor a serving smoke test establishes dataset accuracy.

## Full-model launch and comparison

The following command uses real NVFP4 weights, the full model, no speculation,
CUDA graphs and capacity for 16 requests/rank (256 total):

```bash
python -m tokenspeed.cli serve \
  --model MODEL_DIR --load-format safetensors \
  --served-model-name kimi-k3-nvfp4 --trust-remote-code --language-model-only \
  --dtype bfloat16 --quantization nvfp4 --kv-cache-dtype fp8 \
  --world-size 16 --nprocs-per-node 4 \
  --attn-tp-size 1 --data-parallel-size 16 --dense-tp-size 1 \
  --moe-tp-size 1 --expert-parallel-size 16 \
  --dist-init-addr HEAD_NODE:PORT --host 0.0.0.0 \
  --attention-backend tokenspeed_mla --kda-backend cutedsl_kda \
  --moe-backend flashinfer_trtllm --all2all-backend flashinfer \
  --max-model-len 4096 --max-num-seqs 256 \
  --chunked-prefill-size 1024 --max-prefill-tokens 1024 \
  --prefix-granularity 128 --enable-prefix-caching --disable-kvstore \
  --max-cudagraph-capture-size 16 --cudagraph-capture-sizes 1 2 4 8 16 \
  --disable-cuda-graph-padding --disable-autotune \
  --gpu-memory-utilization 0.83 --seed 1 --enable-output-logprobs \
  --policy cache_aware --dp-aware
```

Keep hardware, checkpoint, MoE transport, prompts and serving settings fixed.
Use explicit request-to-rank affinity, fixed token IDs and greedy generation.
Compare all-TP1 against the selected TP settings; for low-message O-projection
reduction, select `TOKENSPEED_O_PROJ_RS_BACKEND=trtllm_lamport` explicitly.

Exercise cold prefill, decode and incremental prefill reusing a cached prefix.
Record active counts, cache hits, TTFT, decode intervals, throughput and total
workflow latency separately. Report weight memory and cache capacity as well
as peak device memory: the fixed budget can turn weight savings into more cache.
If generated strings diverge, also score identical reference prefixes; later
logits on different generated contexts cannot establish a projection error.

C32/C64/C128 per rank mean 512/1024/2048 concurrent requests across 16 ranks.
Increase both the global sequence budget and graph size. Confirm cache/state
admission and actual occupancy; weight loading or `max-num-seqs` alone does not
prove that many requests can run. Report insufficient memory rather than
silently reducing model depth.

### Short combined C128 profile

Enable all three TP sizes at `4`, use `tokenspeed_a2a_lamport` A2A and
`trtllm_lamport` O-projection reduction, and keep DEP16 attention/cache and
routed-expert ownership. Set `--max-num-seqs 2048`,
`--max-cudagraph-capture-size 128`, and include `128` in
`--cudagraph-capture-sizes`.

Only for an explicitly approved 48-layer experiment, retain real checkpoint
weights and override both `num_hidden_layers` and the attention-layer lists:
full-attention layers are 4, 8, ..., 48 (one-based); the rest are KDA. Remove
indices for omitted layers and label reports **48-layer real-weight**.

Warm up before capture. For an agentic workload, prime a 1024-token prefix,
then submit a 1280-token input reusing it and request 512 output tokens.
Wait until all 2048 requests have emitted a token and none have completed,
then capture 20 decode steps. Verify 128 real active requests on every rank,
without prefill or padding rows replacing requests.

Use one NSYS launcher per node, CUDA-profiler-API capture boundaries, runtime
NVTX annotations, and
`--trace=cuda-sw,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none`.
Check QKV/O-projection and auxiliary-stream shared-expert collectives across
all 16 GPUs. Save clearly named per-node `.nsys-rep` files and ZIP them together.
Retain input IDs and cache-hit counts. Use traces to explain costs; report
performance from separate unprofiled measurements.
