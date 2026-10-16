# Beam

## What must change together

| Boundary | Beam contract |
| --- | --- |
| Config/entry | `model_type=beam`, `BeamForCausalLM`, per-layer-type RoPE and `mlp_layer_types` |
| RMSNorm | Normalize and multiply affine weights in FP32, then cast to activation dtype |
| Q/K and positions | Materialize normalized BF16 Q/K; rotate in FP32 and cast back; NoPE Q is scaled in FP32 after its BF16 norm boundary |
| Attention gate | Bias addition, sigmoid and multiplication in FP32, one cast to the activation dtype (the shared `sigmoid_mul` on NVIDIA, the same expression in Torch elsewhere; the checkpoint reference casts after each of the three) |
| Residual | Activation-dtype scaled products, then activation-dtype addition |
| Expert combine | BF16 expert output times unscaled FP32 route weight; round each route to BF16, sum and scale in FP32, cast to BF16 before TP all-reduce |
| LM head | BF16 projection, then FP32 logit softcap |

The implementation reuses shared transformer/communication, attention, linear,
normalization and expert layers. Treat each newly admitted fusion as a
separate correctness and performance candidate.

The general shared MoE implementation uses Triton with the
explicit `bf16_route_accum_fp32` policy. Automatic selection chooses this path
wherever Beam's vendor MoE default (below) does not apply, for example under
TP16 or on another platform; on NVIDIA SM100 to SM103 the flag-free default is
`flashinfer_trtllm`. Requesting FlashInfer-CUTLASS or another unsupported backend fails. Applying a
scale to an already-finalized BF16 CUTLASS result does not recover scaling before
the final cast.

## Serving

For serving, Beam requires `--dtype bfloat16`. Its MoE
backend defaults as described below; `--moe-backend triton` selects the shared
Triton BF16 MoE (`bf16_route_accum_fp32`), the previous flag-free path.

The validated Day-0 launch uses `xgrammar==0.2.8` and
`tokenspeed-scheduler>=0.1.39` from main. It runs TP8 across two nodes, leaves
`--gpu-memory-utilization` at its default of 0.95 and does not pass
`--max-total-tokens`.

Run this command on both nodes, changing only `--node-rank`. Point
`--dist-init-addr` at node 0, the only rank that serves the HTTP API. Use the
same local checkpoint path on every node.

```bash
# node 0 (serves the HTTP API); node 1 runs it with --node-rank 1
ts serve --model <beam-checkpoint-dir> \
  --nnodes 2 --node-rank 0 --nprocs-per-node 4 --world-size 8 \
  --dist-init-addr <node0-host>:29000 \
  --tensor-parallel-size 8 \
  --dtype bfloat16 \
  --max-model-len 1048576 \
  --grammar-backend xgrammar \
  --enable-mixed-batch --prefill-graph-max-tokens 8192 \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path <drafter-checkpoint-dir> \
  --drafter-attention-backend trtllm \
  --served-model-name reflection/Beam-1.0-501B-A23B-NVFP4 \
  --host 0.0.0.0 --port 8000
```

Without `--served-model-name`, the API model id is the full checkpoint path.

Pass `--max-model-len 1048576`, Beam's context of 1,048,576 tokens (1M, the
checkpoint's `max_position_embeddings`); without the flag TokenSpeed takes the
same value from `config.json`. Pass the same value with the DFlash2 drafter
(below).

Beam's grammar backend defaults to `xgrammar` on every node. Forced tool choice
and JSON `response_format` need it (see "Constrained outputs"). An explicit
`--grammar-backend` wins; `none` is intended only for benchmarking and rejects
constrained decoding requests.

Pass `--enable-mixed-batch --prefill-graph-max-tokens 8192` on every node.
With them a prompt chunk runs in the same step as the running decodes, so a
long prompt no longer pauses every running request, and steps of up to 8,192
tokens, not 2,048, replay a CUDA graph. Measured on GB200, TP8 over two trays,
with the DFlash2 drafter, against the same launch without the two flags: with
NVFP4, while 8 or 32 uncached 150K-token prompts arrived at once, the longest
pause in the output of 4 running requests fell from 21.4 s and 83.1 s to
0.135 s and 0.149 s. Output throughput at 64 and 128 concurrent requests, at
1K/1K, 1K/8K and 8K/1K input/output tokens, was 1.02x to 1.27x as high with
NVFP4, 1.03x to 1.35x with FP8 and 0.95x to 1.12x with BF16. With multi-turn
agentic traffic at 8 users, each user got 34% (NVFP4), 11% (FP8) and 4%
(BF16) more tokens per second, and the first token of a conversation came 3x
to 3.6x sooner. The cost: the KV pool is 2% (NVFP4, FP8) to 5% (BF16)
smaller, CUDA graphs take 2.0 to 2.5 GiB more per GPU, and startup takes 15
to 30 s longer.

Beam serves without TF32 on every platform: when the checkpoint's
`config.json` names `BeamForCausalLM`, the launch sets `--disable-tf32` and
replaces `NVIDIA_TF32_OVERRIDE` and `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE` with
`0`, logging each value it replaces, so the FP32 router's Torch matmul (past
16 rows, where neither of the two router kernels below takes the call) runs in
IEEE FP32. The launch reads `config.json` from the model
directory or the local Hugging Face cache; for a hub id the cache does not hold
yet, it downloads that file alone before it starts any process, unless
`HF_HUB_OFFLINE=1`. If it cannot read the file, it logs a warning and TF32
stays at its default: pass the local snapshot directory as the model, or pass
`--disable-tf32`.

From 17 to 96 rows on NVIDIA the router runs an FP32 CUDA-core GEMV on its FP32
weight. From 97 rows on SM100 to SM103 it multiplies by BF16 pieces of that
weight, made once after the weights load (1.5x the router weights' bytes on
each rank).

Beam's flag-free defaults below (the `trtllm` attention backend, the
`flashinfer_trtllm` MoE backend and the fused kernels of "Beam's kernels") apply on NVIDIA
SM100 to SM103 (`beam_default_platform` in `model_config.py`): Blackwell up to
SM 10.3, the bound FlashInfer's TRT-LLM MoE kernels are registered for, so a
later 10.x, Hopper, SM120 and AMD keep the previous flag-free path, each process
logging why. There Beam starts on the previous path, unvalidated:
no TP8 run, byte-identity check or serving run has been made outside SM100 to
SM103. The defaults were validated on SM100 (GB200) only. SM103 (B300/GB300) is
admitted but not validated: no GPU test, byte-identity check or serving run has
been made there. The GPU tests that would check it (the TRT-LLM BF16 MoE
adapter's GPU test and the `qk_rmsnorm_ssmax` byte-identity test) run on SM100
to SM103; run them on an SM103 host before serving there. There FlashInfer
compiles its JIT modules, the MoE routing module included, for the running
device (`compute_103a`, in its own `103a` workspace) unless
`FLASHINFER_CUDA_ARCH_LIST` names another architecture, so an SM103 host builds
its own modules and must not inherit a `10.0a` setting. With
`FLASHINFER_NO_DOWNLOAD=1` an SM103 host also needs the sm103a TRT-LLM-gen FMHA
context cubins staged (below).

Beam's attention backend defaults to `trtllm`, the TRT-LLM-gen paged FMHA leaf
(`layers/attention/backends/paged/trtllm.py`) that the MiniMax M3 and Qwen3.8
recipes (`models.md`) and the Qwen3.5 CI configs name on the command line. Beam
declares it as a model default (`beam_default_attention_backend` in
`_ATTENTION_FAMILY_SPECS`), so a launch without `--attention-backend` no longer
lands on the unset `mha` leaf, whose FA4 and FlashInfer TRT-LLM extend and
decode kernels share one priority and are ordered only by registration. An
explicit `--attention-backend` always wins, e.g. `triton`, `fa4`, or `mha`
(exactly the previous unset path). The leaf calls FlashInfer's TRT-LLM context
and decode kernels directly: no registry selection, no fallback to another
backend, and a missing kernel or cubin raises. It was admitted for NVIDIA SM100
to SM103 (validated on SM100) with a BF16 KV cache (head dim 128, 64-token pages, sliding and full layers,
prefix and chunked extend, CUDA-graph decode), so the default applies only
there, only under `--numerics auto` and only without an `attention.mha_*` override
variable, which this leaf would ignore. Otherwise --
another platform; `--kv-cache-dtype fp8` / `fp8_e4m3` / `mxfp8`;
`--kv-cache-dtype auto` under a `--dtype` other than `bfloat16` or `auto`; a
`--prefix-granularity` that is not a multiple of 64, such as 16, 32 or 96 (the
`mha` leaf takes it as its page, while this leaf's 64-token pages would fail
startup); `--numerics rl-bitwise`, whose bits must not depend on the batch while
this leaf's decode picks its KV split count per batch bucket; such an override
-- the backend stays unset and runs as before. The default reads the launch
arguments only. It takes `--dtype auto` for BF16, which a checkpoint declaring
`bfloat16` or no dtype resolves to; one declaring another dtype (`float32`
resolves to float16) is not caught. An FP8 KV cache the model runner derives
later from a checkpoint's `kv_cache_quant_algo` is not seen either (no Beam
checkpoint declares one). Each process logs `Beam default
attention_backend=trtllm; pass --attention-backend to choose another.` or `Beam
default attention_backend=trtllm not applied, left unset: <reason>`; the
`server_args=` line precedes the model default and still shows
`attention_backend=None`. Beam passes no attention override under this leaf (it
ranks nothing), the output gate runs after it as after every backend, and an
`attention.mha_*` override given together with an explicit
`--attention-backend trtllm` has no effect. Running it needs FlashInfer's
`fmha_gen` JIT module and the TRT-LLM-gen cubins of these shapes (BF16, head dim
128, 64-token pages: the causal and sliding context kernels and their decode
variants) staged beforehand: with `FLASHINFER_NO_DOWNLOAD=1` a missing cubin
fails the first prefill (the startup autotune unless it is disabled). The
context kernels are per architecture (sm100a on SM100, sm103a on SM103) while
the decode kernels are sm100f and serve both, so a cubin directory staged on an
SM100 host lacks SM103's context kernels. The decode
kernel variant and its KV split count follow the CUDA-graph batch size and
`--max-model-len`, so one prompt may decode differently in two batch buckets;
repeats in one bucket are expected to match.

`--moe-backend flashinfer_trtllm` runs the experts on FlashInfer's TRT-LLM BF16
kernels under the `bf16_vendor_fp32_route` policy
(`tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/README.md`). It changes the
expert arithmetic contract. On NVIDIA SM100 to SM103 it is Beam's default when
`--moe-backend` is left `auto` (`beam_default_moe_backend` in `model_config.py`):
for a BF16 model whose checkpoint is unquantized or in a quantized format Beam
admits (FP8 and NVFP4, below), without `--all2all-backend`, with an `--ep-size`
(if any) that divides the experts, and with an intermediate size per MoE rank
that is a multiple of the kernels' alignment. For BF16 experts that is 64 where
FlashInfer's launcher is relaxed (TP 1 to 8 on Beam's 2,560, and EP; not TP16),
else 128 (then TP8's 320 keeps the Triton MoE); for NVFP4 experts it is the
NVFP4 kernels' 64 (`TRTLLM_NVFP4_ISPP_ALIGNMENT`); FP8 experts take any size.
Otherwise `auto` stays,
i.e. the Triton `bf16_route_accum_fp32` MoE, as before. Each process logs `Beam
default moe_backend=flashinfer_trtllm; pass --moe-backend to choose another.` or
`Beam default moe_backend=flashinfer_trtllm not applied, left auto: <reason>`; an
explicit `--moe-backend` is never replaced, and `--moe-backend triton` is
exactly the previous flag-free path. An explicit `flashinfer_trtllm` whose
intermediate size per rank the kernels do not take fails startup instead of
padding.

On that backend Beam routes inside the kernel when the kernel's routing with an
FP32 correction bias can be built (FlashInfer's DeepSeekV3 routing with the
#5557 edit, `thirdparty/flashinfer/trtllm_bf16_moe.py`): the kernel selects
Beam's grouped top-k from the FP32 router logits and applies the routed scale to
its BF16 route weights. This requires normalized top-k weights
(`norm_topk_prob: true`) and a finite `e_score_correction_bias`, as Beam's
configs and checkpoints have. The first start on a machine and FlashInfer JIT
cache compiles that module (several minutes, logged) before CUDA graph capture;
such a start may need a larger `tokenspeed serve --engine-startup-timeout`
(3600 s by default). Where it cannot be built (the edit does not apply to the
installed FlashInfer, or the compile fails) the process logs one warning and
Beam routes with its own router kernels on precomputed top-k, the routed scale
folded into the FP32 route weights. All ranks must route alike: when the model
is built the ranks compare their MoE class and routing, and startup fails if
they differ. Each rank logs one line naming its routing and apply kernel,
`Beam vendor MoE rank R/N: routing=... apply=...`. Kernel and Beam routes
both hand the experts BF16 route weights rounded once from FP32, in a
different operation order; the two are not byte-identical.

Expert parallelism is an opt-in of that vendor path only: `--ep-size 8` with
`--tensor-parallel-size 8` gives each rank 32 whole experts (MoE TP 1) while
attention and dense layers stay TP8, so the MoE still ends in one all-reduce
over the same group. On SM100 to SM103 the MoE default above applies to it
(elsewhere pass `--moe-backend flashinfer_trtllm`). It always routes with Beam's
router kernels on precomputed top-k. It changes the expert arithmetic against
vendor TP8 (whole-expert down projection, per-rank partial sums by expert set)
and is not bitwise equal to it. The Triton policy, redundant experts and mixed
MoE TP x EP refuse it, and the attention TP size must equal the EP size.

Each of Beam's 2L all-reduce boundaries (the TP all-reduce of an attention or
MLP output, its sandwich norm and the norm that takes the depth-scale pair) runs
as one MNNVL launch where the attention TP group has an MNNVL workspace.
`--enable-allreduce-fusion` is auto-enabled for Beam on Hopper and newer at any
node count (for other models on one node only); there is no Beam flag. The
boundaries take the fused path with attention TP of two or more ranks, no DP
attention or context parallelism, EP 1 (`--ep-size 8` keeps them unfused) and
the rank-order MoE combine. On a launch across nodes they also need the
attention TP group's MNNVL workspace, armed at startup where the nodes share
multi-node NVLink (GB200 trays): without it every boundary keeps the unfused
all-reduce and two norms. Each process logs
`Beam sandwich all-reduce boundaries: through ...` or
`Beam sandwich all-reduce boundaries: unfused (<reason>)`. On one node a call
that no MNNVL workspace serves, and any call of more than
`--comm-fusion-max-num-tokens` (2048) tokens, runs those unfused kernels, so
`--comm-fusion-max-num-tokens 0` keeps every boundary unfused. Unfused, the
boundaries run the residual norm kernel of the table below with the same
arguments. The fused boundary performs the same
operations with the same BF16 rounding points as the unfused one; only the order
of the two sums of squares differs (and that of the all-reduce sum where the
unfused all-reduce does not take the MNNVL path), so the two are not
byte-identical. `--numerics rl-bitwise`, `--layer-boundary-norm unfused` and
`--moe-combine-order slot` turn the option off. The auto-enable reads the
checkpoint's `config.json` as the TF32 default above does; across nodes every
node must read it (a node that cannot keeps its boundaries unfused while the
others fuse theirs), so pass the local snapshot directory or a hub id that
every node can download.

Beam's default kernels also use FlashInfer modules that FlashInfer compiles with
nvcc on first use and keeps in its JIT workspace (`FLASHINFER_WORKSPACE_BASE`,
by default the home directory). On SM100 these are the TRT-LLM-gen attention
launcher (`fmha_gen`), SwiGLU (`silu_and_mul`), the cuBLASLt module that the
BF16 GEMM startup tuning builds (`mm_bf16_cublaslt`), the autotuner's
`trtllm_utils`, the expert path's TRT-LLM MoE module (and
`fp4_quantization_100` for NVFP4 experts) and, with the default `flashinfer`
sampler, FlashInfer's `sampling` module, which is built on the first request
that samples. The first start on a machine therefore needs a CUDA toolkit with
nvcc, and ninja, and compiles for several minutes before CUDA graph capture.
FlashInfer takes its CUDA home from `CUDA_HOME`, else `CUDA_PATH`, else the nvcc
on `PATH`, else `/usr/local/cuda`. Later starts reuse the workspace without
compiling only while the workspace path and that CUDA home stay the same.

The standard source install does not install `flashinfer-cubin` or
`flashinfer-jit-cache`. For an offline first start on GB200 (`sm100a`) with
CUDA 13.0, optionally install the matching prebuilt kernels while the node
still has network access:

```bash
python -m pip install --upgrade --no-deps \
  --index-url https://flashinfer.ai/whl/cu130 \
  "flashinfer-jit-cache==0.7.1rc2+cu130" \
  "flashinfer-jit-cache-sm100a==0.7.1rc2+cu130" \
  "https://github.com/flashinfer-ai/flashinfer/releases/download/v0.7.1rc2/flashinfer_cubin-0.7.1rc2-py3-none-any.whl"
```

`--no-deps` keeps the install to the shim and the GB200 provider named above.
These versions match `flashinfer-python==0.7.1rc2`. Without the prebuilt
packages, the first start needs network access to download the cubins. With
`FLASHINFER_NO_DOWNLOAD=1`, a missing cubin fails the start.

`FLASHINFER_DISABLE_JIT=1` is not supported: with it FlashInfer loads only the
modules that an installed `flashinfer-jit-cache` package ships, never the ones
in its JIT workspace, so without that package Beam fails at startup in every
precision (`MissingJITCacheError`).

Every start also tunes kernels before CUDA graph capture and logs
`Kernel startup tuning finished in ...`. With the trained BF16, FP8 and NVFP4
checkpoints on GB200, TP8 over two trays, the first start for a checkpoint path
tuned for 626 to 693 s and was ready after 930 to 1,030 s, inside the default
`--engine-startup-timeout` of 3600 s. These ready times are from nodes that had
already compiled Beam's Triton kernels: a node's first start also compiles them
during CUDA graph capture, about 3 minutes more. On a fresh GB200 machine with
empty caches, the same two-node NVFP4 launch tuned for 2,340 s, including the
FlashInfer JIT builds and 360 cubin downloads (101 MB per node). The engine was
ready after 2,614 s, 73% of the default timeout, and the HTTP API served after
45.5 minutes. Pass `--engine-startup-timeout 5400` for the cold first start.
Later starts reuse the JIT, cubin and tuning caches. TokenSpeed saves the tuning
result under `TOKENSPEED_FLASHINFER_AUTOTUNE_CACHE_DIR`, else
`$XDG_CACHE_HOME/tokenspeed/flashinfer-autotune` (`~/.cache` by default), keyed
by the checkpoint path and the launch settings. A restart with the same path
and settings loads it: tuning then took 5 to 9 s and the server was ready
after 241 to 430 s. A checkpoint at a new path is tuned again.

No new baseline is production-approved until the release checks are satisfied on
a pinned integrated artifact.

## Speculative decoding with the DFlash2 drafter

NextN/MTP is not enabled for Beam; speculative decoding uses Beam's DFlash2
drafter checkpoint instead. Its architecture, `DFlash2BeamDraftModel`, runs as a
DFlash2 drafter, and the same drafter serves the BF16, FP8 and NVFP4
checkpoints. The validated Day-0 drafter is
`reflection/Beam-1.0-501B-A23B-DFlash2` at revision `8d26c952`. Add
`--speculative-algorithm DFLASH`,
`--speculative-draft-model-path <drafter>` (the drafter's checkpoint directory)
and `--drafter-attention-backend trtllm` to the launch. Beam's `trtllm`
attention default (above) covers the target only, so the last flag names it
for the drafter. The drafter keeps KV for the whole context in its 5 layers:
a KV cache page then holds 14 layer slots instead of 9 (see the KV pool under
"Batches above 160"). With the drafter the default batch limit is 80
requests, not 160 (also under "Batches above 160").

With the drafter, pass `--max-model-len 1048576` as well. Speculative decoding
can write up to three draft blocks past the context (48 positions with the
drafter's 16-token blocks); Beam's RoPE table covers them and one more block,
1,048,640 rows at that context.

The final drafter declares 1,048,576 positions. It needs no context-length
override at `--max-model-len 1048576`; leave
`TOKENSPEED_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN` unset.

On the final Day-0 stack, a cold 1,045,524-token NVFP4 prompt with the drafter
enabled returned the correct passkey and reached its first token in 25.96 s.

With the trained checkpoints, greedy sampling, prompts from GSM8K's training
split and chat, and up to 2,048 output tokens, the drafter cut the mean time
per output token by these factors:

| Concurrent requests | 1 | 4 | 16 | 64 |
| --- | --- | --- | --- | --- |
| BF16 | 2.11x | 1.72x | 1.98x | 2.27x |
| FP8 | 2.18x | 1.74x | 1.93x | 2.24x |
| NVFP4 | 2.63x | 2.17x | 1.87x | 1.79x |

The FP8 row was measured when the FP8 checkpoint ran dequantized to BF16, before
the FP8 kernels above.

With the drafter, the final Day-0 stack scored 97.2% on NVFP4 and 97% on BF16
and FP8 on GSM8K (1,319 test problems, greedy).

## Quantized checkpoints

Besides BF16, Beam admits two quantized checkpoint formats (`beam_quant_policy`
in `models/beam_quant.py`); the checkpoint's own quantization config selects
them, with no flag. Any other quantization is refused when the model is built,
naming the field and the admitted formats. Neither format quantizes the KV
cache (a declared KV-cache scheme is refused), and the router and its
correction bias must be stored unquantized FP32.

### FP8 (FP8 linears and experts)

The FP8 checkpoint is compressed-tensors FP8 dynamic W8A8 (llm-compressor's
`FP8_DYNAMIC`): FP8 E4M3 weights with a static scale per output channel and
activations quantized per token at run time. It loads and runs natively in FP8,
with no BF16 copy of the weights: the experts on TensorRT-LLM's per-channel FP8
MoE, attention and the dense MLP on FP8 GEMMs with the activation quantization
fused into the neighbouring kernels. At TP8 its weights take 61.2 GiB per GPU
(experts 58.4), about half of BF16's 119.1 GiB; GSM8K (1,319 test problems)
scored 97% at TP8 and 97.7% at EP8.

Its weights and channel scales
load unchanged. Attention (Q, K, V, output and the output gate) and the dense
MLP layers run the compressed-tensors FP8 W8A8 linear: the input quantized per
token, an FP8 GEMM applying the token and channel scales to the FP32
accumulator (PyTorch's row-wise CUTLASS GEMM from 65 rows on SM100, the Triton
small-M FP8 GEMM up to 64 rows on SM100 and SM103, the Triton scaled GEMM
otherwise). Up to 64 rows the input is quantized by one Triton launch, so a
decode projection is two launches. Q, K, V and the output gate run as one FP8
GEMM: their weights and channel scales are put in one buffer at load (the
projections' parameters are views of it). With a vendor per-tensor FP8 GEMM
(cuBLASLt), the decode rows that the one-launch per-token quantizer takes (up to
64, SM100 and newer) of that GEMM and of the dense MLP's gate/up run it with
unit scales, and the kernel that reads the output applies
the per-token and per-channel scales: the Q/K norm (with RoPE or SSMax), which
also writes V and the output gate, and the SwiGLU, which also quantizes the
down projection's input. Layer 0's input norm, a plain RMSNorm, writes its
decode rows' FP8 codes in the same launch; the other boundaries' norms run in
the fused all-reduce kernel and the rows are quantized by the Triton launch.
The experts run on FlashInfer's TRT-LLM FP8 per-channel
routed MoE under the `fp8ch_vendor_fp32_route` policy: Beam's router kernels
choose the experts and their FP32 route weights (precomputed top-k), the input
is quantized per token (up to 64 rows by the one-launch Triton quantizer on
SM100 and newer, otherwise by TRT-LLM's), the kernel quantizes the FC2 input per
route, and the per-route
outputs are combined with the FP32 route weights in FP32. It needs
`--moe-backend flashinfer_trtllm` (Beam's default where it applies); another
backend is refused; Beam's default applies at any MoE partition. Beam's
partitions run unpadded: 320 per rank at TP8, and 32 whole experts of 2,560 per
rank under `--ep-size 8` with `--tensor-parallel-size 8` (MoE TP 1, as for the
other vendor formats). On GB200 the TP8 output is bit-identical to the same
experts padded to 384; other sizes pad to a multiple of 64 with zero rows that
add exactly zero. The router stays
FP32 and the LM head and embedding unquantized. Each load logs one line, `Beam
FP8 checkpoint (compressed-tensors FP8 dynamic W8A8) loaded as FP8, FP8 GEMMs
and FP8 experts run it: ...`, with the count per role. Other compressed-tensors
schemes (another weight or activation strategy, static activation scales,
sparsity or transforms) are refused. The kernels' cubins (per-token and
per-channel scaled E4M3 batched GEMMs) must be in FlashInfer's cubin cache for
an offline launch.

### NVFP4 (experts on the TRT-LLM MoE)

The NVFP4 checkpoint is ModelOpt NVFP4 with group size 16 that quantizes the
routed experts only: attention, the dense MLP layers and the LM head are stored
and run in BF16 (the router in FP32), and a quantized tensor of any of them is
refused. The expert tensors load unchanged into FlashInfer's TRT-LLM NVFP4
routed MoE under the `nvfp4_vendor_fp32_route` policy: Beam's router kernels
choose the experts and their FP32 route weights (precomputed top-k; the kernel
never routes), the kernel quantizes the BF16 input, and the output is the
kernel's finalized combine, as under the BF16 vendor policy. This runs only on
`--moe-backend flashinfer_trtllm`, Beam's default above with the NVFP4
alignment of 64 (TP8's 320 and TP4's 640 per rank, or 32 whole experts per rank
under `--ep-size 8`). Any other MoE backend fails startup with the reason, and
so does a launch left `auto` where the default does not apply (another platform
than SM100 to SM103, or an intermediate size per rank that is not a multiple of
64). Each expert's `weight_scale_2` must be one positive finite FP32 value and
its `input_scale` one finite FP32 value, positive or 0 (an expert no
calibration token reached): the MoE uses each layer's largest input scale, and
a layer whose w13 or w2 input scales are 0 in every expert is refused. Each
load logs `Beam NVFP4 checkpoint (ModelOpt NVFP4, experts only) loaded: ...`
with the counts. The expert GEMMs run FlashInfer's TRT-LLM-gen cubins, which
FlashInfer downloads when its cubin directory lacks them; with
`FLASHINFER_NO_DOWNLOAD=1` stage them beforehand, as for the attention backend.

## Batches above 160

Without batch flags a launch runs at most 160 requests at once
(`--max-num-seqs` defaults to 160) and captures decode CUDA graphs for batches
of 1, 2, 4 and every multiple of 8 up to 160 (`--max-cudagraph-capture-size`
defaults to 160 on GPUs with 25 GB of memory or more). With the DFlash2
drafter (above) both default to 80 instead: at most 80 requests at once, and
decode graphs for 1, 2, 4, every multiple of 8 up to 80, and 79. A decode
batch replays the graph of the smallest captured size that holds it, padded
to that size; a batch above the largest captured size runs eagerly, without a
graph. A larger batch therefore takes all three flags; for 512 requests,
`--max-num-seqs 512`, `--max-cudagraph-capture-size 512` and
`--cudagraph-capture-sizes 1 8 32 64 128 160 256 384 512`.

`--max-num-seqs 512` alone, or with `--max-cudagraph-capture-size 512`, keeps
the default list, so every decode batch from 161 to 512 runs eagerly (from 81
with the drafter and `--max-num-seqs 512` alone); a listed size above
`--max-cudagraph-capture-size` (160 when it is not given, 80 with the drafter)
is dropped. `--cudagraph-capture-sizes` replaces the default list: under the
list above a batch of 9 replays the 32-request graph. Listing more sizes pads
less and costs startup capture time and graph memory per size. The `trtllm`
attention backend picks its decode kernel per captured size (above), so each
new size is a new batch bucket.

`--max-num-seqs` reserves no KV cache. The KV pool takes what the startup
memory profile leaves after the weights, the headroom that
`--gpu-memory-utilization` keeps free (5% of the GPU by default) and the
CUDA-graph reserve, which grows with the captured sizes; `--max-total-tokens`
caps it. Running requests share the pool, so a batch reaches 512 only while the
pool holds all their KV. Each rank logs a `Device memory summary` table at
startup (weights, KV cache, free memory), and the scheduler its 64-token pages
(`num_device_pages`).

Beam's KV cache is BF16 in every precision. It is one pool of 64-token pages
shared by six cache groups: the 9 full-attention layers, and the 43
sliding-window layers split into five groups of at most 9
(`BeamConfig.cache_layer_types`; the split changes only where the KV is
stored, not what each layer keeps). A page belongs to one group at a time and
has one slot per layer of the largest group: 9 slots, or 14 with the DFlash2
drafter, whose 5 layers keep KV for the whole context in the full-attention
group. At TP8 a slot holds one of the 8 KV heads (head dim 128, K and V), 512 B
per token, so the cache takes 4,608 B per token per GPU, or 7,168 B with the
drafter. Each halving of the TP size doubles these (TP4: 9,216 B and 14,336 B).
A full-attention page stays for the whole request. A sliding page leaves the
pool once its tokens slide out of the 512-token window, so a request keeps
about 10 pages per sliding group, about 50 in all, at any context length. A
full-context request therefore needs 16,384 full-attention pages, 4.5 GiB
per GPU at TP8 (7 GiB with the drafter), plus its sliding pages (under 30
MiB). The pool's token count, which `--max-total-tokens` caps and the startup
log reports, is its pages times 64 over all groups, sliding pages included, so
each running request also takes about 3,200 of it for its sliding pages.
With the drafter, the memory summary's `KV cache` row counts only the target's
9 slots of each page (the drafter's 5 fall in `Activations + device graphs`).
Each rank's `Cache profile` log line gives the whole pool: `parents` pages of
`parent_bytes` each.

Measured at startup on GB200, TP8 over two nodes, with the trained checkpoints
and the final DFlash2 drafter, `--max-model-len 1048576`,
`--enable-mixed-batch`, `--prefill-graph-max-tokens 8192`, the default 0.95
GPU memory utilization and no `--max-total-tokens`:

| Checkpoint | Target weights per GPU | KV pool (tokens) |
| --- | --- | --- |
| NVFP4 | 36.5 GiB | 18.8M |
| FP8 | 61.2 GiB | 15.2M |
| BF16 | 119.1 GiB | 6.53M |

The final drafter adds 1.40 GiB of weights per GPU. KV capacity can vary
slightly with free memory at startup.

The Host KVStore (Host L2) stays on by default. Each rank sizes it to fit its
share of the host memory its node has available: when the default size, twice
the rank's KV pool, does not fit, the rank uses a smaller size that fits on
every rank and logs one warning (`Capped the compact Host L2 ...`).

Measured decode-step throughput on GB200, TP8 over two trays of four GPUs (one
MNNVL domain), **with random-valued checkpoints in Beam's shapes (BF16, and the
NVFP4 format above), not the trained weights**; Beam's defaults with
`--max-model-len 33792` and no drafter, greedy sampling, prompts of 16 to 48
tokens, and the batch exactly at a captured size:

| Requests | NVFP4 step (ms) | NVFP4 tok/s per GPU | BF16 step (ms) | BF16 tok/s per GPU |
| --- | --- | --- | --- | --- |
| 160 (default limit) | 11.17 | 1,791 | 23.76 | 842 |
| 256 | 11.59 | 2,761 | 24.54 | 1,304 |
| 384 | 13.73 | 3,496 | not measured | not measured |
| 512 | 14.37 | 4,454 | not measured | not measured |

The step is the 52 decoder layers' time on the GPU timeline (the median layer
period times 52, median over the 8 GPUs); tok/s per GPU is requests / step / 8.
The LM head, sampling and per-step host work are not in it (for NVFP4 the whole
decode graph replay was 1% longer at 160 requests and 6% at 512). A second
NVFP4 launch agreed within 1%. These are decode steps only: no prefill,
queueing, gateway or time to first token. The BF16 columns were measured on an
earlier build, before the router's 17 to 96 and 97+ row kernels and the
cross-node fusion of the all-reduce boundaries (above) were added; the NVFP4
columns include them. The random-valued checkpoints route tokens to experts
differently from the trained weights, which changes the expert GEMM time. With
these short prompts attention kernels took 0.37 ms of the NVFP4 step at 64
requests; with 8K-token prompts 0.81 ms and with 32K-token prompts 1.76 ms.

Every running request gets one token per step, so a longer step is a longer
time per output token. From 160 to 512 requests NVFP4 throughput per GPU grew
2.5x and the step 1.29x (11.2 to 14.4 ms); 256 requests gave 1.5x for 1.04x,
and BF16 at 256 1.55x for 1.03x. Prefill adds to that: by default (no
`--enable-mixed-batch`, as in these runs) a prefill step runs alone, so each
arriving prompt pauses every running decode for its prefill (in the NVFP4
launch about 33 ms for a 2,040-token prompt and 142 ms for an 8,192-token
chunk), and more running requests bring more such pauses and a longer queue
before the first token. The two flags under "Serving" run a prompt chunk in
the same step as the running decodes instead. Time to first token and
inter-token latency under an arrival load were not measured in these runs.

## Chat format (`ts serve`)

The chat format is the checkpoint's own: `chat_template.jinja` renders the
prompt, and `tokenizer_config.json` names Beam's output format
(`reflection_grammar_version: harmony_xml`). TokenSpeed changes neither.

The supported checkpoints are these Hub revisions:

| Checkpoint | Revision |
| --- | --- |
| BF16 | `f40a3a29` |
| FP8 | `21052137` |
| NVFP4 | `81a5ab6b` |

All three have the same `chat_template.jinja`, `tokenizer.json`,
`tokenizer_config.json` and `generation_config.json`, byte for byte, so prompt
token ids, the stop ids (`<|return|>`, `<|call|>`) and xgrammar's tokenizer
info are the same in all three. Pass the local checkpoint directory as the
model, the same path on every node: a hub id resolves through the cache's
`refs/main`, which can name another revision.

The gateway must recognize Beam to split the output into reasoning, content
and tool calls. `tokenspeed-smg==<NEW_SMG_VERSION>`, the version TokenSpeed pins, does it
from the checkpoint alone: it compiles the `response_template` that
`tokenizer_config.json` declares into the reader of the output. The gateway
also needs smg#3013 or later: the chat template uses transformers'
`{% generation %}` blocks, and an older gateway fails to load the tokenizer
(`unknown statement generation`) and answers every chat request with 500.
Earlier releases do not recognize Beam, and those built before smg#3013 merged
fail this way. A gateway that does not recognize Beam returns the raw output
markup (control tokens such as `<|channel|>` and `<|start|>`, and the XML
parameter tags of tool calls) as `content`, with no `reasoning_content` or
`tool_calls`.

The seven control tokens (`<|start|>`, `<|end|>`, `<|message|>`,
`<|channel|>`, `<|constrain|>`, `<|return|>`, `<|call|>`) are ordinary added
tokens (`special: false`), and `skip_special_tokens` strips at most the EOS
`<|return|>`. The gateway's template reader parses them out; every other path
returns them in the text: `separate_reasoning: false`, an explicit
`--tool-call-parser` or `--reasoning-parser`, `passthrough` included (naming
either parser turns the template reader off for both), and the text of the
Engine API.
The supported path is `ts serve` with no parser named. When the `--model`
directory holds Beam's tokenizer files, whose `tokenizer_config.json` declares
a `response_template`, and no other `--tokenizer` is given, it leaves both
parsers to the gateway and prints one line saying so. Otherwise it passes
`--reasoning-parser passthrough`.

The template maps the `system` and `developer` roles to one developer
`# Instructions` block. Consecutive ones at the start of `messages` are
joined, a blank line apart, and empty ones are left out. A `system` or
`developer` message after the first message of another role is not rendered,
and no error says so: send instructions first. A `bash` tool whose only
parameter is an unconstrained `command` string is declared in the system block
with the definition Beam was trained with, and the request's descriptions of it
are not rendered. Any other `bash` tool is an ordinary tool.

Multi-turn requests should send back each earlier assistant turn's reasoning
unchanged: `reasoning_content` (Chat), the reasoning item (Responses) or the
thinking block (Messages); the template notes that dropping it degrades the
model's output. By default the template renders an assistant turn whose
reasoning is missing or empty with an empty reasoning block, and no error says
so. A Chat request that passes `chat_template_kwargs:
{"allow_missing_thinking": false}` and has such a turn is refused instead: the
template refuses the generation prompt, and the gateway answers 400 with the
template's message. The template recommends `false` for serving, but neither
`ts serve` nor the bundled Day-0 gateway sends it, the kwarg is not part of
the provider's API, and a Responses or Messages request has no way to pass it
on the bundled Day-0 gateway. The assistant turns that gateway rebuilds
without reasoning for a Responses request (from `previous_response_id` or a
`conversation`, or after its MCP tool loop ran a call) render the same way.
Keeping the reasoning through stored conversations and the MCP tool loop is a
follow-up.

A request's `seed` keys its random draws on the seed and the token position,
so the same request with the same seed draws the same random numbers whether or
not its prompt hits the prefix cache, and however its prefill is chunked. The
tokens can still differ when batching or the prefill's shape changes the
logits: the kernels are not batch-invariant.

Logprobs need `--enable-output-logprobs` on the launch, including prompt
logprobs (`echo` with `logprobs` on `/v1/completions`), which the token parity
check of the provider's `beam-verifier` sends.

### Reasoning effort

The template alone renders `Reasoning effort: 512` (`max`) when a request names
no effort or passes `null`. It takes an integer or the labels `minimal`, `low`,
`medium`, `high`, `xhigh` and `max`, which render 8, 32, 64, 128, 256 and 512;
`minimal` is the smallest effort the model was trained with, and a smaller
number is not respected. The provider's API defaults to `medium` and accepts
only `low`, `medium`, `high`, `xhigh` and `max`. The bundled Day-0 gateway
applies the API's contract to Beam itself, with no flag to set: a request that
names no effort (Chat `reasoning_effort` or `thinking.effort`, Responses
`reasoning.effort`, Messages `output_config.effort`) renders
`Reasoning effort: 64`, the five values render 32, 64, 128, 256 and 512, and
any other value is a 400. Chat
`reasoning_effort` and Messages `output_config.effort` are lenient and take a
string or a number: a number, or a string that reads as one (`"64"`), is
`invalid_type`, and any other string is `invalid_value` (`none` and `minimal`
included). `thinking.effort` takes a string only: the same two codes for a
string, while a JSON number fails request parsing (`json_parse_error`).
Responses `reasoning.effort` is a strict enum of OpenAI's labels: `none` and
`minimal` are `invalid_value`, and any value that is not one of the labels
fails request parsing (`json_parse_error`). `chat_template_kwargs:
{"reasoning_effort": ...}` reaches the template unchecked (`minimal` renders 8,
`null` 512, a number as given); it is not part of the provider's API either.
Without this handling the template's 512 applies, the longest reasoning: on a
gateway that does not recognize Beam, and in any other render of the template
that names no effort, such as a direct `apply_chat_template` call. Naming either
parser (`--reasoning-parser` or `--tool-call-parser`, `passthrough` included)
turns this handling off, so the template's 512 applies.

### Constrained outputs

The launch enables the `xgrammar` grammar backend by default (see "Serving").
This is what lets forced tool
choice (`required`, a named function) and JSON `response_format` work.
With an explicit `--grammar-backend none`, such a request is a 400. Every node
must pass the same explicit `--grammar-backend`, or all must omit it. A grammar
that does not compile is a 400 as well. The gateway constrains these requests
to Beam's own call format only when it recognizes Beam (above).
Under `tool_choice` `auto` (the default with tools) and no JSON
`response_format` the output is not constrained, so a tool's `strict: true` is
not enforced. Each rank keeps the grammars that requests were admitted with,
up to 1 GiB of their estimated size, and drops the least recently used beyond
that; one of Beam's structural tags (forced calls, or a JSON response or calls)
holds about 37 to 106 MB of host memory for 1 to 128 tools. With attention
TP > 1 (TP8) the ranks compile on their own but admit a constrained request
together: a request whose schema or tool set is still compiling on some rank
waits until every rank has it, and one whose grammar failed or timed out on
any rank is refused on every rank.

## Beam's kernels

Beam calls its fused kernels directly where they were checked byte for byte
against the Torch statements they replace (NVIDIA SM100 to SM103, validated on
SM100) and the call is inside the kernel's host guard; any other call (another
platform, an empty batch, a layout the guard refuses) runs the statements. The
sliding layers' Q/K norm with RoPE runs its kernel on every GPU, as before. Two
call sites choose among kernels: the router's `mv` by row count, and
the NoPE Q/K norm, a registered operator whose ranked selection the registry
override surface (`TOKENSPEED_KERNEL_OVERRIDE_LAYERNORM_QK_RMSNORM_SSMAX=...` or
a `tokenspeed_kernel.selection.kernel_override()` context, set before CUDA-graph
capture) can replace.

| Call site | Kernel | Otherwise |
| --- | --- | --- |
| `BeamRouter.forward` (FP32 expert router on the BF16 rows) | `gemm.mv`, by row count: `triton_rowcta_gemm_fp32` (1-16 rows); `gluon_simt_gemm_fp32` (17-96 rows on NVIDIA: FP32 CUDA cores on the FP32 weight in one fixed order, the same bits for any row count and tile); `triton_bf16x3_gemm_fp32` (97+ rows on SM100 to SM103: three BF16 tensor-core MMAs per 64-wide K block on the exact BF16 split of the FP32 weight that the router makes once after loading) | `torch_decode_gemv` (`x.float() @ weight.t()`, cuBLAS FP32) |
| `beam_route_weights` (the grouped sigmoid route) | `triton_router_route_lanepar`: one launch, one row per program on one warp, for Beam's routing shape (Torch's FP32 sigmoid recipe, the selection with Torch's radix keys and fill order, the epilogue with the Torch CUDA K8 sum order) | `torch_grouped_route` (the route's statements) |
| `BeamRMSNorm.forward` (residual add + RMSNorm) | the shared `rmsnorm`'s residual arm with `round_residual_sum_bf16=True`: the FP32 sum rounded once to BF16 in the kernel and, with the depth-scale pair, `residual_pair`'s two products rounded to BF16 before the add (`SCALE_INPUTS_BF16`), for BF16 contiguous `[rows, 5120]` inputs | the Torch products and BF16 add, then the shared `rmsnorm` |
| `BeamAttention.prepare_qkv` (NoPE-layer Q/K RMSNorm + SSMax scale) | `layernorm.qk_rmsnorm_ssmax`, ranked: `triton_fused_qk_rmsnorm_ssmax` (one launch, the FP32 scale computed in the kernel from the position) for the per-rank head pairs 48/8 (TP1) and 6/1 (TP8) | `torch_qk_rmsnorm_ssmax` (other TP sizes and platforms) |
| `BeamAttention.qk_norm` (sliding-layer Q/K RMSNorm + RoPE) | `fused_qk_rmsnorm_rope` on every GPU: one program per token and head; with `dequant` it also applies an unscaled FP8 GEMM's scales | none |
| `BeamLogitsProcessor` (final logit softcap) | `tanh_softcap`: one pass over 2-D CUDA logits with dense columns and a cap whose FP32 value and reciprocal are positive normal numbers | `(x.float() / cap).tanh() * cap` |

All but the RoPE norm are byte-identical to their statements (the RoPE norm
reduces the squares in another order). The route's byte identity rests on this
torch build's FP32 CUDA sigmoid (`1 / (1 + expf(-x))`, IEEE add and div,
non-FTZ), its CUDA K8 sum order (`((v0+v4)+(v2+v6))+((v1+v5)+(v3+v7))`, then
`+1e-20`, then `div.rn`) and its `topk` tie order, all pinned by
`tokenspeed-kernel/test/nvidia/ops/moe/test_router_select_cuda.py` (a mismatch
fails, it never skips): a torch upgrade must re-run that test. The residual
kernel absorbs the 104 residual adds of a step: the 52 attention-boundary adds
that `BeamDecoderLayer.attn_residual_norm` routes through
`post_attention_layernorm(o, residual=r, x_scale=..., residual_scale=...)`, and
the 52 MLP adds whose pair layer i+1's `input_layernorm` and the final norm
(`BeamResidualFoldRMSNorm`) hold as `residual_fold`. `--numerics rl-bitwise`
pins none of these kernels.

Attention stays the backend's: Beam passes no attention override under any
`--attention-backend`, its `trtllm` default (above) included. The attention
output gate is not a default: every backend returns the attention output
ungated, and `BeamAttention.forward` gates it once,
computing gate + bias, the sigmoid and the product in FP32 and rounding once to
BF16. On any NVIDIA GPU this is one in-place launch of the shared
`sigmoid_mul(attended, gate, head_bias=gate_bias)` (the kernel's biased
branch, at floor 0 and temperature 1). That
branch calls `libdevice.div_rn`, which ROCm's libdevice does not have, so on
other platforms the forward runs the same expression in Torch
(`apply_attention_output_gate` at floor 0 and temperature 1, a new tensor)
instead of failing to compile (unlike the kernel, which flushes a sigmoid below
the FP32 normal range, gate + bias below about -87.3, to 0, Torch keeps it).
The checkpoint reference rounds each of the three steps to BF16 instead, so
this is an FP-order difference, not a byte-identical default; no override
restores the reference chain.

On the vendor MoE with Beam's routes the
route runs as on the Triton MoE: the vendor layer keeps `BeamMoE._route`, and
the route epilogue folds the routed scale once into its fresh FP32 weights
(the bytes of TopK's multiply, one kernel fewer); the int32 ids pass through.
