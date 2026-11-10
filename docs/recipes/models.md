# Model recipes

These recipes start from a known model family, pick the hardware topology, then
set only the parameters that change runtime behavior.

The commands below are templates. Validate exact model IDs, checkpoint formats,
and backend choices against the build you deploy.

## Muse Spark Open

[Muse Spark Open](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B)
runs on the shared model path: vendor-neutral Triton kernels
with capability-gated NVIDIA fast paths that keep the same numerics. The
checkpoint must expose the model type `muse_spark` and the architecture
`MuseSparkForConditionalGeneration`; the runtime reads the sliding-attention
and no-RoPE schedule from that configuration. The gateway loads the
checkpoint's `chat_template.jinja`; pass `--chat-template` only to override
it. The residual stream, the layer boundaries, and the final norm run in FP32;
the attention and MoE branch outputs and the final-norm output are BF16.

`ts serve` selects the `muse` reasoning and tool parsers: the `self` channel
becomes `reasoning_content`, `user` and `commentary` stay visible, `<|eom|>`
closes a channel, and `<|eot|>` closes the turn. The bundled gateway reads them
with its Symphony parser, which `ts serve` selects with `--parser symphony`;
forced tool calls and JSON answers run through the engine's xgrammar backend,
which `ts serve` turns on for this model (`--grammar-backend xgrammar`). These
parser and grammar-backend defaults need a `tokenspeed-smg` build with smg's
Muse Spark support (smg main from 2026-10-16 on). Explicit parser and
grammar-backend flags override these defaults. Sample with the checkpoint's
`generation_config.json` defaults
(`temperature=1.0`, `top_p=1.0`, `top_k=64`); greedy decoding is loop-prone
for this checkpoint. The checkpoint's
`chat_template.jinja` renders `reasoning_effort` (`minimal`, `low`, `medium`,
`high`, `xhigh`) as a reasoning-strength line and defaults to `medium`; the
OpenAI field and `chat_template_kwargs.reasoning_effort` both reach it.
`chat_template_kwargs.force_nonempty_content=true` promotes an unterminated
`self` channel to visible content.

Images and video use the chat template's `<image>` and `<video>` anchors and
the checkpoint's vision tower. Audio uses the `<audio>` anchor: one `<|audio|>`
token per 1,920 samples of mono 24 kHz, encoded by the checkpoint's audio
tower. SMG performs the preprocessing and anchor expansion.

### MI355X MXFP4

The
[AMD Quark MXFP4](https://huggingface.co/amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4)
checkpoint quantizes the routed-expert weights and activations of all but the
first and last layers to MXFP4 and keeps everything else in BF16. On eight
MI355X GPUs (gfx950), MoE TP8 with EP1 shards every expert over its
intermediate dimension, so each GPU holds all 256 experts and no tokens are
dispatched between GPUs. The launch below adds the
[DFlash2](https://huggingface.co/RadixArk/Muse-Spark-Open-DFlash2) draft and an
FP8 KV cache. The draft proposes seven tokens per step, and
`--enable-speculative-sampling` verifies them by rejection sampling, so sampled
requests keep the target model's output distribution:

```bash
DEBUG_HIP_FORCE_GRAPH_QUEUES=1 DEBUG_HIP_GRAPH_BATCH_SIZE=1 ts serve \
    --model amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4 \
    --served-model-name muse-spark \
    --tp 8 \
    --moe-tp-size 8 \
    --ep-size 1 \
    --dtype bfloat16 \
    --quantization mxfp4 \
    --moe-backend gluon \
    --kv-cache-dtype fp8_e4m3 \
    --max-model-len 1048576 \
    --max-num-seqs 2 \
    --prefill-decode-interval 192 \
    --max-prefill-tokens 8192 \
    --chunked-prefill-size 8192 \
    --cudagraph-capture-sizes 1 2 3 4 5 6 8 12 16 32 64 \
    --max-cudagraph-capture-size 64 \
    --disable-prefill-graph \
    --gpu-memory-utilization 0.92 \
    --sampling-backend triton \
    --disable-kvstore \
    --kvstore-ratio 0.0 \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path RadixArk/Muse-Spark-Open-DFlash2 \
    --speculative-draft-model-quantization fp8_weight_only \
    --speculative-num-steps 7 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 8 \
    --enable-speculative-sampling \
    --host 127.0.0.1 \
    --port 8000
```

`--max-model-len 1048576` keeps the checkpoint's native context, which agentic
workloads with long cached prefixes need; prefix caching is on by default.

`--speculative-draft-model-quantization fp8_weight_only` streams the draft's
MLP, `o_proj` and LM-head weights in FP8 with one scale per output row, which
shortens each draft step at unchanged acceptance.

With MXFP4 experts at MoE TP8 and EP1, startup also builds a column-split copy
of the expert down-projection weights for prefill. The copy holds one more set
of those weights on each GPU, which leaves less memory for the KV cache. While
the copy exists, TokenSpeed refuses live weight updates.

The two HIP runtime settings are needed for speed; TokenSpeed does not set them
itself. `DEBUG_HIP_FORCE_GRAPH_QUEUES=1` matters most for the two-stream
speculative decode graph. The extra capture sizes (3, 5, 6, 12) cut padding at
the small batch sizes that speculative decoding runs at.

For multi-turn agentic traffic, set `--max-num-seqs` and
`--prefill-decode-interval` by concurrency. The earlier launch uses the row for
a concurrency of 2:

| Concurrency | `--max-num-seqs` | `--prefill-decode-interval` |
| ----------- | ---------------- | --------------------------- |
| 1           | 1                | 0                           |
| 2           | 2                | 192                         |
| 4           | 2                | 96                          |
| 8           | 2                | 256                         |
| 16          | 3                | 384                         |

A `--max-num-seqs` below the concurrency queues the extra requests. With an
interval of N, the scheduler runs N decode rounds after each prefill chunk
before the next one. Both keep decode batches small and uninterrupted, which
lowers TPOT, but queued and deferred prompts wait longer for their first token.
At a concurrency of 8, time-to-first-token is noticeably higher. If
time-to-first-token matters more, raise `--max-num-seqs` and lower the
interval.

`--prefill-decode-interval` needs a `tokenspeed-scheduler` release that
includes the prefill/decode interval. With an older scheduler, leave it at 0.

### Blackwell NVFP4

The NVFP4 checkpoint quantizes the experts of all but the first and last
layers and keeps everything else in BF16. On a four-GPU B300 node the decode
path pairs TRT-LLM attention with the TRT-LLM FP4 MoE kernels:

```bash
ts serve \
    --model meta-models/Muse-Spark-Open-816B-A42B-NVFP4 \
    --served-model-name muse-spark \
    --tp 4 \
    --ep-size 4 \
    --attention-backend trtllm \
    --moe-backend flashinfer_trtllm \
    --dtype bfloat16 \
    --max-model-len 65536 \
    --max-num-seqs 8 \
    --cudagraph-capture-sizes 1 2 4 8 \
    --max-cudagraph-capture-size 8 \
    --sampling-backend flashinfer
```

Overlap scheduling and the prefill graph stay on: the graph serves text and
multimodal prefills up to its largest bucket, where it cuts short-prompt
time-to-first-token by several times. Decode is weight-stream bound, so the layer
fuses its norms, residual gate, expert-input quantization, and the sharded
pre-expert and post-expert projection gathers into few launches, all under programmatic
dependent launch; at decode batch sizes the top-k kernel also writes the
expert GEMMs' routing metadata, so the experts launch without a routing
kernel of their own.

Expert inputs are scaled by the checkpoint's calibrated static input scales,
folded into the MoE kernel's dequantization scalars.

`--kv-cache-dtype fp8_e4m3` halves the KV cache on the same decode path; the
RoPE kernel writes K/V in FP8 directly.

## Inkling

Blog: https://lightseek.org/blog/tokenspeed-inkling.html

```bash
## Docker

### nvidia
docker pull lightseekorg/tokenspeed:latest
### amd
docker pull lightseekorg/tokenspeed-amd:latest

## Launch command

# nvidia
ts serve \
    --model thinkingmachines/Inkling-NVFP4 \
    --attn-tp-size 4 \
    --moe-tp-size 4 \
    --max-model-len 81920 \
    --max-num-seqs 16 \
    --max-prefill-tokens 8192 \
    --chunked-prefill-size 8192 \
    --disable-cuda-graph-padding \
    --trust-remote-code \
    --attention-backend fa4 \
    --moe-backend flashinfer_trtllm \
    --enable-prefix-caching \
    --disable-kvstore \
    --speculative-algorithm MTP \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4

# amd
ts serve \
    --model lightseekorg/Inkling-MXFP4 \
    --attn-tp-size 4 \
    --moe-tp-size 4 \
    --max-model-len 81920 \
    --max-num-seqs 16 \
    --max-prefill-tokens 8192 \
    --chunked-prefill-size 8192 \
    --disable-cuda-graph-padding \
    --trust-remote-code \
    --enable-prefix-caching \
    --disable-kvstore \
    --speculative-algorithm MTP \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4
```

## MiniMax M3

MiniMax M3 uses 128-token MSA blocks. TokenSpeed configures its dense and sparse
attention layers automatically. Select the dense backend with
`--attention-backend` and run with `--disable-kvstore`.

### EAGLE3 draft

```bash
tokenspeed serve nvidia/MiniMax-M3-NVFP4 \
    --tensor-parallel-size 4 \
    --max-model-len 81920 \
    --max-num-seqs 16 \
    --max-prefill-tokens 8192 \
    --chunked-prefill-size 8192 \
    --disable-cuda-graph-padding \
    --attention-backend trtllm \
    --kv-cache-dtype fp8 \
    --moe-backend flashinfer_trtllm \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path Inferact/MiniMax-M3-EAGLE3 \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --disable-kvstore \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000
```

### DSpark draft

`nvidia/MiniMax-M3-DSpark` is a six-layer Qwen3-shaped GQA block drafter with a
vanilla Markov head. Keep the target launch shape and swap the speculative
options:

```bash
tokenspeed serve nvidia/MiniMax-M3-NVFP4 \
    --tensor-parallel-size 4 \
    --max-model-len 262144 \
    --max-num-seqs 16 \
    --max-prefill-tokens 8192 \
    --chunked-prefill-size 8192 \
    --disable-cuda-graph-padding \
    --attention-backend trtllm \
    --kv-cache-dtype fp8 \
    --moe-backend flashinfer_trtllm \
    --speculative-algorithm DSPARK \
    --speculative-draft-model-path nvidia/MiniMax-M3-DSpark \
    --speculative-num-steps 8 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 9 \
    --disable-kvstore \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000
```

Notes:

- This checkpoint's `block_size` is 8, and a DSpark `block_size` is the drafted
  token count, so launch it with `--speculative-num-steps 8` and
  `--speculative-num-draft-tokens 9`: the verify window is one anchor row plus
  eight draft queries. Startup checks both widths against the checkpoint, so a
  mismatched launch fails fast instead of drafting a wrong-width block. See
  [Speculative decoding](../configuration/server.md#speculative-decoding)
  for the DSpark and DFlash conventions.
- The draft writes its KV at the target's cache locations and shares the
  target's page table, so it inherits the target's 128-token MSA prefix
  granularity.
- The draft's 1024-token sliding window is an attention mask its own layers
  apply. It is deliberately not a cache-retention policy, because the draft's
  pages are the target's pages.
- The target model captures features from the residual stream after each
  layer in `dflash_config.target_layer_ids` (`[1, 12, 23, 35, 46, 57]` of M3's
  60 layers); the draft concatenates them in ascending layer order to feed its
  `fc`.
- The draft checkpoint stores fp32 master weights. TokenSpeed loads it in the
  target's dtype rather than the standalone fp32-to-fp16 default, because the
  two exchange hidden states and share the target's embedding and LM head.
- Measured on 4x GB300 with the launch above: gsm8k `mean_acc` 0.9719 versus
  0.9704 without speculative decoding (paired disagreement 10 versus 8, McNemar
  p ~ 0.81 -- within run-to-run noise), so the draft does not move accuracy.

## MiniMax M3.1 {#minimax-m3-1}

For checkpoints that declare `quant_method: mxfp8` with
`moe_quant_algo: NVFP4`, TokenSpeed automatically selects `mxfp8_nvfp4`:
projections use MXFP8 and routed experts use NVFP4. This quantization requires
Blackwell.

Download the target and bundled DSpark draft together. The draft is in the
`dspark/` subdirectory and declares `DSparkMiniMaxDraftModel`. This example uses
seven draft queries and an eight-token verify window; match those widths to
the draft checkpoint's configuration.

```bash
model_dir=$(hf download MiniMaxAI/MiniMax-M3.1-Flash --quiet)

tokenspeed serve "$model_dir" \
    --tensor-parallel-size 4 \
    --max-model-len 1048576 \
    --gpu-memory-utilization 0.95 \
    --attention-backend trtllm \
    --kv-cache-dtype fp8 \
    --moe-backend flashinfer_trtllm \
    --speculative-algorithm DSPARK \
    --speculative-draft-model-path "$model_dir/dspark" \
    --speculative-num-steps 7 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 8 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000
```

Set `--speculative-num-steps` to the DSpark checkpoint's `block_size` and
`--speculative-num-draft-tokens` to `block_size + 1`. See
[Speculative decoding](../configuration/server.md#speculative-decoding) for
the width conventions. To serve the target alone, drop the `--speculative-*`
options.

## Kimi K2.5 / K2.6

Kimi-style MoE launches usually need remote code, long context, reasoning and
tool parsers, and explicit MLA/MoE backends.

```bash
tokenspeed serve nvidia/Kimi-K2.5-NVFP4 \
  --served-model-name kimi-k2.5 \
  --trust-remote-code \
  --max-model-len 262144 \
  --kv-cache-dtype fp8 \
  --quantization nvfp4 \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --chunked-prefill-size 8192 \
  --max-num-seqs 256 \
  --attention-backend trtllm_mla \
  --moe-backend flashinfer_trtllm \
  --reasoning-parser kimi_k25 \
  --tool-call-parser kimik2 \
  --host 0.0.0.0 \
  --port 8000
```

For K2.6, keep the same parameter shape and change the checkpoint and parser
only if the model card requires a different value.

To enable a compatible DFlash draft model, keep the target launch shape and add
the draft model path plus DFlash speculative decoding options:

```bash
tokenspeed serve nvidia/Kimi-K2.6-NVFP4 \
  --served-model-name kimi-k2.6 \
  --trust-remote-code \
  --max-model-len 262144 \
  --kv-cache-dtype fp8 \
  --quantization nvfp4 \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --chunked-prefill-size 8192 \
  --max-num-seqs 256 \
  --attention-backend tokenspeed_mla \
  --moe-backend flashinfer_trtllm \
  --reasoning-parser kimi_k25 \
  --tool-call-parser kimik2 \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path /path/to/kimi-k2.6-dflash \
  --speculative-num-draft-tokens 8 \
  --speculative-num-steps 7 \
  --drafter-attention-backend fa4 \
  --host 0.0.0.0 \
  --port 8000
```

Official DFlash2 checkpoints that declare `DFlash2DraftModel` use the same
`--speculative-algorithm DFLASH` launch. TokenSpeed enables their grouped
dynamic convolutions and candidate selector automatically from the draft
architecture. On GPU each convolution runs as a single fused Triton kernel.
The torch node graph stays as the CPU reference. An MLA DFlash2 draft also
writes its context KV through one stacked projection plus one fused
norm/RoPE/scatter launch, which
lets that write overlap the draft forward and accumulate as the target
produces each captured layer. The server logs at INFO which of those paths it
took, and why when it declined one.
Draft proposals greedily follow the selector's transition-conditioned path.
Candidate selection takes each vocabulary shard's local top-k and gathers only
those, instead of gathering whole logits rows, whenever the head's shards carry
no padding or added tokens.
The target's verification step applies a request's `temperature`, `top_k`, and
`top_p`, never the proposal, so the served distribution is the target's
whatever the drafter proposed.

## Kimi K3

For Hopper MXFP4 serving with pipeline prefill, attention-DP decode, DeepEP,
DSpark and decode CUDA graphs, see the
[K3 Hopper PD configuration and validation guide](../guides/kimi-k3-hopper-pd.md).

For attention-DP serving with independently sharded QKV, output projections,
and shared experts, see
[Kimi-K3 DP with TP subgroups](../serving/parallelism.md#tensor-parallel-subgroups-within-dp).

Kimi-K3 combines a MoonViT vision encoder with a hybrid KDA
(linear-attention) / NoPE-MLA (full-attention) decoder and a
DeepSeek-V3-style latent MoE. The KDA layers currently use
flash-linear-attention kernels on NVIDIA, so install it first:

```bash
pip install flash-linear-attention
```

Notes:

- K3 uses the cache-group scheduler and KDA state groups.
- On Blackwell, `--dense-gemm-backend trtllm_cutedsl` opts the
  block-FP8 attention projections into TRT-LLM CuTe-DSL: KDA fused QKV/gates and output,
  and MLA fused QKV-a/gate, Q-b, and output. With `auto`, compatible GEMMs prefer
  CuTe-DSL at M > 16 rows and FlashInfer at M <= 16. Unsupported CuTe-DSL inputs
  retain compatible fallback selection. M is the actual GEMM row count,
  not necessarily request concurrency. Explicit `trtllm_cutedsl` also selects
  CuTe-DSL for smaller M. This shared option applies across models
  to standard 128x128 block-FP8 dense linears. BF16/NVFP4 linears, MXFP8,
  per-tensor FP8, specialized grouped projections, and routed experts are
  unchanged. It does not change tensor-parallel mapping.
- This backend preserves checkpoint FP8 values and their FP32 128x128 block
  scales. It does not requantize weights or convert scales to E8M0.
  Activations use the existing 1x128 FP8 quantization path. The kernel uses
  FP32 accumulation and BF16 output. Validate model accuracy on your workload.
  Preserving the quantization contract does not guarantee bitwise equality
  across GEMM implementations.
- It requires Blackwell, CuTe-DSL with TVM-FFI support, BF16 outputs, and aligned
  local weight dimensions. Kernel variants compile during preparation before
  CUDA-graph capture. Unsupported selected linears fail instead of silently
  falling back. Faster GEMMs alone do not establish an end-to-end speedup.
- KDA dispatch is vendor-neutral at the runtime boundary. The kernel registry
  selects the existing FLA-derived NVIDIA implementation or the native AMD
  implementation, including each backend's preferred recurrent-state layout.
  The runtime does not transpose or reinterpret that state.
- The NVIDIA default selects `tokenspeed_mla` (FP8 KV), whose current kernels
  support Blackwell. On Hopper, explicitly select `--attention-backend flashmla`
  or `mla` with `--kv-cache-dtype bfloat16`. AMD uses the `mla` backend.
- `tokenspeed serve` auto-selects the `kimi_k3` reasoning and tool-call
  parsers. Explicit parser flags override these defaults.
- The SMG packages pinned by TokenSpeed resolve `moonshotai/Kimi-K3` directly.
  You no longer need a flattened local checkpoint and a separately staged
  remote-code cache.
- K3's target and DSpark draft MLA caches follow the shared `--kv-cache-dtype`
  setting, as with EAGLE3: `fp8` selects FP8; `auto` and `bfloat16` select BF16.
- DSpark proposal blocks use non-causal MLA draft attention. Both the `mla` and
  `trtllm_mla` draft backends preserve every block row during eager execution
  and CUDA graph capture. When K3's 128-token logical cache pages feed the
  64-token TRT-LLM MLA kernel, the backend expands each logical page into its
  two physical kernel pages before draft attention.
- K3 DSpark context injection supports KVP (`--kv-parallel-size > 1`) by
  translating virtual slots into shard-local addresses and masking nonowner
  writes. The selected draft backend must support DCP decode and non-causal
  block attention on the device and cache dtype in use.
- A K3 DFlash2 draft declares `sliding_attention` layers, so it needs a drafter
  backend that applies per-layer sliding windows: `--drafter-attention-backend
  mla`. Those layers dispatch to the CuteDSL windowed decode on Blackwell,
  which walks the KV from the window rather than from token zero, and fall back
  to the portable Triton kernel anywhere its shape gate does not hold. The
  draft's full-attention layer is unaffected either way.
- For Kimi K3, an eight-token verify window uses seven DSpark draft queries.
  The anchor query directly predicts the first draft through the Markov head.
  Do not pad it with an eighth, unused mask row.
- The target model captures features from K3's completed-layer prefix stream
  before the model-level AttnRes mix and final norm, matching the DSpark
  checkpoint's vLLM training and inference contract. A draft trained instead
  against the pre-norm AttnRes mixture declares
  `"aux_hidden_stream": "attn_res"` in its config, and TokenSpeed serves it
  that stream. Feeding a draft the other stream raises nothing and shows up
  only as a lower acceptance rate, so the server logs the choice next to the
  tap ids at startup.
- A draft whose config sets `"fc_norm": true` normalizes each target tap on its
  own before it concatenates and projects the taps, and ships one
  `fc_norm.N.weight` per tap. Declaring it without the weights (or shipping the
  weights without declaring it) fails the load rather than serving an
  identity-weight norm.
- Under tensor parallelism, the TP ranks reduce the draft's final row-parallel
  MLP output before `final_norm` and shared target-head sampling.
- The vision encoder has 12 attention heads. For an 8-way text TP deployment,
  use `--mm-encoder-tp-mode data` so each rank runs the vision encoder at TP1
  on a different whole image.
- The pinned SMG frontend registers Kimi-K3's chat renderer and multimodal
  processor. Preserve the checkpoint's
  `media_proc_cfg.in_patch_limit=65536`. Silently falling back to K2.5's
  16384-patch default reduces OCR resolution.
- KDA recurrent-state pages register for prefix-cache reuse only when a
  prefill chunk ends exactly on a logical cache-page boundary. The engine floors
  `--chunked-prefill-size` to the plan's page grain automatically (logged as
  a warning when it adjusts). The page grain is budget-dependent (such as 1472
  at 32k context, 1536 at 1M), so do not hand-tune the chunk size against a
  hard-coded page value. Prefix hits are page-granular.

### NVIDIA

Serve with expert parallelism (recommended) on 8x B300:

```bash
tokenspeed serve moonshotai/Kimi-K3 \
  --served-model-name kimi-k3 \
  --trust-remote-code \
  --max-model-len 32768 \
  --kv-cache-dtype fp8 \
  --tensor-parallel-size 8 \
  --mm-encoder-tp-mode data \
  --ep-size 8 \
  --moe-backend flashinfer_trtllm \
  --max-num-seqs 32 \
  --disable-kvstore \
  --host 0.0.0.0 \
  --port 8000
```

Plain TP8 (drop `--ep-size 8`) works too. The fused MoE path needs a
Blackwell GPU (B200/B300). On other NVIDIA platforms, use
`--moe-backend triton`.

### AMD

On 8x MI350X/MI355X (gfx950), serve with the Eagle3 draft at TP8 and expert
parallelism disabled. This is the configuration of the per-commit MI35x
50K-input/500-output perf job. Against the speculator-free TP8/EP8 launch
below, TP8/EP1 cuts prefill latency enough to improve end-to-end latency at
concurrency 16, despite a higher per-token decode latency.

```bash
tokenspeed serve moonshotai/Kimi-K3 \
  --served-model-name kimi-k3 \
  --trust-remote-code \
  --max-model-len 65536 \
  --kv-cache-dtype fp8 \
  --tensor-parallel-size 8 \
  --ep-size 1 \
  --mm-encoder-tp-mode data \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path lightseekorg/kimi-k3-eagle3-mla \
  --speculative-num-steps 3 \
  --speculative-num-draft-tokens 4 \
  --speculative-eagle-topk 1 \
  --speculative-draft-model-quantization unquant \
  --eagle3-layers-to-capture 2,46,90 \
  --attention-backend gluon \
  --drafter-attention-backend gluon \
  --moe-backend auto \
  --max-num-seqs 16 \
  --chunked-prefill-size 8192 \
  --max-prefill-tokens 8192 \
  --max-cudagraph-capture-size 16 \
  --cudagraph-capture-sizes 1 2 4 8 16 \
  --disable-prefill-graph \
  --gpu-memory-utilization 0.92 \
  --disable-kvstore \
  --host 0.0.0.0 \
  --port 8000
```

- `gluon` requires Gluon MLA kernels for both the target and its MLA drafter.
  The explicit policy fails fast when the current GPU, dtype, or attention shape
  has no registered Gluon kernel, instead of silently selecting another
  solution. `--attention-backend mla --drafter-attention-backend mla` lets
  TokenSpeed select the MLA kernels automatically.
- `--disable-prefill-graph` skips prefill CUDA graphs, which need 40 buckets on
  top of the decode ones and run out of memory at this memory fraction.
  Speculative decoding runs on the decode graphs.
- To raise `--max-num-seqs`, extend `--cudagraph-capture-sizes` and
  `--max-cudagraph-capture-size` with it.
- AMD defaults to greedy sampling, which ignores per-request `temperature`,
  `top_p`, and `top_k`.

Without speculative decoding, use TP8/EP8. Automatic MoE selection then uses
the specialized Gluon SiTU kernels:

```bash
tokenspeed serve moonshotai/Kimi-K3 \
  --served-model-name kimi-k3 \
  --trust-remote-code \
  --max-model-len 8192 \
  --kv-cache-dtype fp8 \
  --tensor-parallel-size 8 \
  --mm-encoder-tp-mode data \
  --enable-expert-parallel \
  --attention-backend mla \
  --moe-backend auto \
  --max-num-seqs 32 \
  --disable-kvstore \
  --host 0.0.0.0 \
  --port 8000
```

On gfx950, the replicated 7168↔3584 latent projections automatically select
among a one-token Triton GEMV, tuned Gluon GEMMs, and the vendor GEMM according
to the current token count. At TP8/EP8, eligible one-token decode also combines
the routed MXFP4 experts with the shared-expert down projection, then applies
their joint reduction before the fused latent up-projection epilogue. Other
shapes and unsupported layouts retain the ordinary composed path. The fused
sigmoid-bias top-k route supports the full scheduled token count.

For Gluon Petit MegaMoE on one node with 8x gfx950 and MXFP4 expert weights,
use attention DP8 and EP8, with at most 1024 tokens per rank:

```bash
TORCH_NCCL_BLOCKING_WAIT=1 tokenspeed serve moonshotai/Kimi-K3 \
  --trust-remote-code --dtype bfloat16 \
  --tensor-parallel-size 1 --data-parallel-size 8 --ep-size 8 \
  --attention-backend mla --kv-cache-dtype fp8 \
  --moe-backend gluon_petit --all2all-backend gluon_petit \
  --max-model-len 8192 --max-num-seqs 32 \
  --chunked-prefill-size 1024 --max-prefill-tokens 1024 \
  --disable-kvstore --disable-autotune
```

## Kimi K3.1 Pro {#kimi-k3-1-pro}

Kimi-K3.1 (`KimiK31ForConditionalGeneration`) is Kimi-K3 with a DSA lightning
indexer on each of its 24 MLA layers: every MLA layer attends only the 2048
history tokens (`index_topk`) its own indexer selects. The KDA layers, the
MoE, the vision encoder, the parsers, and the cache family are Kimi-K3's, so
the [Kimi K3](#kimi-k3) notes apply unless a note below says otherwise.

Notes:

- Leave `--attention-backend` unset (or pass `dsa`): the MLA layers then run
  the `dsa` sparse-MLA backend inside the KDA hybrid wrapper. The dense MLA
  backends do not run the indexer.
- The indexer keys live in an index-K plane beside each MLA layer's latent
  plane in the full-attention cache group, so prefix caching, PD transfer, and
  frees treat them with the latent rows. The plane stores the keys as FP8 with
  one FP32 scale per 128 channels, 132 bytes per token per layer, whatever
  `--kv-cache-dtype` sets for the latent.
- The checkpoint ships no modeling code. The indexer follows DeepSeek-V3.2:
  queries from the normalized MLA q latent, LayerNorm keys without RoPE, and
  per-head ReLU scores weighted by `weights_proj`. Loading validates
  `dsa_config` against that and fails on anything else, on an indexer tensor
  without a parameter, and on an indexer parameter the checkpoint does not
  fill.
- Speculative verify takes up to 8 query rows per request on Blackwell and
  up to 6 on gfx950.
- The DSA layers run attention TP only; TokenSpeed refuses the Kimi-K3 QKV
  and output projection TP subgroups at startup.
- On NVIDIA, prefill CUDA graphs replay with the DSA attention at eager
  breaks on the PD prefill role (with `--disaggregation-layerwise-interval 0`)
  and on the aggregated role unless `--enable-mixed-batch` is on; elsewhere
  prefill runs eager. `--disable-prefill-graph` turns them off. On the
  aggregated role the default bucket ladder takes about 7.5 GB more per GPU,
  out of the KV cache, and adds about 7 minutes to startup;
  `--prefill-graph-capture-token-sizes` captures fewer buckets for less.
- The Kimi-K3.1 DSpark draft (`Inferact/Kimi-K3.1-Pro-DSpark-preview`)
  proposes 7 tokens per step, so it takes
  `--speculative-num-draft-tokens 8`. Unlike Kimi-K3's draft, it attends
  causally within its block (its config sets `dflash_config.causal`).
  `--drafter-attention-backend tokenspeed_mla` runs causal blocks; the other
  MLA drafter backends refuse the draft at startup.
- A checkpoint with a Python XTML encoder and no Jinja chat template needs
  its tokenizer's `apply_chat_template` to encode messages. Pass the token
  IDs as `input_ids` to the native `/generate` endpoint.

### NVIDIA

Install `flash-linear-attention` for the KDA layers, as for
[Kimi K3](#kimi-k3), then serve with the DSpark draft on 8 Blackwell GPUs.
This launch was validated on 2 GB300 nodes with 4 GPUs each. On such nodes,
start each one with the command below plus
`--nnodes 2 --node-rank <rank> --dist-init-addr <first-node>:<port>`, where
`<rank>` is the node's rank (`0` on the first node, `1` on the second) and
`<first-node>:<port>` is the first node's address and a free port on it. Drop
the 4 speculative flags to serve without the draft. The context length is the
checkpoint's 1,048,576 tokens.

```bash
tokenspeed serve moonshotai/Kimi-K3.1-Pro \
  --served-model-name kimi-k3.1 \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --tensor-parallel-size 8 \
  --mm-encoder-tp-mode data \
  --moe-backend flashinfer_trtllm \
  --max-num-seqs 64 \
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path Inferact/Kimi-K3.1-Pro-DSpark-preview \
  --speculative-num-draft-tokens 8 \
  --drafter-attention-backend tokenspeed_mla \
  --host 0.0.0.0 \
  --port 8000
```

### AMD

On 8x MI350X/MI355X (gfx950), serve at TP8 with Kimi-K3's EAGLE-3.1 draft
(`lightseekorg/kimi-k3-eagle3.1-mla`), which accepts slightly more tokens per
step than the EAGLE3 draft. The index-K planes make a K3.1 cache block about
22% larger than K3's, so at the default `--gpu-memory-utilization`, 15 requests
of 50K input and 500 output tokens run at once.

```bash
tokenspeed serve moonshotai/Kimi-K3.1-Pro \
  --served-model-name kimi-k3.1 \
  --trust-remote-code \
  --max-model-len 65536 \
  --kv-cache-dtype fp8 \
  --tensor-parallel-size 8 \
  --mm-encoder-tp-mode data \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path lightseekorg/kimi-k3-eagle3.1-mla \
  --drafter-attention-backend gluon \
  --max-num-seqs 64 \
  --disable-kvstore \
  --host 0.0.0.0 \
  --port 8000
```

- The rest comes from the defaults and the draft's config: a 3-step EAGLE3
  chain (a 4-token verify window), the draft's capture layers
  (2, 46, 90) and BF16 weights, TP8 without expert parallelism, automatic MoE
  selection, an 8192-token prefill budget, and CUDA graph capture sizes up to
  `--max-num-seqs`.
- `--drafter-attention-backend gluon` pins the draft's MLA layers to the Gluon
  kernels and fails fast where none is registered; leaving it unset lets the
  `mla` backend select the kernels.
- Prefill runs eager on gfx950: the DSA backend replays prefill CUDA graphs on
  NVIDIA only.
- The Kimi-K3 AMD note on greedy sampling applies.

## GLM5 / GLM5.2

GLM5 launches usually need remote code, long context, expert parallelism, FP8 KV
cache, and the TRTLLM MoE backend. GLM5.2 FP8 is available on Hugging Face as
`zai-org/GLM-5.2-FP8`. TokenSpeed defaults the reasoning parser to `glm45`.
Pass an explicit parser flag to override it. GLM5 DSA prefill planning reuses
the host-side request lengths and the packed page/row plan across full-indexer
layers. When integrating a custom attention backend, keep the
scheduler-provided CPU length mirrors populated to avoid device
synchronization in this path.

```bash
tokenspeed serve zai-org/GLM-5.2-FP8 \
  --served-model-name glm-5.2 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --moe-backend flashinfer_trtllm \
  --kv-cache-dtype fp8 \
  --max-model-len 262144 \
  --chunked-prefill-size 8192 \
  --max-num-seqs 128 \
  --host 0.0.0.0 \
  --port 8000
```

## GLM 5.3

GLM-5.3 follows the GLM-5.2 DSA serving path. Its base checkpoint includes the
NextN draft layer, so MTP does not require a separate draft checkpoint.

```bash
ts serve zai-org/GLM-5.3 \
  --served-model-name glm-5.3 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --moe-backend flashinfer_trtllm \
  --kv-cache-dtype fp8 \
  --max-model-len 262144 \
  --max-num-seqs 128 \
  --draft-model-path-use-base \
  --speculative-algorithm MTP \
  --speculative-num-steps 3
```

## GLM 5.3 Flash

GLM-5.3-Flash automatically configures its KDA/DSA backends and supports MTP from
the base checkpoint.

Install `ffmpeg` before serving multimodal requests:

```bash
apt-get update && apt-get install -y ffmpeg
```

On platforms without DeepGEMM, four-stream mHC uses a portable Triton path and
switches to its tiled prefill projection above 256 tokens.

### NVIDIA

```bash
ts serve zai-org/GLM-5.3-Flash \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --moe-backend flashinfer_trtllm \
  --kv-cache-dtype fp8 \
  --draft-model-path-use-base \
  --speculative-algorithm MTP \
  --speculative-num-steps 2 \
  --speculative-num-draft-tokens 3
```

### AMD

On MI350X/MI355X (gfx950), use tensor parallel size 4 with expert parallelism
disabled for the BF16 or block-FP8 checkpoints. The block-FP8 path retains
compact expert weights for its decode-specialized Gluon kernel and
materializes BF16 expert copies once at load time for prefill. Pass
`--sampling-backend triton`: without it AMD defaults to greedy sampling, which
ignores per-request `temperature`, `top_p`, and `top_k`.

```bash
ts serve \
  --model zai-org/GLM-5.3-Flash \
  --trust-remote-code \
  --attn-tp-size 4 \
  --ep-size 1 \
  --kv-cache-dtype fp8 \
  --max-model-len 65536 \
  --max-num-seqs 16 \
  --max-total-tokens 1048576 \
  --chunked-prefill-size 8192 \
  --max-prefill-tokens 8192 \
  --max-cudagraph-capture-size 16 \
  --cudagraph-capture-sizes 1 2 4 8 16 \
  --sampling-backend triton \
  --speculative-algorithm MTP \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4
```

`--max-total-tokens 1048576` holds sixteen concurrent 50K-token requests in the
KV cache on four GPUs; a 524288-token budget queues some of them.

## Qwen3 dense / Qwen3 30B-A3B

Qwen2, dense Qwen3, and Qwen3 MoE checkpoints use different architecture names.
For Qwen3 30B-A3B, the Hugging Face config advertises `qwen3_moe` and
`Qwen3MoeForCausalLM`, so launch it as a MoE model.

### Qwen3-0.6B on Ascend NPU

The Ascend path supports unquantized Qwen3-0.6B on one or more NPUs. It uses
the normal TokenSpeed scheduler and paged KV cache: prefill runs eagerly, while
the runtime captures fixed-shape decode batches as ACL Graphs. Aggregate
serving can schedule prefill and decode work in the same deployment.

The validated environment is CANN 9.0.0, PyTorch 2.9.0, `torch_npu`
2.9.0.post2, Transformers 5.12.0, Triton 3.2.0, and Triton-Ascend 3.2.1.
During validation, upgrading Transformers from 4.51.0 to 5.12.0 also changed
`huggingface-hub` from 0.36.2 to 1.28.0, `tokenizers` from 0.21.4 to 0.22.2,
and `hf-xet` from 1.5.1 to 1.6.0. It newly installed `typer==0.27.1`,
`shellingham==1.5.4`, and `annotated-doc==0.0.5`. The setup also newly installs
`apache-tvm-ffi==0.1.13` and editable `tokenspeed-kernel-npu==0.1.0`. These are
all Python-environment mutations made during this validation. Use matching
PyTorch and `torch_npu` builds. From the repository root, install the Ascend
dependencies, source CANN, and expose the three source packages:

```bash
test/ci_system/install_triton_ascend.sh
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
export PYTHONPATH="${PWD}/python:${PWD}/tokenspeed-kernel/python:${PWD}/tokenspeed-kernel-npu/python:${PYTHONPATH:-}"

python -m tokenspeed.cli serve Qwen/Qwen3-0.6B \
  --served-model-name qwen3-0.6b \
  --device npu \
  --dtype bfloat16 \
  --kv-cache-dtype auto \
  --attention-backend mha \
  --sampling-backend greedy \
  --disable-prefill-graph \
  --disable-pdl \
  --max-model-len 4096 \
  --max-num-seqs 4 \
  --max-total-tokens 16384 \
  --chunked-prefill-size 4096 \
  --max-cudagraph-capture-size 4 \
  --cudagraph-capture-sizes 1 2 4 \
  --disable-autotune \
  --host 0.0.0.0 \
  --port 31889
```

The setup script installs Transformers 5.12.0 and Triton-Ascend 3.2.1, installs
`tokenspeed-kernel-npu` from this checkout in editable mode, and compiles and
executes both a vector-add kernel and a TokenSpeed KV-cache kernel on the
visible NPU. Set `TOKENSPEED_CANN_ROOT` when CANN is installed outside the
default path.

The standard `mha` runtime backend selects the registered Ascend kernels on an
NPU. It requires eager prefill and disabled PDL, so pass
`--disable-prefill-graph` and `--disable-pdl` explicitly. Do not pass
`--enforce-eager`, because that also disables the decode ACL Graph. Although the
graph flags retain their CUDA-oriented names for CLI compatibility, they control
ACL Graph capture on an NPU. The command above captures decode batches 1, 2,
and 4 and was validated with a 16,384-token KV pool. Increase
`--max-model-len`, `--max-num-seqs`, `--max-total-tokens`, and the capture sizes
together when scaling the deployment. `--disable-autotune` shortens bring-up.
Remove it after validation when you want startup tuning.

The current Ascend sampling path is validated with greedy decoding.
Because `--sampling-backend greedy` always performs argmax, send
`temperature=0` and do not expect request-level `top_p` or `top_k` to take
effect.

Multi-card deployments use tensor parallelism. The world size must satisfy
the model's standard divisibility constraints. Context, pipeline, and data
parallel sizes must remain 1. Set `--world-size N` and expose `N` NPUs, for
example `ASCEND_RT_VISIBLE_DEVICES=0,1` with `--world-size 2`.

Verify the OpenAI-compatible endpoint with the served model name rather than
the checkpoint path:

```bash
curl http://127.0.0.1:31889/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-0.6b",
    "messages": [{"role": "user", "content": "你好，请用一句话介绍你自己。"}],
    "temperature": 0,
    "max_tokens": 64,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

```bash
tokenspeed serve Qwen/Qwen3-30B-A3B \
  --served-model-name qwen3-30b-a3b \
  --tensor-parallel-size 2 \
  --enable-expert-parallel \
  --moe-backend flashinfer_cutlass \
  --max-model-len 40960 \
  --reasoning-parser qwen3 \
  --host 0.0.0.0 \
  --port 8000
```

## Qwen3.8

Qwen3.8 shares the hybrid linear-attention (GDN) / full-attention layer
pattern with Qwen3.5.

### Qwen3.8-2.4T-A95B

Qwen3.8-max needs 16 GPUs, so it runs on two 8-GPU nodes. Launch
`tokenspeed serve` on every node with the same command, changing only
`--node-rank`. Every node points `--dist-init-addr` at node 0, which is the
only rank that serves the HTTP API. See [Parallelism](../serving/parallelism.md)
for the multi-node rules.

This family has no parser auto-selection, so set `--reasoning-parser` and
`--tool-call-parser` explicitly. `--speculative-algorithm MTP` without
`--speculative-draft-model-path` drafts from the base checkpoint. Set
`--dist-init-addr` to node 0's own address and port throughout
(`<node0-host>:25000` below).

#### TP16

One replica across both nodes. `--ep-size` defaults to 1, so the experts stay
tensor-parallel over the full world and all-to-all stays out of the path. Keep
`--moe-backend auto`: the block-scale FP8 `deep_gemm` experts implement only the
DeepEP legs and are unavailable without `--all2all-backend deepep`.

```bash
# node 0 (serves the HTTP API)
tokenspeed serve Qwen/Qwen3.8-2.4T-A95B \
  --served-model-name Qwen/Qwen3.8-2.4T-A95B \
  --nnodes 2 --node-rank 0 --nprocs-per-node 8 --world-size 16 \
  --dist-init-addr <node0-host>:25000 \
  --attn-tp-size 16 \
  --moe-backend auto \
  --quantization fp8 --kv-cache-dtype fp8 \
  --attention-backend trtllm \
  --chunked-prefill-size 8192 \
  --max-num-seqs 128 \
  --speculative-algorithm MTP --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --reasoning-parser qwen3_thinking --tool-call-parser qwen_coder \
  --host 0.0.0.0 --port 8000

# node 1 (same command, --node-rank 1)
tokenspeed serve Qwen/Qwen3.8-2.4T-A95B \
  --served-model-name Qwen/Qwen3.8-2.4T-A95B \
  --nnodes 2 --node-rank 1 --nprocs-per-node 8 --world-size 16 \
  --dist-init-addr <node0-host>:25000 \
  --attn-tp-size 16 \
  --moe-backend auto \
  --quantization fp8 --kv-cache-dtype fp8 \
  --attention-backend trtllm \
  --chunked-prefill-size 8192 \
  --max-num-seqs 128 \
  --speculative-algorithm MTP --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --reasoning-parser qwen3_thinking --tool-call-parser qwen_coder \
  --host 0.0.0.0 --port 8000
```

#### TP8 DP2 EP16 (DeepEP)

Two TP8 attention replicas, experts sharded across all 16 ranks, and expert
routing on DeepEP dispatch/combine instead of all-gather:

```bash
# node 0 (serves the HTTP API)
tokenspeed serve Qwen/Qwen3.8-2.4T-A95B \
  --served-model-name Qwen/Qwen3.8-2.4T-A95B \
  --nnodes 2 --node-rank 0 --nprocs-per-node 8 --world-size 16 \
  --dist-init-addr <node0-host>:25000 \
  --attn-tp-size 8 --data-parallel-size 2 --ep-size 16 \
  --moe-backend deep_gemm \
  --all2all-backend deepep --deepep-mode auto \
  --low-latency-max-num-tokens-per-gpu 64 \
  --quantization fp8 --kv-cache-dtype fp8 \
  --attention-backend trtllm \
  --chunked-prefill-size 8192 \
  --max-num-seqs 128 \
  --speculative-algorithm MTP --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --reasoning-parser qwen3_thinking --tool-call-parser qwen_coder \
  --host 0.0.0.0 --port 8000

# node 1 (same command, --node-rank 1)
tokenspeed serve Qwen/Qwen3.8-2.4T-A95B \
  --served-model-name Qwen/Qwen3.8-2.4T-A95B \
  --nnodes 2 --node-rank 1 --nprocs-per-node 8 --world-size 16 \
  --dist-init-addr <node0-host>:25000 \
  --attn-tp-size 8 --data-parallel-size 2 --ep-size 16 \
  --moe-backend deep_gemm \
  --all2all-backend deepep --deepep-mode auto \
  --low-latency-max-num-tokens-per-gpu 64 \
  --quantization fp8 --kv-cache-dtype fp8 \
  --attention-backend trtllm \
  --chunked-prefill-size 8192 \
  --max-num-seqs 128 \
  --speculative-algorithm MTP --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --reasoning-parser qwen3_thinking --tool-call-parser qwen_coder \
  --host 0.0.0.0 --port 8000
```

Notes:

- `--low-latency-max-num-tokens-per-gpu` sizes DeepEP's NVSHMEM heap (roughly
  2.0 GB at 64, 8.1 GB at 256), and startup claims that heap after profiling
  the KV pool. An oversized value therefore fails late, when the first
  dispatch runs out of fabric memory. Size it to the real per-rank decode
  token bound and no lower: the runtime rejects a batch above the capacity
  rather than truncating it.
- Internode DeepEP rides NVSHMEM IBGDA. On a RoCE fabric, mirror the NCCL
  values into `NVSHMEM_IB_GID_INDEX`, `NVSHMEM_IB_TRAFFIC_CLASS`, and
  `NVSHMEM_IB_SL`, and point `NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME` at the same
  interface as `NCCL_SOCKET_IFNAME`.

#### Choosing a layout

- TP16 has the lower TTFT and TPOT at batch 1-2: no dispatch/combine hop, and
  the single replica owns the whole batch.
- The DeepEP layout pulls ahead from mid batch up, where its expert kernels and
  the second attention replica both pay off.

### Qwen3.8-27B

A dense 27B-class Qwen3.8 FP8 checkpoint on a single GPU, with self-speculative
MTP (the draft model path points at the same checkpoint):

```bash
tokenspeed serve Qwen/Qwen3.8-27B-FP8 \
  --served-model-name Qwen/Qwen3.8-27B-FP8 \
  --world-size 1 \
  --attention-backend trtllm \
  --moe-backend flashinfer_trtllm \
  --chunked-prefill-size 8192 \
  --max-model-len 262144 \
  --max-num-seqs 128 \
  --kv-cache-dtype fp8_e4m3 \
  --speculative-algorithm MTP \
  --speculative-draft-model-path Qwen/Qwen3.8-27B-FP8 \
  --speculative-num-steps 3
```

## Qwen3.8 Flash Next

Qwen3.8-Flash-Next is a multimodal MoE model and an early preview of the
Qwen4 architecture, playing for Qwen4 the role Qwen3-Next played for Qwen3.5.
It pairs a 125B-parameter main model with 51B of N-gram embeddings (6B
activated per token), supports 262,144 tokens of context natively and 1M with
YaRN, and upgrades four axes of the hybrid design:

- GDN + QSA hybrid attention
- 4-branch Gated Residual
- N-gram (predictive latent) embeddings
- Muon optimization

In TokenSpeed it runs as a hybrid linear-attention (GDN) / full-attention model
with optional predictive latent embeddings (PLE), optional QSA sparse
attention, and a one-layer MTP draft. Dense and MoE checkpoints share the same
launch command.

### NVIDIA

```bash
ts serve \
    --model Qwen/Qwen3.8-Flash-Next-FP8 \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --quantization fp8 \
    --moe-backend flashinfer_trtllm \
    --speculative-algorithm MTP \
    --speculative-num-steps 3
```

### AMD

On AMD MI350X/MI355X (gfx950), serve the
[AMD Quark MXFP4](https://huggingface.co/amd/Qwen3.8-Flash-Next-Quark-MXFP4)
checkpoint, which quantizes the routed and shared experts to MXFP4 and keeps
the rest, including the MTP layer, in BF16:

```bash
ts serve \
    --model amd/Qwen3.8-Flash-Next-Quark-MXFP4 \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --quantization mxfp4 \
    --max-num-seqs 16 \
    --max-cudagraph-capture-size 16 \
    --kvstore-size 32 \
    --speculative-algorithm MTP \
    --speculative-num-steps 3
```

The preceding configuration caps the batch at 16 sequences for latency; raise
`--max-num-seqs` and `--max-cudagraph-capture-size` together for throughput.
Keep ReplaySSM, the default.

`--kvstore-size 32` caps the Host KV cache at 32 GB per GPU. The default
(twice the Device KV, about 96 GB per GPU here) keeps much more Host memory
mapped for the GPU, and while the Linux kernel compacts memory and migrates
those pages, the GPU stalls; on hosts with fragmented memory this showed as
10-30 s pauses during decode. Raise the size if long prefixes are reused
after the Device cache evicts them, or drop the Host cache with
`--disable-kvstore` if they are not. On the host side, disabling proactive
compaction (`sysctl vm.compaction_proactiveness=0`) removes the background
trigger.

### Optional `--hf-overrides`

All keys are optional and can be combined in a single `--hf-overrides` JSON
object:

```bash
--hf-overrides \
  '{"ple_embed_dtype":"float8_e4m3fn","index_share_for_mtp_iteration":true}'
```

- `ple_embed_dtype: "float8_e4m3fn"`: store the PLE n-gram embedding table in
  FP8 to save memory, quantizing a BF16 table per row at load. When omitted,
  the table keeps the checkpoint's format: checkpoints that ship it in FP8,
  such as `Qwen/Qwen3.8-Flash-Next-FP8` and `nvidia/Qwen3.8-Flash-Next-NVFP4`,
  keep it in FP8, and other checkpoints store it in the model's compute dtype.
- `ple_offload_embedding`: keep the PLE table in pinned host memory when `true`
  or GPU memory when `false`. When omitted, TokenSpeed enables offloading on
  NVIDIA CUDA and disables it on other platforms.
- `index_share_for_mtp_iteration: true`: reuse the QSA top-k selection across
  MTP steps. Checkpoints that already set
  `text_config.index_share_for_mtp_iteration=true` do not need this flag.

## GPT-OSS 20B / 120B

Small GPT-OSS launches can start simple. Large GPT-OSS launches usually tune
tensor parallelism, scheduler token budget, and KV cache dtype.

```bash
tokenspeed serve openai/gpt-oss-20b \
  --served-model-name gpt-oss-20b \
  --tensor-parallel-size 1 \
  --max-model-len 131072 \
  --chunked-prefill-size 8192 \
  --reasoning-parser base \
  --host 0.0.0.0 \
  --port 8000
```

```bash
tokenspeed serve openai/gpt-oss-120b \
  --served-model-name gpt-oss-120b \
  --tensor-parallel-size 4 \
  --max-model-len 131072 \
  --kv-cache-dtype fp8 \
  --chunked-prefill-size 8192 \
  --max-num-seqs 256 \
  --reasoning-parser base \
  --host 0.0.0.0 \
  --port 8000
```

### Gluon Petit MegaMoE on AMD CDNA4

GPT-OSS 120B with MXFP4 weights supports Gluon Petit MegaMoE on one node
with eight gfx950 GPUs:

```bash
TORCH_NCCL_BLOCKING_WAIT=1 tokenspeed serve openai/gpt-oss-120b \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --expert-parallel-size 8 \
  --dtype bfloat16 \
  --moe-backend gluon_petit \
  --all2all-backend gluon_petit \
  --chunked-prefill-size 1024 \
  --max-prefill-tokens 1024 \
  --disable-autotune \
  --disable-kvstore
```

This backend supports at most 1024 prefill or decode tokens per rank.

## DeepSeek V4.1-Flash

The B200 and AMD GSM8K CI jobs, the GB300 Slurm 1P1D GSM8K job, and the
manual 1P1D smoke test use `deepseek-ai/DeepSeek-V4.1-Flash-1017`.
The 1P1D CI tasks set `MODEL` explicitly so both serving roles and the
evaluation client select the same checkpoint.

The target LM head masks BOS, PAD, and the image control/placeholder token IDs
`0, 2, 129262..129264` to `-inf`. The mask applies before ordinary sampling,
fused argmax, and speculative verification, and to target-side logprob scoring;
draft-model proposals are unchanged. Pass
`--disable-deepseek-v41-token-blacklist` to allow these tokens.

DeepSeek V4.1 (`deepseek_v41`) runs on its own FlatKV attention backend
with a four-group KV cache: the global KV chains, the SWA rows, and the
compressor tails. The recipe declares the last two **replayable**: they
never enter the prefix cache, and a prefix
hit re-feeds the cached prefix's last 128 tokens so the model regenerates
them into the request's own pages (SWA bounded replay,
[`docs/design/scheduler.md` §1.3](../design/scheduler.md#13-bounded-replay)).
The backend masks the global KV and index rows those replayed tokens
recompute, so the shared rows stay exactly what the first computation
produced. The CED decoder (layers 20–39) runs only on each prompt's
last 128 positions (one row per chunk that does not complete its
prompt), so the prefill row
count changes at layer 20. The prefill CUDA graph therefore captures the
model in two halves around that layer, which runs eager: encoder graphs per
token bucket and decoder graphs per decoder-row bucket, the latter shared by
every token bucket and capped at 128 rows per request
(`--max-num-seqs` × 128, see
[`docs/design/unified_path.md`](../design/unified_path.md#prefill-graphs-around-a-row-narrowing)).
`--prefill-graph-capture-sizes` sets both ladders. `--disable-prefill-graph`
turns both off. Decode CUDA graphs are unaffected.

```bash
tokenspeed serve deepseek-ai/DeepSeek-V4.1-Flash \
  --served-model-name deepseek-v41-flash \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --max-total-tokens 262144 \
  --max-num-seqs 32 \
  --chunked-prefill-size 8192 \
  --max-cudagraph-capture-size 32 \
  --disable-kvstore \
  --host 0.0.0.0 \
  --port 8000
```

On Hopper the routed MXFP4 experts run on FlashInfer's CUTLASS mixed-input
grouped GEMM (`flashinfer_cutlass_mxfp4_w4a16_moe_apply`), which `auto`
selects over Marlin. Marlin dequantizes in registers and scales linearly with
the token count. The CUTLASS kernel stays weight-bandwidth bound. Measured per
MoE layer on one EP8 rank of an H20 (µs, CUDA-graph replay): 502 vs 656 at 96
tokens, 621 vs 1552 at 192, 671 vs 2855 at 576, 4517 vs 7999 at an 8192-token
prefill chunk. Two consequences:

- Startup runs FlashInfer's tactic autotuner inside the kernel tuning window
  (about five minutes for this kernel on H20), including each distinct draft
  expert geometry. Decode-capable roles also traverse the shared speculative
  path once to discover the draft model's other operators. Prefill-only roles
  skip that traversal because they do not allocate decode/verify scratch.
  `--disable-autotune` loads a matching persistent cache and uses heuristic
  tactics for uncovered shapes, which is fine for bring-up. Pipeline-parallel
  launches skip tuning but can reuse a cache from a full-model run with the
  same engine role, tensor/expert parallel layout and environment. Independent
  prefill and decode roles keep separate caches. The tuner creates cache
  directories on the first successful save.
- `--moe-mxfp4-fp8-activation` switches to the W4A8 variant (FP8 activations,
  Humming residual scales): 282/338/380/2195 µs at the same token counts,
  another 1.8x, at a few percent of relative error on the expert outputs
  (FP8 activation rounding). Treat it as opt-in and confirm the served model
  on GSM8K or a similar check before relying on it.

`--moe-backend marlin` keeps the previous kernel and stays required for
DeepEP all-to-all layouts and for Kimi-K3's SiTU experts, which the CUTLASS
epilogue does not implement.

Add `--speculative-algorithm DSPARK` for same-checkpoint DSpark decoding.
The draft seeds its context windows from the decoder's kept rows, and each
draft stage attends its 128-row window plus the non-causal proposal block
through the same `selected_attention` workspace kernel as the target's
prefill (FlashMLA on sm90+, Triton elsewhere). The fp32 arithmetic remains
the CPU reference. A hit
re-feeds the groups' whole retention window, which is exactly the 128-token
attention window (the compressor-tail group retains its unfinished pair the
same way). The scheduler requires
`--chunked-prefill-size` of at least that window plus one prefix page and
never leaves a prompt's final chunk shorter than it. The replayed rows attend
SWA keys from the replay start only, the truncation the model is trained
for. The runtime never recomputes the cached global KV from them.
`usage.prompt_tokens_details.cached_tokens` reports the hit through the end
of the replayed window, so it stays a multiple of the prefix granularity.
Under prefill/decode disaggregation the prefill node replays on its own
prefix hits exactly as above and ships each group's retained tail. The
decode node lands the tail and never re-feeds. Even on one machine, let
Mooncake pick an RDMA transport rather than forcing the intra-node NVLink one.

Same-checkpoint DSpark computes its public FP32 base logits straight from the
target's BF16 LM-head shard: BF16 products are exact in FP32, so a BF16 GEMM
with an FP32 accumulator reproduces the reference's FP32 head math up to
summation order without an FP32 copy of the shard. The draft path reads the
head in place under CUDA Graph replay, so in-place target weight updates need
no refresh.

The greedy block sampler runs one Triton kernel per block step on each
tensor-parallel rank: it gathers the previous token's BF16 Markov bigram row
from a replicated table, adds the bigram bias (a BF16 tensor-core dot against
the rank's Markov projection shard, sharded like the LM head) to the
rank's base logits, and packs the best `(logit, token)` of every vocabulary
tile into one int64 candidate. One all-gather per step shares the candidates.
The next step's kernel (or the final resolve) reduces them to the
`torch.argmax` winner over the whole vocabulary, ties broken toward the lowest
token ID. A five-token block is therefore six kernels and five all-gathers
instead of a per-step chain of masked embedding lookups, GEMMs, reductions, and
two all-gathers.

Before each forward, one pinned upload carries every request's Engram history
snapshot. `prepare_ngram_inputs` resolves the previous-three tokens and validity
mask, clears padding, and computes table rows for all Engram layers in one
launch. The model consumes these prepared hashes through stable input buffers.
Preparation leaves accepted request state unchanged; after sampling,
`commit_ngram_inputs` advances accepted history and cache lengths together in
a second launch. Hashes are bit-identical to the eager reference, which remains
the CPU path. Per-batch token and request counts stay runtime kernel arguments;
only bounded tile sizes and fixed model/pool geometry specialize compilation.
Both kernels are checked against the default per-thread stack budget, so first
use does not reserve extra spill memory on every SM. Retraction snapshots carry
the accepted frontier; the first restored decode reseeds Engram history from
its token snapshot, and commits it only after verification.

The CUDA draft path also preserves the checkpoint's UE8M0-scaled FP8 activation
round-trip with a fused `tokenspeed-kernel` operation. It computes the same
per-group power-of-two scale and returns dequantized values in the input dtype.
The fusion removes intermediate reduction and elementwise launches but does not
change the model's quantization contract.

DSpark attention RMSNorm uses the platform kernel on CUDA while retaining its
explicit FP32-accumulating PyTorch expression as the CPU reference. The fused
path preserves the existing output dtype and is safe to capture and replay in
the target CUDA Graph.

### GB300 Slurm 1P1D CI

[`deepseek-v4.1-flash-pd-1p1d-dspark-evalscope-gsm8k-gb300-slurm.yaml`](../../test/ci/eval/deepseek-v4.1-flash-pd-1p1d-dspark-evalscope-gsm8k-gb300-slurm.yaml)
runs one TP4 prefill engine and one TP4 decode engine on two four-GPU nodes.
Node 0 also hosts the SMG gateway and the Slurm evaluation client, which
connects to `127.0.0.1:8000`. Both engines use same-checkpoint DSpark,
`mega_moe` with expert parallelism on Blackwell, host-resident Engram tables,
and Mooncake transfer after the completed prompt (`layerwise-interval=0`).
The YAML disables decode prefix caching. The gateway uses `deepseek_v31`
reasoning parsing. The 262144-token cache budget limits admission
independently of the 32768-token per-request context and 16-sequence cap.

The gate checks GSM8K accuracy of at least 0.90 on 100 samples with EvalScope
1.11.1, greedy decoding, concurrency 8, and up to 30000 generated tokens.
It participates in the GB300 Slurm per-commit workflow. To run only this case,
select its YAML in **Slurm Dispatch**, choose cluster `gb300`, and optionally
provide a pull request number.

The launcher uses `PD_SLURM=1` to assign one role per node and clears Slurm
topology discovery only inside each worker process. Node 0 starts the gateway
with the prefill worker on a same-host socket and the decode worker on
`tcp://<node 0>:DECODE_HANDSHAKE_PORT` (18348), then publishes its address in
the launcher's job-and-step-scoped artifact directory, which both nodes share;
node 1's decode engine dials it there. The gateway is ready once both engines
registered. Worker logs remain separate as `prefill.log`, `decode.log`, and
`lb.log`. When you need an explicit RDMA device selection, set
`DISAGGREGATION_IB_DEVICE`. The GB300 task selects
`mlx5_0,mlx5_1,mlx5_2,mlx5_3` to keep transfers on the InfiniBand fabric.
Automatic discovery also includes Ethernet RNICs, which can
cause incompatible RoCE/InfiniBand endpoint pairings during the RDMA handshake.
Without `PD_SLURM=1`, the same launcher retains the single-node smoke topology.

## Nemotron-3 Super

`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` is a Nemotron-H hybrid. Its
88 blocks are 40 Mamba2 layers, 40 latent MoE layers, and 8 grouped-query
attention layers without positional encoding. Only the Mamba2 and attention
blocks own cache state, so the cache plans 48 layers. The routed experts use
squared ReLU with NVFP4 weights on the FlashInfer TRT-LLM MoE, routed inside
the kernel. Some dense projections are FP8 and the rest are BF16, as the
checkpoint's per-layer ModelOpt config says.

The checkpoint's KV cache scheme is FP8 with unit scales, which TokenSpeed does
not detect automatically, so pass `--kv-cache-dtype fp8`. The Mamba2 SSM state
is FP32.

```bash
tokenspeed serve nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --max-model-len 16384 \
  --world-size 1 \
  --host 0.0.0.0 \
  --port 8000
```

The checkpoint's multi-token prediction head serves as the draft model for
speculative decoding. ReplaySSM is on by default: verification leaves the
committed Mamba2 state untouched and one kernel then rebuilds every layer's
accepted state, instead of storing a state per draft token
(`--disable-replay-ssm` stores them).

```bash
tokenspeed serve nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --max-model-len 16384 \
  --world-size 1 \
  --speculative-algorithm MTP \
  --speculative-num-steps 3 \
  --speculative-num-draft-tokens 4
```

## Beam

Beam (`model_type=beam`, `BeamForCausalLM`) is a mixture-of-experts model
with 52 decoder layers: 9 full-attention layers and 43 sliding-window layers
with a 512-token window. Its BF16, FP8, and NVFP4 checkpoints run with the same
NVIDIA command below: the checkpoint's own quantization config selects the
path, with no flag. Use these supported Hub revisions:

| Checkpoint | Revision |
| --- | --- |
| BF16 | `f40a3a29` |
| FP8 | `21052137` |
| NVFP4 | `81a5ab6b` |

For FP8 decode, eligible MNNVL sandwich all-reduce boundaries also write
per-token E4M3 activations and FP32 scales for the following Q/K/V/gate or
dense gate/up projection. This reuses the BF16-rounded norm output and its
existing quantizer contract. The consumer accepts at most 64 rows; other
shapes and checkpoint formats retain their existing path. No extra flag is
required. If the fused collective declines, the second norm fills the same
buffers before the projection consumes them.

The command below runs one TP8 replica over two 4-GPU nodes, as in the
recipe's GB200 measurements. Run it on both nodes, changing only
`--node-rank`, and point `--dist-init-addr` at node 0, the only rank that
serves the HTTP API. Pass the local checkpoint directory as the model, the
same path on every node. Speculative decoding uses Beam's DFlash2 drafter
checkpoint, which serves all three checkpoints.

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

Without `--served-model-name`, the API model ID is the full checkpoint path.

Notes:

- Serving requires `--dtype bfloat16`. Beam's flag-free defaults, the `trtllm`
  attention backend and the `flashinfer_trtllm` MoE backend, apply on NVIDIA
  SM100 to SM103 and were validated on SM100 (GB200) only. Beam serves without
  TF32 on every platform. Beam's own Triton kernels (routing, SSMax, the
  residual norm and the soft cap) and the BF16x3 router GEMM run on NVIDIA SM90
  to SM103.
- On NVIDIA SM90 the BF16 checkpoint's experts run on the Triton BF16 MoE and
  the FP8 checkpoint's experts on the Triton FP8 per-channel MoE; the FP8
  checkpoint's dense linears are dequantized to BF16 at load. NVFP4 experts
  need SM100 to SM103.
- `--grammar-backend xgrammar` lets forced tool choice and JSON
  `response_format` work, and `--enable-mixed-batch
  --prefill-graph-max-tokens 8192` runs a prompt chunk in the same step as the
  running decodes. Pass the same values on every node.
- The `reflection/Beam-1.0-501B-A23B-DFlash2` drafter at revision `8d26c952`
  serves all three checkpoints. It declares 1,048,576 positions, so leave
  `TOKENSPEED_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN` unset. With the drafter the
  default batch limit is 80 requests, not 160.
- Name no `--reasoning-parser` or `--tool-call-parser`. The bundled
  `tokenspeed-smg` version pinned in
  [python/pyproject.toml](https://github.com/lightseekorg/tokenspeed/blob/main/python/pyproject.toml)
  reads the output through the checkpoint's `response_template`.

The first start on a fresh machine tunes kernels for 30 to 45 minutes; pass
`--engine-startup-timeout 5400`. Later starts reuse the caches.

API notes:

- Reasoning effort is `low`, `medium` (the default), `high`, `xhigh`, or `max`.
- Keep `--grammar-backend xgrammar`. With it, the bundled gateway constrains
  every Beam turn to Beam's output format, including tool calls under
  `tool_choice: auto`, forced tool choice (`required` or a named function) and
  JSON `response_format` with `json_schema`.

For batches above 160, or above 80 with the drafter, set all three scheduling
and graph flags. For 512 requests, use `--max-num-seqs 512`,
`--max-cudagraph-capture-size 512`, and
`--cudagraph-capture-sizes 1 8 32 64 128 160 256 384 512`. Setting only the
first one or two leaves larger decode batches outside the captured graph list.

### AMD gfx950

Beam supports BF16 and compressed-tensors dynamic W8A8 FP8 checkpoints on
gfx950. The following configuration runs an eight-GPU FP8 target with a BF16
DFlash2 drafter, mixed prefill/decode batching, and graph replay:

```bash
ts serve --model <beam-fp8-checkpoint-dir> \
  --tensor-parallel-size 8 --ep-size 1 \
  --dtype bfloat16 --kv-cache-dtype bfloat16 \
  --attention-backend mha --moe-backend gluon --all2all-backend none \
  --sampling-backend triton_full --grammar-backend xgrammar \
  --disable-tf32 --layer-boundary-norm unfused \
  --max-model-len 262144 --max-num-seqs 16 --chunked-prefill-size 8192 \
  --prefill-graph-max-tokens 8192 \
  --max-cudagraph-capture-size 16 --cudagraph-capture-sizes 1 2 4 8 16 \
  --gpu-memory-utilization 0.85 --disable-kvstore --disable-prefix-caching \
  --enable-mixed-batch --speculative-algorithm DFLASH \
  --speculative-draft-model-path <beam-dflash2-checkpoint-dir> \
  --speculative-draft-model-quantization unquant \
  --drafter-attention-backend mha \
  --speculative-num-steps 15 --speculative-num-draft-tokens 16 \
  --host 127.0.0.1 --port 8000
```

For a BF16 target, use its checkpoint and `--moe-backend triton` with the
same attention, dtype and unfused-boundary settings. Omit the speculative
arguments for target-only decoding. The context and memory fraction above
are starting budgets, not a capacity guarantee; check the startup allocation
for the chosen checkpoint and request lengths.

This recipe covers BF16 and FP8 checkpoints, not other quantization formats.

Use the gateway dependencies pinned by this checkout and the original
checkpoint chat template. Check raw generation, parsed chat/tool APIs,
numerical parity and performance independently before deployment.

## Tuning order

1. Set model ID, trust policy, tokenizer mode, and served model name.
2. Set context length and KV cache dtype.
3. Set tensor, data, and expert parallelism to match the node topology.
4. Set scheduler budgets: `--chunked-prefill-size`, `--max-num-seqs`, and only then `--max-total-tokens`.
5. Set attention, MoE, and sampling backends explicitly for benchmark runs.
6. Add reasoning, tool-call, grammar, or speculative decoding only when the model and workload need them.
