# Model Recipes

These recipes start from a known model family, pick the hardware topology, then
set only the parameters that change runtime behavior.

The commands below are templates. Validate exact model IDs, checkpoint formats,
and backend choices against the build you deploy.

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
    --gpu-memory-utilization 0.95 \
    --disable-cuda-graph-padding \
    --trust-remote-code \
    --attention-backend fa4 \
    --moe-backend flashinfer_trtllm \
    --enable-prefix-caching \
    --disable-kvstore \
    --block-size 128 \
    --enable-cache-report \
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
    --gpu-memory-utilization 0.95 \
    --disable-cuda-graph-padding \
    --trust-remote-code \
    --enable-prefix-caching \
    --disable-kvstore \
    --block-size 128 \
    --enable-cache-report \
    --speculative-algorithm MTP \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4
```

## MiniMax M3

MiniMax M3 uses 128-token MSA blocks. TokenSpeed configures its dense and sparse
attention layers automatically; select the dense backend with
`--attention-backend` and run with `--disable-kvstore`.

```bash
tokenspeed serve nvidia/MiniMax-M3-NVFP4 \
    --tensor-parallel-size 4 \
    --max-model-len 81920 \
    --max-num-seqs 16 \
    --max-prefill-tokens 8192 \
    --chunked-prefill-size 8192 \
    --gpu-memory-utilization 0.95 \
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
    --block-size 128 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000
```

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

Known limitation: native TokenSpeed DFlash currently uses full-history draft
attention. It does not yet expose an equivalent of SGLang's
`--speculative-dflash-draft-window-size`; add such a flag before relying on
bounded draft attention for long-context deployments.

## Kimi K3

Kimi-K3 combines a MoonViT vision encoder with a hybrid KDA
(linear-attention) / NoPE-MLA (full-attention) decoder and a
DeepSeek-V3-style latent MoE. The KDA layers currently use
flash-linear-attention kernels on NVIDIA, so install it first:

```bash
pip install flash-linear-attention
```

Notes:

- K3 uses the grouped paged-cache scheduler and KDA state groups.
- KDA dispatch is vendor-neutral at the runtime boundary. The kernel registry
  selects the existing FLA-derived NVIDIA implementation or the native AMD
  implementation, including each backend's preferred recurrent-state layout.
  The runtime does not transpose or reinterpret that state.
- NVIDIA auto-selects `--attention-backend tokenspeed_mla` for K3
  (fp8 KV required). AMD uses the `mla` backend.
- `tokenspeed serve` auto-selects the `kimi_k3` reasoning and tool-call
  parsers. Explicit parser flags override these defaults.
- The SMG packages pinned by TokenSpeed resolve `moonshotai/Kimi-K3` directly;
  a flattened local checkpoint and separately staged remote-code cache are no
  longer required.
- The checkpoint carries no FP8 KV scaling factors. When the target K3 uses its
  required FP8 LCM cache, TokenSpeed keeps the separate K3 DSpark draft cache in
  BF16 so context injection and draft attention match the reference precision.
- DSpark proposal blocks use non-causal MLA draft attention. Both the `mla` and
  `trtllm_mla` draft backends preserve every block row during eager execution
  and CUDA graph capture. When K3's 128-token logical cache pages feed the
  64-token TRT-LLM MLA kernel, the backend expands each logical page into its
  two physical kernel pages before draft attention.
- For Kimi K3, an eight-token verify window uses seven DSpark draft queries.
  The anchor query directly predicts the first draft through the Markov head;
  it must not be padded with an eighth, unused mask row.
- Target features are captured from K3's completed-layer prefix stream before
  the model-level AttnRes mix and final norm, matching the DSpark checkpoint's
  vLLM training and inference contract.
- Under tensor parallelism, the draft's final row-parallel MLP output is reduced
  across TP ranks before `final_norm` and shared target-head sampling.
- The vision encoder has 12 attention heads. For an 8-way text TP deployment,
  use `--mm-encoder-tp-mode data` so each rank runs the vision encoder at TP1
  on a different whole image.
- The pinned SMG frontend registers Kimi-K3's chat renderer and multimodal
  processor. Preserve the checkpoint's
  `media_proc_cfg.in_patch_limit=65536`; silently falling back to K2.5's
  16384-patch default reduces OCR resolution.
- KDA recurrent-state pages register for prefix-cache reuse only when a
  prefill chunk ends exactly on a logical cache-page boundary. The engine floors
  `--chunked-prefill-size` to the plan's page grain automatically (logged as
  a warning when it adjusts); the page grain is budget-dependent (e.g. 1472
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
  --gpu-memory-utilization 0.94 \
  --max-num-seqs 32 \
  --disable-kvstore \
  --host 0.0.0.0 \
  --port 8000
```

Plain TP8 (drop `--ep-size 8`) works too. The fused MoE path needs a
Blackwell GPU (B200/B300); on other NVIDIA platforms use
`--moe-backend triton`.

### AMD

The standard AMD path on 8x gfx950 uses the `mla` backend. For TP8/EP8,
automatic MoE selection uses the specialized Gluon SiTU kernels:

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
  --gpu-memory-utilization 0.92 \
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

## GLM5 / GLM5.2

GLM5 launches usually need remote code, long context, expert parallelism, FP8 KV
cache, and the TRTLLM MoE backend. GLM5.2 FP8 is available on Hugging Face as
`zai-org/GLM-5.2-FP8`. TokenSpeed defaults the reasoning parser to `glm45`;
pass an explicit parser flag to override it.

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

## Qwen3 Dense / Qwen3 30B-A3B

Qwen2, dense Qwen3, and Qwen3 MoE checkpoints use different architecture names.
For Qwen3 30B-A3B, the Hugging Face config advertises `qwen3_moe` and
`Qwen3MoeForCausalLM`, so launch it as a MoE model.

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
`--node-rank`; every node points `--dist-init-addr` at node 0, which is the only
rank that serves the HTTP API. See [Parallelism](../serving/parallelism.md) for
the multi-node rules.

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
  --gpu-memory-utilization 0.95 --max-num-seqs 128 \
  --disable-kvstore \
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
  --gpu-memory-utilization 0.95 --max-num-seqs 128 \
  --disable-kvstore \
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
  --gpu-memory-utilization 0.95 --max-num-seqs 128 \
  --disable-kvstore \
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
  --gpu-memory-utilization 0.95 --max-num-seqs 128 \
  --disable-kvstore \
  --speculative-algorithm MTP --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --reasoning-parser qwen3_thinking --tool-call-parser qwen_coder \
  --host 0.0.0.0 --port 8000
```

Notes:
- `--low-latency-max-num-tokens-per-gpu` sizes DeepEP's NVSHMEM heap (roughly
  2.0 GB at 64, 8.1 GB at 256), and that heap is claimed after the KV pool is
  profiled. An oversized value therefore fails late, when the first dispatch
  runs out of fabric memory. Size it to the real per-rank decode token bound
  and no lower: a batch above the capacity is rejected, not truncated.
- Internode DeepEP rides NVSHMEM IBGDA. On a RoCE fabric, mirror the NCCL
  values into `NVSHMEM_IB_GID_INDEX`, `NVSHMEM_IB_TRAFFIC_CLASS`, and
  `NVSHMEM_IB_SL`, and point `NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME` at the same
  interface as `NCCL_SOCKET_IFNAME`.

#### Choosing a layout

- TP16 has the lower TTFT and TPOT at batch 1-2: no dispatch/combine hop, and
  the single replica owns the whole batch.
- The DeepEP layout pulls ahead from mid batch up, where its expert kernels and
  the second attention replica both pay off.

### Qwen3.8-27B(Blackwell)

A dense 27B-class Qwen3.8 FP8 checkpoint on a single GPU, with self-speculative
MTP (the draft model path points at the same checkpoint):

```bash
tokenspeed serve Qwen/Qwen3.8-27B-FP8 \
  --served-model-name Qwen/Qwen3.8-27B-FP8 \
  --world-size 1 \
  --gpu-memory-utilization 0.9 \
  --attention-backend trtllm \
  --moe-backend flashinfer_trtllm \
  --chunked-prefill-size 8192 \
  --max-model-len 262144 \
  --max-num-seqs 128 \
  --kv-cache-dtype fp8_e4m3 \
  --speculative-algorithm MTP \
  --speculative-draft-model-path Qwen/Qwen3.8-27B-FP8 \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --disable-kvstore \
  --host 0.0.0.0 --port 8000
```

### Qwen3.8-27B(Hopper)

A dense 27B-class Qwen3.8 FP8 checkpoint on a single GPU, with self-speculative
MTP (the draft model path points at the same checkpoint):

```bash
tokenspeed serve Qwen/Qwen3.8-27B-FP8 \
  --served-model-name Qwen/Qwen3.8-27B-FP8 \
  --world-size 1 \
  --gpu-memory-utilization 0.9 \
  --drafter-attention-backend fa3 \
  --chunked-prefill-size 8192 \
  --max-model-len 256000 \
  --max-num-seqs 64 \
  --kv-cache-dtype fp8_e4m3 \
  --speculative-algorithm MTP \
  --speculative-draft-model-path Qwen/Qwen3.8-27B-FP8 \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen \
  --disable-kvstore \
  --host 0.0.0.0 --port 8000
```


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

## DeepSeek V4-Flash / V4-Pro

DeepSeek V4 needs FP8 KV cache, the DeepGEMM `mega_moe` experts, and the FP4
indexer cache. `tokenspeed serve` auto-selects `--reasoning-parser deepseek_v31`
and `--tool-call-parser deepseek_v4`, and auto-sets `block_size=256` (pass
`--block-size N` with `N != 64` to override). Requires
`tokenspeed-deepgemm>=2.5.0.post20260629` and `tokenspeed-flashmla`.

**V4-Flash** — 4× B200 (SM100), data-parallel + expert-parallel:

```bash
tokenspeed serve deepseek-ai/DeepSeek-V4-Flash \
  --served-model-name deepseek-v4-flash \
  --trust-remote-code \
  --data-parallel-size 4 \
  --enable-expert-parallel \
  --kv-cache-dtype fp8_e4m3 \
  --moe-backend mega_moe \
  --attention-use-fp4-indexer-cache \
  --max-model-len 80000 \
  --max-total-tokens 163840 \
  --chunked-prefill-size 8192 \
  --enable-mixed-batch \
  --gpu-memory-utilization 0.9 \
  --disable-kvstore \
  --host 0.0.0.0 \
  --port 8000
```

**V4-Pro** — 8× B200, tensor-parallel:

```bash
tokenspeed serve deepseek-ai/DeepSeek-V4-Pro \
  --served-model-name deepseek-v4-pro \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --kv-cache-dtype fp8_e4m3 \
  --moe-backend flashinfer_trtllm \
  --attention-use-fp4-indexer-cache \
  --max-model-len 80000 \
  --max-total-tokens 2560000 \
  --chunked-prefill-size 8192 \
  --gpu-memory-utilization 0.9 \
  --disable-kvstore \
  --host 0.0.0.0 \
  --port 8000
```

For the expert-parallel topology, swap `--tensor-parallel-size 8` for
`--tensor-parallel-size 8 --enable-expert-parallel --dense-tp-size 1` and
`--moe-backend flashinfer_trtllm` for `--moe-backend mega_moe`.

### MTP speculative decoding

Both variants can drive the checkpoint's NextN/MTP draft layers. Keep the launch
flags above and add:

```bash
--speculative-algorithm MTP \
--speculative-num-steps 3
```

With `--speculative-draft-model-path` omitted, V4 uses the same checkpoint as the
draft source (`DeepseekV4ForCausalLMNextN`). MTP runs on the non-overlap
scheduler — the runtime disables overlap scheduling automatically when
speculative decoding and paged-cache groups are both active — and prefix caching
stays on by default. Add `--enable-metrics` to read `Decoded Tok/Iter` and the
speculative accept rate from the run summary.

### DSpark speculative decoding with Prefix Replay

A V4-Flash checkpoint that includes complete DSpark draft weights can use
same-checkpoint DSpark decoding. Prefix Replay keeps prefix caching enabled
while recomputing the bounded prompt suffix needed to rebuild DSpark's
request-persistent state:

```bash
tokenspeed serve deepseek-ai/DeepSeek-V4-Flash \
  --served-model-name deepseek-v4-flash \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --kv-cache-dtype fp8_e4m3 \
  --moe-backend flashinfer_trtllm \
  --attention-use-fp4-indexer-cache \
  --max-model-len 96000 \
  --max-num-seqs 80 \
  --max-total-tokens 2560000 \
  --max-prefill-tokens 8192 \
  --chunked-prefill-size 8192 \
  --enable-mixed-batch \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.90 \
  --disable-kvstore \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5}' \
  --speculative-eagle-topk 1 \
  --max-cudagraph-capture-size 80 \
  --prefill-graph-max-tokens 2048 \
  --enable-metrics \
  --enable-cache-report \
  --host 0.0.0.0 \
  --port 8000
```

The replay window comes from the checkpoint's DSpark configuration; there is
no user-tuned replay-length flag. Startup fails closed when same-checkpoint
DSpark weights are incomplete, the replay capability is missing, KVStore is
enabled, or the draft checkpoint contains only MTP/NextN weights. External
DSpark checkpoints that do not advertise this capability keep the generic
scheduler behavior.

For a two-node TP8 deployment, run one process per node with four local workers
and the same command on both nodes. See [Multi-Node](../serving/parallelism.md#multi-node)
for explicit topology flags and launcher-derived settings. Before applying
production load, confirm that every rank reports a nonzero Prefix Replay window,
then check completion, speculative acceptance, and cache-hit metrics with fixed
prompts and package/model revisions.

## Tuning Order

1. Set model ID, trust policy, tokenizer mode, and served model name.
2. Set context length and KV cache dtype.
3. Set tensor, data, and expert parallelism to match the node topology.
4. Set scheduler budgets: `--chunked-prefill-size`, `--max-num-seqs`, and only then `--max-total-tokens`.
5. Set attention, MoE, and sampling backends explicitly for benchmark runs.
6. Add reasoning, tool-call, grammar, or speculative decoding only when the model and workload need them.
