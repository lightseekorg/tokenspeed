# Model Recipes

These recipes start from a known model family, pick the hardware topology, then
set only the parameters that change runtime behavior.

The commands below are templates. Validate exact model IDs, checkpoint formats,
and backend choices against the build you deploy.

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
`tokenspeed-deepgemm>=2.5.0.post20260604` and `tokenspeed-flashmla`.

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

## Tuning Order

1. Set model ID, trust policy, tokenizer mode, and served model name.
2. Set context length and KV cache dtype.
3. Set tensor, data, and expert parallelism to match the node topology.
4. Set scheduler budgets: `--chunked-prefill-size`, `--max-num-seqs`, and only then `--max-total-tokens`.
5. Set attention, MoE, and sampling backends explicitly for benchmark runs.
6. Add reasoning, tool-call, grammar, or speculative decoding only when the model and workload need them.
