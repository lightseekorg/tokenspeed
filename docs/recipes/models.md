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
  --reasoning-parser kimi_k2 \
  --tool-call-parser kimi_k2 \
  --host 0.0.0.0 \
  --port 8000
```

For K2.6, keep the same parameter shape and change the checkpoint and parser
only if the model card requires a different value.

## GPT-OSS 20B / 120B

Small GPT-OSS launches can start simple. Large GPT-OSS launches usually tune
tensor parallelism, scheduler token budget, and KV cache dtype.

```bash
tokenspeed serve openai/gpt-oss-20b \
  --served-model-name gpt-oss-20b \
  --tensor-parallel-size 1 \
  --max-model-len 131072 \
  --chunked-prefill-size 8192 \
  --reasoning-parser gpt-oss \
  --tool-call-parser gpt-oss \
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
  --reasoning-parser gpt-oss \
  --tool-call-parser gpt-oss \
  --host 0.0.0.0 \
  --port 8000
```

## DeepSeek V3 / V3.1

DeepSeek V3 and V3.1 models use MLA attention and MoE architecture. They
require `--trust-remote-code` and benefit from expert parallelism.

### DeepSeek-V3-0324 (FP8)

```bash
tokenspeed serve deepseek-ai/DeepSeek-V3-0324   --served-model-name deepseek-v3   --trust-remote-code   --tensor-parallel-size 4   --enable-expert-parallel   --max-model-len 131072   --kv-cache-dtype fp8   --chunked-prefill-size 8192   --max-num-seqs 256   --attention-backend trtllm_mla   --moe-backend flashinfer_trtllm   --reasoning-parser deepseekv3   --tool-call-parser deepseekv3   --host 0.0.0.0   --port 8000
```

### DeepSeek-V3.1 (latest)

V3.1 uses the same architecture as V3 with updated weights and improved
tool-call formatting.

```bash
tokenspeed serve deepseek-ai/DeepSeek-V3.1   --served-model-name deepseek-v3.1   --trust-remote-code   --tensor-parallel-size 4   --enable-expert-parallel   --max-model-len 131072   --kv-cache-dtype fp8   --chunked-prefill-size 8192   --max-num-seqs 256   --attention-backend trtllm_mla   --moe-backend flashinfer_trtllm   --reasoning-parser deepseekv31   --tool-call-parser deepseekv31   --host 0.0.0.0   --port 8000
```

## Qwen3 / Qwen3.5

Qwen3 models support thinking mode (reasoning) and tool calling. Use
`--reasoning-parser qwen3` for thinking mode and `--tool-call-parser qwen3`
for function calling.

### Qwen3-8B (single GPU)

```bash
tokenspeed serve Qwen/Qwen3-8B   --served-model-name qwen3-8b   --tensor-parallel-size 1   --max-model-len 131072   --chunked-prefill-size 8192   --reasoning-parser qwen3   --tool-call-parser qwen3   --host 0.0.0.0   --port 8000
```

### Qwen3-30B-A3B (MoE, multi-GPU)

Qwen3 MoE models benefit from expert parallelism for better throughput.

```bash
tokenspeed serve Qwen/Qwen3-30B-A3B   --served-model-name qwen3-30b-a3b   --tensor-parallel-size 2   --enable-expert-parallel   --max-model-len 131072   --kv-cache-dtype fp8   --chunked-prefill-size 8192   --max-num-seqs 256   --reasoning-parser qwen3   --tool-call-parser qwen3   --host 0.0.0.0   --port 8000
```

### Qwen3-Coder-30B-A3B (code generation)

Qwen3-Coder uses a different tool-call format optimized for code generation.

```bash
tokenspeed serve Qwen/Qwen3-Coder-30B-A3B   --served-model-name qwen3-coder-30b   --tensor-parallel-size 2   --enable-expert-parallel   --max-model-len 262144   --kv-cache-dtype fp8   --chunked-prefill-size 8192   --max-num-seqs 128   --reasoning-parser qwen3   --tool-call-parser qwen3_coder   --host 0.0.0.0   --port 8000
```

### Qwen3.5-27B (latest generation)

Qwen3.5 models use the same parser configuration as Qwen3.

```bash
tokenspeed serve Qwen/Qwen3.5-27B   --served-model-name qwen3.5-27b   --tensor-parallel-size 2   --max-model-len 131072   --kv-cache-dtype fp8   --chunked-prefill-size 8192   --max-num-seqs 256   --reasoning-parser qwen3   --tool-call-parser qwen3.5   --host 0.0.0.0   --port 8000
```

## Tuning Order

1. Set model ID, trust policy, tokenizer mode, and served model name.
2. Set context length and KV cache dtype.
3. Set tensor, data, and expert parallelism to match the node topology.
4. Set scheduler budgets: `--chunked-prefill-size`, `--max-num-seqs`, and only then `--max-total-tokens`.
5. Set attention, MoE, and sampling backends explicitly for benchmark runs.
6. Add reasoning, tool-call, grammar, or speculative decoding only when the model and workload need them.
