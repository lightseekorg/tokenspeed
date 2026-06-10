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

## Nemotron-H NVFP4

Nemotron-H is a hybrid Mamba2, attention, MLP, and non-gated MoE model. Use the
hybrid attention backend, the in-tree Triton Mamba2 kernels from
`tokenspeed-kernel`, and FlashInfer TRT-LLM NVFP4 MoE for the `relu2` experts.

```bash
tokenspeed serve nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
  --served-model-name nemotron-3-super-120b-a12b \
  --quantization modelopt_mixed \
  --tensor-parallel-size 1 \
  --attention-backend flashinfer \
  --moe-backend flashinfer_trtllm \
  --mamba-ssm-dtype float32 \
  --max-model-len 2048 \
  --max-num-seqs 1 \
  --chunked-prefill-size 512 \
  --block-size 1 \
  --skip-server-warmup \
  --enforce-eager \
  --disable-overlap-schedule \
  --host 127.0.0.1 \
  --port 7999
```

The smg gateway binds to `--port`; TokenSpeed's OpenAI-compatible control
server binds to the next port. With the command above, use
`http://127.0.0.1:8000/v1/chat/completions`:

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nemotron-3-super-120b-a12b",
    "messages": [
      {
        "role": "user",
        "content": "what is the capital of Chad? Answer with only the city name."
      }
    ],
    "max_tokens": 16,
    "temperature": 0,
    "reasoning_effort": "none"
  }'
```

The `modelopt_mixed` path keeps the checkpoint's ModelOpt mixed-precision
layout: FP8 dense projections are loaded through the FP8 path, NVFP4 experts
through the FlashInfer TRT-LLM MoE path, and CUTLASS-capable CUDA targets use
SGLang-compatible channelwise FP8 scale layout while preserving the checkpoint
FP8 shard scales.

The Mamba2 kernel code was vendored from `sgl-project/sglang` commit
`03c77dc33d0a051aa15c1235407440d9d107b98f`, which carries vLLM and
state-spaces/mamba lineage in the copied files.

`ts serve` detects Nemotron-H / Nemotron-3 checkpoints and applies the SGLang
reasoning behavior automatically: the engine uses the `nemotron_3` reasoning
parser for grammar deferral, while the smg gateway receives its equivalent
`qwen3` reasoning parser and `qwen_coder` tool parser. Chat-completion requests
with `reasoning_effort: "none"` are normalized to the model template's
`enable_thinking: false` switch before they reach the gateway, matching SGLang's
OpenAI request handling.

## Tuning Order

1. Set model ID, trust policy, tokenizer mode, and served model name.
2. Set context length and KV cache dtype.
3. Set tensor, data, and expert parallelism to match the node topology.
4. Set scheduler budgets: `--chunked-prefill-size`, `--max-num-seqs`, and only then `--max-total-tokens`.
5. Set attention, MoE, and sampling backends explicitly for benchmark runs.
6. Add reasoning, tool-call, grammar, or speculative decoding only when the model and workload need them.
