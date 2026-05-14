# DeepSeek V4-Flash

## Launch command

DP=4 + expert parallel + mega_moe + FP8 KV cache (B200, 4× SM100):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 exec ts serve \
    --model deepseek-ai/DeepSeek-V4-Flash \
    --host localhost --port 30100 \
    --dist-init-addr 127.0.0.1:4013 \
    --trust-remote-code \
    --data-parallel-size 4 \
    --enable-expert-parallel \
    --kv-cache-dtype fp8_e4m3 \
    --moe-backend mega_moe \
    --attention-use-fp4-indexer-cache \
    --max-model-len 4096 \
    --max-total-tokens 16384 \
    --chunked-prefill-size 8192 \
    --gpu-memory-utilization 0.9 \
    --disable-kvstore
```

## Required flags

| Flag | Why |
|---|---|
| `--data-parallel-size 4` + `--enable-expert-parallel` | V4 ships with EP=4 weight sharding. |
| `--kv-cache-dtype fp8_e4m3` | V4 SWA cache rows are uint8-packed FP8 NoPE + BF16 RoPE + UE8M0 scale; FP8 e4m3 is the only supported KV dtype. |
| `--moe-backend mega_moe` | Activates the DeepGEMM `fp8_fp4_mega_moe` fused experts. Requires `tokenspeed-deepgemm>=2.5.0.post20260424`. |
| `--attention-use-fp4-indexer-cache` | Stores indexer keys as MXFP4 (`[values \| ue8m0 scales]`); the FP8 fallback path is reference-only. |
| `--trust-remote-code` | The HF config uses model-class architectures registered via remote code. |

## Block size

V4 uses `block_size=256` (`block_size / compress_ratio` cleanly divides the
HCA/CSA/SWA layouts). The model loader auto-overrides `block_size` to 256 at
config-init time when the value is the `ServerArgs` class default (currently
`64`); pass `--block-size <N>` with `N != 64` to keep `<N>`. (Passing
`--block-size 64` explicitly is indistinguishable from the default and will
also be bumped to 256.)

## Optional flags

- `--disable-deepseek-v4-fast-mhc`: disables the tilelang+DeepGEMM fused mHC
  path; falls back to the PyTorch reference. Use when tilelang or DeepGEMM JIT
  is unavailable.
- `--deepseek-v4-mega-moe-max-num-tokens N`: caps the DeepGEMM mega_moe
  workspace (`0` lets the kernel pick).
- `--deepseek-v4-indexer-prefill-max-logits-mb N`: caps the FP4 indexer
  prefill logits buffer in MB (default 512).

## MTP speculative decoding

DeepSeek V4 can use the checkpoint's NextN/MTP draft layers through the standard
speculative flags. For `num_steps > 1`, keep the main V4 launch flags and add:

```bash
--speculative-algorithm MTP \
--speculative-num-steps 3
```

When `--speculative-draft-model-path` is omitted for MTP, TokenSpeed uses the
same V4 checkpoint as the draft source and loads the `DeepseekV4ForCausalLMNextN`
architecture.

## Hardware / dependency requirements

- 4× NVIDIA Blackwell SM100 (B200) GPUs.
- `tokenspeed-deepgemm>=2.5.0.post20260424` (mega_moe + FP4 indexer symbols).
- `tilelang==0.1.9` (fast mHC fused kernels). Pulled in automatically via
  `tokenspeed-kernel`'s `pyproject.toml`.
- `flash_mla` (provided by `tokenspeed-flashmla`) — required for sparse decode
  and prefill.

## Validating the deployment

GSM8K 5-shot, 50 samples is the standard quick-validation harness for V4:

```bash
HF_DATASETS_TRUST_REMOTE_CODE=1 lm_eval run \
    --model local-completions \
    --model_args "model=deepseek-ai/DeepSeek-V4-Flash,base_url=http://127.0.0.1:30100/v1/completions,tokenized_requests=False,tokenizer_backend=None,num_concurrent=4,max_retries=1,timeout=600,max_gen_toks=256" \
    --tasks gsm8k --num_fewshot 5 --limit 50 --batch_size 1 \
    --gen_kwargs temperature=0
```

Expected `exact_match`: **0.96-0.98 ± 0.04**. Below ~0.86 indicates a real
regression.

`tokenizer_backend=None` is required because the V4-Flash tokenizer config
does not load through `transformers.AutoTokenizer.from_pretrained`.
