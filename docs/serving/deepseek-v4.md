# DeepSeek V4-Flash

## Launch command

DP=4 + expert parallel + mega_moe + FP8 KV cache (B200, 4× SM100):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 tokenspeed serve deepseek-ai/DeepSeek-V4-Flash \
    --host localhost --port 8000 \
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
    --enable-mixed-batch \
    --gpu-memory-utilization 0.9 \
    --disable-kvstore
```

## Required flags

| Flag | Why |
|---|---|
| `--data-parallel-size 4` + `--enable-expert-parallel` | V4 ships with EP=4 weight sharding. |
| `--kv-cache-dtype fp8_e4m3` | V4 SWA cache rows are uint8-packed FP8 NoPE + BF16 RoPE + UE8M0 scale; FP8 e4m3 is the only supported KV dtype. |
| `--moe-backend mega_moe` | Activates the DeepGEMM `fp8_fp4_mega_moe` fused experts. Requires `tokenspeed-deepgemm>=2.5.0.post20260604`. |
| `--attention-use-fp4-indexer-cache` | Stores indexer keys as MXFP4 (`[values \| ue8m0 scales]`); the FP8 fallback path is reference-only. |
| `--enable-mixed-batch` | Lets the scheduler issue prefill and decode requests in the same iteration. Off by default globally; opt in per workload. |
| `--trust-remote-code` | The HF config uses model-class architectures registered via remote code. |

## Parser defaults

`tokenspeed serve deepseek-ai/DeepSeek-V4-Flash` automatically selects
`--reasoning-parser deepseek_v31` and `--tool-call-parser deepseek_v4`.
Pass explicit parser flags to override these defaults.

## Block size

V4 uses `block_size=256` (`block_size / compress_ratio` cleanly divides the
HCA/CSA/SWA layouts). The model loader auto-overrides `block_size` to 256 at
config-init time when the value is the `ServerArgs` class default (currently
`64`); pass `--block-size <N>` with `N != 64` to keep `<N>`. (Passing
`--block-size 64` explicitly is indistinguishable from the default and will
also be bumped to 256.)

## Optional flags

- `--deepseek-v4-mega-moe-max-num-tokens N`: caps the DeepGEMM mega_moe
  workspace (`0` lets the kernel pick).
- `--deepseek-v4-indexer-prefill-max-logits-mb N`: caps the FP4 indexer
  prefill logits buffer in MB (default 512).

## MTP speculative decoding

DeepSeek V4 can use the checkpoint's NextN/MTP draft layers through the standard
speculative flags. For `num_steps > 1`, keep the main V4 launch flags and add:

```bash
--speculative-algorithm MTP \
--speculative-num-steps 3 \
--enable-metrics
```

When `--speculative-draft-model-path` is omitted for MTP, TokenSpeed uses the
same V4 checkpoint as the draft source and loads the `DeepseekV4ForCausalLMNextN`
architecture.

For multi-step MTP, draft decode metadata must advance from the accepted prefix
length (`valid_cache_len + accept_len`), not from the full verify width. This
keeps later draft steps from attending over rejected verify-tail KV after partial
accepts.

DeepSeek V4 attention also carries separate paged cache groups for SWA,
compressed KV, compressor state, and CSA indexer state. Speculative target and
draft metadata paths must forward those group block tables and base logical-page
offsets together; falling back to the ordinary request page table is not a valid
V4 MTP cache layout. Compressed attention reads, compressed KV/indexer inserts,
and indexer decode plans must treat compact group tables as `logical_page -
base_offset`. Multi-token decode metadata should derive compressed cache slots
from each token's position within the verify span.

Draft decode SWA metadata has the same requirement. The drafter cache uses V4
compact SWA pages (64 rows in the default layout), while ordinary request pages
can be larger. Multi-step draft decode must keep the compact SWA block table from
the draft-extend/prefill metadata and refresh `decode_swa_indices`/`decode_swa_lens`
after each accepted-prefix advance; otherwise later draft steps can read stale or
out-of-range SWA rows.

Scheduler prefix caching is supported for the phase-1 V4 MTP path and is enabled
by default. No extra flag is needed for normal serving; use
`--no-enable-prefix-caching` only for ablation or debugging. During MTP decode,
TokenSpeed reuses previously accepted prefix state while keeping unaccepted draft
tokens out of the reusable prefix cache.

Keep this path on the non-overlap scheduler. The runtime disables overlap
scheduling when speculative decoding and paged-cache groups are both active, and
that is the supported phase-1 boundary. State-family reuse remains conservative:
if SWA, compressed KV, compressor state, or CSA indexer state cannot be restored
from a complete accepted snapshot, the request should fall back to recomputing
that state rather than treating the speculative tail as reusable prefix state.

SWA cache insert slot mappings are validated against the current SWA group cache
capacity before invoking fused cache writers. Multi-step MTP can carry padded or
draft-tail slot entries that are valid as ignored tokens but not valid physical
SWA cache locations; these entries must stay negative or be masked before the
writer touches paged cache memory.

MTP advances each request by the sampled accepted length, not by a fixed verify
width. The next target-verify step must be scheduled only after the scheduler has
observed the previous accepted length, so group block tables and base
logical-page offsets are built from accepted scheduler truth rather than from the
speculative reserve width.

With `--enable-metrics`, check the run summary under `--outputs-dir` for
`Decoded Tok/Iter` and speculative accept-rate fields when comparing prefix-cache
default runs against `--no-enable-prefix-caching` ablations.

## Hardware / dependency requirements

- 4× NVIDIA Blackwell SM100 (B200) GPUs.
- `tokenspeed-deepgemm>=2.5.0.post20260604` (mega_moe + FP4 indexer symbols).
- `tilelang==0.1.11` (fast mHC fused kernels). Pulled in automatically via
  `tokenspeed-kernel`'s `pyproject.toml`.
- `flash_mla` (provided by `tokenspeed-flashmla`) — required for sparse decode
  and prefill.

## Validating the deployment

GSM8K 5-shot, 50 samples is the standard quick-validation harness for V4:

```bash
HF_DATASETS_TRUST_REMOTE_CODE=1 lm_eval run \
    --model local-completions \
    --model_args "model=deepseek-ai/DeepSeek-V4-Flash,base_url=http://127.0.0.1:8000/v1/completions,tokenized_requests=False,tokenizer_backend=None,num_concurrent=4,max_retries=1,timeout=600,max_gen_toks=256" \
    --tasks gsm8k --num_fewshot 5 --limit 50 --batch_size 1 \
    --gen_kwargs temperature=0
```

Expected `exact_match`: **0.96-0.98 ± 0.04**. Below ~0.86 indicates a real
regression.

`tokenizer_backend=None` is required because the V4-Flash tokenizer config
does not load through `transformers.AutoTokenizer.from_pretrained`.
