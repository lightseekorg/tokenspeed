# Ascend DSA operators

## Indexer RoPE

`interleave_rope` fuses position lookup, interleaved-pair rotation into split
halves, and the unchanged tail copy. Validated fast layouts are BF16/FP16,
width 128, rotary width 64, and 1/4/8/16/32/64 heads. The Ascend framework entry
registers BF16 and FP32; FP32 retains the CANN path. FP16 is supported only
through the direct NPU helper and is not registered at the framework entry.
Products and additions compute separately in FP32 (no FMA
contraction), then round once to the output dtype. The tail is copied exactly.
This rotates indexer keys; it does not write the attention KV cache.
Inputs outside the fused kernel's metadata requirements fall back to the
original composition, preserving its accepted inputs and error handling.

Tests compare against both CANN and an independent CPU FP32 formula, including
token/head strides, empty inputs, fallback layouts and graph-updated positions.
Tests cover 264 CANN/CPU comparisons, 48 graph-update cases,
24 BF16 public-framework dispatch cases and 15 empty/fallback cases, including
metadata layouts accepted by the original composition.
The measured token counts are T = 1/8/64/65/128/256/512/1024. H = 1 represents
2D keys; H = 4/8/16/32/64 covers local query head counts for total head counts
32 or 64 with TP1/2/4/8, without requiring the caller to shard heads.
Both contiguous and outer-strided input views keep the feature axis contiguous.
Graph position updates are checked at T = 64 and 1024 for every head count,
dtype and input layout.

The dispatcher only fuses the ranges that improve both full-core and half-core
measurements; other token counts, head counts and layouts retain CANN:

| Local heads H | Maximum fused T |
|---:|---:|
| 1 / 4 | 256 |
| 8 / 16 / 32 | 512 |
| 64 | 1024 |

Each program handles one token, so large T with few heads can be slower than
the CANN composition. Tests also cover 257, 513 and 1025 to check the dispatch
boundaries. The framework entry is exercised with BF16, both layouts and
every supported head count at T = 64 and 1024.

```bash
pytest -q tokenspeed-kernel-npu/test/test_dsa_rope.py
python tokenspeed-kernel-npu/benchmark/benchmark_dsa_rope.py \
  --mode graph --dtype bf16 --layout contiguous \
  --samples 100 --repetitions 20 --output rope-bf16.json
```

Repeat with `--dtype fp16`, `--layout outer-strided` or `--mode eager`.
Use `--tokens` to select T (`--rows` remains an alias). The synthetic benchmark includes
the entire original lookup/repeat/CANN-RoPE/concat chain. It warms up JIT and
capture first, alternates paired A/B order, retains every sample and checks
outputs. Device events enclose a batch of repetitions; reported per-call time
includes amortized graph launch overhead, or host submission gaps in eager mode.
It is not a whole-layer or serving-throughput benchmark.

## Synthetic graph measurement

Ascend 910B2C, CANN 9.0.0, PyTorch 2.9.0, torch-npu 2.9.0.post2;
100 alternating pairs, 20 calls per graph, device-event microseconds per call.
The capture stream explicitly receives the caller's core limits, and its actual
limits are recorded in the benchmark JSON. These synthetic measurements are
not a layer speedup claim.

First sweep: BF16, contiguous inputs, full device (48 vector / 24 cube cores):

| T | H | CANN chain median µs | Selected path median µs | Reduction |
|---:|---:|---:|---:|---:|
| 64 | 1 | 33.377 | 7.347 | 78.0% |
| 64 | 4 | 34.358 | 7.571 | 78.0% |
| 64 | 8 | 32.181 | 7.105 | 77.9% |
| 64 | 16 | 33.967 | 7.476 | 78.0% |
| 64 | 32 | 34.832 | 8.213 | 76.4% |
| 64 | 64 | 37.483 | 9.197 | 75.5% |
| 512 | 8 | 42.731 | 38.237 | 10.5% |
| 1024 | 64 | 111.512 | 78.435 | 29.7% |

The candidate benchmark spans 384 dtype/T/H/layout/core-limit combinations:
both dtypes, both input layouts and 48-vector/24-cube or 24-vector/12-cube
limits. Of those, 328 combinations fall inside the enabled ranges above;
all 328 improve, by 7.4–88.0%, with at least 99 of 100 paired samples faster.
All candidate outputs agree with CANN. Large-T/few-head cases remain on CANN;
for example, T=1024/H=1 fusion regressed by up to 60.1%. No samples are trimmed.

A final dispatch check repeats all 144 combinations at T=256/512/1024:
the 88 fused cases improve by 2.5–60.5%, with at least 99/100 paired samples
faster. The other 56 cases execute CANN; their measured median differences
range from 1.9% slower to 1.1% faster. These results retain both runs' timing
variation rather than treating the first sweep as a guaranteed speedup.
