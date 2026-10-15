# Ascend DSA operators

## Indexer RoPE

`interleave_rope` fuses position lookup, interleaved-pair rotation into split
halves, and the unchanged tail copy. Validated fast layouts are BF16/FP16,
width 128, rotary width 64, and 1/8/64 heads. Unsupported layouts retain the
CANN composition. Products and additions compute separately in FP32 (no FMA
contraction), then round once to the output dtype. The tail is copied exactly.
This rotates indexer keys; it does not write the attention KV cache.

Tests compare against both CANN and an independent CPU FP32 formula, including
row/head strides, empty inputs, fallback layouts and graph-updated positions:

```bash
pytest -q tokenspeed-kernel-npu/test/test_dsa_rope.py
python tokenspeed-kernel-npu/benchmark/benchmark_dsa_rope.py \
  --mode graph --dtype bf16 --samples 100 --repetitions 20 --output rope-bf16.json
```

Repeat with `--dtype fp16` or `--mode eager`. The synthetic benchmark includes
the entire original lookup/repeat/CANN-RoPE/concat chain. It warms up JIT and
capture first, alternates paired A/B order, retains every sample and checks
outputs. Device events enclose a batch of repetitions; reported per-call time
includes amortized graph launch overhead, or host submission gaps in eager mode.
It is not a whole-layer or serving-throughput benchmark.

## Synthetic graph measurement

Ascend 910B2C, CANN 9.0.0, PyTorch 2.9.0, torch-npu 2.9.0.post2;
100 alternating pairs, 20 calls per graph, device-event microseconds per call.
These full-device synthetic measurements are not a layer speedup claim.

| Dtype | Rows | Heads | CANN chain median µs | Fused median µs | Reduction |
|---|---:|---:|---:|---:|---:|
| BF16 | 64 | 1 | 32.851 | 6.990 | 78.7% |
| BF16 | 64 | 8 | 31.935 | 6.898 | 78.4% |
| BF16 | 64 | 64 | 37.896 | 9.037 | 76.2% |
| FP16 | 64 | 1 | 33.095 | 7.081 | 78.6% |
| FP16 | 64 | 8 | 32.162 | 7.310 | 77.3% |
| FP16 | 64 | 64 | 37.954 | 9.216 | 75.7% |

All 24 measured dtype/shape cases matched the original output exactly. No
sample was trimmed; the FP16 64-row/1-head CANN case had a 205.786 µs maximum
and 34.809 µs mean (33.095 µs median). The benchmark JSON preserves all samples.
