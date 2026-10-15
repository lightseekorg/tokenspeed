# Head-major DSA value projection

`AscendDSAKernels.project_value(attention, weight)` wraps CANN's existing
TransposeBatchMatMul. It accepts contiguous attention `[H,T,512]` and weights
`[H,512,Dv]`, and returns contiguous `[T,H*Dv]`. The matmul writes token-major
output, replacing BMM followed by a materialized head/token transpose. No new
native kernel is added.

The packed-query adapter can hand off its native head-major layout directly. The helper itself can also consume the old
backend's head-major result. It is an explicit NPU operator helper, not an
automatic rewrite of model code. Model implementations must call it at their
value projection; this PR does not integrate any model consumer. The model
continues to own value weights, gating and output projection. Consequently,
merging this helper alone does not change a model's serving latency.

```bash
pytest -q tokenspeed-kernel-npu/test/test_dsa_value.py
python tokenspeed-kernel-npu/benchmark/benchmark_dsa_value.py \
  --mode graph --dtype bf16 --samples 100 --repetitions 20 --output value-bf16.json
```

The synthetic A/B starts with the same head-major attention on both sides,
matching the packed-query change's output. It measures the incremental value
projection change, not packed-query or complete-layer gains. It alternates A/B
order after warmup, reports every sample and checks exact outputs. Device event
timings include amortized graph launch overhead. Eager mode additionally
includes host submission gaps. Shape and hardware affect results.

## Synthetic graph measurement

Ascend 910B2C, CANN 9.0.0, PyTorch 2.9.0, torch-npu 2.9.0.post2;
BF16, 100 alternating pairs, 20 repetitions per graph, device-event µs per call.
Both sides start from the same head-major input. All 12 shape cases matched
exactly in the benchmark, with every timing sample retained.

| Tokens | Heads | BMM + transpose median µs | TBMM median µs | Change |
|---|---:|---:|---:|---:|
| 1 | 8 | 4.917 | 8.806 | slower by 3.889 µs |
| 64 | 8 | 18.176 | 9.361 | saved 8.815 µs |
| 64 | 32 | 28.634 | 10.476 | saved 18.158 µs |
| 64 | 64 | 19.821 | 11.905 | saved 7.916 µs |

The single-token case regresses: its head/token transpose need not copy data.
The helper is opt-in; callers must benchmark their shapes instead of replacing
all BMM calls unconditionally. These numbers are operator-chain measurements,
not whole-layer gains, and must not be added to packed-query savings.
