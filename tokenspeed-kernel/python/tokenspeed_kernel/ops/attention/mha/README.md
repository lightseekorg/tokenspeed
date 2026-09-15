# Multi-head attention

## Sliding-window Triton prefill

The portable Triton prefill kernel bounds its KV tile loop when the sliding
window is positive. The lower bound keeps the first tile that intersects a
query's window. For causal attention without a custom mask, the upper bound
also excludes tiles after the query block.

A custom mask replaces the causal mask and can admit future keys, so it does
not use that upper bound. Full attention and disabled windows retain their
original loop bounds. The optimization skips only tiles that the existing
window mask would reject completely; valid tiles keep the same processing
order and per-element masks.

The regression suite covers cached suffixes, permuted physical pages, ragged
query lengths, query/tile/window boundaries, sinks, and optional log-sum-exp
outputs. It compares against an FP32 reference built from logical KV tensors
and checks input immutability.

```bash
python -m pytest tokenspeed-kernel/test/ops/test_attention_window_bounds.py -q
python -m pytest tokenspeed-kernel/test/ops/test_attention.py \
  -k 'triton and bf16 and (prefill or extend)' -q
```

These checks exercise the portable Triton path on the selected test device.
End-to-end serving latency also includes KV writes and other model work and
must be measured separately from the attention kernel.
