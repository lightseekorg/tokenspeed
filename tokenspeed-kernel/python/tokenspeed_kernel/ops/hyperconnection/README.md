# Gated-residual hyperconnection kernels

This family implements the Qwen gated residual stream behind the public
`tokenspeed_kernel` boundary. Runtime model code owns parameters and checkpoint
mapping; all GPU implementation and dispatch logic lives here.
The public family is GPU-only; numerical references live in tests rather than
as a second registered runtime backend.

For normalized branches `N` shaped `[T, C * H]`, the mix is:

```text
d = SiLU(scale * N @ W_down.T)
g = sigmoid(d @ W_up.T).reshape(T, C, H)
mixed = mean_c(g * N.reshape(T, C, H))
inject = scale * N @ W_inject.T
```

`W_down` and `W_inject` are stacked into one `[R + C, C * H]` parameter so the
wide input is projected once. `gated_residual_mix` returns both `mixed` and the
inject logits. `gated_residual_combine` applies `2 * sigmoid(inject)` while
adding the sublayer output back into every branch without materializing a
broadcast tensor.

## Backends and dispatch

- `triton_persistent_hyperconnection_mix` serves production decode shape
  `C=4, H=2560, R=320, T<=16` on NVIDIA. One resident grid performs both
  projections, activation, sigmoid weighting, and branch reduction. Atomic
  split-K accumulation is disabled when PyTorch deterministic algorithms are
  enabled. Barrier counters and FP32 scratch are keyed by CUDA stream, so
  overlapping streams and separate CUDA graphs do not share synchronization
  state. On Hopper+, the launch waits for its PDL producer before touching
  stream-private scratch and signals the next dependent after its outputs and
  barrier state are complete.
- `triton_hyperconnection_mix` is the portable general path. Dense GEMMs handle
  arbitrary token counts; Triton fuses projection scaling with SiLU and fuses
  sigmoid, branch weighting, and reduction. It is the default for prefill and
  deterministic execution.
- `cute_dsl_hyperconnection_mix` composes the vendored CuTeDSL low-latency BF16
  GEMMs with the same fused Triton epilogues on Blackwell. It remains selectable
  for explicit tuning. Current production measurements favor the one-launch
  persistent path through `T=16` and the general path above it, so CuTeDSL has a
  lower heuristic priority rather than silently regressing the default.
- Grouped Gemma RMSNorm is implemented in `ops/layernorm/triton.py`. One program
  owns each `(token, branch)` group and emits no unused inverse-RMS tensor.

## Full-chain PDL

TokenSpeed serving has one PDL switch: `ServerArgs.disable_pdl`, exposed as
`--disable-pdl`. It defaults to false, so PDL is enabled on NVIDIA Hopper and
newer GPUs. `ServerArgs` applies the effective hardware-gated value to
`tokenspeed-kernel`, TorchInductor, and TRT-LLM in the parent and reapplies it in
every spawned worker before kernel compilation. Disabling the flag explicitly
sets every backend to zero instead of retaining an earlier process setting.

Grouped RMSNorm, the Triton projection epilogue, persistent mix, mix epilogue,
and combine all carry matching PDL wait/launch-dependents hooks. CuTeDSL GEMMs
compile PDL and non-PDL variants under separate cache keys, so the server switch
cannot reuse a kernel compiled with the opposite policy. Vendor GEMMs in the
general prefill path retain ordinary stream dependency semantics; all custom
stages surrounding them remain PDL-aware.

The persistent path is intentionally not copied from implementations that use
one process-global barrier tensor. A process-global tensor can race when CUDA
graphs or model work overlap on multiple streams.

## Projection scaling

Qwen divides down and inject projections by `C`. Runtime loading folds this
factor into BF16/FP16 weights only when `C` is a power of two, where division is
an exact exponent change. For other branch counts, checkpoint weights remain
unchanged and `projection_scale=1/C` is applied to projection results in the
kernel epilogue.

## Verification and benchmark

`tokenspeed-kernel/test/ops/test_hyperconnection.py` checks BF16/FP16 against an
FP64 reference, production dimensions, zero rows, reduce-scatter row views,
full-chain CUDA Graph replay with PDL both enabled and disabled, and concurrent
streams.

The standalone benchmark covers the serving shape matrix
`T=0,1,4,8,16,24,32,128,512,2048,8192` in eager and CUDA Graph modes:

```bash
PYTHONPATH=tokenspeed-kernel/python python \
  tokenspeed-kernel/test/ops/bench_hyperconnection.py --mode both
```

Use `--backend triton`, `--backend persistent`, or `--backend cute_dsl` to pin a
mix implementation.
