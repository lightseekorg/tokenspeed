# GDN programmatic dependent launch

All GDN compute kernel launchers use the existing global
`tokenspeed_kernel.platform.pdl_enabled()` switch. It defaults to enabled on
NVIDIA SM90 and newer; unsupported platforms always return false. Calling
`pdl_enabled(False)` disables PDL and `pdl_enabled(True)` enables it where
supported. There is no backend-specific or decode-mode-specific switch.

## Dependency contract

A PDL launch attribute alone is insufficient. Each participating CTA executes
grid dependency synchronization **before any input or metadata load**, including
indirect state-pool addresses, replay payloads, gates and the Z branch. The wait
establishes visibility of predecessor writes, including in-place convolution
and recurrent-state updates. Launch readiness is signaled at the end of the
computation; CTAs returning early for padding implicitly signal on exit. The
signal permits the dependent grid to launch but does not replace its wait.
Signaling at entry caused resource contention in MTP measurements; late
signaling avoids that regression.

Triton launchers pass `ENABLE_PDL` and, when enabled, `launch_pdl=True`.
The device specialization emits `gdc_launch_dependents` and `gdc_wait`.
When disabled, neither instruction nor the NVIDIA launch option is present;
AMD continues through the same launchers. PDL is a scheduling opportunity,
not a correctness dependency on concurrent execution.

This follows NVIDIA's [programmatic dependent launch contract](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization).

## Coverage

- Qwen3.5 QKVZBA repacking, causal convolution prefill/update, fused QKV split
  with optional L2 normalization/replay packing, GDN gating and gated RMSNorm.
- Triton single-token decode, MTP, accepted-prefix recurrent replay and the
  prefill stages: L2 norm, cumulative gates, KKT, triangular inverse, WY
  reconstruction, recurrent state and output accumulation.
- FlashInfer FP32 and BF16 single-token decode/MTP and SM100 chunk prefill.
- State-page index/commit kernels and batched state-row copies used by verify.

Triton decode accepts explicit Q/K/V/A/B strides, so packed decode views do
not introduce standalone contiguous-copy kernels between conv and recurrence.
The FP32 MTP adapter retains its empty output and unused intermediate-buffer
view; PDL does not reintroduce either zero fill.

Ordinary PyTorch bookkeeping, conversions and optional checkpoint preparation
retain their normal stream ordering. A non-PDL kernel is an implicit producer
at completion; a non-PDL consumer waits for predecessor completion. These
boundaries are safe but do not overlap through PDL. The switch governs kernel
scheduling; it does not rewrite PyTorch operations or change cache ownership,
padding, checkpoint layout or allocation contracts.

## FlashInfer boundary

FlashInfer's GDN entry points currently omit PDL controls. The adapter under
`tokenspeed-kernel/thirdparty/flashinfer` inlines each original CuTe device body
inside a PDL kernel and uses the original host launch geometry, shared-memory
sizes and argument ABI. It preserves constexpr arguments separately from
runtime arguments, including prefill's kernel instance.

Explicit host entry points bind into a private Python namespace with fresh
compilation/buffer caches. The PDL and upstream non-PDL executors cannot share
a cache entry. Cloning uses CuTe's saved original code if the function has
already been compiled, so an off/on/off switch sequence also works after an
upstream warmup. No installed FlashInfer code, global CuTe launch API or shared
module binding is patched. Missing upstream symbols fail explicitly rather
than silently omitting synchronization after an incompatible dependency update.
Optional backend availability checks remain in place.

## CUDA Graphs and validation

Eager and capture use the same launchers. Choose the switch **before capture**:
CUDA Graph records programmatic dependency edges and replay retains that
choice even if the global switch changes. Recapture graphs to change their PDL
configuration.

`test/runtime/test_gdn_pdl.py` inspects actual CUDA Graph dependency edges,
compares outputs and written states with PDL disabled, and replays with changed
inputs after changing the global switch. An intentionally delayed PDL producer
exposes missing waits before the repack. The tests cover FP32/BF16 states,
T=1/T=3, small/regular batches, both registered decode solutions, and both
prefill solutions. Existing state-paging, ReplaySSM and zero-fill regressions
continue to verify padding and state isolation. The B200 validation passed
95 runtime tests (plus 9 subtests), 76 GDN kernel tests and 35 state index/copy
tests; `pre-commit run --all-files` also passed.

Run runtime and kernel suites separately (their pytest import roots differ):

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m pytest -q test/runtime/test_gdn_pdl.py \
  test/runtime/test_gdn_verify_launches.py test/runtime/test_gdn_state_paging.py \
  test/runtime/models/test_qwen3_5_fused_qkvzba.py \
  test/runtime/layers/test_gdn_qkv_split_fused.py \
  test/runtime/layers/test_gdn_flashinfer_fastpath.py
CUDA_VISIBLE_DEVICES=0 python3 -m pytest -q \
  tokenspeed-kernel/test/ops/test_attention_gdn.py \
  tokenspeed-kernel/test/ops/attention/test_gdn_replay_commit.py
```

## B200 measurements

A CUDA Graph contains 100 consecutive GDN chains (repack, conv, decode or MTP,
optional QKV split, gated RMSNorm), excluding the projection/output GEMMs and
scheduler. Inputs use 4 QK heads, 12 V heads, dimension 128, BF16 activations
and a width-4 convolution. The two PDL settings are measured in alternating
order with CUDA events; entries are medians of 30 samples, in microseconds per
chain. FP32 state uses FlashInfer and BF16 state uses Triton, matching runtime
backend selection. These are chain measurements, not model throughput.

| State/backend | Batch | Tokens/request | PDL off | PDL on |
| --- | ---: | ---: | ---: | ---: |
| FP32/FlashInfer | 1 | 1 | 7.699 | 7.723 |
| FP32/FlashInfer | 8 | 1 | 9.135 | 9.093 |
| FP32/FlashInfer | 1 | 3 | 9.360 | 8.990 |
| FP32/FlashInfer | 8 | 3 | 14.188 | 13.433 |
| BF16/Triton | 1 | 1 | 7.575 | 7.374 |
| BF16/Triton | 8 | 1 | 8.583 | 8.102 |
| BF16/Triton | 1 | 3 | 13.352 | 12.985 |
| BF16/Triton | 8 | 3 | 15.279 | 14.782 |

Single-token FlashInfer is approximately unchanged; the tested MTP chains
improve by 2.7–5.3%. The switch allows deployments to measure their own shapes.
