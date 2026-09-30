# Communication kernels

Stateful AllGather/ReduceScatter APIs require a collectively prepared workspace.
Their `stateful_allgather` and `stateful_reduce_scatter` registrations are
excluded from stateless autodispatch; callers own initialization, capacity,
stream ordering and fallback selection.

TokenSpeed Lamport A2A is exported from `communication/cuda.py`. Its private
Python implementation and CUDA sources live under `_cuda/`. The public APIs
are `tokenspeed_a2a_lamport` and `tokenspeed_a2a_lamport_fp8_quantize`. Inputs
must be 16-byte aligned even when contiguous; see
[the protocol contract](tokenspeed_a2a_lamport.md).

## Fused AllGather and FP8 quantization

`TrtllmAllGatherQuantState` and `trtllm_allgather_fp8_quantize` combine BF16
Lamport AllGather with 128-element FP8 activation quantization. Each ready
group is quantized directly into FP8 values and MN-major FP32 scales for the
prepared GEMM. Communication remains BF16. Fusion removes the gathered-BF16
output write/read and a separate quantization launch; it does not reduce
network traffic.

Kimi-K3 QKV projection TP selects this path by default for TP2/TP4, up to 128
physical rows/rank, and compatible prepared block-FP8 GEMM plans. Other plans
keep separate gather and linear execution. See the unified
[TP-sharding recipe](../../../../../docs/recipes/kimi-k3-tp-sharding.md).
This does not apply RMSNorm or use PDL. A following GEMM waits for the fused
kernel to complete normally; there is no overlapping scale consumer.

Create the state collectively before CUDA-graph capture. Pass an explicit
positive `num_blocks` no larger than the device's SM count. Inputs must be
contiguous, finite BF16 with a width divisible by 128 and equal physical row
counts of 1..128 across the subgroup. Empty owners participate with zero
padding. Calls sharing a state must be serialized, and returned buffers are
borrowed until the next fused call. The state retains the ordinary BF16 output
buffer for fallback and reference testing, so fusion does not reduce allocated
scratch.

Quantization matches the prepared-scale path, including the native TRT-LLM
small-value clamp for aligned rows and the existing Triton padding contract
otherwise. Check exact FP8 bytes and scales when changing quantizer
dependencies. TP2/TP4 have GPU correctness coverage; template dispatch also
accepts TP8/TP16, which need their own multi-GPU validation before use.

Run the four-GPU test from the repository root:

```bash
python -m pytest -q \
  tokenspeed-kernel/test/nvidia/ops/communication/test_trtllm_allgather_quant.py
```

It covers eager calls, repeated graph replay with changing inputs, batch-size
changes, signed zeros, tiny/large finite values, padded/empty owners, subgroup
isolation, and interchange with ordinary AllGather on the same workspace.

Measure the complete projection as well as the fused kernel. Removing a
quantization launch does not guarantee lower model latency, and projection
sharding can still lose to a replicated projection once communication is
included. Kernel-unit results are not full-model accuracy or performance
results.
