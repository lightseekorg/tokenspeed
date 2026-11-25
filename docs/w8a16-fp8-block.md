# FP8 block storage with BF16 computation

Select `--quantization w8a16_fp8_block --dtype bfloat16` explicitly for an
E4M3 checkpoint with FP32 128 × 128 block scales. Checkpoint metadata can use
`quant_method: fp8` or `quant_method: w8a16_fp8_block`; ordinary `fp8` execution
continues to use its existing activation policy.

Absent or null checkpoint quantization metadata leaves the model unquantized.
Configuration normalization and wrapped text-config extraction preserve that
meaning; a model provider may explicitly declare the optional field as `None`.

Each quantized matrix has a `.weight` tensor in `float8_e4m3fn` and a
`.weight_scale_inv` tensor in `float32`. For `[N, K]` weights, scales have shape
`[ceil(N / 128), ceil(K / 128)]`. Expert storage adds the expert dimension.
`modules_to_not_convert` / `ignored_layers` retain mixed BF16 layers.

The loader preserves codes and scales. Immediately before GEMM, the decoder
multiplies them in FP32 and rounds once to BF16. Activations and the selected
GEMM remain BF16. No persistent decoded weight or regenerated encoding is kept.
The Linear method has quantized storage, so fast paths requiring an ordinary
unquantized BF16 weight do not accept it.
Merged projections and tensor-parallel shards require block-aligned boundaries.
The numerical GEMM policy remains explicit through the existing kernel planner.
The `rl-bitwise` envelope requires a batch-invariant BF16 GEMM implementation;
when none is registered for the device and shape, layer construction fails.
MoE execution requires a registered kernel advertising `w8a16_fp8_block`;
unsupported backends fail selection.

Complete online updates must include both codes and scales. Models using this
format must validate loaded operands after a complete update and preserve
canonical storage through memory release/resume. The runtime tests cover dense
reloads and numerical-policy admission. The shared kernel tests cover irregular
shapes, expert batches beyond int32 element offsets and graph replay with changed
codes/scales on supported GPU vendors. Model providers own their numerical and
lifecycle acceptance tests.
