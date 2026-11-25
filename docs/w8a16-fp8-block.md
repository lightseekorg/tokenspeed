# FP8 block storage with BF16 computation

Select `--quantization w8a16_fp8_block --dtype bfloat16` explicitly for an
E4M3 checkpoint with FP32 128 × 128 block scales. Checkpoint metadata can use
`quant_method: fp8` or `quant_method: w8a16_fp8_block`; ordinary `fp8` execution
continues to use its existing activation policy.

Each quantized matrix has a `.weight` tensor in `float8_e4m3fn` and a
`.weight_scale_inv` tensor in `float32`. For `[N, K]` weights, scales have shape
`[ceil(N / 128), ceil(K / 128)]`. Expert storage adds the expert dimension.
`modules_to_not_convert` / `ignored_layers` retain mixed BF16 layers.

The loader preserves codes and scales. Immediately before GEMM, the decoder
multiplies them in FP32 and rounds once to BF16. Activations and the selected
GEMM remain BF16. No persistent decoded weight or regenerated encoding is kept.
Merged projections and tensor-parallel shards require block-aligned boundaries.
The numerical GEMM policy remains explicit through the existing kernel planner.
MoE execution requires a registered kernel advertising `w8a16_fp8_block`;
unsupported backends fail selection.

Complete online updates must include both codes and scales. Models using this
format must validate loaded operands after a complete update and preserve
canonical storage through memory release/resume. The runtime tests cover dense
reloads; the kernel tests cover irregular shapes, expert batches and graph replay
with changed codes/scales. Model providers own their numerical and lifecycle
acceptance tests.
