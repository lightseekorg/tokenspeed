# Grouped sigmoid routing precision

The `bf16_vendor_fp32_route`, `nvfp4_vendor_fp32_route` and `fp8ch_vendor_fp32_route` policies require FP32 router logits and correction bias for grouped expert selection.

The kernel registry admits selection inside the expert invocation only after its capability hook prepares the checked routing implementation. Unrecognized source, unavailable compilation, explicit precomputed routing and expert parallelism retain separate routing. Quantized policies also retain separate routing for other routing methods and for a single group, whose backend router does not receive the exponential-sigmoid source correction. Model code uses the existing layer and bypassed TopK container; it does not invoke backend kernels.

The fused route computes sigmoid as `1 / (1 + exp(-logit))` in FP32 and adds the selection-only correction bias without a BF16 cast. Group selection uses the two largest corrected scores per group; the chosen groups supply the final experts. The unbiased sigmoid scores supply the route weights.

Quantized fused routes retain FP32 coefficients through routing, packed route storage and expert finalization. The packed buffer reserves eight bytes per FP32 score/index pair. The top-eight normalization uses the reference shuffle-down reduction tree, divides by the score sum plus `1e-20`, then applies the routed scale. BF16 expert routing retains its existing BF16 coefficient contract. Quantized qualification compares weights by expert ID with `rtol=1e-6, atol=1e-7` and requires identical selected expert sets, including tie and near-tie cases. This is an empirical regression tolerance, not a universal error bound: the backend's fast-math sigmoid and tie ordering can still differ from the separate implementation, so bitwise-identical model outputs are not promised.

The routed scale is applied once, in route weights; output scale remains one. Quantized EP uses precomputed FP32 routes. NVFP4 keeps its existing input-quantization overlap and generic tactic adapter when the FP32-bias capability is not requested. Empty batches bypass the backend, and readiness is resolved before CUDA graph capture.

Deferred finalization uses an owned coefficient buffer only when the native launcher returns one. Ungrouped NVFP4 routing borrows the caller's buffer and retains its BF16 coefficient interpretation, including the live BF16 prefix of an FP32-typed carrier. The general adapter preserves this contract even though the model precision policies route single groups separately.

Output qualification compares against the existing FP32-coefficient path. Readiness requires the private FP32 coefficient implementation even if the installed backend already preserves FP32 selection bias. Unsupported source layouts fail closed to separate routing.

This moves selection into the MoE API invocation. Router matrix multiplication remains separate, and the backend still launches internal routing, permutation and expert kernels.
