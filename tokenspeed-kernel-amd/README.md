# AMD-specific High Performance Kernels

This directory contains high-performance kernel implementations for AMD GPUs.

## MoE kernels

The MXFP/FP8 Gluon MoE kernels live under
`python/tokenspeed_kernel_amd/ops/moe/`.

* `fused_mxfp_gfx950.py` provides the production gfx950 API, including routed
  dispatch/combine wrappers and fused routing helpers.
* `fused_mxfp_gfx1250.py` vendors the gfx1250 Gluon MoE matmul kernel and
  exposes the same lower-level wrapper shape (`matmul`,
  `gluon_mxfp_dispatch_swiglu`, `gluon_mxfp_combine`, and
  `gluon_mxfp_ragged_matmul`) for gfx1250 bring-up. It intentionally keeps
  to the source kernel's feature set; gfx950-only fused route and gate-scaling
  paths are not implied by this wrapper.
