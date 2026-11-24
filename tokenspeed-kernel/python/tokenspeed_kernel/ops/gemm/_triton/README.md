# GEMM Triton helpers

Private implementations support the [GEMM APIs](../README.md), including
portable dense and grouped projections. The SM120 scale-packing addition preserves G32 UE8M0 exponent bytes.
The public APIs decide eligibility and keep vendor calls behind the kernel
package boundary.
