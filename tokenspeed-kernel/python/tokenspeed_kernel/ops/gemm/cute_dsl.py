# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

from typing import Optional, Tuple

from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.registry import error_fn

platform = current_platform()

nvfp4_gemm_swiglu_nvfp4_quant = error_fn

if platform.is_nvidia:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import torch
    from flashinfer.cute_dsl.utils import (
        get_cutlass_dtype,
        get_max_active_clusters,
        make_ptr,
    )
    from flashinfer.utils import get_compute_capability
    from tokenspeed_kernel.thirdparty.cute_dsl.nvfp4_gemm_swiglu_nvfp4_quant import (
        Sm100BlockScaledPersistentDenseGemmKernel,
    )

    _nvfp4_gemm_swiglu_nvfp4_quant_kernel_cache: dict[tuple, object] = {}

    def _round_up(value: int, multiple: int) -> int:
        return (value + multiple - 1) // multiple * multiple

    def _get_compiled_nvfp4_gemm_swiglu_nvfp4_quant_kernel(
        *,
        a_ptr,
        b_ptr,
        a_sf_ptr,
        b_sf_ptr,
        c_ptr,
        c_sf_ptr,
        alpha_ptr,
        norm_const_ptr,
        m: int,
        n: int,
        k: int,
        l: int,
        max_active_clusters: int,
        stream,
        ab_dtype: str,
        sf_dtype: str,
        c_dtype: str,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_prefetch: bool,
        prefetch_dist: int,
        vectorized_f32: bool,
        enable_pdl: bool,
    ):
        cache_key = (
            n,
            k,
            ab_dtype,
            sf_dtype,
            c_dtype,
            sf_vec_size,
            mma_tiler_mn,
            cluster_shape_mn,
            use_prefetch,
            prefetch_dist,
            vectorized_f32,
            enable_pdl,
        )

        if cache_key not in _nvfp4_gemm_swiglu_nvfp4_quant_kernel_cache:
            gemm = Sm100BlockScaledPersistentDenseGemmKernel(
                sf_vec_size=sf_vec_size,
                mma_tiler_mn=mma_tiler_mn,
                cluster_shape_mn=cluster_shape_mn,
                use_prefetch=use_prefetch,
                prefetch_dist=prefetch_dist,
                vectorized_f32=vectorized_f32,
            )
            _nvfp4_gemm_swiglu_nvfp4_quant_kernel_cache[cache_key] = cute.compile(
                gemm.wrapper,
                a_ptr,
                b_ptr,
                a_sf_ptr,
                b_sf_ptr,
                c_ptr,
                c_sf_ptr,
                alpha_ptr,
                norm_const_ptr,
                m,
                n,
                k,
                l,
                scaling_vector_size=sf_vec_size,
                max_active_clusters=max_active_clusters,
                stream=stream,
                use_pdl=enable_pdl,
            )

        return _nvfp4_gemm_swiglu_nvfp4_quant_kernel_cache[cache_key]

    def nvfp4_gemm_swiglu_nvfp4_quant(
        a: torch.Tensor,
        a_scale: torch.Tensor,
        b: torch.Tensor,
        b_scale: torch.Tensor,
        alpha: torch.Tensor,
        output_global_scale: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
        out_scale: Optional[torch.Tensor] = None,
        ab_dtype: str = "float4_e2m1fn",
        sf_dtype: str = "float8_e4m3fn",
        c_dtype: str = "float4_e2m1fn",
        sf_vec_size: int = 16,
        use_prefetch: bool = False,
        prefetch_dist: int = 3,
        vectorized_f32: bool = True,
        enable_pdl: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """NVFP4 GEMM fused with SwiGLU and NVFP4 output quantization.

        Args:
            a: FP4-packed input activation, shape ``[M, K / 2]``.
            a_scale: Swizzled NVFP4 input scales, shape ``[round_up(M,128), round_up(K/16,4)]``.
            b: FP4-packed interleaved FC1 weight, shape ``[2 * I, K / 2]``.
            b_scale: Swizzled interleaved FC1 weight scales.
            alpha: GEMM global dequant scale, scalar or ``[1, 1]``.
            output_global_scale: Output quantization scale-up factor.
            enable_pdl: Enable Programmatic Dependent Launch for this fused kernel.

        Returns:
            ``(out_fp4, out_scale)`` directly consumable by NVFP4 ``down_proj``.
        """
        if ab_dtype != "float4_e2m1fn" or c_dtype != "float4_e2m1fn":
            raise ValueError(
                "nvfp4_gemm_swiglu_nvfp4_quant currently supports NVFP4 input and output only"
            )
        if a.device.type != "cuda" or b.device.type != "cuda":
            raise ValueError("nvfp4_gemm_swiglu_nvfp4_quant requires CUDA tensors")

        major, minor = get_compute_capability(a.device)
        if major != 10:
            raise ValueError(
                "nvfp4_gemm_swiglu_nvfp4_quant requires Blackwell SM100 family, "
                f"got SM{major}{minor}"
            )

        m = a.shape[0]
        k = a.shape[1] * 2
        n = b.shape[0]
        if b.shape[1] * 2 != k:
            raise ValueError(f"Shape mismatch: A K={k}, B K={b.shape[1] * 2}")
        if n % 2 != 0:
            raise ValueError(f"Interleaved FC1 N must be even, got {n}")

        l = 1
        n_out = n // 2
        if n_out % sf_vec_size != 0:
            raise ValueError(
                f"Output N={n_out} must be divisible by sf_vec_size={sf_vec_size}"
            )
        scale_n_out = n_out // sf_vec_size
        padded_m = _round_up(m, 128)
        padded_scale_n = _round_up(scale_n_out, 4)

        ab_dtype_cutlass = get_cutlass_dtype(ab_dtype)
        sf_dtype_cutlass = get_cutlass_dtype(sf_dtype)
        c_dtype_cutlass = get_cutlass_dtype(c_dtype)

        # Select the tile shape from the current M dimension.
        if m <= 128:
            mma_tiler_mn, cluster_shape_mn = (128, 128), (1, 2)
        else:
            mma_tiler_mn, cluster_shape_mn = (256, 128), (2, 1)

        if not Sm100BlockScaledPersistentDenseGemmKernel.can_implement(
            ab_dtype_cutlass,
            sf_dtype_cutlass,
            sf_vec_size,
            c_dtype_cutlass,
            mma_tiler_mn,
            cluster_shape_mn,
            m,
            n,
            k,
            l,
            a_major="k",
            b_major="k",
            c_major="n",
        ):
            raise ValueError(
                "Unsupported nvfp4_gemm_swiglu_nvfp4_quant configuration: "
                f"shape=(M={m}, N={n}, K={k}), mma_tiler_mn={mma_tiler_mn}, "
                f"cluster_shape_mn={cluster_shape_mn}"
            )

        if out is None:
            out = torch.empty((m, n_out // 2), dtype=torch.uint8, device=a.device)
        if out_scale is None:
            out_scale = torch.empty(
                (padded_m, padded_scale_n),
                dtype=torch.float8_e4m3fn,
                device=a.device,
            )

        if alpha.dim() == 0:
            alpha = alpha.view(1, 1)
        elif alpha.dim() == 1:
            alpha = alpha.view(1, 1)
        if output_global_scale.dim() == 0:
            output_global_scale = output_global_scale.view(1)

        a_ptr = make_ptr(
            ab_dtype_cutlass,
            a.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=32,
        )
        b_ptr = make_ptr(
            ab_dtype_cutlass,
            b.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=32,
        )
        a_sf_ptr = make_ptr(
            sf_dtype_cutlass,
            a_scale.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        b_sf_ptr = make_ptr(
            sf_dtype_cutlass,
            b_scale.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        c_ptr = make_ptr(
            c_dtype_cutlass,
            out.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=32,
        )
        c_sf_ptr = make_ptr(
            sf_dtype_cutlass,
            out_scale.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        alpha_ptr = make_ptr(cutlass.Float32, alpha.data_ptr(), cute.AddressSpace.gmem)
        norm_const_ptr = make_ptr(
            cutlass.Float32,
            output_global_scale.data_ptr(),
            cute.AddressSpace.gmem,
        )

        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        max_active_clusters = get_max_active_clusters(
            cluster_shape_mn[0] * cluster_shape_mn[1]
        )

        compiled_gemm = _get_compiled_nvfp4_gemm_swiglu_nvfp4_quant_kernel(
            a_ptr=a_ptr,
            b_ptr=b_ptr,
            a_sf_ptr=a_sf_ptr,
            b_sf_ptr=b_sf_ptr,
            c_ptr=c_ptr,
            c_sf_ptr=c_sf_ptr,
            alpha_ptr=alpha_ptr,
            norm_const_ptr=norm_const_ptr,
            m=m,
            n=n,
            k=k,
            l=l,
            max_active_clusters=max_active_clusters,
            stream=stream,
            ab_dtype=ab_dtype,
            sf_dtype=sf_dtype,
            c_dtype=c_dtype,
            sf_vec_size=sf_vec_size,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            use_prefetch=use_prefetch,
            prefetch_dist=prefetch_dist,
            vectorized_f32=vectorized_f32,
            enable_pdl=bool(enable_pdl),
        )

        compiled_gemm(
            a_ptr,
            b_ptr,
            a_sf_ptr,
            b_sf_ptr,
            c_ptr,
            c_sf_ptr,
            alpha_ptr,
            norm_const_ptr,
            m,
            n,
            k,
            l,
            stream=stream,
        )
        return out, out_scale


__all__ = ["nvfp4_gemm_swiglu_nvfp4_quant"]
