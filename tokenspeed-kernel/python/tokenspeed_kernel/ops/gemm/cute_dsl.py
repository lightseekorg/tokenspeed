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

import functools
import itertools
from typing import Optional, Tuple

from tokenspeed_kernel.platform import current_platform, pdl_enabled
from tokenspeed_kernel.registry import error_fn

platform = current_platform()

nvfp4_gemm_swiglu_nvfp4_quant = error_fn

if platform.is_nvidia:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import torch
    from flashinfer.autotuner import (
        AutoTuner,
        ConstraintSpec,
        DynamicTensorSpec,
        TunableRunner,
        TuningConfig,
        autotuner_initializer_empty,
    )
    from flashinfer.cute_dsl.utils import (
        get_cutlass_dtype,
        get_max_active_clusters,
        make_ptr,
    )
    from flashinfer.fused_moe.utils import (
        get_hybrid_num_tokens_buckets,
        map_to_hybrid_bucket_uncapped,
    )
    from flashinfer.utils import get_compute_capability
    from tokenspeed_kernel.thirdparty.cute_dsl.nvfp4_gemm_swiglu_nvfp4_quant import (
        Sm100BlockScaledPersistentDenseGemmKernel,
    )

    def _round_up(value: int, multiple: int) -> int:
        return (value + multiple - 1) // multiple * multiple

    def _init_packed_fp4(shapes, dtype, device):
        """Autotuner initializer for FP4-packed operands."""
        return torch.randint(0, 256, shapes, dtype=dtype, device=device)

    class _Nvfp4GemmSwigluNvfp4QuantRunner(TunableRunner):
        """Autotuner adapter over the fused NVFP4 FC1 kernel.

        A tactic is a ``(mma_tiler_mn, cluster_shape_mn)`` pair drawn from
        :attr:`TACTICS`; ``-1`` is the mandatory always-safe fallback, a
        ``(128, 128)`` tile on a 1x1 cluster. Instances are
        stateless with respect to the problem shape -- M, N and K are read back
        from ``inputs`` -- so one instance per compile-option set is enough,
        which is what keeps the autotuner's per-runner caches stable.
        """

        # tile_n=64 is excluded: can_implement() accepts it but the vendored
        # kernel then writes a wrong result.
        MMA_TILER_MN_CANDIDATES = ((128, 128), (256, 128), (128, 256), (256, 256))
        CLUSTER_N_CANDIDATES = (1, 2, 4)

        # cluster_m is derived rather than swept: can_implement() requires it to
        # be 1 for a 1-CTA tile and even for a 2-CTA one, and cluster_m=4 never
        # won a cell in the standalone sweep.
        TACTICS: Tuple[Tuple[Tuple[int, int], Tuple[int, int]], ...] = tuple(
            (mma_tiler_mn, (mma_tiler_mn[0] // 128, cluster_n))
            for mma_tiler_mn, cluster_n in itertools.product(
                MMA_TILER_MN_CANDIDATES, CLUSTER_N_CANDIDATES
            )
        )

        _kernel_cache: dict[tuple, object] = {}

        def __init__(
            self,
            ab_dtype: str,
            sf_dtype: str,
            c_dtype: str,
            sf_vec_size: int,
            use_prefetch: bool,
            prefetch_dist: int,
            vectorized_f32: bool,
            enable_pdl: bool,
        ):
            self.ab_dtype = ab_dtype
            self.sf_dtype = sf_dtype
            self.c_dtype = c_dtype
            self.sf_vec_size = sf_vec_size
            self.use_prefetch = use_prefetch
            self.prefetch_dist = prefetch_dist
            self.vectorized_f32 = vectorized_f32
            self.enable_pdl = enable_pdl

        @classmethod
        @functools.lru_cache(maxsize=None)
        def get(
            cls,
            ab_dtype: str,
            sf_dtype: str,
            c_dtype: str,
            sf_vec_size: int,
            use_prefetch: bool,
            prefetch_dist: int,
            vectorized_f32: bool,
            enable_pdl: bool,
        ) -> "_Nvfp4GemmSwigluNvfp4QuantRunner":
            """Memoized runner for one set of compile-time options."""
            return cls(
                ab_dtype,
                sf_dtype,
                c_dtype,
                sf_vec_size,
                use_prefetch,
                prefetch_dist,
                vectorized_f32,
                enable_pdl,
            )

        def _key(self) -> tuple:
            return (
                self.ab_dtype,
                self.sf_dtype,
                self.c_dtype,
                self.sf_vec_size,
                self.use_prefetch,
                self.prefetch_dist,
                self.vectorized_f32,
                self.enable_pdl,
            )

        def __hash__(self) -> int:
            return hash(self._key())

        def __eq__(self, other: object) -> bool:
            return (
                isinstance(other, _Nvfp4GemmSwigluNvfp4QuantRunner)
                and self._key() == other._key()
            )

        def get_cache_key_extras(self, inputs) -> tuple:
            return self._key()

        @staticmethod
        def _mnk(inputs) -> Tuple[int, int, int]:
            a, b = inputs[0], inputs[2]
            return a.shape[0], b.shape[0], a.shape[1] * 2

        def _can_implement(
            self,
            mma_tiler_mn: Tuple[int, int],
            cluster_shape_mn: Tuple[int, int],
            m: int,
            n: int,
            k: int,
        ) -> bool:
            return Sm100BlockScaledPersistentDenseGemmKernel.can_implement(
                get_cutlass_dtype(self.ab_dtype),
                get_cutlass_dtype(self.sf_dtype),
                self.sf_vec_size,
                get_cutlass_dtype(self.c_dtype),
                mma_tiler_mn,
                cluster_shape_mn,
                m,
                n,
                k,
                1,
                a_major="k",
                b_major="k",
                c_major="n",
            )

        def get_valid_tactics(self, inputs, profile) -> list:
            m, n, k = self._mnk(inputs)
            return [t for t in self.TACTICS if self._can_implement(*t, m, n, k)]

        def _compile(
            self,
            ptrs: dict,
            m: int,
            n: int,
            k: int,
            mma_tiler_mn: Tuple[int, int],
            cluster_shape_mn: Tuple[int, int],
            stream,
        ):
            """Compile-and-cache the kernel; M stays dynamic so it is not keyed."""
            cache_key = (*self._key(), n, k, mma_tiler_mn, cluster_shape_mn)
            if cache_key not in self._kernel_cache:
                gemm = Sm100BlockScaledPersistentDenseGemmKernel(
                    sf_vec_size=self.sf_vec_size,
                    mma_tiler_mn=mma_tiler_mn,
                    cluster_shape_mn=cluster_shape_mn,
                    use_prefetch=self.use_prefetch,
                    prefetch_dist=self.prefetch_dist,
                    vectorized_f32=self.vectorized_f32,
                )
                self._kernel_cache[cache_key] = cute.compile(
                    gemm.wrapper,
                    *ptrs.values(),
                    m,
                    n,
                    k,
                    1,
                    scaling_vector_size=self.sf_vec_size,
                    max_active_clusters=get_max_active_clusters(
                        cluster_shape_mn[0] * cluster_shape_mn[1]
                    ),
                    stream=stream,
                    use_pdl=self.enable_pdl,
                )
            return self._kernel_cache[cache_key]

        def _make_ptrs(self, inputs) -> dict:
            a, a_scale, b, b_scale, alpha, output_global_scale, out, out_scale = inputs
            ab = get_cutlass_dtype(self.ab_dtype)
            sf = get_cutlass_dtype(self.sf_dtype)
            c = get_cutlass_dtype(self.c_dtype)
            gmem = cute.AddressSpace.gmem
            # Order matches Sm100BlockScaledPersistentDenseGemmKernel.wrapper.
            return {
                "a": make_ptr(ab, a.data_ptr(), gmem, assumed_align=32),
                "b": make_ptr(ab, b.data_ptr(), gmem, assumed_align=32),
                "a_sf": make_ptr(sf, a_scale.data_ptr(), gmem, assumed_align=16),
                "b_sf": make_ptr(sf, b_scale.data_ptr(), gmem, assumed_align=16),
                "c": make_ptr(c, out.data_ptr(), gmem, assumed_align=32),
                "c_sf": make_ptr(sf, out_scale.data_ptr(), gmem, assumed_align=16),
                "alpha": make_ptr(cutlass.Float32, alpha.data_ptr(), gmem),
                "norm_const": make_ptr(
                    cutlass.Float32, output_global_scale.data_ptr(), gmem
                ),
            }

        def forward(self, inputs, tactic=-1, do_preparation: bool = False):
            m, n, k = self._mnk(inputs)

            if tactic is not None and tactic != -1:
                mma_tiler_mn, cluster_shape_mn = tactic
            else:
                mma_tiler_mn, cluster_shape_mn = (128, 128), (1, 1)
                if not self._can_implement(mma_tiler_mn, cluster_shape_mn, m, n, k):
                    raise ValueError(
                        "Unsupported nvfp4_gemm_swiglu_nvfp4_quant configuration: "
                        f"shape=(M={m}, N={n}, K={k}), mma_tiler_mn={mma_tiler_mn}, "
                        f"cluster_shape_mn={cluster_shape_mn}"
                    )

            ptrs = self._make_ptrs(inputs)
            stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
            compiled_gemm = self._compile(
                ptrs, m, n, k, mma_tiler_mn, cluster_shape_mn, stream
            )
            compiled_gemm(*ptrs.values(), m, n, k, 1, stream=stream)
            return inputs[6], inputs[7]

        # Scale tensors are padded derivations of M, so they are constrained
        # rather than tuned; the autotuner's default initializer handles their
        # dtype. Cold L2 matches the conditions this GEMM meets in a decode
        # step, as in flashinfer's own CuteDSL tuning configs.
        TUNING_CONFIG = TuningConfig(
            dynamic_tensor_specs=(
                DynamicTensorSpec(
                    (0, 6),  # a, out
                    (0, 0),
                    get_hybrid_num_tokens_buckets,
                    map_to_hybrid_bucket_uncapped,
                ),
            ),
            constraint_specs=(
                ConstraintSpec(1, 0, lambda shapes: _round_up(shapes[0][0], 128)),
                ConstraintSpec(7, 0, lambda shapes: _round_up(shapes[0][0], 128)),
            ),
            tensor_initializers=(
                (0, _init_packed_fp4),
                (6, autotuner_initializer_empty),
            ),
            use_cold_l2_cache=True,
        )

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
        enable_pdl: bool | None = None,
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
                Uses the platform default when omitted.

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

        n_out = n // 2
        if n_out % sf_vec_size != 0:
            raise ValueError(
                f"Output N={n_out} must be divisible by sf_vec_size={sf_vec_size}"
            )
        scale_n_out = n_out // sf_vec_size
        padded_m = _round_up(m, 128)
        padded_scale_n = _round_up(scale_n_out, 4)

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

        enable_pdl = pdl_enabled() if enable_pdl is None else enable_pdl
        runner = _Nvfp4GemmSwigluNvfp4QuantRunner.get(
            ab_dtype,
            sf_dtype,
            c_dtype,
            sf_vec_size,
            use_prefetch,
            prefetch_dist,
            vectorized_f32,
            bool(enable_pdl),
        )
        inputs = [a, a_scale, b, b_scale, alpha, output_global_scale, out, out_scale]
        chosen, tactic = AutoTuner.get().choose_one(
            custom_op="nvfp4_gemm_swiglu_nvfp4_quant",
            runners=[runner],
            tuning_config=_Nvfp4GemmSwigluNvfp4QuantRunner.TUNING_CONFIG,
            inputs=inputs,
        )
        chosen(inputs=inputs, tactic=tactic)
        return out, out_scale


__all__ = ["nvfp4_gemm_swiglu_nvfp4_quant"]
