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

"""Dense W8A16 Marlin: E4M3 weights, E8M0 group-32 scales, BF16 activations."""

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import ScaleFormat, format_signature, tensor_format
from tokenspeed_kernel.thirdparty.cuda.marlin import gptq_marlin_repack
from tokenspeed_kernel.thirdparty.cuda.marlin_mxfp8 import load_marlin_mxfp8_module

_BLOCK = 32
_SUPPORTED_SMS = frozenset({90, 100, 103, 107})
_SCALE_PERM = tuple(i + 8 * j for i in range(8) for j in range(8))
_BIAS_PERM = tuple(2 * i + j for i in range(4) for j in (0, 1, 8, 9, 16, 17, 24, 25))


def marlin_mxfp8_padded_shape(n: int, k: int) -> tuple[int, int]:
    """Return the smallest valid Marlin (N, K) thread-tile extent."""
    if n <= 0 or k <= 0 or k % _BLOCK:
        raise ValueError(
            "MXFP8 requires positive N/K and a K partition divisible by 32"
        )
    candidates = (
        ((n + 63) // 64 * 64, (k + 127) // 128 * 128),
        ((n + 127) // 128 * 128, (k + 63) // 64 * 64),
    )
    return min(candidates, key=lambda nk: (nk[0] * nk[1], nk[0] + nk[1]))


def marlin_mxfp8_permute_scales(
    scales: torch.Tensor, *, padded_n: int, padded_k: int
) -> torch.Tensor:
    """Reorder raw E8M0 bytes into Marlin [K/32, N] storage without rounding.

    Padding uses scale 1 (encoding 127); its E4M3 weight is exactly zero.
    The caller validates the logical shape and rejects unsupported scale bytes.
    """
    n, groups = scales.shape
    if padded_n < n or padded_k // _BLOCK < groups or padded_n % 64 or padded_k % 64:
        raise ValueError("Invalid padded MXFP8 scale extent")
    raw = scales.view(torch.uint8).T.contiguous()
    padded = torch.full(
        (padded_k // _BLOCK, padded_n), 127, dtype=torch.uint8, device=scales.device
    )
    padded[:groups, :n].copy_(raw)
    perm = torch.tensor(_SCALE_PERM, dtype=torch.long, device=scales.device)
    result = padded.reshape(-1, 64).index_select(1, perm)
    interleave = torch.tensor((0, 2, 1, 3), dtype=torch.long, device=scales.device)
    return (
        result.reshape(-1, 4)
        .index_select(1, interleave)
        .reshape(padded_k // _BLOCK, padded_n)
        .contiguous()
    )


class MarlinMxfp8LinearKernel:
    """Prepared native dense Marlin weights; never dequantize to a BF16 GEMM.

    Args:
        weight: CUDA E4M3 [N, K] checkpoint weight, K divisible by 32.
        weight_scale: Colocated uint8/E8M0 [N, K/32] scale bytes.
        use_atomic_add: Explicit reduction policy; false is reproducible for a
            fixed launch. True deliberately permits atomic reduction ordering.
        use_fp32_reduce: Explicit split-K accumulation policy.

    Preparation owns the packed weights and warms the native module. Each apply
    owns its locks and reduction scratch, so concurrent streams/graphs do not
    alias mutable workspace. M stays a runtime argument, not a JIT cache key.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        *,
        use_atomic_add: bool,
        use_fp32_reduce: bool,
    ) -> None:
        if not isinstance(use_atomic_add, bool) or not isinstance(
            use_fp32_reduce, bool
        ):
            raise TypeError("Marlin reduction policies must be bool")
        if weight.dtype != torch.float8_e4m3fn:
            raise TypeError("Dense Marlin MXFP8 requires E4M3 checkpoint weights")
        if weight_scale.dtype not in (torch.uint8, torch.float8_e8m0fnu):
            raise TypeError("Dense Marlin MXFP8 requires raw E8M0 scale bytes")
        if weight.ndim != 2:
            raise ValueError("Dense Marlin MXFP8 weight must have shape [N, K]")
        self.size_n, self.size_k = weight.shape
        self.padded_n, self.padded_k = marlin_mxfp8_padded_shape(
            self.size_n, self.size_k
        )
        if weight_scale.shape != (self.size_n, self.size_k // _BLOCK):
            raise ValueError("MXFP8 weight_scale must have shape [N, K/32]")
        if not weight.is_cuda or weight_scale.device != weight.device:
            raise ValueError("MXFP8 weights and scales must share a CUDA device")
        self.device = weight.device
        self.use_atomic_add = use_atomic_add
        self.use_fp32_reduce = use_fp32_reduce
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Prepare Marlin weights before graph capture")
            major, minor = torch.cuda.get_device_capability(self.device)
            if major * 10 + minor not in _SUPPORTED_SMS:
                raise ValueError(
                    "Dense Marlin MXFP8 requires supported NVIDIA SM90/SM10x"
                )
            raw_scale = weight_scale.contiguous().view(torch.uint8)
            # Public Marlin assumes the 2**-127 scale is absent (it flushes it).
            # Reject it explicitly rather than silently changing a real weight.
            if bool(((raw_scale == 0) | (raw_scale == 255)).any()):
                raise ValueError(
                    "Marlin MXFP8 cannot represent scale encoding 0 or NaN 255"
                )
            if not bool(torch.isfinite(weight.float()).all()):
                raise ValueError("Marlin MXFP8 weight contains NaN or infinity")
            raw_weight = weight.contiguous()
            if raw_weight.data_ptr() % 4:
                raw_weight = raw_weight.clone()
            packed = raw_weight.view(torch.int32).T.contiguous()
            packed = torch.nn.functional.pad(
                packed,
                (0, self.padded_n - self.size_n, 0, (self.padded_k - self.size_k) // 4),
            )
            self.weight = gptq_marlin_repack(
                packed,
                torch.empty(0, device=self.device, dtype=torch.int32),
                self.padded_k,
                self.padded_n,
                8,
            )
            self.weight_scale = marlin_mxfp8_permute_scales(
                raw_scale, padded_n=self.padded_n, padded_k=self.padded_k
            )
            self.sm_count = torch.cuda.get_device_properties(
                self.device
            ).multi_processor_count
            self.bias_permutation = torch.tensor(
                _BIAS_PERM, device=self.device, dtype=torch.long
            )
            self.module = load_marlin_mxfp8_module()

    def apply(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        """Compute x @ weight.T (+ bias), preserving all leading input dimensions."""
        return _marlin_mm_mxfp8_w8a16(x, self, bias=bias)

    def dequantize_for_absorption(self) -> torch.Tensor:
        """Materialize MLA's absorbed K/V projections, not a dense-GEMM fallback."""
        # This happens once during loading. Invert the byte layout directly so
        # absorption does not retain raw weights or depend on an MoE backend.
        packed = self.weight.view(torch.uint8).reshape(
            self.padded_k // 16, self.padded_n // 64, 8, 4, 4, 2, 2, 2
        )
        values = (
            packed.permute(1, 4, 5, 2, 0, 7, 3, 6)
            .contiguous()
            .view(torch.float8_e4m3fn)
            .reshape(self.padded_n, self.padded_k)
        )
        raw_scale = self.weight_scale.reshape(-1, 4)[:, [0, 2, 1, 3]]
        undo = torch.argsort(
            torch.tensor(_SCALE_PERM, dtype=torch.long, device=self.device)
        )
        raw_scale = (
            raw_scale.reshape(-1, 64)
            .index_select(1, undo)
            .reshape(self.padded_k // _BLOCK, self.padded_n)
            .T
        )
        values = values[: self.size_n, : self.size_k].to(torch.bfloat16)
        scale = torch.exp2(
            raw_scale[: self.size_n, : self.size_k // _BLOCK].float() - 127
        ).to(torch.bfloat16)
        return (
            values.reshape(self.size_n, self.size_k // _BLOCK, _BLOCK)
            * scale.unsqueeze(-1)
        ).reshape(self.size_n, self.size_k)


@register_kernel(
    "gemm",
    "mxfp8_w8a16",
    name="marlin_mxfp8_linear",
    solution="marlin",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(9, 0),
        max_arch_version=ArchVersion(10, 7),
        vendors=frozenset({"nvidia"}),
    ),
    signatures=frozenset(
        {
            format_signature(
                a=tensor_format("dense", torch.bfloat16),
                b=tensor_format(
                    "mxfp8",
                    torch.float8_e4m3fn,
                    scale=ScaleFormat(
                        storage_dtype=torch.uint8,
                        granularity="block",
                        block_shape=(1, 32),
                    ),
                ),
            )
        }
    ),
    traits={
        "n_min": frozenset({1}),
        "k_align": frozenset({32}),
        "out_dtype": frozenset({torch.bfloat16}),
        "use_atomic_add": frozenset({False, True}),
        "use_fp32_reduce": frozenset({False, True}),
    },
    priority=Priority.PERFORMANT,
)
def marlin_mxfp8_linear(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    input_dtype: torch.dtype,
    use_atomic_add: bool,
    use_fp32_reduce: bool,
) -> MarlinMxfp8LinearKernel:
    """Prepare raw E4M3/E8M0 weights for the registered BF16 W8A16 backend."""
    if input_dtype != torch.bfloat16:
        raise TypeError("Marlin MXFP8 requires BF16 activations and output")
    return MarlinMxfp8LinearKernel(
        weight,
        weight_scale,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=use_fp32_reduce,
    )


def _marlin_mm_mxfp8_w8a16(
    x: torch.Tensor,
    kernel: MarlinMxfp8LinearKernel,
    *,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the prepared W8A16 kernel; no activation quantization or BF16 fallback."""
    if x.ndim < 2 or x.shape[-1] != kernel.size_k:
        raise ValueError(
            "Marlin input must have shape [..., K] with at least two dimensions"
        )
    if x.dtype != torch.bfloat16 or x.device != kernel.device:
        raise TypeError("Marlin input must be BF16 on the weight device")
    if bias is not None and (
        bias.shape != (kernel.size_n,)
        or bias.dtype != x.dtype
        or bias.device != x.device
    ):
        raise ValueError("Marlin bias must be colocated BF16 [N]")
    output_shape = (*x.shape[:-1], kernel.size_n)
    with torch.cuda.device(kernel.device):
        a = x.reshape(-1, kernel.size_k).contiguous()
        # A contiguous offset view can still violate the native vector load
        # alignment; materialize it without changing the arithmetic.
        if a.data_ptr() % 16:
            a = a.clone()
        m = a.shape[0]
        if m == 0:
            return torch.empty(output_shape, dtype=x.dtype, device=x.device)
        if (
            max(
                m * kernel.padded_k,
                m * kernel.padded_n,
                kernel.padded_n * kernel.padded_k,
            )
            > 2**31 - 1
        ):
            raise ValueError("Marlin matrix extent exceeds the INT32 indexing contract")
        if kernel.padded_k != kernel.size_k:
            a = torch.nn.functional.pad(a, (0, kernel.padded_k - kernel.size_k))
        if bias is None:
            b_bias = torch.empty(0, dtype=x.dtype, device=x.device)
        else:
            b_bias = torch.nn.functional.pad(bias, (0, kernel.padded_n - kernel.size_n))
            b_bias = (
                b_bias.reshape(-1, 32)
                .index_select(1, kernel.bias_permutation)
                .reshape(-1)
                .contiguous()
            )
        output = torch.empty((m, kernel.padded_n), dtype=x.dtype, device=x.device)
        locks = torch.zeros(kernel.sm_count, dtype=torch.int32, device=x.device)
        tmp_size = (
            kernel.sm_count * min((m + 15) // 16 * 16, 64) * 256
            if kernel.use_fp32_reduce
            else 0
        )
        scratch = torch.empty(tmp_size, dtype=torch.float32, device=x.device)
        kernel.module.marlin_mxfp8_gemm(
            output,
            a,
            kernel.weight,
            kernel.weight_scale,
            b_bias,
            locks,
            scratch,
            kernel.use_atomic_add,
            kernel.use_fp32_reduce,
        )
        return output[:, : kernel.size_n].contiguous().reshape(output_shape)
