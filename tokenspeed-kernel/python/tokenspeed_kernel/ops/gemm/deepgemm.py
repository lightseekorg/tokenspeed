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

from functools import cache

import torch
from tokenspeed_kernel._optional import has_optional_package
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.tuning import is_warming_up
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    Platform,
    pdl_enabled,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import (
    ScaleFormat,
    dense_tensor_format,
    format_signature,
    format_signatures,
)
from tokenspeed_kernel.thirdparty.deepgemm import (
    block_fp8_mm,
    fp8_mm,
    fp8_weight_scales,
    grouped_fp8_mm,
    grouped_fp8_weight_scales,
    prepare_block_fp8_weight,
)

try:
    from flashinfer import autotuner
except ImportError:
    autotuner = None


_SPLIT_K_MAX_ROWS = 32
_TUNING_OP = "tokenspeed_deepgemm_fp8_split_k"
_ShapeKey = tuple[torch.device, int, int, bool, int, int]
# Only dispatch tactics that were warmed in this process.
_prepared_tactics: dict[_ShapeKey, tuple[int, ...]] = {}


@cache
def _valid_tactics(k: int) -> tuple[int, ...]:
    return (1,) + tuple(splits for splits in (2, 3, 4, 6) if k % (splits * 512) == 0)


@triton.jit(do_not_specialize=["elements"])
def _reduce_split_k_kernel(
    partial,
    out,
    elements,
    N: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    offset = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    row, col = offset // N, offset % N
    first = partial + row * (NUM_SPLITS * N) + col
    value = tl.load(first, offset < elements, other=0)
    for split in tl.static_range(1, NUM_SPLITS):
        value += tl.load(first + split * N, offset < elements, other=0)
    tl.store(out + offset, value, offset < elements)


def block_fp8_mm_split_k(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    *,
    num_splits: int,
    enable_pdl: bool,
) -> torch.Tensor:
    """Use views of packed operands, then reduce FP32 partials to BF16 out."""
    import deep_gemm

    m, k = a.shape
    n = b.shape[0]
    split_k = k // num_splits
    scale_words = split_k // 512
    # Preserve the MN-major strides, including the singleton M=1 case.
    sa = a_scale.as_strided(
        (m, num_splits, scale_words),
        (1, scale_words * a_scale.stride(1), a_scale.stride(1)),
    )
    sb = b_scale.as_strided(
        (num_splits, n, scale_words),
        (scale_words * b_scale.stride(1), 1, b_scale.stride(1)),
    )
    partial = torch.empty((m, num_splits, n), dtype=torch.float32, device=a.device)
    deep_gemm.set_pdl(enable_pdl)
    deep_gemm.fp8_einsum(
        "bhr,hdr->bhd",
        (a.view(m, num_splits, split_k), sa),
        (b.view(n, num_splits, split_k).transpose(0, 1), sb),
        partial,
        recipe=(1, 1, 128),
    )
    _reduce_split_k_kernel[(triton.cdiv(m * n, 256),)](
        partial,
        out,
        m * n,
        N=n,
        NUM_SPLITS=num_splits,
        ENABLE_PDL=enable_pdl,
        BLOCK=256,
        num_warps=4,
        **({"launch_pdl": True} if enable_pdl else {}),
    )
    return out


if autotuner is not None:
    # Tune exact input shapes so packed scale strides never have to be resized
    # by a dynamic-M profile.
    _TUNING = autotuner.TuningConfig(
        use_cold_l2_cache=True, use_cuda_graph=True, cuda_graph_profile_replays=1
    )

    class _SplitKRunner(autotuner.TunableRunner):
        def __init__(self, *, enable_pdl: bool, extras: tuple):
            self.enable_pdl = enable_pdl
            self.extras = extras

        def get_valid_tactics(self, inputs, profile):
            return list(_valid_tactics(inputs[0].shape[1]))

        def get_cache_key_extras(self, inputs):
            return self.extras

        def forward(self, inputs, tactic=-1, do_preparation=False):
            if tactic in (-1, 1):
                return block_fp8_mm(*inputs, enable_pdl=self.enable_pdl)
            return block_fp8_mm_split_k(
                *inputs, num_splits=tactic, enable_pdl=self.enable_pdl
            )


def _warm_tactics(b: torch.Tensor, b_scale: torch.Tensor, key: _ShapeKey) -> None:
    device, n, k, enable_pdl, num_sms, tc_util = key
    runner = _SplitKRunner(
        enable_pdl=enable_pdl,
        extras=(
            "packed_group128_fp32_reduce_v1",
            torch.cuda.get_device_name(device),
            torch.cuda.get_device_capability(device),
            enable_pdl,
            num_sms,
            tc_util,
        ),
    )
    tuner = autotuner.AutoTuner.get()
    a = b.new_zeros((_SPLIT_K_MAX_ROWS, k))
    out = torch.empty((_SPLIT_K_MAX_ROWS, n), dtype=torch.bfloat16, device=device)
    words = k // 512
    storage = b_scale.new_full((words * _SPLIT_K_MAX_ROWS,), 0x7F7F7F7F)
    candidates = _valid_tactics(k)
    tactics = []
    for rows in range(1, _SPLIT_K_MAX_ROWS + 1):
        padded = (rows + 3) // 4 * 4
        scales = storage[: words * padded].view(words, padded).t()[:rows]
        inputs = [a[:rows], b, scales, b_scale, out[:rows]]
        _, tactic = tuner.choose_one(_TUNING_OP, [runner], _TUNING, inputs)
        tactic = tactic if tactic in candidates else 1
        runner(inputs, tactic=tactic)
        tactics.append(tactic)
    _prepared_tactics[key] = tuple(tactics)


def block_fp8_mm_tuned(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    *,
    enable_pdl: bool,
) -> torch.Tensor:
    """Profile at startup and run the prepared tactic for the exact row count."""
    m, k = a.shape
    warming_up = is_warming_up()
    if (
        autotuner is not None
        and (warming_up or m <= _SPLIT_K_MAX_ROWS)
        and len(_valid_tactics(k)) > 1
    ):
        import deep_gemm

        key = (
            a.device,
            b.shape[0],
            k,
            enable_pdl,
            deep_gemm.get_num_sms(),
            deep_gemm.get_tc_util(),
        )
        if (
            warming_up
            and key not in _prepared_tactics
            and not torch.cuda.is_current_stream_capturing()
        ):
            _warm_tactics(b, b_scale, key)
        tactics = _prepared_tactics.get(key)
        if tactics is not None and m <= _SPLIT_K_MAX_ROWS:
            num_splits = tactics[m - 1]
            if num_splits > 1:
                return block_fp8_mm_split_k(
                    a,
                    b,
                    a_scale,
                    b_scale,
                    out,
                    num_splits=num_splits,
                    enable_pdl=enable_pdl,
                )
    return block_fp8_mm(a, b, a_scale, b_scale, out, enable_pdl=enable_pdl)


def prepare_deepgemm_fp8_weight(
    weight: torch.Tensor, scales: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare standard block-FP8 weights for explicit DeepGEMM serving.

    Args:
        weight: E4M3 [N,K] weights, N%64=0 and K%128=0.
        scales: Canonical FP32 [ceil(N/128),K/128] block scales.

    Returns:
        Hopper's unchanged canonical tensors, or Blackwell's requantized
        weights and native packed UE8M0 scales. Call once after loading.
    """
    platform = Platform.get()
    if not (
        platform.is_hopper
        or platform.arch_version in (ArchVersion(10, 0), ArchVersion(10, 3))
    ):
        raise ValueError("DeepGEMM dense FP8 requires SM90, SM100 or SM103")
    n, k = weight.shape
    if n % 64 or k % 128:
        raise ValueError("DeepGEMM weights require N%64=0 and K%128=0")
    if platform.is_hopper:
        return weight, scales
    return prepare_block_fp8_weight(weight, scales)


@register_kernel(
    "gemm",
    "mm",
    name="deepgemm_mm_fp8_blockscale",
    solution="deepgemm",
    capability=CapabilityRequirement(
        vendors=frozenset({"nvidia"}),
        min_arch_version=ArchVersion(9, 0),
        max_arch_version=ArchVersion(10, 3),
    ),
    signatures=frozenset().union(
        *(
            format_signatures(
                ("a", "b"),
                "mxfp8",
                {torch.float8_e4m3fn},
                scale=ScaleFormat(
                    storage_dtype=dtype, granularity="block", block_shape=(128, 128)
                ),
            )
            for dtype in (torch.float32, torch.int32)
        )
    ),
    traits={
        "n_align": frozenset({64}),
        "k_align": frozenset({128}),
        "a_contiguous": frozenset({True}),
        "b_contiguous": frozenset({True}),
        "block_scale_layout": frozenset({"canonical", "canonical_blackwell"}),
        "out_contiguous": frozenset({True}),
        "out_dtype": frozenset({torch.bfloat16}),
    },
    weight_preprocessor=prepare_deepgemm_fp8_weight,
    # Below the existing Triton block-FP8 leaf: auto keeps its current choice.
    priority=Priority.PERFORMANT,
)
def deepgemm_mm_fp8_blockscale(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scales: torch.Tensor | None,
    B_scales: torch.Tensor | None,
    out_dtype: torch.dtype,
    *,
    alpha: torch.Tensor | None = None,
    block_size: list[int] | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Explicit E4M3 GEMM with Hopper FP32 or Blackwell packed UE8M0 scales.

    Activations use 1x128 scales and weights use 128x128 scales. Output is
    BF16; a supplied contiguous output buffer is written in place. Hopper
    converts scale layouts per call; Blackwell consumes prepared packed scales.
    Unsupported inputs are rejected even for a forced kernel override.
    """
    platform = Platform.get()
    packed = platform.arch_version in (ArchVersion(10, 0), ArchVersion(10, 3))
    if not (platform.is_hopper or packed):
        raise ValueError("DeepGEMM block FP8 requires SM90, SM100 or SM103")
    if alpha is not None or block_size is None or tuple(block_size) != (128, 128):
        raise ValueError("DeepGEMM requires 128x128 block scales and alpha=None")
    if (
        out_dtype != torch.bfloat16
        or A.dtype != torch.float8_e4m3fn
        or B.dtype != A.dtype
    ):
        raise ValueError("DeepGEMM requires E4M3 inputs and BF16 output")
    if A.ndim != 2 or B.ndim != 2 or not A.is_contiguous() or not B.is_contiguous():
        raise ValueError("DeepGEMM requires contiguous 2D A[M,K] and B[N,K]")
    m, k = A.shape
    n, kb = B.shape
    if n < 1 or k < 1 or k != kb or n % 64 or k % 128:
        raise ValueError(
            "DeepGEMM requires positive N and K, matching K, N%64=0 and K%128=0"
        )
    if not A.is_cuda or B.device != A.device:
        raise ValueError("DeepGEMM operands must share a CUDA device")
    for scale, rows, shape in (
        (A_scales, m, (m, (k // 128 + 3) // 4) if packed else (m, k // 128)),
        (
            B_scales,
            n,
            (n, (k // 128 + 3) // 4) if packed else ((n + 127) // 128, k // 128),
        ),
    ):
        if (
            scale is None
            or scale.dtype != (torch.int32 if packed else torch.float32)
            or tuple(scale.shape) != shape
            or scale.device != A.device
        ):
            raise ValueError(
                "DeepGEMM requires matching canonical FP32 or packed UE8M0 scales"
            )
        if (
            packed
            and m
            and (scale.stride(0) != 1 or scale.stride(1) != (rows + 3) // 4 * 4)
        ):
            raise ValueError(
                "DeepGEMM packed scales require TMA-aligned MN-major strides"
            )
    if not packed and not B_scales.is_contiguous():
        raise ValueError("DeepGEMM requires contiguous weight scales")
    if out is None:
        out = torch.empty((m, n), device=A.device, dtype=out_dtype)
    elif (
        tuple(out.shape) != (m, n)
        or out.dtype != out_dtype
        or out.device != A.device
        or not out.is_contiguous()
    ):
        raise ValueError(
            "DeepGEMM out must be contiguous BF16 [M,N] on the operand device"
        )
    if not m:
        return out
    enable_pdl = pdl_enabled()
    if packed and n <= 4096 and k >= 4096:
        return block_fp8_mm_tuned(A, B, A_scales, B_scales, out, enable_pdl=enable_pdl)
    return block_fp8_mm(A, B, A_scales, B_scales, out, enable_pdl=enable_pdl)


if has_optional_package("deep_gemm"):
    # SM100's MMA takes the per-32 UE8M0 scales natively.
    _SM100 = CapabilityRequirement(
        vendors=frozenset({"nvidia"}),
        min_arch_version=ArchVersion(10, 0),
        max_arch_version=ArchVersion(10, 3),
    )
    _FP8_OPERANDS = frozenset(
        {
            format_signature(
                x=dense_tensor_format(torch.float8_e4m3fn),
                weight=dense_tensor_format(torch.float8_e4m3fn),
            )
        }
    )
    _TRAITS = {"aligned": frozenset({True}), "is_cuda": frozenset({True})}

    @register_kernel(
        "gemm",
        "grouped_fp8_projection",
        name="deepgemm_grouped_fp8_projection",
        solution="deepgemm",
        capability=_SM100,
        signatures=_FP8_OPERANDS,
        traits=_TRAITS,
        priority=Priority.PERFORMANT,
        weight_preprocessor=grouped_fp8_weight_scales,
    )
    def deepgemm_grouped_fp8_projection(
        x: torch.Tensor,
        x_scales: torch.Tensor,
        weight: torch.Tensor,
        weight_scales: torch.Tensor,
        out: torch.Tensor,
    ) -> torch.Tensor:
        """DeepGEMM grouped E4M3 projection; see ``grouped_fp8_projection``."""
        return grouped_fp8_mm(x, x_scales, weight, weight_scales, out, None)

    @register_kernel(
        "gemm",
        "grouped_fp8_projection_fp8",
        name="deepgemm_grouped_fp8_projection_fp8",
        solution="deepgemm",
        capability=_SM100,
        signatures=_FP8_OPERANDS,
        traits=_TRAITS,
        priority=Priority.PERFORMANT,
        weight_preprocessor=grouped_fp8_weight_scales,
    )
    def deepgemm_grouped_fp8_projection_fp8(
        x: torch.Tensor,
        x_scales: torch.Tensor,
        weight: torch.Tensor,
        weight_scales: torch.Tensor,
        out: torch.Tensor,
        out_scales: torch.Tensor,
    ) -> torch.Tensor:
        """DeepGEMM grouped E4M3 projection with E4M3 output; see
        ``grouped_fp8_projection_quantize_fp8``."""
        return grouped_fp8_mm(x, x_scales, weight, weight_scales, out, out_scales)

    @register_kernel(
        "gemm",
        "fp8_projection",
        name="deepgemm_fp8_projection",
        solution="deepgemm",
        capability=_SM100,
        signatures=_FP8_OPERANDS,
        traits=_TRAITS,
        priority=Priority.PERFORMANT,
        weight_preprocessor=fp8_weight_scales,
    )
    def deepgemm_fp8_projection(
        x: torch.Tensor,
        x_scales: torch.Tensor,
        weight: torch.Tensor,
        weight_scales: torch.Tensor,
        out: torch.Tensor,
    ) -> torch.Tensor:
        """DeepGEMM E4M3 projection; see ``fp8_projection``."""
        return fp8_mm(x, x_scales, weight, weight_scales, out)
