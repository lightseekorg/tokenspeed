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

"""BF16 decode GEMV dispatch and Triton row-CTA kernels.

Eligible small-M BF16 projections use FlashInfer joint runner/tactic selection.
The registry keeps the architecture-specific and portable fallbacks, while the
row-CTA implementation also provides the independent fused add3 epilogue.

BF16 activations against an FP32 weight give an FP32 result. Torch widens the
activations first. A layer that keeps the weight split from
:func:`decode_gemv_weight_split` passes it to :func:`decode_gemv`, and the
registry can then run the product on BF16 tensor cores
(``triton_bf16x3_gemm_fp32``). From 17 to 96 rows, for weights of up to 256
rows (a multiple of 4) with K a multiple of 512 up to 8192, the registry runs
it on CUDA cores from the FP32 weight instead, with or without the split
(``gluon_simt_gemm_fp32``).
"""

from __future__ import annotations

import functools

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.compile_monitor import is_serving
from tokenspeed_kernel.ops.gemm.flashinfer import (
    BF16_GEMM_MAX_M,
    autotune_bf16_gemm,
    flashinfer_bf16_gemm,
    flashinfer_joint_bf16_supported,
)
from tokenspeed_kernel.ops.gemm.triton_bf16x3 import (
    BF16X3_MIN_M,
    split_fp32_weight_bf16x3,
    triton_bf16x3_gemm_fp32,
)
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import KernelRegistry, Priority, register_kernel
from tokenspeed_kernel.selection import spec_matches_shape_traits, spec_matches_traits
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

__all__ = [
    "decode_gemv",
    "decode_gemv_weight_split",
    "triton_rowcta_gemv",
    "use_decode_gemv",
]


@triton.jit
def _rowcta_gemv_add3_kernel(
    x_ptr,
    w_ptr,
    a_ptr,
    c_ptr,
    out_ptr,
    K: tl.constexpr,
    BK: tl.constexpr,
):
    """Row dot-product with a fused two-addend epilogue:
    ``out[n] = a[n] + x . w[n] + c[n]`` (the MoE residual accumulate rides
    the up-projection store; a/c row strides support lane column slices)."""
    n = tl.program_id(0)
    acc = tl.zeros([BK], tl.float32)
    for kb in tl.static_range(0, K, BK):
        offs = kb + tl.arange(0, BK)
        mask = offs < K
        xv = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        wv = tl.load(w_ptr + n * K + offs, mask=mask, other=0.0).to(tl.float32)
        acc += wv * xv
    av = tl.load(a_ptr + n).to(tl.float32)
    cv = tl.load(c_ptr + n).to(tl.float32)
    tl.store(
        out_ptr + n,
        (av + tl.sum(acc) + cv).to(out_ptr.dtype.element_ty),
    )


@triton.jit
def _row_dot(x_ptr, w_ptr, K: tl.constexpr, BK: tl.constexpr):
    acc = tl.zeros([BK], tl.float32)
    for kb in tl.static_range(0, K, BK):
        offs = kb + tl.arange(0, BK)
        mask = offs < K
        xv = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        wv = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        acc += wv * xv
    return tl.sum(acc)


@triton.jit
def _rowcta_gemv_kernel(x_ptr, w_ptr, out_ptr, K: tl.constexpr, BK: tl.constexpr):
    n = tl.program_id(0)
    value = _row_dot(x_ptr, w_ptr + n * K, K, BK)
    tl.store(out_ptr + n, value.to(out_ptr.dtype.element_ty))


@triton.jit
def _grouped_rowcta_gemv_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    K: tl.constexpr,
    BK: tl.constexpr,
    X_GROUP_STRIDE: tl.constexpr,
    W_GROUP_STRIDE: tl.constexpr,
    W_ROW_STRIDE: tl.constexpr,
    OUT_GROUP_STRIDE: tl.constexpr,
):
    n = tl.program_id(0).to(tl.int64)
    group = tl.program_id(1).to(tl.int64)
    value = _row_dot(
        x_ptr + group * X_GROUP_STRIDE,
        w_ptr + group * W_GROUP_STRIDE + n * W_ROW_STRIDE,
        K,
        BK,
    )
    tl.store(out_ptr + group * OUT_GROUP_STRIDE + n, value.to(out_ptr.dtype.element_ty))


# Registry dispatch: rowcta owns M == 1 while torch handles other shapes.
_BF16_SIG = frozenset(
    {
        format_signature(
            x=dense_tensor_format(torch.bfloat16),
            weight=dense_tensor_format(torch.bfloat16),
        )
    }
)
_FP32_SIG = frozenset(
    {
        format_signature(
            x=dense_tensor_format(torch.float32),
            weight=dense_tensor_format(torch.float32),
        )
    }
)
# BF16 activations against an FP32 weight, with an FP32 result.
_BF16_FP32_SIG = frozenset(
    {
        format_signature(
            x=dense_tensor_format(torch.bfloat16),
            weight=dense_tensor_format(torch.float32),
        )
    }
)
# Registry signature of each served (x dtype, weight dtype) pair.
_SIGNATURES = {
    (torch.bfloat16, torch.bfloat16): next(iter(_BF16_SIG)),
    (torch.float32, torch.float32): next(iter(_FP32_SIG)),
    (torch.bfloat16, torch.float32): next(iter(_BF16_FP32_SIG)),
}


@register_kernel(
    "gemm",
    "decode_gemv",
    name="triton_rowcta_gemv",
    solution="triton",
    signatures=_BF16_SIG,
    traits={
        "m": frozenset({1}),
        "n_min": frozenset({128}),
        "k_min": frozenset({128}),
    },
    priority=Priority.SPECIALIZED,
)
def triton_rowcta_gemv(
    x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    """``x @ weight.T`` for ``M == 1`` decode activations.

    Args:
        x: ``[1, K]`` contiguous bf16 activation row.
        weight: ``[N, K]`` contiguous bf16 weight.
        out: optional ``[1, N]`` destination.

    Returns:
        ``[1, N]`` output in ``x``'s dtype.
    """
    assert x.shape[0] == 1 and x.stride(-1) == 1 and weight.stride(-1) == 1
    n, k = weight.shape
    if out is None:
        out = torch.empty(1, n, dtype=x.dtype, device=x.device)
    # BK=512 (4 fp32 accumulator regs/thread): standalone parity, and aux-stream kernels co-reside instead of stalling behind the GEMV wave.
    _rowcta_gemv_kernel[(n,)](
        x.view(-1),
        weight,
        out.view(-1),
        K=k,
        BK=512,
        num_warps=4,
    )
    return out


def _gfx1250_wmma_dense_problem(m: int, n: int, k: int) -> bool:
    from tokenspeed_kernel_amd.ops.gfx1250.gemm.fp16.mm import (
        use_gluon_wmma_dense_gfx1250,
    )

    return use_gluon_wmma_dense_gfx1250(m, k, n)


@register_kernel(
    "gemm",
    "decode_gemv",
    name="gluon_wmma_dense_gemv_gfx1250",
    solution="gluon",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(12, 5),
        max_arch_version=ArchVersion(12, 5),
        vendors=frozenset({"amd"}),
    ),
    signatures=_BF16_SIG,
    traits={
        "m": frozenset(range(2, 65)),
        "mnk_problem_filter": frozenset({_gfx1250_wmma_dense_problem}),
    },
    priority=Priority.SPECIALIZED,
)
def gluon_wmma_dense_gemv_gfx1250(
    x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    """``x @ weight.T`` for small-M decode activations on CDNA5.

    Args:
        x: ``[M, K]`` contiguous bf16 activation, K a multiple of 128.
        weight: ``[N, K]`` contiguous bf16 weight, N a multiple of 16.
        out: optional ``[M, N]`` destination.

    Returns:
        ``[M, N]`` output in ``x``'s dtype.
    """
    from tokenspeed_kernel_amd.ops.gfx1250.gemm.fp16.mm import (
        gluon_wmma_tdm_dense_gfx1250,
    )

    return gluon_wmma_tdm_dense_gfx1250(x, weight, out=out, split_k=None)


@register_kernel(
    "gemm",
    "decode_gemv",
    name="torch_decode_gemv",
    solution="torch",
    signatures=_BF16_SIG | _FP32_SIG | _BF16_FP32_SIG,
    traits={},
    priority=Priority.PORTABLE,
)
def torch_decode_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if x.dtype == torch.bfloat16 and weight.dtype == torch.float32:
        x = x.float()
    if out is not None:
        return torch.mm(x, weight.t(), out=out)
    return x @ weight.t()


@functools.lru_cache(maxsize=64)
def _select(
    m: int,
    n: int,
    k: int,
    on_cuda: bool,
    x_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    weight_split: bool,
):
    signature = _SIGNATURES.get((x_dtype, weight_dtype))
    if not on_cuda or signature is None:
        return torch_decode_gemv

    reg = KernelRegistry.get()
    # Honor each registered implementation's architecture gate and dtype
    # signature before its shape traits, including the specialized CDNA5
    # kernels. Kernels that take a split weight declare the weight_split trait.
    traits = {"m": m, "n": n, "k": k, "weight_split": weight_split}
    for spec in reg.get_for_operator(
        "gemm",
        "decode_gemv",
        platform=current_platform(),
        format_signature=signature,
    ):
        if spec_matches_traits(spec, traits) and spec_matches_shape_traits(
            spec, traits
        ):
            return reg.get_impl(spec.name)
    return torch_decode_gemv


def use_decode_gemv(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """Whether a dense projection should use the specialized decode entry.

    Args:
        x: Activation tensor shaped ``[M, K]``.
        weight: Projection weight shaped ``[N, K]``.

    Returns:
        True for eligible small-M FI inputs or a registered CDNA4/CDNA5
        kernel; False when the caller should retain its ordinary GEMM path.
    """
    if (
        flashinfer_joint_bf16_supported(x, weight, None)
        and x.shape[0] <= BF16_GEMM_MAX_M
    ):
        # FI compiles per row count; serving's eager rows take the caller's GEMM.
        return not is_serving()
    if (
        not x.is_cuda
        or x.ndim != 2
        or weight.ndim != 2
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or not x.is_contiguous()
        or not weight.is_contiguous()
    ):
        return False
    m, k = x.shape
    platform = current_platform()
    n = weight.shape[0]
    if platform.is_cdna4:
        return m >= 2 and _select(m, n, k, True, x.dtype, weight.dtype, False) is not (
            torch_decode_gemv
        )
    if not platform.is_cdna5 or k < 256:
        return False
    return _select(m, n, k, True, x.dtype, weight.dtype, False) is not torch_decode_gemv


def decode_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    weight_split: torch.Tensor | None = None,
) -> torch.Tensor:
    """``x @ weight.T`` through joint FI tuning or the ordinary registry fallback.

    FlashInfer owns runner/tactic selection on the supported BF16 range.
    The registry retains other architectures, unsupported input layouts and
    FP32 weights, which also take BF16 activations and return FP32.
    Noncontiguous inputs and other dtypes take Torch.

    Args:
        x: ``[M, K]`` activations.
        weight: ``[N, K]`` weight.
        out: optional ``[M, N]`` destination in the promoted dtype.
        weight_split: the weight's :func:`decode_gemv_weight_split`, for an
            FP32 weight; ``None`` keeps the kernels that read ``weight``.

    Returns:
        ``[M, N]`` output in the promoted dtype of ``x`` and ``weight``.
    """

    expected = (x.shape[0], weight.shape[0])
    if out is not None:
        if (
            tuple(out.shape) != expected
            or out.dtype != torch.promote_types(x.dtype, weight.dtype)
            or out.device != x.device
            or out.stride(-1) != 1
        ):
            raise ValueError(f"out must match x and have shape {expected}")
        if not out.is_contiguous():
            return torch_decode_gemv(x, weight, out)

    autotune_bf16_gemm(x, weight)
    if (
        flashinfer_joint_bf16_supported(x, weight, out)
        and x.shape[0] <= BF16_GEMM_MAX_M
    ):
        # Serving never compiles: FI keys runners on the row count, rowcta may be cold.
        if is_serving():
            return torch_decode_gemv(x, weight, out)
        return flashinfer_bf16_gemm(x, weight, out)
    if not x.is_contiguous() or not weight.is_contiguous():
        return torch_decode_gemv(x, weight, out)
    impl = _select(
        x.shape[0],
        weight.shape[0],
        weight.shape[1],
        x.is_cuda,
        x.dtype,
        weight.dtype,
        weight_split is not None,
    )
    if impl is triton_bf16x3_gemm_fp32:
        return impl(x, weight_split, out)
    return impl(x, weight, out)


def decode_gemv_weight_split(weight: torch.Tensor) -> torch.Tensor | None:
    """Split an FP32 weight for the tensor-core path of :func:`decode_gemv`.

    The layer that owns the weight calls this once after loading it and
    passes the result as ``decode_gemv(..., weight_split=...)``. The split
    takes 1.5x the FP32 weight's memory, so it is only made where the
    registration of ``triton_bf16x3_gemm_fp32`` takes this weight on this
    device. The kernel the registry prefers at one row count does not decide
    it: a kernel registered ahead of the split kernel for some row counts
    serves those rows from ``weight``, and the split serves the others.

    Args:
        weight: ``[N, K]`` weight.

    Returns:
        ``[3, N, K]`` BF16 pieces, or ``None`` when the split kernel does not
        take ``weight`` on this device.
    """
    if not weight.is_cuda or weight.ndim != 2 or not weight.is_contiguous():
        return None
    signature = _SIGNATURES.get((torch.bfloat16, weight.dtype))
    if signature is None:
        return None
    n, k = weight.shape
    # _select()'s filters on the split kernel's own registration, at its
    # first row count.
    traits = {"m": BF16X3_MIN_M, "n": n, "k": k, "weight_split": True}
    reg = KernelRegistry.get()
    if not any(
        reg.get_impl(spec.name) is triton_bf16x3_gemm_fp32
        and spec_matches_traits(spec, traits)
        and spec_matches_shape_traits(spec, traits)
        for spec in reg.get_for_operator(
            "gemm",
            "decode_gemv",
            platform=current_platform(),
            format_signature=signature,
        )
    ):
        return None
    return split_fp32_weight_bf16x3(weight)


def rowcta_gemv_add3(
    x: torch.Tensor,
    weight: torch.Tensor,
    a: torch.Tensor,
    c: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """``a + x @ weight.T + c`` for ``M == 1`` (fused MoE residual epilogue).

    Args:
        x: ``[1, K]`` bf16 latent row; weight: ``[N, K]``.
        a/c: ``[1, N]`` addends (``c`` may be a wider-lane column slice --
            only unit inner stride is required).

    Returns:
        ``[1, N]`` prefix row.
    """
    assert x.shape[0] == 1 and a.shape == (1, weight.shape[0])
    assert a.stride(1) == 1 and c.stride(1) == 1 and c.shape[1] == weight.shape[0]
    n, k = weight.shape
    if out is None:
        out = torch.empty(1, n, dtype=x.dtype, device=x.device)
    _rowcta_gemv_add3_kernel[(n,)](
        x.view(-1),
        weight,
        a,
        c,
        out,
        K=k,
        BK=512,
        num_warps=4,
    )
    return out


@register_kernel(
    "gemm",
    "grouped_bf16_projection",
    name="grouped_bf16_projection_rowcta",
    solution="triton",
    capability=CapabilityRequirement(
        vendors=frozenset({"nvidia"}), min_arch_version=ArchVersion(9, 0)
    ),
    signatures=frozenset(
        {
            format_signature(
                x=dense_tensor_format(torch.bfloat16),
                weight=dense_tensor_format(torch.bfloat16),
            )
        }
    ),
    traits={
        "batch": frozenset({2}),
        "m": frozenset({1}),
        "n": frozenset({1024}),
        "k": frozenset({4096}),
        "a_inner_stride_one": frozenset({True}),
        "b_inner_stride_one": frozenset({True}),
        "is_cuda": frozenset({True}),
    },
    priority=Priority.SPECIALIZED,
)
def grouped_bf16_projection_rowcta(
    x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor | None
) -> torch.Tensor:
    """Single-token grouped projection; the public API validates the layout."""
    groups, rows, dim = weight.shape
    if out is None:
        out = torch.empty((1, groups, rows), dtype=x.dtype, device=x.device)
    _grouped_rowcta_gemv_kernel[(rows, groups)](
        x,
        weight,
        out,
        K=dim,
        BK=4096,
        X_GROUP_STRIDE=x.stride(1),
        W_GROUP_STRIDE=weight.stride(0),
        W_ROW_STRIDE=weight.stride(1),
        OUT_GROUP_STRIDE=out.stride(1),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out
