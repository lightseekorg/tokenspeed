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

"""Beam's NoPE-layer Q/K RMSNorm plus the SSMax query scale: the ``layernorm.qk_rmsnorm_ssmax`` operator.

``BeamAttention.prepare_qkv`` (``tokenspeed/runtime/models/beam.py``) on a layer
without rotary embedding normalizes the query and key heads with
``BeamRMSNorm`` -- the shared kernel ``rmsnorm(x, weight, eps,
residual=None, out=None, enable_pdl=None)`` of
:mod:`tokenspeed_kernel.ops.layernorm.triton` on ``[tokens, heads, 128]`` views,
FP32 ``x * rsqrt(mean(x^2) + eps) * w`` stored once as BF16 -- and then scales
the query with ``apply_ssmax_root10(q, positions, 1, 2048.0)``: the FP32 scale
``exp(0.1 * log1p((p + 1) / 2048))`` per token, one FP32 multiply of the widened
BF16 query, one BF16 rounding. Those statements are the operator
``("layernorm", "qk_rmsnorm_ssmax")``: :func:`qk_rmsnorm_ssmax` takes ``q``,
``k``, the two ``[128]`` weights, ``eps`` and the ``[tokens]`` positions and
returns ``(q_out, k_out)``, fresh contiguous BF16 ``[tokens, heads * 128]``
tensors. Two kernels are registered:

* ``torch_qk_rmsnorm_ssmax`` (solution ``torch``, PORTABLE, any device the shared
  kernel runs on): the NoPE branch transcribed -- ``BeamRMSNorm.forward``
  evaluated for ``residual=None, out=None`` on the q heads and on the k heads
  (the same ``rmsnorm`` call with the module's weight and eps), then the three
  statements of ``apply_ssmax_root10`` with Beam's constants. Admits every shape
  the statements admit.
* ``triton_fused_qk_rmsnorm_ssmax`` (solution ``triton``, SPECIALIZED, NVIDIA
  SM100 to SM103, checked on SM100): one launch of the shared
  ``_fused_qk_rmsnorm_kernel`` through
  ``qk_rmsnorm(..., ssmax_positions=positions)`` with the constexprs
  ``ROUND_AFTER_AFFINE`` (the affine output rounded once to BF16, the value the
  eager norm stores) and ``SSMAX_INKERNEL`` (the FP32 scale computed in the
  kernel from the token's position with the CUDA math library, the multiply by
  ``2^-11`` in place of the division by 2048, one IEEE round-to-nearest
  multiply, one BF16 store; no table). It admits BF16 rank-2 ``q``/``k`` with a
  contiguous last axis and token strides covering the row (Beam's ``qkv.split``
  column views or contiguous tensors), the head pairs ``(48, 8)`` and ``(6, 1)``
  of head dimension 128, contiguous BF16 ``[128]`` weights, a contiguous int32 /
  int64 ``[tokens]`` positions vector, a finite positive ``eps``, ``tokens > 0``
  and one CUDA device. Ranked selection picks it wherever its traits admit the
  call and keeps ``torch_qk_rmsnorm_ssmax`` everywhere else; reached by name it
  raises on any other call (:func:`qk_rmsnorm_ssmax_rejection`), an empty batch
  included: no in-function fallback. ``positions >= 0`` is the caller's
  contract on both kernels, as it is for ``apply_ssmax_root10`` (never read on
  the host: the product path runs under CUDA-graph capture).

The fused kernel's scale equals Torch's bit for bit on every position of
``[0, 2^20]`` for int32 and int64 (measured, zero mismatches), and the two kernels
agree byte for byte on ``q_out`` and ``k_out``, which
``test/nvidia/ops/layernorm/test_beam_norms_cuda.py`` checks.
"""

from __future__ import annotations

import math

import torch
from tokenspeed_kernel.ops.layernorm.triton import (
    UnscaledFP8QKVGate,
    qk_rmsnorm,
    rmsnorm,
)
from tokenspeed_kernel.ops.quantization.triton import dequantize_unscaled_fp8
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.profiling import ShapeCapture, kernel_scope
from tokenspeed_kernel.registry import (
    KernelRegistry,
    KernelSpec,
    Priority,
    register_kernel,
)
from tokenspeed_kernel.selection import SelectedKernel, select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

__all__ = [
    "BEAM_HEADS",
    "BEAM_HEAD_DIM",
    "POSITION_DTYPES",
    "QK_RMSNORM_SSMAX_FAMILY",
    "QK_RMSNORM_SSMAX_MODE",
    "QK_RMSNORM_SSMAX_SIGNATURES",
    "SM100",
    "SM103",
    "SSMAX_DENOMINATOR",
    "SSMAX_OFFSET",
    "TORCH_QK_RMSNORM_SSMAX",
    "TRITON_FUSED_QK_RMSNORM_SSMAX",
    "qk_rmsnorm_ssmax",
    "qk_rmsnorm_ssmax_rejection",
    "qk_rmsnorm_ssmax_traits",
    "select_qk_rmsnorm_ssmax_kernel",
    "torch_qk_rmsnorm_ssmax",
    "triton_fused_qk_rmsnorm_ssmax",
    "validate_qk_norm_arguments",
]

QK_RMSNORM_SSMAX_FAMILY = "layernorm"
QK_RMSNORM_SSMAX_MODE = "qk_rmsnorm_ssmax"
# Registered names: the portable statements and the fused Triton kernel.
TORCH_QK_RMSNORM_SSMAX = "torch_qk_rmsnorm_ssmax"
TRITON_FUSED_QK_RMSNORM_SSMAX = "triton_fused_qk_rmsnorm_ssmax"
# The head geometry the fused kernel admits: Beam's head dimension and its
# (q heads, kv heads) pairs at TP1 and on the TP8 shard.
BEAM_HEAD_DIM = 128
BEAM_HEADS = ((48, 8), (6, 1))
# ``apply_ssmax_root10(q, positions, 1, 2048.0)`` in ``BeamAttention.prepare_qkv``;
# the fused kernel carries the same constants as literals.
SSMAX_OFFSET = 1
SSMAX_DENOMINATOR = 2048.0
POSITION_DTYPES = (torch.int32, torch.int64)
# The feature a call requests when it passes ``dequant``; both kernels declare it.
DEQUANT = "dequant"
# The architectures the fused kernel admits: NVIDIA SM100 to SM103, the range of
# Beam's flag-free defaults; its byte identity was checked on SM100 (GB200) only.
# The capability filter states the range for ranked selection, the host guard for a
# by-name override.
SM100 = ArchVersion(10, 0)
SM103 = ArchVersion(10, 3)
# The dtypes Beam's NoPE branch produces (``BeamRMSNorm`` is built with
# ``torch.bfloat16``; the runtime hands int64 or int32 positions). Both kernels
# declare both signatures, so a by-name switch never changes the facade's
# filtering.
QK_RMSNORM_SSMAX_SIGNATURES = frozenset(
    format_signature(
        q=dense_tensor_format(torch.bfloat16),
        k=dense_tensor_format(torch.bfloat16),
        q_weight=dense_tensor_format(torch.bfloat16),
        k_weight=dense_tensor_format(torch.bfloat16),
        positions=dense_tensor_format(dtype),
    )
    for dtype in POSITION_DTYPES
)


def qk_rmsnorm_ssmax_traits(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    positions: torch.Tensor,
) -> dict[str, object]:
    """The trait dict :func:`qk_rmsnorm_ssmax` selects with (metadata only).

    ``tokens`` is the token count the shared kernel derives (``q.numel() //
    q.shape[-1]``), ``head_dim`` the weight's length, ``heads`` the pair ``(q
    heads, kv heads)``, ``ndim`` the rank of ``q`` and ``k`` (0 when they
    differ), ``unit_stride`` whether both have a contiguous last axis and
    ``rows_disjoint`` whether their token strides cover their widths;
    ``weights_contiguous``, ``positions_contiguous``, ``eps_ok`` (a finite
    positive float) and ``is_cuda`` (``q`` is a CUDA tensor) complete the
    clauses of :func:`qk_rmsnorm_ssmax_rejection` that the format signatures,
    the capability and the facade's own checks leave open, so ranked selection
    admits ``triton_fused_qk_rmsnorm_ssmax`` exactly where its host guard does.
    """
    head_dim = int(q_weight.shape[0])
    q_width, kv_width = int(q.shape[-1]), int(k.shape[-1])
    return {
        "tokens": int(q.numel() // q_width) if q_width else 0,
        "head_dim": head_dim,
        "eps_ok": (
            isinstance(eps, float)
            and not isinstance(eps, bool)
            and math.isfinite(eps)
            and eps > 0.0
        ),
        "heads": ((q_width // head_dim, kv_width // head_dim) if head_dim else (0, 0)),
        "is_cuda": q.is_cuda,
        "ndim": q.ndim if q.ndim == k.ndim else 0,
        "positions_contiguous": positions.is_contiguous(),
        "rows_disjoint": q.stride(0) >= q_width and k.stride(0) >= kv_width,
        "unit_stride": q.stride(-1) == 1 and k.stride(-1) == 1,
        "weights_contiguous": q_weight.is_contiguous() and k_weight.is_contiguous(),
    }


def select_qk_rmsnorm_ssmax_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    positions: torch.Tensor,
    *,
    dequant: bool,
    solution: str | None,
    override: str | None,
) -> SelectedKernel:
    """The kernel :func:`qk_rmsnorm_ssmax` runs for this call; nothing is launched.

    Args:
        q, k, q_weight, k_weight, eps, positions: The facade's arguments
            (metadata only).
        dequant: Whether the call passes ``dequant`` (the ``dequant`` feature).
        solution: Restrict ranked selection to one solution, or ``None``.
        override: Exact kernel name to force, or ``None`` for ranked selection;
            the ``kernel_override`` context and
            ``TOKENSPEED_KERNEL_OVERRIDE_LAYERNORM_QK_RMSNORM_SSMAX`` outrank it.

    Returns:
        The selected kernel; ``.name`` is its registered name.

    Raises:
        NoKernelFoundError: No registered kernel serves the dtype signature.
    """
    # An unregistered dtype signature makes ``select_kernel`` raise
    # ``NoKernelFoundError`` naming the signature.
    signature = format_signature(
        q=dense_tensor_format(q.dtype),
        k=dense_tensor_format(k.dtype),
        q_weight=dense_tensor_format(q_weight.dtype),
        k_weight=dense_tensor_format(k_weight.dtype),
        positions=dense_tensor_format(positions.dtype),
    )
    return select_kernel(
        QK_RMSNORM_SSMAX_FAMILY,
        QK_RMSNORM_SSMAX_MODE,
        signature,
        features=frozenset({DEQUANT}) if dequant else None,
        traits=qk_rmsnorm_ssmax_traits(q, k, q_weight, k_weight, eps, positions),
        solution=solution,
        override=override,
    )


def _operator_spec(kernel_name: str) -> KernelSpec:
    """The registry spec of ``kernel_name`` if it belongs to ``layernorm.qk_rmsnorm_ssmax``.

    Raises:
        ValueError: ``kernel_name`` is unregistered or registered under another
            operator.
            ``select_kernel`` resolves an override by name alone, so the
            operator check is made here before the kernel is called with the
            norm's tensors.
    """
    spec = KernelRegistry.get().get_by_name(kernel_name)
    if spec is None or (spec.family, spec.mode) != (
        QK_RMSNORM_SSMAX_FAMILY,
        QK_RMSNORM_SSMAX_MODE,
    ):
        raise ValueError(
            f"{kernel_name!r} is not registered under "
            f"{QK_RMSNORM_SSMAX_FAMILY}.{QK_RMSNORM_SSMAX_MODE}"
        )
    return spec


def validate_qk_norm_arguments(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[int, int]:
    """Raise ``ValueError`` unless the arguments form a per-head Q/K norm call.

    Metadata only (:func:`qk_rmsnorm_ssmax`):
    ``q`` and ``k`` have a last (head) axis, the weights are one non-empty
    ``[head_dim]`` vector each, both widths are multiples of ``head_dim``, both
    hold the same token count, ``positions`` is ``[tokens]`` and every tensor is
    on one device.

    Returns:
        ``(tokens, head_dim)``.
    """
    if q.ndim < 1 or k.ndim < 1:
        raise ValueError("q and k must have shape [..., heads * head_dim]")
    if q_weight.ndim != 1 or q_weight.shape[0] == 0:
        raise ValueError(
            f"q_weight must be a non-empty [head_dim] vector, got "
            f"{tuple(q_weight.shape)}"
        )
    if k_weight.shape != q_weight.shape:
        raise ValueError(
            f"k_weight shape {tuple(k_weight.shape)} does not match q_weight "
            f"{tuple(q_weight.shape)}"
        )
    head_dim = int(q_weight.shape[0])
    if q.shape[-1] % head_dim or k.shape[-1] % head_dim:
        raise ValueError(
            f"q and k widths must be multiples of head_dim {head_dim}, got "
            f"{int(q.shape[-1])} / {int(k.shape[-1])}"
        )
    tokens = int(q.numel() // q.shape[-1]) if q.shape[-1] else 0
    k_tokens = int(k.numel() // k.shape[-1]) if k.shape[-1] else 0
    if tokens != k_tokens:
        raise ValueError(
            f"q and k must share the token count, got {tokens} / {k_tokens}"
        )
    if tuple(positions.shape) != (tokens,):
        raise ValueError(
            f"positions must have shape ({tokens},) (one per token), got "
            f"{tuple(positions.shape)}"
        )
    devices = {q.device, k.device, q_weight.device, k_weight.device, positions.device}
    if len(devices) != 1:
        raise ValueError(
            "q, k, q_weight, k_weight and positions must share one device, got "
            f"{sorted(str(device) for device in devices)}"
        )
    return tokens, head_dim


def qk_rmsnorm_ssmax(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    positions: torch.Tensor,
    *,
    solution: str | None,
    override: str | None,
    dequant: UnscaledFP8QKVGate | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize the q and k heads and scale q by position through ``layernorm.qk_rmsnorm_ssmax``.

    Args:
        q: ``[..., q_heads * head_dim]`` queries (BF16 for the registered
            kernels); a column view of a wider tensor is admitted.
        k: ``[..., kv_heads * head_dim]`` keys with the same token count.
        q_weight: ``[head_dim]`` affine weight of the query norm.
        k_weight: ``[head_dim]`` affine weight of the key norm.
        eps: Epsilon added to the FP32 mean of squares of both norms.
        positions: ``[tokens]`` absolute positions (int32 / int64 for the
            registered kernels); ``positions >= 0`` is the caller's contract and
            is never read on the host.
        solution: Restrict ranked selection to one solution (``"torch"``/``"triton"``)
            or ``None`` for any.
        override: Exact kernel name to force, or ``None`` for ranked selection; the
            ``kernel_override`` context and
            ``TOKENSPEED_KERNEL_OVERRIDE_LAYERNORM_QK_RMSNORM_SSMAX`` outrank it (see
            ``select_kernel``).
        dequant: ``None`` (the default) for the call as before. Given, ``q``
            and ``k`` are the Q and K columns of an FP8 GEMM run with unit
            scales: either kernel applies the left-out scales first (the scaled
            GEMM's BF16 output) and writes the dequantized V and output gate
            columns to ``dequant.v_out`` / ``dequant.gate_out``
            (:class:`~tokenspeed_kernel.ops.layernorm.triton.UnscaledFP8QKVGate`).

    Returns:
        ``(q_out, k_out)``: fresh BF16 ``[tokens, q_heads * head_dim]`` and
        ``[tokens, kv_heads * head_dim]`` tensors -- the normalized, scaled
        queries and the normalized keys; the inputs are not modified.

    Raises:
        ValueError: ``q`` or ``k`` has no head dimension, the weights are not one
            ``[head_dim]`` vector each, a width is not a multiple of ``head_dim``,
            the token counts disagree, ``positions`` is not ``[tokens]``, the
            tensors are not on one device, or the resolved kernel is not
            registered under ``layernorm.qk_rmsnorm_ssmax``. The Triton kernel
            raises its own ``ValueError`` on a call outside its envelope (no
            fallback).
        TypeError: The kernel returned something other than a tensor pair.
        NoKernelFoundError: No registered kernel serves the dtype signature.
    """
    tokens, head_dim = validate_qk_norm_arguments(q, k, q_weight, k_weight, positions)
    kernel = select_qk_rmsnorm_ssmax_kernel(
        q,
        k,
        q_weight,
        k_weight,
        eps,
        positions,
        dequant=dequant is not None,
        solution=solution,
        override=override,
    )
    _operator_spec(kernel.name)
    shape_params = {
        "tokens": tokens,
        "head_dim": head_dim,
        "q_heads": int(q.shape[-1]) // head_dim,
        "kv_heads": int(k.shape[-1]) // head_dim,
    }
    ShapeCapture.get().record(
        QK_RMSNORM_SSMAX_FAMILY,
        QK_RMSNORM_SSMAX_MODE,
        kernel.name,
        q.dtype,
        shape_params,
    )
    with kernel_scope(
        QK_RMSNORM_SSMAX_FAMILY,
        QK_RMSNORM_SSMAX_MODE,
        q.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        result = kernel(
            q,
            k,
            q_weight,
            k_weight,
            eps,
            positions,
            **({} if dequant is None else {"dequant": dequant}),
        )
    if not (
        isinstance(result, tuple)
        and len(result) == 2
        and all(isinstance(value, torch.Tensor) for value in result)
    ):
        raise TypeError(
            f"{kernel.name} returned {type(result).__name__}, not a "
            "(q_out, k_out) tensor pair"
        )
    return result


@register_kernel(
    "layernorm",
    "qk_rmsnorm_ssmax",
    name=TORCH_QK_RMSNORM_SSMAX,
    features={DEQUANT},
    solution="torch",
    signatures=QK_RMSNORM_SSMAX_SIGNATURES,
    # Admits every shape: the statements are general.
    traits={},
    priority=Priority.PORTABLE,
)
def torch_qk_rmsnorm_ssmax(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    positions: torch.Tensor,
    *,
    dequant: UnscaledFP8QKVGate | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``BeamAttention.prepare_qkv``'s NoPE branch, transcribed: ``BeamRMSNorm.forward``
    on the q heads and on the k heads (``residual=None, out=None`` -- the shared
    kernel with the module's weight and eps), then the three statements
    of ``apply_ssmax_root10(q, positions, 1, 2048.0)``; runs on any device the
    shared kernel runs on. Returns 2-D views of the fresh 3-D tensors. With
    ``dequant`` the four column groups are dequantized first
    (``dequantize_unscaled_fp8``)."""
    if dequant is not None:
        tokens = dequant.token_scales
        q = dequantize_unscaled_fp8(q, tokens, dequant.q_scales)
        k = dequantize_unscaled_fp8(k, tokens, dequant.k_scales)
        dequant.v_out.copy_(
            dequantize_unscaled_fp8(dequant.v, tokens, dequant.v_scales)
        )
        dequant.gate_out.copy_(
            dequantize_unscaled_fp8(dequant.gate, tokens, dequant.gate_scales)
        )
    head_dim = q_weight.shape[0]
    q = rmsnorm(
        q.reshape(-1, q.shape[-1] // head_dim, head_dim),
        q_weight,
        eps,
        residual=None,
        out=None,
        enable_pdl=None,
    )
    k = rmsnorm(
        k.reshape(-1, k.shape[-1] // head_dim, head_dim),
        k_weight,
        eps,
        residual=None,
        out=None,
        enable_pdl=None,
    )
    ratio = (positions.float() + float(SSMAX_OFFSET)) / float(SSMAX_DENOMINATOR)
    scale = torch.exp(0.1 * torch.log1p(ratio))
    q = (q * scale[..., None, None]).to(q.dtype)
    return q.flatten(1), k.flatten(1)


def qk_rmsnorm_ssmax_rejection(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    positions: torch.Tensor,
) -> str:
    """Why ``triton_fused_qk_rmsnorm_ssmax`` cannot serve this call; empty when admitted.

    Metadata only (no device read, no synchronization: the product path runs
    under CUDA-graph capture): rank-2 ``q`` and ``k`` with one shared, positive
    token count (an empty grid is not launched); BF16 ``q``, ``k``, ``q_weight``,
    ``k_weight``; widths exact multiples of 128 and ``(q heads, kv heads)`` in
    ``BEAM_HEADS``; a contiguous last axis and token strides covering the row
    width (Beam's ``qkv.split`` column views or contiguous tensors); contiguous
    ``[128]`` weights; ``positions`` a contiguous int32 / int64 ``[tokens]``
    vector; a finite positive float ``eps`` (a kernel constexpr); every tensor a
    CUDA tensor on ``q``'s device; one NVIDIA SM100 to SM103 device.
    ``positions >= 0`` is not a clause: it is the caller's contract, as it is
    for ``apply_ssmax_root10``. The reason names the first rejecting clause; the
    device and platform clauses come last so that shape and layout findings are
    reported on any device. Ranked selection applies the same clauses as the
    spec's traits and capability (:func:`qk_rmsnorm_ssmax_traits`); this guard
    stops a by-name call outside them.

    Args:
        q: Queries the kernel normalizes and scales.
        k: Keys the kernel normalizes.
        q_weight: Affine weight of the query norm.
        k_weight: Affine weight of the key norm.
        eps: Variance epsilon (the kernel constexpr).
        positions: Absolute positions the kernel reads the scale from.

    Returns:
        The rejection reason, or ``""`` when the kernel admits the call.
    """
    if q.ndim != 2 or k.ndim != 2:
        return (
            f"q and k must be rank 2 [tokens, heads * {BEAM_HEAD_DIM}], got "
            f"{tuple(q.shape)} / {tuple(k.shape)}"
        )
    if q.shape[0] != k.shape[0]:
        return (
            f"q and k must share the token count, got {int(q.shape[0])} / "
            f"{int(k.shape[0])}"
        )
    tokens = int(q.shape[0])
    if tokens == 0:
        return "tokens == 0: an empty grid is not launched (tokens must be positive)"
    for name, tensor in (
        ("q", q),
        ("k", k),
        ("q_weight", q_weight),
        ("k_weight", k_weight),
    ):
        if tensor.dtype is not torch.bfloat16:
            return f"{name} must be {torch.bfloat16}, got {tensor.dtype}"
    if q.shape[1] % BEAM_HEAD_DIM or k.shape[1] % BEAM_HEAD_DIM:
        return (
            f"q and k widths must be multiples of head_dim {BEAM_HEAD_DIM}, got "
            f"{int(q.shape[1])} / {int(k.shape[1])}"
        )
    heads = (int(q.shape[1]) // BEAM_HEAD_DIM, int(k.shape[1]) // BEAM_HEAD_DIM)
    if heads not in BEAM_HEADS:
        return f"(q heads, kv heads) must be one of {BEAM_HEADS}, got {heads}"
    if q.stride(-1) != 1 or k.stride(-1) != 1:
        return (
            "q and k require a contiguous last axis, got strides "
            f"{tuple(q.stride())} / {tuple(k.stride())}"
        )
    if q.stride(0) < q.shape[1] or k.stride(0) < k.shape[1]:
        return (
            "q and k token strides must cover the row width (column views of a wider "
            f"tensor or contiguous), got strides {tuple(q.stride())} / "
            f"{tuple(k.stride())}"
        )
    for name, weight in (("q_weight", q_weight), ("k_weight", k_weight)):
        if tuple(weight.shape) != (BEAM_HEAD_DIM,):
            return f"{name} must be [{BEAM_HEAD_DIM}], got {tuple(weight.shape)}"
        if not weight.is_contiguous():
            return f"{name} must be contiguous"
    if positions.ndim != 1 or positions.shape[0] != tokens:
        return (
            f"positions must be [{tokens}] (one per token), got "
            f"{tuple(positions.shape)}"
        )
    if not positions.is_contiguous():
        return "positions must be contiguous"
    if positions.dtype not in POSITION_DTYPES:
        return f"positions must be int32 or int64, got {positions.dtype}"
    if (
        isinstance(eps, bool)
        or not isinstance(eps, float)
        or not math.isfinite(eps)
        or eps <= 0.0
    ):
        return f"eps must be a finite positive float, got {eps!r}"
    if not q.is_cuda:
        return f"q must be a CUDA tensor, got device {q.device}"
    for name, tensor in (
        ("k", k),
        ("q_weight", q_weight),
        ("k_weight", k_weight),
        ("positions", positions),
    ):
        if tensor.device != q.device:
            return f"{name} is on {tensor.device}, q is on {q.device}"
    platform = current_platform()
    if not (platform.is_nvidia and SM100 <= platform.arch_version <= SM103):
        return (
            f"{TRITON_FUSED_QK_RMSNORM_SSMAX} runs on NVIDIA SM100 to SM103 only, "
            f"got {platform.device_name}"
        )
    return ""


@register_kernel(
    "layernorm",
    "qk_rmsnorm_ssmax",
    name=TRITON_FUSED_QK_RMSNORM_SSMAX,
    features={DEQUANT},
    solution="triton",
    capability=CapabilityRequirement(
        vendors=frozenset({"nvidia"}),
        min_arch_version=SM100,
        max_arch_version=SM103,
    ),
    # The same BF16 q/k/weights x int32|int64 positions signatures as
    # ``torch_qk_rmsnorm_ssmax``, so a name override between the two never
    # changes the facade's filtering.
    signatures=QK_RMSNORM_SSMAX_SIGNATURES,
    # qk_rmsnorm_ssmax_rejection's clauses (qk_rmsnorm_ssmax_traits): at least
    # one token, Beam's head pairs of 128, rank-2 views with disjoint rows and a
    # contiguous last axis, contiguous weights and positions, a valid eps, CUDA.
    traits={
        "tokens_min": frozenset({1}),
        "head_dim": frozenset({BEAM_HEAD_DIM}),
        "eps_ok": frozenset({True}),
        "heads": frozenset(BEAM_HEADS),
        "is_cuda": frozenset({True}),
        "ndim": frozenset({2}),
        "positions_contiguous": frozenset({True}),
        "rows_disjoint": frozenset({True}),
        "unit_stride": frozenset({True}),
        "weights_contiguous": frozenset({True}),
    },
    priority=Priority.SPECIALIZED,
)
def triton_fused_qk_rmsnorm_ssmax(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    positions: torch.Tensor,
    *,
    dequant: UnscaledFP8QKVGate | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One launch of the shared Q/K RMSNorm kernel with the SSMax fold on an
    admitted call; raise otherwise. ``dequant`` folds the unscaled FP8 GEMM's
    scales into the same launch (``qk_rmsnorm``'s DEQUANT mode).

    Args:
        q: ``[tokens, 48 * 128]`` or ``[tokens, 6 * 128]`` BF16 CUDA queries with a
            contiguous last axis (a column view of ``qkv`` or contiguous).
        k: ``[tokens, 8 * 128]`` or ``[tokens, 1 * 128]`` BF16 keys on ``q``'s
            device, the pair matching ``q``'s.
        q_weight: ``[128]`` contiguous BF16 weight on ``q``'s device.
        k_weight: ``[128]`` contiguous BF16 weight on ``q``'s device.
        eps: Finite positive variance epsilon (a kernel constexpr, as in the
            product launch).
        positions: ``[tokens]`` contiguous int32 / int64 positions on ``q``'s
            device; ``>= 0`` by the caller's contract.

    Returns:
        ``(q_out, k_out)``, fresh contiguous BF16 ``[tokens, heads * 128]``
        tensors, byte-identical to ``torch_qk_rmsnorm_ssmax``'s pair on the same
        inputs.

    Raises:
        ValueError: :func:`qk_rmsnorm_ssmax_rejection` rejects the call (zero
            tokens and a device outside SM100 to SM103 included). There is no
            fallback to the torch statements.
    """
    rejection = qk_rmsnorm_ssmax_rejection(q, k, q_weight, k_weight, eps, positions)
    if rejection:
        raise ValueError(
            f"{TRITON_FUSED_QK_RMSNORM_SSMAX} cannot serve this call: {rejection}"
        )
    # Beam scales by the weight itself; WEIGHT_OFFSET=0.0 compiles the add out.
    return qk_rmsnorm(
        q,
        k,
        q_weight,
        k_weight,
        eps,
        weight_offset=0.0,
        ssmax_positions=positions,
        dequant=dequant,
    )
