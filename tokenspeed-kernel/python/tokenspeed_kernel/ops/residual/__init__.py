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
#
from __future__ import annotations

import math

import torch as _torch
from tokenspeed_kernel.platform import Platform
from tokenspeed_kernel.profiling import ShapeCapture, kernel_scope
from tokenspeed_kernel.selection import NoKernelFoundError, select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

# ===-----------------------------------------------------------------------===#
# Kimi K3 Attention Residual
# ===-----------------------------------------------------------------------===#

# The Blackwell launcher instantiates aligned hidden sizes in [4096, 8192].
# Its block-residual source stride is passed to CUDA as a signed 32-bit int.
# AMD Gluon currently specializes Kimi K3's H=7168 fused-output-norm path.
_MAX_BLACKWELL_TOKENS = ((1 << 31) - 1) // 7168
_MAX_AMD_GLUON_TOKENS = 65536
_MAX_N = 12


def _select_attn_res_kernel(
    layer_residual,
    block_residual,
    res_weight,
    rms_weight,
    *,
    out_norm_weight,
    output_eps,
    eps,
    delta,
    num_valid_blocks,
    block_write_idx,
):
    tokens, hidden_size = layer_residual.shape
    valid_blocks = (
        block_residual.shape[0] if num_valid_blocks is None else int(num_valid_blocks)
    )
    if not 0 <= valid_blocks <= block_residual.shape[0]:
        raise ValueError("num_valid_blocks is outside block_residual")
    if block_write_idx != -1 and (
        block_write_idx != valid_blocks or block_write_idx >= block_residual.shape[0]
    ):
        raise ValueError("block_write_idx must append within block_residual")
    num_sources = valid_blocks + 1
    input_tensors = [
        layer_residual,
        block_residual,
        res_weight,
        rms_weight,
    ]
    if out_norm_weight is not None:
        input_tensors.append(out_norm_weight)
    if delta is not None:
        input_tensors.append(delta)
    inputs_on_same_gpu = layer_residual.is_cuda and all(
        tensor.is_cuda and tensor.device == layer_residual.device
        for tensor in input_tensors
    )
    hidden_dimension_contiguous = (
        layer_residual.stride(-1) == 1
        and block_residual.ndim == 3
        and block_residual.stride(-1) == 1
        and res_weight.stride(-1) == 1
        and rms_weight.stride(-1) == 1
        and (out_norm_weight is None or out_norm_weight.stride(-1) == 1)
    )
    delta_compatible = delta is None or (
        delta.shape == layer_residual.shape
        and delta.dtype == layer_residual.dtype
        and delta.device == layer_residual.device
        and delta.stride(-1) == 1
    )
    platform = Platform.get()
    if platform.is_cdna4 or platform.is_cdna5:
        eligible = (
            hidden_size == 7168
            and out_norm_weight is not None
            and 1 <= tokens <= _MAX_AMD_GLUON_TOKENS
        )
    else:
        eligible = (
            4096 <= hidden_size <= 8192
            and hidden_size % 1024 == 0
            and 1 <= tokens <= _MAX_BLACKWELL_TOKENS
        )
    eligible = eligible and 1 <= num_sources <= _MAX_N
    signature = format_signature(
        layer_residual=dense_tensor_format(layer_residual.dtype),
        block_residual=dense_tensor_format(block_residual.dtype),
    )
    kernel = select_kernel(
        "residual",
        "attn_res_fwd",
        signature,
        traits={
            "fused_output_norm": out_norm_weight is not None,
            "has_delta": delta is not None,
            "hidden_dimension_contiguous": hidden_dimension_contiguous,
            "inputs_on_same_gpu": inputs_on_same_gpu,
            "large_prefill": tokens > 32,
            "hidden_size": hidden_size,
            "delta_compatible": delta_compatible,
            "partial_block_storage": valid_blocks != block_residual.shape[0],
            "separate_output_eps": out_norm_weight is not None and output_eps != eps,
            "writes_block": block_write_idx >= 0,
        },
        solution=None if eligible else "torch",
    )
    return kernel, valid_blocks


def attn_res_fwd(
    layer_residual,
    block_residual,
    res_weight,
    rms_weight,
    eps=1e-6,
    out_norm_weight=None,
    out_norm_eps=None,
    *,
    delta=None,
    num_valid_blocks=None,
    block_write_idx=-1,
):
    """Fused Attention-Residual forward.

    Candidates are ``block_residual[0..K-1]`` followed by ``layer_residual``
    (N = K + 1). Computes ``softmax_n(<RMSNorm(v_n), rms_weight * res_weight>)``
    over candidates, then the weighted sum of the raw candidates.

    Args:
        layer_residual: bf16 ``[T, H]`` current residual stream.
        block_residual: bf16 ``[K, T, H]`` periodic-snapshot storage.
        res_weight: bf16 ``[H]`` scorer projection weight.
        rms_weight: bf16 ``[H]`` RMSNorm weight.
        eps: RMSNorm epsilon.
        out_norm_weight: optional bf16 ``[H]``; when given, the following
            RMSNorm is fused into the epilogue and the return value is the
            normed mix.
        out_norm_eps: Optional output RMSNorm epsilon. Defaults to ``eps``.
        delta: Optional bf16 ``[T, H]`` update added to ``layer_residual``.
            The BF16-rounded sum is written back to ``layer_residual`` before
            it participates in the AttnRes mix.
        num_valid_blocks: Number of leading snapshots to include. Defaults to
            all rows in ``block_residual``.
        block_write_idx: Optional snapshot row receiving the updated layer
            residual. It must immediately follow the valid snapshots.

    Returns:
        bf16 ``[T, H]`` mixed residual (normed when ``out_norm_weight`` given).
    """
    output_eps = (
        eps if out_norm_weight is None or out_norm_eps is None else out_norm_eps
    )
    kernel, valid_blocks = _select_attn_res_kernel(
        layer_residual,
        block_residual,
        res_weight,
        rms_weight,
        out_norm_weight=out_norm_weight,
        output_eps=output_eps,
        eps=eps,
        delta=delta,
        num_valid_blocks=num_valid_blocks,
        block_write_idx=block_write_idx,
    )
    return kernel(
        layer_residual=layer_residual,
        block_residual=block_residual,
        res_weight=res_weight,
        rms_weight=rms_weight,
        eps=eps,
        out_norm_weight=out_norm_weight,
        out_norm_eps=output_eps,
        delta=delta,
        num_valid_blocks=valid_blocks,
        block_write_idx=block_write_idx,
    )


def attn_res_fwd_available(
    layer_residual,
    block_residual,
    res_weight,
    rms_weight,
    eps=1e-6,
    out_norm_weight=None,
    out_norm_eps=None,
    *,
    delta=None,
    num_valid_blocks=None,
    block_write_idx=-1,
):
    """Return whether a specialized kernel supports the exact AttnRes call.

    Args:
        layer_residual: Current residual stream shaped ``[tokens, hidden_size]``.
        block_residual: Periodic snapshots shaped ``[blocks, tokens, hidden_size]``.
        res_weight: AttnRes projection weight shaped ``[hidden_size]``.
        rms_weight: AttnRes RMSNorm weight shaped ``[hidden_size]``.
        eps: AttnRes RMSNorm epsilon.
        out_norm_weight: Optional following RMSNorm weight.
        out_norm_eps: Optional following RMSNorm epsilon. Defaults to ``eps``.
        delta: Optional update added in place to ``layer_residual``.
        num_valid_blocks: Number of leading snapshots included in the mix.
        block_write_idx: Optional row receiving the updated residual; it must
            immediately follow the valid snapshots.

    Returns:
        ``True`` when registry dispatch can run a specialized implementation.
    """
    output_eps = (
        eps if out_norm_weight is None or out_norm_eps is None else out_norm_eps
    )
    try:
        kernel, _ = _select_attn_res_kernel(
            layer_residual,
            block_residual,
            res_weight,
            rms_weight,
            out_norm_weight=out_norm_weight,
            output_eps=output_eps,
            eps=eps,
            delta=delta,
            num_valid_blocks=num_valid_blocks,
            block_write_idx=block_write_idx,
        )
    except (NoKernelFoundError, ValueError):
        return False
    from tokenspeed_kernel.ops.residual.torch import torch_attn_res_fwd

    return kernel.impl is not torch_attn_res_fwd


# ===-----------------------------------------------------------------------===#
# Qwen4 Gated Hyperconnection
# ===-----------------------------------------------------------------------===#


def _flatten_rows(value: _torch.Tensor, width: int, name: str) -> _torch.Tensor:
    if value.ndim < 1 or value.shape[-1] != width:
        raise ValueError(
            f"{name} must have last dimension {width}, got {tuple(value.shape)}"
        )
    return value.reshape(-1, width)


def _same_tensor_contract(
    reference: _torch.Tensor, value: _torch.Tensor, name: str
) -> None:
    if value.dtype != reference.dtype:
        raise ValueError(
            f"{name} dtype must match the input ({reference.dtype}), got {value.dtype}"
        )
    if value.device != reference.device:
        raise ValueError(
            f"{name} device must match the input ({reference.device}), got {value.device}"
        )


def prepare_gated_residual_weight_cache(up_weight: _torch.Tensor, lowrank: int) -> bool:
    """Prepare derived mix-up weights after an initial or online weight load.

    CUDA graphs retain the address of backend-specific derived weights. The
    first call creates that fixed-address allocation outside forward; later
    calls update it in place after online weight synchronization. It is a no-op
    when the selected platform does not need a derived weight.

    Args:
        up_weight: Source mix-up weight shaped ``[wide, lowrank]``.
        lowrank: Rank of the mix gate bottleneck.

    Returns:
        Whether a backend-specific derived allocation was prepared.
    """
    if lowrank <= 0:
        raise ValueError("lowrank must be positive")
    if up_weight.ndim != 2 or int(up_weight.shape[1]) != lowrank:
        raise ValueError(
            f"up_weight must have shape [wide, {lowrank}], got "
            f"{tuple(up_weight.shape)}"
        )
    from tokenspeed_kernel.ops.residual.cute_dsl import (
        _prepare_padded_up_weight,
    )

    return _prepare_padded_up_weight(up_weight, lowrank)


def gated_residual_mix(
    normalized: _torch.Tensor,
    projection_weight: _torch.Tensor,
    up_weight: _torch.Tensor,
    hc_count: int,
    hidden_size: int,
    lowrank: int,
    *,
    projection_scale: float = 1.0,
    override: str | None = None,
    solution: str | None = None,
) -> tuple[_torch.Tensor, _torch.Tensor | None]:
    """Mix normalized hyperconnection branches and optionally form inject logits.

    The first projection is stored as one matrix. Its leading ``lowrank`` rows
    are the mix-down weight and, when present, its final ``hc_count`` rows are
    the block-injection weight. This preserves a single read of the wide input.

    Args:
        normalized: Normalized GPU residual streams shaped
            ``[..., hc_count * hidden_size]``.
        projection_weight: Fused down/inject weight shaped either
            ``[lowrank, hc_count * hidden_size]`` or
            ``[lowrank + hc_count, hc_count * hidden_size]``.
        up_weight: Mix-up weight shaped
            ``[hc_count * hidden_size, lowrank]``.
        hc_count: Number of residual branches.
        hidden_size: Width of one branch.
        lowrank: Rank of the mix gate bottleneck.
        projection_scale: Scale applied to down and inject projection results.
            It is ``1`` when an exact power-of-two scale was folded into the
            checkpoint weight and ``1 / hc_count`` otherwise.
        override: Optional exact registered kernel name.
        solution: Optional registered solution name.

    Returns:
        A pair containing the mixed tensor shaped ``[..., hidden_size]`` and
        optional inject logits shaped ``[..., hc_count]``.
    """
    if hc_count <= 1 or hidden_size <= 0 or lowrank <= 0:
        raise ValueError("hc_count must exceed one and all dimensions must be positive")
    if not math.isfinite(projection_scale) or projection_scale <= 0:
        raise ValueError("projection_scale must be finite and positive")

    wide = hc_count * hidden_size
    flat = _flatten_rows(normalized, wide, "normalized")
    if projection_weight.ndim != 2 or projection_weight.shape[1] != wide:
        raise ValueError(
            "projection_weight must have shape "
            f"[{lowrank} or {lowrank + hc_count}, {wide}], got "
            f"{tuple(projection_weight.shape)}"
        )
    projection_rows = int(projection_weight.shape[0])
    if projection_rows not in (lowrank, lowrank + hc_count):
        raise ValueError(
            f"projection_weight has {projection_rows} rows; expected {lowrank} "
            f"or {lowrank + hc_count}"
        )
    if up_weight.shape != (wide, lowrank):
        raise ValueError(
            f"up_weight must have shape {(wide, lowrank)}, got {tuple(up_weight.shape)}"
        )
    _same_tensor_contract(flat, projection_weight, "projection_weight")
    _same_tensor_contract(flat, up_weight, "up_weight")
    if not flat.is_cuda:
        raise ValueError("gated_residual_mix requires GPU tensors")

    rows = int(flat.shape[0])
    has_inject = projection_rows != lowrank
    leading_shape = normalized.shape[:-1]
    if rows == 0:
        mixed = normalized.new_empty((*leading_shape, hidden_size))
        inject = (
            normalized.new_empty((*leading_shape, hc_count)) if has_inject else None
        )
        return mixed, inject

    traits = {
        "num_tokens": rows,
        "hc_count": hc_count,
        "hidden_size": hidden_size,
        "lowrank": lowrank,
        "has_inject": has_inject,
        "contiguous": bool(
            flat.is_contiguous()
            and projection_weight.is_contiguous()
            and up_weight.is_contiguous()
        ),
        "folded_scale": projection_scale == 1.0,
        "deterministic": _torch.are_deterministic_algorithms_enabled(),
        "capturing": bool(flat.is_cuda and _torch.cuda.is_current_stream_capturing()),
    }
    signature = format_signature(
        normalized=dense_tensor_format(flat.dtype),
        projection_weight=dense_tensor_format(projection_weight.dtype),
        up_weight=dense_tensor_format(up_weight.dtype),
    )
    kernel = select_kernel(
        "residual",
        "hyperconnection_mix",
        signature,
        traits=traits,
        override=override,
        solution=solution,
    )
    ShapeCapture.get().record(
        "residual", "hyperconnection_mix", kernel.name, flat.dtype, traits
    )
    with kernel_scope(
        "residual",
        "hyperconnection_mix",
        flat.dtype,
        kernel_name=kernel.name,
        **traits,
    ):
        mixed, inject = kernel(
            flat,
            projection_weight,
            up_weight,
            hc_count,
            hidden_size,
            lowrank,
            projection_scale,
        )
    mixed = mixed.reshape(*leading_shape, hidden_size)
    if inject is not None:
        inject = inject.reshape(*leading_shape, hc_count)
    return mixed, inject


def gated_residual_combine(
    block_output: _torch.Tensor,
    residual: _torch.Tensor,
    inject_logits: _torch.Tensor,
    hc_count: int,
    hidden_size: int,
    *,
    override: str | None = None,
    solution: str | None = None,
) -> _torch.Tensor:
    """Gate one sublayer output and inject it into every residual branch.

    Args:
        block_output: GPU sublayer output shaped ``[..., hidden_size]``.
        residual: Hyperconnection stream shaped
            ``[..., hc_count * hidden_size]``.
        inject_logits: Per-branch logits shaped ``[..., hc_count]``.
        hc_count: Number of residual branches.
        hidden_size: Width of one branch.
        override: Optional exact registered kernel name.
        solution: Optional registered solution name.

    Returns:
        Updated residual stream with the same shape as ``residual``.
    """
    wide = hc_count * hidden_size
    block_flat = _flatten_rows(block_output, hidden_size, "block_output")
    residual_flat = _flatten_rows(residual, wide, "residual")
    inject_flat = _flatten_rows(inject_logits, hc_count, "inject_logits")
    rows = int(block_flat.shape[0])
    if residual_flat.shape[0] != rows or inject_flat.shape[0] != rows:
        raise ValueError("block_output, residual, and inject_logits must share rows")
    _same_tensor_contract(block_flat, residual_flat, "residual")
    _same_tensor_contract(block_flat, inject_flat, "inject_logits")
    if not block_flat.is_cuda:
        raise ValueError("gated_residual_combine requires GPU tensors")
    if rows == 0:
        return residual.to(block_output.dtype)

    traits = {
        "num_tokens": rows,
        "hc_count": hc_count,
        "hidden_size": hidden_size,
    }
    signature = format_signature(
        block_output=dense_tensor_format(block_flat.dtype),
        residual=dense_tensor_format(residual_flat.dtype),
        inject_logits=dense_tensor_format(inject_flat.dtype),
    )
    kernel = select_kernel(
        "residual",
        "hyperconnection_combine",
        signature,
        traits=traits,
        override=override,
        solution=solution,
    )
    ShapeCapture.get().record(
        "residual", "hyperconnection_combine", kernel.name, block_flat.dtype, traits
    )
    with kernel_scope(
        "residual",
        "hyperconnection_combine",
        block_flat.dtype,
        kernel_name=kernel.name,
        **traits,
    ):
        result = kernel(
            block_flat,
            residual_flat,
            inject_flat,
            hc_count,
            hidden_size,
        )
    return result.reshape(residual.shape)


# ===-----------------------------------------------------------------------===#
# DeepSeek V4 mHC
# ===-----------------------------------------------------------------------===#


def mhc_pre(
    residual: _torch.Tensor,
    fn: _torch.Tensor,
    hc_scale: _torch.Tensor,
    hc_base: _torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    override: str | None = None,
    solution: str | None = None,
    *,
    norm_weight: _torch.Tensor | None,
    norm_eps: float | None,
) -> tuple[_torch.Tensor, _torch.Tensor, _torch.Tensor]:
    """Compute the mHC pre-mapping for one residual stream.

    Args:
        residual: BF16 residual streams shaped ``[..., hc_mult, hidden_size]``.
        fn: FP32 mixing projection shaped
            ``[2 * hc_mult + hc_mult**2, hc_mult * hidden_size]``.
        hc_scale: FP32 pre, post, and combine scales shaped ``[3]``.
        hc_base: FP32 mixing biases shaped
            ``[2 * hc_mult + hc_mult**2]``.
        rms_eps: Epsilon used by the residual RMS normalization.
        hc_eps: Epsilon added during pre-mix and Sinkhorn normalization.
        sinkhorn_iters: Number of Sinkhorn row/column normalization iterations.
        override: Optional exact registered kernel name.
        solution: Optional registered solution name.
        norm_weight: Optional BF16 RMSNorm weight fused into the selected kernel.
        norm_eps: Optional RMSNorm epsilon. This must be provided together with
            ``norm_weight``.

    Returns:
        A tuple of the BF16 layer input ``[..., hidden_size]``, FP32 post mix
        ``[..., hc_mult, 1]``, and FP32 combine mix
        ``[..., hc_mult, hc_mult]``.
    """
    if (norm_weight is None) != (norm_eps is None):
        raise ValueError("norm_weight and norm_eps must be provided together")

    hc_mult = int(residual.shape[-2])
    hidden_size = int(residual.shape[-1])
    num_tokens = int(residual.numel() // (hc_mult * hidden_size))
    traits = {
        "num_tokens": num_tokens,
        "hc_mult": hc_mult,
        "hidden_size": hidden_size,
        "sinkhorn_iters": int(sinkhorn_iters),
    }
    signature = format_signature(
        residual=dense_tensor_format(residual.dtype),
        fn=dense_tensor_format(fn.dtype),
        hc_scale=dense_tensor_format(hc_scale.dtype),
        hc_base=dense_tensor_format(hc_base.dtype),
    )
    kernel = select_kernel(
        "residual",
        "mhc_pre",
        signature,
        traits=traits,
        override=override,
        solution=solution,
    )
    ShapeCapture.get().record(
        "residual", "mhc_pre", kernel.name, residual.dtype, traits
    )
    with kernel_scope(
        "residual",
        "mhc_pre",
        residual.dtype,
        kernel_name=kernel.name,
        **traits,
    ):
        return kernel(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_eps,
            sinkhorn_iters,
            norm_weight,
            norm_eps,
        )


def mhc_post(
    hidden_states: _torch.Tensor,
    residual: _torch.Tensor,
    post: _torch.Tensor,
    comb: _torch.Tensor,
    override: str | None = None,
    solution: str | None = None,
) -> _torch.Tensor:
    """Compute the mHC post-mapping and residual-stream update.

    Args:
        hidden_states: BF16 layer output shaped ``[..., hidden_size]``.
        residual: BF16 residual streams shaped
            ``[..., hc_mult, hidden_size]``.
        post: FP32 post mix shaped ``[..., hc_mult, 1]``.
        comb: FP32 combine mix shaped ``[..., hc_mult, hc_mult]``.
        override: Optional exact registered kernel name.
        solution: Optional registered solution name.

    Returns:
        Updated BF16 residual streams with the same shape as ``residual``.
    """
    hc_mult = int(residual.shape[-2])
    hidden_size = int(residual.shape[-1])
    num_tokens = int(residual.numel() // (hc_mult * hidden_size))
    traits = {
        "num_tokens": num_tokens,
        "hc_mult": hc_mult,
        "hidden_size": hidden_size,
    }
    signature = format_signature(
        hidden_states=dense_tensor_format(hidden_states.dtype),
        residual=dense_tensor_format(residual.dtype),
        post=dense_tensor_format(post.dtype),
        comb=dense_tensor_format(comb.dtype),
    )
    kernel = select_kernel(
        "residual",
        "mhc_post",
        signature,
        traits=traits,
        override=override,
        solution=solution,
    )
    ShapeCapture.get().record(
        "residual", "mhc_post", kernel.name, residual.dtype, traits
    )
    with kernel_scope(
        "residual",
        "mhc_post",
        residual.dtype,
        kernel_name=kernel.name,
        **traits,
    ):
        return kernel(hidden_states, residual, post, comb)


def mhc_fused_hc(
    x_prev: _torch.Tensor,
    residual_prev: _torch.Tensor,
    post_prev: _torch.Tensor,
    comb_prev: _torch.Tensor,
    fn: _torch.Tensor,
    hc_scale: _torch.Tensor,
    hc_base: _torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    norm_weight: _torch.Tensor | None,
    norm_eps: float | None,
) -> tuple[_torch.Tensor, _torch.Tensor, _torch.Tensor, _torch.Tensor]:
    """Compose the registered previous post-mapping and current pre-mapping.

    Args:
        x_prev: BF16 previous layer output shaped ``[..., hidden_size]``.
        residual_prev: BF16 previous residual streams shaped
            ``[..., hc_mult, hidden_size]``.
        post_prev: FP32 previous post mix shaped ``[..., hc_mult, 1]``.
        comb_prev: FP32 previous combine mix shaped
            ``[..., hc_mult, hc_mult]``.
        fn: FP32 current mixing projection shaped
            ``[2 * hc_mult + hc_mult**2, hc_mult * hidden_size]``.
        hc_scale: FP32 current pre, post, and combine scales shaped ``[3]``.
        hc_base: FP32 current mixing biases shaped
            ``[2 * hc_mult + hc_mult**2]``.
        rms_eps: Epsilon used by the residual RMS normalization.
        hc_eps: Epsilon added during pre-mix and Sinkhorn normalization.
        sinkhorn_iters: Number of Sinkhorn row/column normalization iterations.
        norm_weight: Optional RMSNorm weight applied to the current layer input.
        norm_eps: RMSNorm epsilon. This must be provided with ``norm_weight``.

    Returns:
        A tuple of the current BF16 residual streams, BF16 layer input, FP32
        post mix, and FP32 combine mix.
    """
    residual_cur = mhc_post(x_prev, residual_prev, post_prev, comb_prev)
    layer_input, post_cur, comb_cur = mhc_pre(
        residual_cur,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_eps,
        sinkhorn_iters,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
    )
    return residual_cur, layer_input, post_cur, comb_cur


# Backend registration (side-effect imports)
# isort: off
import tokenspeed_kernel.ops.residual.cuda  # noqa: E402,F401
import tokenspeed_kernel.ops.residual.cute_dsl  # noqa: E402,F401
import tokenspeed_kernel.ops.residual.deep_gemm  # noqa: E402,F401
import tokenspeed_kernel.ops.residual.gluon  # noqa: E402,F401
import tokenspeed_kernel.ops.residual.torch  # noqa: E402,F401
import tokenspeed_kernel.ops.residual.triton  # noqa: E402,F401

# isort: on


__all__ = [
    "attn_res_fwd",
    "attn_res_fwd_available",
    "gated_residual_combine",
    "gated_residual_mix",
    "prepare_gated_residual_weight_cache",
    "mhc_fused_hc",
    "mhc_post",
    "mhc_pre",
]
