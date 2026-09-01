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

"""FlashInfer fused KDA decode adapter."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

_HEAD_DIM = 128
_SUPPORTED_HEADS = frozenset({12, 24, 32, 48, 96})


def _load_fused_kda_decode() -> Callable[..., torch.Tensor] | None:
    try:
        from flashinfer import kda_decode
    except (ImportError, OSError, RuntimeError):
        return None
    # FlashInfer keeps the public wrapper importable when its CuTe backend is
    # unavailable; selecting that wrapper would fail at execution and suppress
    # TokenSpeed's safe Triton fallback.
    if not bool(getattr(kda_decode, "_FUSED_KDA_DECODE_AVAILABLE", False)):
        return None
    return kda_decode.fused_kda_decode


_fused_kda_decode = _load_fused_kda_decode()


@dataclass(frozen=True)
class _FlashInferKdaDecodeWeights:
    conv: torch.Tensor
    norm: torch.Tensor


def _fused_kda_decode_available() -> bool:
    """Return whether the public fused KDA backend is usable on this device."""
    if _fused_kda_decode is None:
        return False
    platform = current_platform()
    return bool(
        platform.is_nvidia
        and ArchVersion(10, 0) <= platform.arch_version <= ArchVersion(10, 3)
    )


def prepare_flashinfer_kda_decode_weights(
    conv_weights: torch.Tensor,
    norm_weight: torch.Tensor,
    prepared_weights: object | None = None,
) -> object | None:
    """Prepare stable weights for fused KDA decode, or return ``None``.

    The returned object is opaque outside ``tokenspeed-kernel``.

    Args:
        conv_weights: TokenSpeed Q/K/V convolution bank ``[3 * H * 128, 4]``.
        norm_weight: Output RMSNorm weight ``[128]``.
        prepared_weights: Existing plan whose storage should be refreshed in
            place after a model-weight update.

    Returns:
        The prepared backend plan, or ``None`` when unsupported.
    """
    if not _fused_kda_decode_available():
        return None
    if conv_weights.ndim != 2 or conv_weights.shape[1] != 4:
        raise ValueError(
            "conv_weights must have shape [3 * H * 128, 4], got "
            f"{tuple(conv_weights.shape)}"
        )
    if conv_weights.shape[0] % (3 * _HEAD_DIM):
        raise ValueError("conv_weights row count must be divisible by 3 * 128")
    num_heads = conv_weights.shape[0] // (3 * _HEAD_DIM)
    if num_heads not in _SUPPORTED_HEADS or norm_weight.shape != (_HEAD_DIM,):
        return None
    hidden_size = num_heads * _HEAD_DIM
    conv = conv_weights.detach().reshape(3, hidden_size, 4).permute(0, 2, 1)
    norm = norm_weight.detach()
    if prepared_weights is not None:
        prepared = _require_prepared_weights(prepared_weights)
        if (
            prepared.conv.shape != conv.shape
            or prepared.norm.shape != norm.shape
            or prepared.conv.device != conv.device
            or prepared.norm.device != norm.device
        ):
            raise ValueError("prepared KDA weights cannot change shape or device")
        prepared.conv.copy_(conv)
        prepared.norm.copy_(norm)
        return prepared
    return _FlashInferKdaDecodeWeights(
        conv=conv.to(torch.float32).contiguous(),
        norm=norm.to(torch.float32).contiguous(),
    )


def _require_prepared_weights(
    prepared_weights: object | None,
) -> _FlashInferKdaDecodeWeights:
    if not isinstance(prepared_weights, _FlashInferKdaDecodeWeights):
        raise TypeError(
            "prepared_weights must be returned by prepare_flashinfer_kda_decode_weights"
        )
    return prepared_weights


def _flashinfer_kda_fused_paged_decode(
    mixed_qkv: torch.Tensor,
    conv_weights: torch.Tensor,
    conv_states: torch.Tensor,
    f_a_out: torch.Tensor,
    f_b_weight: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    state_pool: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    num_heads: int,
    head_dim: int,
    cu_seqlens: torch.Tensor,
    lower_bound: float | None,
    output_gate: torch.Tensor | None,
    norm_weight: torch.Tensor | None,
    norm_eps: float | None,
    prepared_weights: object | None,
) -> torch.Tensor:
    """Adapt the registered KDA contract to FlashInfer's fused operator."""
    del conv_weights, norm_weight
    if _fused_kda_decode is None:
        raise RuntimeError("FlashInfer fused KDA decode is unavailable")
    if read_indices is not write_indices:
        raise ValueError(
            "FlashInfer KDA decode requires copy-on-write staging and aliased "
            "read/write indices"
        )
    if output_gate is None or norm_eps is None:
        raise ValueError("FlashInfer fused KDA decode requires output normalization")
    if head_dim != _HEAD_DIM or num_heads not in _SUPPORTED_HEADS:
        raise ValueError("unsupported FlashInfer KDA head geometry")
    batch = mixed_qkv.shape[0]
    hidden_size = num_heads * head_dim
    if mixed_qkv.shape != (batch, 3 * hidden_size):
        raise ValueError("mixed_qkv must have shape [batch, 3 * H * 128]")
    if f_a_out.shape != (batch, head_dim):
        raise ValueError("f_a_out must have shape [batch, 128]")
    if f_b_weight.shape != (hidden_size, head_dim):
        raise ValueError("f_b_weight must have shape [H * 128, 128]")
    if beta_logits.shape != (batch, num_heads):
        raise ValueError("beta_logits must have shape [batch, H]")
    if cu_seqlens.numel() != batch + 1:
        raise ValueError("cu_seqlens must contain one boundary per decode row")
    if (
        conv_states.ndim != 3
        or conv_states.shape[1:] != (3 * hidden_size, 3)
        or conv_states.stride(1) != 1
        or conv_states.stride(2) != 3 * hidden_size
    ):
        raise ValueError(
            "conv_states must use the sequence-major [block, channel, tap] layout"
        )

    weights = _require_prepared_weights(prepared_weights)
    if (
        weights.conv.device != mixed_qkv.device
        or weights.norm.device != mixed_qkv.device
    ):
        raise ValueError("prepared KDA weights must be on the input device")
    raw_gate = torch.mm(f_a_out, f_b_weight.t()).view(1, batch, num_heads, head_dim)
    return _fused_kda_decode(
        x=mixed_qkv,
        weight=weights.conv,
        conv_state=conv_states,
        raw_gate=raw_gate,
        raw_beta=beta_logits.unsqueeze(0),
        A_log=A_log,
        dt_bias=dt_bias,
        # TokenSpeed bulk-copies distinct COW sources into their destinations
        # once per decode step. The public FlashInfer 0.6.18 kernel can then
        # safely read and update the destination slot in place.
        state_indices=write_indices,
        state=state_pool,
        output_gate=output_gate.view(batch, num_heads, head_dim),
        norm_weight=weights.norm,
        lower_bound=lower_bound,
        norm_eps=norm_eps,
    )


if _fused_kda_decode is not None:
    flashinfer_kda_fused_paged_decode = register_kernel(
        "attention",
        "kda_fused_paged_decode",
        name="flashinfer_kda_fused_paged_decode",
        solution="flashinfer",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            max_arch_version=ArchVersion(10, 3),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=format_signatures(("q", "k", "v"), "dense", {torch.bfloat16}),
        priority=Priority.SPECIALIZED + 1,
        traits={
            "paged_state": frozenset({True}),
            "fused_output_norm": frozenset({True}),
            "num_heads": _SUPPORTED_HEADS,
            "head_dim": frozenset({_HEAD_DIM}),
            "conv_kernel_size": frozenset({4}),
            "recurrent_layout": frozenset({"v_major"}),
            "prepared_weights": frozenset({True}),
            "staged_state": frozenset({True}),
        },
        tags={"nvidia", "paged_cache", "cuda_graph", "fusion"},
    )(_flashinfer_kda_fused_paged_decode)


__all__ = [
    "prepare_flashinfer_kda_decode_weights",
]
