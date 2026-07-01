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

"""TRT-LLM DeepSeek-V4 sparse-indexer preparation kernels."""

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures
from tokenspeed_kernel.thirdparty.cuda.trtllm_deepseek_v4_indexer import (
    has_trtllm_indexer_q_kernels,
    trtllm_fused_cat_fp4,
    trtllm_mla_rope_inplace,
)

try:
    from tokenspeed_kernel.thirdparty.fast_hadamard_transform import (
        hadamard_transform,
    )
except (ImportError, OSError):
    hadamard_transform = None

_HEAD_DIM = 128
_ROPE_DIM = 64
_MXFP4_VALUE_BYTES = _HEAD_DIM // 2
_BLACKWELL_CAPABILITY = CapabilityRequirement(
    min_arch_version=ArchVersion(10, 0),
    max_arch_version=ArchVersion(10, 9),
    vendors=frozenset({"nvidia"}),
)


def has_trtllm_deepseek_v4_indexer_q_prepare() -> bool:
    """Return whether the TRT-LLM Q preparation chain is importable."""

    return has_trtllm_indexer_q_kernels() and hadamard_transform is not None


def supports_trtllm_deepseek_v4_indexer_q_prepare(
    index_q: torch.Tensor,
    positions: torch.Tensor | None = None,
    cos_sin_cache: torch.Tensor | None = None,
) -> bool:
    """Return whether inputs satisfy the specialized SM100 kernel contract.

    Args:
        index_q: Candidate BF16 tensor shaped ``[tokens, 64, 128]``.
        positions: Optional INT32/INT64 position vector for full validation.
        cos_sin_cache: Optional contiguous FP32 RoPE cache for full validation.

    Returns:
        ``True`` when the vendored kernels, fast Hadamard transform, device,
        dtype, and production DeepSeek-V4 shape are all supported.
    """

    if (
        not has_trtllm_deepseek_v4_indexer_q_prepare()
        or not index_q.is_cuda
        or index_q.dtype != torch.bfloat16
        or index_q.ndim != 3
        or tuple(index_q.shape[1:]) != (64, _HEAD_DIM)
        or not index_q.is_contiguous()
        or index_q.data_ptr() % 16 != 0
    ):
        return False
    if positions is not None and (
        not positions.is_cuda
        or positions.device != index_q.device
        or positions.dtype not in (torch.int32, torch.int64)
        or positions.ndim != 1
        or positions.shape[0] != index_q.shape[0]
        or not positions.is_contiguous()
    ):
        return False
    if cos_sin_cache is not None and (
        not cos_sin_cache.is_cuda
        or cos_sin_cache.device != index_q.device
        or cos_sin_cache.dtype != torch.float32
        or cos_sin_cache.ndim != 2
        or cos_sin_cache.shape[1] != _ROPE_DIM
        or not cos_sin_cache.is_contiguous()
    ):
        return False
    device_index = index_q.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    major, _minor = torch.cuda.get_device_capability(device_index)
    return major == 10


def trtllm_deepseek_v4_indexer_q_prepare_mxfp4(
    *,
    index_q: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    weights: torch.Tensor,
    softmax_scale: float,
    head_scale: float,
    enable_pdl: bool = False,
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Prepare DeepGEMM MXFP4 indexer Q with vendored TRT-LLM kernels.

    The composite follows TRT-LLM's intended sequence: in-place interleaved
    MLA RoPE, normalized fast Hadamard transform, fused per-block MXFP4 pack,
    and FP32 index-head weight scaling. ``index_q`` is mutated by the RoPE op
    and must not be reused by the caller.

    Args:
        index_q: Contiguous BF16 ``[tokens, 64, 128]`` projection output.
        positions: INT32/INT64 absolute positions shaped ``[tokens]``.
        cos_sin_cache: FP32 RoPE table shaped ``[max_position, 64]``.
        weights: Index-head weights shaped ``[tokens, 64]`` or
            ``[tokens, 64, 1]``.
        softmax_scale: Indexer attention scale.
        head_scale: Per-head normalization scale.
        enable_pdl: Whether the RoPE launch may use PDL.

    Returns:
        ``((packed_q, packed_scales), scaled_weights)``. Packed Q is UINT8
        ``[tokens, 64, 64]``, scales are INT32 ``[tokens, 64]``, and weights
        are FP32 ``[tokens, 64]``.
    """

    if not supports_trtllm_deepseek_v4_indexer_q_prepare(
        index_q,
        positions,
        cos_sin_cache,
    ):
        raise RuntimeError(
            "TRT-LLM DeepSeek-V4 indexer Q preparation requires built kernels, "
            "fast Hadamard, SM100, and contiguous BF16 [tokens, 64, 128] input"
        )
    if positions.ndim != 1 or positions.shape[0] != index_q.shape[0]:
        raise ValueError(
            f"positions must be [{index_q.shape[0]}], got {tuple(positions.shape)}"
        )
    if cos_sin_cache.shape[-1] != _ROPE_DIM:
        raise ValueError(
            f"cos_sin_cache width must be {_ROPE_DIM}, got {cos_sin_cache.shape[-1]}"
        )
    if weights.ndim == 3:
        weights = weights.squeeze(-1)
    if weights.shape != index_q.shape[:2]:
        raise ValueError(
            f"weights must be {tuple(index_q.shape[:2])}, got {tuple(weights.shape)}"
        )

    num_tokens, num_heads, _head_dim = index_q.shape
    if num_tokens == 0:
        packed_q = torch.empty(
            (0, num_heads, _MXFP4_VALUE_BYTES),
            dtype=torch.uint8,
            device=index_q.device,
        )
        packed_scales = torch.empty(
            (0, num_heads), dtype=torch.int32, device=index_q.device
        )
        return (packed_q, packed_scales), weights.float()

    trtllm_mla_rope_inplace(
        index_q,
        positions,
        cos_sin_cache,
        enable_pdl=enable_pdl,
    )
    rotated = hadamard_transform(
        index_q.reshape(-1, _HEAD_DIM),
        scale=_HEAD_DIM**-0.5,
    )
    first, second = rotated.split((_ROPE_DIM, _HEAD_DIM - _ROPE_DIM), dim=-1)
    packed_q, packed_scales = trtllm_fused_cat_fp4(first, second)
    scaled_weights = weights.float() * float(softmax_scale * head_scale)
    return (
        packed_q.view(num_tokens, num_heads, _MXFP4_VALUE_BYTES),
        packed_scales.view(num_tokens, num_heads),
    ), scaled_weights


if has_trtllm_deepseek_v4_indexer_q_prepare() and current_platform().is_nvidia:
    trtllm_deepseek_v4_indexer_q_prepare_mxfp4 = register_kernel(
        "attention",
        "deepseek_v4_indexer_q_prepare_mxfp4",
        name="trtllm_deepseek_v4_indexer_q_prepare_mxfp4",
        solution="trtllm",
        capability=_BLACKWELL_CAPABILITY,
        signatures=format_signatures("index_q", "dense", {torch.bfloat16}),
        traits={
            "num_heads": frozenset({64}),
            "head_dim": frozenset({_HEAD_DIM}),
            "rope_dim": frozenset({_ROPE_DIM}),
        },
        priority=Priority.SPECIALIZED,
        tags={"latency", "throughput"},
    )(trtllm_deepseek_v4_indexer_q_prepare_mxfp4)
