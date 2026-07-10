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

"""TRT-LLM-inspired DeepSeek-V4 attention preparation kernels."""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.transform import hadamard_transform
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures
from tokenspeed_kernel.thirdparty.cuda.trtllm_deepseek_v4_c128_prefill import (
    has_trtllm_deepseek_v4_c128_prefill_kernels,
    trtllm_deepseek_v4_c128_prefill_compress_cache_raw,
)
from tokenspeed_kernel.thirdparty.cuda.trtllm_deepseek_v4_indexer import (
    has_trtllm_indexer_q_kernels,
    trtllm_fused_cat_fp4,
    trtllm_mla_rope_inplace,
)

_HEAD_DIM = 128
_ROPE_DIM = 64
_MXFP4_VALUE_BYTES = _HEAD_DIM // 2
_C128_HEAD_DIM = 512
_C128_STATE_ROW_WIDTH = 2 * _C128_HEAD_DIM
_C128_CACHE_ROW_BYTES = 584
_BLACKWELL_CAPABILITY = CapabilityRequirement(
    min_arch_version=ArchVersion(10, 0),
    max_arch_version=ArchVersion(10, 9),
    vendors=frozenset({"nvidia"}),
)


def has_trtllm_deepseek_v4_indexer_q_prepare() -> bool:
    """Return whether the TRT-LLM Q preparation chain is importable."""

    return has_trtllm_indexer_q_kernels()


def has_trtllm_deepseek_v4_c128_prefill_compress_cache() -> bool:
    """Return whether the native C128 pure-prefill compressor is built."""

    return has_trtllm_deepseek_v4_c128_prefill_kernels()


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


def supports_trtllm_deepseek_v4_c128_prefill_compress_cache(
    state_cache: torch.Tensor,
    scratch: torch.Tensor,
    positions: torch.Tensor,
    compressor_slot_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    rms_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    *,
    block_table_base_offsets: torch.Tensor | None = None,
    state_block_size: int,
    kv_block_size: int,
    max_outputs: int,
) -> bool:
    """Return whether inputs satisfy the SM100 C128 pure-prefill contract.

    This predicate is intentionally strict: the launch path never allocates,
    converts dtypes, or makes a tensor contiguous on behalf of the caller.

    Args:
        state_cache: FP32 combined state pages shaped
            ``[blocks, state_block_size, 1024]``.
        scratch: Mutable contiguous FP32 projection scratch shaped
            ``[tokens, >=512]``.
        positions: Contiguous INT64 absolute positions shaped ``[tokens]``.
        compressor_slot_mapping: Contiguous INT64 compressor-state slots
            shaped ``[tokens]``. Negative boundary slots suppress outputs.
        query_start_loc: Contiguous INT32 packed offsets shaped ``[batch+1]``.
        seq_lens: Contiguous INT32 final sequence lengths shaped ``[batch]``.
        block_table: Contiguous INT32 compact state table shaped
            ``[batch, width]``.
        rms_norm_weight: Contiguous BF16 or FP32 weights shaped ``[512]``.
        cos_sin_cache: Contiguous FP32 RoPE table shaped ``[maxpos,64]``. It
            must cover all referenced 128-aligned compressed positions.
        kv_cache: Mutable contiguous UINT8 fp8_ds_mla pages.
        kv_slot_mapping: Contiguous INT64 compressed slots shaped ``[tokens]``.
        block_table_base_offsets: Optional contiguous INT32 logical page bases
            shaped ``[batch]``.
        state_block_size: Number of rows per state page.
        kv_block_size: Number of compressed rows per KV page.
        max_outputs: Positive per-request launch bound, at most 65535, that
            must cover every C128 output in the batch; an undersized bound
            skips outputs.

    Returns:
        ``True`` only when the native op, SM100-family device, layouts, dtypes,
        shapes, and scalar launch parameters are all supported.
    """

    if not has_trtllm_deepseek_v4_c128_prefill_compress_cache():
        return False
    tensors = (
        state_cache,
        scratch,
        positions,
        compressor_slot_mapping,
        query_start_loc,
        seq_lens,
        block_table,
        rms_norm_weight,
        cos_sin_cache,
        kv_cache,
        kv_slot_mapping,
    )
    if any(
        not tensor.is_cuda
        or tensor.device != state_cache.device
        or not tensor.is_contiguous()
        for tensor in tensors
    ):
        return False
    if block_table_base_offsets is not None and (
        not block_table_base_offsets.is_cuda
        or block_table_base_offsets.device != state_cache.device
        or not block_table_base_offsets.is_contiguous()
        or block_table_base_offsets.dtype != torch.int32
        or block_table_base_offsets.ndim != 1
    ):
        return False
    if (
        state_cache.dtype != torch.float32
        or state_cache.ndim != 3
        or state_cache.shape[1] != state_block_size
        or state_cache.shape[2] != _C128_STATE_ROW_WIDTH
        or state_cache.data_ptr() % 16 != 0
        or state_block_size <= 0
        or kv_block_size <= 0
        or max_outputs <= 0
        or max_outputs > 65535
    ):
        return False
    if (
        scratch.dtype != torch.float32
        or scratch.ndim != 2
        or scratch.shape[0] == 0
        or scratch.shape[1] < _C128_HEAD_DIM
        or scratch.stride(0) % 4 != 0
        or scratch.data_ptr() % 16 != 0
        or positions.dtype != torch.int64
        or positions.ndim != 1
        or positions.shape[0] != scratch.shape[0]
        or compressor_slot_mapping.dtype != torch.int64
        or compressor_slot_mapping.ndim != 1
        or compressor_slot_mapping.shape[0] != scratch.shape[0]
        or kv_slot_mapping.dtype != torch.int64
        or kv_slot_mapping.ndim != 1
        or kv_slot_mapping.shape[0] != scratch.shape[0]
    ):
        return False
    batch_size = seq_lens.shape[0] if seq_lens.ndim == 1 else 0
    if (
        batch_size == 0
        or seq_lens.dtype != torch.int32
        or query_start_loc.dtype != torch.int32
        or query_start_loc.ndim != 1
        or query_start_loc.shape[0] != batch_size + 1
        or block_table.dtype != torch.int32
        or block_table.ndim != 2
        or block_table.shape[0] != batch_size
        or block_table.shape[1] == 0
        or (
            block_table_base_offsets is not None
            and block_table_base_offsets.shape[0] != batch_size
        )
    ):
        return False
    if (
        rms_norm_weight.dtype not in (torch.bfloat16, torch.float32)
        or rms_norm_weight.ndim != 1
        or rms_norm_weight.shape[0] != _C128_HEAD_DIM
        or cos_sin_cache.dtype != torch.float32
        or cos_sin_cache.ndim != 2
        or cos_sin_cache.shape[0] == 0
        or cos_sin_cache.shape[1] != _ROPE_DIM
        or kv_cache.dtype != torch.uint8
        or kv_cache.ndim != 2
        or kv_cache.shape[1] < kv_block_size * _C128_CACHE_ROW_BYTES
        or kv_cache.stride(0) % 16 != 0
        or kv_cache.data_ptr() % 16 != 0
    ):
        return False
    device_index = state_cache.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    major, _minor = torch.cuda.get_device_capability(device_index)
    return major == 10


def trtllm_deepseek_v4_c128_prefill_compress_cache(
    *,
    state_cache: torch.Tensor,
    scratch: torch.Tensor,
    positions: torch.Tensor,
    compressor_slot_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    rms_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    block_table_base_offsets: torch.Tensor | None = None,
    state_block_size: int,
    kv_block_size: int,
    max_outputs: int,
    rms_norm_eps: float,
) -> None:
    """Compress complete C128 prefill windows and scatter fp8_ds_mla rows.

    The first launch reduces already-saved FP32 compressor-state rows with a
    four-group online softmax and writes each FP32 result into the first 512
    columns of its boundary row in ``scratch``. The second launch applies
    RMSNorm, BF16-rounded 64-value E4M3 quantization for the 448 NoPE values,
    interleaved RoPE for the 64 tail values, and paged cache scatter.

    Args:
        state_cache: FP32 ``[blocks, state_block_size, 1024]`` combined
            ``[kv | score+APE]`` state cache.
        scratch: Mutable contiguous FP32 ``[tokens, >=512]`` buffer. Only
            boundary rows in ``:512`` are overwritten.
        positions: Contiguous INT64 absolute positions, ``[tokens]``.
        compressor_slot_mapping: Contiguous INT64 compressor-state slots,
            ``[tokens]``. Negative boundary slots suppress outputs.
        query_start_loc: Contiguous INT32 packed offsets, ``[batch+1]``.
        seq_lens: Contiguous INT32 final sequence lengths, ``[batch]``.
        block_table: Contiguous INT32 compact state page table.
        rms_norm_weight: Contiguous BF16 or FP32 RMSNorm weights, ``[512]``.
        cos_sin_cache: Contiguous FP32 RoPE table, ``[maxpos,64]``. The table
            must cover every referenced 128-aligned compressed position; this
            device-resident bound is a caller precondition.
        kv_cache: Mutable contiguous UINT8 fp8_ds_mla pages.
        kv_slot_mapping: Contiguous INT64 compressed output slots, ``[tokens]``.
        block_table_base_offsets: Optional contiguous INT32 logical-page bases.
        state_block_size: State page size in token rows.
        kv_block_size: Compressed KV page size in rows.
        max_outputs: Positive per-request output launch bound, at most 65535,
            that must cover every C128 output in the batch; an undersized bound
            skips outputs.
        rms_norm_eps: RMSNorm epsilon.

    Returns:
        ``None``. ``scratch`` and ``kv_cache`` are mutated in place.

    Raises:
        RuntimeError: If the native op, device, or strict tensor contract is
            unsupported. Launch errors propagate; there is no post-launch
            fallback.
    """

    if not supports_trtllm_deepseek_v4_c128_prefill_compress_cache(
        state_cache,
        scratch,
        positions,
        compressor_slot_mapping,
        query_start_loc,
        seq_lens,
        block_table,
        rms_norm_weight,
        cos_sin_cache,
        kv_cache,
        kv_slot_mapping,
        block_table_base_offsets=block_table_base_offsets,
        state_block_size=state_block_size,
        kv_block_size=kv_block_size,
        max_outputs=max_outputs,
    ):
        raise RuntimeError(
            "TRT-inspired DeepSeek-V4 C128 prefill compression requires the "
            "built native op, an SM100-family GPU, and exact contiguous "
            "FP32/INT64/INT32/BF16-or-FP32/UINT8 tensor layouts"
        )
    if rms_norm_eps < 0.0:
        raise ValueError(f"rms_norm_eps must be non-negative, got {rms_norm_eps}")
    trtllm_deepseek_v4_c128_prefill_compress_cache_raw(
        state_cache=state_cache,
        scratch=scratch,
        positions=positions,
        compressor_slot_mapping=compressor_slot_mapping,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        block_table=block_table,
        rms_norm_weight=rms_norm_weight,
        cos_sin_cache=cos_sin_cache,
        kv_cache=kv_cache,
        kv_slot_mapping=kv_slot_mapping,
        block_table_base_offsets=block_table_base_offsets,
        state_block_size=state_block_size,
        kv_block_size=kv_block_size,
        max_outputs=max_outputs,
        rms_norm_eps=rms_norm_eps,
    )


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


if (
    has_trtllm_deepseek_v4_c128_prefill_compress_cache()
    and current_platform().is_nvidia
):
    trtllm_deepseek_v4_c128_prefill_compress_cache = register_kernel(
        "attention",
        "deepseek_v4_c128_prefill_compress_cache",
        name="trtllm_deepseek_v4_c128_prefill_compress_cache",
        solution="trtllm",
        capability=_BLACKWELL_CAPABILITY,
        signatures=format_signatures("scratch", "dense", {torch.float32}),
        traits={
            "compress_ratio": frozenset({128}),
            "head_dim": frozenset({_C128_HEAD_DIM}),
        },
        priority=Priority.SPECIALIZED,
        tags={"latency", "throughput"},
    )(trtllm_deepseek_v4_c128_prefill_compress_cache)
