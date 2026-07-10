"""DeepSeek V4 attention CUDA kernel wrappers."""

from __future__ import annotations

import functools
from pathlib import Path

import torch
import tvm_ffi


def _objs_dir() -> Path:
    return Path(__file__).resolve().parent / "objs"


@functools.cache
def _load_deepseek_v4_attention_module():
    so_path = _objs_dir() / "deepseek_v4_attention" / "deepseek_v4_attention.so"
    if not so_path.exists():
        raise RuntimeError(
            f"tokenspeed_kernel DeepSeek V4 attention library not found at {so_path}. "
            "Run `pip install -e tokenspeed-kernel/python/` to build."
        )
    return tvm_ffi.load_module(str(so_path))


def has_fused_qnorm_rope_kv_insert() -> bool:
    try:
        module = _load_deepseek_v4_attention_module()
    except Exception:
        return False
    return hasattr(module, "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert")


def has_fused_qnorm_rope_kv_insert_padded() -> bool:
    """Return whether the optional padded-Q producer symbol is available."""

    try:
        module = _load_deepseek_v4_attention_module()
    except Exception:
        return False
    return all(
        hasattr(module, name)
        for name in (
            "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert",
            "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_padded",
        )
    )


def has_indexer_topk_prefill() -> bool:
    try:
        module = _load_deepseek_v4_attention_module()
    except Exception:
        return False
    return hasattr(module, "deepseek_v4_indexer_topk_prefill")


def has_indexer_mxfp4_paged_gather() -> bool:
    try:
        module = _load_deepseek_v4_attention_module()
    except Exception:
        return False
    return hasattr(module, "deepseek_v4_gather_paged_indexer_mxfp4_cache")


def has_persistent_topk() -> bool:
    try:
        module = _load_deepseek_v4_attention_module()
    except Exception:
        return False
    return hasattr(module, "deepseek_v4_persistent_topk")


def fused_qnorm_rope_kv_insert(
    q: torch.Tensor,
    kv: torch.Tensor,
    k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rms_norm_eps: float,
    block_size: int,
    enable_pdl: bool = False,
    q_out: torch.Tensor | None = None,
) -> None:
    """Normalize Q and insert the normalized/quantized KV into the SWA cache.

    Args:
        q: Contiguous ``[tokens, local_heads, 512]`` FP16 or BF16 input. It is
            normalized and rotated in place when ``q_out`` is omitted, and is
            preserved when ``q_out`` is provided.
        kv: Contiguous ``[tokens, 512]`` input with the same dtype as ``q``.
        k_cache: UINT8 paged SWA cache flattened to ``[blocks, block_bytes]``.
        slot_mapping: Cache slot for each KV row to insert.
        positions: Absolute position for every Q/KV row.
        cos_sin_cache: FP32 rotary embedding cache with width 64.
        rms_norm_eps: Epsilon used by RMS normalization.
        block_size: Number of cache slots per block.
        enable_pdl: Whether to enable programmatic dependent launch.
        q_out: Optional contiguous ``[tokens, padded_heads, 512]`` output. The
            active heads contain normalized/rotated Q, the padded tail is
            bitwise +0, and the tensor must not overlap any input tensor.

    Returns:
        None.
    """
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"q must be float16 or bfloat16, got {q.dtype}")
    if kv.dtype != q.dtype:
        raise TypeError(f"kv dtype {kv.dtype} must match q dtype {q.dtype}")
    if k_cache.dtype != torch.uint8:
        raise TypeError(f"k_cache must be uint8, got {k_cache.dtype}")
    if cos_sin_cache.dtype != torch.float32:
        raise TypeError(f"cos_sin_cache must be float32, got {cos_sin_cache.dtype}")
    if q_out is not None:
        if q_out.dtype != q.dtype:
            raise TypeError(f"q_out dtype {q_out.dtype} must match q dtype {q.dtype}")
        if q_out.device != q.device:
            raise ValueError("q_out must be on the same device as q")
        if q_out.ndim != 3 or q_out.shape[0] != q.shape[0]:
            raise ValueError("q_out must be [tokens, padded_heads, head_dim]")
        if q_out.shape[1] < q.shape[1] or q_out.shape[2] != q.shape[2]:
            raise ValueError("q_out must preserve all q heads and the q head_dim")
        if not q_out.is_contiguous():
            raise ValueError("q_out must be contiguous")
    if slot_mapping.dtype != torch.int64:
        slot_mapping = slot_mapping.to(torch.int64)
    if positions.dtype != torch.int64:
        positions = positions.to(torch.int64)

    module = _load_deepseek_v4_attention_module()
    args = (
        q,
        kv,
        k_cache,
        slot_mapping.contiguous(),
        positions.contiguous(),
        cos_sin_cache.contiguous(),
        float(rms_norm_eps),
        int(block_size),
        bool(enable_pdl),
    )
    if q_out is None:
        module.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(*args)
        return
    if not hasattr(module, "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_padded"):
        raise RuntimeError(
            "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_padded is "
            "unavailable; rebuild tokenspeed-kernel"
        )
    module.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_padded(
        q,
        q_out,
        *args[1:],
    )


def indexer_topk_prefill(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    output: torch.Tensor,
    k: int,
) -> None:
    if logits.dtype != torch.float32:
        raise TypeError(f"logits must be float32, got {logits.dtype}")
    if row_starts.dtype != torch.int32:
        row_starts = row_starts.to(torch.int32)
    if row_ends.dtype != torch.int32:
        row_ends = row_ends.to(torch.int32)
    if output.dtype != torch.int32:
        raise TypeError(f"output must be int32, got {output.dtype}")
    _load_deepseek_v4_attention_module().deepseek_v4_indexer_topk_prefill(
        logits.contiguous(),
        row_starts.contiguous(),
        row_ends.contiguous(),
        output,
        int(k),
    )


def indexer_mxfp4_paged_gather(
    kv_cache: torch.Tensor,
    values_out: torch.Tensor,
    scales_out: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    cache_block_size: int,
) -> None:
    if kv_cache.dtype != torch.uint8:
        raise TypeError(f"kv_cache must be uint8, got {kv_cache.dtype}")
    if values_out.dtype != torch.uint8:
        raise TypeError(f"values_out must be uint8, got {values_out.dtype}")
    if scales_out.dtype != torch.uint8:
        raise TypeError(f"scales_out must be uint8, got {scales_out.dtype}")
    if block_table.dtype != torch.int32:
        block_table = block_table.to(torch.int32)
    if cu_seq_lens.dtype != torch.int32:
        cu_seq_lens = cu_seq_lens.to(torch.int32)
    if values_out.shape[0] != scales_out.shape[0]:
        raise ValueError(
            "DeepSeek V4 paged gather output value/scale rows must match, "
            f"got values={values_out.shape[0]}, scales={scales_out.shape[0]}"
        )
    _load_deepseek_v4_attention_module().deepseek_v4_gather_paged_indexer_mxfp4_cache(
        kv_cache,
        values_out,
        scales_out,
        block_table.contiguous(),
        cu_seq_lens.contiguous(),
        int(cache_block_size),
    )


def persistent_topk(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    output: torch.Tensor,
    workspace: torch.Tensor,
    k: int,
    max_seq_len: int,
) -> None:
    if logits.dtype != torch.float32:
        raise TypeError(f"logits must be float32, got {logits.dtype}")
    if lengths.dtype != torch.int32:
        lengths = lengths.to(torch.int32)
    if output.dtype != torch.int32:
        raise TypeError(f"output must be int32, got {output.dtype}")
    if workspace.dtype != torch.uint8:
        raise TypeError(f"workspace must be uint8, got {workspace.dtype}")
    if not logits.is_contiguous():
        logits = logits.contiguous()
    if not lengths.is_contiguous():
        lengths = lengths.contiguous()
    _load_deepseek_v4_attention_module().deepseek_v4_persistent_topk(
        logits,
        lengths,
        output,
        workspace,
        int(k),
        int(max_seq_len),
    )
