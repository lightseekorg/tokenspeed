"""DeepSeek V4 family persistent top-k CUDA kernel wrapper."""

from __future__ import annotations

import functools
from pathlib import Path

import torch


def _objs_dir() -> Path:
    return Path(__file__).resolve().parent / "objs"


@functools.cache
def _load_dsv4_attention_module():
    so_path = _objs_dir() / "dsv4_attention" / "dsv4_attention.so"
    if not so_path.exists():
        raise RuntimeError(
            f"tokenspeed_kernel DeepSeek V4 attention library not found at {so_path}. "
            "Run `pip install -e tokenspeed-kernel/python/` to build."
        )
    import tvm_ffi

    return tvm_ffi.load_module(str(so_path))


@functools.cache
def persistent_topk_unavailable_reason() -> str | None:
    """Return a cached failure summary, or None when native top-k is available.

    Restart after repairing the install. Report only exception types here:
    raw loader errors can contain private deployment paths.
    """
    try:
        if not (_objs_dir() / "dsv4_attention" / "dsv4_attention.so").is_file():
            return "missing native library dsv4_attention.so"
        module = _load_dsv4_attention_module()
        if not hasattr(module, "dsv4_persistent_topk"):
            return "missing export dsv4_persistent_topk in dsv4_attention.so"
    except Exception as exc:
        return f"native module load failed ({type(exc).__name__})"
    return None


def has_persistent_topk() -> bool:
    return persistent_topk_unavailable_reason() is None


def persistent_topk(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    output: torch.Tensor,
    k: int,
    max_seq_len: int,
    q_len_per_req: int,
    *,
    enable_pdl: bool,
) -> None:
    """Select ragged row indices with optional CUDA launch dependencies.

    Args:
        logits: Contiguous FP32 scores shaped ``[rows, columns]``.
        lengths: Valid lengths per row, or per request with grouped queries.
        output: Int32 ``[rows, k]`` buffer, filled with indices and ``-1`` padding.
        k: Number of selected columns: 512, 1024 or 2048.
        max_seq_len: Maximum columns scanned by a row.
        q_len_per_req: Query rows per request; 1 selects per-row lengths.
        enable_pdl: Enable CUDA programmatic launch dependencies.

    Returns:
        None. Writes selected indices to ``output`` in place.
    """
    if logits.dtype != torch.float32:
        raise TypeError(f"logits must be float32, got {logits.dtype}")
    if lengths.dtype != torch.int32:
        lengths = lengths.to(torch.int32)
    if output.dtype != torch.int32:
        raise TypeError(f"output must be int32, got {output.dtype}")
    if not logits.is_contiguous():
        logits = logits.contiguous()
    if not lengths.is_contiguous():
        lengths = lengths.contiguous()
    _load_dsv4_attention_module().dsv4_persistent_topk(
        logits,
        lengths,
        output,
        int(k),
        int(max_seq_len),
        int(q_len_per_req),
        bool(enable_pdl),
    )
