"""MiniMax-M3 fused qk-norm + RoPE + KV/index-insert CUDA kernel wrapper.

Loads the vendored ``minimax_m3_fused`` TVM-FFI module and exposes a typed
wrapper around ``fused_minimax_m3_qknorm_rope_kv_insert``. The device kernel is
vLLM's; only the build/binding were ported to TokenSpeed.
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch
import tvm_ffi

# kv_cache_dtype int codes (must match the C++ Fp8KVCacheDataType enum).
_KV_DTYPE_CODE = {"auto": 0, "fp8_e4m3": 1, "fp8": 1, "fp8_e5m2": 2}


def _objs_dir() -> Path:
    return Path(__file__).resolve().parent / "objs"


@functools.cache
def _load_module():
    so_path = _objs_dir() / "minimax_m3_fused" / "minimax_m3_fused.so"
    if not so_path.exists():
        raise RuntimeError(
            f"tokenspeed_kernel minimax_m3_fused library not found at {so_path}. "
            "Run `pip install -e tokenspeed-kernel/python/` to build."
        )
    return tvm_ffi.load_module(str(so_path))


def fused_qknorm_rope_kv_insert(
    qkv: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    rotary_dim: int,
    eps: float,
    *,
    index_q_norm_weight: torch.Tensor | None = None,
    index_k_norm_weight: torch.Tensor | None = None,
    num_index_heads: int = 0,
    slot_mapping: torch.Tensor | None = None,
    index_slot_mapping: torch.Tensor | None = None,
    k_cache: torch.Tensor | None = None,
    v_cache: torch.Tensor | None = None,
    index_cache: torch.Tensor | None = None,
    block_size: int = 0,
    q_out: torch.Tensor | None = None,
    index_q_out: torch.Tensor | None = None,
    kv_cache_dtype: str = "auto",
    skip_index_branch: bool = False,
    enable_pdl: bool = False,
) -> None:
    """Fused Gemma qk-norm + partial-NeoX RoPE (main q/k + index q/k), with
    optional K/V and index-K cache scatter-insert.

    ``qkv`` is the fused ``[q | k | v | index_q | index_k]`` projection output
    (dense layers pass ``[q | k | v]`` with ``num_index_heads == 0``). q/k and,
    on the sparse path, index_q/index_k are rewritten in place; when given,
    contiguous ``q_out`` / ``index_q_out`` receive the de-interleaved outputs.
    Norm weights are the folded Gemma weights (``1 + w``). ``cos_sin_cache`` is
    ``[max_pos, rotary_dim]`` ([cos | sin] halves) in the qkv dtype.

    ``enable_pdl`` requests Programmatic Dependent Launch (SM90+); when False the
    kernel launches without the stream-serialization attribute (its in-body
    grid-dependency intrinsics become no-ops).
    """
    if qkv.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"qkv must be float16 or bfloat16, got {qkv.dtype}")
    code = _KV_DTYPE_CODE.get(kv_cache_dtype)
    if code is None:
        raise ValueError(f"unsupported kv_cache_dtype {kv_cache_dtype!r}")
    if positions.dtype != torch.int64:
        positions = positions.to(torch.int64)
    if slot_mapping is not None and slot_mapping.dtype != torch.int64:
        slot_mapping = slot_mapping.to(torch.int64)
    if index_slot_mapping is not None and index_slot_mapping.dtype != torch.int64:
        index_slot_mapping = index_slot_mapping.to(torch.int64)

    _load_module().fused_minimax_m3_qknorm_rope_kv_insert(
        qkv,
        q_norm_weight,
        k_norm_weight,
        cos_sin_cache.contiguous(),
        positions.contiguous(),
        int(num_heads),
        int(num_kv_heads),
        int(rotary_dim),
        float(eps),
        index_q_norm_weight,
        index_k_norm_weight,
        int(num_index_heads),
        None if slot_mapping is None else slot_mapping.contiguous(),
        None if index_slot_mapping is None else index_slot_mapping.contiguous(),
        k_cache,
        v_cache,
        index_cache,
        int(block_size),
        q_out,
        index_q_out,
        int(code),
        bool(skip_index_branch),
        bool(enable_pdl),
    )
