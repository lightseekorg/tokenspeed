from __future__ import annotations

import os

import torch

_NATIVE_GLUON_SPARSE_ATTN = None
_NATIVE_GLUON_SPARSE_ATTN_IMPORT_ERROR: Exception | None = None
_NATIVE_GLUON_SPARSE_ATTN_LOGGED = False


def deepseek_v4_selected_attn_impl() -> str:
    """Return the configured DeepSeek-V4 prefill selected-attention impl.

    ``default`` preserves TokenSpeed's built-in kernel path. ``auto`` and
    ``gluon_sparse`` enable the native gfx950 Gluon sparse-attention path when
    shape guards pass.
    """

    impl = os.environ.get("TOKENSPEED_DSV4_SELECTED_ATTN_IMPL", "").strip().lower()
    if impl:
        return impl
    return "default"


def load_native_gluon_sparse_attn():
    """Load the native gfx950 Gluon sparse-attention wrapper on demand."""

    global _NATIVE_GLUON_SPARSE_ATTN, _NATIVE_GLUON_SPARSE_ATTN_IMPORT_ERROR
    if _NATIVE_GLUON_SPARSE_ATTN is not None:
        return _NATIVE_GLUON_SPARSE_ATTN
    if _NATIVE_GLUON_SPARSE_ATTN_IMPORT_ERROR is not None:
        raise _NATIVE_GLUON_SPARSE_ATTN_IMPORT_ERROR
    try:
        from tokenspeed.runtime.layers.attention.deepseek_v4.gluon_sparse_attn.sparse_attn import (
            sparse_attn,
        )

        _NATIVE_GLUON_SPARSE_ATTN = sparse_attn
        return _NATIVE_GLUON_SPARSE_ATTN
    except Exception as exc:  # pragma: no cover - exercised in integration runs.
        _NATIVE_GLUON_SPARSE_ATTN_IMPORT_ERROR = exc
        raise


def native_gluon_sparse_selected_attention(
    *,
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    lens: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor | None:
    """Adapter from TokenSpeed's selected-attention ABI to the native wrapper.

    The Gluon wrapper accepts dense ``[1, tokens, heads, 512]`` query and
    ``[1, rows, 512]`` KV tensors, while TokenSpeed's backend keeps
    ``[tokens, heads, 512]`` queries and a selected-attention ``indices/lens``
    ABI. This adapter reshapes the dense tensors and forwards ``lens`` through
    the wrapper's ``topk_lens`` argument.
    """

    impl = deepseek_v4_selected_attn_impl()
    if impl in {"default", "off", "false", "0"}:
        return None
    if impl not in {"gluon_sparse", "native_gluon", "auto"}:
        raise RuntimeError(
            "Unsupported TOKENSPEED_DSV4_SELECTED_ATTN_IMPL="
            f"{impl!r}; expected default, auto, or gluon_sparse"
        )
    if q.dim() != 3 or kv.dim() != 3 or indices.dim() != 2:
        return None
    if kv.shape[0] != 1 or q.shape[-1] != 512 or kv.shape[-1] != 512:
        return None
    if q.shape[1] not in (64, 128):
        return None
    if indices.shape[0] != q.shape[0] or lens.numel() != q.shape[0]:
        return None
    if indices.shape[1] < 128:
        return None
    min_tokens = int(
        os.environ.get("TOKENSPEED_DSV4_GLUON_SPARSE_ATTN_MIN_TOKENS", "8192")
    )
    if q.shape[0] < min_tokens:
        return None

    selected_width = int(indices.shape[1])
    topk_idxs = indices.to(dtype=torch.int32).contiguous()
    topk_lens = lens.reshape(1, -1).to(device=q.device, dtype=torch.int32).contiguous()

    global _NATIVE_GLUON_SPARSE_ATTN_LOGGED
    if not _NATIVE_GLUON_SPARSE_ATTN_LOGGED:
        label = os.environ.get(
            "TOKENSPEED_DSV4_GLUON_SPARSE_ATTN_LABEL",
            "native Gluon DSV4 sparse_attn",
        )
        print(
            f"[TokenSpeed] Using {label} "
            f"(q={tuple(q.shape)}, kv={tuple(kv.shape)}, "
            f"topk={selected_width}, has_lens=True)",
            flush=True,
        )
        _NATIVE_GLUON_SPARSE_ATTN_LOGGED = True

    sparse_attn = load_native_gluon_sparse_attn()
    q4 = q.unsqueeze(0).contiguous()
    kv4 = kv.contiguous()
    topk3 = topk_idxs.unsqueeze(0).contiguous()
    out4 = sparse_attn(q4, kv4, attn_sink, topk3, softmax_scale, topk_lens=topk_lens)
    return out4.squeeze(0).contiguous()
