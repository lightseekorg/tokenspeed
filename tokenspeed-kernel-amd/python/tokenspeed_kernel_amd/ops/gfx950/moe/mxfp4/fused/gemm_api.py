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

"""Per-GEMM public entry points: dispatch GEMM with fused SwiGLU, combine
GEMM with scatter/gate scaling, and the ragged-matmul router over both."""

from __future__ import annotations

from typing import Optional

import torch
from tokenspeed_kernel_amd.ops.gfx950.moe._common import (
    RaggedTensorMetadata,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused._common import (
    _CDNA4_NUM_CUS,
    _NON_K_PRESHUFFLE_BLOCK_SIZE,
    _PERSISTENT_TILES_THRESHOLD,
    _extract_gluon_raw_s,
    _extract_gluon_raw_w,
    _extract_gluon_raw_w_unshuffled,
    _global_scale_passthrough,
    _maybe_extract_swiglu_args,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused._layouts import (
    _moe_partial_reduce,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused.launch import (
    _dense_grid_dims,
    _launch_kernel,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused.tuning import (
    _align_block_n_to_preshuffled_layout,
    _autotune_block,
    _is_single_k_tile,
    _prefill_launch_tuning,
    _ragged_slice_size,
    _resolve_prefill_slice_modes,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.scale_layout import (
    MXFP4_BLOCK,
    empty_swizzled_cdna4_mxfp4_scale,
)


def gluon_mxfp_dispatch_swiglu(
    x: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    *,
    x_scale: torch.Tensor | None = None,
    x_format: str = "e2m1",
    x_global_scale: torch.Tensor | float = 1.0,
    bias: torch.Tensor | None,
    a_ragged_metadata,
    gather_indx,
    out_dtype: torch.dtype = torch.bfloat16,
    swiglu_alpha: float = 1.0,
    swiglu_limit: float = 0.0,
    swiglu_beta: float = 1.0,
    block_m: int | None = None,
    block_n: int | None = None,
    block_k: int | None = None,
    num_warps: int | None = None,
    num_buffers: int = 2,
    use_warp_pipeline: bool | None = None,
    use_slice_mn: bool | None = None,
    use_slice_n: bool | None = None,
    scale_load_mode: str = "transpose",
    w_transpose: bool = False,
    persistent: bool | None = None,
    num_ctas: int | None = None,
    out_quant_scale: torch.Tensor | float | None = None,
    out_quant_format: str | None = None,
    w_preshuffle: bool = False,
    x_scale_ragged_padded: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    assert w.ndim == 3 and w.shape[-1] % 2 == 0
    M_X = int(x.shape[-2])
    if gather_indx is not None:
        gather_t = (
            gather_indx.src_indx if hasattr(gather_indx, "src_indx") else gather_indx
        )
        M = int(gather_t.shape[0])
    else:
        M = x.shape[-2]
    N = w.shape[-1]
    div_x = 2 if x_format == "e2m1" else 1
    K = x.shape[-1] * div_x
    requested_block_m = block_m
    requested_block_n = block_n
    requested_block_k = block_k
    slice_size = _ragged_slice_size(a_ragged_metadata, M)
    (
        block_m,
        block_n,
        block_k,
        nw,
        use_slice_n,
        use_small_prefill_m,
    ) = _autotune_block(
        M,
        N,
        K,
        do_swiglu=True,
        x_format=x_format,
        scale_load_mode=scale_load_mode,
        slice_size=slice_size,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        use_slice_n=use_slice_n,
        large_slice_size=128,
        large_m=16384,
    )
    if out_quant_format not in (None, "mxfp4"):
        raise ValueError(f"unsupported out_quant_format={out_quant_format!r}")
    has_mxfp4_quant_out = out_quant_format == "mxfp4"
    if has_mxfp4_quant_out and out_quant_scale is not None:
        raise ValueError("MXFP4 output quantization does not use out_quant_scale")

    medium_decode_dispatch_shape = (
        8 <= M_X <= 16
        and a_ragged_metadata is not None
        and gather_indx is not None
        and out_quant_scale is not None
        and x_format == "e4m3"
        and x_scale is None
        and scale_load_mode == "swizzle"
        and w_transpose
        and requested_block_m is None
        and requested_block_n is None
        and requested_block_k is None
        and not w_preshuffle
        and N % 128 == 0
    )
    if medium_decode_dispatch_shape:
        block_m, block_n, block_k, nw = 16, 128, 256, 4
        use_slice_n = False
        use_slice_mn = False
        use_small_prefill_m = False
    if (
        use_slice_n is None
        and x_format == "e2m1"
        and scale_load_mode == "swizzle"
        and a_ragged_metadata is not None
    ):
        use_slice_n = False
    if _is_single_k_tile(K, block_k):
        use_slice_n = False
        use_slice_mn = False
    if w_preshuffle:
        block_n, use_slice_mn, use_slice_n = _align_block_n_to_preshuffled_layout(
            w,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            scale_load_mode=scale_load_mode,
            x_format=x_format,
            has_x_block_scale=x_format == "e2m1",
            has_w_block_scale=True,
            use_slice_mn=use_slice_mn,
            use_slice_n=use_slice_n,
        )
    num_warps = num_warps or nw
    if w_preshuffle:
        num_warps = 4
    use_warp_pipeline = (
        bool(use_warp_pipeline) if use_warp_pipeline is not None else False
    )
    use_slice_mn, use_slice_n = _resolve_prefill_slice_modes(
        use_slice_mn=use_slice_mn,
        use_slice_n=use_slice_n,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_buffers=num_buffers,
        scale_load_mode=scale_load_mode,
        x_format=x_format,
        has_x_block_scale=x_format == "e2m1",
        has_w_block_scale=True,
    )
    medium_decode_dispatch_eligible = (
        medium_decode_dispatch_shape
        and block_m == 16
        and block_n == 128
        and block_k == 256
        and not use_slice_mn
        and not use_slice_n
        and not w_preshuffle
    )
    if persistent is None and use_small_prefill_m:
        persistent = False
    if persistent is None and use_slice_n:
        grid_n = (N + block_n - 1) // block_n
        if a_ragged_metadata is not None:
            n_slices = int(a_ragged_metadata.slice_sizes.shape[0])
            grid_m_upper = RaggedTensorMetadata.n_blocks(n_slices, M, block_m)
        else:
            grid_m_upper, _ = _dense_grid_dims(M, block_m)
        persistent = (grid_m_upper * grid_n) >= _PERSISTENT_TILES_THRESHOLD
        if persistent and num_ctas is None:
            num_ctas = _CDNA4_NUM_CUS
    group_m, xcd_swizzle, w_cache_cg, _ = _prefill_launch_tuning(
        "dispatch",
        m=M,
        use_slice_mn=use_slice_mn,
    )
    out_block_n = block_n // 2
    if has_mxfp4_quant_out:
        out_features = N // 2
        y = torch.empty((M, out_features // 2), device=x.device, dtype=torch.uint8)
        if a_ragged_metadata is not None:
            n_slices = int(a_ragged_metadata.slice_sizes.shape[0])
            scale_rows = (
                RaggedTensorMetadata.n_blocks(
                    n_slices,
                    M,
                    _NON_K_PRESHUFFLE_BLOCK_SIZE,
                )
                * _NON_K_PRESHUFFLE_BLOCK_SIZE
            )
        else:
            scale_rows = M
        y_mx_scale = empty_swizzled_cdna4_mxfp4_scale(
            scale_rows,
            out_features // MXFP4_BLOCK,
            device=x.device,
        )
    else:
        y_dtype = torch.float8_e4m3fn if out_quant_scale is not None else out_dtype
        y = torch.empty((M, N // 2), device=x.device, dtype=y_dtype)
        y_mx_scale = None
    _launch_kernel(
        x,
        w,
        y=y,
        bias=bias,
        gather_indx=gather_indx,
        scatter_indx=None,
        gate_scal=None,
        a_ragged_metadata=a_ragged_metadata,
        swiglu=(float(swiglu_alpha), float(swiglu_limit), float(swiglu_beta)),
        out_block_n=out_block_n,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        x_format=x_format,
        w_format="e2m1",
        x_scale=x_scale,
        w_scale=w_scale,
        x_global_scale=x_global_scale,
        scale_load_mode=scale_load_mode,
        w_transpose=w_transpose,
        num_buffers=num_buffers,
        use_warp_pipeline=use_warp_pipeline,
        use_slice_mn=use_slice_mn,
        use_slice_n=use_slice_n,
        persistent=persistent,
        num_ctas=num_ctas,
        group_m=group_m,
        xcd_swizzle=xcd_swizzle,
        out_quant_scale=out_quant_scale,
        out_mx_scale=y_mx_scale,
        w_preshuffle=w_preshuffle,
        w_cache_cg=w_cache_cg,
        medium_decode_dispatch=medium_decode_dispatch_eligible,
        x_scale_ragged_padded=x_scale_ragged_padded,
    )
    if has_mxfp4_quant_out:
        return y, y_mx_scale
    return y


def gluon_mxfp_combine(
    x: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    *,
    x_scale: torch.Tensor | None = None,
    x_format: str = "e2m1",
    x_global_scale: torch.Tensor | float = 1.0,
    bias: torch.Tensor | None,
    a_ragged_metadata,
    scatter_indx,
    gate_scal: torch.Tensor | None = None,
    n_tokens: int | None = None,
    n_expts_act: int | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
    block_m: int | None = None,
    block_n: int | None = None,
    block_k: int | None = None,
    num_warps: int | None = None,
    num_buffers: int = 2,
    use_warp_pipeline: bool | None = None,
    use_slice_mn: bool | None = None,
    use_slice_n: bool | None = None,
    scale_load_mode: str = "transpose",
    w_transpose: bool = False,
    persistent: bool | None = None,
    num_ctas: int | None = None,
    w_preshuffle: bool = False,
    x_scale_ragged_padded: bool = False,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    assert w.ndim == 3
    M = x.shape[-2]
    N = w.shape[-1]
    div_x = 2 if x_format == "e2m1" else 1
    K = x.shape[-1] * div_x
    requested_block_m = block_m
    requested_block_n = block_n
    requested_block_k = block_k
    slice_size = _ragged_slice_size(a_ragged_metadata, M)
    (
        block_m,
        block_n,
        block_k,
        nw,
        use_slice_n,
        use_small_prefill_m,
    ) = _autotune_block(
        M,
        N,
        K,
        ragged=a_ragged_metadata is not None,
        x_format=x_format,
        scale_load_mode=scale_load_mode,
        slice_size=slice_size,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        use_slice_n=use_slice_n,
        large_slice_size=256,
        large_m=32768,
    )
    medium_decode_combine_shape = (
        a_ragged_metadata is not None
        and scatter_indx is not None
        and gate_scal is not None
        and n_tokens in (8, 16)
        and n_expts_act is not None
        and x_format == "e4m3"
        and x_scale is None
        and scale_load_mode == "swizzle"
        and w_transpose
        and requested_block_m is None
        and requested_block_n is None
        and requested_block_k is None
        and not w_preshuffle
        and N % 128 == 0
    )
    if medium_decode_combine_shape:
        block_m, block_n, block_k, nw = 16, 128, 256, 4
        use_slice_n = False
        use_slice_mn = False
        use_small_prefill_m = False
    if _is_single_k_tile(K, block_k):
        use_slice_n = False
        use_slice_mn = False
    if w_preshuffle:
        block_n, use_slice_mn, use_slice_n = _align_block_n_to_preshuffled_layout(
            w,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            scale_load_mode=scale_load_mode,
            x_format=x_format,
            has_x_block_scale=x_format == "e2m1",
            has_w_block_scale=True,
            use_slice_mn=use_slice_mn,
            use_slice_n=use_slice_n,
        )
    num_warps = num_warps or nw
    if w_preshuffle:
        num_warps = 4
    use_warp_pipeline = (
        bool(use_warp_pipeline) if use_warp_pipeline is not None else False
    )
    use_slice_mn, use_slice_n = _resolve_prefill_slice_modes(
        use_slice_mn=use_slice_mn,
        use_slice_n=use_slice_n,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_buffers=num_buffers,
        scale_load_mode=scale_load_mode,
        x_format=x_format,
        has_x_block_scale=x_format == "e2m1",
        has_w_block_scale=True,
    )
    medium_decode_combine_eligible = (
        medium_decode_combine_shape
        and block_m == 16
        and block_n == 128
        and block_k == 256
        and not use_slice_mn
        and not use_slice_n
        and not w_preshuffle
    )
    if persistent is None and use_small_prefill_m:
        persistent = False
    if persistent is None:
        grid_n = (N + block_n - 1) // block_n
        if a_ragged_metadata is not None:
            n_slices = int(a_ragged_metadata.slice_sizes.shape[0])
            grid_m_upper = RaggedTensorMetadata.n_blocks(n_slices, M, block_m)
        else:
            grid_m_upper, _ = _dense_grid_dims(M, block_m)
        persistent = (grid_m_upper * grid_n) >= _PERSISTENT_TILES_THRESHOLD
        if persistent and num_ctas is None:
            num_ctas = _CDNA4_NUM_CUS
    group_m, xcd_swizzle, w_cache_cg, use_narrow_n_store_layout = (
        _prefill_launch_tuning(
            "combine",
            m=M,
            use_slice_mn=use_slice_mn,
        )
    )
    n_act_eff = int(n_expts_act) if n_expts_act is not None else 1
    if n_tokens is None:
        n_rows = M
        n_tokens_eff = M
    else:
        n_tokens_eff = int(n_tokens)
        n_rows = n_tokens_eff * n_act_eff
    # W may be padded in N to satisfy the packed layout. Keep padded N for
    # tiling/W-scale reads, but store only the caller-visible width.
    logical_n = int(getattr(w, "original_n", N))
    y_n = logical_n if logical_n < N else N
    expected_out_shape = (n_tokens_eff, y_n)
    if out is not None and (
        out.shape != expected_out_shape
        or out.dtype != out_dtype
        or out.device != x.device
        or not out.is_contiguous()
    ):
        raise ValueError(
            f"MXFP4 combine output must be contiguous {expected_out_shape} "
            f"{out_dtype} on {x.device}, got {tuple(out.shape)} {out.dtype} "
            f"on {out.device}"
        )
    y = (
        out
        if out is not None and n_act_eff == 1
        else torch.empty((n_rows, y_n), device=x.device, dtype=out_dtype)
    )
    # GEMM2 X is already in expert-sorted ragged order. Store through
    # scatter_indx to recover flat token/top-k row order before reduction.
    _launch_kernel(
        x,
        w,
        y=y,
        bias=bias,
        gather_indx=None,
        scatter_indx=scatter_indx,
        gate_scal=gate_scal,
        a_ragged_metadata=a_ragged_metadata,
        swiglu=None,
        out_block_n=block_n,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        x_format=x_format,
        w_format="e2m1",
        x_scale=x_scale,
        w_scale=w_scale,
        x_global_scale=x_global_scale,
        scale_load_mode=scale_load_mode,
        w_transpose=w_transpose,
        num_buffers=num_buffers,
        use_warp_pipeline=use_warp_pipeline,
        use_slice_mn=use_slice_mn,
        use_slice_n=use_slice_n,
        persistent=persistent,
        num_ctas=num_ctas,
        group_m=group_m,
        xcd_swizzle=xcd_swizzle,
        w_preshuffle=w_preshuffle,
        y_n_const=y_n if y_n != N else 0,
        w_cache_cg=w_cache_cg,
        use_narrow_n_store_layout=use_narrow_n_store_layout,
        medium_decode_combine=medium_decode_combine_eligible,
        x_scale_ragged_padded=x_scale_ragged_padded,
    )
    if n_act_eff > 1:
        if medium_decode_combine_eligible:
            # Fused top-k reduction over the scatter rows (graph-capturable).
            # The TOPK partials for a token are consecutive rows of y.
            y_reduced = (
                out
                if out is not None
                else torch.empty((n_tokens_eff, y_n), device=x.device, dtype=out_dtype)
            )
            R_BLOCK_N = 256
            r_grid = (n_tokens_eff * ((y_n + R_BLOCK_N - 1) // R_BLOCK_N),)
            _moe_partial_reduce[r_grid](
                y,
                y_reduced,
                n_tokens_eff,
                y_n,
                y.stride(0),
                n_act_eff * y.stride(0),
                y.stride(1),
                y_reduced.stride(0),
                y_reduced.stride(1),
                SPLIT_K=n_act_eff,
                BLOCK_N=R_BLOCK_N,
                num_warps=1,
            )
            y = y_reduced
        else:
            y_rows = y.view(n_tokens_eff, n_act_eff, y_n)
            y = (
                torch.sum(y_rows, dim=1, out=out)
                if out is not None
                else y_rows.sum(dim=1)
            )
    # Unpad N if the caller padded W for w_preshuffle. Padded W bytes
    # are 0 and padded scales are 127 so acc[:, N:N_padded] == 0.
    if logical_n != y.shape[-1]:
        y = y[..., :logical_n].contiguous()
    return y


_TUNING_KW = frozenset(
    {"block_m", "block_n", "block_k", "num_warps", "num_buffers", "dtype"}
)


# Gluon-only kwargs; explicitly stripped before forwarding upstream.
_GLUON_PRIVATE_KW = frozenset(
    {"out", "out_quant_format", "out_quant_scale", "x_scale_ragged_padded"}
)


def gluon_mxfp_ragged_matmul(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    w_mx_scale: torch.Tensor,
    x_global_scale: Optional[torch.Tensor] = None,
    x_mx_scale: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    x_format: str = "e4m3",
    a_ragged_metadata=None,
    gather_indx=None,
    scatter_indx=None,
    fused_activation=None,
    n_tokens=None,
    n_expts_act=None,
    **extra_kwargs,
) -> torch.Tensor | None:
    assert w_mx_scale is not None

    if x_format == "e4m3":
        x_global_scale = _global_scale_passthrough(x_global_scale)
        x_view = x.view(torch.uint8) if x.dtype != torch.uint8 else x
        x_scale = None
    elif x_format == "e2m1":
        if x_mx_scale is None:
            raise RuntimeError("x_mx_scale is required for e2m1 input X")
        x_global_scale = 1.0
        x_view = x.view(torch.uint8) if x.dtype != torch.uint8 else x
        x_scale = _extract_gluon_raw_s(x_mx_scale)
        assert isinstance(x_scale, torch.Tensor)
    else:
        raise RuntimeError(f"unsupported input X format: {x_format}")

    if out_dtype is None and x.dtype.is_floating_point:
        out_dtype = x.dtype
    elif out_dtype is None:
        out_dtype = torch.bfloat16

    prefer_unshuffled_w = bool(extra_kwargs.get("prefer_unshuffled_w", False))
    w_raw = (
        _extract_gluon_raw_w_unshuffled(w)
        if prefer_unshuffled_w
        else _extract_gluon_raw_w(w)
    )
    s_raw = _extract_gluon_raw_s(w_mx_scale)

    assert isinstance(w_raw, torch.Tensor) and isinstance(s_raw, torch.Tensor)
    assert w_raw.ndim == 3

    # Wrap bare tensors into ``.<attr>``-typed adapters; the launcher
    # consults gather_indx.src_indx / scatter_indx.dst_indx.
    def _adapt_indx(obj, attr):
        if obj is None:
            return None
        if hasattr(obj, attr):
            return obj
        if isinstance(obj, torch.Tensor):
            return type("IndxAdapter", (), {attr: obj})()
        return obj

    gather_indx = _adapt_indx(gather_indx, "src_indx")
    scatter_indx = _adapt_indx(scatter_indx, "dst_indx")

    swiglu_args = _maybe_extract_swiglu_args(fused_activation)
    has_gather = gather_indx is not None
    has_scatter = scatter_indx is not None

    if fused_activation is not None:
        assert swiglu_args is not None, "SwiGLU activation requires swiglu_args"

    gammas = extra_kwargs.get("gammas")
    out_quant_scale = extra_kwargs.get("out_quant_scale")
    out_quant_format = extra_kwargs.get("out_quant_format")
    x_scale_ragged_padded = bool(extra_kwargs.get("x_scale_ragged_padded", False))
    output = extra_kwargs.get("out")
    scale_load_mode = extra_kwargs.get("scale_load_mode", "swizzle")
    launch_kwargs = {
        key: extra_kwargs[key]
        for key in (
            "block_m",
            "block_n",
            "block_k",
            "num_warps",
            "num_buffers",
            "use_warp_pipeline",
            "use_slice_mn",
            "use_slice_n",
        )
        if key in extra_kwargs
    }

    if has_scatter and not has_gather:
        # gemm + combine
        w_preshuffle = bool(getattr(w_raw, "is_shuffled_for_gluon_dot", False))
        out = gluon_mxfp_combine(
            x_view,
            w_raw,
            s_raw,
            x_scale=x_scale,
            x_format=x_format,
            x_global_scale=x_global_scale,
            bias=bias,
            a_ragged_metadata=a_ragged_metadata,
            scatter_indx=scatter_indx,
            gate_scal=gammas,
            n_tokens=n_tokens,
            n_expts_act=n_expts_act,
            out_dtype=out_dtype,
            scale_load_mode=scale_load_mode,
            w_transpose=True,
            w_preshuffle=w_preshuffle,
            x_scale_ragged_padded=x_scale_ragged_padded,
            out=output,
            **launch_kwargs,
        )
        return out

    if not has_scatter and swiglu_args is not None:
        swiglu_alpha, swiglu_limit, swiglu_beta = swiglu_args
        w_preshuffle = bool(getattr(w_raw, "is_shuffled_for_gluon_dot", False))
        out = gluon_mxfp_dispatch_swiglu(
            x_view,
            w_raw,
            s_raw,
            x_scale=x_scale,
            x_format=x_format,
            x_global_scale=x_global_scale,
            bias=bias,
            a_ragged_metadata=a_ragged_metadata,
            gather_indx=gather_indx,
            out_dtype=out_dtype,
            swiglu_alpha=swiglu_alpha,
            swiglu_limit=swiglu_limit,
            swiglu_beta=swiglu_beta,
            scale_load_mode=scale_load_mode,
            w_transpose=True,
            out_quant_scale=out_quant_scale,
            out_quant_format=out_quant_format,
            w_preshuffle=w_preshuffle,
            x_scale_ragged_padded=x_scale_ragged_padded,
            **launch_kwargs,
        )
        return out
