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

"""Host-side launch marshaling for the pipelined ragged GEMM kernel, plus
AMDGCN static profiling / spill-check helpers."""

from __future__ import annotations

from typing import Any

import torch
from tokenspeed_kernel_amd.ops.gfx950.moe._common import (
    RaggedTensorMetadata,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused._common import (
    _BLOCK_SIZES_FROZEN,
    _SCALE_LOAD_MODES,
    _SCALED_FORMATS,
    _as_int32,
    _make_dummy,
    _ragged_block_offs,
    _ragged_block_schedule,
    _ragged_scale_block_offs,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused.pipelined_kernel import (
    _pipelined_moe_kernel_scaled,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused.tuning import (
    _MFMA_M,
    _MFMA_SCALED_K,
    _autotune_pid_swizzle,
    _effective_scale_load_mode,
    _persistent_grid_size,
    _preshuffled_layout_block_n,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.scale_layout import (
    is_swizzled_cdna4_mxfp4_scale,
    swizzle_cdna4_mxfp4_scale,
)

# ---------------------------------------------------------------------------
# Static profile helper (sgpr/vgpr spill detection)
# ---------------------------------------------------------------------------


def _parse_amdgcn_metric(amdgcn: str, key: str) -> int | None:
    """Look for ``.<key>: N`` or ``;  Key: N`` in the AMDGCN dump."""
    import re

    m = re.search(rf"\.{key}:\s+(\d+)", amdgcn)
    if m is not None:
        return int(m.group(1))
    m = re.search(rf";\s+{key}\s*[:=]?\s+(\d+)", amdgcn)
    if m is not None:
        return int(m.group(1))
    return None


def static_profile(kernel: Any, *, label: str = "") -> dict:
    amdgcn = kernel.asm.get("amdgcn", "")
    fields = [
        "sgpr_count",
        "sgpr_spill_count",
        "vgpr_count",
        "vgpr_spill_count",
        "ScratchSize",
        "codeLenInByte",
        "Occupancy",
    ]
    profile = {f: _parse_amdgcn_metric(amdgcn, f) for f in fields}
    if label:
        profile["label"] = label
    return profile


_LAST_KERNEL_PROFILE: dict | None = None


_PROFILE_BY_KERNEL_ID: dict[int, dict] = {}


def _capture_launch_profile(k: Any) -> None:
    global _LAST_KERNEL_PROFILE
    key = id(k)
    prof = _PROFILE_BY_KERNEL_ID.get(key)
    if prof is None:
        prof = static_profile(k)
        name = getattr(k, "name", None)
        if name is None:
            md = getattr(k, "metadata", None)
            name = getattr(md, "name", None) if md is not None else None
        if name is not None:
            prof["kernel_name"] = str(name)
        md = getattr(k, "metadata", None)
        if md is not None:
            shared = getattr(md, "shared", None)
            if shared is not None:
                prof["shared"] = int(shared)
        _PROFILE_BY_KERNEL_ID[key] = prof
    _LAST_KERNEL_PROFILE = prof


def last_kernel_profile() -> dict | None:
    return _LAST_KERNEL_PROFILE


def assert_no_spills(profile: dict, *, allow_scratch: int = 0) -> None:
    sgpr_spill = profile.get("sgpr_spill_count") or 0
    vgpr_spill = profile.get("vgpr_spill_count") or 0
    scratch = profile.get("ScratchSize") or 0
    msg = []
    if sgpr_spill:
        msg.append(f"sgpr_spill={sgpr_spill}")
    if vgpr_spill:
        msg.append(f"vgpr_spill={vgpr_spill}")
    if scratch > allow_scratch:
        msg.append(f"scratch={scratch} (allowed={allow_scratch})")
    if msg:
        raise AssertionError(
            f"Gluon MoE kernel '{profile.get('label', '?')}' "
            f"shows static spills: {', '.join(msg)}"
        )


def _dense_grid_dims(M: int, block_m: int) -> tuple[int, int]:
    """Return ``(num_active, blocks_per_expert)`` for the no-ragged
    (dense / gating GEMM) path."""
    return 1, (M + block_m - 1) // block_m


def _preprocess_scale(data: torch.Tensor | None, mode: str) -> torch.Tensor | None:
    if data is None:
        return None
    if mode not in _SCALE_LOAD_MODES:
        raise ValueError(
            f"_preprocess_scale: SCALE_LOAD_MODE must be one of "
            f"{_SCALE_LOAD_MODES}, got {mode!r}"
        )
    if mode == "swizzle":
        if is_swizzled_cdna4_mxfp4_scale(data):
            return data
        assert data.dtype == torch.uint8, (
            f"_preprocess_scale: expected uint8 e8m0 scales, " f"got {data.dtype}"
        )
        return swizzle_cdna4_mxfp4_scale(data)
    return data


# ---------------------------------------------------------------------------
# Public launcher: software-pipelined ragged matmul (scaled-MFMA only)
# ---------------------------------------------------------------------------


def _scale_strides(scale: torch.Tensor | None, mode: str = "bypass") -> tuple[int, int]:
    if scale is None:
        return 0, 0
    if mode == "swizzle":
        return scale.stride(-1), scale.stride(-2)
    return scale.stride(-2), scale.stride(-1)


def _launch_kernel(
    x: torch.Tensor,
    w: torch.Tensor,
    *,
    y: torch.Tensor,
    bias: torch.Tensor | None,
    gather_indx,
    scatter_indx,
    gate_scal: torch.Tensor | None,
    a_ragged_metadata,
    swiglu: tuple[float, float, float] | None,
    out_block_n: int,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    num_buffers: int = 2,
    x_format: str,
    w_format: str = "e2m1",
    x_scale: torch.Tensor | None = None,
    w_scale: torch.Tensor | None = None,
    x_global_scale: torch.Tensor | float | None = 1.0,
    scale_load_mode: str = "bypass",
    w_transpose: bool = False,
    apply_x_global_scale: bool | None = None,
    use_warp_pipeline: bool = False,
    use_slice_mn: bool = False,
    use_slice_n: bool = False,
    persistent: bool | None = False,
    num_ctas: int | None = None,
    group_m: int | None = None,
    xcd_swizzle: int | None = None,
    out_quant_scale: torch.Tensor | float | None = None,
    out_mx_scale: torch.Tensor | None = None,
    w_preshuffle: bool = False,
    y_n_const: int = 0,
    w_cache_cg: bool | None = None,
    use_narrow_n_store_layout: bool = False,
    medium_decode_dispatch: bool = False,
    medium_decode_combine: bool = False,
    x_scale_ragged_padded: bool = False,
):
    assert x_format in _SCALED_FORMATS, f"unknown x_format={x_format!r}"
    assert w_format in _SCALED_FORMATS, f"unknown w_format={w_format!r}"
    if apply_x_global_scale is None:
        apply_x_global_scale = True
    assert scale_load_mode in _SCALE_LOAD_MODES, (
        f"scale_load_mode must be one of {_SCALE_LOAD_MODES}, "
        f"got {scale_load_mode!r}"
    )
    has_x_block_scale = x_format == "e2m1"
    has_w_block_scale = w_format == "e2m1"
    if has_x_block_scale:
        assert x_scale is not None, "mxfp4 A requires a block-scale tensor"
    if has_w_block_scale:
        assert w_scale is not None, "mxfp4 W requires a block-scale tensor"
    if has_x_block_scale and gather_indx is not None:
        raise ValueError(
            "gathered MXFP4 activations must be quantized into gathered row "
            "order before gluon_mxfp_ragged_matmul"
        )

    M_X = x.shape[-2]
    if gather_indx is not None:
        gather_buf_for_m = gather_indx.src_indx
        M = int(gather_buf_for_m.shape[0])
    else:
        M = M_X
    K_phys = x.shape[-1]
    div_x = 2 if x_format == "e2m1" else 1
    div_w = 2 if w_format == "e2m1" else 1
    K = K_phys * div_x

    scale_load_mode = _effective_scale_load_mode(
        scale_load_mode,
        block_m,
        block_n,
        block_k,
        scale_block=32,
        has_x_scale=has_x_block_scale,
        has_w_scale=has_w_block_scale,
        k=K,
        x_format=x_format,
        num_buffers=num_buffers,
    )

    if w.ndim == 3:
        _, K_w_phys, N_w_phys = w.shape
    else:
        K_w_phys, N_w_phys = w.shape
    K_w = K_w_phys * div_w
    if w_preshuffle and getattr(w, "is_shuffled_for_gluon_dot", False):
        # Host pre-shuffle zero-pads K_pk to a multiple of 128 and W
        # scale to padded N (combine launcher trims output back).
        original_k_pk = getattr(w, "original_k_pk", K_w_phys)
        assert (
            K == original_k_pk * div_w
        ), f"K mismatch: A logical K={K} vs W original logical K={original_k_pk * div_w}"
        assert (
            K_w_phys >= original_k_pk and K_w_phys % 128 == 0
        ), f"shuffled W K_pk ({K_w_phys}) must be K_pk_padded (multiple of 128)"
        N = N_w_phys
    else:
        assert K == K_w, f"K mismatch: A logical K={K} vs W logical K={K_w}"
        N = N_w_phys

    assert (
        block_k % _MFMA_SCALED_K == 0
    ), f"BLOCK_K={block_k} must be a multiple of MFMA K dim ({_MFMA_SCALED_K})"
    assert (
        block_k >= _MFMA_SCALED_K
    ), f"scaled MFMA requires BLOCK_K >= {_MFMA_SCALED_K} (got {block_k})"
    assert block_m % _MFMA_M == 0
    if w_preshuffle:
        packed_block_n = _preshuffled_layout_block_n(w)
        expected_packed_block_n = block_n // 2 if use_slice_n else block_n
        assert not use_slice_mn, "preshuffled W LDS path does not support USE_SLICE_MN"
        assert num_warps == 4, "preshuffled W LDS load layout requires NUM_WARPS=4"
        assert packed_block_n == 128, (
            f"preshuffled W LDS path currently supports only block_n=128 "
            f"packed layouts; got block_n={packed_block_n}"
        )
        assert expected_packed_block_n == packed_block_n, (
            f"preshuffled W packed block_n={packed_block_n} is incompatible with "
            f"execution BLOCK_N={block_n}, USE_SLICE_N={use_slice_n}"
        )
        assert (
            N % packed_block_n == 0
        ), f"preshuffled W N={N} must be divisible by packed block_n={packed_block_n}"

    grid_n = (N + block_n - 1) // block_n

    # Per-expert ragged offsets needed when per-expert size < BLOCK_M
    # (else off_m would walk past the expert tail into the next one).
    has_ragged_offs = a_ragged_metadata is not None
    if has_ragged_offs:
        slice_offs_buf = _as_int32(a_ragged_metadata.slice_offs)
        slice_sizes_buf = _as_int32(a_ragged_metadata.slice_sizes)
    else:
        slice_offs_buf = _make_dummy(x.device, torch.int32)
        slice_sizes_buf = _make_dummy(x.device, torch.int32)
    has_padded_x_scale_rows = (
        has_ragged_offs and has_x_block_scale and bool(x_scale_ragged_padded)
    )
    if has_padded_x_scale_rows:
        x_scale_block_offs_buf = _as_int32(_ragged_scale_block_offs(a_ragged_metadata))
    else:
        x_scale_block_offs_buf = _make_dummy(x.device, torch.int32)

    # Block-schedule path: host picks grid_m as an integer upper bound
    # (no D2H sync, graph-capturable) and the kernel decodes
    # (expert_id, block_in_expert) from block_schedule[pid_m]. The
    # dense fallback is only valid when ``a_ragged_metadata is None``.
    use_block_schedule = (
        has_ragged_offs
        and block_m in _BLOCK_SIZES_FROZEN
        and getattr(a_ragged_metadata, "block_offs_data", None) is not None
        and getattr(a_ragged_metadata, "block_schedule_data", None) is not None
    )

    if use_block_schedule:
        n_slices = int(a_ragged_metadata.slice_sizes.shape[0])
        grid_m_upper = RaggedTensorMetadata.n_blocks(n_slices, M, block_m)
        num_tiles_total = grid_m_upper * grid_n
        block_offs_buf = _as_int32(_ragged_block_offs(a_ragged_metadata, block_m))
        block_schedule_buf = _as_int32(
            _ragged_block_schedule(a_ragged_metadata, block_m)
        )
        blocks_per_expert = 1  # unused constexpr sentinel in schedule mode
    else:
        # Only ``a_ragged_metadata is None`` (dense GEMM) is accepted;
        # hand-built ragged metadata without schedule tables is rejected
        # to avoid the historical D2H ``counts.tolist()`` path.
        assert not has_ragged_offs, (
            f"_launch_kernel requires a_ragged_metadata to either be None "
            f"(dense / gating GEMM) or to have populated "
            f"block_offs_data + block_schedule_data and "
            f"block_m={block_m} in {sorted(_BLOCK_SIZES_FROZEN)}. Build "
            f"the metadata via triton_kernels' make_ragged_tensor_metadata."
        )
        _, blocks_per_expert = _dense_grid_dims(M, block_m)
        num_tiles_total = blocks_per_expert * grid_n
        block_offs_buf = _make_dummy(x.device, torch.int32)
        block_schedule_buf = _make_dummy(x.device, torch.int32)
        n_slices = 0

    if persistent:
        if num_ctas is None:
            num_ctas = _persistent_grid_size(num_tiles_total)
        else:
            num_ctas = max(1, min(num_ctas, num_tiles_total))
    else:
        num_ctas = max(1, num_tiles_total)
    grid = (num_ctas, 1)

    grid_m_for_swizzle = num_tiles_total // grid_n
    auto_group_m, auto_xcd = _autotune_pid_swizzle(
        num_tiles_total=num_tiles_total,
        grid_m_padded=grid_m_for_swizzle,
        block_m=block_m,
    )
    if group_m is None:
        group_m = auto_group_m
    if xcd_swizzle is None:
        xcd_swizzle = auto_xcd
    if group_m > 1 and grid_m_for_swizzle % group_m != 0:
        group_m = 1
    if xcd_swizzle > 1 and num_tiles_total % xcd_swizzle != 0:
        xcd_swizzle = 1
    if w_cache_cg is None:
        w_cache_cg = block_m <= 32

    bias_buf = bias if bias is not None else _make_dummy(x.device, torch.float32)
    gather_buf = (
        gather_indx.src_indx
        if gather_indx is not None
        else _make_dummy(x.device, torch.int32)
    )
    scatter_buf = (
        scatter_indx.dst_indx
        if scatter_indx is not None
        else _make_dummy(x.device, torch.int32)
    )
    gate_scal_buf = (
        gate_scal if gate_scal is not None else _make_dummy(x.device, torch.float32)
    )

    swiglu_alpha = swiglu[0] if swiglu is not None else 0.0
    swiglu_limit = swiglu[1] if swiglu is not None else 0.0
    swiglu_beta = swiglu[2] if swiglu is not None else 0.0

    w3 = w if w.ndim == 3 else w.unsqueeze(0)

    if w_preshuffle:
        # Host pre-shuffled into 5-D HBM byte layout. The preshuffled
        # descriptor uses N directly for the K-iter stride and stages the
        # tile through LDS; .contiguous() would clobber the HBM layout.
        # stride_wn/stride_wk are not consulted; only stride_we matters.
        # w_transpose is irrelevant on this path.
        stride_wn, stride_wk = w3.stride(-2), w3.stride(-1)
    elif w_transpose:
        # K-contig W staged as [BN, BK] in LDS; view permuted for dot.
        w3 = w3.transpose(-1, -2).contiguous()
        stride_wn, stride_wk = w3.stride(-2), w3.stride(-1)
    else:
        # N-contig W staged as [BK, BN] in LDS.
        stride_wn, stride_wk = w3.stride(-1), w3.stride(-2)

    x_scale_load_mode = scale_load_mode
    w_scale_load_mode = scale_load_mode
    x_scale_via_lds = (
        x_scale_load_mode == "swizzle"
        and has_x_block_scale
        and a_ragged_metadata is None
    )
    w_scale_via_lds = w_scale_load_mode == "swizzle" and has_w_block_scale

    if has_w_block_scale:
        w_scale3 = w_scale if w_scale.ndim == 3 else w_scale.unsqueeze(0)
        w_scale_proc3 = _preprocess_scale(w_scale3, w_scale_load_mode)
        stride_wse = w_scale_proc3.stride(0)
        stride_wsn, stride_wsk = _scale_strides(w_scale_proc3, w_scale_load_mode)
        w_scale_buf = w_scale_proc3
    else:
        stride_wse = stride_wsn = stride_wsk = 0
        w_scale_buf = _make_dummy(x.device, torch.uint8)

    x_scale_proc = (
        _preprocess_scale(x_scale, x_scale_load_mode) if has_x_block_scale else None
    )
    stride_xsm, stride_xsk = _scale_strides(x_scale_proc, x_scale_load_mode)

    x_scale_buf = (
        x_scale_proc if x_scale_proc is not None else _make_dummy(x.device, torch.uint8)
    )

    if use_slice_mn:
        NUM_SUBTILES = (2, 2, 1)
    elif use_slice_n:
        NUM_SUBTILES = (1, 2, 1)
    else:
        NUM_SUBTILES = (1, 1, 1)
    EVEN_K = K % block_k == 0
    K_ITERS = (K + block_k - 1) // block_k

    needs_scale_load = apply_x_global_scale and not has_x_block_scale
    if not needs_scale_load:
        x_global_scale_buf = _make_dummy(x.device, torch.float32)
    elif isinstance(x_global_scale, torch.Tensor):
        # Production: zero-copy passthrough of the precision config scale.
        scale_view = x_global_scale.detach().reshape(-1)[:1]
        if scale_view.device == x.device and scale_view.dtype == torch.float32:
            x_global_scale_buf = scale_view
        else:
            x_global_scale_buf = scale_view.to(device=x.device, dtype=torch.float32)
    else:
        x_global_scale_buf = torch.tensor(
            [float(x_global_scale)], dtype=torch.float32, device=x.device
        )

    has_mxfp4_quant_out = out_mx_scale is not None
    has_fp8_quant_out = out_quant_scale is not None
    if has_fp8_quant_out and has_mxfp4_quant_out:
        raise ValueError("FP8 and MXFP4 output quantization are mutually exclusive")
    if has_fp8_quant_out:
        if isinstance(out_quant_scale, torch.Tensor):
            qscale_view = out_quant_scale.detach().reshape(-1)[:1]
            if qscale_view.device == x.device and qscale_view.dtype == torch.float32:
                out_quant_scale_buf = qscale_view
            else:
                out_quant_scale_buf = qscale_view.to(
                    device=x.device, dtype=torch.float32
                )
        else:
            out_quant_scale_buf = torch.tensor(
                [float(out_quant_scale)], dtype=torch.float32, device=x.device
            )
        assert y.dtype == torch.float8_e4m3fn, (
            f"out_quant_scale requires a float8_e4m3fn output buffer, "
            f"got y.dtype={y.dtype}"
        )
        if not swiglu:
            raise ValueError(
                "out_quant_scale is currently only wired through the SwiGLU "
                "epilogue (GEMM1 fused quant). For combine-GEMM (DO_SWIGLU=False) "
                "quant, see follow-up P0-1 step 5."
            )
    else:
        out_quant_scale_buf = _make_dummy(x.device, torch.float32)

    if has_mxfp4_quant_out:
        if not swiglu:
            raise ValueError("MXFP4 output quantization is only supported for SwiGLU")
        if y.dtype != torch.uint8:
            raise ValueError(
                f"MXFP4 output quantization requires uint8 y, got {y.dtype}"
            )
        out_mx_scale_buf = out_mx_scale
        stride_out_mxs_kswizzled = out_mx_scale.stride(0)
        stride_out_mxs_mblock = out_mx_scale.stride(1)
    else:
        out_mx_scale_buf = _make_dummy(x.device, torch.uint8)
        stride_out_mxs_kswizzled = 0
        stride_out_mxs_mblock = 0

    # Common args / constexprs shared by both kernel entries.
    common_args = (
        x,
        w3,
        x_scale_buf,
        w_scale_buf,
        bias_buf,
        y,
        gather_buf,
        scatter_buf,
        gate_scal_buf,
        slice_offs_buf,
        slice_sizes_buf,
        x_scale_block_offs_buf,
    )
    common_strides = (
        x.stride(-2),
        x.stride(-1),
        w3.stride(0),
        stride_wn,
        stride_wk,
        stride_xsm,
        stride_xsk,
        stride_wse,
        stride_wsn,
        stride_wsk,
        y.stride(-1),
        y.stride(-2),
        bias.stride(0) if bias is not None else 0,
        bias.stride(-1) if bias is not None else 0,
    )
    common_dims = (
        M,
        M_X,
        N,
        K,
        x_global_scale_buf,
        out_quant_scale_buf,
        out_mx_scale_buf,
        stride_out_mxs_kswizzled,
        stride_out_mxs_mblock,
        num_tiles_total,
    )
    common_kwargs = dict(
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        BLOCKS_PER_EXPERT=blocks_per_expert,
        X_FORMAT=x_format,
        W_FORMAT=w_format,
        UPCAST_INDICES=False,
        HAS_X_BLOCK_SCALE=has_x_block_scale,
        HAS_W_BLOCK_SCALE=has_w_block_scale,
        HAS_BIAS=bias is not None,
        HAS_GATHER=gather_indx is not None,
        HAS_SCATTER=scatter_indx is not None,
        DO_SWIGLU=swiglu is not None,
        SWIGLU_ALPHA=float(swiglu_alpha),
        SWIGLU_LIMIT=float(swiglu_limit),
        SWIGLU_BETA=float(swiglu_beta),
        OUT_BLOCK_N=out_block_n,
        APPLY_GATE_SCAL=gate_scal is not None,
        HAS_RAGGED_OFFS=has_ragged_offs,
        NUM_WARPS=num_warps,
        NUM_BUFFERS=num_buffers,
        SCALE_LOAD_MODE=scale_load_mode,
        W_TRANSPOSE=w_transpose,
        NUM_SUBTILES=NUM_SUBTILES,
        EVEN_K=EVEN_K,
        K_ITERS=K_ITERS,
        N_CONST=N if w_preshuffle else 0,
        Y_N_CONST=int(y_n_const),
        APPLY_X_GLOBAL_SCALE=apply_x_global_scale,
        USE_WARP_PIPELINE=use_warp_pipeline,
        USE_SLICE_MN=use_slice_mn,
        USE_SLICE_N=use_slice_n,
        HAS_FP8_QUANT_OUT=has_fp8_quant_out,
        HAS_MXFP4_QUANT_OUT=has_mxfp4_quant_out,
        W_PRESHUFFLED=w_preshuffle,
        W_VIA_VGPR=False,
        W_PREFETCH=False,
        W_CACHE_CG=bool(w_cache_cg),
        X_SCALE_VIA_LDS=bool(x_scale_via_lds),
        W_SCALE_VIA_LDS=bool(w_scale_via_lds),
        USE_NARROW_N_STORE_LAYOUT=bool(use_narrow_n_store_layout),
        X_SCALE_RAGGED_PADDED=bool(has_padded_x_scale_rows),
        GRID_N=grid_n,
        GROUP_M=group_m,
        XCD_SWIZZLE=xcd_swizzle,
        num_warps=num_warps,
    )

    common_kwargs["waves_per_eu"] = num_warps // 4

    if medium_decode_dispatch or medium_decode_combine:
        # M=8/16 decode reuses the regular launcher preprocessing, but selects
        # the single-buffer direct-load body under a constexpr. Keep the
        # identity swizzles/grid shape the direct kernels were tuned for and
        # avoid the occupancy hint that bloats the medium body's VGPR footprint.
        medium_kwargs = dict(common_kwargs)
        medium_kwargs["GROUP_M"] = 1
        medium_kwargs["XCD_SWIZZLE"] = 1
        medium_kwargs["num_stages"] = 1
        medium_kwargs.pop("waves_per_eu", None)
        k = _pipelined_moe_kernel_scaled[(num_tiles_total,)](
            *common_args,
            block_offs_buf,
            block_schedule_buf,
            *common_strides,
            *common_dims,
            USE_BLOCK_SCHEDULE=use_block_schedule,
            N_EXPTS_TOT=n_slices,
            IS_MEDIUM_DECODE=True,
            MEDIUM_COMBINE=medium_decode_combine,
            **medium_kwargs,
        )
        _capture_launch_profile(k)
        return

    k = _pipelined_moe_kernel_scaled[grid](
        *common_args,
        block_offs_buf,
        block_schedule_buf,
        *common_strides,
        *common_dims,
        USE_BLOCK_SCHEDULE=use_block_schedule,
        N_EXPTS_TOT=n_slices,
        **common_kwargs,
    )

    _capture_launch_profile(k)
