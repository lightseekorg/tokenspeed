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

"""Host-side block-size autotuning and launch heuristics for the
pipelined ragged GEMM."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused._common import (
    _CDNA4_NUM_CUS,
    _CDNA4_NUM_XCDS,
    _GLUON_DOT_N_LANE,
    _NON_K_PRESHUFFLE_BLOCK_SIZE,
    _PERSISTENT_OVERSUBSCRIBE,
    _SWIZZLE_K_S_INNER,
    _TCP_INFLIGHT_CAP_BYTES,
)


def _effective_scale_load_mode(
    mode: str,
    block_m: int,
    block_n: int,
    block_k: int,
    scale_block: int,
    has_x_scale: bool,
    has_w_scale: bool,
    k: int | None = None,
    x_format: str | None = None,
    num_buffers: int | None = None,
) -> str:
    del k, x_format, num_buffers
    if mode != "swizzle":
        return mode
    # CDNA4MXScaleLayout requires BLOCK_K_S >= 8 and BLOCK_{M,N} %
    # 32 == 0 when the corresponding scale is present. Hard-assert
    # (no fallback) -- the input scale tensor is already in the
    # upstream swizzled storage.
    bk_s = block_k // scale_block
    assert bk_s >= _SWIZZLE_K_S_INNER, (
        f"swizzle requires BLOCK_K // SCALE_BLOCK >= "
        f"{_SWIZZLE_K_S_INNER} (got BLOCK_K={block_k}, "
        f"SCALE_BLOCK={scale_block} -> BLOCK_K_S={bk_s}). Bump "
        f"BLOCK_K to >= {_SWIZZLE_K_S_INNER * scale_block}."
    )
    if has_x_scale:
        assert block_m % _NON_K_PRESHUFFLE_BLOCK_SIZE == 0, (
            f"swizzle requires BLOCK_M % "
            f"{_NON_K_PRESHUFFLE_BLOCK_SIZE} == 0 when x_scale is "
            f"present (got BLOCK_M={block_m})."
        )
    if has_w_scale:
        assert block_n % _NON_K_PRESHUFFLE_BLOCK_SIZE == 0, (
            f"swizzle requires BLOCK_N % "
            f"{_NON_K_PRESHUFFLE_BLOCK_SIZE} == 0 when w_scale is "
            f"present (got BLOCK_N={block_n})."
        )
    return "swizzle"


# CDNA4 MFMA scaled = 16x16x128.
_MFMA_SCALED_K = 128


_MFMA_M = 16


def _round_up_int(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m


def _clamp_block_m(block_m: int, M: int) -> int:
    target = max(_MFMA_M, min(block_m, _round_up_int(M, _MFMA_M)))
    return 1 << (target.bit_length() - 1)


def _ragged_slice_size(a_ragged_metadata, M: int) -> int | None:
    """Per-expert M hint for autotune (mirrors upstream
    ``opt_flags_amd``'s formula). Returns ``None`` on no metadata."""
    if a_ragged_metadata is None:
        return None
    expected = getattr(a_ragged_metadata, "expected_slice_size", None)
    if expected is not None:
        return int(expected)
    try:
        n_slices = int(a_ragged_metadata.slice_sizes.shape[0])
    except (AttributeError, IndexError):
        return None
    return max(1, M // max(1, n_slices))


def _autotune_block(
    M: int,
    N: int,
    K: int,
    *,
    do_swiglu: bool = False,
    ragged: bool = False,
    x_format: str = "e2m1",
    scale_load_mode: str = "transpose",
    slice_size: int | None = None,
    block_m: int | None = None,
    block_n: int | None = None,
    block_k: int | None = None,
    use_slice_n: bool | None = None,
    large_slice_size: int | None = None,
    large_m: int | None = None,
) -> tuple[int, int, int, int, bool | None, bool]:
    """Pick the scaled-MFMA tile route.

    Sweep-tuned on gpt-oss-120b (H=I=2880, E=128, top_k=4) at MI355.
    Tiers off logical ``M`` and the per-expert ``slice_size`` hint;
    ``BLOCK_K`` must be a multiple of 128 (MFMA 16x16x128).
    """
    del ragged
    is_fp8 = x_format == "e4m3"
    if slice_size is not None and slice_size <= 16:
        # tinny ragged decode
        bm, bn, bk, nw = 64, 128, 256, 4
    elif slice_size is not None and slice_size <= 64 and M <= 8192:
        # mid ragged decode
        bm, bn, bk, nw = 64, 128, 256, 4
    elif M <= 512:
        bm, bn, bk, nw = 64, 128, 512, 8
    elif is_fp8:
        # fp8 X is 1 byte/elem (lower VGPR pressure); prefill promotes
        # to (128, 256, 256, NW=4) -- sliceMN sweet spot for dispatch.
        if M <= 8192:
            # combine + preshuffled W requires NW=4 (LinearLayout bases);
            # dispatch tolerates NW=8 since OUT_BLOCK_N halving sidesteps
            # the BN=256 / SLICE_N constraint at the BN=256 tile.
            bm, bn, bk, nw = (64, 256, 128, 8) if do_swiglu else (64, 256, 128, 4)
        elif do_swiglu:
            # Preshuffled dispatch may lower BM in the launcher so BN=256 can
            # use SliceN over two 128-wide packed half-tiles.
            bm, bn, bk, nw = 128, 256, 128, 4
        else:
            # combine path: keep BN=256 throughput but force BM<=64
            # so ``_resolve_use_slice_n`` enables USE_SLICE_N=True
            # (half-tile path), which the preshuffled-W static_assert
            # explicitly accepts. NW=4 also required.
            bm, bn, bk, nw = 64, 256, 128, 4
    else:
        # mxfp4 X dequant adds VGPR pressure; same sliceMN sweet spot
        # at the prefill tier.
        if M <= 8192:
            bm, bn, bk, nw = 64, 256, 256, 4
        elif do_swiglu:
            bm, bn, bk, nw = 128, 256, 256, 4
        else:
            bm, bn, bk, nw = 64, 256, 256, 4
    # Clamp tile to actual shape (avoid over-tile + NaN-padded
    # reduction on tiny test shapes).
    bm = _clamp_block_m(bm, M)
    bn = max(_MFMA_M, min(bn, _round_up_int(N, _MFMA_M)))
    bk = max(_MFMA_SCALED_K, min(bk, _round_up_int(K, _MFMA_SCALED_K)))
    # Swizzle unswizzle reshape requires BLOCK_K_S >= 8 (= BLOCK_K
    # >= 256 with SCALE_BLOCK=32).
    if scale_load_mode == "swizzle":
        bk = max(bk, 256)
        bk = min(bk, _round_up_int(K, _MFMA_SCALED_K))

    requested_block_m = block_m
    requested_block_n = block_n
    block_m = block_m or bm
    block_n = block_n or bn
    block_k = block_k or bk

    use_small_m = (
        requested_block_m is None
        and slice_size is not None
        and slice_size < 16
        and 1024 <= M < 2048
    )
    use_medium_m = (
        requested_block_m is None
        and slice_size is not None
        and slice_size <= 16
        and 2048 <= M < 4096
    )
    use_large_m = (
        requested_block_m is None
        and slice_size is not None
        and large_slice_size is not None
        and large_m is not None
        and slice_size >= large_slice_size
        and M >= large_m
    )
    if use_small_m:
        block_m = 16
    elif use_medium_m:
        block_m = 32
    elif use_large_m:
        block_m = 128
        if use_slice_n is None:
            use_slice_n = True

    if requested_block_n is None and block_n == 128 and N >= 256:
        # The tuned prefill route consumes a 256-wide execution tile through the
        # SliceN pipeline.  Preshuffled W reads two 128-wide packed half-tiles;
        # non-preshuffled W reads the same two half-tiles from the normal LDS
        # layout, keeping the compute schedule aligned across both variants.
        block_n = 256

    if block_n == 256 and block_m > 64 and not use_large_m:
        block_m = 64

    if (
        requested_block_m is None
        and scale_load_mode == "swizzle"
        and x_format == "e2m1"
        and block_m < _NON_K_PRESHUFFLE_BLOCK_SIZE
    ):
        block_m = _NON_K_PRESHUFFLE_BLOCK_SIZE

    return block_m, block_n, block_k, nw, use_slice_n, use_small_m


def _autotune_pid_swizzle(
    num_tiles_total: int,
    grid_m_padded: int,
    block_m: int,
) -> tuple[int, int]:
    if num_tiles_total < 256:
        return 1, 1
    if block_m < 32:
        return 1, 1
    if grid_m_padded >= 8 and grid_m_padded % 4 == 0:
        group_m = 4
    elif grid_m_padded >= 2 and grid_m_padded % 2 == 0:
        group_m = 2
    else:
        group_m = 1
    xcd_swizzle = _CDNA4_NUM_XCDS if num_tiles_total % _CDNA4_NUM_XCDS == 0 else 1
    return group_m, xcd_swizzle


_LaunchTuning = tuple[int | None, int | None, bool | None, bool]


def _default_prefill_launch_tuning() -> _LaunchTuning:
    return 1, None, None, False


def _dispatch_prefill_launch_tuning(m: int) -> _LaunchTuning:
    if m <= 1024:
        return 1, _CDNA4_NUM_XCDS, None, False
    if m <= 2048:
        return 1, 4, None, False
    if m <= 4096:
        return 1, _CDNA4_NUM_XCDS, True, False
    return _default_prefill_launch_tuning()


def _combine_prefill_launch_tuning(m: int) -> _LaunchTuning:
    if m <= 1024:
        return 1, _CDNA4_NUM_XCDS, None, False
    if m <= 2048:
        return 1, 4, None, False
    if m <= 4096:
        return 1, _CDNA4_NUM_XCDS, True, False
    if m < 16384:
        return 1, 4, None, False
    return 1, 4, None, True


def _prefill_launch_tuning(
    op: str,
    *,
    m: int,
    use_slice_mn: bool,
) -> _LaunchTuning:
    if use_slice_mn:
        return _default_prefill_launch_tuning()

    if op == "dispatch":
        return _dispatch_prefill_launch_tuning(m)
    if op == "combine":
        return _combine_prefill_launch_tuning(m)
    return _default_prefill_launch_tuning()


def _persistent_grid_size(num_tiles_total: int) -> int:
    if num_tiles_total <= 0:
        return 1
    return max(1, min(num_tiles_total, _CDNA4_NUM_CUS * _PERSISTENT_OVERSUBSCRIBE))


def _needs_scale_lds(
    x_format: str, has_x_block_scale: bool, has_w_block_scale: bool
) -> bool:
    return (has_x_block_scale and x_format == "e2m1") or has_w_block_scale


def _can_use_slice_n(
    bm: int,
    bn: int,
    *,
    scale_load_mode: str,
    x_format: str,
    has_x_block_scale: bool,
    has_w_block_scale: bool,
) -> bool:
    if bn < 256 or bm < 16 or (bn // 2) % 64 != 0:
        return False
    if _needs_scale_lds(x_format, has_x_block_scale, has_w_block_scale):
        return scale_load_mode == "swizzle"
    return True


def _resolve_use_slice_n(
    use_slice_n: bool | None,
    bm: int,
    bn: int,
    *,
    scale_load_mode: str,
    x_format: str,
    has_x_block_scale: bool,
    has_w_block_scale: bool,
    bk: int,
) -> bool:
    if use_slice_n is not None:
        enabled = bool(use_slice_n)
    else:
        w_bytes = (bn * bk) // 2
        enabled = bn >= 256 and bm <= 64 and w_bytes >= _TCP_INFLIGHT_CAP_BYTES
    return enabled and _can_use_slice_n(
        bm,
        bn,
        scale_load_mode=scale_load_mode,
        x_format=x_format,
        has_x_block_scale=has_x_block_scale,
        has_w_block_scale=has_w_block_scale,
    )


def _can_use_slice_mn(
    bm: int,
    bn: int,
    *,
    num_buffers: int,
    scale_load_mode: str,
    x_format: str,
    has_x_block_scale: bool,
    has_w_block_scale: bool,
) -> bool:
    if bm < 128 or bn < 128:
        return False
    if (bm // 2) % 64 != 0 or (bn // 2) % 64 != 0:
        return False
    if num_buffers < 2:
        return False
    if _needs_scale_lds(x_format, has_x_block_scale, has_w_block_scale):
        return scale_load_mode == "swizzle"
    return True


def _resolve_use_slice_mn(
    use_slice_mn: bool | None,
    bm: int,
    bn: int,
    *,
    num_buffers: int,
    scale_load_mode: str,
    x_format: str,
    has_x_block_scale: bool,
    has_w_block_scale: bool,
    use_slice_n: bool = False,
    bk: int,
) -> bool:
    if use_slice_n:
        return False
    if use_slice_mn is not None:
        enabled = bool(use_slice_mn)
    else:
        w_bytes = (bn * bk) // 2 if x_format == "e2m1" else bn * bk
        enabled = bm >= 128 and bn >= 128 and w_bytes >= 16 * 1024 and (bm + bn) >= 384
    return enabled and _can_use_slice_mn(
        bm,
        bn,
        num_buffers=num_buffers,
        scale_load_mode=scale_load_mode,
        x_format=x_format,
        has_x_block_scale=has_x_block_scale,
        has_w_block_scale=has_w_block_scale,
    )


def _preshuffled_layout_block_n(w: torch.Tensor) -> int:
    block_n = int(getattr(w, "gluon_dot_block_n", 128))
    if block_n <= 0 or block_n % _GLUON_DOT_N_LANE != 0:
        raise ValueError(
            f"invalid preshuffled Gluon W layout block_n={block_n}; "
            f"expected a positive multiple of {_GLUON_DOT_N_LANE}"
        )
    return block_n


def _align_block_n_to_preshuffled_layout(
    w: torch.Tensor,
    *,
    block_m: int,
    block_n: int,
    block_k: int,
    scale_load_mode: str,
    x_format: str,
    has_x_block_scale: bool,
    has_w_block_scale: bool,
    use_slice_mn: bool | None,
    use_slice_n: bool | None,
) -> tuple[int, bool, bool | None]:
    """Constrain the execution tile to the host-preshuffled W layout.

    The current preshuffled kernel supports the 128-wide packed layout. A
    256-wide execution tile is still legal by consuming two adjacent 128-wide
    packed half-tiles through USE_SLICE_N.
    """
    packed_block_n = _preshuffled_layout_block_n(w)
    if packed_block_n != 128:
        raise ValueError(
            f"preshuffled Gluon W layout block_n={packed_block_n} is not "
            "supported by the current preshuffled W load layout"
        )

    # USE_SLICE_MN is not wired for the preshuffled W descriptor.
    use_slice_mn = False

    if block_n == packed_block_n:
        return block_n, use_slice_mn, use_slice_n

    auto_slice_n = (
        block_n >= 256
        and block_m <= 64
        and ((block_n * block_k) // 2) >= _TCP_INFLIGHT_CAP_BYTES
    )
    wants_slice_n = use_slice_n is True or (use_slice_n is None and auto_slice_n)
    can_use_slice_n = wants_slice_n and _can_use_slice_n(
        block_m,
        block_n,
        scale_load_mode=scale_load_mode,
        x_format=x_format,
        has_x_block_scale=has_x_block_scale,
        has_w_block_scale=has_w_block_scale,
    )

    if block_n == 2 * packed_block_n and can_use_slice_n:
        return block_n, use_slice_mn, use_slice_n

    return packed_block_n, use_slice_mn, False


def _resolve_prefill_slice_modes(
    *,
    use_slice_mn: bool | None,
    use_slice_n: bool | None,
    block_m: int,
    block_n: int,
    block_k: int,
    num_buffers: int,
    scale_load_mode: str,
    x_format: str,
    has_x_block_scale: bool,
    has_w_block_scale: bool,
) -> tuple[bool, bool]:
    if use_slice_mn is True:
        use_slice_mn_resolved = _resolve_use_slice_mn(
            True,
            block_m,
            block_n,
            num_buffers=num_buffers,
            scale_load_mode=scale_load_mode,
            x_format=x_format,
            has_x_block_scale=has_x_block_scale,
            has_w_block_scale=has_w_block_scale,
            bk=block_k,
        )
        if use_slice_mn_resolved:
            return True, False

    use_slice_n_resolved = _resolve_use_slice_n(
        use_slice_n,
        block_m,
        block_n,
        scale_load_mode=scale_load_mode,
        x_format=x_format,
        has_x_block_scale=has_x_block_scale,
        has_w_block_scale=has_w_block_scale,
        bk=block_k,
    )
    use_slice_mn_resolved = _resolve_use_slice_mn(
        use_slice_mn,
        block_m,
        block_n,
        num_buffers=num_buffers,
        scale_load_mode=scale_load_mode,
        x_format=x_format,
        has_x_block_scale=has_x_block_scale,
        has_w_block_scale=has_w_block_scale,
        use_slice_n=use_slice_n_resolved,
        bk=block_k,
    )
    return use_slice_mn_resolved, use_slice_n_resolved


def _is_single_k_tile(k: int, block_k: int) -> bool:
    """Whether the K reduction fits in a single BLOCK_K tile (K_ITERS == 1).

    The SliceN and pipelined schedules are double-buffered and require at least
    two K tiles; a single tile has no dedicated SliceN path, so such shapes must
    run on the full-N decode schedule instead.
    """
    return (k + block_k - 1) // block_k == 1
