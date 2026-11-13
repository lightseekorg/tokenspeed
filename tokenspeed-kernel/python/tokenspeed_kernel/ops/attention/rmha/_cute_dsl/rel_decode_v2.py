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

"""Rel-bias decode v2 on the standalone tokenspeed-mha decode kernel.

Replaces the fa4-fork fwd + triton shear + split-KV combine three-launch
path with ONE kernel (CUTLASS gqa_decode dataflow: K @ Q first GEMM,
grouped-head packing, rel bias fused into the log2 online softmax, SWA
tile culling, split-KV with in-kernel deterministic reduction). The
kernel consumes the COMPACT per-head rel table directly — no shear
prepass.

Serving page sizes run NATIVELY (64/128/256): the kernel assembles its
K/V MMA tiles from sub-page TMA boxes, so the engine's per-step flat
group table is consumed directly — no cache re-view, no page-table
expansion, no per-call table-rewrite ops. ``-1`` window holes are safe
by two independent mechanisms (both verified bit-exact and under
compute-sanitizer): the page index is a bounds-checked TMA coordinate
(out-of-range loads zero-fill, never fault), and the in-kernel window
mask is pure index math that -infs out-of-window columns regardless of
loaded content. NOTE the mask alone does not neutralize NaNs — the
zero-fill is load-bearing; never point an out-of-window table entry at
real (potentially uninitialized) memory.

Compiled once per static config (page size included) with TVM FFI:
runtime calls take torch tensors, dynamic batch / page-count / seqlens /
KV split count, and are CUDA-graph safe (``reduction_mode="kernel"`` rewrites its
workspaces every launch).

An MXFP8 K/V cache (FP8 E4M3 data plus the UE8M0 scales
``quantize_store_kv_mxfp8`` writes) is read directly with a BF16 query: the
kernel dequantizes each 128-token tile to BF16 in shared memory, so the result
equals the BF16 kernel's on the dequantized cache bit for bit.
"""

from __future__ import annotations

import math

import torch

# KV tokens per loop iteration, at least one page (the kernel assembles a tile
# from whole-page TMA boxes). On B300 (8 query rows per request, 12 query
# heads over 2 KV heads, D 128) 128-token tiles beat 256 at every batch,
# context and split count measured.
_SEQUENCE_TILE = 128
# More splits of a full-attention row stop paying: at one request, 24 splits
# take 10.3 / 14.6 / 36.7 us at 8K / 32K / 128K keys, 48 take 15.0 / 19.2 / 47.8.
_MAX_FULL_SPLITS = 24
# Full-attention splits fill the SMs about this many times. A batch's rows
# differ in length and one CTA streams its tiles at a fixed rate, so with one
# wave the longest row's few CTAs set the launch's time: on B300, 8-128 rows of
# mixed 8K-256K contexts took 1.05-3.4x as long with one wave as with seven
# (BF16 K/V; 1.3-6.6x with MXFP8). Rows of one length pay up to 26 us for it.
_FULL_WAVES = 7
_SM_COUNTS: dict[int, int] = {}


def _kv_splits(
    window_left: int,
    batch: int,
    prediction: int,
    num_q_heads: int,
    num_kv_heads: int,
    dtype: torch.dtype,
    device: torch.device,
) -> int:
    """KV splits of a decode launch: enough to fill the SMs once, at most one
    per tile a window spans; for full attention, enough to fill them
    ``_FULL_WAVES`` times, at most ``_MAX_FULL_SPLITS``. The batch shape alone
    sets it, so CUDA graphs keep it; it is a runtime argument of the one
    compiled kernel."""
    if device.index not in _SM_COUNTS:
        _SM_COUNTS[device.index] = torch.cuda.get_device_properties(
            device
        ).multi_processor_count
    grouped = num_q_heads // num_kv_heads
    head_tile = 1 << (grouped - 1).bit_length()
    prediction_tile = _prediction_tile(prediction, head_tile, window_left, dtype)
    base_ctas = (
        batch
        * num_kv_heads
        * math.ceil(grouped / head_tile)
        * math.ceil(prediction / prediction_tile)
    )
    waves = 1
    if window_left >= 0:
        # Counted in 128-token tiles: with larger pages the extra splits find
        # no tile and exit at once.
        cap = math.ceil((window_left + 1 + prediction_tile) / _SEQUENCE_TILE)
    else:
        cap = _MAX_FULL_SPLITS
        waves = _FULL_WAVES
    return max(1, min(cap, waves * _SM_COUNTS[device.index] // base_ctas))


class _CompiledCache:
    def __init__(self):
        self._cache = {}

    def get(self, *key):
        if key not in self._cache:
            self._cache[key] = _compile_decode(*key)
        return self._cache[key]


_DECODE = _CompiledCache()
# Split-KV workspaces, exact-B contiguous; see the allocation comment in the wrapper.
_WORKSPACES: dict = {}


def _prediction_tile(
    prediction: int, grouped_head_tile: int, window_left: int, dtype: torch.dtype
) -> int:
    """Query-token packing per CTA. SWA defers to the kernel's own packing
    model (short bf16 windows want few output rows per CTA); full attention
    keeps maximum legal packing (grouped_head_tile * tile <= 32)."""
    if prediction == 1:
        return 1
    if window_left >= 0:
        from .flash_fwd_sm100_bias_decode import (
            choose_swa_prediction_tile,
        )

        return choose_swa_prediction_tile(
            prediction,
            grouped_head_tile,
            window_left,
            bf16_eager=dtype == torch.bfloat16,
        )
    tile = 1 << (prediction - 1).bit_length()
    return min(32 // grouped_head_tile, tile)


def _compile_decode(
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    extent: int,
    window_left: int,
    dtype: torch.dtype,
    device_index: int,
    page_size: int,
    use_pdl: bool,
    prediction: int,
    has_tau: bool,
    out_dtype: torch.dtype,
    with_lse: bool,
    gated: bool,
    kv_mxfp8: bool,
):
    from cutlass import cute
    from flash_attn.cute.cute_dsl_utils import to_cute_tensor

    from .flash_fwd_sm100_bias_decode import (
        FlashAttentionDecodeSm100Bias,
    )

    grouped = num_q_heads // num_kv_heads
    grouped_head_tile = 1 << (grouped - 1).bit_length()  # next pow2
    kernel = FlashAttentionDecodeSm100Bias(
        head_dim,
        grouped_head_tile,
        prediction_tile=_prediction_tile(
            prediction, grouped_head_tile, window_left, dtype
        ),
        sequence_tile=max(_SEQUENCE_TILE, page_size),
        reduction_mode="kernel",
        window_size_left=window_left if window_left >= 0 else None,
        page_size=page_size,
        use_pdl=use_pdl,
        kv_mxfp8=kv_mxfp8,
    )
    dev = torch.device("cuda", device_index)
    # The split count, like the batch, is a runtime argument of the kernel.
    FB, FPAGES, P, kv_splits = 2, 8, prediction, 2
    q = torch.empty(FB, P, num_q_heads, head_dim, device=dev, dtype=dtype)
    kv_dtype = torch.float8_e4m3fn if kv_mxfp8 else dtype
    k = torch.empty(
        FPAGES, page_size, num_kv_heads, head_dim, device=dev, dtype=kv_dtype
    )
    v = torch.empty_like(k)
    # Per page and KV head, the 512 scale bytes of 128 tokens x 4 groups.
    k_sf = v_sf = None
    if kv_mxfp8:
        k_sf = torch.empty(
            FPAGES, num_kv_heads, 512, device=dev, dtype=torch.float8_e8m0fnu
        )
        v_sf = torch.empty_like(k_sf)
    o = torch.empty(FB, P, num_q_heads, head_dim, device=dev, dtype=out_dtype)
    lse = (
        torch.empty(FB, P, num_q_heads, device=dev, dtype=torch.float32)
        if with_lse
        else None
    )
    bias = torch.empty(FB, P, num_q_heads, extent, device=dev, dtype=dtype)
    op = torch.empty(
        kv_splits, FB, P, num_q_heads, head_dim, device=dev, dtype=torch.float32
    )
    mp = torch.empty(kv_splits, FB, P, num_q_heads, device=dev, dtype=torch.float32)
    lp = torch.empty_like(mp)
    pt = torch.empty(FB * 4, device=dev, dtype=torch.int32)
    po = torch.empty(FB, device=dev, dtype=torch.int32)
    su = torch.empty(FB, device=dev, dtype=torch.int32)
    tau = torch.empty(FB, P, device=dev, dtype=torch.float32) if has_tau else None
    # Gate and sink are strided views of the query projection.
    gate = torch.empty(FB, P, num_q_heads, head_dim, device=dev, dtype=dtype)
    sink = torch.empty(FB, P, num_q_heads, device=dev, dtype=dtype)
    return cute.compile(
        kernel,
        kv_splits,
        to_cute_tensor(q),
        to_cute_tensor(k),
        to_cute_tensor(v),
        None,  # q_sf
        to_cute_tensor(k_sf) if kv_mxfp8 else None,
        to_cute_tensor(v_sf) if kv_mxfp8 else None,
        to_cute_tensor(o),
        to_cute_tensor(bias),
        to_cute_tensor(op, assumed_align=4),
        to_cute_tensor(mp, assumed_align=4),
        to_cute_tensor(lp, assumed_align=4),
        1.0 / math.sqrt(head_dim),
        to_cute_tensor(tau, assumed_align=4) if has_tau else None,
        None,  # mCuSeqlensQ
        None,  # mCuSeqlensK
        None,  # mSeqUsedQ
        to_cute_tensor(su, assumed_align=4, leading_dim=0),
        None,  # max_seqlen_q
        None,  # max_seqlen_k
        to_cute_tensor(pt, assumed_align=4, leading_dim=0),
        to_cute_tensor(po, assumed_align=4, leading_dim=0),
        to_cute_tensor(lse, assumed_align=4) if with_lse else None,
        to_cute_tensor(gate, assumed_align=4) if gated else None,
        to_cute_tensor(sink, assumed_align=2) if gated else None,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _ladder(B: int, cap: int = 64):
    b = 1
    while b < max(B, 1):
        b <<= 1
    sizes = {b}
    x = 1
    while x <= max(cap, b):
        sizes.add(x)
        x <<= 1
    return sorted(sizes)


def _alloc_workspaces(kv_splits, b, prediction, H, D, device):
    """Allocate the split-KV (O, M, L) partial-result workspace triple."""
    return (
        torch.empty(kv_splits, b, prediction, H, D, device=device, dtype=torch.float32),
        torch.empty(kv_splits, b, prediction, H, device=device, dtype=torch.float32),
        torch.empty(kv_splits, b, prediction, H, device=device, dtype=torch.float32),
    )


def _ensure_buffers(window_left, H, HK, D, dtype, device, page_table, B, prediction=1):
    """Pre-allocate every per-B buffer OUTSIDE graph capture.

    Buffers first allocated INSIDE a capture come from the graph mempool,
    whose blocks are reused by other captures/replays — persistent state
    there gets clobbered (the bs>1 garbage root cause). The static table's
    base pointer is shared by every [:b] slice, so the whole pow2 ladder is
    pre-allocatable from any eager call."""
    P = prediction
    row_stride = page_table.stride(0)
    for b in _ladder(B):
        kv_splits = _kv_splits(window_left, b, P, H, HK, dtype, device)
        wkey = (kv_splits, H, D, device.index, b, P)
        if wkey not in _WORKSPACES:
            _WORKSPACES[wkey] = _alloc_workspaces(kv_splits, b, P, H, D, device)
        okey = (b, row_stride, device.index)
        if okey not in _OFFSETS:
            _OFFSETS[okey] = (
                torch.arange(b, device=device, dtype=torch.int32) * row_stride
            )


def v2_buffers_ready(
    q, window_left, page_table, num_kv_heads: int, prediction: int = 1
) -> bool:
    """Capture-time gate: True iff every buffer this call needs already
    exists (so nothing would be allocated from the capture mempool)."""
    rows, H, D = q.shape
    B = rows // max(prediction, 1)
    kv_splits = _kv_splits(
        window_left, B, prediction, H, num_kv_heads, q.dtype, q.device
    )
    if (kv_splits, H, D, q.device.index, B, prediction) not in _WORKSPACES:
        return False
    if (B, page_table.stride(0), q.device.index) not in _OFFSETS:
        return False
    return True


def is_compiled_for_v2(
    q,
    k_cache,
    rel_logits,
    window_left,
    enable_pdl: bool = True,
    prediction: int = 1,
    has_tau: bool = False,
    *,
    output_dtype: torch.dtype | None,
    return_lse: bool,
    gated: bool,
) -> bool:
    """Whether the config's kernel is already compiled (graph-capture guard)."""
    key = (
        q.shape[1],
        k_cache.shape[2],
        q.shape[-1],
        rel_logits.shape[-1],
        window_left if window_left >= 0 else -1,
        q.dtype,
        q.device.index,
        k_cache.shape[1],
        enable_pdl,
        prediction,
        has_tau,
        output_dtype or q.dtype,
        return_lse,
        gated,
        k_cache.dtype == torch.float8_e4m3fn,
    )
    return key in _DECODE._cache


# Contents are a pure function of the key (write-once), so graphs may bake the addresses.
_OFFSETS: dict = {}


def _page_scale_view(scale: torch.Tensor, cache: torch.Tensor) -> torch.Tensor:
    """``(pages, heads, 512)`` view of a ``(pages, heads, 1, 32, 4, 4)`` UE8M0 slab.

    One 128-token page of one KV head owns 512 contiguous bytes (128 tokens x
    4 groups of 32 head_dim elements), which the kernel loads with one TMA.
    """
    pages, heads = cache.shape[0], cache.shape[2]
    if (
        scale.dtype != torch.float8_e8m0fnu
        or scale.shape != (pages, heads, 1, 32, 4, 4)
        or scale.stride()[3:] != (16, 4, 1)
        or scale.stride(0) % 16
        or scale.stride(1) % 16
    ):
        raise ValueError(
            f"MXFP8 K/V scales must be float8_e8m0fnu ({pages}, {heads}, 1, 32, "
            f"4, 4) with contiguous 512-byte page-head blocks, got {scale.dtype} "
            f"{tuple(scale.shape)} strides {scale.stride()}"
        )
    return torch.as_strided(
        scale, (pages, heads, 512), (scale.stride(0), scale.stride(1), 1)
    )


def _offsets_for(B: int, row_blocks: int, device) -> torch.Tensor:
    key = (B, row_blocks, device.index)
    off = _OFFSETS.get(key)
    if off is None:
        off = _OFFSETS[key] = (
            torch.arange(B, device=device, dtype=torch.int32) * row_blocks
        )
    return off


def rel_mha_decode_tsmha_v2(
    q: torch.Tensor,  # (B * prediction, num_q_heads, head_dim) bf16/fp16
    k_cache: torch.Tensor,  # (num_pages, page_tokens, num_kv_heads, head_dim)
    v_cache: torch.Tensor,
    page_table: torch.Tensor,  # (B, max_pages) int32, -1 holes
    cache_seqlens: torch.Tensor,  # (B,) int32
    rel_logits: torch.Tensor,  # (B * prediction, num_q_heads, extent)
    window_left: int,
    softmax_scale: float | None = None,
    tau: torch.Tensor | None = None,  # (B * prediction,) fp32
    enable_pdl: bool = True,
    prediction: int = 1,
    *,
    output_dtype: torch.dtype | None,
    return_lse: bool,
    gate: torch.Tensor | None,
    sink: torch.Tensor | None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Rel-bias decode on the standalone kernel (native multi-query).

    ``prediction`` is the kernel's native multi-token dimension: q carries
    ``prediction`` token rows per request (token-major), ``cache_seqlens``
    is per REQUEST (total length including all prediction tokens), and the
    kernel masks row t of request b to positions
    ``[.., seq_b - prediction + t]`` in-kernel — KV pages are read once
    per request, not once per row. 1 is plain decode; spec verify passes
    ``spec_num_tokens``.

    ``page_table`` is consumed at its NATIVE page granularity (64, 128, or
    256 tokens — the kernel's sub-page TMA boxes cover all three). It must
    be (a view of) the engine's address-stable per-step table with a dense
    innermost dim (column-sliced row strides are fine) and ``-1`` holes
    only outside the sliding window. ``tau`` is an optional fp32 per-query-row
    multiplier fused onto the total pre-softmax logits,
    ``tau * (softmax_scale * q@k^T + rel)``; values must be positive. Returns
    ``(B * prediction, num_q_heads * head_dim)`` in ``output_dtype`` (None
    keeps the query dtype; torch.float32 stores the unrounded FP32 result),
    plus the natural-log LSE ``(B * prediction, num_q_heads)`` when
    ``return_lse``. With ``gate`` (``(B * prediction, num_q_heads *
    head_dim)``) and ``sink`` (``(B * prediction, num_q_heads)``), rows of the
    query projection, the output is ``(out * sigmoid(gate)) * sigmoid(lse -
    sink)`` from the unrounded FP32 result, as ``rel_mha_gate_sink`` computes
    it, stored once in ``output_dtype``.

    A float8_e4m3fn ``k_cache``/``v_cache`` is an MXFP8 cache: ``k_scale``
    and ``v_scale`` are then its UE8M0 scales, ``(num_pages, num_kv_heads, 1,
    32, 4, 4)`` per ``quantize_store_kv_mxfp8`` (one byte per token, head and
    32 head_dim elements), with 128-token pages, head_dim 128 and a BF16
    query. The kernel dequantizes K and V to BF16 in shared memory; the
    result is the BF16 kernel's on the dequantized cache, bit for bit. Cache
    rows past a request's length may hold anything: they read as zero.
    """
    if (gate is None) != (sink is None):
        raise ValueError("gate and sink go together")
    kv_mxfp8 = k_cache.dtype == torch.float8_e4m3fn
    if (k_scale is not None, v_scale is not None) != (kv_mxfp8, kv_mxfp8):
        raise ValueError(
            "an MXFP8 (float8_e4m3fn) K/V cache takes k_scale and v_scale, a "
            f"{k_cache.dtype} one neither"
        )
    P = max(prediction, 1)
    rows, H, D = q.shape
    B = rows // P
    assert rows == B * P, f"q rows {rows} not divisible by prediction {P}"
    assert cache_seqlens.shape[0] >= B and page_table.shape[0] >= B
    HK = k_cache.shape[2]
    page_tokens = k_cache.shape[1]
    assert page_tokens in (64, 128, 256), f"unsupported page size {page_tokens}"
    if kv_mxfp8:
        if not (
            q.dtype == torch.bfloat16
            and v_cache.dtype == k_cache.dtype
            and page_tokens == 128
            and D == 128
        ):
            raise ValueError(
                "MXFP8 K/V decode needs a BF16 query, FP8 E4M3 K and V, "
                f"128-token pages and head_dim 128 (got {q.dtype}, "
                f"{v_cache.dtype}, page {page_tokens}, head_dim {D})"
            )
        k_scale = _page_scale_view(k_scale, k_cache)
        v_scale = _page_scale_view(v_scale, v_cache)
    # page_table is a column-sliced view; fold row stride into offsets to consume it zero-copy.
    assert page_table.stride(-1) == 1, "page_table innermost dim must be dense"
    extent = rel_logits.shape[-1]
    kv_splits = _kv_splits(window_left, B, P, H, HK, q.dtype, q.device)
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(D)

    compiled = _DECODE.get(
        H,
        HK,
        D,
        extent,
        window_left if window_left >= 0 else -1,
        q.dtype,
        q.device.index,
        page_tokens,
        enable_pdl,
        P,
        tau is not None,
        output_dtype or q.dtype,
        return_lse,
        gate is not None,
        kv_mxfp8,
    )

    row_stride = page_table.stride(0)
    span = (B - 1) * row_stride + page_table.shape[1]
    flat_table = torch.as_strided(page_table, (span,), (1,))
    offsets = _offsets_for(B, row_stride, q.device)

    # Exact-B contiguous: sliced views mis-address the split reduction (dropped tail KV tiles).
    if not torch.cuda.is_current_stream_capturing():
        _ensure_buffers(window_left, H, HK, D, q.dtype, q.device, page_table, B, P)
    key = (kv_splits, H, D, q.device.index, B, P)
    ws = _WORKSPACES.get(key)
    if ws is None:
        ws = _WORKSPACES[key] = _alloc_workspaces(kv_splits, B, P, H, D, q.device)
    op_w, mp_w, lp_w = ws
    o = torch.empty(B, P, H, D, device=q.device, dtype=output_dtype or q.dtype)
    lse = (
        torch.empty(B, P, H, device=q.device, dtype=torch.float32)
        if return_lse
        else None
    )

    # TVM FFI carries FP8 data as uint8 storage.
    compiled(
        kv_splits,
        q.view(B, P, H, D),
        k_cache.view(torch.uint8) if kv_mxfp8 else k_cache,
        v_cache.view(torch.uint8) if kv_mxfp8 else v_cache,
        None,
        k_scale,
        v_scale,
        o,
        rel_logits.view(B, P, H, extent),
        op_w,
        mp_w,
        lp_w,
        softmax_scale,
        tau.view(B, P) if tau is not None else None,
        None,
        None,
        None,
        cache_seqlens[:B],
        None,
        None,
        flat_table,
        offsets,
        lse,
        None if gate is None else gate.view(B, P, H, D),
        None if sink is None else sink.view(B, P, H),
    )
    if return_lse:
        return o.view(B * P, H * D), lse.view(B * P, H)
    return o.view(B * P, H * D)
