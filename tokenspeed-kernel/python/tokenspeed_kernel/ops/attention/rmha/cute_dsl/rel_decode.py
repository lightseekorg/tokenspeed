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

"""Sliding-window rel-bias decode through the tokenspeed-mha fwd kernel.

One query token per request (max_seqlen_q == 1). The sheared-bias prepass is
a small triton gather: for the query of request b at absolute position
``seqused[b] - 1`` (phase ``p = pos % 128``), physical bias column ``c`` of the
``extent + 256``-wide row maps to relative distance

    rel = (n_tiles - 1 - c // 128) * 128 + p - c % 128

taking ``rel_table[h, rel]`` when ``0 <= rel < extent``, ``-inf`` for negative
distances, ``0`` beyond the extent (the fwd kernel applies the causal/window
mask itself). This matches flash_attn's ShearingBias layout, so
the fwd kernel consumes it unchanged.

The fwd kernel is compiled once per static config with TVM FFI enabled, so
runtime calls take torch tensors, dynamic batch/page-count, and are
CUDA-graph safe (seqlens are read from device memory).
"""

from __future__ import annotations

import math
import os as _os

import torch
from tokenspeed_kernel._triton import tl, triton


@triton.jit
def _shear_decode_kernel(
    rel_ptr,  # (B * P, H, EXTENT) compact per-head rel table per query row
    out_ptr,  # (B, R>=P, H, EXTENT_PADDED) sheared tile rows; strided dims ok
    seqused_ptr,  # (B,) int32 visible KV lengths (incl. all P tokens)
    out_batch_stride,
    out_row_stride,
    P,  # query tokens per request
    H: tl.constexpr,
    EXTENT: tl.constexpr,
    EXTENT_PADDED: tl.constexpr,
    BLOCK: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    i = tl.program_id(0)  # flattened (request, query-row)
    h = tl.program_id(1)
    cb = tl.program_id(2)
    b = i // P
    t = i - b * P
    cols = cb * BLOCK + tl.arange(0, BLOCK)
    # rel/seqused predate the KV-store producer, so this gather is safe PDL prologue overlap.
    seq = tl.load(seqused_ptr + b)
    # Query row t sits at absolute position seq - P + t. The fwd kernel
    # anchors the bias tile columns at the LAST query row's 128-tile, so
    # every row's phase is expressed relative to that anchor — negative
    # when the P rows straddle a tile boundary (rel < 0 stays -inf, which
    # the causal mask covers anyway).
    anchor = ((seq - 1) // 128) * 128
    phase = seq - P + t - anchor
    n_tiles: tl.constexpr = EXTENT_PADDED // 128
    rel = (n_tiles - 1 - cols // 128) * 128 + phase - cols % 128

    in_extent = (rel >= 0) & (rel < EXTENT)
    vals = tl.load(
        rel_ptr + (i * H + h) * EXTENT + tl.where(in_extent, rel, 0),
        mask=in_extent,
        other=0.0,
    )
    # ShearingBias match: rel<0 -> -inf; >=extent stays 0 or masked-tile row maxima are poisoned.
    out = tl.where(rel < 0, float("-inf"), vals.to(tl.float32))
    if ENABLE_PDL:
        # Conservative alias guard: complete the producer before we store.
        tl.extra.cuda.gdc_wait()
    tl.store(
        out_ptr + b * out_batch_stride + t * out_row_stride + h * EXTENT_PADDED + cols,
        out.to(out_ptr.dtype.element_ty),
        mask=cols < EXTENT_PADDED,
    )
    if ENABLE_PDL:
        # All stores issued; let the dependent kernel begin its prologue.
        tl.extra.cuda.gdc_launch_dependents()


def shear_decode_bias(
    rel: torch.Tensor,
    seqused: torch.Tensor,
    out: torch.Tensor,
    enable_pdl: bool = False,
    prediction: int = 1,
) -> None:
    """Fill the fwd kernel's bias tile with sheared rows for batched decode.

    Each request's ``prediction`` query rows land in tile rows
    ``0..prediction-1``, phase-anchored to the LAST row's 128-tile (rows
    that straddle a tile boundary get negative phases, whose ``rel < 0``
    region is -inf — inside the causal mask either way). ``prediction=1``
    is plain decode: one row at phase ``(seqused - 1) % 128``.

    Args:
        rel: Compact rel-bias rows, (B * prediction, H, extent), bf16/fp16.
        seqused: Visible KV length per request incl. all prediction tokens,
            (B,) int32.
        out: Preallocated bias tile slice, same dtype as rel:
            (B, H, extent + 256) for prediction == 1 (the tile's row 0), or
            (B, R >= prediction, H, extent + 256) for prediction > 1.
    """
    rows, H, extent = rel.shape
    P = max(prediction, 1)
    B = rows // P
    assert rows == B * P
    extent_padded = extent + 256
    BLOCK = 128
    assert out.stride(-1) == 1 and out.stride(-2) == extent_padded
    if out.dim() == 4:
        out_batch_stride, out_row_stride = out.stride(0), out.stride(1)
    else:
        assert P == 1, "prediction > 1 needs the 4-D bias tile slice"
        out_batch_stride, out_row_stride = out.stride(0), 0
    # HIP triton has no launch_pdl; PDL is NVIDIA-only.
    use_pdl = enable_pdl and torch.version.hip is None
    kwargs = {}
    if use_pdl:
        kwargs["launch_pdl"] = True
    _shear_decode_kernel[(B * P, H, extent_padded // BLOCK)](
        rel,
        out,
        seqused,
        out_batch_stride,
        out_row_stride,
        P,
        H=H,
        EXTENT=extent,
        EXTENT_PADDED=extent_padded,
        BLOCK=BLOCK,
        ENABLE_PDL=use_pdl,
        **kwargs,
    )


class _FwdCache:
    """Compile cache for the varlen paged bf16 rel-bias fwd kernel."""

    def __init__(self):
        self._cache = {}

    def get(
        self,
        num_q_heads,
        num_kv_heads,
        head_dim,
        extent,
        is_local,
        dtype,
        num_splits,
        use_pdl,
        blocks_per_page=1,
        blockscaled=False,
    ):
        key = (
            num_q_heads,
            num_kv_heads,
            head_dim,
            extent,
            is_local,
            dtype,
            num_splits,
            use_pdl,
            blocks_per_page,
            blockscaled,
        )
        if key not in self._cache:
            self._cache[key] = _compile_fwd(*key)
        return self._cache[key]


def _compile_fwd(
    num_q_heads,
    num_kv_heads,
    head_dim,
    extent,
    is_local,
    dtype,
    num_splits=1,
    use_pdl=False,
    blocks_per_page=1,
    blockscaled=False,
):
    """``dtype`` is the bias/output dtype. With ``blockscaled`` the data
    tensors are fp8-e4m3 (MXFP8: UE8M0 vec-32 scales, fp8 V dequantized
    in-kernel to ``dtype`` for the P@V MMA)."""
    import cutlass.cute as cute
    from flash_attn.cute.cute_dsl_utils import to_cute_tensor

    from .flash_fwd_sm100_bias import FlashAttentionForwardSm100

    kernel = FlashAttentionForwardSm100(
        head_dim=head_dim,
        head_dim_v=head_dim,
        qhead_per_kvhead=num_q_heads // num_kv_heads,
        is_causal=not is_local,
        is_local=is_local,
        is_split_kv=num_splits > 1,
        pack_gqa=True,
        m_block_size=128,
        n_block_size=128,
        bias_block_size=128,
        q_stage=1,
        is_persistent=False,
        has_bias=True,
        rel_extent_padded=extent + 256,
        paged_kv_blocks_per_page=blocks_per_page,
        use_pdl=use_pdl,
        qk_blockscaled=blockscaled,
        v_dequant=blockscaled,
        q_sf_interleaved=False,
        kv_sf_interleaved=blockscaled,
    )
    # TVM FFI + dynamic layouts: the artifact reuses across batch size, page count, seqlens.
    B, PAGES, PAGE = 2, 4, 128 * blocks_per_page
    dev = torch.device("cuda")
    data_dtype = torch.float8_e4m3fn if blockscaled else dtype
    q = torch.empty(B, 1, num_q_heads, head_dim, device=dev, dtype=data_dtype)
    if num_splits > 1:
        o = torch.empty(
            num_splits, B, 1, num_q_heads, head_dim, device=dev, dtype=torch.float32
        )
        lse = torch.empty(
            num_splits, B, num_q_heads, 1, device=dev, dtype=torch.float32
        )
    else:
        o = torch.empty(B, 1, num_q_heads, head_dim, device=dev, dtype=dtype)
        lse = None
    k = torch.empty(PAGES, PAGE, num_kv_heads, head_dim, device=dev, dtype=data_dtype)
    v = torch.empty_like(k)
    bias = torch.empty(B, 128, num_q_heads, extent + 256, device=dev, dtype=dtype)
    pt = torch.empty(B, PAGES // B, device=dev, dtype=torch.int32)
    seqused = torch.empty(B, device=dev, dtype=torch.int32)
    if blockscaled:
        sf_dim = head_dim // 32
        sfq = torch.empty(
            B, 1, num_q_heads, sf_dim, device=dev, dtype=torch.float8_e8m0fnu
        )
        sfk = torch.empty(
            PAGES,
            num_kv_heads,
            blocks_per_page,
            32,
            sf_dim,
            sf_dim,
            device=dev,
            dtype=torch.float8_e8m0fnu,
        )
        sfv = torch.empty_like(sfk)

    # Leading-dim pinned (fork convention); fully-dynamic strides mis-tile the bias TMA for B>0.
    m_q, m_k, m_v, m_o, m_bias = [to_cute_tensor(t) for t in (q, k, v, o, bias)]
    if blockscaled:
        m_sfq, m_sfk, m_sfv = [
            to_cute_tensor(t, assumed_align=4) for t in (sfq, sfk, sfv)
        ]
        sf_vec = 32
    else:
        m_sfq = m_sfk = m_sfv = sf_vec = None
    m_pt, m_seqused = [
        to_cute_tensor(t, assumed_align=4, leading_dim=t.ndim - 1)
        for t in (pt, seqused)
    ]
    m_lse = to_cute_tensor(lse, assumed_align=4) if lse is not None else None
    compile_args = (
        m_q,
        m_k,
        m_v,
        m_o,
        m_lse,  # mLSE (split partials when num_splits > 1)
        None,  # mRowMax
        1.0 / math.sqrt(head_dim),
        m_sfq,
        m_sfk,
        m_sfv,
        sf_vec,
        sf_vec,  # qk/v sf vec sizes (constexpr)
        None,  # mCuSeqlensQ: batch mode
        None,  # mCuSeqlensK
        None,  # mSeqUsedQ
        m_seqused,  # mSeqUsedK
        m_pt,  # mPageTable
        extent - 1 if is_local else None,  # window_size_left (runtime Int32)
        0 if is_local else None,  # window_size_right
        None,  # learnable_sink
        None,  # blocksparse
        None,  # aux
        m_bias,
        None,  # num_splits_dynamic_ptr
        None,  # tile_count_semaphore
        1,  # max_seqlen_q (runtime Int32)
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
    )
    return cute.compile(kernel, *compile_args, options="--enable-tvm-ffi")


class _CombineCache:
    """Compile cache for the split-KV combine kernel."""

    def __init__(self):
        self._cache = {}

    def get(self, head_dim, dtype, num_splits, use_pdl):
        key = (head_dim, dtype, num_splits, use_pdl)
        if key not in self._cache:
            self._cache[key] = _compile_combine(*key)
        return self._cache[key]


def _compile_combine(head_dim, dtype, num_splits, use_pdl=False):
    import cutlass
    import cutlass.cute as cute
    from cutlass import Float32
    from quack.compile_utils import make_fake_tensor as fake_tensor

    from .flash_fwd_combine import FlashAttentionForwardCombine

    k_block_size = 64 if head_dim <= 64 else 128
    tile_m = 8 if k_block_size % 128 == 0 else (16 if k_block_size % 64 == 0 else 32)
    log_max_splits = max(math.ceil(math.log2(num_splits)), 4)
    if tile_m == 8:
        log_max_splits = max(log_max_splits, 5)
    out_dtype = cutlass.BFloat16 if dtype == torch.bfloat16 else cutlass.Float16
    fa_combine = FlashAttentionForwardCombine(
        dtype=out_dtype,
        dtype_partial=Float32,
        head_dim=head_dim,
        tile_m=tile_m,
        k_block_size=k_block_size,
        log_max_splits=log_max_splits,
        use_pdl=use_pdl,
    )
    # Mirror the wheel's _compile_fwd_combine; artifact stays reusable across batch/split count.
    sym = cute.sym_int
    div = 128 // Float32.width
    num_splits_s, batch, seqlen, nheads = sym(), sym(), sym(), sym()
    mO_partial = fake_tensor(
        Float32, (num_splits_s, batch, seqlen, nheads, head_dim), divisibility=div
    )
    mLSE_partial = fake_tensor(
        Float32, (num_splits_s, batch, seqlen, nheads), divisibility=1, leading_dim=2
    )
    mO = fake_tensor(out_dtype, (batch, seqlen, nheads, head_dim), divisibility=div)
    return cute.compile(
        fa_combine,
        mO_partial,
        mLSE_partial,
        mO,
        None,
        None,
        None,
        None,
        None,
        None,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


_FWD = _FwdCache()
_COMBINE = _CombineCache()
_BIAS_SCRATCH: dict = {}
_BIAS_SCRATCH_RETIRED: list = []
_PARTIAL_SCRATCH: dict = {}

# Splits for full-attention decode; 1 disables splitting (A/B knob).
_FULL_ATTN_SPLITS = max(1, int(_os.environ.get("TSMHA_FULL_SPLITS", "16")))


def rel_mha_decode_tsmha(
    q: torch.Tensor,  # (B * prediction, num_q_heads, head_dim) bf16
    k_cache: torch.Tensor,  # (num_pages, 128, num_kv_heads, head_dim)
    v_cache: torch.Tensor,
    page_table: torch.Tensor,  # (B, max_pages) int32
    cache_seqlens: torch.Tensor,  # (B,) int32
    rel_logits: torch.Tensor,  # (B * prediction, num_q_heads, extent)
    cu_seqlens_q: torch.Tensor,  # unused (batch mode); kept for call parity
    window_left: int,
    softmax_scale: float | None = None,
    enable_pdl: bool = False,
    q_scale: torch.Tensor | None = None,  # (B * P, H, D // 32) e8m0
    k_scale: torch.Tensor | None = None,  # (pages, h_kv, k, 32, 4, 4) e8m0
    v_scale: torch.Tensor | None = None,  # same layout as k_scale
    prediction: int = 1,
) -> torch.Tensor:
    """Batched rel-bias decode on the tokenspeed-mha kernel.

    Batch mode: q is viewed (B, prediction, H, D) and the sheared bias
    carries one 128-row tile per request (rows 0..prediction-1 hold the
    query rows, phase-corrected per row by the triton shear); the compiled
    fwd applies bottom-right causal/window masking itself, so row t of
    request b sees keys up to position ``seq_b - prediction + t``.
    ``prediction=1`` is plain decode. Varlen is deliberately avoided — the
    kernel's multi-request varlen path mis-addresses KV for request > 0.
    Requires page_size % 128 == 0 and extent % 128 == 0.

    Providing ``q_scale``/``k_scale``/``v_scale`` selects the MXFP8
    block-scaled path: q/k_cache/v_cache must be float8_e4m3fn, scales are
    UE8M0 vec-32 (KV scales in the paged interleaved layout, one
    BlockScaledBasicChunk atom per in-page 128-row chunk), and the output
    is the bias dtype (bf16/fp16).

    ``enable_pdl`` launches the shear, fwd, and combine kernels with
    Programmatic Dependent Launch (Hopper+); pass
    ``tokenspeed.runtime.utils.pdl.pdl_enabled()`` from the runtime.
    """
    P = max(prediction, 1)
    rows, H, D = q.shape
    B = rows // P
    assert rows == B * P, f"q rows {rows} not divisible by prediction {P}"
    assert P <= 128, "prediction rows must fit the 128-row bias tile"
    extent = rel_logits.shape[-1]
    is_local = window_left >= 0
    page_size = k_cache.shape[1]
    assert (
        page_size % 128 == 0
    ), f"tokenspeed-mha paged TMA needs page_size % 128 == 0, got {page_size}"
    blocks_per_page = page_size // 128
    blockscaled = q_scale is not None
    assert not blockscaled or (k_scale is not None and v_scale is not None)
    out_dtype = rel_logits.dtype if blockscaled else q.dtype

    key = (H, extent, out_dtype, q.device.index)
    bias = _BIAS_SCRATCH.get(key)
    if bias is None or bias.shape[0] < B:
        if bias is not None:
            # Smaller-B capture graphs hold pointers here; retain so replays store into live memory.
            _BIAS_SCRATCH_RETIRED.append(bias)
        bias = torch.zeros(
            max(B, 8), 128, H, extent + 256, device=q.device, dtype=out_dtype
        )
        _BIAS_SCRATCH[key] = bias
    shear_decode_bias(
        rel_logits,
        cache_seqlens,
        bias[:B, 0] if P == 1 else bias[:B],
        enable_pdl=enable_pdl,
        prediction=P,
    )

    # Split KV + combine for full attention (~2 CTAs scan whole context); SWA gains nothing.
    num_splits = 1 if is_local else _FULL_ATTN_SPLITS
    compiled = _FWD.get(
        H,
        k_cache.shape[2],
        D,
        extent,
        is_local,
        out_dtype,
        num_splits,
        enable_pdl,
        blocks_per_page,
        blockscaled,
    )
    o = torch.empty(q.shape, device=q.device, dtype=out_dtype)
    if num_splits > 1:
        pkey = (B, H, D, num_splits, q.device.index, P)
        partials = _PARTIAL_SCRATCH.get(pkey)
        if partials is None:
            partials = (
                torch.empty(
                    num_splits, B, P, H, D, device=q.device, dtype=torch.float32
                ),
                torch.empty(num_splits, B, H, P, device=q.device, dtype=torch.float32),
            )
            _PARTIAL_SCRATCH[pkey] = partials
        o_arg, lse_arg = partials
    else:
        o_arg, lse_arg = o.view(B, P, H, D), None
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(D)
    if blockscaled:
        # TVM-FFI carries fp8 as uint8 storage; SFQ rides the 4D batch-mode Q transpose.
        data = lambda t: t.view(torch.uint8)
        sf_args = (q_scale.view(B, P, H, D // 32), k_scale, v_scale)
    else:
        data = lambda t: t
        sf_args = (None, None, None)
    compiled(
        data(q).view(B, P, H, D),
        data(k_cache),
        data(v_cache),
        o_arg,
        lse_arg,
        None,
        scale,
        *sf_args,
        None,  # mCuSeqlensQ: batch mode
        None,
        None,
        cache_seqlens[:B],
        page_table,
        window_left if is_local else None,
        0 if is_local else None,
        None,
        None,
        None,
        bias[:B],
        None,
        None,
        P,  # max_seqlen_q (runtime)
    )
    if num_splits > 1:
        _COMBINE.get(D, out_dtype, num_splits, enable_pdl)(
            o_arg,
            lse_arg.transpose(-1, -2),
            o.view(B, P, H, D),
            None,
            None,
            None,
            None,
            None,
            None,
        )
    return o


def is_compiled_for(
    q, k_cache, rel_logits, window_left, enable_pdl=False, blockscaled=False
) -> bool:
    """True when both prepass and fwd kernels for this config are cached."""
    H = q.shape[1]
    extent = rel_logits.shape[-1]
    is_local = window_left >= 0
    num_splits = 1 if is_local else _FULL_ATTN_SPLITS
    out_dtype = rel_logits.dtype if blockscaled else q.dtype
    fwd_key = (
        H,
        k_cache.shape[2],
        q.shape[-1],
        extent,
        is_local,
        out_dtype,
        num_splits,
        enable_pdl,
        k_cache.shape[1] // 128,
        blockscaled,
    )
    if fwd_key not in _FWD._cache:
        return False
    if num_splits > 1:
        return (q.shape[-1], out_dtype, num_splits, enable_pdl) in _COMBINE._cache
    return True


def warmup(
    num_q_heads,
    num_kv_heads,
    head_dim,
    extents,
    dtype=torch.bfloat16,
    enable_pdl=False,
    blocks_per_page=1,
    blockscaled=False,
):
    """Pre-compile shear + fwd kernels outside CUDA-graph capture.

    Call from attention-backend init (device already selected, capture not
    yet started): importing the kernel modules remounts flash_attn.cute and
    cute.compile allocates — both are unsafe mid-capture and unsafe before
    per-rank device setup. Pass the same ``enable_pdl`` the runtime will use
    at call time so the cached artifacts match.
    """
    dev = torch.device("cuda")
    for extent, is_local in extents:
        _FWD.get(
            num_q_heads,
            num_kv_heads,
            head_dim,
            extent,
            is_local,
            dtype,
            1 if is_local else _FULL_ATTN_SPLITS,
            enable_pdl,
            blocks_per_page,
            blockscaled,
        )
        if not is_local and _FULL_ATTN_SPLITS > 1:
            _COMBINE.get(head_dim, dtype, _FULL_ATTN_SPLITS, enable_pdl)
        rel = torch.zeros(1, num_q_heads, extent, device=dev, dtype=dtype)
        bias = torch.zeros(1, 128, num_q_heads, extent + 256, device=dev, dtype=dtype)
        seqused = torch.ones(1, device=dev, dtype=torch.int32)
        shear_decode_bias(
            rel,
            seqused,
            bias[:1, 0],
            enable_pdl=enable_pdl,
        )
