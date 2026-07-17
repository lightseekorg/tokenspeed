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

"""Varlen rel-bias prefill/extend through the tokenspeed-mha fwd kernel.

Multi-row packed queries (``cu_seqlens_q``) against packed KV (prefill) or
the paged KV cache (extend, page_size 128). The fwd kernel is compiled with
``is_varlen_q=True`` — the constructor flag, not the presence of cu_seqlens,
selects the varlen tile scheduler; without it requests > 0 are mis-mapped.

The sheared-bias prepass is the ported ShearingBias kernel (varlen
configuration), fed by two prep kernels that build the block prefix sums
over ``cu_seqlens_q`` and the block-to-batch index map. ShearingBias anchors
each 128-row q-tile's rightmost bias block to key block ``n_block_max - 1``
of that tile (``n_block_max = min(ceil(seqlen_k/128), tile-end diagonal
block + 1)``); with ``has_bias`` the attention kernel applies no explicit
causal mask on diagonal blocks — the sheared ``-inf`` pattern is the mask —
so this anchor convention is load-bearing.

All kernels compile once per static config with TVM FFI enabled; runtime
calls take torch tensors with dynamic batch, row, and page counts.
"""

from __future__ import annotations

import math

import torch

_FWD_VARLEN: dict = {}
_SHEAR: dict = {}
_PREP: dict = {}


def _compile_fwd_varlen(
    num_q_heads,
    num_kv_heads,
    head_dim,
    extent,
    is_local,
    dtype,
    paged,
    use_pdl=False,
    blocks_per_page=1,
    blockscaled=False,
):
    """``dtype`` is the bias/output dtype. ``blockscaled`` (paged only)
    compiles the MXFP8 variant: fp8-e4m3 q/k/v, UE8M0 vec-32 scales (flat
    per-token SFQ — interleaved SFQ is forbidden with varlen q — and paged
    interleaved KV scales), fp8 V dequantized in-kernel to ``dtype``."""
    assert not blockscaled or paged, "blockscaled varlen requires the paged path"
    import cutlass.cute as cute
    from flash_attn.cute.cute_dsl_utils import to_cute_tensor

    from .flash_fwd_sm100_bias import FlashAttentionForwardSm100

    kernel = FlashAttentionForwardSm100(
        head_dim=head_dim,
        head_dim_v=head_dim,
        qhead_per_kvhead=num_q_heads // num_kv_heads,
        is_causal=not is_local,
        is_local=is_local,
        is_split_kv=False,
        pack_gqa=False,
        m_block_size=128,
        n_block_size=128,
        bias_block_size=128,
        q_stage=2,
        is_persistent=False,
        is_varlen_q=True,
        has_bias=True,
        rel_extent_padded=extent + 256,
        use_pdl=use_pdl,
        paged_kv_blocks_per_page=blocks_per_page,
        qk_blockscaled=blockscaled,
        v_dequant=blockscaled,
        q_sf_interleaved=False,
        kv_sf_interleaved=blockscaled,
    )
    # TVM FFI + dynamic layouts: the artifact reuses across batch, rows, pages, seqlens.
    B, TOTAL_Q, PAGES, PAGE = 2, 256, 4, 128 * blocks_per_page
    dev = torch.device("cuda")
    data_dtype = torch.float8_e4m3fn if blockscaled else dtype
    q = torch.empty(TOTAL_Q, num_q_heads, head_dim, device=dev, dtype=data_dtype)
    o = torch.empty(TOTAL_Q, num_q_heads, head_dim, device=dev, dtype=dtype)
    bias = torch.empty(
        TOTAL_Q + 128, num_q_heads, extent + 256, device=dev, dtype=dtype
    )
    cu_q = torch.empty(B + 1, device=dev, dtype=torch.int32)
    m_q, m_o, m_bias = [to_cute_tensor(t) for t in (q, o, bias)]
    m_cu_q = to_cute_tensor(cu_q, assumed_align=4, leading_dim=0)
    if paged:
        k = torch.empty(
            PAGES, PAGE, num_kv_heads, head_dim, device=dev, dtype=data_dtype
        )
        v = torch.empty_like(k)
        pt = torch.empty(B, PAGES // B, device=dev, dtype=torch.int32)
        seqused = torch.empty(B, device=dev, dtype=torch.int32)
        m_k, m_v = [to_cute_tensor(t) for t in (k, v)]
        m_pt, m_used_k = [
            to_cute_tensor(t, assumed_align=4, leading_dim=t.ndim - 1)
            for t in (pt, seqused)
        ]
        m_cu_k = None
    else:
        k = torch.empty(TOTAL_Q, num_kv_heads, head_dim, device=dev, dtype=dtype)
        v = torch.empty_like(k)
        m_k, m_v = [to_cute_tensor(t) for t in (k, v)]
        m_pt, m_used_k, m_cu_k = None, None, m_cu_q
    if blockscaled:
        sf_dim = head_dim // 32
        sfq = torch.empty(
            TOTAL_Q, num_q_heads, sf_dim, device=dev, dtype=torch.float8_e8m0fnu
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
        m_sfq, m_sfk, m_sfv = [
            to_cute_tensor(t, assumed_align=4) for t in (sfq, sfk, sfv)
        ]
        sf_vec = 32
    else:
        m_sfq = m_sfk = m_sfv = sf_vec = None
    compile_args = (
        m_q,
        m_k,
        m_v,
        m_o,
        None,  # mLSE
        None,  # mRowMax
        1.0 / math.sqrt(head_dim),
        m_sfq,
        m_sfk,
        m_sfv,
        sf_vec,
        sf_vec,  # qk/v sf vec sizes (constexpr)
        m_cu_q,  # mCuSeqlensQ
        m_cu_k,  # mCuSeqlensK (packed prefill)
        None,  # mSeqUsedQ
        m_used_k,  # mSeqUsedK (paged extend)
        m_pt,  # mPageTable
        extent - 1 if is_local else None,  # window_size_left (runtime Int32)
        0 if is_local else None,  # window_size_right
        None,  # learnable_sink
        None,  # blocksparse
        None,  # aux
        m_bias,
        None,  # num_splits_dynamic_ptr
        None,  # tile_count_semaphore
        128,  # max_seqlen_q (runtime Int32)
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
    )
    return cute.compile(kernel, *compile_args, options="--enable-tvm-ffi")


def _compile_shear(extent, is_local, dtype, paged):
    import cutlass.cute as cute
    from flash_attn.cute.cute_dsl_utils import to_cute_tensor

    from .shearing_bias import ShearingBias

    shear = ShearingBias(
        rel_extent=extent,
        is_causal=not is_local,
        is_local=is_local,
        pack_gqa=False,
        qhead_per_kvhead=1,
        rows_per_cta=4,
        tile_m=128,
        max_m_blocks_leq_one=False,
    )
    B, TOTAL_Q, H = 2, 256, 2
    dev = torch.device("cuda")
    rel = torch.empty(TOTAL_Q, H, extent, device=dev, dtype=dtype)
    bias = torch.empty(TOTAL_Q + 128, H, extent + 256, device=dev, dtype=dtype)
    cu_q = torch.empty(B + 1, device=dev, dtype=torch.int32)
    cu_blocks = torch.empty(B + 1, device=dev, dtype=torch.int32)
    b2b = torch.empty(TOTAL_Q // 128 + B, device=dev, dtype=torch.int32)
    m_rel, m_bias = [to_cute_tensor(t) for t in (rel, bias)]
    m_cu_q, m_cu_blocks, m_b2b = [
        to_cute_tensor(t, assumed_align=4, leading_dim=0)
        for t in (cu_q, cu_blocks, b2b)
    ]
    if paged:
        seqused_k = torch.empty(B, device=dev, dtype=torch.int32)
        m_used_k = to_cute_tensor(seqused_k, assumed_align=4, leading_dim=0)
        m_cu_k = None
    else:
        m_cu_k, m_used_k = m_cu_q, None
    compile_args = (
        m_rel,
        m_bias,
        128,  # max_seqlen_q (runtime Int32)
        128,  # max_seqlen_k (runtime Int32)
        m_cu_q,
        m_cu_k,
        None,  # mSeqUsedQ
        m_used_k,
        m_cu_blocks,
        m_b2b,
        extent - 1 if is_local else None,  # window_size_left (runtime Int32)
        0 if is_local else None,  # window_size_right
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
    )
    return cute.compile(shear, *compile_args, options="--enable-tvm-ffi")


def _compile_prep():
    import cutlass.cute as cute
    from flash_attn.cute.cute_dsl_utils import to_cute_tensor

    from .cu_blocks_kernels import (
        CuBlocksToBatchKernel,
        CuSeqlensToBlocksKernel,
    )

    B, BLOCKS = 2, 4
    dev = torch.device("cuda")
    cu_blocks = torch.empty(B + 1, device=dev, dtype=torch.int32)
    cu_q = torch.empty(B + 1, device=dev, dtype=torch.int32)
    b2b = torch.empty(BLOCKS, device=dev, dtype=torch.int32)
    t_blocks, t_cu, t_b2b = [
        to_cute_tensor(t, assumed_align=4, leading_dim=0)
        for t in (cu_blocks, cu_q, b2b)
    ]
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    cu2blocks = cute.compile(
        CuSeqlensToBlocksKernel(tile=128, seqlen_multiple=1),
        t_blocks,
        t_cu,
        stream,
        options="--enable-tvm-ffi",
    )
    blocks2batch = cute.compile(
        CuBlocksToBatchKernel(), t_blocks, t_b2b, stream, options="--enable-tvm-ffi"
    )
    return cu2blocks, blocks2batch


def _get_prep():
    if "prep" not in _PREP:
        _PREP["prep"] = _compile_prep()
    return _PREP["prep"]


def is_compiled_for(
    q,
    num_kv_heads,
    extent,
    is_local,
    paged,
    enable_pdl=False,
    blocks_per_page=1,
    blockscaled=False,
    out_dtype=None,
) -> bool:
    """True when the prep, shear, and fwd kernels for this config are cached."""
    dtype = out_dtype or (torch.bfloat16 if blockscaled else q.dtype)
    fwd_key = (
        q.shape[1],
        num_kv_heads,
        q.shape[-1],
        extent,
        is_local,
        dtype,
        paged,
        enable_pdl,
        blocks_per_page,
        blockscaled,
    )
    shear_key = (extent, is_local, dtype, paged)
    return fwd_key in _FWD_VARLEN and shear_key in _SHEAR and "prep" in _PREP


def rel_mha_varlen_tsmha(
    q: torch.Tensor,  # (total_q, num_q_heads, head_dim) bf16/fp16
    k: torch.Tensor,  # (total_k, h_kv, d) packed, or (num_pages, 128, h_kv, d)
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,  # (B + 1,) int32
    max_seqlen_q: int,
    rel_logits: torch.Tensor,  # (total_q, num_q_heads, extent)
    window_left: int,
    max_seqlen_k: int,
    cu_seqlens_k: torch.Tensor | None = None,  # packed prefill
    page_table: torch.Tensor | None = None,  # (B, max_pages) int32, paged extend
    cache_seqlens: torch.Tensor | None = None,  # (B,) int32, paged extend
    softmax_scale: float | None = None,
    enable_pdl: bool = False,
    q_scale: torch.Tensor | None = None,  # (total_q, H, D // 32) e8m0
    k_scale: torch.Tensor | None = None,  # (pages, h_kv, k, 32, 4, 4) e8m0
    v_scale: torch.Tensor | None = None,  # same layout as k_scale
) -> torch.Tensor:
    """Multi-row rel-bias attention over packed varlen queries.

    Exactly one of ``cu_seqlens_k`` (packed prefill, K/V share the query
    packing) or ``page_table`` + ``cache_seqlens`` (extend against the paged
    cache) must be provided. Requires ``extent % 128 == 0`` and, for the
    paged path, page_size 128. Sliding-window callers must satisfy
    ``window_left + 1 == extent`` (the sheared table is sliced to the
    window). Returns the packed (total_q, num_q_heads, head_dim) output.

    ``enable_pdl`` launches the fwd kernel with Programmatic Dependent
    Launch (Hopper+); pass ``tokenspeed.runtime.utils.pdl.pdl_enabled()``
    from the runtime.
    """
    total_q, H, D = q.shape
    B = cu_seqlens_q.shape[0] - 1
    extent = rel_logits.shape[-1]
    is_local = window_left >= 0
    paged = page_table is not None
    assert paged == (cache_seqlens is not None) and paged != (cu_seqlens_k is not None)
    blockscaled = q_scale is not None
    assert not blockscaled or (
        paged and k_scale is not None and v_scale is not None
    ), "MXFP8 varlen requires the paged path and all three scale tensors"
    out_dtype = rel_logits.dtype if blockscaled else q.dtype
    blocks_per_page = 1
    if paged:
        assert (
            k.shape[1] % 128 == 0
        ), f"tokenspeed-mha paged TMA needs page_size % 128 == 0, got {k.shape[1]}"
        blocks_per_page = k.shape[1] // 128
    num_kv_heads = k.shape[2] if paged else k.shape[1]
    dev = q.device
    if not rel_logits.is_contiguous():
        rel_logits = rel_logits.contiguous()

    cu2blocks, blocks2batch = _get_prep()
    shear_key = (extent, is_local, out_dtype, paged)
    shear = _SHEAR.get(shear_key)
    if shear is None:
        shear = _SHEAR[shear_key] = _compile_shear(extent, is_local, out_dtype, paged)
    fwd_key = (
        H,
        num_kv_heads,
        D,
        extent,
        is_local,
        out_dtype,
        paged,
        enable_pdl,
        blocks_per_page,
        blockscaled,
    )
    fwd = _FWD_VARLEN.get(fwd_key)
    if fwd is None:
        fwd = _FWD_VARLEN[fwd_key] = _compile_fwd_varlen(
            H,
            num_kv_heads,
            D,
            extent,
            is_local,
            out_dtype,
            paged,
            enable_pdl,
            blocks_per_page,
            blockscaled,
        )

    cu_blocks = torch.empty(B + 1, device=dev, dtype=torch.int32)
    total_blocks_max = (total_q + B * 127) // 128
    b2b = torch.empty(max(total_blocks_max, 1), device=dev, dtype=torch.int32)
    cu2blocks(cu_blocks, cu_seqlens_q)
    blocks2batch(cu_blocks, b2b)

    # Tail rows past total_q are TMA-read; garbage there only feeds masked-out rows.
    bias = torch.empty(total_q + 128, H, extent + 256, device=dev, dtype=out_dtype)
    seqused_k = cache_seqlens if paged else None
    shear(
        rel_logits,
        bias,
        max_seqlen_q,
        max_seqlen_k,
        cu_seqlens_q,
        cu_seqlens_k,
        None,
        seqused_k,
        cu_blocks,
        b2b,
        window_left if is_local else None,
        0 if is_local else None,
    )

    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(D)
    o = torch.empty(q.shape, device=dev, dtype=out_dtype)
    if blockscaled:
        # fp8 goes as uint8 via TVM-FFI; SFQ stays flat (interleaved SFQ forbidden with varlen q).
        data = lambda t: t.view(torch.uint8)
        sf_args = (q_scale, k_scale, v_scale)
    else:
        data = lambda t: t
        sf_args = (None, None, None)
    fwd(
        data(q),
        data(k),
        data(v),
        o,
        None,
        None,
        scale,
        *sf_args,
        cu_seqlens_q,
        cu_seqlens_k,
        None,
        seqused_k,
        page_table,
        window_left if is_local else None,
        0 if is_local else None,
        None,
        None,
        None,
        bias,
        None,
        None,
        max_seqlen_q,
    )
    return o
