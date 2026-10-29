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

"""Read-only GDN chain verify in the chunked form on 16-bit tensor cores (see the GDN README)."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.attention.gdn import (
    GDN_TREE_VERIFY_CHUNKED_MAX_NODES,
    gdn_chain_verify_supported,
)
from tokenspeed_kernel.platform import pdl_enabled


@triton.jit
def _window_matmul(x, y, BS: tl.constexpr):
    """``x @ y`` of fp32 tiles: a dot from 8 rows, in-order outer products below."""
    if BS >= 8:
        out = tl.dot(x, y, input_precision="ieee")
    else:
        cols = tl.arange(0, BS)
        out = tl.zeros([x.shape[0], y.shape[1]], dtype=tl.float32)
        for j in tl.static_range(BS):
            x_j = tl.sum(tl.where(cols[None, :] == j, x, 0.0), 1)
            y_j = tl.sum(tl.where(cols[:, None] == j, y, 0.0), 0)
            out += x_j[:, None] * y_j[None, :]
    return out


# Gates are projection views: unspecialized, so dense warm gates compile what any layout runs.
@triton.jit(
    do_not_specialize=["T", "stride_a_t", "stride_b_t"],
    do_not_specialize_on_alignment=["a", "b"],
)
def gdn_chain_verify_kernel(
    q,
    k,
    v,
    a,
    b,
    A_log,
    dt_bias,
    o,
    h0,
    h0_indices,
    scale,
    T,
    stride_q_t,
    stride_k_t,
    stride_v_t,
    stride_a_t,
    stride_b_t,
    stride_h0,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    BS: tl.constexpr,
    LOG2_BS: tl.constexpr,
    BK: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    """Chain verify of ``T <= BS`` request-major rows per request from the K-last pool ``h0``."""
    i_v = tl.program_id(0)
    i_n = tl.program_id(1).to(tl.int64)
    i_hv = tl.program_id(2)
    i_h = i_hv // (HV // H)
    nodes = tl.arange(0, BS)
    valid = nodes < T
    # Rows past T alias row 0 of the request (in bounds); zero norms null them.
    rows = i_n * T + tl.where(valid, nodes, 0)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_v = o_v < V

    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
        # Release successor setup; its wait still guards all dependent reads.
        tl.extra.cuda.gdc_launch_dependents()

    # Transposed output tile [BV, BS]: the value tile is the MMA's wide side.
    p_o = o + (rows[None, :] * HV + i_hv) * V + o_v[:, None]
    out_mask = mask_v[:, None] & valid[None, :]
    idx = tl.load(h0_indices + i_n).to(tl.int64)
    if idx < 0:
        tl.store(p_o, tl.zeros([BV, BS], dtype=p_o.dtype.element_ty), mask=out_mask)
        return

    b_A = tl.exp(tl.load(A_log + i_hv).to(tl.float32))
    b_dt_bias = tl.load(dt_bias + i_hv).to(tl.float32)
    b_a = tl.load(a + rows * stride_a_t + i_hv, mask=valid, other=0).to(tl.float32)
    b_b = tl.load(b + rows * stride_b_t + i_hv, mask=valid, other=0).to(tl.float32)
    b_vT = tl.load(
        v + rows[None, :] * stride_v_t + i_hv * V + o_v[:, None],
        mask=out_mask,
        other=0,
    ).to(tl.float32)

    p_q = q + rows[:, None] * stride_q_t + i_h * K
    p_k = k + rows[:, None] * stride_k_t + i_h * K
    r_q = tl.where(valid, scale, 0.0)
    r_k = tl.where(valid, 1.0, 0.0)
    if USE_QK_L2NORM_IN_KERNEL:
        sq_q = tl.zeros([BS], dtype=tl.float32)
        sq_k = tl.zeros([BS], dtype=tl.float32)
        for kc in tl.static_range(0, K, BK):
            o_k = kc + tl.arange(0, BK)
            x_q = tl.load(p_q + o_k[None, :]).to(tl.float32)
            x_k = tl.load(p_k + o_k[None, :]).to(tl.float32)
            sq_q += tl.sum(x_q * x_q, 1)
            sq_k += tl.sum(x_k * x_k, 1)
        r_q = r_q / tl.sqrt(sq_q + 1e-6)
        r_k = r_k / tl.sqrt(sq_k + 1e-6)

    kk = tl.zeros([BS, BS], dtype=tl.float32)
    qk = tl.zeros([BS, BS], dtype=tl.float32)
    hkT = tl.zeros([BV, BS], dtype=tl.float32)
    hqT = tl.zeros([BV, BS], dtype=tl.float32)
    p_h0 = h0 + idx * stride_h0 + i_hv * V * K + o_v[:, None] * K
    for kc in tl.range(0, K, BK):
        o_k = kc + tl.arange(0, BK)
        # Rows load unmasked (rows past T alias row 0, nulled by their norms).
        c_q = (tl.load(p_q + o_k[None, :]).to(tl.float32) * r_q[:, None]).to(
            h0.dtype.element_ty
        )
        c_kf = tl.load(p_k + o_k[None, :]).to(tl.float32) * r_k[:, None]
        c_k = c_kf.to(h0.dtype.element_ty)
        # What rounding the normalized key lost, as a second key operand.
        c_kr = (c_kf - c_k.to(tl.float32)).to(h0.dtype.element_ty)
        # The K-last pool row loads as [BV, BK] with K contiguous, as stored.
        c_h = tl.load(p_h0 + o_k[None, :], mask=mask_v[:, None], other=0)
        c_kT = tl.trans(c_k)
        c_krT = tl.trans(c_kr)
        kk += tl.dot(c_k, c_kT)
        kk += tl.dot(c_k, c_krT)
        kk += tl.dot(c_kr, c_kT)
        qk += tl.dot(c_q, c_kT)
        qk += tl.dot(c_q, c_krT)
        hkT += tl.dot(c_h, c_kT)
        hkT += tl.dot(c_h, c_krT)
        hqT += tl.dot(c_h, tl.trans(c_q))

    x = b_a + b_dt_bias
    softplus_x = tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x)
    # A finite floor keeps an infinite gate's zero decay out of 0 * inf below.
    b_g = tl.where(valid, tl.maximum(-b_A * softplus_x, -1e30), 0.0)
    b_beta = tl.where(valid, 1.0 / (1.0 + tl.exp(-b_b)), 0.0)

    anc = nodes[None, :] <= nodes[:, None]
    path_gates = tl.where(anc, b_g[None, :], 0.0)
    gate = tl.sum(path_gates, 1)
    # Each segment sums its own gates; a difference of prefix sums would cancel an extreme gate.
    segment = _window_matmul(path_gates, tl.trans(tl.where(anc, 0.0, 1.0)), BS)
    decay = tl.where(anc, tl.exp(segment), 0.0)
    m = tl.where(
        anc & (nodes[:, None] != nodes[None, :]), b_beta[:, None] * decay * kk, 0.0
    )
    # (I + m)^-1 by doubling blocks: [[A, 0], [C, B]]^-1 = [[A^-1, 0], [-B^-1 C A^-1, B^-1]].
    block = nodes[:, None] ^ nodes[None, :]
    inv = (block == 0).to(tl.float32) - tl.where(block == 1, m, 0.0)
    for level in tl.static_range(1, LOG2_BS):
        c = tl.where((block >> level) == 1, m, 0.0)
        inv -= _window_matmul(inv, _window_matmul(c, inv, BS), BS)

    decay_h0 = tl.exp(gate)[None, :]
    # A vanished decay drops the h0 terms even where an extreme state overflowed them.
    hkT = tl.where(decay_h0 > 0, decay_h0 * hkT, 0.0)
    hqT = tl.where(decay_h0 > 0, decay_h0 * hqT, 0.0)
    uT = _window_matmul(b_beta[None, :] * (b_vT - hkT), tl.trans(inv), BS)
    b_oT = hqT + _window_matmul(uT, tl.trans(decay * qk), BS)
    tl.store(p_o, b_oT.to(p_o.dtype.element_ty), mask=out_mask)


# Row bucket -> (min_request_heads, block_v, key_splits, warps, stages), measured on Blackwell.
_CHAIN_VERIFY_LAUNCH_TIERS = {
    4: ((0, 32, 1, 4, 2), (48, 64, 2, 2, 3), (384, 128, 2, 2, 2)),
    8: ((0, 32, 1, 2, 2), (144, 128, 2, 2, 3), (384, 128, 4, 1, 3)),
    16: ((0, 32, 2, 2, 3), (48, 64, 2, 2, 3), (192, 128, 4, 2, 3)),
}


def _row_bucket(width: int) -> int:
    """Rows per tile: a power of two of at least 4 (narrower tiles were not measured)."""
    return max(4, triton.next_power_of_2(width))


def gdn_chain_verify_launch(num_requests: int, num_v_heads: int, width: int) -> dict:
    """The launch tier that ``num_requests * num_v_heads`` reach for ``width`` rows."""
    request_heads = num_requests * num_v_heads
    for min_heads, block_v, nk, num_warps, num_stages in reversed(
        _CHAIN_VERIFY_LAUNCH_TIERS[_row_bucket(width)]
    ):
        if request_heads >= min_heads:
            break
    return dict(block_v=block_v, nk=nk, num_warps=num_warps, num_stages=num_stages)


def gdn_chain_verify_warm_batch_sizes(
    batch_sizes: Iterable[int], num_v_heads: int, width: int
) -> list[int]:
    """The first of ``batch_sizes`` at each launch tier of a ``width``-row chain verify."""
    first = {}
    for batch in batch_sizes:
        cfg = gdn_chain_verify_launch(batch, num_v_heads, width)
        first.setdefault(tuple(sorted(cfg.items())), batch)
    return list(first.values())


def launch_gdn_chain_verify(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    initial_state: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: float | None,
    use_qk_l2norm: bool,
) -> torch.Tensor:
    """Launch the chain verify; see ``gdn_chain_verify`` for the contract."""
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    if not gdn_chain_verify_supported(K, V, initial_state.dtype):
        raise ValueError(
            "the chunked chain verify reads a 16-bit state of equal key and value "
            f"head dims, a power of two from 64; got {initial_state.dtype}, {K}, {V}"
        )
    if initial_state.shape[1:] != (HV, V, K) or initial_state.stride()[1:] != (
        V * K,
        K,
        1,
    ):
        raise ValueError(
            f"initial_state must be a K-last [pages, {HV}, {V}, {K}] pool with "
            f"dense pages, got {tuple(initial_state.shape)} strides "
            f"{initial_state.stride()}"
        )
    if (
        q.stride(-1) != 1
        or k.stride(-1) != 1
        or v.stride(-1) != 1
        or q.stride(2) != K
        or k.stride(2) != K
        or v.stride(2) != V
        or a.stride(-1) != 1
        or b.stride(-1) != 1
    ):
        raise ValueError("q, k, v need dense heads and a, b a unit head stride")
    if B > 1 and any(t.stride(0) != T * t.stride(1) for t in (q, k, v, a, b)):
        raise ValueError("q, k, v, a and b need request-major rows (T rows apart)")
    if not 1 <= T <= GDN_TREE_VERIFY_CHUNKED_MAX_NODES:
        raise ValueError(
            "the chunked chain verify holds 1 to "
            f"{GDN_TREE_VERIFY_CHUNKED_MAX_NODES} rows, got {T}"
        )
    if (
        HV % H
        or k.shape != q.shape
        or v.shape[:2] != (B, T)
        or a.shape != (B, T, HV)
        or b.shape != (B, T, HV)
        or A_log.shape != (HV,)
        or dt_bias.shape != (HV,)
        or not A_log.is_contiguous()
        or not dt_bias.is_contiguous()
        or initial_state_indices.shape != (B,)
        or not initial_state_indices.is_contiguous()
        or initial_state_indices.is_floating_point()
    ):
        raise ValueError(
            "need HV % H == 0, k shaped like q, [B, T] rows of v, [B, T, HV] "
            "gates, contiguous [HV] A_log and dt_bias and a contiguous integer "
            f"[B] page index; got q {tuple(q.shape)}, k {tuple(k.shape)}, "
            f"v {tuple(v.shape)}, a {tuple(a.shape)}, b {tuple(b.shape)}, "
            f"A_log {tuple(A_log.shape)}, dt_bias {tuple(dt_bias.shape)}, "
            f"indices {tuple(initial_state_indices.shape)} "
            f"{initial_state_indices.dtype}"
        )
    if scale is None:
        scale = K**-0.5
    cfg = gdn_chain_verify_launch(B, HV, T)
    nk = cfg["nk"]
    BV = min(cfg["block_v"], V)
    BS = _row_bucket(T)
    o = q.new_empty(B, T, HV, V)
    if B == 0:
        return o
    enable_pdl = pdl_enabled()
    gdn_chain_verify_kernel[(triton.cdiv(V, BV), B, HV)](
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        o,
        initial_state,
        initial_state_indices,
        scale,
        T,
        q.stride(1),
        k.stride(1),
        v.stride(1),
        a.stride(1),
        b.stride(1),
        initial_state.stride(0),
        H=H,
        HV=HV,
        K=K,
        V=V,
        BV=BV,
        BS=BS,
        LOG2_BS=BS.bit_length() - 1,
        BK=K // nk,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm,
        ENABLE_PDL=enable_pdl,
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
        **({"launch_pdl": True} if enable_pdl else {}),
    )
    return o
