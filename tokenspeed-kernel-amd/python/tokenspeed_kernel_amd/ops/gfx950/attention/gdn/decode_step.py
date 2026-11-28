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

"""GFX950 Gluon GDN single-token decode step over a K-last state pool."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon

cdna4 = gl.amd.cdna4

# Head dims and pool dtypes the launcher accepts; the registration declares
# the same sets so selection routes other shapes to another solution. Each
# program covers 16 value rows, so V may be any multiple of 16 (the set
# enumerates them up to 1024).
GDN_DECODE_STEP_KEY_DIM = 128
GDN_DECODE_STEP_VALUE_DIMS = frozenset(range(16, 1025, 16))
GDN_DECODE_STEP_STATE_DTYPES = frozenset({torch.float16, torch.bfloat16, torch.float32})

# Each lane holds 4 consecutive K entries of a state row, repeated every 32
# entries, so one 16-byte access per lane covers a full 128-byte line per row.
_ROW_LANES = 8
# (rows per lane, waves): 16 V rows per one-wave program measured fastest.
_ROWS_PER_THREAD = 2
_NUM_WARPS = 1


def _gdn_decode_step_launch_metadata(grid, kernel, args):
    """Report the step's state traffic and FLOPs to Proton."""
    q, v = args["q"], args["v"]
    batch = q.shape[0]
    heads, value_dim, key_dim = args["HV"], args["V"], args["K"]
    state_bytes = heads * value_dim * key_dim * args["state"].element_size()
    activations = (
        2 * args["H"] * key_dim * q.element_size()
        + 2 * heads * value_dim * v.element_size()
    )
    return {
        "name": kernel.name,
        # Decay, prediction, rank-1 update and output: about 6 FLOPs per entry.
        "flops32": 6 * batch * heads * value_dim * key_dim,
        "bytes": batch * (2 * state_bytes + activations),
    }


@gluon.jit(launch_metadata=_gdn_decode_step_launch_metadata)
def gluon_gdn_decode_step_gfx950(
    q,
    k,
    v,
    a,
    b,
    A_log,
    dt_bias,
    o,
    state,
    read_indices,
    write_indices,
    scale,
    q_batch_stride,
    k_batch_stride,
    v_batch_stride,
    a_batch_stride,
    a_head_stride,
    b_batch_stride,
    b_head_stride,
    state_row_stride,
    H: gl.constexpr,
    HV: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    BV: gl.constexpr,
    ROWS_PER_THREAD: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    USE_QK_L2NORM: gl.constexpr,
):
    """Sigmoid-gated delta-rule step of one token for one V block of one head.

    q/k: ``[B, 1, H, K]``; v: ``[B, 1, HV, V]``; a/b: ``[B, 1, HV]``; o:
    contiguous ``[B, 1, HV, V]``. ``state`` is a K-last ``[rows, HV, V, K]``
    pool. Request ``n`` reads row ``read_indices[n]`` (negative: zero state)
    and writes the updated state to row ``write_indices[n]`` (negative:
    skipped).

    Each lane holds a ``[ROWS_PER_THREAD, 16]`` patch of the ``[BV, K]`` tile
    (four 4-wide K runs, 32 apart), so a reduction over K is 16 in-register
    FMAs plus a three-step cross-lane butterfly and every 16-byte state access
    fills whole 128-byte lines.
    """
    state_layout: gl.constexpr = gl.BlockedLayout(
        [ROWS_PER_THREAD, 4], [8, 8], [NUM_WARPS, 1], [1, 0]
    )
    row_layout: gl.constexpr = gl.SliceLayout(1, state_layout)
    key_layout: gl.constexpr = gl.SliceLayout(0, state_layout)

    value_block = gl.program_id(0)
    sequence_head = gl.program_id(1)
    seq = sequence_head // HV
    head_v = sequence_head % HV
    head_k = head_v // (HV // H)

    value_offsets = value_block * BV + gl.arange(0, BV, layout=row_layout)
    key_offsets = gl.arange(0, K, layout=key_layout)
    tile_offsets = value_offsets[:, None] * K + key_offsets[None, :]
    head_offset = head_v * (V * K)

    read_idx = gl.load(read_indices + seq).to(gl.int64)
    s = cdna4.buffer_load(
        state + gl.maximum(read_idx, 0) * state_row_stride + head_offset,
        tile_offsets,
        mask=(read_idx >= 0) & (tile_offsets >= 0),
        other=0.0,
    ).to(gl.float32)

    q_t = cdna4.buffer_load(q, seq * q_batch_stride + head_k * K + key_offsets).to(
        gl.float32
    )
    k_t = cdna4.buffer_load(k, seq * k_batch_stride + head_k * K + key_offsets).to(
        gl.float32
    )
    if USE_QK_L2NORM:
        q_t = q_t * (gl.rsqrt(gl.sum(q_t * q_t, axis=0) + 1e-6) * scale)
        k_t = k_t * gl.rsqrt(gl.sum(k_t * k_t, axis=0) + 1e-6)
    else:
        q_t = q_t * scale
    v_t = cdna4.buffer_load(v, seq * v_batch_stride + head_v * V + value_offsets).to(
        gl.float32
    )
    a_t = gl.load(a + seq * a_batch_stride + head_v * a_head_stride).to(gl.float32)
    b_t = gl.load(b + seq * b_batch_stride + head_v * b_head_stride).to(gl.float32)
    x = a_t + gl.load(dt_bias + head_v).to(gl.float32)
    softplus = gl.where(x <= 20.0, gl.log(1.0 + gl.exp(x)), x)
    decay = gl.exp(-gl.exp(gl.load(A_log + head_v).to(gl.float32)) * softplus)
    beta = 1.0 / (1.0 + gl.exp(-b_t))

    s = s * decay
    prediction = gl.sum(s * k_t[None, :], axis=1)
    delta = beta * (v_t - prediction)
    s = s + delta[:, None] * k_t[None, :]
    out = gl.sum(s * q_t[None, :], axis=1)
    gl.store(o + (seq * HV + head_v) * V + value_offsets, out.to(o.dtype.element_ty))

    write_idx = gl.load(write_indices + seq).to(gl.int64)
    cdna4.buffer_store(
        s.to(state.dtype.element_ty),
        state + gl.maximum(write_idx, 0) * state_row_stride + head_offset,
        tile_offsets,
        mask=(write_idx >= 0) & (tile_offsets >= 0),
    )


def launch_gluon_gdn_decode_step_gfx950(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    state: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    scale: float,
    use_qk_l2norm: bool,
) -> torch.Tensor:
    """Run one gfx950 GDN decode step (one token per request).

    Args:
        q, k: ``[B, 1, H, 128]`` with contiguous heads (any batch stride).
        v: ``[B, 1, HV, V]`` with contiguous heads, V a multiple of 16 up to
            1024.
        a, b: ``[B, 1, HV]`` decay and beta logits (any strides).
        A_log, dt_bias: ``[HV]`` per-head decay parameters.
        state: K-last ``[rows, HV, V, K]`` pool (BF16, FP16 or FP32, any row
            stride) holding the initial states.
        read_indices: int32 ``[B]`` rows of ``state``; a negative row starts
            from a zero state.
        write_indices: int32 ``[B]`` rows of ``state`` receiving the updated
            state; negative rows are skipped. May equal ``read_indices``.
        scale: Query scale applied after the optional L2 norm.
        use_qk_l2norm: Whether to L2-normalize q and k over K.

    Returns:
        The ``[B, 1, HV, V]`` output in q's dtype.
    """
    batch, tokens, heads, key_dim = q.shape
    value_heads, value_dim = v.shape[2], v.shape[3]
    if (
        key_dim != GDN_DECODE_STEP_KEY_DIM
        or value_dim not in GDN_DECODE_STEP_VALUE_DIMS
    ):
        raise ValueError(f"unsupported GDN head dims K={key_dim} V={value_dim}")
    if tokens != 1:
        raise ValueError(f"the gfx950 GDN decode step runs one token, got {tokens}")
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if tensor.stride(3) != 1 or tensor.stride(2) != tensor.shape[3]:
            raise ValueError(f"{name} heads must be contiguous")
    if (
        state.shape[1:] != (value_heads, value_dim, key_dim)
        or not state[0].is_contiguous()
    ):
        raise ValueError("state must be a K-last [rows, HV, V, K] pool")
    if state.dtype not in GDN_DECODE_STEP_STATE_DTYPES:
        raise ValueError(f"unsupported GDN state dtype {state.dtype}")
    out = q.new_empty(batch, tokens, value_heads, value_dim)
    if batch == 0:
        return out

    block_v = _ROWS_PER_THREAD * _ROW_LANES * _NUM_WARPS
    gluon_gdn_decode_step_gfx950[(value_dim // block_v, batch * value_heads)](
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        out,
        state,
        read_indices,
        write_indices,
        scale,
        q.stride(0),
        k.stride(0),
        v.stride(0),
        a.stride(0),
        a.stride(2),
        b.stride(0),
        b.stride(2),
        state.stride(0),
        H=heads,
        HV=value_heads,
        K=key_dim,
        V=value_dim,
        BV=block_v,
        ROWS_PER_THREAD=_ROWS_PER_THREAD,
        NUM_WARPS=_NUM_WARPS,
        USE_QK_L2NORM=use_qk_l2norm,
        num_warps=_NUM_WARPS,
    )
    return out
