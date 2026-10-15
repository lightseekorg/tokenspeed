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

"""GFX950 Gluon GDN decode / MTP-verify recurrence over a K-last state pool."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, triton

cdna4 = gl.amd.cdna4

# Where each token's post-update state goes (STEP_MODE).
GDN_DECODE_STEPS_NONE = 0
# Row ``step_indices[n, t]`` of ``step_states``.
GDN_DECODE_STEPS_INDEXED = 1
# Row ``n * T + t`` of a dense ``step_states`` buffer.
GDN_DECODE_STEPS_DENSE = 2

# Each lane holds 4 consecutive K entries of a state row, repeated every 32
# entries, so one 16-byte access per lane covers a full 128-byte line per row.
_KEYS_PER_ACCESS = 4
_KEY_LANES = 8
_ROW_LANES = 8
# (rows per lane, waves): 16 V rows per one-wave program measured fastest.
_ROWS_PER_THREAD = 2
_NUM_WARPS = 1


def _gdn_decode_launch_metadata(grid, kernel, args):
    """Report the recurrence's state traffic and FLOPs to Proton."""
    q, v = args["q"], args["v"]
    batch, tokens = q.shape[0], args["T"]
    heads, value_dim, key_dim = args["HV"], args["V"], args["K"]
    state_bytes = heads * value_dim * key_dim * args["state_in"].element_size()
    writes = (tokens if args["STEP_MODE"] else 0) + int(args["WRITE_FINAL"])
    activations = (
        2 * batch * tokens * args["H"] * key_dim * q.element_size()
        + 2 * batch * tokens * heads * value_dim * v.element_size()
    )
    return {
        "name": kernel.name,
        # Decay, prediction, rank-1 update and output: about 6 FLOPs per entry.
        "flops32": 6 * batch * tokens * heads * value_dim * key_dim,
        "bytes": batch * (1 + writes) * state_bytes + activations,
    }


@gluon.jit
def _gdn_decode_store_state(state, pool, row, row_stride, head_offset, tile_offsets):
    """Store a ``[BV, K]`` state tile to one pool row; a negative row skips it."""
    cdna4.buffer_store(
        state.to(pool.dtype.element_ty),
        pool + gl.maximum(row, 0) * row_stride + head_offset,
        tile_offsets,
        mask=(row >= 0) & (tile_offsets >= 0),
    )


@gluon.jit(launch_metadata=_gdn_decode_launch_metadata, do_not_specialize=("T",))
def gluon_gdn_decode_gfx950(
    q,
    k,
    v,
    a,
    b,
    A_log,
    dt_bias,
    o,
    state_in,
    read_indices,
    final_indices,
    step_states,
    step_indices,
    scale,
    T,
    q_batch_stride,
    q_token_stride,
    k_batch_stride,
    k_token_stride,
    v_batch_stride,
    v_token_stride,
    a_batch_stride,
    a_token_stride,
    a_head_stride,
    b_batch_stride,
    b_token_stride,
    b_head_stride,
    state_in_row_stride,
    step_states_row_stride,
    H: gl.constexpr,
    HV: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    BV: gl.constexpr,
    ROWS_PER_THREAD: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    STEP_MODE: gl.constexpr,
    WRITE_FINAL: gl.constexpr,
    USE_QK_L2NORM: gl.constexpr,
    T_BLOCK: gl.constexpr,
):
    """Sigmoid-gated delta-rule recurrence of T tokens for one V block of one head.

    q/k: ``[B, T, H, K]``; v: ``[B, T, HV, V]``; a/b: ``[B, T, HV]``; o:
    contiguous ``[B, T, HV, V]``. ``state_in`` and ``step_states`` are K-last
    ``[rows, HV, V, K]`` pools (they may alias). Request ``n`` reads row
    ``read_indices[n]`` (negative: zero state), writes each token's state per
    STEP_MODE and, with WRITE_FINAL, the last state to ``state_in`` row
    ``final_indices[n]``; negative destination rows are skipped.

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
    state = cdna4.buffer_load(
        state_in + gl.maximum(read_idx, 0) * state_in_row_stride + head_offset,
        tile_offsets,
        mask=(read_idx >= 0) & (tile_offsets >= 0),
        other=0.0,
    ).to(gl.float32)

    # Stage up to T_BLOCK tokens' normalized q/k, beta-scaled v, beta and decay
    # in LDS at a time: one load round trip per chunk instead of one per step.
    token_layout: gl.constexpr = gl.BlockedLayout(
        [1, 4], [8, 8], [NUM_WARPS, 1], [1, 0]
    )
    token_value_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1], [4, 16], [NUM_WARPS, 1], [1, 0]
    )
    # Unswizzled row-major buffers, stored as [T_BLOCK, N] tiles and read back
    # one token row at a time through a rank-1 view of the same bytes.
    smem_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])
    row_smem_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[0])
    smem_q = gl.allocate_shared_memory(gl.float32, [T_BLOCK, K], smem_layout)
    smem_k = gl.allocate_shared_memory(gl.float32, [T_BLOCK, K], smem_layout)
    smem_v = gl.allocate_shared_memory(gl.float32, [T_BLOCK, BV], smem_layout)
    smem_beta = gl.allocate_shared_memory(gl.float32, [T_BLOCK, BV], smem_layout)
    smem_decay = gl.allocate_shared_memory(gl.float32, [T_BLOCK, BV], smem_layout)
    rows_q = smem_q.reinterpret(layout=row_smem_layout)
    rows_k = smem_k.reinterpret(layout=row_smem_layout)
    rows_v = smem_v.reinterpret(layout=row_smem_layout)
    rows_beta = smem_beta.reinterpret(layout=row_smem_layout)
    rows_decay = smem_decay.reinterpret(layout=row_smem_layout)
    keys = gl.arange(0, K, layout=gl.SliceLayout(0, token_layout))
    v_rows = value_block * BV + gl.arange(
        0, BV, layout=gl.SliceLayout(0, token_value_layout)
    )
    neg_a = -gl.exp(gl.load(A_log + head_v).to(gl.float32))
    bias = gl.load(dt_bias + head_v).to(gl.float32)
    zeros_v = gl.zeros([T_BLOCK, BV], gl.float32, layout=token_value_layout)

    o_row = o + (seq * T * HV + head_v) * V
    for t0 in range(0, T, T_BLOCK):
        tokens = t0 + gl.arange(0, T_BLOCK, layout=gl.SliceLayout(1, token_layout))
        token_mask = (tokens < T)[:, None] & (keys >= 0)[None, :]
        q_all = cdna4.buffer_load(
            q,
            seq * q_batch_stride + head_k * K + tokens[:, None] * q_token_stride
            + keys[None, :],
            mask=token_mask,
            other=0.0,
        ).to(gl.float32)  # fmt: skip
        k_all = cdna4.buffer_load(
            k,
            seq * k_batch_stride + head_k * K + tokens[:, None] * k_token_stride
            + keys[None, :],
            mask=token_mask,
            other=0.0,
        ).to(gl.float32)  # fmt: skip
        if USE_QK_L2NORM:
            q_all = (
                q_all
                * (gl.rsqrt(gl.sum(q_all * q_all, axis=1) + 1e-6) * scale)[:, None]
            )
            k_all = k_all * gl.rsqrt(gl.sum(k_all * k_all, axis=1) + 1e-6)[:, None]
        else:
            q_all = q_all * scale

        v_tokens = t0 + gl.arange(
            0, T_BLOCK, layout=gl.SliceLayout(1, token_value_layout)
        )
        v_live = v_tokens < T
        v_all = cdna4.buffer_load(
            v,
            seq * v_batch_stride + head_v * V + v_tokens[:, None] * v_token_stride
            + v_rows[None, :],
            mask=v_live[:, None] & (v_rows >= 0)[None, :],
            other=0.0,
        ).to(gl.float32)  # fmt: skip
        a_all = gl.load(
            a
            + seq * a_batch_stride
            + head_v * a_head_stride
            + v_tokens * a_token_stride,
            mask=v_live,
            other=0.0,
        ).to(gl.float32)
        b_all = gl.load(
            b
            + seq * b_batch_stride
            + head_v * b_head_stride
            + v_tokens * b_token_stride,
            mask=v_live,
            other=0.0,
        ).to(gl.float32)
        x = a_all + bias
        softplus = gl.where(x <= 20.0, gl.log(1.0 + gl.exp(x)), x)
        beta_all = 1.0 / (1.0 + gl.exp(-b_all))
        smem_q.store(q_all)
        smem_k.store(k_all)
        smem_v.store(v_all * beta_all[:, None])
        # Per-token scalars are stored broadcast over the V rows so each step
        # reads them in the state's row layout.
        smem_beta.store(zeros_v + beta_all[:, None])
        smem_decay.store(zeros_v + gl.exp(neg_a * softplus)[:, None])

        for t in range(t0, gl.minimum(t0 + T_BLOCK, T)):
            slot = t - t0
            q_t = rows_q.index(slot).load(key_layout)
            k_t = rows_k.index(slot).load(key_layout)
            beta_v = rows_v.index(slot).load(row_layout)
            beta = rows_beta.index(slot).load(row_layout)
            decay = rows_decay.index(slot).load(row_layout)

            state = state * decay[:, None]
            prediction = gl.sum(state * k_t[None, :], axis=1)
            delta = beta_v - beta * prediction
            state = state + delta[:, None] * k_t[None, :]
            out = gl.sum(state * q_t[None, :], axis=1)
            gl.store(o_row + value_offsets, out.to(o.dtype.element_ty))
            o_row += HV * V

            if STEP_MODE == 1:
                step_row = gl.load(step_indices + seq * T + t).to(gl.int64)
                _gdn_decode_store_state(
                    state, step_states, step_row, step_states_row_stride,
                    head_offset, tile_offsets,
                )  # fmt: skip
            elif STEP_MODE == 2:
                _gdn_decode_store_state(
                    state, step_states, (seq * T + t).to(gl.int64),
                    step_states_row_stride, head_offset, tile_offsets,
                )  # fmt: skip

    if WRITE_FINAL:
        final_row = gl.load(final_indices + seq).to(gl.int64)
        _gdn_decode_store_state(
            state, state_in, final_row, state_in_row_stride, head_offset, tile_offsets
        )


def launch_gluon_gdn_decode_gfx950(
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
    final_indices: torch.Tensor | None,
    step_states: torch.Tensor | None,
    step_indices: torch.Tensor | None,
    scale: float,
    use_qk_l2norm: bool,
) -> torch.Tensor:
    """Run the gfx950 GDN recurrence of ``T = q.shape[1]`` tokens per request.

    Args:
        q, k: ``[B, T, H, 128]`` with contiguous heads (any batch/token strides).
        v: ``[B, T, HV, V]`` with contiguous heads, V a multiple of 16.
        a, b: ``[B, T, HV]`` decay and beta logits (any strides).
        A_log, dt_bias: ``[HV]`` per-head decay parameters.
        state: K-last ``[rows, HV, V, K]`` pool holding the initial states.
        read_indices: int32 ``[B]`` rows of ``state``; a negative row starts
            from a zero state.
        final_indices: int32 ``[B]`` rows of ``state`` receiving the last
            state, or ``None`` for no final write.
        step_states: K-last pool receiving every token's state, or ``None``.
        step_indices: int32 ``[B, T]`` rows of ``step_states``; ``None`` with
            ``step_states`` writes the dense row ``n * T + t``.
        scale: Query scale applied after the optional L2 norm.
        use_qk_l2norm: Whether to L2-normalize q and k over K.

    Negative destination rows are skipped. Returns the ``[B, T, HV, V]``
    output in q's dtype.
    """
    batch, tokens, heads, key_dim = q.shape
    value_heads, value_dim = v.shape[2], v.shape[3]
    block_v = _ROWS_PER_THREAD * _ROW_LANES * _NUM_WARPS
    if key_dim != _KEYS_PER_ACCESS * _KEY_LANES * 4 or value_dim % block_v:
        raise ValueError(f"unsupported GDN head dims K={key_dim} V={value_dim}")
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if tensor.stride(3) != 1 or tensor.stride(2) != tensor.shape[3]:
            raise ValueError(f"{name} heads must be contiguous")
    pools = [("state", state)]
    if step_states is not None:
        pools.append(("step_states", step_states))
    elif step_indices is not None:
        raise ValueError("step_indices requires step_states")
    for name, pool in pools:
        if (
            pool.shape[1:] != (value_heads, value_dim, key_dim)
            or not pool[0].is_contiguous()
        ):
            raise ValueError(f"{name} must be a K-last [rows, HV, V, K] pool")
    out = q.new_empty(batch, tokens, value_heads, value_dim)
    if batch == 0:
        return out

    step_mode = GDN_DECODE_STEPS_NONE
    if step_states is not None:
        step_mode = (
            GDN_DECODE_STEPS_DENSE if step_indices is None else GDN_DECODE_STEPS_INDEXED
        )
    gluon_gdn_decode_gfx950[(value_dim // block_v, batch * value_heads)](
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
        read_indices if final_indices is None else final_indices,
        state if step_states is None else step_states,
        read_indices if step_indices is None else step_indices,
        scale,
        tokens,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        a.stride(0),
        a.stride(1),
        a.stride(2),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        state.stride(0),
        state.stride(0) if step_states is None else step_states.stride(0),
        H=heads,
        HV=value_heads,
        K=key_dim,
        V=value_dim,
        BV=block_v,
        ROWS_PER_THREAD=_ROWS_PER_THREAD,
        NUM_WARPS=_NUM_WARPS,
        STEP_MODE=step_mode,
        WRITE_FINAL=final_indices is not None,
        USE_QK_L2NORM=use_qk_l2norm,
        # LDS staging chunk, bucketed so a server's fixed T compiles once.
        T_BLOCK=min(triton.next_power_of_2(tokens), 16),
        num_warps=_NUM_WARPS,
    )
    return out
