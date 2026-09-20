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

"""TS input producers for the FlashInfer KDA state/layout contract.

Single-token preparation publishes convolution history to destination slots.
Multi-token preparation gathers frozen state and can capture replay inputs.
The compact SM103 path preserves the generic producer's reduction order.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import gl, gluon, tl, triton
from tokenspeed_kernel.platform import ArchVersion, current_platform, pdl_enabled


@gluon.jit
def _compact_decode_producer(
    raw_qkv,
    gate_input,
    gate_weight,
    beta_logits,
    conv_weight,
    conv_pool,
    read_indices,
    write_indices,
    qkv_out,
    gate_out,
    beta_out,
    state_pool,
    ROWS: gl.constexpr,
    HEADS: gl.constexpr,
    RAW_STRIDE: gl.constexpr,
    GATE_INPUT_STRIDE: gl.constexpr,
    BETA_STRIDE: gl.constexpr,
    CONV_STRIDE: gl.constexpr,
    BLOCK_CHANNELS: gl.constexpr,
    PDL: gl.constexpr,
    STATE_STRIDE: gl.constexpr,
    PREFETCH_STATE: gl.constexpr,
):
    head, row, channel_tile = (gl.program_id(0), gl.program_id(1), gl.program_id(2))
    GATE_LAYOUT: gl.constexpr = gl.BlockedLayout([1, 8], [2, 16], [1, 1], [1, 0])
    c = channel_tile * BLOCK_CHANNELS + gl.arange(
        0, BLOCK_CHANNELS, layout=gl.SliceLayout(1, GATE_LAYOUT)
    )
    f = gl.arange(0, 128, layout=gl.SliceLayout(0, GATE_LAYOUT))
    # Immutable low-rank weights may be prefetched before the dependency wait.
    w = gl.load(gate_weight + (head * 128 + c[:, None]) * 128 + f[None, :])
    CONV_LAYOUT: gl.constexpr = gl.BlockedLayout([1, 1], [2, 16], [1, 1], [1, 0])
    WEIGHT_LAYOUT: gl.constexpr = gl.BlockedLayout(
        [1, 1, 4], [2, 16, 1], [1, 1, 1], [1, 0, 2]
    )
    planes = gl.arange(0, 4, layout=gl.SliceLayout(1, CONV_LAYOUT))
    channels = channel_tile * BLOCK_CHANNELS + gl.arange(
        0, BLOCK_CHANNELS, layout=gl.SliceLayout(0, CONV_LAYOUT)
    )
    offset = planes[:, None] * HEADS * 128 + head * 128 + channels[None, :]
    p4 = gl.arange(0, 4, layout=gl.SliceLayout(1, gl.SliceLayout(2, WEIGHT_LAYOUT)))
    c4 = channel_tile * BLOCK_CHANNELS + gl.arange(
        0, BLOCK_CHANNELS, layout=gl.SliceLayout(0, gl.SliceLayout(2, WEIGHT_LAYOUT))
    )
    taps = gl.arange(0, 4, layout=gl.SliceLayout(0, gl.SliceLayout(1, WEIGHT_LAYOUT)))
    off4 = p4[:, None] * HEADS * 128 + head * 128 + c4[None, :]
    wc = gl.load(
        conv_weight + off4[:, :, None] * 4 + taps[None, None, :],
        mask=p4[:, None, None] < 3,
        other=0,
    )
    if PDL:
        gl.inline_asm_elementwise(
            "griddepcontrol.wait; mov.u32 $0,0;",
            constraints="=r",
            args=[],
            dtype=gl.uint32,
            is_pure=False,
            pack=1,
        )
    read_index = gl.load(read_indices + row).to(gl.int64)
    write_index = gl.load(write_indices + row).to(gl.int64)
    active = write_index >= 0
    if PREFETCH_STATE and read_index >= 0 and active:
        # Exactly one hardware thread per CTA issues the aligned read hint.
        # Each channel tile covers a disjoint contiguous V-row region; all
        # tiles together cover the head's unchanged [128,128] BF16 state.
        pointer = (
            state_pool
            + read_index * STATE_STRIDE
            + head * 16384
            + channel_tile * BLOCK_CHANNELS * 128
        )
        gl.inline_asm_elementwise(
            "{ .reg .b32 tid; .reg .pred p; mov.u32 tid, %tid.x; "
            "setp.eq.u32 p, tid, 0; "
            "@p cp.async.bulk.prefetch.L2.global [$1], $2; mov.u32 $0,0; }",
            constraints="=r,l,r",
            args=[pointer, BLOCK_CHANNELS * 128 * 2],
            dtype=gl.uint32,
            is_pure=False,
            pack=1,
        )
    # Prepare all four tap weights and issue Q/K/V history/input loads before
    # the gate reduction, overlapping independent memory work. The fourth
    # plane is masked; both gate reduction and conv accumulation are unchanged.
    w02, w13 = gl.split(gl.reshape(wc.to(gl.float32), (4, BLOCK_CHANNELS, 2, 2)))
    w0, w2 = gl.split(w02)
    w1, w3 = gl.split(w13)
    w0 = gl.convert_layout(w0, CONV_LAYOUT)
    w1 = gl.convert_layout(w1, CONV_LAYOUT)
    w2 = gl.convert_layout(w2, CONV_LAYOUT)
    w3 = gl.convert_layout(w3, CONV_LAYOUT)
    valid = active & (planes[:, None] < 3)
    s0 = gl.load(
        conv_pool + read_index * CONV_STRIDE + offset * 3,
        mask=valid & (read_index >= 0),
        other=0,
    ).to(gl.float32)
    s1 = gl.load(
        conv_pool + read_index * CONV_STRIDE + offset * 3 + 1,
        mask=valid & (read_index >= 0),
        other=0,
    ).to(gl.float32)
    s2 = gl.load(
        conv_pool + read_index * CONV_STRIDE + offset * 3 + 2,
        mask=valid & (read_index >= 0),
        other=0,
    ).to(gl.float32)
    x = gl.load(raw_qkv + row * RAW_STRIDE + offset, mask=valid, other=0).to(gl.float32)
    xf = gl.load(gate_input + row * GATE_INPUT_STRIDE + f).to(gl.float32)
    g = gl.sum(w.to(gl.float32) * xf[None, :], 1)
    gl.store(gate_out + row * HEADS * 128 + head * 128 + c, gl.where(active, g, 0.0))
    a = s0 * w0
    a += s1 * w1
    a += s2 * w2
    a += x * w3
    y = a * (1.0 / (1.0 + gl.exp(-a)))
    gl.store(
        qkv_out
        + planes[:, None] * ROWS * HEADS * 128
        + row * HEADS * 128
        + head * 128
        + channels[None, :],
        y,
        mask=planes[:, None] < 3,
    )
    gl.store(conv_pool + write_index * CONV_STRIDE + offset * 3, s1, mask=valid)
    gl.store(conv_pool + write_index * CONV_STRIDE + offset * 3 + 1, s2, mask=valid)
    gl.store(conv_pool + write_index * CONV_STRIDE + offset * 3 + 2, x, mask=valid)
    if channel_tile == 0:
        b = gl.load(beta_logits + row * BETA_STRIDE + head, mask=active, other=0)
        gl.store(beta_out + row * HEADS + head, b)


@triton.jit
def _kda_recurrent_producer(
    raw,
    fa,
    fb,
    beta,
    conv_w,
    conv_pool,
    state_pool,
    reads,
    writes,
    qkv_out,
    gate_out,
    beta_out,
    state_out,
    slots,
    replay_raw,
    replay_fa,
    replay_beta,
    REPLAY_RAW_STRIDE: tl.constexpr,
    REPLAY_FA_STRIDE: tl.constexpr,
    REPLAY_BETA_STRIDE: tl.constexpr,
    CAPTURE_REPLAY: tl.constexpr,
    ROWS: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    DFA: tl.constexpr,
    TOKENS: tl.constexpr,
    RAW_STRIDE: tl.constexpr,
    FA_STRIDE: tl.constexpr,
    BETA_STRIDE: tl.constexpr,
    CONV_STRIDE: tl.constexpr,
    STATE_STRIDE: tl.constexpr,
    STATE_OUT_STRIDE: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    STORE_CONV: tl.constexpr,
    GATHER_STATE: tl.constexpr,
    DOT: tl.constexpr,
    PDL: tl.constexpr,
):
    h, tb, kb = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    row = tb * BT + tl.arange(0, BT)
    channel = kb * BK + tl.arange(0, BK)
    c = h * K + channel
    n, t = row // TOKENS, row % TOKENS
    cmask = channel < K
    f = tl.arange(0, DFA)
    if STORE_CONV:
        # Immutable f_b weights are independent of the incoming projection.
        # Dynamic metadata and projected activations are consumed only after
        # the PDL dependency wait. The downstream FI call remains unchanged.
        w = tl.load(fb + c[:, None] * DFA + f[None, :], mask=cmask[:, None], other=0)
    if PDL:
        tl.extra.cuda.gdc_wait()
    valid = row < ROWS
    ri = tl.load(reads + n, mask=valid, other=-1).to(tl.int64)
    if STORE_CONV:
        wi = tl.load(writes + n, mask=valid, other=-1).to(tl.int64)
        valid = valid & (wi >= 0)
    else:
        wi = n.to(tl.int64)
    mask = valid[:, None] & cmask[None, :]
    for plane in tl.static_range(3):
        feature = plane * H * K + c
        if STORE_CONV:
            # T=1: load the three old samples and the new sample once. Reuse
            # the loaded values for publication instead of re-reading them.
            s0 = tl.load(
                conv_pool + ri[:, None] * CONV_STRIDE + feature[None, :] * 3,
                mask=mask & (ri[:, None] >= 0),
                other=0,
            ).to(tl.float32)
            s1 = tl.load(
                conv_pool + ri[:, None] * CONV_STRIDE + feature[None, :] * 3 + 1,
                mask=mask & (ri[:, None] >= 0),
                other=0,
            ).to(tl.float32)
            s2 = tl.load(
                conv_pool + ri[:, None] * CONV_STRIDE + feature[None, :] * 3 + 2,
                mask=mask & (ri[:, None] >= 0),
                other=0,
            ).to(tl.float32)
            current = tl.load(
                raw + row[:, None] * RAW_STRIDE + feature[None, :], mask=mask, other=0
            ).to(tl.float32)
            w0 = tl.load(conv_w + feature * 4, mask=cmask, other=0).to(tl.float32)
            w1 = tl.load(conv_w + feature * 4 + 1, mask=cmask, other=0).to(tl.float32)
            w2 = tl.load(conv_w + feature * 4 + 2, mask=cmask, other=0).to(tl.float32)
            w3 = tl.load(conv_w + feature * 4 + 3, mask=cmask, other=0).to(tl.float32)
            acc = s0 * w0[None, :]
            acc += s1 * w1[None, :]
            acc += s2 * w2[None, :]
            acc += current * w3[None, :]
        else:
            acc = tl.full((BT, BK), 0, tl.float32)
            for tap in tl.static_range(4):
                source_t = t + tap - 3
                from_history = source_t < 0
                sample_base = tl.where(
                    from_history,
                    conv_pool + ri * CONV_STRIDE + t + tap,
                    raw + (n * TOKENS + source_t) * RAW_STRIDE,
                )
                feature_stride = tl.where(from_history, 3, 1)
                new_value = tl.load(
                    sample_base[:, None] + feature[None, :] * feature_stride[:, None],
                    mask=mask & (~from_history[:, None] | (ri[:, None] >= 0)),
                    other=0,
                )
                # Retain the old masked-load sum's positive-zero boundary.
                conv_sample = new_value.to(tl.float32) + 0.0
                if CAPTURE_REPLAY and tap == 3:
                    tl.store(
                        replay_raw
                        + row[:, None] * REPLAY_RAW_STRIDE
                        + feature[None, :],
                        new_value,
                        mask=mask,
                    )
                weight = tl.load(conv_w + feature * 4 + tap, mask=cmask, other=0).to(
                    tl.float32
                )
                acc += conv_sample * weight[None, :]
        result = acc * tl.sigmoid(acc)
        tl.store(
            qkv_out + plane * ROWS * H * K + row[:, None] * H * K + c[None, :],
            result,
            mask=(row[:, None] < ROWS) & cmask[None, :],
        )
        if STORE_CONV:
            for tap in tl.static_range(3):
                value = s1 if tap == 0 else (s2 if tap == 1 else current)
                tl.store(
                    conv_pool + wi[:, None] * CONV_STRIDE + feature[None, :] * 3 + tap,
                    value,
                    mask=mask,
                )
    if not STORE_CONV:
        w = tl.load(fb + c[:, None] * DFA + f[None, :], mask=cmask[:, None], other=0)
    if DOT:
        x = tl.load(
            fa + row[:, None] * FA_STRIDE + f[None, :], mask=valid[:, None], other=0
        )
        g = tl.dot(x, tl.trans(w))
        if CAPTURE_REPLAY and h == 0 and kb == 0:
            tl.store(
                replay_fa + row[:, None] * REPLAY_FA_STRIDE + f[None, :],
                x,
                mask=valid[:, None],
            )
        tl.store(
            gate_out + row[:, None] * H * K + c[None, :],
            g,
            mask=(row[:, None] < ROWS) & cmask[None, :],
        )
    else:
        for j in tl.static_range(BT):
            r = tb * BT + j
            x_raw = tl.load(fa + r * FA_STRIDE + f, mask=r < ROWS, other=0)
            x = x_raw.to(tl.float32)
            if CAPTURE_REPLAY and h == 0 and kb == 0:
                tl.store(replay_fa + r * REPLAY_FA_STRIDE + f, x_raw, mask=r < ROWS)
            g = tl.sum(w.to(tl.float32) * x[None, :], axis=1)
            if STORE_CONV:
                active = tl.load(writes + r, mask=r < ROWS, other=-1) >= 0
                g = tl.where(active, g, 0.0)
            tl.store(gate_out + r * H * K + c, g, mask=(r < ROWS) & cmask)
    if kb == 0:
        b = tl.load(beta + row * BETA_STRIDE + h, mask=valid, other=0)
        tl.store(beta_out + row * H + h, b, mask=row < ROWS)
        if CAPTURE_REPLAY:
            tl.store(replay_beta + row * REPLAY_BETA_STRIDE + h, b, mask=row < ROWS)
    if GATHER_STATE:
        if TOKENS % BT == 0:
            # A tile is wholly inside one sequence. Only the first tile
            # copies that sequence's state; assign contiguous word ranges.
            if tb % (TOKENS // BT) == 0:
                seq = tb // (TOKENS // BT)
                src_index = tl.load(
                    reads + seq, mask=seq < ROWS // TOKENS, other=-1
                ).to(tl.int64)
                words = (
                    h * (K * K // 4) + kb * (K * BK // 4) + tl.arange(0, K * BK // 4)
                )
                source = state_pool.to(tl.pointer_type(tl.int64))
                target = state_out.to(tl.pointer_type(tl.int64))
                value = tl.load(
                    source + src_index * (STATE_STRIDE // 4) + words,
                    mask=(seq < ROWS // TOKENS) & (src_index >= 0),
                    other=0,
                )
                tl.store(
                    target + seq * (STATE_OUT_STRIDE // 4) + words,
                    value,
                    mask=seq < ROWS // TOKENS,
                )
                if h == 0 and kb == 0:
                    tl.store(slots + seq, seq, mask=seq < ROWS // TOKENS)
        elif BT % TOKENS == 0 and BT // TOKENS <= 2:
            # Copy one row per request, not one row per speculative token.
            seq = tb * (BT // TOKENS) + tl.arange(0, BT // TOKENS)
            src_index = tl.load(reads + seq, mask=seq < ROWS // TOKENS, other=-1).to(
                tl.int64
            )
            words = h * (K * K // 4) + kb * (K * BK // 4) + tl.arange(0, K * BK // 4)
            source = state_pool.to(tl.pointer_type(tl.int64))
            target = state_out.to(tl.pointer_type(tl.int64))
            value = tl.load(
                source + src_index[:, None] * (STATE_STRIDE // 4) + words[None, :],
                mask=(seq[:, None] < ROWS // TOKENS) & (src_index[:, None] >= 0),
                other=0,
            )
            tl.store(
                target + seq[:, None] * (STATE_OUT_STRIDE // 4) + words[None, :],
                value,
                mask=seq[:, None] < ROWS // TOKENS,
            )
            if h == 0 and kb == 0:
                tl.store(slots + seq, seq, mask=seq < ROWS // TOKENS)
        else:
            # Assign a request to the unique tile containing its first token.
            # Copy contiguous state words instead of expanding an inactive
            # [token, value, key] gather for all rows in the tile.
            first = tl.cdiv(tb * BT, TOKENS)
            limit = tl.minimum(tl.cdiv((tb + 1) * BT, TOKENS), ROWS // TOKENS)
            words = h * (K * K // 4) + kb * (K * BK // 4) + tl.arange(0, K * BK // 4)
            source = state_pool.to(tl.pointer_type(tl.int64))
            target = state_out.to(tl.pointer_type(tl.int64))
            for seq in range(first, limit):
                src_index = tl.load(reads + seq).to(tl.int64)
                value = tl.load(
                    source + src_index * (STATE_STRIDE // 4) + words,
                    mask=src_index >= 0,
                    other=0,
                )
                tl.store(target + seq * (STATE_OUT_STRIDE // 4) + words, value)
                if h == 0 and kb == 0:
                    tl.store(slots + seq, seq)


def prepare_kda_recurrent_inputs(
    raw: torch.Tensor,
    fa: torch.Tensor,
    fb: torch.Tensor,
    beta: torch.Tensor,
    conv_weights: torch.Tensor,
    conv_pool: torch.Tensor,
    state_pool: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    *,
    tokens: int,
    state_scratch: torch.Tensor | None,
    replay_payload: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Compose convolution, gate GEMM, input packing and frozen-state gathering.

    Ordinary decode publishes convolution to the TS destination index while
    recurrence consumes the same independent read/write indices. Frozen verify gathers
    active state rows and leaves both persistent pools untouched. Optional
    replay destinations are preallocated raw-QKV, low-rank input and beta
    buffers; their original BF16 words are published by the same producer.
    """
    assert state_pool.dtype == torch.bfloat16
    assert conv_pool.stride(1) == 3 and conv_pool.stride(2) == 1
    rows = raw.shape[0]
    # These views refer to existing persistent replay buffers. No allocation,
    # cache ownership or accepted-prefix commit semantics are changed.
    if replay_payload is not None:
        if tokens <= 1:
            raise ValueError("Replay capture is only valid for frozen verification")
        if len(replay_payload) != 3:
            raise ValueError("Replay capture requires three destination tensors")
        for src, dst in zip((raw, fa, beta), replay_payload):
            if (
                dst.ndim != 2
                or dst.shape[0] < rows
                or dst.shape[1] < src.shape[1]
                or dst.stride(1) != 1
                or dst.dtype != src.dtype
                or dst.device != src.device
            ):
                raise ValueError("Invalid replay capture destination")
    heads, dim = state_pool.shape[1:3]
    assert dim == 128 and fb.shape == (heads * dim, fa.shape[1])
    qkv = torch.empty((3, rows, heads, dim), device=raw.device, dtype=raw.dtype)
    gate = torch.empty((rows, heads, dim), device=raw.device, dtype=raw.dtype)
    beta_dense = torch.empty((rows, heads), device=raw.device, dtype=raw.dtype)
    if tokens > 1:
        if state_scratch is None:
            state = torch.empty(
                (rows // tokens, heads, dim, dim),
                device=raw.device,
                dtype=state_pool.dtype,
            )
        else:
            assert state_scratch.shape[0] >= rows // tokens
            assert state_scratch.shape[1:] == state_pool.shape[1:]
            assert (
                state_scratch.dtype == state_pool.dtype
                and state_scratch.is_contiguous()
            )
            state = state_scratch[: rows // tokens]
        slots = torch.empty(rows // tokens, device=raw.device, dtype=torch.int32)
    else:
        state, slots = state_pool, None
    enable_pdl = tokens == 1 and pdl_enabled()
    # Use one producer for compatible single-token calls regardless of batch
    # size. Multi-token verification keeps its existing preparation contract.
    if (
        tokens == 1
        and rows > 0
        and heads == 12
        and fa.shape[1] == 128
        and raw.dtype == torch.bfloat16
        and fa.dtype == fb.dtype == beta.dtype == conv_weights.dtype == torch.bfloat16
        and current_platform().arch_version in (ArchVersion(10, 0), ArchVersion(10, 3))
    ):
        block_channels = 16
        # PTX bulk prefetch requires a 16-byte aligned source/range. A valid
        # but unaligned staging view retains exactly the same computation
        # with its optional cache hint disabled; no value-dependent sync.
        prefetch_state = (
            state_pool.data_ptr() % 16 == 0
            and state_pool.stride(0) * state_pool.element_size() % 16 == 0
        )
        _compact_decode_producer[(heads, rows, dim // block_channels)](
            raw,
            fa,
            fb,
            beta,
            conv_weights,
            conv_pool,
            read_indices,
            write_indices,
            qkv,
            gate,
            beta_dense,
            state_pool,
            ROWS=rows,
            HEADS=heads,
            RAW_STRIDE=raw.stride(0),
            GATE_INPUT_STRIDE=fa.stride(0),
            BETA_STRIDE=beta.stride(0),
            CONV_STRIDE=conv_pool.stride(0),
            BLOCK_CHANNELS=block_channels,
            PDL=enable_pdl,
            STATE_STRIDE=state_pool.stride(0),
            PREFETCH_STATE=prefetch_state,
            num_warps=1,
            **({"launch_pdl": True} if enable_pdl else {}),
        )
        return qkv, gate, beta_dense, state, slots
    if tokens == 1:
        if rows <= 4:
            bt, bk, warps = 1, 8, 1
        elif rows <= 16:
            bt, bk, warps = 1, 16, 2
        elif rows <= 32:
            bt, bk, warps = 1, 32, 2
        else:
            bt, bk, warps = 4, 16, 2
    else:
        bt, bk, warps = (
            (1, 16, 2) if rows <= 32 else (4, 16, 2) if rows <= 64 else (16, 32, 4)
        )
    _kda_recurrent_producer[(heads, triton.cdiv(rows, bt), triton.cdiv(dim, bk))](
        raw,
        fa,
        fb,
        beta,
        conv_weights,
        conv_pool,
        state_pool,
        read_indices,
        write_indices,
        qkv,
        gate,
        beta_dense,
        state,
        slots,
        *(replay_payload if replay_payload is not None else (None, None, None)),
        REPLAY_RAW_STRIDE=0 if replay_payload is None else replay_payload[0].stride(0),
        REPLAY_FA_STRIDE=0 if replay_payload is None else replay_payload[1].stride(0),
        REPLAY_BETA_STRIDE=0 if replay_payload is None else replay_payload[2].stride(0),
        CAPTURE_REPLAY=replay_payload is not None,
        ROWS=rows,
        H=heads,
        K=dim,
        DFA=fa.shape[1],
        TOKENS=tokens,
        RAW_STRIDE=raw.stride(0),
        FA_STRIDE=fa.stride(0),
        BETA_STRIDE=beta.stride(0),
        CONV_STRIDE=conv_pool.stride(0),
        STATE_STRIDE=state_pool.stride(0),
        STATE_OUT_STRIDE=state.stride(0),
        BT=bt,
        BK=bk,
        STORE_CONV=tokens == 1,
        GATHER_STATE=tokens > 1,
        DOT=bt >= 16,
        PDL=enable_pdl,
        num_warps=warps,
        **({"launch_pdl": True} if enable_pdl else {}),
    )
    return qkv, gate, beta_dense, state, slots
