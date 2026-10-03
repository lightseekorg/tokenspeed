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

"""Mamba2 SSD prefill scan and paged state update against a sequential oracle."""

import dataclasses
import math

import pytest
import torch
import torch.nn.functional as F
from tokenspeed_kernel.ops.attention.mamba2 import (
    build_mamba2_chunk_metadata,
    mamba2_chunk_scan,
    mamba2_replay_commit,
    mamba2_replay_commit_supported,
    mamba2_state_update,
    mamba2_verify_scan,
)
from tokenspeed_kernel.ops.attention.mamba2.triton import (
    _mamba2_replay_commit_kernel,
    _mamba2_verify_scan_kernel,
    _ssd_chunk_cb_kernel,
    _ssd_chunk_scan_kernel,
    _ssd_chunk_state_kernel,
    _ssd_state_passing_kernel,
)
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.selection import NoKernelFoundError
from utils import assert_no_triton_compile

pytestmark = pytest.mark.skipif(
    not current_platform().is_nvidia, reason="Mamba2 Triton kernels are NVIDIA-only"
)

# Nemotron-3 Super geometry at TP4: 32 heads of 64, 2 groups, d_state 128.
HEADS, HEAD_DIM, GROUPS, D_STATE, CHUNK = 32, 64, 2, 128, 128
DT_LIMIT = (0.0, float("inf"))


def _params(seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    A_log = torch.log(torch.arange(1, HEADS + 1, dtype=torch.float32, device="cuda"))
    D = torch.rand(HEADS, generator=g, device="cuda")
    # dt_bias = softplus^-1 of a step in [1e-3, 1e-1], as the model initializes it.
    step = torch.exp(
        torch.rand(HEADS, generator=g, device="cuda")
        * (torch.log(torch.tensor(0.1)) - torch.log(torch.tensor(1e-3)))
        + torch.log(torch.tensor(1e-3))
    )
    dt_bias = step + torch.log(-torch.expm1(-step))
    return A_log, D, dt_bias


def _inputs(tokens: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    # One packed row, as the in-projection produces them, sliced into strided views.
    width = HEADS * HEAD_DIM + 2 * GROUPS * D_STATE + HEADS
    row = torch.randn(tokens, width, generator=g, device="cuda").to(torch.bfloat16)
    x, B, C, dt = row.split(
        [HEADS * HEAD_DIM, GROUPS * D_STATE, GROUPS * D_STATE, HEADS], dim=-1
    )
    return (
        x.view(tokens, HEADS, HEAD_DIM),
        dt,
        B.view(tokens, GROUPS, D_STATE),
        C.view(tokens, GROUPS, D_STATE),
    )


def _oracle(x, dt, A_log, B, C, D, dt_bias, state):
    """Token-by-token recurrence in fp64 for one sequence; returns (y, final state)."""
    heads_per_group = x.shape[1] // B.shape[1]
    A = -torch.exp(A_log.double())
    s = state.double().clone()
    ys = []
    for t in range(x.shape[0]):
        step = F.softplus(dt[t].double() + dt_bias.double()).clamp(*DT_LIMIT)
        b = B[t].double().repeat_interleave(heads_per_group, dim=0)
        c = C[t].double().repeat_interleave(heads_per_group, dim=0)
        xt = x[t].double()
        s = (
            s * torch.exp(step * A)[:, None, None]
            + (step[:, None] * xt)[:, :, None] * b[:, None, :]
        )
        ys.append(torch.einsum("hpn,hn->hp", s, c) + D.double()[:, None] * xt)
    return torch.stack(ys).float(), s.float()


def _relative(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


def _scan(x, dt, B, C, A_log, D, dt_bias, lengths, initial, chunk=CHUNK, out=None):
    bounds = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32
    )
    out = torch.empty_like(x) if out is None else out
    final = mamba2_chunk_scan(
        x,
        dt,
        A_log,
        B,
        C,
        D,
        dt_bias,
        dt_limit=DT_LIMIT,
        initial_states=initial,
        cu_seqlens=bounds.cuda(),
        chunk_metadata=build_mamba2_chunk_metadata(bounds, chunk, torch.device("cuda")),
        out=out,
    )
    return out, final


def test_chunk_metadata_splits_at_sequence_and_chunk_boundaries():
    meta = build_mamba2_chunk_metadata(
        torch.tensor([0, 100, 300, 301], dtype=torch.int32),
        CHUNK,
        torch.device("cuda"),
    )
    # 100 | 28 + 128 + 44 | 1: chunks never cross 128 or a sequence end.
    assert meta.cu_chunk_seqlens.tolist() == [0, 100, 128, 256, 300, 301]
    assert meta.seq_idx.tolist() == [0, 1, 1, 1, 2]
    assert meta.last_chunk_indices.tolist() == [0, 3, 4]


def test_chunk_metadata_rejects_vectors_the_kernels_cannot_index():
    plan = build_mamba2_chunk_metadata(
        torch.tensor([0, 1, 2], dtype=torch.int32), CHUNK, torch.device("cuda")
    )
    spread = torch.tensor([0, 3, 1, 3, 2], dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="dense int32 vectors"):
        dataclasses.replace(plan, cu_chunk_seqlens=spread[::2])
    with pytest.raises(ValueError, match="dense int32 vectors"):
        dataclasses.replace(plan, seq_idx=plan.seq_idx.long())
    A_log, D, dt_bias = _params(54)
    x, dt, B, C = _inputs(3, seed=55)
    with pytest.raises(ValueError, match="different batch"):
        mamba2_chunk_scan(
            x,
            dt,
            A_log,
            B,
            C,
            D,
            dt_bias,
            dt_limit=DT_LIMIT,
            initial_states=torch.zeros(1, HEADS, HEAD_DIM, D_STATE, device="cuda"),
            cu_seqlens=torch.tensor([0, 3], dtype=torch.int32, device="cuda"),
            chunk_metadata=plan,
            out=torch.empty_like(x),
        )


def test_chunk_metadata_rejects_an_empty_sequence():
    # State passing finds a sequence's first chunk from its predecessor's last.
    with pytest.raises(ValueError, match="sequence 1 is empty"):
        build_mamba2_chunk_metadata(
            torch.tensor([0, 100, 100, 101], dtype=torch.int32),
            CHUNK,
            torch.device("cuda"),
        )


@pytest.mark.parametrize("lengths", [[1], [5, 128, 129], [300, 1, 77, 256]])
def test_chunk_scan_matches_the_sequential_recurrence(lengths):
    A_log, D, dt_bias = _params(0)
    x, dt, B, C = _inputs(sum(lengths), seed=1)
    g = torch.Generator(device="cuda").manual_seed(2)
    initial = (
        torch.randn(len(lengths), HEADS, HEAD_DIM, D_STATE, generator=g, device="cuda")
        * 0.1
    )
    initial[0] = 0  # a sequence without history starts from a zero row
    out, final = _scan(x, dt, B, C, A_log, D, dt_bias, lengths, initial)
    start = 0
    for i, n in enumerate(lengths):
        ref_y, ref_s = _oracle(
            x[start : start + n],
            dt[start : start + n],
            A_log,
            B[start : start + n],
            C[start : start + n],
            D,
            dt_bias,
            initial[i],
        )
        assert _relative(out[start : start + n], ref_y) < 1e-2, f"sequence {i} output"
        # The chunk state is accumulated from split BF16 operands: near-FP32 states.
        assert _relative(final[i], ref_s) < 2e-5, f"sequence {i} state"
        start += n
    assert final.dtype == torch.float32


def test_scanning_in_two_parts_equals_one_scan():
    """Chunked prefill resumes from the state the first part returns."""
    A_log, D, dt_bias = _params(3)
    x, dt, B, C = _inputs(500, seed=4)
    initial = torch.zeros(1, HEADS, HEAD_DIM, D_STATE, device="cuda")
    whole_y, whole_s = _scan(x, dt, B, C, A_log, D, dt_bias, [500], initial)
    head_y, head_s = _scan(
        x[:200], dt[:200], B[:200], C[:200], A_log, D, dt_bias, [200], initial
    )
    tail_y, tail_s = _scan(
        x[200:], dt[200:], B[200:], C[200:], A_log, D, dt_bias, [300], head_s
    )
    assert _relative(torch.cat([head_y, tail_y]), whole_y) < 5e-3
    assert _relative(tail_s, whole_s) < 2e-5


def test_state_update_reads_and_writes_the_named_slots():
    A_log, D, dt_bias = _params(5)
    x, dt, B, C = _inputs(3, seed=6)
    g = torch.Generator(device="cuda").manual_seed(7)
    pool = torch.randn(8, HEADS, HEAD_DIM, D_STATE, generator=g, device="cuda") * 0.1
    before = pool.clone()
    src = torch.tensor([2, 5, -1], dtype=torch.int32, device="cuda")
    dst = torch.tensor([6, 1, -1], dtype=torch.int32, device="cuda")
    out = torch.empty_like(x)
    mamba2_state_update(
        pool,
        x,
        dt,
        A_log,
        B,
        C,
        D,
        dt_bias,
        state_indices=src,
        dst_state_indices=dst,
        null_slot=-1,
        out=out,
    )
    for row, (s, d) in enumerate([(2, 6), (5, 1)]):
        ref_y, ref_s = _oracle(
            x[row : row + 1],
            dt[row : row + 1],
            A_log,
            B[row : row + 1],
            C[row : row + 1],
            D,
            dt_bias,
            before[s],
        )
        assert _relative(out[row], ref_y[0]) < 1e-2
        assert _relative(pool[d], ref_s) < 1e-4
        assert torch.equal(pool[s], before[s]), "the read slot stays as it was"
    untouched = [i for i in range(8) if i not in (6, 1)]
    assert torch.equal(
        pool[untouched], before[untouched]
    ), "a padded row writes nothing"


def test_decode_steps_continue_a_prefill_scan():
    """Prefill then single-token steps equal one scan over all tokens."""
    A_log, D, dt_bias = _params(8)
    x, dt, B, C = _inputs(140, seed=9)
    zero = torch.zeros(1, HEADS, HEAD_DIM, D_STATE, device="cuda")
    whole_y, whole_s = _scan(x, dt, B, C, A_log, D, dt_bias, [140], zero)
    _, prefill_s = _scan(
        x[:130], dt[:130], B[:130], C[:130], A_log, D, dt_bias, [130], zero
    )
    pool = torch.zeros(2, HEADS, HEAD_DIM, D_STATE, device="cuda")
    pool[0] = prefill_s[0]
    slot = torch.zeros(1, dtype=torch.int32, device="cuda")
    ys = []
    for t in range(130, 140):
        out = torch.empty_like(x[t : t + 1])
        mamba2_state_update(
            pool,
            x[t : t + 1],
            dt[t : t + 1],
            A_log,
            B[t : t + 1],
            C[t : t + 1],
            D,
            dt_bias,
            state_indices=slot,
            dst_state_indices=slot,
            null_slot=-1,
            out=out,
        )
        ys.append(out)
    # The decode kernel is exact fp32 arithmetic on the state it continues from.
    ref_y, ref_s = _oracle(
        x[130:], dt[130:], A_log, B[130:], C[130:], D, dt_bias, prefill_s[0]
    )
    assert _relative(torch.cat(ys), ref_y) < 1e-2
    assert _relative(pool[0], ref_s) < 1e-4
    # Against one chunked scan both paths carry bf16-input rounding.
    assert _relative(torch.cat(ys), whole_y[130:]) < 1e-2
    assert _relative(pool[0], whole_s[0]) < 5e-3


def _verify_window(batch: int, steps: int, seed: int):
    """``[batch, T]`` verify tokens as request-major strided views of one projection."""
    x, dt, B, C = _inputs(batch * steps, seed)
    return (
        x.view(batch, steps, HEADS, HEAD_DIM),
        dt.view(batch, steps, HEADS),
        B.view(batch, steps, GROUPS, D_STATE),
        C.view(batch, steps, GROUPS, D_STATE),
    )


def _decode_steps(pool, x, dt, B, C, A_log, D, dt_bias, reads, scratch_rows):
    """T single-token decode updates per request, each into its own scratch row."""
    batch, steps = x.shape[:2]
    outs = []
    src = reads
    for t in range(steps):
        out = torch.empty_like(x[:, t])
        dst = scratch_rows[:, t].contiguous()
        mamba2_state_update(
            pool,
            x[:, t],
            dt[:, t],
            A_log,
            B[:, t],
            C[:, t],
            D,
            dt_bias,
            state_indices=src,
            dst_state_indices=dst,
            null_slot=-1,
            out=out,
        )
        outs.append(out)
        src = dst
    return torch.stack(outs, dim=1)


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_verify_scan_steps_exactly_like_consecutive_decode_updates(state_dtype):
    """Verify rounds the state through the pool dtype per token, as decode's reload does."""
    batch, steps = 3, 4
    A_log, D, dt_bias = _params(5)
    x, dt, B, C = _verify_window(batch, steps, 6)
    g = torch.Generator(device="cuda").manual_seed(7)
    pool = (
        0.1
        * torch.randn(
            1 + batch * (steps + 1),
            HEADS,
            HEAD_DIM,
            D_STATE,
            generator=g,
            device="cuda",
        )
    ).to(state_dtype)
    reads = torch.tensor([1, 2, 3], dtype=torch.int32, device="cuda")
    rows = 4 + torch.arange(batch * steps, dtype=torch.int32, device="cuda").view(
        batch, steps
    )
    decode_pool = pool.clone()
    expected = _decode_steps(decode_pool, x, dt, B, C, A_log, D, dt_bias, reads, rows)

    out = torch.empty_like(x)
    mamba2_verify_scan(
        pool,
        x,
        dt,
        A_log,
        B,
        C,
        D,
        dt_bias,
        state_indices=reads,
        dst_state_indices=rows,
        null_slot=-1,
        out=out,
    )
    assert torch.equal(pool, decode_pool)
    assert torch.equal(out, expected)

    # Without destinations the pool is untouched and the outputs are the same.
    untouched = pool.clone()
    replay_out = torch.empty_like(x)
    mamba2_verify_scan(
        pool,
        x,
        dt,
        A_log,
        B,
        C,
        D,
        dt_bias,
        state_indices=reads,
        dst_state_indices=None,
        null_slot=-1,
        out=replay_out,
    )
    assert torch.equal(pool, untouched)
    assert torch.equal(replay_out, out)


def test_verify_scan_rejects_strided_heads_and_malformed_parameters():
    batch, steps = 2, 3
    A_log, D, dt_bias = _params(12)
    x, dt, B, C = _verify_window(batch, steps, 13)
    pool = torch.zeros(4, HEADS, HEAD_DIM, D_STATE, device="cuda")
    args = dict(
        state_indices=torch.tensor([1, 2], dtype=torch.int32, device="cuda"),
        dst_state_indices=None,
        null_slot=-1,
        out=torch.empty_like(x),
    )
    strided_dt = dt.transpose(1, 2).contiguous().transpose(1, 2)
    with pytest.raises(ValueError, match="contiguous in their last dim"):
        mamba2_verify_scan(pool, x, strided_dt, A_log, B, C, D, dt_bias, **args)
    per_dim_D = D[:, None].expand(HEADS, HEAD_DIM)
    with pytest.raises(ValueError, match="dense \\[heads\\] vectors"):
        mamba2_verify_scan(pool, x, dt, A_log, B, C, per_dim_D, dt_bias, **args)


def test_verify_scan_skips_padded_requests_and_destinations():
    batch, steps = 2, 3
    A_log, D, dt_bias = _params(8)
    x, dt, B, C = _verify_window(batch, steps, 9)
    pool = torch.zeros(8, HEADS, HEAD_DIM, D_STATE, device="cuda")
    pool[1] = 0.5
    before = pool.clone()
    rows = torch.tensor([[2, -1, 3], [4, 5, 6]], dtype=torch.int32, device="cuda")
    out = torch.empty_like(x)
    mamba2_verify_scan(
        pool,
        x,
        dt,
        A_log,
        B,
        C,
        D,
        dt_bias,
        state_indices=torch.tensor([1, -1], dtype=torch.int32, device="cuda"),
        dst_state_indices=rows,
        null_slot=-1,
        out=out,
    )
    assert torch.equal(pool[1], before[1])
    assert torch.equal(pool[7], before[7])
    zero_start = torch.zeros(HEADS, HEAD_DIM, D_STATE, device="cuda")
    _, first = _oracle(
        x[1, :1], dt[1, :1], A_log, B[1, :1], C[1, :1], D, dt_bias, zero_start
    )
    assert _relative(pool[4], first) < 1e-4


def _replay_payload(x, dt, B):
    """The verify split's ``[K | V | a | b]`` rows; SSD repeats dt in the unused b slot."""
    batch, steps = x.shape[:2]
    rows = batch * steps
    return torch.cat(
        [
            B.reshape(rows, -1),
            x.reshape(rows, -1),
            dt.reshape(rows, -1),
            dt.reshape(rows, -1),
        ],
        dim=-1,
    )


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_replay_commit_rebuilds_the_accepted_state_of_every_layer(state_dtype):
    batch, steps, layers = 4, 3, 2
    accepted = torch.tensor([0, 1, 3, 2], dtype=torch.int32, device="cuda")
    g = torch.Generator(device="cuda").manual_seed(11)
    pools, payloads, params, per_token = [], [], [], []
    for layer in range(layers):
        A_log, D, dt_bias = _params(20 + layer)
        x, dt, B, C = _verify_window(batch, steps, 30 + layer)
        # Separate pools with different row strides, as separate cache planes are.
        storage = (
            0.1
            * torch.randn(
                1 + batch * (steps + 2),
                HEADS * HEAD_DIM * D_STATE + 64 * layer,
                generator=g,
                device="cuda",
            )
        ).to(state_dtype)
        pool = storage[:, : HEADS * HEAD_DIM * D_STATE].view(
            -1, HEADS, HEAD_DIM, D_STATE
        )
        scratch = pool.clone()
        rows = (
            1
            + batch
            + torch.arange(batch * steps, dtype=torch.int32, device="cuda").view(
                batch, steps
            )
        )
        mamba2_verify_scan(
            scratch,
            x,
            dt,
            A_log,
            B,
            C,
            D,
            dt_bias,
            state_indices=torch.arange(1, batch + 1, dtype=torch.int32, device="cuda"),
            dst_state_indices=rows,
            null_slot=-1,
            out=torch.empty_like(x),
        )
        pools.append((storage, pool))
        per_token.append((scratch, rows))
        payloads.append(_replay_payload(x, dt, B))
        params.append(torch.stack([A_log, dt_bias]))

    payload = torch.stack(payloads).contiguous()
    reads = torch.arange(1, batch + 1, dtype=torch.int32, device="cuda").repeat(
        layers, 1
    )
    writes = reads.clone() + batch * (steps + 1)
    writes[1, 2] = -1
    committed = [pool.clone() for _, pool in pools]
    mamba2_replay_commit(
        payload,
        torch.stack(params).contiguous(),
        state_addresses=torch.tensor(
            [p.data_ptr() for _, p in pools], dtype=torch.uint64, device="cuda"
        ),
        state_row_strides=torch.tensor(
            [p.stride(0) for _, p in pools], dtype=torch.int64, device="cuda"
        ),
        read_indices=reads,
        write_indices=writes,
        accepted_length=accepted,
        draft_token_num=steps,
        geometry=(GROUPS, HEADS, D_STATE, HEAD_DIM),
        state_dtype=state_dtype,
    )
    for layer, ((_, pool), (scratch, rows)) in enumerate(zip(pools, per_token)):
        for req in range(batch):
            dst = int(writes[layer, req])
            if dst < 0:
                continue
            k = int(accepted[req])
            source = scratch[1 + req] if k == 0 else scratch[int(rows[req, k - 1])]
            assert torch.equal(pool[dst], source), (layer, req, k)
    # The skipped destination left its row alone; committed pages are never written.
    assert torch.equal(
        pools[1][1][1 + batch * (steps + 1) + 2],
        committed[1][1 + batch * (steps + 1) + 2],
    )
    assert torch.equal(pools[0][1][1 : batch + 1], committed[0][1 : batch + 1])


def test_replay_commit_launches_past_the_65535_grid_cap():
    """40 layers x 16 requests x 128 heads overflows a y/z grid axis."""
    layers, batch, steps, heads = 40, 16, 2, 128
    payload = torch.zeros(
        layers,
        batch * steps,
        GROUPS * D_STATE + heads * HEAD_DIM + 2 * heads,
        dtype=torch.bfloat16,
        device="cuda",
    )
    pool = torch.zeros(2 * batch, heads, HEAD_DIM, D_STATE, device="cuda")
    pool[:batch] = 1
    pages = torch.arange(batch, dtype=torch.int32, device="cuda").repeat(layers, 1)
    mamba2_replay_commit(
        payload,
        torch.zeros(layers, 2, heads, device="cuda"),
        state_addresses=torch.full(
            (layers,), pool.data_ptr(), dtype=torch.uint64, device="cuda"
        ),
        state_row_strides=torch.full(
            (layers,), pool.stride(0), dtype=torch.int64, device="cuda"
        ),
        read_indices=pages,
        write_indices=pages + batch,
        accepted_length=torch.zeros(batch, dtype=torch.int32, device="cuda"),
        draft_token_num=steps,
        geometry=(GROUPS, heads, D_STATE, HEAD_DIM),
        state_dtype=torch.float32,
    )
    torch.cuda.synchronize()
    assert bool((pool == 1).all())


def test_verify_and_replay_compile_once_across_batch_sizes():
    steps, max_batch = 3, 16
    A_log, D, dt_bias = _params(40)
    pool = torch.zeros(1 + 2 * max_batch, HEADS, HEAD_DIM, D_STATE, device="cuda")
    addresses = torch.tensor([pool.data_ptr()], dtype=torch.uint64, device="cuda")
    row_strides = torch.tensor([pool.stride(0)], dtype=torch.int64, device="cuda")
    parameters = torch.stack([A_log, dt_bias]).unsqueeze(0).contiguous()

    def run(batch: int) -> None:
        x, dt, B, C = _verify_window(batch, steps, batch)
        reads = torch.arange(1, batch + 1, dtype=torch.int32, device="cuda")
        mamba2_verify_scan(
            pool,
            x,
            dt,
            A_log,
            B,
            C,
            D,
            dt_bias,
            state_indices=reads,
            dst_state_indices=None,
            null_slot=-1,
            out=torch.empty_like(x),
        )
        mamba2_replay_commit(
            _replay_payload(x, dt, B).unsqueeze(0).contiguous(),
            parameters,
            state_addresses=addresses,
            state_row_strides=row_strides,
            read_indices=reads.unsqueeze(0).contiguous(),
            write_indices=(reads + max_batch).unsqueeze(0).contiguous(),
            accepted_length=torch.full(
                (batch,), steps, dtype=torch.int32, device="cuda"
            ),
            draft_token_num=steps,
            geometry=(GROUPS, HEADS, D_STATE, HEAD_DIM),
            state_dtype=torch.float32,
        )

    run(3)
    with assert_no_triton_compile(
        _mamba2_verify_scan_kernel, _mamba2_replay_commit_kernel
    ):
        for batch in (1, 2, 8, max_batch):
            run(batch)


def test_chunk_scan_compiles_once_across_batch_shapes():
    A_log, D, dt_bias = _params(41)
    x, dt, B, C = _inputs(900, seed=42)
    initial = torch.zeros(8, HEADS, HEAD_DIM, D_STATE, device="cuda")

    def run(lengths: list[int]) -> None:
        total = sum(lengths)
        _scan(
            x[:total],
            dt[:total],
            B[:total],
            C[:total],
            A_log,
            D,
            dt_bias,
            lengths,
            initial[: len(lengths)],
        )

    run([3])
    with assert_no_triton_compile(
        _ssd_chunk_state_kernel,
        _ssd_state_passing_kernel,
        _ssd_chunk_cb_kernel,
        _ssd_chunk_scan_kernel,
    ):
        for lengths in ([1], [128], [129, 7], [300, 1, 77, 256], [16] * 8):
            run(lengths)


def _small_problem(tokens, heads, head_dim, groups, d_state, seqs, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    width = heads * head_dim + 2 * groups * d_state + heads
    row = torch.randn(tokens, width, generator=g, device="cuda").to(torch.bfloat16)
    x, B, C, dt = row.split(
        [heads * head_dim, groups * d_state, groups * d_state, heads], dim=-1
    )
    A_log = torch.log(torch.rand(heads, generator=g, device="cuda") * 8 + 1)
    D = torch.rand(heads, generator=g, device="cuda")
    dt_bias = torch.rand(heads, generator=g, device="cuda") - 4
    initial = (
        torch.randn(seqs, heads, head_dim, d_state, generator=g, device="cuda") * 0.1
    )
    return (
        x.view(tokens, heads, head_dim),
        dt,
        B.view(tokens, groups, d_state),
        C.view(tokens, groups, d_state),
        A_log,
        D,
        dt_bias,
        initial,
    )


@pytest.mark.parametrize("chunk", [16, 64, 256])
@pytest.mark.parametrize("geometry", [(4, 40, 2, 24), (2, 8, 1, 7)])
def test_chunk_scan_handles_small_and_uneven_geometry(chunk, geometry):
    """Dimensions below the 16-wide tensor-core minimum and off powers of two."""
    heads, head_dim, groups, d_state = geometry
    lengths = [1, 300, 17]
    x, dt, B, C, A_log, D, dt_bias, initial = _small_problem(
        sum(lengths), heads, head_dim, groups, d_state, len(lengths), seed=chunk
    )
    out, final = _scan(x, dt, B, C, A_log, D, dt_bias, lengths, initial, chunk=chunk)
    start = 0
    for i, n in enumerate(lengths):
        span = slice(start, start + n)
        ref_y, ref_s = _oracle(
            x[span], dt[span], A_log, B[span], C[span], D, dt_bias, initial[i]
        )
        assert _relative(out[span], ref_y) < 1e-2, f"sequence {i} output"
        assert _relative(final[i], ref_s) < 2e-5, f"sequence {i} state"
        start += n


def test_chunk_scan_rejects_more_chunks_than_its_grid_holds():
    """70,000 single-token sequences are 70,000 chunks, past the 65,535 grid cap."""
    seqs = 70_000
    x, dt, B, C, A_log, D, dt_bias, initial = _small_problem(
        seqs, 1, 16, 1, 16, seqs, seed=5
    )
    with pytest.raises(ValueError, match="scan grid holds"):
        _scan(x, dt, B, C, A_log, D, dt_bias, [1] * seqs, initial, chunk=16)


def test_ops_reject_operands_the_kernels_cannot_index():
    A_log, D, dt_bias = _params(50)
    x, dt, B, C = _inputs(8, seed=51)
    wide = torch.zeros(9, HEADS, HEAD_DIM, D_STATE, device="cuda")
    with pytest.raises(ValueError, match="initial_states"):
        _scan(x, dt, B, C, A_log, D, dt_bias, [8], wide[:, :1])
    strided = torch.zeros(8, HEADS, 2 * HEAD_DIM, device="cuda").bfloat16()[..., ::2]
    with pytest.raises(ValueError, match="contiguous in their last dim"):
        _scan(strided, dt, B, C, A_log, D, dt_bias, [8], wide[:1])
    expanded = torch.empty(1, HEADS, HEAD_DIM, device="cuda").bfloat16()
    with pytest.raises(ValueError, match="out contiguous"):
        _scan(
            x,
            dt,
            B,
            C,
            A_log,
            D,
            dt_bias,
            [8],
            wide[:1],
            out=expanded.expand(8, -1, -1),
        )
    shifted = torch.zeros(9, HEADS, HEAD_DIM, device="cuda").bfloat16()
    with pytest.raises(ValueError, match="must not share storage"):
        _scan(shifted[:8], dt, B, C, A_log, D, dt_bias, [8], wide[:1], out=shifted[1:])
    # Meta storage: validation rejects this row before a launch could index it.
    far = torch.empty(HEADS, 2**22, HEAD_DIM, device="meta", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="over int32"):
        _scan(
            far.transpose(0, 1)[:8],
            dt,
            B,
            C,
            A_log,
            D,
            dt_bias,
            [8],
            wide[:1],
            out=torch.empty_like(x),
        )

    slots = torch.tensor([1, 2], dtype=torch.int32, device="cuda")

    def decode(pool, step=dt[:2], out=None):
        mamba2_state_update(
            pool,
            x[:2],
            step,
            A_log,
            B[:2],
            C[:2],
            D,
            dt_bias,
            state_indices=slots,
            dst_state_indices=slots,
            null_slot=-1,
            out=torch.empty_like(x[:2]) if out is None else out,
        )

    pool = torch.zeros(4, HEADS, HEAD_DIM, D_STATE, device="cuda")
    with pytest.raises(ValueError, match="contiguous in their last dim"):
        decode(pool, step=torch.zeros(2, 2 * HEADS, device="cuda").bfloat16()[:, ::2])
    with pytest.raises(ValueError, match="dense within a slot"):
        decode(torch.zeros(4, 2 * HEADS, HEAD_DIM, D_STATE, device="cuda")[:, ::2])
    with pytest.raises(ValueError, match="slots must not overlap"):
        decode(pool[:1].expand(4, -1, -1, -1))
    in_pool = pool[3].view(torch.bfloat16).flatten()[: 2 * HEADS * HEAD_DIM]
    with pytest.raises(ValueError, match="must not share storage"):
        decode(pool, out=in_pool.view(2, HEADS, HEAD_DIM))


def test_ops_take_bf16_activations_only():
    A_log, D, dt_bias = _params(52)
    x, dt, B, C = (t.half() for t in _inputs(8, seed=53))
    initial = torch.zeros(1, HEADS, HEAD_DIM, D_STATE, device="cuda")
    with pytest.raises(NoKernelFoundError):
        _scan(x, dt, B, C, A_log, D, dt_bias, [8], initial)
    assert not mamba2_replay_commit_supported(torch.float16)


def test_verify_and_decode_reject_more_requests_than_their_grid_holds():
    batch, zeros = 65536, torch.zeros(1, device="cuda")
    x = torch.zeros(batch, 1, 1, 16, device="cuda").bfloat16()
    dt = torch.zeros(batch, 1, 1, device="cuda").bfloat16()
    B = torch.zeros(batch, 1, 1, 16, device="cuda").bfloat16()
    reads = torch.full((batch,), -1, dtype=torch.int32, device="cuda")
    pool = torch.zeros(1, 1, 16, 16, device="cuda")
    args = dict(state_indices=reads, dst_state_indices=None, null_slot=-1)
    with pytest.raises(ValueError, match="verify grid holds"):
        mamba2_verify_scan(
            pool, x, dt, zeros, B, B, zeros, zeros, **args, out=torch.empty_like(x)
        )
    args["dst_state_indices"] = reads
    with pytest.raises(ValueError, match="verify grid holds"):
        mamba2_state_update(
            pool,
            x[:, 0],
            dt[:, 0],
            zeros,
            B[:, 0],
            B[:, 0],
            zeros,
            zeros,
            **args,
            out=torch.empty_like(x[:, 0]),
        )


def test_tiny_steps_survive_softplus():
    """softplus(-20) is 2e-9, which fp32 loses in log(exp(x) + 1)."""
    zeros = torch.zeros(HEADS, device="cuda")
    x = torch.ones(1, HEADS, HEAD_DIM, device="cuda").bfloat16()
    dt = torch.full((1, HEADS), -20.0, device="cuda").bfloat16()
    B = torch.full((1, GROUPS, D_STATE), 2.0**20, device="cuda").bfloat16()
    initial = torch.zeros(1, HEADS, HEAD_DIM, D_STATE, device="cuda")
    _, final = _scan(x, dt, B, B, zeros, zeros, zeros, [1], initial)
    expected = torch.full_like(final, math.log1p(math.exp(-20.0)) * 2.0**20)
    torch.testing.assert_close(final, expected, rtol=1e-5, atol=0)
    pool = torch.zeros(2, HEADS, HEAD_DIM, D_STATE, device="cuda")
    slots = torch.tensor([0], dtype=torch.int32, device="cuda")
    mamba2_state_update(
        pool,
        x,
        dt,
        zeros,
        B,
        B,
        zeros,
        zeros,
        state_indices=slots,
        dst_state_indices=slots + 1,
        null_slot=-1,
        out=torch.empty_like(x),
    )
    torch.testing.assert_close(pool[1], expected[0], rtol=1e-5, atol=0)
