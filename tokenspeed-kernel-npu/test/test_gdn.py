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

"""Numerics of the Ascend Gated Delta Net against an FP32 recurrence.

The reference is the recurrence itself, one token at a time, with the gating
computed independently::

    S' = S * exp(g) + beta * (v - (S * exp(g)) @ k) @ k^T
    o  = S' @ q * scale

so a wrong fusion order, a wrong index layout or a wrong beta activation shows
up as a numeric difference rather than as a shape error.
"""

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from tokenspeed_kernel_npu.ops.gdn import (
    gdn_chunk_prefill,
    gdn_decode_mtp,
    gdn_decode_step,
    gdn_gating,
)

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="Ascend GDN tests require an NPU"
)

# Qwen3.5/3.6 geometry: 16 key heads, 32 value heads, 128-wide both ways.
NUM_K_HEADS, NUM_V_HEADS, HEAD_DIM = 16, 32, 128
RTOL = 0.02


def _reference(q, k, v, g, beta, initial, lengths):
    """Step-by-step Gated Delta Rule in float32."""
    total = q.shape[0]
    out = torch.zeros(
        total, NUM_V_HEADS, HEAD_DIM, dtype=torch.float32, device=q.device
    )
    finals = torch.zeros_like(initial, dtype=torch.float32)
    group = NUM_V_HEADS // NUM_K_HEADS
    start = 0
    for request, length in enumerate(lengths):
        state = initial[request].float().clone()
        for step in range(length):
            token = start + step
            for head in range(NUM_V_HEADS):
                key = k[token, head // group].float()
                value = v[token, head].float()
                query = q[token, head // group].float()
                state[head] = state[head] * torch.exp(g[token, head].float())
                projected = (state[head] * key[None, :]).sum(-1)
                state[head] += ((value - projected) * beta[token, head].float())[
                    :, None
                ] * key[None, :]
                out[token, head] = (state[head] * query[None, :]).sum(-1)
        finals[request] = state
        start += length
    return out, finals


def _inputs(batch, length, seed, rows=None):
    """Q/K are L2-normalized, as the real model feeds them."""
    torch.manual_seed(seed)
    total = batch * length
    q = torch.nn.functional.normalize(
        torch.randn(total, NUM_K_HEADS, HEAD_DIM, device="npu").float(), p=2, dim=-1
    ).to(torch.bfloat16)
    k = torch.nn.functional.normalize(
        torch.randn(total, NUM_K_HEADS, HEAD_DIM, device="npu").float(), p=2, dim=-1
    ).to(torch.bfloat16)
    v = torch.randn(total, NUM_V_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="npu")
    g = -torch.rand(total, NUM_V_HEADS, dtype=torch.float32, device="npu")
    beta = torch.rand(total, NUM_V_HEADS, dtype=torch.bfloat16, device="npu")
    pool_rows = rows if rows is not None else max(batch, total)
    pool = (
        torch.randn(
            pool_rows,
            NUM_V_HEADS,
            HEAD_DIM,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device="npu",
        )
        * 0.1
    )
    return {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "pool": pool,
        "batch": batch,
        "length": length,
        "total": total,
    }


def _gates(batch, length, seed):
    torch.manual_seed(seed)
    shape = (batch, length, NUM_V_HEADS)
    return (
        torch.randn(NUM_V_HEADS, dtype=torch.float32, device="npu") * 0.1,
        torch.randn(NUM_V_HEADS, dtype=torch.float32, device="npu") * 0.1,
        torch.randn(shape, dtype=torch.bfloat16, device="npu"),
        torch.randn(shape, dtype=torch.bfloat16, device="npu"),
    )


def _assert_close(got, expected, what):
    error = (got.float() - expected).abs().max().item()
    peak = expected.abs().max().item() + 1e-9
    assert error / peak < RTOL, f"{what}: rel error {error / peak:.4f}"


def _assert_state_close(got, expected):
    # The recurrent state is float32 in the model and bf16 in the pool, so the
    # absolute error is compared against its own scale.
    error = (got.float() - expected).abs().max().item()
    scale = expected.abs().max().item() + 1e-9
    assert error / scale < 0.05, f"state: rel error {error / scale:.4f}"


@pytest.mark.parametrize(("lengths", "seed"), [([3, 3], 61), ([1, 5, 2], 67)])
def test_chunk_prefill_matches_recurrence(lengths, seed):
    batch = len(lengths)
    tensors = _inputs(batch, max(lengths), seed)
    # Repack the variable lengths into one contiguous batch.
    total = sum(lengths)
    q = torch.cat(
        [
            tensors["q"][i * max(lengths) : i * max(lengths) + length]
            for i, length in enumerate(lengths)
        ]
    )
    k = torch.cat(
        [
            tensors["k"][i * max(lengths) : i * max(lengths) + length]
            for i, length in enumerate(lengths)
        ]
    )
    v = torch.cat(
        [
            tensors["v"][i * max(lengths) : i * max(lengths) + length]
            for i, length in enumerate(lengths)
        ]
    )
    g = torch.cat(
        [
            tensors["g"][i * max(lengths) : i * max(lengths) + length]
            for i, length in enumerate(lengths)
        ]
    )
    beta = torch.cat(
        [
            tensors["beta"][i * max(lengths) : i * max(lengths) + length]
            for i, length in enumerate(lengths)
        ]
    )
    assert q.shape[0] == total

    initial = tensors["pool"][:batch].clone()
    cumulative = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(lengths), 0).tolist()),
        dtype=torch.int32,
        device="npu",
    )
    out, final = gdn_chunk_prefill(
        q.unsqueeze(0),
        k.unsqueeze(0),
        v.unsqueeze(0),
        g.unsqueeze(0),
        beta.unsqueeze(0),
        scale=None,
        initial_state=initial,
        cu_seqlens=cumulative,
        qk_l2norm=False,
    )
    expected_out, expected_final = _reference(
        q * HEAD_DIM**-0.5, k, v, g, beta, tensors["pool"][:batch], lengths
    )
    _assert_close(out[0].reshape(total, NUM_V_HEADS, HEAD_DIM), expected_out, "output")
    _assert_state_close(final, expected_final)


def test_decode_step_writes_the_target_row_and_leaves_the_read_row():
    batch, seed = 4, 71
    tensors = _inputs(batch, 1, seed, rows=8)
    A_log, dt_bias, a, b = _gates(batch, 1, seed + 1)
    pool = tensors["pool"].clone()
    read = torch.arange(batch, dtype=torch.int32, device="npu")
    write = torch.arange(batch, 2 * batch, dtype=torch.int32, device="npu")

    out = gdn_decode_step(
        tensors["q"].view(batch, 1, NUM_K_HEADS, HEAD_DIM),
        tensors["k"].view(batch, 1, NUM_K_HEADS, HEAD_DIM),
        tensors["v"].view(batch, 1, NUM_V_HEADS, HEAD_DIM),
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        initial_state=pool,
        initial_state_indices=read,
        output_state_indices=write,
        use_qk_l2norm=False,
    )
    decay = gdn_gating(A_log, a, dt_bias).reshape(batch, NUM_V_HEADS)
    beta = torch.sigmoid(b.float()).to(torch.bfloat16).reshape(batch, NUM_V_HEADS)
    expected_out, expected_final = _reference(
        tensors["q"] * HEAD_DIM**-0.5,
        tensors["k"],
        tensors["v"],
        decay,
        beta,
        tensors["pool"][:batch],
        [1] * batch,
    )
    _assert_close(out.reshape(batch, NUM_V_HEADS, HEAD_DIM), expected_out, "output")
    _assert_state_close(pool[write.long()], expected_final)
    assert torch.equal(
        pool[read.long()], tensors["pool"][:batch]
    ), "the committed rows the call only read were overwritten"


def test_decode_mtp_stages_one_state_row_per_token():
    """Verify writes each position's state to its own row and leaves the
    committed rows untouched, which is the runtime's scratch-grid layout."""
    batch, steps, seed = 2, 4, 73
    tensors = _inputs(batch, steps, seed)
    A_log, dt_bias, a, b = _gates(batch, steps, seed + 1)
    read = torch.arange(batch, dtype=torch.int32, device="npu")
    committed = tensors["pool"][:batch].clone()

    # The runtime's grid: request r owns rows r*(T+1), the seeded window sits
    # on the first and position t leaves its state on r*(T+1)+1+t.
    rows_needed = batch * (steps + 1)
    scratch = torch.zeros(
        rows_needed,
        NUM_V_HEADS,
        HEAD_DIM,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="npu",
    )
    base = torch.arange(batch, device="npu") * (steps + 1)
    grid = base.unsqueeze(1) + torch.arange(1, steps + 1, device="npu").unsqueeze(0)
    assert grid[0, 0].item() == 1

    out = gdn_decode_mtp(
        tensors["q"].view(batch, steps, NUM_K_HEADS, HEAD_DIM),
        tensors["k"].view(batch, steps, NUM_K_HEADS, HEAD_DIM),
        tensors["v"].view(batch, steps, NUM_V_HEADS, HEAD_DIM),
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        initial_state=tensors["pool"],
        initial_state_indices=read,
        output_state=scratch,
        output_state_indices=grid,
        use_qk_l2norm=False,
    )
    decay = gdn_gating(A_log, a, dt_bias).reshape(-1, NUM_V_HEADS)
    beta = torch.sigmoid(b.float()).to(torch.bfloat16).reshape(-1, NUM_V_HEADS)
    expected_out, _ = _reference(
        tensors["q"] * HEAD_DIM**-0.5,
        tensors["k"],
        tensors["v"],
        decay,
        beta,
        committed,
        [steps] * batch,
    )
    _assert_close(out.reshape(-1, NUM_V_HEADS, HEAD_DIM), expected_out, "output")
    assert torch.equal(
        tensors["pool"][:batch], committed
    ), "a staged verify wrote the committed state"
    # The grid's first row is both the window's start and position 0's
    # destination, so it holds that position's state afterwards.
    for request in range(batch):
        got = scratch[grid[request, 0]].float()
        want = _reference(
            tensors["q"][request * steps : request * steps + 1] * HEAD_DIM**-0.5,
            tensors["k"][request * steps : request * steps + 1],
            tensors["v"][request * steps : request * steps + 1],
            decay[request * steps : request * steps + 1],
            beta[request * steps : request * steps + 1],
            committed[request : request + 1],
            [1],
        )[1][0]
        _assert_state_close(got, want)


def test_decode_mtp_writes_the_final_state_to_one_row_per_request():
    """A uniform grid (every position naming the same row) leaves each request's
    state after its last token on that row and its committed row untouched."""
    batch, steps, seed = 2, 3, 77
    tensors = _inputs(batch, steps, seed, rows=8)
    A_log, dt_bias, a, b = _gates(batch, steps, seed + 1)
    pool = tensors["pool"].clone()
    read = torch.arange(batch, dtype=torch.int32, device="npu")
    write = torch.arange(batch, 2 * batch, dtype=torch.int32, device="npu")
    grid = write.unsqueeze(1).repeat(1, steps)

    out = gdn_decode_mtp(
        tensors["q"].view(batch, steps, NUM_K_HEADS, HEAD_DIM),
        tensors["k"].view(batch, steps, NUM_K_HEADS, HEAD_DIM),
        tensors["v"].view(batch, steps, NUM_V_HEADS, HEAD_DIM),
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        initial_state=pool,
        initial_state_indices=read,
        output_state_indices=grid,
        use_qk_l2norm=False,
    )
    decay = gdn_gating(A_log, a, dt_bias).reshape(-1, NUM_V_HEADS)
    beta = torch.sigmoid(b.float()).to(torch.bfloat16).reshape(-1, NUM_V_HEADS)
    expected_out, expected_final = _reference(
        tensors["q"] * HEAD_DIM**-0.5,
        tensors["k"],
        tensors["v"],
        decay,
        beta,
        tensors["pool"][:batch],
        [steps] * batch,
    )
    _assert_close(out.reshape(-1, NUM_V_HEADS, HEAD_DIM), expected_out, "output")
    _assert_state_close(pool[write.long()], expected_final)
    assert torch.equal(
        pool[read.long()], tensors["pool"][:batch]
    ), "the committed rows the call only read were overwritten"


def test_decode_mtp_leaves_the_null_page_and_live_rows_alone_for_padding():
    """A padding row (``-1``) reads the zero state and writes nothing: the
    reserved null page (row 0) keeps its contents and no live row moves."""
    real, steps, seed = 2, 3, 71
    tensors = _inputs(real, steps, seed, rows=16)
    A_log, dt_bias, a, b = _gates(real, steps, seed + 1)
    pool = tensors["pool"].clone()
    # Row 0 is the runtime's reserved null page.
    pool[0] = 0
    null_page = pool[0].clone()
    live = pool[list(range(1, 5))].clone()

    # One padding row appended to the batch.
    q = torch.cat(
        [
            tensors["q"].view(real, steps, NUM_K_HEADS, HEAD_DIM),
            torch.zeros(
                1, steps, NUM_K_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="npu"
            ),
        ]
    )
    k = torch.cat(
        [
            tensors["k"].view(real, steps, NUM_K_HEADS, HEAD_DIM),
            torch.zeros(
                1, steps, NUM_K_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="npu"
            ),
        ]
    )
    v = torch.cat(
        [
            tensors["v"].view(real, steps, NUM_V_HEADS, HEAD_DIM),
            torch.zeros(
                1, steps, NUM_V_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="npu"
            ),
        ]
    )
    a_pad = torch.cat(
        [a, torch.zeros(1, steps, NUM_V_HEADS, dtype=torch.bfloat16, device="npu")]
    )
    b_pad = torch.cat(
        [b, torch.zeros(1, steps, NUM_V_HEADS, dtype=torch.bfloat16, device="npu")]
    )
    indices = torch.tensor([1, 3, -1], dtype=torch.int32, device="npu")
    grid = torch.tensor(
        [[1, 2, 3], [5, 6, 7], [9, 10, 11]], dtype=torch.int32, device="npu"
    )
    scratch = torch.zeros(
        16, NUM_V_HEADS, HEAD_DIM, HEAD_DIM, dtype=torch.bfloat16, device="npu"
    )

    gdn_decode_mtp(
        q,
        k,
        v,
        A_log=A_log,
        a=a_pad,
        dt_bias=dt_bias,
        b=b_pad,
        initial_state=pool,
        initial_state_indices=indices,
        output_state=scratch,
        output_state_indices=grid,
        tree_ancestors=None,
        use_qk_l2norm=False,
    )
    assert torch.equal(pool[0], null_page), "the padding row dirtied the null page"
    assert torch.equal(pool[1:5], live), "the padding row dirtied a live row"


def test_decode_step_accepts_a_float32_pool():
    """Qwen3.6 selects a float32 recurrent pool through ``--mamba-ssm-dtype``."""
    batch, seed = 2, 91
    tensors = _inputs(batch, 1, seed, rows=16)
    A_log, dt_bias, a, b = _gates(batch, 1, seed + 1)
    pool = tensors["pool"].to(torch.float32)
    read = torch.tensor([1, 3], dtype=torch.int32, device="npu")
    write = torch.tensor([2, 4], dtype=torch.int32, device="npu")
    committed = pool[read.long()].clone()

    out = gdn_decode_step(
        tensors["q"].view(batch, 1, NUM_K_HEADS, HEAD_DIM),
        tensors["k"].view(batch, 1, NUM_K_HEADS, HEAD_DIM),
        tensors["v"].view(batch, 1, NUM_V_HEADS, HEAD_DIM),
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        initial_state=pool,
        initial_state_indices=read,
        output_state_indices=write,
        use_qk_l2norm=False,
    )
    decay = gdn_gating(A_log, a, dt_bias).reshape(batch, NUM_V_HEADS)
    beta = torch.sigmoid(b.float()).to(torch.bfloat16).reshape(batch, NUM_V_HEADS)
    expected_out, expected_final = _reference(
        tensors["q"] * HEAD_DIM**-0.5,
        tensors["k"],
        tensors["v"],
        decay,
        beta,
        committed,
        [1] * batch,
    )
    _assert_close(out.reshape(batch, NUM_V_HEADS, HEAD_DIM), expected_out, "output")
    _assert_state_close(pool[write.long()], expected_final)
    assert torch.equal(
        pool[read.long()], committed
    ), "the committed rows the call only read were overwritten"


def test_recurrent_window_rejects_a_pool_smaller_than_the_token_count():
    batch, steps, seed = 2, 4, 83
    tensors = _inputs(batch, steps, seed, rows=3)
    A_log, dt_bias, a, b = _gates(batch, steps, seed + 1)
    with pytest.raises(ValueError, match="BlockNum"):
        gdn_decode_mtp(
            tensors["q"].view(batch, steps, NUM_K_HEADS, HEAD_DIM),
            tensors["k"].view(batch, steps, NUM_K_HEADS, HEAD_DIM),
            tensors["v"].view(batch, steps, NUM_V_HEADS, HEAD_DIM),
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            b=b,
            initial_state=tensors["pool"],
            initial_state_indices=torch.zeros(batch, dtype=torch.int32, device="npu"),
            disable_state_update=False,
            use_qk_l2norm=False,
        )


def test_gating_is_minus_exp_A_log_times_clamped_softplus():
    torch.manual_seed(89)
    A_log = torch.randn(NUM_V_HEADS, dtype=torch.float32, device="npu")
    dt_bias = torch.randn(NUM_V_HEADS, dtype=torch.float32, device="npu")
    a = torch.tensor(
        [[[30.0] * NUM_V_HEADS, [0.0] * NUM_V_HEADS]], device="npu"
    ).squeeze(0)

    got = gdn_gating(A_log, a, dt_bias)
    # Above the threshold softplus(x) == x, so the 30.0 row is exactly linear.
    expected_high = -torch.exp(A_log) * (30.0 + dt_bias)
    expected_low = -torch.exp(A_log) * torch.log1p(torch.exp(dt_bias))
    _assert_close(got[0], expected_high, "clamped softplus")
    _assert_close(got[1], expected_low, "small softplus")


def test_gating_matches_float64_softplus():
    torch.manual_seed(97)
    A_log = torch.randn(NUM_V_HEADS, dtype=torch.float32, device="npu") * 0.5
    dt_bias = torch.randn(NUM_V_HEADS, dtype=torch.float32, device="npu")
    a = torch.randn(3, NUM_V_HEADS, dtype=torch.float32, device="npu")

    got = gdn_gating(A_log, a, dt_bias)
    x = a.double() + dt_bias.double()
    expected = -torch.exp(A_log.double()) * torch.log1p(torch.exp(x))
    assert torch.allclose(
        got.double(), expected, rtol=1e-6, atol=1e-6
    ), "gating differs from fp64 softplus"


def test_qk_l2norm_is_applied_when_requested():
    """The scan normalizes Q/K itself, so an already-normalized input is unchanged."""
    batch, length, seed = 1, 4, 101
    tensors = _inputs(batch, length, seed)
    initial = tensors["pool"][:batch].clone()
    cumulative = torch.tensor([0, length], dtype=torch.int32, device="npu")

    args = (
        tensors["q"].unsqueeze(0),
        tensors["k"].unsqueeze(0),
        tensors["v"].unsqueeze(0),
        tensors["g"].unsqueeze(0),
        tensors["beta"].unsqueeze(0),
    )
    plain, _ = gdn_chunk_prefill(
        *args,
        scale=None,
        initial_state=initial.clone(),
        cu_seqlens=cumulative,
        qk_l2norm=False,
    )
    normalized, _ = gdn_chunk_prefill(
        *args,
        scale=None,
        initial_state=initial.clone(),
        cu_seqlens=cumulative,
        qk_l2norm=True,
    )
    # Inputs are constructed normalized, so re-normalizing changes nothing.
    _assert_close(
        normalized[0].reshape(length, NUM_V_HEADS, HEAD_DIM),
        plain[0].reshape(length, NUM_V_HEADS, HEAD_DIM).float(),
        "double normalization",
    )
