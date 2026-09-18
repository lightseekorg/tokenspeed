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

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.attention._triton.prefill_state_checkpoints import (
    merge_prefill_checkpoint_outputs,
    pack_prefill_recurrent_checkpoint_inputs,
    write_prefill_conv_checkpoints,
    write_prefill_recurrent_checkpoints,
)


def _device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Triton checkpoint kernels")
    return torch.device("cuda")


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_inactive_conv_checkpoint_destinations_do_not_access_state(device_name):
    device = _device() if device_name == "cuda" else torch.device("cpu")
    raw = torch.arange(24, dtype=torch.float32, device=device).reshape(8, 3)
    pool = torch.randn(5, 3, 3, device=device)
    before = pool.clone()
    rows = torch.tensor([0, 1], device=device)
    blocks = torch.tensor([-1, 3], device=device)
    write_prefill_conv_checkpoints(
        raw,
        pool,
        torch.tensor([1, 2], device=device),
        torch.tensor([2, 4], device=device),
        blocks,
        rows,
        torch.tensor([1000000, 4], device=device),
        torch.tensor([1000000, 4], device=device),
    )
    before[3] = raw[5:8].T
    torch.testing.assert_close(pool, before, rtol=0, atol=0)


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_inactive_recurrent_checkpoint_rows_preserve_destination(device_name):
    device = _device() if device_name == "cuda" else torch.device("cpu")
    state = torch.randn(3, 2, 4, 4, device=device).transpose(-1, -2)
    state[0].fill_(float("nan"))
    state[2].fill_(float("nan"))
    pool = torch.randn(4, 2, 4, 4, device=device).transpose(-1, -2)
    before = pool.clone()
    # Slot 0 has no destination row; slot 2 has a row but no checkpoint page.
    # Destination zero remains valid for merging tail results into body states.
    write_prefill_recurrent_checkpoints(
        state,
        pool,
        torch.tensor([0, 2, -1], device=device),
        torch.tensor([-1, 0, 2], device=device),
    )
    before[0] = state[1]
    torch.testing.assert_close(pool, before, rtol=0, atol=0)


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
@pytest.mark.parametrize("token_dim", [0, 1])
def test_padded_checkpoint_pack_and_output_merge(device_name, token_dim):
    device = _device() if device_name == "cuda" else torch.device("cpu")
    # Noncontiguous token rows and state features match projection/scan views.
    q = torch.arange(7 * 3 * 2 * 4, dtype=torch.float32, device=device).view(
        1, 7, 3, 2, 4
    )[:, :, 1]
    state = (
        torch.arange(3 * 2 * 4 * 4, dtype=torch.float32, device=device)
        .view(3, 2, 4, 4)
        .transpose(-1, -2)
    )
    rows = torch.tensor([2, 0], device=device)
    body_indices = torch.tensor([0, 1, 4, -1, -1, -1, -1, -1], device=device)
    tail_indices = torch.tensor([2, 3, 5, 6, -1], device=device)
    gate = q.squeeze(0)

    def pack(indices):
        return pack_prefill_recurrent_checkpoint_inputs(
            q, q, q, state, rows, indices, gate, gate, gate, gate, gate
        )

    body, tail = pack(body_indices), pack(tail_indices)
    for name in ("query", "key", "value", "a", "b", "g_raw", "f_a_out", "beta_raw"):
        actual = getattr(body, name)
        if actual.ndim == q.ndim:
            actual = actual.squeeze(0)
        torch.testing.assert_close(actual[:3], gate[[0, 1, 4]], rtol=0, atol=0)
        assert torch.count_nonzero(actual[3:]) == 0
    torch.testing.assert_close(body.recurrent_state, state[rows], rtol=0, atol=0)
    # Exercise feature strides in scan outputs as well as token-axis conventions.
    body_out, tail_out = body.value.transpose(-1, -2), tail.value.transpose(-1, -2)
    expected = q.transpose(-1, -2)
    if token_dim == 0:
        body_out, tail_out, expected = (
            x.squeeze(0) for x in (body_out, tail_out, expected)
        )
    merged = merge_prefill_checkpoint_outputs(
        body_out, tail_out, body_indices, tail_indices, token_dim, 9, None
    )
    torch.testing.assert_close(merged.narrow(token_dim, 0, 7), expected, rtol=0, atol=0)
    assert torch.count_nonzero(merged.narrow(token_dim, 7, 2)) == 0


@pytest.mark.parametrize("token_dim", [0, 1])
@pytest.mark.parametrize("strided", [False, True])
def test_shared_inverse_gather_replay(token_dim, strided):
    device = _device()
    body = torch.randn(8, 3, 4, device=device)
    tail = torch.randn(5, 3, 4, device=device)
    if strided:
        body, tail = body.transpose(-1, -2), tail.transpose(-1, -2)
    if token_dim == 1:
        body, tail = body.unsqueeze(0), tail.unsqueeze(0)
    body_map = torch.empty(8, dtype=torch.int64, device=device)
    tail_map = torch.empty(5, dtype=torch.int64, device=device)
    inverse = torch.full((10,), -1, dtype=torch.int64, device=device)

    def run():
        return merge_prefill_checkpoint_outputs(
            body, tail, body_map, tail_map, token_dim, 10, inverse
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    for live_body, live_tail in [
        ([0, 1, 4], [2, 3, 5, 6]),
        ([5], [0, 2]),
        ([], []),
        ([1, 3, 5, 7, 9], [0, 2, 4, 6, 8]),
    ] * 3:
        body_map.fill_(-1)
        tail_map.fill_(-1)
        inverse.fill_(-1)
        for source, mapping, destinations, start in (
            (body, body_map, live_body, 0),
            (tail, tail_map, live_tail, 8),
        ):
            source.copy_(torch.randn_like(source))
            source.narrow(
                token_dim,
                len(destinations),
                source.shape[token_dim] - len(destinations),
            ).fill_(float("nan"))
            if destinations:
                dest = torch.tensor(destinations, device=device)
                mapping[: len(destinations)].copy_(dest)
                inverse[dest] = torch.arange(len(destinations), device=device) + start
        expected = merge_prefill_checkpoint_outputs(
            body, tail, body_map, tail_map, token_dim, 10, None
        )
        graph.replay()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.fixture
def large_offset_pool():
    device = _device()
    # Keep the stride itself in int32: a larger stride would make Triton
    # promote the product even before the fix. Page 2 crosses INT32_MAX.
    page_stride = 2**30 + 4096
    storage_bytes = 2 * page_stride + 4
    free_bytes, _ = torch.cuda.mem_get_info(device)
    if free_bytes < storage_bytes + 512 * 1024**2:
        pytest.skip("large-offset regression requires 2.5 GiB of free GPU memory")
    storage = torch.empty(storage_bytes, dtype=torch.uint8, device=device)
    # Sparse views require only ~2 GiB and touch a few bytes. Never clone or
    # initialize the entire pool. Without widened IDs the kernels below access
    # a wrapped negative offset, which can poison the GPU context.
    return storage, page_stride


@pytest.mark.parametrize(
    ("source", "continuation", "destination"), [(2, 1, 1), (0, 2, 1), (1, 1, 2)]
)
def test_conv_checkpoint_offsets_beyond_int32(
    large_offset_pool, source, continuation, destination
):
    storage, stride = large_offset_pool
    pool = storage.as_strided((3, 1, 4), (stride, 4, 1))
    pool[0].fill_(99)
    pool[1].copy_(torch.tensor([[11, 12, 13, 14]], device=storage.device))
    pool[2].copy_(torch.tensor([[21, 22, 23, 24]], device=storage.device))
    raw = torch.tensor([[101]], dtype=torch.uint8, device=storage.device)
    expected = torch.cat((pool[source or continuation, :, 1:].clone(), raw), dim=1)
    write_prefill_conv_checkpoints(
        raw,
        pool,
        torch.tensor([source], dtype=torch.int32, device=storage.device),
        torch.tensor([continuation], dtype=torch.int32, device=storage.device),
        torch.tensor([destination], dtype=torch.int32, device=storage.device),
        torch.tensor([0], dtype=torch.int64, device=storage.device),
        torch.tensor([0], dtype=torch.int64, device=storage.device),
        torch.tensor([1], dtype=torch.int64, device=storage.device),
    )
    torch.testing.assert_close(pool[destination], expected, rtol=0, atol=0)
    assert torch.all(pool[0] == 99)


def test_recurrent_checkpoint_offsets_beyond_int32(large_offset_pool):
    storage, stride = large_offset_pool
    pool = storage.as_strided((3, 1, 2, 2), (stride, 4, 2, 1))
    pool[1].fill_(99)
    pool[2].zero_()
    checkpoint = torch.tensor(
        [[[[41, 42], [43, 44]]]], dtype=torch.float32, device=storage.device
    )
    write_prefill_recurrent_checkpoints(
        checkpoint,
        pool,
        torch.tensor([2], dtype=torch.int32, device=storage.device),
        torch.tensor([0], dtype=torch.int64, device=storage.device),
    )
    torch.testing.assert_close(pool[2], checkpoint[0].to(torch.uint8), rtol=0, atol=0)
    assert torch.all(pool[1] == 99)


def _reference_conv(
    raw_inputs: torch.Tensor,
    conv_states: torch.Tensor,
    state_in: torch.Tensor,
    state_out: torch.Tensor,
    destinations: torch.Tensor,
    rows: torch.Tensor,
    starts: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    expected = conv_states.clone()
    old = conv_states.clone()
    state_len = conv_states.shape[-1]
    for selected, request_row in enumerate(rows.cpu().tolist()):
        checkpoint_len = int(lengths[selected])
        start = int(starts[selected])
        source = int(state_in[request_row])
        if source <= 0:
            source = int(state_out[request_row])
        destination = int(destinations[request_row])
        for position in range(state_len):
            relative = checkpoint_len - state_len + position
            if relative >= 0:
                expected[destination, :, position] = raw_inputs[start + relative]
            else:
                expected[destination, :, position] = old[
                    source, :, state_len + relative
                ]
    return expected


@pytest.mark.parametrize(
    ("rows_list", "starts_list", "lengths_list", "alias"),
    [
        ([0], [0], [6], False),
        ([0], [0], [2], False),
        ([0], [0], [2], True),
        ([0, 2], [0, 7], [2, 5], False),
    ],
)
def test_conv_checkpoint_fused_kernel(
    rows_list: list[int],
    starts_list: list[int],
    lengths_list: list[int],
    alias: bool,
) -> None:
    device = _device()
    raw_inputs = torch.arange(60, dtype=torch.float32, device=device).view(15, 4)
    conv_states = torch.arange(10 * 4 * 3, dtype=torch.float32, device=device).view(
        10, 4, 3
    )
    state_in = torch.tensor([1, 2, 3], dtype=torch.int32, device=device)
    state_out = torch.tensor([4, 5, 6], dtype=torch.int32, device=device)
    destinations = torch.tensor(
        [1 if alias else 7, 8, 9], dtype=torch.int32, device=device
    )
    rows = torch.tensor(rows_list, dtype=torch.int64, device=device)
    starts = torch.tensor(starts_list, dtype=torch.int64, device=device)
    lengths = torch.tensor(lengths_list, dtype=torch.int64, device=device)
    expected = _reference_conv(
        raw_inputs,
        conv_states,
        state_in,
        state_out,
        destinations,
        rows,
        starts,
        lengths,
    )
    write_prefill_conv_checkpoints(
        raw_inputs,
        conv_states,
        state_in,
        state_out,
        destinations,
        rows,
        starts,
        lengths,
    )
    torch.testing.assert_close(conv_states, expected, rtol=0, atol=0)


@pytest.mark.parametrize("num_rows", [1, 2])
@pytest.mark.parametrize("row_padding", [0, 7])
@pytest.mark.parametrize("cuda_graph", [False, True])
@pytest.mark.parametrize("transpose_state", [False, True])
def test_recurrent_input_pack_is_one_semantic_path(
    num_rows: int, row_padding: int, cuda_graph: bool, transpose_state: bool
) -> None:
    device = _device()
    tokens = 7
    query = torch.arange(tokens * 2 * 3, dtype=torch.bfloat16, device=device).view(
        1, tokens, 2, 3
    )
    key = query + 100
    value = torch.arange(tokens * 2 * 4, dtype=torch.bfloat16, device=device).view(
        1, tokens, 2, 4
    )
    state = torch.arange(3 * 2 * 4 * 3, dtype=torch.float32, device=device).view(
        3, 2, 4, 3
    )
    if transpose_state:
        # GDN/FLA expose a transposed final-state view in the public layout.
        state = state.transpose(-1, -2).contiguous().transpose(-1, -2)
        assert not state[0].is_contiguous()
    all_rows = torch.tensor([0, 2], dtype=torch.int64, device=device)
    all_indices = torch.tensor([0, 1, 4, 5, 6], dtype=torch.int64, device=device)
    rows = all_rows[:num_rows]
    token_indices = all_indices[: 2 if num_rows == 1 else 5]
    a = torch.arange(tokens * 2, dtype=torch.bfloat16, device=device).view(tokens, 2)
    b = a + 20
    g_raw = torch.arange(tokens * 2 * 3, dtype=torch.bfloat16, device=device).view(
        tokens, 2, 3
    )
    f_a_out = g_raw + 30
    beta_raw = a + 40

    def pad_rows(tensor: torch.Tensor, row_dim: int) -> torch.Tensor:
        if row_padding == 0:
            return tensor
        strides = list(tensor.stride())
        strides[row_dim] += row_padding
        if row_dim == 1:
            strides[0] = tensor.shape[1] * strides[1]
        padded = torch.empty_strided(
            tensor.shape, strides, dtype=tensor.dtype, device=device
        )
        padded.copy_(tensor)
        assert not padded.is_contiguous()
        return padded

    query, key, value = (pad_rows(tensor, 1) for tensor in (query, key, value))
    state, a, b, g_raw, f_a_out, beta_raw = (
        pad_rows(tensor, 0) for tensor in (state, a, b, g_raw, f_a_out, beta_raw)
    )

    packed = pack_prefill_recurrent_checkpoint_inputs(
        query,
        key,
        value,
        state,
        rows,
        token_indices,
        a,
        b,
        g_raw,
        f_a_out,
        beta_raw,
    )
    if cuda_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            packed = pack_prefill_recurrent_checkpoint_inputs(
                query,
                key,
                value,
                state,
                rows,
                token_indices,
                a,
                b,
                g_raw,
                f_a_out,
                beta_raw,
            )
        # Replay must read current data through the captured strided addresses.
        for tensor in (query, key, value, state, a, b, g_raw, f_a_out, beta_raw):
            tensor.add_(1)
        graph.replay()
    torch.testing.assert_close(packed.query, query.index_select(1, token_indices))
    torch.testing.assert_close(packed.key, key.index_select(1, token_indices))
    torch.testing.assert_close(packed.value, value.index_select(1, token_indices))
    torch.testing.assert_close(packed.recurrent_state, state.index_select(0, rows))
    torch.testing.assert_close(packed.a, a.index_select(0, token_indices))
    torch.testing.assert_close(packed.b, b.index_select(0, token_indices))
    torch.testing.assert_close(packed.g_raw, g_raw.index_select(0, token_indices))
    torch.testing.assert_close(packed.f_a_out, f_a_out.index_select(0, token_indices))
    torch.testing.assert_close(packed.beta_raw, beta_raw.index_select(0, token_indices))


@pytest.mark.parametrize("num_rows", [1, 2])
def test_recurrent_state_scatter_supports_noncontiguous_result(num_rows: int) -> None:
    device = _device()
    base = torch.arange(num_rows * 2 * 3 * 4, dtype=torch.float32, device=device).view(
        num_rows, 2, 3, 4
    )
    checkpoint_state = base.transpose(-1, -2)
    assert not checkpoint_state.is_contiguous()
    pool = torch.zeros(8, 2, 4, 3, dtype=torch.float32, device=device)
    rows = torch.tensor([0, 2], dtype=torch.int64, device=device)[:num_rows]
    blocks = torch.tensor([5, -1, 7], dtype=torch.int32, device=device)
    write_prefill_recurrent_checkpoints(checkpoint_state, pool, blocks, rows)
    torch.testing.assert_close(
        pool[blocks.index_select(0, rows).long()], checkpoint_state
    )
