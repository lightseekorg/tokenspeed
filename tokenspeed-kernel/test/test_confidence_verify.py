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

"""Packed verify kernels retain causal positions and accepted state rows."""

import pytest
import torch
from tokenspeed_kernel.ops.attention.gdn import gdn_decode_mtp
from tokenspeed_kernel.ops.attention.gdn.triton import _fused_gdn_decode_update_kernel
from tokenspeed_kernel.ops.attention.qsa import qsa_sparse_attention
from tokenspeed_kernel.ops.attention.qsa.triton import (
    _qwen4_exp_qsa_compress_and_store_kernel,
    _qwen4_exp_qsa_prepare_metadata_kernel,
    qwen4_exp_qsa_compress_and_store,
    qwen4_exp_qsa_prepare_metadata,
)
from tokenspeed_kernel.ops.conv import causal_conv1d_update
from tokenspeed_kernel.ops.conv._triton.causal_conv1d import (
    _causal_conv1d_update_kernel,
)
from tokenspeed_kernel.ops.ple import ple_conv_sequences, ple_ngram_ids
from utils import assert_no_triton_compile

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")


def test_qsa_device_offsets_replay_positions_and_filler():
    lengths = torch.tensor([3, 1, 0], dtype=torch.int32, device="cuda")
    offsets = torch.tensor([0, 3, 4, 4], dtype=torch.int32, device="cuda")
    prefixes = torch.tensor([12, 25, 0], dtype=torch.int32, device="cuda")
    ends = torch.empty_like(prefixes)
    pages = torch.tensor([[1], [2], [0]], dtype=torch.int32, device="cuda")

    def prepare():
        ends.copy_(prefixes + lengths)
        return qwen4_exp_qsa_prepare_metadata(
            ends,
            lengths,
            6,
            pages,
            64,
            pages,
            64,
            4,
            enable_pdl=False,
            query_start_loc=offsets,
        )

    prepare()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        logical, requests, compressed, recent, complete = prepare()
    graph.replay()
    assert logical.tolist() == [12, 13, 14, 25, -1, -1]
    assert requests.tolist() == [0, 0, 0, 1, 0, 0]
    assert recent.tolist() == [76, 77, 78, 153, 0, 0]
    assert compressed.tolist() == recent.tolist()
    assert complete.tolist() == [3, 3, 3, 6, 0, 0]
    lengths.copy_(torch.tensor([1, 4, 1], device="cuda"))
    offsets.copy_(torch.tensor([0, 1, 5, 6], device="cuda"))
    graph.replay()
    assert logical.tolist() == [12, 25, 26, 27, 28, 0]
    assert requests.tolist() == [0, 1, 1, 1, 1, 2]
    assert recent[-1].item() == 0


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("replay", [False, True])
def test_gdn_packed_prefixes_match_uniform_verify(state_dtype, replay):
    torch.manual_seed(12)
    batch, width, heads, value_heads, dim = 3, 4, 2, 4, 32
    q = torch.randn(batch, width, heads, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(batch, width, value_heads, dim, device="cuda", dtype=q.dtype)
    a = torch.randn(batch, width, value_heads, device="cuda", dtype=q.dtype)
    b = torch.randn_like(a)
    state = torch.randn(
        batch + 1, value_heads, dim, dim, device="cuda", dtype=state_dtype
    )
    initial = state.clone()
    read = torch.arange(1, batch + 1, dtype=torch.int32, device="cuda")
    destinations = torch.arange(batch * width, dtype=torch.int32, device="cuda").view(
        batch, width
    )
    kwargs = {
        "A_log": torch.zeros(value_heads, device="cuda"),
        "a": a,
        "dt_bias": torch.zeros(value_heads, device="cuda"),
        "b": b,
        "initial_state": state,
        "initial_state_indices": read,
        "tree_ancestors": None,
        "solution": "triton",
        "disable_state_update": False,
        "output_state_indices": destinations,
    }
    full_states = torch.empty(
        batch * width, value_heads, dim, dim, device="cuda", dtype=state_dtype
    )
    expected = gdn_decode_mtp(q, k, v, output_state=full_states, **kwargs)
    offsets = torch.tensor([0, 3, 4, 4], device="cuda", dtype=torch.int32)
    selected = torch.tensor([0, 1, 2, 4, 0, 0], device="cuda")
    packed_q, packed_k, packed_v = [x.flatten(0, 1)[selected][None] for x in (q, k, v)]
    kwargs.update(a=a.flatten(0, 1)[selected][None], b=b.flatten(0, 1)[selected][None])
    packed_states = torch.full_like(full_states, float("nan"))
    payload = (
        torch.full(
            (batch * width, heads * dim + value_heads * dim + 2 * value_heads),
            float("nan"),
            dtype=q.dtype,
            device="cuda",
        )
        if replay
        else None
    )
    kwargs.update(
        disable_state_update=replay,
        output_state_indices=None if replay else destinations,
    )

    def run():
        return gdn_decode_mtp(
            packed_q,
            packed_k,
            packed_v,
            output_state=None if replay else packed_states,
            query_start_loc=offsets,
            max_seqlen_q=width,
            replay_payload=payload,
            **kwargs,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    graph.replay()
    torch.testing.assert_close(
        actual[0, :4], expected.flatten(0, 1)[selected[:4]], atol=0, rtol=0
    )
    assert torch.count_nonzero(actual[0, 4:]).item() == 0
    if replay:
        raw = torch.cat((k.flatten(2), v.flatten(2), a, b), dim=-1).flatten(0, 1)
        torch.testing.assert_close(
            payload[selected[:4]], raw[selected[:4]], atol=0, rtol=0
        )
        assert payload[3].isnan().all()
    else:
        torch.testing.assert_close(
            packed_states[selected[:4]], full_states[selected[:4]], atol=0, rtol=0
        )
        assert packed_states[3].isnan().all()
    torch.testing.assert_close(state, initial, atol=0, rtol=0)
    offsets.copy_(torch.tensor([0, 1, 3, 4], device="cuda"))
    graph.replay()
    # The first token of the newly active third request reads its own state.
    want = gdn_decode_mtp(
        packed_q[:, 3:4],
        packed_k[:, 3:4],
        packed_v[:, 3:4],
        A_log=kwargs["A_log"],
        a=kwargs["a"][:, 3:4],
        dt_bias=kwargs["dt_bias"],
        b=kwargs["b"][:, 3:4],
        initial_state=state,
        initial_state_indices=read[2:],
        tree_ancestors=None,
        solution="triton",
        disable_state_update=True,
    )
    torch.testing.assert_close(actual[:, 3:4], want, atol=0, rtol=0)


@pytest.mark.parametrize("separate_output", [False, True])
@pytest.mark.parametrize("save_replay", [False, True])
@pytest.mark.parametrize("filler_rows", [0, 2])
def test_packed_conv_stages_only_each_requests_prefix(
    separate_output, save_replay, filler_rows
):
    torch.manual_seed(25)
    batch, width, channels = 3, 4, 64
    original = torch.randn(batch, width, channels, dtype=torch.bfloat16, device="cuda")
    weights = torch.randn(channels, 4, dtype=original.dtype, device="cuda")
    seed = torch.randn(20, channels, 3, dtype=original.dtype, device="cuda")
    read = torch.tensor([1, 2, 3], dtype=torch.int32, device="cuda")
    destinations = torch.arange(4, 16, dtype=torch.int32, device="cuda").view(
        batch, width
    )
    full_state = seed.clone()
    expected = causal_conv1d_update(
        original.clone().transpose(1, 2),
        full_state,
        weights,
        activation="silu",
        conv_state_indices=read,
        output_state_indices=destinations,
        parent_indices=None,
    ).transpose(1, 2)
    selected = torch.tensor([0, 1, 2, 4] + [0] * filler_rows, device="cuda")
    packed = original.flatten(0, 1)[selected].clone()
    raw_packed = packed.clone()
    packed_state = seed.clone()
    offsets = torch.tensor([0, 3, 4, 4], dtype=torch.int32, device="cuda")
    output = torch.empty_like(packed) if separate_output else None
    heads, first_channel = 4, 24
    payload = torch.full(
        (batch * width, channels - first_channel + 2 * heads),
        float("nan"),
        dtype=original.dtype,
        device="cuda",
    )
    gates = tuple(
        torch.randn(selected.numel(), heads, dtype=original.dtype, device="cuda")
        for _ in range(2)
    )
    actual = causal_conv1d_update(
        packed,
        packed_state,
        weights,
        activation="silu",
        conv_state_indices=read,
        output_state_indices=destinations,
        parent_indices=None,
        query_start_loc=offsets,
        out=output,
        replay_payload=(payload, *gates) if save_replay else None,
    )
    torch.testing.assert_close(
        actual[:4], expected.flatten(0, 1)[selected[:4]], atol=0, rtol=0
    )
    torch.testing.assert_close(
        packed_state[[4, 5, 6, 8]], full_state[[4, 5, 6, 8]], atol=0, rtol=0
    )
    torch.testing.assert_close(
        packed_state[[7, 9, 10, 11, 12, 13, 14, 15]],
        seed[[7, 9, 10, 11, 12, 13, 14, 15]],
        atol=0,
        rtol=0,
    )
    if separate_output:
        torch.testing.assert_close(packed, raw_packed, atol=0, rtol=0)
    if save_replay:
        expected_rows = torch.cat(
            (
                expected.flatten(0, 1)[selected[:4], first_channel:],
                *(g[:4] for g in gates),
            ),
            dim=-1,
        )
        torch.testing.assert_close(payload[[0, 1, 2, 4]], expected_rows, atol=0, rtol=0)
        assert payload[[3, 5, 6, 7, 8, 9, 10, 11]].isnan().all()


def test_ple_packed_graph_replay_preserves_context_and_conv_windows():
    torch.manual_seed(31)
    device = "cuda"
    batch, width, rows, channels, state_len = 3, 4, 6, 32, 9
    full_ids = torch.randint(1, 20, (batch * width,), device=device)
    context = torch.randint(1, 20, (batch, 2), device=device)
    full_values = torch.randn(
        batch * width, channels, device=device, dtype=torch.bfloat16
    )
    initial_conv = torch.randn(
        batch, channels, state_len, device=device, dtype=full_values.dtype
    )
    weights = torch.randn(channels, 4, device=device, dtype=full_values.dtype)
    multipliers = torch.tensor([7, 17, 29], device=device)
    sizes = torch.tensor([101, 103], device=device)
    bases = torch.tensor([0, 101], device=device)
    full_req = torch.arange(batch, device=device).repeat_interleave(width)
    full_col = torch.arange(width, device=device).repeat(batch)
    full_lengths = torch.full((batch,), width, device=device)
    full_starts = torch.arange(batch, device=device) * width
    hash_kwargs = {
        "ngram_size": 3,
        "heads_per_ngram": 1,
        "eos_token_id": 0,
        "uniform_length": 0,
        "mod_reciprocals": None,
    }
    expected_ids, expected_tails = ple_ngram_ids(
        full_ids,
        context,
        full_req,
        full_col,
        full_lengths,
        full_starts,
        multipliers,
        sizes,
        bases,
        need_tail=True,
        **hash_kwargs,
    )
    conv_kwargs = {
        "dilation": 3,
        "kernel_size": 4,
        "state_len": state_len,
        "write_final": False,
        "weights_independent": True,
        "scatter_windows": True,
        "windows_block_rows": width + 1,
        "batch_size": batch,
    }
    expected_windows = torch.full(
        (batch * (width + 1), channels, state_len),
        float("nan"),
        device=device,
        dtype=full_values.dtype,
    )
    expected_values, _, _ = ple_conv_sequences(
        full_values,
        initial_conv,
        weights,
        full_req,
        full_col,
        full_lengths,
        full_starts,
        total_tokens=batch * width,
        windows=expected_windows,
        **conv_kwargs,
    )
    req = torch.tensor([0, 0, 0, 1, 2, 2], device=device)
    col = torch.tensor([0, 1, 2, 0, -1, -1], device=device)
    lengths = torch.tensor([3, 1, 0], device=device)
    starts = torch.tensor([0, 3, 4], device=device)
    selected = torch.tensor([0, 1, 2, 4, 0, 0], device=device)
    packed_ids = full_ids[selected].clone()
    packed_values = full_values[selected].clone()
    context_scratch = torch.full(
        (batch * (width + 1), 2), -9, device=device, dtype=context.dtype
    )
    conv_scratch = torch.full_like(expected_windows, float("nan"))

    def run():
        ids, _ = ple_ngram_ids(
            packed_ids,
            context,
            req,
            col,
            lengths,
            starts,
            multipliers,
            sizes,
            bases,
            tail_out=context_scratch,
            tail_block_rows=width + 1,
            **hash_kwargs,
        )
        output, _, _ = ple_conv_sequences(
            packed_values,
            initial_conv,
            weights,
            req,
            col,
            lengths,
            starts,
            total_tokens=rows,
            windows=conv_scratch,
            **conv_kwargs,
        )
        return ids, output

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual_ids, actual_values = run()

    def check(selected_rows):
        context_scratch.fill_(-9)
        conv_scratch.fill_(float("nan"))
        graph.replay()
        torch.testing.assert_close(
            actual_ids[:4], expected_ids[selected_rows], atol=0, rtol=0
        )
        torch.testing.assert_close(
            actual_values[:4], expected_values[selected_rows], atol=0, rtol=0
        )
        assert actual_values[4:].count_nonzero().item() == 0
        torch.testing.assert_close(actual_ids[4:], bases.expand(2, -1), atol=0, rtol=0)
        seed_rows = torch.arange(batch, device=device) * (width + 1)
        torch.testing.assert_close(context_scratch[seed_rows], context, atol=0, rtol=0)
        torch.testing.assert_close(
            conv_scratch[seed_rows], initial_conv, atol=0, rtol=0
        )
        destinations = req[:4] * (width + 1) + col[:4] + 1
        torch.testing.assert_close(
            context_scratch[destinations], expected_tails[selected_rows], atol=0, rtol=0
        )
        torch.testing.assert_close(
            conv_scratch[destinations], expected_windows[destinations], atol=0, rtol=0
        )

    check(selected[:4])
    req.copy_(torch.tensor([0, 1, 2, 2, 2, 2], device=device))
    col.copy_(torch.tensor([0, 0, 0, 1, -1, -1], device=device))
    lengths.copy_(torch.tensor([1, 1, 2], device=device))
    starts.copy_(torch.tensor([0, 1, 2], device=device))
    selected.copy_(torch.tensor([0, 4, 8, 9, 0, 0], device=device))
    packed_ids.copy_(full_ids[selected])
    packed_values.copy_(full_values[selected])
    check(selected[:4])


def test_qsa_compact_rows_stage_in_original_verify_windows():
    device = "cuda"
    dim, width, batch = 8, 4, 3
    token_k = torch.randn(6, 1, dim, dtype=torch.bfloat16, device=device)
    logical = torch.tensor([8, 9, 10, 20, -1, -1], device=device)
    requests = torch.tensor([0, 0, 0, 1, 0, 0], device=device)
    positions = logical.clamp_min(0)[:, None].expand(-1, 3).contiguous()
    slots = torch.zeros(6, dtype=torch.int32, device=device)
    raw = torch.zeros(4, 4, 1, dim, dtype=token_k.dtype, device=device)
    raw_positions = torch.zeros(4, 3, dtype=torch.int64, device=device)
    compressed = torch.zeros(4, 16, 1, dim, dtype=token_k.dtype, device=device)
    rotary = torch.cat(
        (
            torch.ones(64, dim // 2, device=device),
            torch.zeros(64, dim // 2, device=device),
        ),
        dim=1,
    )
    staged_k = torch.full(
        (batch, width, 1, dim), float("nan"), dtype=token_k.dtype, device=device
    )
    staged_positions = torch.full(
        (batch, width, 3), -99, dtype=torch.int64, device=device
    )
    staged_logical = torch.full((batch, width), -99, dtype=torch.int64, device=device)
    staged_recent = torch.full((batch, width), -99, dtype=torch.int32, device=device)
    row_map = torch.tensor([0, 1, 2, 4, -1, -1], device=device)
    qwen4_exp_qsa_compress_and_store(
        token_k,
        logical,
        requests,
        slots,
        raw,
        positions,
        raw_positions,
        torch.ones(dim, device=device),
        1e-6,
        rotary,
        slots,
        compressed,
        4,
        4,
        64,
        stage_verify_buffers=(
            staged_k,
            staged_positions,
            staged_logical,
            staged_recent,
        ),
        stage_draft=False,
        stage_verify_rows=row_map,
    )
    rows = row_map[:4]
    torch.testing.assert_close(
        staged_k.flatten(0, 1)[rows], token_k[:4], atol=0, rtol=0
    )
    torch.testing.assert_close(
        staged_logical.flatten()[rows], logical[:4], atol=0, rtol=0
    )
    torch.testing.assert_close(
        staged_positions.flatten(0, 1)[rows], positions[:4], atol=0, rtol=0
    )
    assert staged_k[0, 3].isnan().all()
    assert staged_logical[2].eq(-99).all()
    with assert_no_triton_compile(_qwen4_exp_qsa_compress_and_store_kernel):
        for count in (3, 5, 6):
            qwen4_exp_qsa_compress_and_store(
                token_k[:count],
                logical[:count],
                requests[:count],
                slots[:count],
                raw,
                positions[:count],
                raw_positions,
                torch.ones(dim, device=device),
                1e-6,
                rotary,
                slots[:count],
                compressed,
                4,
                4,
                64,
                stage_verify_buffers=(
                    staged_k,
                    staged_positions,
                    staged_logical,
                    staged_recent,
                ),
                stage_draft=False,
                stage_verify_rows=row_map[:count],
            )


def test_qsa_compact_attention_matches_full_window_queries():
    torch.manual_seed(43)
    device = "cuda"
    q = torch.randn(16, 6, 256, dtype=torch.bfloat16, device=device)
    k = torch.randn(128, 1, 256, dtype=q.dtype, device=device)
    v = torch.randn_like(k)
    k[0].fill_(float("nan"))
    v[0].fill_(float("nan"))
    slots = torch.zeros(16, 2051, dtype=torch.int32, device=device)
    for row in range(16):
        count = row % 8 + 2
        slots[row, :count] = torch.arange(2, count + 2, device=device)
    kwargs = {
        "scale": 256**-0.5,
        "metadata_capacity_rows": 16,
        "k_scale": None,
        "v_scale": None,
        "override": None,
        "solution": None,
        "sparse_metadata": None,
    }
    full = qsa_sparse_attention(q, k, v, slots, max_seqlen_q=8, **kwargs)
    selected = torch.tensor([0, 1, 2, 3, 4, 8], device=device)
    packed_q = q[selected].clone()
    packed_slots = slots[selected].clone()
    qsa_sparse_attention(packed_q, k, v, packed_slots, max_seqlen_q=1, **kwargs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compact = qsa_sparse_attention(
            packed_q, k, v, packed_slots, max_seqlen_q=1, **kwargs
        )
    graph.replay()
    torch.testing.assert_close(compact, full[selected], atol=1e-2, rtol=1e-2)
    for i, row in enumerate(selected.tolist()):
        keys = k[slots[row, : row % 8 + 2], 0].float()
        values = v[slots[row, : row % 8 + 2], 0].float()
        reference = ((q[row].float() @ keys.T * 256**-0.5).softmax(-1) @ values).to(
            q.dtype
        )
        torch.testing.assert_close(compact[i], reference, atol=1e-2, rtol=1e-2)


def test_packed_verify_does_not_specialize_on_request_or_query_counts():
    device = "cuda"
    state = torch.randn(8, 4, 32, 32, dtype=torch.float32, device=device)
    staged = torch.empty(40, 4, 32, 32, dtype=state.dtype, device=device)
    conv_state = torch.randn(70, 64, 3, dtype=torch.bfloat16, device=device)
    weights = torch.randn(64, 4, dtype=conv_state.dtype, device=device)

    def run(batch, rows, columns):
        offsets = torch.arange(batch + 1, dtype=torch.int32, device=device)
        read = torch.arange(batch, dtype=torch.int32, device=device)
        destinations = torch.arange(batch * 4, dtype=torch.int32, device=device).view(
            batch, 4
        )
        q = torch.randn(1, rows, 2, 32, dtype=torch.bfloat16, device=device)
        v = torch.randn(1, rows, 4, 32, dtype=q.dtype, device=device)
        a = torch.randn(1, rows, 4, dtype=q.dtype, device=device)
        gdn_decode_mtp(
            q,
            q,
            v,
            A_log=torch.zeros(4, device=device),
            a=a,
            dt_bias=torch.zeros(4, device=device),
            b=a,
            initial_state=state,
            initial_state_indices=read,
            output_state_indices=destinations,
            output_state=staged,
            disable_state_update=False,
            tree_ancestors=None,
            query_start_loc=offsets,
            max_seqlen_q=4,
        )
        x = torch.randn(rows, 64, dtype=q.dtype, device=device)
        causal_conv1d_update(
            x,
            conv_state,
            weights,
            activation="silu",
            conv_state_indices=read,
            output_state_indices=destinations,
            parent_indices=None,
            query_start_loc=offsets,
        )
        pages = torch.ones(batch, columns, dtype=torch.int32, device=device)
        qwen4_exp_qsa_prepare_metadata(
            torch.full((batch,), 13, dtype=torch.int32, device=device),
            torch.ones(batch, dtype=torch.int32, device=device),
            rows,
            pages,
            64,
            pages,
            64,
            4,
            enable_pdl=False,
            query_start_loc=offsets,
        )

    run(4, 16, 4)
    with assert_no_triton_compile(
        _fused_gdn_decode_update_kernel,
        _causal_conv1d_update_kernel,
        _qwen4_exp_qsa_prepare_metadata_kernel,
    ):
        for batch, rows, columns in ((1, 9, 7), (3, 11, 16), (5, 21, 32)):
            run(batch, rows, columns)
