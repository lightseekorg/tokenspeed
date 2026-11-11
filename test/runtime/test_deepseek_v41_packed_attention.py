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

"""Packed V4.1 decode metadata and target attention against uniform windows."""

from test.runtime.test_deepseek_v41_cache import _tables, _verify_backend

import pytest
import torch
from tokenspeed_kernel.ops.attention import dsv41

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA/ROCm")
    return torch.device(request.param)


def test_packed_decode_rows_preserve_reserved_window_origin(device):
    seq = torch.tensor([12, 24], dtype=torch.int32, device=device)
    pools = torch.tensor([19, 7], dtype=torch.int64, device=device)
    indptr = torch.tensor([0, 1, 4, 4], dtype=torch.int64, device=device)
    positions = torch.empty(6, dtype=torch.int64, device=device)
    requests = torch.empty_like(positions)
    lengths = torch.empty(3, dtype=torch.int32, device=device)
    pool_rows = torch.empty(3, dtype=torch.int64, device=device)
    dsv41.decode_rows(
        seq,
        pools,
        positions,
        requests,
        lengths,
        pool_rows,
        2,
        0,
        4,
        query_start_loc=indptr,
    )
    assert positions.tolist() == [8, 20, 21, 22, -1, -1]
    assert requests.tolist() == [0, 1, 1, 1, -1, -1]
    assert lengths.tolist() == [9, 23, 0]
    assert pool_rows.tolist() == [19, 7, -1]


def test_packed_backend_addresses_match_uniform_prefixes(device):
    backend = _verify_backend(device, 2, 4)
    tables = _tables(device)
    seq = torch.tensor([12, 24], dtype=torch.int32, device=device)
    pools = torch.tensor([19, 7], dtype=torch.int64, device=device)
    common = dict(
        forward_mode=ForwardMode.DECODE, block_tables=tables, for_graph_replay=False
    )
    backend.refresh_decode_metadata(2, 2, pools, seq, **common)
    uniform = backend.query_metadata(ForwardMode.DECODE)
    indices = torch.tensor([0, 4, 5, 6], device=device)
    expected = [
        t[indices].clone()
        for t in (
            uniform.positions,
            uniform.request_indices,
            uniform.swa_write_slots,
            uniform.swa_read_slots,
            uniform.swa_read_lens,
        )
    ]
    backend.refresh_decode_metadata(
        2,
        2,
        pools,
        seq,
        num_tokens=6,
        decode_query_indptr=torch.tensor([0, 1, 4], device=device),
        **common,
    )
    packed = backend.query_metadata(ForwardMode.DECODE)
    for actual, reference in zip(
        (
            packed.positions,
            packed.request_indices,
            packed.swa_write_slots,
            packed.swa_read_slots,
            packed.swa_read_lens,
        ),
        expected,
        strict=True,
    ):
        torch.testing.assert_close(actual[:4], reference, rtol=0, atol=0)
    assert packed.positions[4:].tolist() == [-1, -1]
    assert packed.swa_write_slots[4:].tolist() == [-1, -1]
    assert packed.swa_read_lens[4:].tolist() == [0, 0]
    assert packed.seq_lens.tolist() == [9, 23]
    assert not packed.compressor.active[4:].any()
    # Odd compressed positions only refer to the preceding row of this request.
    active = packed.compressor.active
    for row in torch.where(active)[0].tolist():
        previous = int(packed.compressor.previous[row])
        if previous >= 0:
            assert packed.request_indices[previous] == packed.request_indices[row]
            assert packed.positions[previous] + 1 == packed.positions[row]


def test_packed_metadata_refresh_replays_changed_boundaries():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA/ROCm")
    backend = _verify_backend("cuda", 2, 4)
    tables = _tables("cuda")
    seq = torch.tensor([12, 24], dtype=torch.int32, device="cuda")
    pools = torch.tensor([19, 7], device="cuda")
    indptr = torch.tensor([0, 1, 4], device="cuda")

    def prepare():
        backend.refresh_decode_metadata(
            2,
            2,
            pools,
            seq,
            num_tokens=6,
            decode_query_indptr=indptr,
            forward_mode=ForwardMode.DECODE,
            block_tables=tables,
            for_graph_replay=True,
        )

    prepare()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        prepare()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        prepare()
    meta = backend.query_metadata(ForwardMode.DECODE)
    indptr.copy_(torch.tensor([0, 3, 4], device="cuda"))
    graph.replay()
    assert meta.positions.tolist() == [8, 9, 10, 20, -1, -1]
    assert meta.request_indices.tolist() == [0, 0, 0, 1, -1, -1]
    assert meta.seq_lens.tolist() == [11, 21]
    graph.reset()


def test_target_attention_on_packed_rows_matches_uniform():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA/ROCm")
    torch.manual_seed(37)
    backend = _verify_backend("cuda", 2, 4)
    tables = _tables("cuda")
    seq = torch.tensor([12, 24], dtype=torch.int32, device="cuda")
    pools = torch.tensor([19, 7], device="cuda")
    q = torch.randn(8, 2, 512, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(8, 512, dtype=torch.bfloat16, device="cuda")
    iq = torch.randn(8, 2, 128, dtype=torch.bfloat16, device="cuda")
    iw = torch.rand(8, 2, dtype=torch.bfloat16, device="cuda")
    sink = torch.zeros(2, device="cuda")

    def run(indices, indptr):
        backend.cache_pool.arena.buffer.zero_()
        backend.refresh_decode_metadata(
            2,
            2,
            pools,
            seq,
            num_tokens=indices.numel(),
            decode_query_indptr=indptr,
            forward_mode=ForwardMode.DECODE,
            block_tables=tables,
            for_graph_replay=False,
        )
        meta = backend.query_metadata(ForwardMode.DECODE)
        selected_kv = kv[indices]
        backend.write_global(
            20,
            selected_kv,
            iq[indices, 0],
            meta.positions,
            meta.request_indices,
            ForwardMode.DECODE,
            main_rope_cache=None,
        )
        return backend.forward_v41(
            q[indices],
            selected_kv,
            layer_id=20,
            positions=meta.positions,
            request_indices=meta.request_indices,
            forward_mode=ForwardMode.DECODE,
            index_q=iq[indices],
            index_weights=iw[indices],
            attn_sink=sink,
            softmax_scale=512**-0.5,
            index_process_group=None,
            swa_rope_cache=None,
            swa_norm_weight=None,
            swa_norm_eps=0.0,
            key_rows=None,
            fused_decode_groups=None,
        ).clone()

    dense = run(torch.arange(8, device="cuda"), None)
    indices = torch.tensor([0, 4, 5, 6], device="cuda")
    packed = run(indices, torch.tensor([0, 1, 4], device="cuda"))
    torch.testing.assert_close(packed, dense[indices], rtol=0, atol=0)


def test_packed_anchors_keep_fixed_output_stride(device):
    width = 4
    # One complete prefill, one incomplete prefill, then three decode requests.
    tokens = torch.arange(100, 113, dtype=torch.int32, device=device)
    accepted = torch.tensor([1, 0, 1, 3, 2], dtype=torch.int32, device=device)
    positions = torch.tensor([10, 20, 21, 22, 30, 31, -1, -1], device=device)
    indptr = torch.tensor([0, 1, 4, 6], device=device)
    proposals = torch.full((5, 6), -7, dtype=torch.int32, device=device)
    starts = torch.empty(3, dtype=torch.int64, device=device)
    dsv41.dspark_anchors(
        tokens,
        accepted,
        positions,
        2,
        width,
        proposals,
        starts,
        num_prefill_outputs=1,
        decode_query_indptr=indptr,
    )
    assert proposals.tolist() == [[100] * 6, [-7] * 6, [101] * 6, [107] * 6, [110] * 6]
    assert starts.tolist() == [10, 22, 31]


def test_packed_anchors_replay_with_new_lengths():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA/ROCm")
    device = torch.device("cuda")
    indptr = torch.tensor([0, 4, 5], device=device)
    positions = torch.tensor([10, 11, 12, 13, 20, -1, -1, -1], device=device)
    accepted = torch.ones(2, dtype=torch.int32, device=device)
    tokens = torch.arange(8, dtype=torch.int32, device=device)
    proposals = torch.empty((2, 4), dtype=torch.int32, device=device)
    starts = torch.empty(2, dtype=torch.int64, device=device)

    def run():
        dsv41.dspark_anchors(
            tokens,
            accepted,
            positions,
            0,
            4,
            proposals,
            starts,
            num_prefill_outputs=0,
            decode_query_indptr=indptr,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for offsets, packed, accepts, expected_starts in [
        ([0, 4, 5], [10, 11, 12, 13, 20], [4, 1], [13, 20]),
        ([0, 1, 5], [10, 20, 21, 22, 23], [1, 3], [10, 22]),
        ([0, 4, 5], [10, 11, 12, 13, 20], [2, 1], [11, 20]),
    ]:
        indptr.copy_(torch.tensor(offsets, device=device))
        positions[:5].copy_(torch.tensor(packed, device=device))
        accepted.copy_(torch.tensor(accepts, dtype=torch.int32, device=device))
        graph.replay()
        assert starts.tolist() == expected_starts
        expected_bonus = tokens[
            torch.tensor(accepts, device=device)
            - 1
            + torch.arange(2, device=device) * 4
        ]
        torch.testing.assert_close(
            proposals, expected_bonus[:, None].expand_as(proposals)
        )
    graph.reset()
