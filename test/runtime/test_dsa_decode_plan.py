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

"""Decode and mixed rounds publish the same graph-stable DSA metadata."""

from test.ci_system.ci_register import register_cuda_ci

import pytest
import torch
from tokenspeed_kernel.ops.attention.dsa import dsa_plan

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.paged.dsa import DSABackend
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.dsa import DSAConfig
from tokenspeed.runtime.utils.env import global_server_args_dict

register_cuda_ci(est_time=10, suite="runtime-1gpu")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA DSA metadata planning required"
)


def _backend(monkeypatch, *, width: int, draft: bool) -> DSABackend:
    monkeypatch.setitem(global_server_args_dict, "numerics", "auto")
    monkeypatch.setitem(global_server_args_dict, "dsa_slot_order", "selection")
    monkeypatch.setitem(global_server_args_dict, "chunked_prefill_size", 2048)
    monkeypatch.setitem(global_server_args_dict, "mla_chunk_multiplier", 1)
    spec = DSAConfig(
        backend_name="dsa",
        num_attention_heads=8,
        num_kv_heads=1,
        head_dim=576,
        attn_tp_size=1,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        kv_cache_dim=576,
        scaling=192**-0.5,
        index_topk=512,
        index_head_dim=128,
        index_n_heads=16,
        index_k_format="fp8_scaled",
    )
    config = AttnConfig(
        device="cuda",
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        kv_cache_quant_method="none",
        prefix_granularity=64,
        context_len=2048,
        max_bs=3,
        components=(spec,),
        speculative_num_draft_tokens=width,
        is_draft=draft,
    )
    backend = DSABackend(config, spec, kernel_page_size=64)
    backend.init_cuda_graph_state(3)
    return backend


def _lengths(*values):
    return torch.tensor(values, dtype=torch.int32, device="cuda")


def _mixed_round(backend, lengths, *, num_extends: int):
    extend = lengths[:num_extends]
    backend.init_forward_metadata(
        len(lengths),
        num_extends,
        lengths,
        torch.zeros((len(lengths), 32), dtype=torch.int32, device="cuda"),
        ForwardMode.MIXED,
        extend_seq_lens=extend,
        extend_seq_lens_cpu=extend.cpu(),
        extend_prefix_lens=torch.zeros_like(extend),
        extend_prefix_lens_cpu=torch.zeros(num_extends, dtype=torch.int32),
        extend_with_prefix=False,
        query_shard=None,
        page_table_cpu=None,
    )


def _expected_plan(lengths, *, width: int, num_extends: int):
    first = num_extends if num_extends < len(lengths) else 0
    rows = lengths[first:].repeat_interleave(width).reshape(-1, 1)
    plan = dsa_plan(seq_lens_2d=rows, page_size=64)
    if not isinstance(plan, torch.Tensor):
        pytest.skip("the selected DSA provider does not use a tensor plan")
    return plan


@pytest.mark.parametrize("width", [1, 2, 6])
def test_decode_graph_reads_the_plan_refreshed_by_mixed_rounds(monkeypatch, width):
    backend = _backend(monkeypatch, width=width, draft=False)
    pages = torch.zeros((3, 32), dtype=torch.int32, device="cuda")
    original = _lengths(900, 950, 1000)
    backend.refresh_decode_metadata(3, 3, original, pages)
    metadata = backend.forward_decode_metadata
    rows, plan = metadata._dsa_seq_lens_2d, metadata._dsa_plan
    if not isinstance(plan, torch.Tensor):
        pytest.skip("the selected DSA provider does not use a tensor plan")
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_rows = rows.clone()
        captured_plan = plan.clone()

    # Exercise shortening and growing lengths before replaying the same graph.
    for lengths, num_extends in (
        (_lengths(100, 120, 130), 2),
        (_lengths(1500, 1600, 1700), 1),
    ):
        _mixed_round(backend, lengths, num_extends=num_extends)
        graph.replay()
        torch.testing.assert_close(
            captured_rows.flatten(), lengths.repeat_interleave(width), rtol=0, atol=0
        )
        torch.testing.assert_close(
            captured_plan,
            _expected_plan(lengths, width=width, num_extends=num_extends),
            rtol=0,
            atol=0,
        )
        assert metadata._dsa_seq_lens_2d is rows
        assert metadata._dsa_plan is plan

    current = _lengths(101, 121, 131)
    backend.refresh_decode_metadata(3, 3, current, pages, for_graph_replay=True)
    graph.replay()
    torch.testing.assert_close(
        captured_rows.flatten(), current.repeat_interleave(width)
    )
    torch.testing.assert_close(
        captured_plan,
        _expected_plan(current, width=width, num_extends=0),
        rtol=0,
        atol=0,
    )
    # First publication must not hold a view into caller-owned seq_lens (k=1).
    torch.testing.assert_close(original, _lengths(900, 950, 1000), rtol=0, atol=0)
    current.fill_(7)
    torch.testing.assert_close(
        rows.flatten(), _lengths(101, 121, 131).repeat_interleave(width)
    )


@pytest.mark.parametrize("width", [1, 2, 6])
def test_first_mixed_draft_refresh_and_mtp_reanchor_share_the_plan(monkeypatch, width):
    backend = _backend(monkeypatch, width=width, draft=True)
    lengths = _lengths(40, 300, 500)
    pages = torch.zeros((3, 32), dtype=torch.int32, device="cuda")
    _mixed_round(backend, lengths, num_extends=1)
    backend.refresh_decode_metadata(3, 3, lengths, pages, num_extends=1)
    metadata = backend.forward_decode_metadata
    rows, plan = metadata._dsa_seq_lens_2d, metadata._dsa_plan
    torch.testing.assert_close(
        plan, _expected_plan(lengths, width=width, num_extends=1), rtol=0, atol=0
    )
    frontier = _lengths(41, 301, 501)
    backend.update_draft_forward_metadata(frontier)
    assert metadata._dsa_seq_lens_2d is rows
    assert metadata._dsa_plan is plan
    torch.testing.assert_close(rows.flatten(), frontier.repeat_interleave(width))
    torch.testing.assert_close(
        plan, _expected_plan(frontier, width=width, num_extends=0), rtol=0, atol=0
    )
    # An all-extend mixed round has no decode consumer but keeps a valid plan.
    backend.refresh_decode_metadata(3, 3, frontier, pages, num_extends=3)
    torch.testing.assert_close(
        plan, _expected_plan(frontier, width=width, num_extends=3), rtol=0, atol=0
    )
