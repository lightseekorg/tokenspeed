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

"""Gluon DCP placement and metadata lifecycle, without model weights."""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from functools import partial
from unittest.mock import Mock

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="runtime-1gpu")


def _check_visibility(local, blocks, ends, *, rank, degree, queries, block):
    expected = []
    for row, length in zip(blocks, ends):
        owners = [b > 0 and (b - 1) % degree == rank for b in row for _ in range(128)]
        bounds = (
            [length] * queries
            if block
            else [max(0, length - queries + j + 1) for j in range(queries)]
        )
        expected.append([sum(owners[:end]) for end in bounds])
    assert local.local_visible_lens.tolist() == expected
    assert local.local_seq_lens.tolist() == [r[-1] for r in expected]


@pytest.fixture
def make_leaf(monkeypatch):
    if not torch.cuda.is_available():
        # Runtime backend discovery probes a GPU even for CPU metadata tests.
        # Mock only that missing hardware; the leaf and placement kernels are real.
        from tokenspeed_kernel._triton import triton
        from tokenspeed_kernel.platform import ArchVersion, Platform, PlatformInfo
        from tokenspeed_triton.backends.compiler import GPUTarget

        driver = Mock()
        driver.get_current_target.return_value = GPUTarget("hip", "gfx950", 64)
        monkeypatch.setattr(triton.runtime.driver, "_active", driver)
        monkeypatch.setattr(
            Platform,
            "_instance",
            PlatformInfo(
                "amd",
                ArchVersion(9, 5),
                "gfx950",
                8,
                288 * 2**30,
                8000,
                256,
                2048,
                65536,
            ),
        )

    from tokenspeed.runtime.layers.attention.backends.paged.mla import (
        GluonMLAAttnBackend,
    )
    from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
    from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
    from tokenspeed.runtime.utils.env import global_server_args_dict

    monkeypatch.setitem(global_server_args_dict, "numerics", "auto")

    def make(*, degree, rank, queries, draft, block, device="cpu", **overrides):
        spec = MLAConfig(
            backend_name="gluon",
            num_attention_heads=96,
            num_kv_heads=1,
            head_dim=576,
            attn_tp_size=8,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            scaling=192**-0.5,
            kv_cache_dim=576,
        )
        config = AttnConfig(
            device="cuda",
            dtype=torch.bfloat16,
            kv_cache_dtype=torch.float8_e4m3fn,
            kv_cache_quant_method="none",
            context_len=512,
            max_bs=3,
            speculative_num_draft_tokens=queries,
            is_draft=draft,
            draft_block_decode=block,
            kvp_size=degree,
            kvp_rank=rank,
            kvp_group=tuple(range(degree)),
            components=(spec,),
        )
        config = replace(config, **overrides)
        leaf = GluonMLAAttnBackend(config, spec, kernel_page_size=64)
        # CPU cases exercise the same metadata implementation, without GPU allocation.
        leaf.device = device
        leaf.configure_runtime(
            block_granularity=128, virtual_block_count=33, shard_count=degree
        )
        leaf.init_cuda_graph_state(3)
        return leaf

    return make


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("degree", [2, 4, 8])
@pytest.mark.parametrize("draft,block", [(False, False), (True, False), (True, True)])
def test_gluon_dcp_metadata_refresh_and_frontier(
    make_leaf, device, degree, draft, block
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU required")
    queries = 4
    for rank in range(degree):
        leaf = make_leaf(
            degree=degree,
            rank=rank,
            queries=queries,
            draft=draft,
            block=block,
            device=device,
        )
        blocks = [[5, 2, 8, 1], [2, 5, 1, 8], [0, 0, 0, 0]]
        table = torch.tensor(
            [[2 * b + sub for b in row for sub in range(2)] for row in blocks],
            dtype=torch.int32,
            device=device,
        )
        lengths = torch.tensor([127, 255, 1], dtype=torch.int32, device=device)
        leaf.refresh_decode_metadata(3, 2, lengths, table)
        metadata = leaf.forward_decode_metadata
        local = metadata.dcp
        pointers = [
            tensor.data_ptr()
            for tensor in (
                local.local_page_table,
                local.local_seq_lens,
                local.local_visible_lens,
            )
        ]

        check = partial(
            _check_visibility,
            local,
            blocks,
            rank=rank,
            degree=degree,
            queries=queries,
            block=block,
        )

        check([127, 255, 1 if draft and not block else 4])
        lengths.add_(2)
        if draft:
            if block:
                leaf.fill_block_decode_seq_lens(3, lengths)
            else:
                leaf.advance_draft_forward_metadata(lengths)
            check([129, 257, 4 if block else 3])
        else:
            leaf.refresh_decode_metadata(3, 2, lengths, table)
            check([129, 257, 4])
        assert [
            tensor.data_ptr()
            for tensor in (
                local.local_page_table,
                local.local_seq_lens,
                local.local_visible_lens,
            )
        ] == pointers
        assert leaf._decode_views(3).dcp is local
        assert not leaf.supports_mla_projected_value_decode
        assert not leaf._should_use_absorbed_cached_extend(
            max_extend_seq_len=4, max_extend_prefix_len=128
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e5m2])
def test_gluon_dcp_rejects_unsupported_cache(make_leaf, dtype):
    with pytest.raises(ValueError, match="E4M3 KV"):
        make_leaf(
            degree=8, rank=0, queries=4, draft=False, block=False, kv_cache_dtype=dtype
        )


def test_gluon_dcp_reinitialization_and_topology(make_leaf):
    leaf = make_leaf(degree=8, rank=0, queries=4, draft=False, block=False)
    previous = leaf._dcp.metadata
    leaf.init_cuda_graph_state(3)
    assert (
        leaf._dcp.metadata.local_page_table.data_ptr()
        != previous.local_page_table.data_ptr()
    )
    with pytest.raises(ValueError, match="topology disagree"):
        leaf.configure_runtime(
            block_granularity=128, virtual_block_count=33, shard_count=4
        )


def test_gluon_dcp_rejects_scaled_cache_rows(make_leaf):
    with pytest.raises(ValueError, match="plain E4M3 cache rows"):
        make_leaf(
            degree=8,
            rank=0,
            queries=4,
            draft=False,
            block=False,
            kv_cache_quant_method="per_token_head",
        )


@pytest.mark.parametrize(
    "argument", ["value_weight", "output_gate", "projected_output"]
)
def test_gluon_dcp_rejects_partial_value_projection(make_leaf, argument):
    leaf = make_leaf(degree=8, rank=0, queries=4, draft=False, block=False)
    # Rejection precedes kernel/cache access: partial attention cannot consume
    # a projection or gate meant for the globally merged output.
    with pytest.raises(ValueError, match="value projection after the LSE merge"):
        leaf.forward_decode(
            torch.empty(0),
            None,
            None,
            None,
            None,
            None,
            0,
            **{argument: torch.empty(0)},
        )


def test_gluon_without_dcp_keeps_existing_metadata(make_leaf):
    leaf = make_leaf(degree=1, rank=0, queries=1, draft=False, block=False)
    assert leaf.cache_placement(None) is None
    lengths = torch.tensor([13, 27], dtype=torch.int32)
    table = torch.tensor([[2] * 8, [3] * 8], dtype=torch.int32)
    leaf.refresh_decode_metadata(2, 2, lengths, table)
    assert leaf.forward_decode_metadata.dcp is None
    torch.testing.assert_close(leaf.forward_decode_metadata.seq_lens, lengths)
    assert leaf.supports_mla_projected_value_decode
