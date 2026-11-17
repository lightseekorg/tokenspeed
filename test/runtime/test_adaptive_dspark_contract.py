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

"""Packed candidate protocol and public adaptive DSpark opt-in contracts."""

import pytest
import torch
from torch import nn

from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.drafter import get_drafter_impl
from tokenspeed.runtime.execution.drafter.base import BaseDrafter
from tokenspeed.runtime.execution.drafter.deepseek_v41_dspark import DeepseekV41DSpark
from tokenspeed.runtime.execution.drafter.dspark import DSpark
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.model_executor import ModelExecutor
from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout
from tokenspeed.runtime.utils.env import global_server_args_dict
from tokenspeed.runtime.utils.server_args import prepare_server_args


@pytest.mark.parametrize(
    "flag", ["enable_dsv41_adaptive_dspark", "enable_dp_local_cuda_graph"]
)
def test_adaptive_opt_in_requires_v41_model(monkeypatch, flag):
    model = nn.Linear(1, 1)
    monkeypatch.setitem(global_server_args_dict, "enable_dsv41_adaptive_dspark", False)
    monkeypatch.setitem(global_server_args_dict, "enable_dp_local_cuda_graph", False)
    assert get_drafter_impl("DSPARK", model) is DSpark
    monkeypatch.setitem(global_server_args_dict, flag, True)
    with pytest.raises(ValueError, match="native V4.1 DSpark model"):
        get_drafter_impl("DSPARK", model)


@pytest.mark.parametrize(
    "option,value",
    [
        ("--dsv41-dspark-confidence-threshold", "0"),
        ("--dsv41-dspark-verify-costs", "costs.json"),
        ("--dsv41-dspark-verify-graph-token-step", "16"),
    ],
)
def test_removed_experimental_options_are_rejected(option, value):
    with pytest.raises(SystemExit) as result:
        prepare_server_args(["--model", "unused", option, value])
    assert result.value.code == 2


@pytest.mark.parametrize("bs,width", [(1, 1), (2, 4), (32, 6)])
def test_native_drafter_query_capacity(bs, width):
    drafter = BaseDrafter.__new__(BaseDrafter)
    drafter.spec_num_tokens = width
    ctx = ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=bs,
        num_extends=0,
        input_num_tokens=bs * width,
        forward_mode=ForwardMode.DECODE,
        output_layout=ForwardOutputLayout(0, 0, bs, width),
    )
    assert drafter.decode_graph_query_capacities(bs) == (bs * width,)
    assert (
        drafter.prepare_decode_graph_token_layout(ctx, actual_bs=bs, padded_bs=bs * 2)
        == bs * 2 * width
    )
    drafter.prepare_decode_graph_token_capture_layout(bs=bs, query_tokens=bs * width)
    with pytest.raises(ValueError, match="full query capacity"):
        drafter.prepare_decode_graph_token_capture_layout(
            bs=bs, query_tokens=bs * width + 1
        )


@pytest.mark.parametrize("width", [4, 6])
def test_oversized_adaptive_buckets_keep_native_capture(width):
    from tokenspeed_kernel.ops.attention.dsv41 import VERIFY_LAYOUT_MAX_TOKENS

    drafter = DeepseekV41DSpark.__new__(DeepseekV41DSpark)
    drafter.dp_local_graph = False
    BaseDrafter.__init__(drafter, spec_num_tokens=width)
    drafter.adaptive_min_batch_size = 4
    drafter.confidence_scores = torch.empty(1)
    drafter.verify_graph_token_step = 16
    drafter._verify_rows = None
    drafter._verify_indptr = None
    largest_supported = VERIFY_LAYOUT_MAX_TOKENS // width

    supported = drafter.decode_graph_query_capacities(largest_supported)
    assert supported[0] == largest_supported
    assert supported[-1] == largest_supported * width
    assert len(supported) > 1
    for bs in [largest_supported + 1, largest_supported * 2]:
        full = bs * width
        assert drafter.decode_graph_query_capacities(bs) == (full,)
        drafter.prepare_decode_graph_token_capture_layout(bs=bs, query_tokens=full)
        with pytest.raises(ValueError, match="full query capacity"):
            drafter.prepare_decode_graph_token_capture_layout(
                bs=bs, query_tokens=full - 1
            )
    assert drafter.decode_graph_query_capacities(largest_supported) == supported


class _PackedCandidateDrafter(BaseDrafter):
    def __init__(self, candidates: torch.Tensor):
        super().__init__(spec_num_tokens=candidates.shape[1])
        self.candidates = candidates

    def verify_candidates(self) -> torch.Tensor:
        return self.candidates


@pytest.fixture
def packed_decode_context():
    return ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=2,
        num_extends=0,
        input_num_tokens=5,
        forward_mode=ForwardMode.DECODE,
        output_layout=ForwardOutputLayout(0, 0, 2, 4),
        decode_query_indptr=torch.tensor([0, 2, 5], dtype=torch.int64),
    )


def test_packed_executor_reads_live_original_candidates(packed_decode_context):
    candidates = torch.arange(8, dtype=torch.int32).reshape(2, 4)
    executor = ModelExecutor.__new__(ModelExecutor)
    executor.drafter = _PackedCandidateDrafter(candidates)

    result = executor._decode_candidates(packed_decode_context)
    torch.testing.assert_close(
        result, torch.arange(8, dtype=torch.int32).reshape(2, 4), rtol=0, atol=0
    )
    candidates.add_(10)
    torch.testing.assert_close(
        result, torch.arange(10, 18, dtype=torch.int32).reshape(2, 4), rtol=0, atol=0
    )
    assert torch.equal(executor._decode_candidates(packed_decode_context), result)


def test_packed_executor_rejects_unsupported_drafter(packed_decode_context):
    executor = ModelExecutor.__new__(ModelExecutor)
    executor.drafter = BaseDrafter(spec_num_tokens=4)
    with pytest.raises(ValueError, match="does not support packed verification"):
        executor._decode_candidates(packed_decode_context)


def test_packed_executor_requires_drafter(packed_decode_context):
    executor = ModelExecutor.__new__(ModelExecutor)
    executor.drafter = None
    with pytest.raises(ValueError, match="requires a drafter"):
        executor._decode_candidates(packed_decode_context)


@pytest.mark.parametrize("packed", [False, True])
def test_executor_without_decodes_has_no_candidates(packed_decode_context, packed):
    executor = ModelExecutor.__new__(ModelExecutor)
    executor.drafter = BaseDrafter(spec_num_tokens=4)
    ctx = packed_decode_context
    ctx.num_extends = ctx.bs
    ctx.forward_mode = ForwardMode.EXTEND
    ctx.output_layout = ForwardOutputLayout(2, 2, 0, 4)
    ctx.decode_query_indptr = torch.tensor([0], dtype=torch.int64) if packed else None
    assert executor._decode_candidates(ctx) is None
