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

"""Public opt-in rejects unsupported models and removed experimental options."""

import pytest
from torch import nn

from tokenspeed.runtime.execution.drafter import get_drafter_impl
from tokenspeed.runtime.execution.drafter.dspark import DSpark
from tokenspeed.runtime.utils.env import global_server_args_dict
from tokenspeed.runtime.utils.server_args import prepare_server_args


def test_adaptive_opt_in_requires_v41_model(monkeypatch):
    model = nn.Linear(1, 1)
    monkeypatch.setitem(global_server_args_dict, "enable_dsv41_adaptive_dspark", False)
    assert get_drafter_impl("DSPARK", model) is DSpark
    monkeypatch.setitem(global_server_args_dict, "enable_dsv41_adaptive_dspark", True)
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
    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.execution.drafter.base import BaseDrafter
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout

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
