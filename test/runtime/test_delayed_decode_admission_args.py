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

"""Decode destination allocation is an explicit serving choice."""

import pytest

from tokenspeed.runtime.utils.server_args import ServerArgs


def test_prefill_requires_explicit_allocation_policy():
    with pytest.raises(ValueError, match="requires explicit"):
        ServerArgs(model="unused", disaggregation_mode="prefill")


@pytest.mark.parametrize("policy", ["early", "independent"])
def test_prefill_preserves_selected_allocation_policy(policy):
    args = ServerArgs(
        model="unused",
        disaggregation_mode="prefill",
        disaggregation_decode_allocation_policy=policy,
        disaggregation_layerwise_interval=0,
    )
    assert args.disaggregation_decode_allocation_policy == policy
    assert args.disaggregation_layerwise_interval == 0


@pytest.mark.parametrize("role", ["null", "decode"])
def test_non_prefill_uses_original_admission_without_a_policy(role):
    args = ServerArgs(model="unused", disaggregation_mode=role)
    assert args.disaggregation_decode_allocation_policy is None
    with pytest.raises(ValueError, match="only valid for prefill"):
        ServerArgs(
            model="unused",
            disaggregation_mode=role,
            disaggregation_decode_allocation_policy="early",
        )


def test_independent_admission_refuses_layerwise_transfer_without_overriding_it():
    with pytest.raises(ValueError, match="layerwise-interval 0"):
        ServerArgs(
            model="unused",
            disaggregation_mode="prefill",
            disaggregation_decode_allocation_policy="independent",
            disaggregation_layerwise_interval=1,
        )


def test_independent_admission_refuses_unsupported_attention_topology():
    with pytest.raises(ValueError, match="TP1/KVP1"):
        ServerArgs(
            model="unused",
            world_size=2,
            attn_tp_size=2,
            disaggregation_mode="prefill",
            disaggregation_decode_allocation_policy="independent",
            disaggregation_layerwise_interval=0,
        )
