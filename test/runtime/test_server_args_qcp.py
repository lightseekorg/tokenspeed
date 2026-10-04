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

"""``--prefill-context-parallel-size``: the layouts the first landing refuses."""

from __future__ import annotations

import pytest

from tokenspeed.runtime.utils.server_args import prepare_server_args, validate_qcp

BASE = [
    "--model",
    "x",
    "--attn-tp-size",
    "2",
    "--prefill-context-parallel-size",
    "2",
    "--disaggregation-mode",
    "prefill",
    "--disable-prefill-graph",
    "--attention-backend",
    "dsa",
]


def test_the_prefill_role_accepts_a_full_tp_query_shard():
    args = prepare_server_args(BASE)
    assert args.mapping.attn.qcp_size == 2
    assert args.mapping.attn.has_qcp
    args.mapping.rank = 1
    assert args.mapping.attn.qcp_group == args.mapping.attn.tp_group == (0, 1)
    assert args.mapping.attn.qcp_rank == 1
    # DCP equal to the shard group is the one sharded-page layout allowed
    # (and it inherits DCP's Host KVStore refusal).
    args = prepare_server_args(
        BASE + ["--decode-context-parallel-size", "2", "--disable-kvstore"]
    )
    assert args.mapping.attn.dcp_size == 2


def test_off_by_default_everywhere():
    args = prepare_server_args(["--model", "x", "--attn-tp-size", "2"])
    assert args.prefill_context_parallel_size == 1
    assert not args.mapping.attn.has_qcp


@pytest.mark.parametrize(
    "argv,match",
    [
        (
            ["--attn-tp-size", "4", "--prefill-context-parallel-size", "2"],
            "must equal the attention TP size",
        ),
        (["--disaggregation-mode", "null"], "requires --disaggregation-mode prefill"),
        (["--disaggregation-mode", "decode"], "requires --disaggregation-mode prefill"),
        (["--enable-mixed-batch"], "--enable-mixed-batch"),
        (["--attention-backend", "flashmla"], "DSA-family attention backend"),
    ],
)
def test_refusals(argv, match):
    # Later flags override earlier ones, so BASE + argv applies the override.
    with pytest.raises(ValueError, match=match):
        prepare_server_args(BASE + argv)


def test_refuses_dcp_narrower_than_the_shard_group():
    argv = [flag if flag != "2" else "4" for flag in BASE]
    with pytest.raises(ValueError, match="--decode-context-parallel-size must be 1"):
        prepare_server_args(argv + ["--decode-context-parallel-size", "2"])


def test_refuses_the_prefill_graph():
    argv = [flag for flag in BASE if flag != "--disable-prefill-graph"]
    with pytest.raises(ValueError, match="--disable-prefill-graph"):
        prepare_server_args(argv)


def test_refuses_attention_dp():
    with pytest.raises(ValueError, match="attention DP 1"):
        prepare_server_args(BASE + ["--data-parallel-size", "2"])


def test_validate_qcp_rejects_a_shard_below_the_tp_width():
    with pytest.raises(ValueError, match="must equal the attention TP size"):
        validate_qcp(
            qcp_size=2,
            attn_tp_size=4,
            attn_dp_size=1,
            dcp_size=1,
            disaggregation_mode="prefill",
            disable_prefill_graph=True,
            enable_mixed_batch=False,
            attention_backend="dsa",
        )
    # An unset backend resolves to the architecture's default; the attention
    # config pins it to GPU DSA when the model is known.
    validate_qcp(
        qcp_size=4,
        attn_tp_size=4,
        attn_dp_size=1,
        dcp_size=4,
        disaggregation_mode="prefill",
        disable_prefill_graph=True,
        enable_mixed_batch=False,
        attention_backend=None,
    )
