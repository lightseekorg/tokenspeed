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
from tokenspeed_kernel.ops.moe import moe_topk


@pytest.mark.parametrize("invalid", [-1, 4])
def test_default_hash_router_rejects_invalid_table_values(invalid: int) -> None:
    logits = torch.zeros((1, 4), dtype=torch.float32)
    table = torch.tensor([[0, invalid]], dtype=torch.int32)
    input_ids = torch.zeros((1,), dtype=torch.int64)

    with pytest.raises(ValueError, match=r"entries must be in \[0, 4\)"):
        moe_topk(
            logits,
            top_k=2,
            score_function="sqrt_softplus",
            selection_method="hash",
            renormalize=True,
            routed_scaling_factor=1.0,
            hash_indices_table=table,
            input_ids=input_ids,
        )


def test_non_hash_router_rejects_input_ids() -> None:
    logits = torch.zeros((1, 4), dtype=torch.float32)

    with pytest.raises(ValueError, match="hash routing inputs"):
        moe_topk(
            logits,
            top_k=2,
            score_function="sqrt_softplus",
            selection_method="topk",
            renormalize=True,
            routed_scaling_factor=1.0,
            input_ids=torch.zeros((1,), dtype=torch.int64),
        )


def test_hash_router_rejects_correction_bias() -> None:
    logits = torch.zeros((1, 4), dtype=torch.float32)
    table = torch.tensor([[0, 1]], dtype=torch.int32)
    input_ids = torch.zeros((1,), dtype=torch.int64)

    with pytest.raises(ValueError, match="correction_bias"):
        moe_topk(
            logits,
            top_k=2,
            score_function="sqrt_softplus",
            selection_method="hash",
            renormalize=True,
            routed_scaling_factor=1.0,
            correction_bias=torch.zeros(4),
            hash_indices_table=table,
            input_ids=input_ids,
        )


@pytest.mark.parametrize("tokens", [0, 2])
@pytest.mark.parametrize("renormalize", [False, True])
@pytest.mark.parametrize("routing", ["plain", "bias", "per_token_bias", "hash"])
@pytest.mark.parametrize("override", [None, "torch_sqrt_softplus_topk"])
def test_sqrt_softplus_router_preserves_output_dtypes(
    tokens: int, renormalize: bool, routing: str, override: str | None
) -> None:
    logits = torch.tensor([[-2.0, 0.5, 3.0, 1.0], [2.0, -1.0, 0.0, 4.0]])[:tokens]
    bias = None
    table = None
    input_ids = None
    if routing == "bias":
        bias = torch.tensor([3.0, 0.0, -2.0, 0.0])
    elif routing == "per_token_bias":
        bias = torch.tensor([[3.0, 0.0, -2.0, 0.0], [0.0, 4.0, 0.0, -3.0]])[:tokens]
    elif routing == "hash":
        table = torch.tensor([[3, 0], [1, 2]], dtype=torch.int32)
        input_ids = torch.tensor([1, 0], dtype=torch.int64)[:tokens]

    weights, ids = moe_topk(
        logits,
        top_k=2,
        score_function="sqrt_softplus",
        selection_method="hash" if routing == "hash" else "topk",
        renormalize=renormalize,
        routed_scaling_factor=2.5,
        correction_bias=bias,
        hash_indices_table=table,
        input_ids=input_ids,
        topk_indices_dtype=torch.int64,
        topk_weights_dtype=torch.float64,
        override=override,
        solution="torch",
    )

    scores = torch.nn.functional.softplus(logits).sqrt()
    if routing == "hash":
        expected_ids = table[input_ids].long()
    else:
        selection_scores = scores if bias is None else scores + bias
        expected_ids = selection_scores.topk(2, dim=-1).indices
    expected_weights = scores.gather(1, expected_ids)
    if renormalize:
        expected_weights /= expected_weights.sum(dim=-1, keepdim=True)
    expected_weights = (expected_weights * 2.5).double()

    torch.testing.assert_close(ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(weights, expected_weights, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("score_function", ["softmax", "sigmoid"])
@pytest.mark.parametrize("renormalize", [False, True])
def test_unified_router_preserves_override_and_output_dtypes(
    score_function: str, renormalize: bool
) -> None:
    logits = torch.tensor([[-2.0, 0.5, 3.0, 1.0], [2.0, -1.0, 0.0, 4.0]])
    bias = torch.tensor([1.0, 0.0, -1.0, 0.0]) if score_function == "sigmoid" else None
    expert_map = torch.tensor([3, 2, 0, 1]) if score_function == "sigmoid" else None
    override = (
        "torch_sigmoid_bias_topk"
        if score_function == "sigmoid"
        else "torch_softmax_topk"
    )
    weights, ids = moe_topk(
        logits,
        top_k=2,
        score_function=score_function,
        selection_method="topk",
        renormalize=renormalize,
        routed_scaling_factor=2.5,
        correction_bias=bias,
        logical_to_physical_map=expert_map,
        topk_indices_dtype=torch.int64,
        topk_weights_dtype=torch.float64,
        override=override,
    )

    scores = logits.sigmoid() if score_function == "sigmoid" else logits.softmax(-1)
    selection_scores = scores if bias is None else scores + bias
    expected_ids = selection_scores.topk(2, dim=-1).indices
    expected_weights = scores.gather(1, expected_ids)
    if renormalize:
        expected_weights /= expected_weights.sum(dim=-1, keepdim=True)
    if expert_map is not None:
        expected_ids = expert_map[expected_ids]

    torch.testing.assert_close(ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(
        weights, (expected_weights * 2.5).double(), rtol=1e-6, atol=1e-7
    )
