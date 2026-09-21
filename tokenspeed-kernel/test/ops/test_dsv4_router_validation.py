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
