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
from tokenspeed_kernel.ops.moe import dsv4_select_experts


@pytest.mark.parametrize("invalid", [-1, 4])
def test_default_hash_router_rejects_invalid_table_values(invalid: int) -> None:
    logits = torch.zeros((1, 4), dtype=torch.float32)
    table = torch.tensor([[0, invalid]], dtype=torch.int32)
    input_ids = torch.zeros((1,), dtype=torch.int64)

    with pytest.raises(ValueError, match=r"entries must be in \[0, 4\)"):
        dsv4_select_experts(
            logits,
            top_k=2,
            renormalize=True,
            hash_indices_table=table,
            input_ids=input_ids,
        )


def test_trusted_hash_table_still_checks_runtime_input_ids() -> None:
    logits = torch.zeros((1, 4), dtype=torch.float32)
    table = torch.tensor([[0, 1]], dtype=torch.int32)
    input_ids = torch.ones((1,), dtype=torch.int64)

    with pytest.raises(ValueError, match=r"input_ids entries must be in \[0, 1\)"):
        dsv4_select_experts(
            logits,
            top_k=2,
            renormalize=True,
            hash_indices_table=table,
            input_ids=input_ids,
            hash_table_values_validated=True,
        )


def test_hash_table_validation_contract_rejects_invalid_option_use() -> None:
    logits = torch.zeros((1, 4), dtype=torch.float32)

    with pytest.raises(ValueError, match="requires hash_indices_table"):
        dsv4_select_experts(
            logits,
            top_k=2,
            renormalize=True,
            hash_table_values_validated=True,
        )
    with pytest.raises(TypeError, match="must be a bool"):
        dsv4_select_experts(
            logits,
            top_k=2,
            renormalize=True,
            hash_table_values_validated=1,
        )
