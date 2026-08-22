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
from tokenspeed_kernel import deepseek_v4_select_experts


def _select(
    table: torch.Tensor,
    input_ids: torch.Tensor | None,
    *,
    device: torch.device | str = "cpu",
) -> None:
    deepseek_v4_select_experts(
        torch.empty((2, 8), dtype=torch.float32, device=device),
        top_k=2,
        renormalize=True,
        hash_indices_table=table,
        input_ids=input_ids,
    )


@pytest.mark.parametrize(
    "table",
    [
        torch.empty((8,), dtype=torch.int32),
        torch.empty((8, 3), dtype=torch.int32),
        torch.empty((0, 2), dtype=torch.int32),
    ],
)
def test_hash_routing_validates_table_shape(table: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="shape.*vocabulary, top_k"):
        _select(table, torch.zeros((2,), dtype=torch.int64))


def test_hash_routing_validates_integer_dtypes() -> None:
    table = torch.empty((8, 2), dtype=torch.float32)
    with pytest.raises(ValueError, match="hash_indices_table.*int32 or int64"):
        _select(table, torch.zeros((2,), dtype=torch.int64))

    table = torch.empty((8, 2), dtype=torch.int32)
    with pytest.raises(ValueError, match="input_ids.*int32 or int64"):
        _select(table, torch.zeros((2,), dtype=torch.float32))


def test_hash_routing_validates_input_id_count() -> None:
    with pytest.raises(ValueError, match="input_ids must contain 2"):
        _select(
            torch.empty((8, 2), dtype=torch.int32),
            torch.zeros((3,), dtype=torch.int64),
        )


@pytest.mark.parametrize("input_id", [-1, 8])
def test_hash_routing_validates_input_id_bounds(input_id: int) -> None:
    with pytest.raises(ValueError, match="input_ids entries must be in"):
        _select(
            torch.zeros((8, 2), dtype=torch.int32),
            torch.tensor([0, input_id], dtype=torch.int64),
        )


@pytest.mark.parametrize("expert_id", [-1, 8])
def test_hash_routing_validates_expert_id_bounds(expert_id: int) -> None:
    table = torch.zeros((8, 2), dtype=torch.int32)
    table[0, 0] = expert_id
    with pytest.raises(ValueError, match="hash_indices_table entries must be in"):
        _select(table, torch.zeros((2,), dtype=torch.int64))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("mismatched", ["table", "input_ids"])
def test_hash_routing_requires_matching_devices(mismatched: str) -> None:
    table = torch.empty((8, 2), dtype=torch.int32, device="cuda")
    input_ids = torch.zeros((2,), dtype=torch.int64, device="cuda")
    if mismatched == "table":
        table = table.cpu()
    else:
        input_ids = input_ids.cpu()

    with pytest.raises(ValueError, match=f"{mismatched}.*same device"):
        _select(table, input_ids, device="cuda")
