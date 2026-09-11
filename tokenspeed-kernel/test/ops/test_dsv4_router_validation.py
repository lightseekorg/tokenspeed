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
            hash_table_values_validated=False,
        )


@pytest.mark.parametrize("input_id", [-1, 2])
def test_validated_hash_router_still_checks_token_ids(input_id: int) -> None:
    with pytest.raises(ValueError, match="input_ids entries"):
        dsv4_select_experts(
            torch.zeros((1, 4)),
            top_k=2,
            renormalize=True,
            hash_indices_table=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
            input_ids=torch.tensor([input_id]),
            hash_table_values_validated=True,
        )


def test_validated_hash_router_skips_only_table_values(monkeypatch) -> None:
    import tokenspeed_kernel.ops.moe as moe

    names = []
    validate = moe._assert_indices_in_range

    def record(indices, limit, name):
        names.append(name)
        validate(indices, limit, name)

    class StopBeforeKernel(Exception):
        pass

    def stop(*args, **kwargs):
        raise StopBeforeKernel

    monkeypatch.setattr(moe, "_assert_indices_in_range", record)
    monkeypatch.setattr(moe, "select_kernel", stop)
    for validated, expected in [
        (False, ["input_ids", "hash_indices_table"]),
        (True, ["input_ids"]),
    ]:
        names.clear()
        with pytest.raises(StopBeforeKernel):
            dsv4_select_experts(
                torch.zeros((1, 4)),
                top_k=2,
                renormalize=True,
                hash_indices_table=torch.tensor([[0, 1]], dtype=torch.int32),
                input_ids=torch.tensor([0]),
                hash_table_values_validated=validated,
            )
        assert names == expected
