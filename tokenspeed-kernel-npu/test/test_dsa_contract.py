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

"""CPU tensor contracts at the Ascend DSA native operator boundary."""

import pytest
import torch

pytest.importorskip("torch_npu")
pytest.importorskip("flash_ops")

from tokenspeed_kernel_npu.ops.dsa import AscendDSAKernels

_CP_OPERATORS = (
    "npu_dcp_merge_topk",
    "npu_dsa_merge_lse_direct",
    "npu_select_local_topk_indices",
    "npu_kvp_attention_merge",
)


@pytest.fixture(autouse=True)
def native_constructor_dependencies(monkeypatch):
    def unavailable(*args, **kwargs):
        raise AssertionError("Indexer/attention execution is outside this CPU test")

    for name in ("npu_lightning_indexer", "npu_sparse_flash_attention_decode"):
        if getattr(torch.ops.custom, name, None) is None:
            monkeypatch.setattr(torch.ops.custom, name, unavailable, raising=False)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_partial_merge_handles_all_empty_and_mixed_rows(monkeypatch, dtype):
    def native_merge(output, lse):
        # Model the native normalized-partial ABI, including its undefined
        # softmax for rows whose LSE values are all -inf.
        assert output.dtype == torch.float32
        assert lse.dtype == torch.float32
        weights = torch.softmax(lse, dim=0)
        return (output * weights[..., None]).sum(0).to(torch.bfloat16)

    monkeypatch.setattr(
        torch.ops.custom, "npu_kvp_attention_merge", native_merge, raising=False
    )
    kernels = AscendDSAKernels()
    output = torch.tensor(
        [
            [[0, 0], [1, 3], [0, 0], [2, 4]],
            [[0, 0], [5, 7], [6, 8], [5, 7]],
            [[0, 0], [0, 0], [10, 12], [8, 10]],
        ],
        dtype=dtype,
    )
    lse = torch.tensor([[0, 1, 0, 1], [0, 3, 1, 1], [0, 0, 1, 1]]).float().log()
    original_output, original_lse = output.clone(), lse.clone()

    result = kernels.merge_partials(output, lse)

    expected = torch.tensor([[0, 0], [4, 6], [8, 10], [5, 7]], dtype=torch.bfloat16)
    torch.testing.assert_close(result, expected, atol=0, rtol=0)
    torch.testing.assert_close(output, original_output, atol=0, rtol=0)
    torch.testing.assert_close(lse, original_lse, atol=0, rtol=0)


@pytest.mark.parametrize("missing", _CP_OPERATORS)
def test_cp_startup_requires_every_execution_operator(monkeypatch, missing):
    def unexpected_call(*args, **kwargs):
        raise AssertionError("Missing operators must fail before the Meta probe")

    for name in _CP_OPERATORS:
        monkeypatch.setattr(torch.ops.custom, name, unexpected_call, raising=False)
    monkeypatch.setattr(torch.ops.custom, missing, None)
    kernels = AscendDSAKernels()

    with pytest.raises(RuntimeError, match=missing):
        kernels.require_fused_context_parallel(rank_size=8, num_heads=64)
