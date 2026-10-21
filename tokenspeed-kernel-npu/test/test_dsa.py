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

import pytest
import torch

pytest.importorskip("torch_npu")

from tokenspeed_kernel_npu.ops.dsa import AscendDSAKernels


def test_full_scan_chunk_hints_are_cached_by_shape():
    kernels = AscendDSAKernels.__new__(AscendDSAKernels)
    kernels._full_scan_chunks = {}
    indices = torch.zeros(3, 1, 2048, dtype=torch.int32)

    first = kernels._full_scan_chunk_hints(indices)
    second = kernels._full_scan_chunk_hints(indices)

    assert first.data_ptr() == second.data_ptr()
    assert first.tolist() == [16, 16, 16]


@pytest.mark.parametrize("missing", ["npu_dcp_merge_topk", "npu_dsa_merge_lse_direct"])
def test_missing_fused_cp_operators_fail_at_device_boundary(monkeypatch, missing):
    kernels = AscendDSAKernels.__new__(AscendDSAKernels)
    for name in ("npu_dcp_merge_topk", "npu_dsa_merge_lse_direct"):
        monkeypatch.setattr(torch.ops.custom, name, lambda: None, raising=False)
    monkeypatch.setattr(torch.ops.custom, missing, None)
    with pytest.raises(RuntimeError, match=missing):
        kernels.require_fused_context_parallel(rank_size=8, num_heads=64)


@pytest.mark.parametrize("degree", [2, 4, 8, 16])
def test_fused_cp_topology_is_validated_by_the_installed_operators(degree):
    pytest.importorskip("flash_ops")
    if any(
        getattr(torch.ops.custom, name, None) is None
        for name in ("npu_dcp_merge_topk", "npu_dsa_merge_lse_direct")
    ):
        pytest.skip("full native DSA binding required")
    kernels = AscendDSAKernels.__new__(AscendDSAKernels)
    if degree == 8:
        kernels.require_fused_context_parallel(rank_size=degree, num_heads=64)
    else:
        # Current native fusion is specialized for eight ranks. A wrapper that
        # silently substitutes eight for the requested degree would pass here.
        with pytest.raises(ValueError, match=f"reject CP{degree}"):
            kernels.require_fused_context_parallel(rank_size=degree, num_heads=64)


def test_fused_cp_contract_validates_the_head_layout():
    pytest.importorskip("flash_ops")
    if getattr(torch.ops.custom, "npu_dsa_merge_lse_direct", None) is None:
        pytest.skip("full native DSA binding required")
    kernels = AscendDSAKernels.__new__(AscendDSAKernels)
    with pytest.raises(ValueError, match="with 32 heads"):
        kernels.require_fused_context_parallel(rank_size=8, num_heads=32)


@pytest.mark.parametrize("missing", ["npu_dcp_merge_topk", "npu_dsa_merge_lse_direct"])
def test_fused_cp_missing_meta_reports_a_binding_upgrade(monkeypatch, missing):
    pytest.importorskip("flash_ops")
    names = ("npu_dcp_merge_topk", "npu_dsa_merge_lse_direct")
    if any(getattr(torch.ops.custom, name, None) is None for name in names):
        pytest.skip("full native DSA binding required")
    # Register the same ABI with a CPU kernel only. The real dispatcher then
    # raises NotImplementedError for the missing Meta kernel during the probe.
    library = torch.library.Library("_tokenspeed_dsa_no_meta", "FRAGMENT")
    original = getattr(torch.ops.custom, missing)
    library.define(str(original.default._schema).split("::", 1)[1])

    def cpu_only(*args, **kwargs):
        raise AssertionError("The topology probe must not execute a CPU kernel")

    library.impl(missing, cpu_only, "CPU")
    monkeypatch.setattr(
        torch.ops.custom, missing, getattr(torch.ops._tokenspeed_dsa_no_meta, missing)
    )
    kernels = AscendDSAKernels.__new__(AscendDSAKernels)
    with pytest.raises(RuntimeError, match="no Meta.*upgrade.*flash_ops") as error:
        kernels.require_fused_context_parallel(rank_size=8, num_heads=64)
    assert isinstance(error.value.__cause__, NotImplementedError)


@pytest.mark.parametrize("degree", [2, 4, 16])
def test_lse_adapter_preserves_the_requested_cp_degree(degree):
    pytest.importorskip("flash_ops")
    if getattr(torch.ops.custom, "npu_dsa_merge_lse_direct", None) is None:
        pytest.skip("full native DSA binding required")
    kernels = AscendDSAKernels.__new__(AscendDSAKernels)
    output = torch.empty((64, 64, 512), dtype=torch.float32, device="meta")
    statistics = torch.empty((1, 64, 64), dtype=torch.float32, device="meta")
    peers = torch.empty(8, dtype=torch.int64, device="meta")
    with pytest.raises(RuntimeError, match="rank group"):
        kernels.merge_lse_direct(
            output,
            statistics,
            statistics,
            peers,
            group="test-dsa-contract",
            rank_size=degree,
            setup_mode=0,
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("strided", [False, True])
def test_prepared_projection_preserves_coefficients_and_owns_storage(dtype, strided):
    import torch_npu

    value = torch.arange(1024, dtype=torch.float32).reshape(32, 32).to(dtype).npu()
    if strided:
        value = value.t()
    expected = value.cpu().clone()
    prepared = AscendDSAKernels.prepare_projection_weight(value)
    assert torch_npu.get_npu_format(prepared) == 29
    assert prepared.dtype == dtype
    assert torch.equal(prepared.cpu(), expected)
    value.zero_()
    assert torch.equal(prepared.cpu(), expected)
