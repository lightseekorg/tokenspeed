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
from tokenspeed_kernel.ops.attn_res import attn_res_fwd, attn_res_fwd_available


def _blackwell_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


pytestmark = pytest.mark.skipif(
    not _blackwell_available(), reason="SM100-family CUDA GPU is required"
)


@pytest.mark.parametrize("num_valid_blocks", [0, 3])
@pytest.mark.parametrize("has_delta", [False, True])
def test_cuda_attn_res_writes_bit_exact_snapshot(
    num_valid_blocks: int, has_delta: bool
) -> None:
    torch.manual_seed(29)
    layer = torch.randn(1, 7168, device="cuda", dtype=torch.bfloat16)
    delta = torch.randn_like(layer) if has_delta else None
    blocks = torch.randn(
        num_valid_blocks + 1, 1, 7168, device="cuda", dtype=torch.bfloat16
    )
    baseline_blocks = blocks.clone()
    res_weight = torch.randn(7168, device="cuda", dtype=torch.bfloat16)
    rms_weight = torch.randn(7168, device="cuda", dtype=torch.bfloat16)
    out_norm_weight = torch.randn(7168, device="cuda", dtype=torch.bfloat16)
    expected_prefix = (
        layer.clone()
        if delta is None
        else (layer.float() + delta.float()).to(torch.bfloat16)
    )

    assert attn_res_fwd_available(
        layer,
        blocks,
        res_weight,
        rms_weight,
        1e-5,
        out_norm_weight=out_norm_weight,
        delta=delta,
        num_valid_blocks=num_valid_blocks,
        block_write_idx=num_valid_blocks,
    )
    actual = attn_res_fwd(
        layer,
        blocks,
        res_weight,
        rms_weight,
        1e-5,
        out_norm_weight=out_norm_weight,
        delta=delta,
        num_valid_blocks=num_valid_blocks,
        block_write_idx=num_valid_blocks,
    )
    baseline = attn_res_fwd(
        expected_prefix.clone(),
        baseline_blocks,
        res_weight,
        rms_weight,
        1e-5,
        out_norm_weight=out_norm_weight,
        num_valid_blocks=num_valid_blocks,
    )

    torch.testing.assert_close(actual, baseline, atol=0, rtol=0)
    torch.testing.assert_close(
        blocks[num_valid_blocks], expected_prefix, atol=0, rtol=0
    )


def test_cuda_attn_res_snapshot_survives_strided_block_storage() -> None:
    """A non-contiguous block store must still receive the snapshot.

    The adapter passes ``block_residual.unsqueeze(2).contiguous()``. For a view
    that is strided in the block dimension that materializes a private copy, so
    the kernel would write the snapshot into the temporary and the caller's row
    would silently stay stale -- only the mixed output is returned.
    """
    torch.manual_seed(31)
    num_valid_blocks = 3
    layer = torch.randn(1, 7168, device="cuda", dtype=torch.bfloat16)
    # Back the blocks with a padded buffer and take a strided view: the hidden
    # dimension stays contiguous, the block dimension does not.
    backing = torch.randn(
        num_valid_blocks + 1, 2, 7168, device="cuda", dtype=torch.bfloat16
    )
    blocks = backing[:, :1, :]
    assert not blocks.is_contiguous()
    assert blocks.stride(-1) == 1
    res_weight = torch.randn(7168, device="cuda", dtype=torch.bfloat16)
    rms_weight = torch.randn(7168, device="cuda", dtype=torch.bfloat16)
    out_norm_weight = torch.randn(7168, device="cuda", dtype=torch.bfloat16)

    if not attn_res_fwd_available(
        layer,
        blocks,
        res_weight,
        rms_weight,
        1e-5,
        out_norm_weight=out_norm_weight,
        delta=None,
        num_valid_blocks=num_valid_blocks,
        block_write_idx=num_valid_blocks,
    ):
        pytest.skip("CUDA AttnRes rejects this storage; nothing to verify")

    attn_res_fwd(
        layer,
        blocks,
        res_weight,
        rms_weight,
        1e-5,
        out_norm_weight=out_norm_weight,
        delta=None,
        num_valid_blocks=num_valid_blocks,
        block_write_idx=num_valid_blocks,
    )
    # The snapshot is the unmodified prefix row (no delta), written through the
    # strided view into the caller's backing buffer.
    torch.testing.assert_close(blocks[num_valid_blocks, 0], layer[0])
    torch.testing.assert_close(backing[num_valid_blocks, 0], layer[0])


def test_cuda_attn_res_delta_writeback_survives_strided_layer_residual() -> None:
    """A strided layer residual must still receive the delta-accumulated prefix.

    Under ``delta``, the kernel writes ``prefix + delta`` back through
    ``layer_residual``; kimi_k3 then hands that same tensor to the MoE as the
    residual. If ``.contiguous()`` materialized a copy the caller would keep the
    stale pre-delta value while the returned mix looked correct.
    """
    torch.manual_seed(37)
    # T must exceed 1: with T == 1 the size-1 dims make unsqueeze(1) contiguous
    # regardless of stride, so the copy path is never taken and the test is void.
    backing = torch.randn(2, 2, 7168, device="cuda", dtype=torch.bfloat16)
    layer = backing[:, 0, :]
    assert not layer.unsqueeze(1).is_contiguous()
    delta = torch.randn(2, 7168, device="cuda", dtype=torch.bfloat16)
    blocks = torch.randn(4, 2, 7168, device="cuda", dtype=torch.bfloat16)
    res_weight = torch.randn(7168, device="cuda", dtype=torch.bfloat16)
    rms_weight = torch.randn(7168, device="cuda", dtype=torch.bfloat16)
    out_norm_weight = torch.randn(7168, device="cuda", dtype=torch.bfloat16)
    expected = (layer.float() + delta.float()).to(torch.bfloat16).clone()

    if not attn_res_fwd_available(
        layer,
        blocks,
        res_weight,
        rms_weight,
        1e-5,
        out_norm_weight=out_norm_weight,
        delta=delta,
        num_valid_blocks=3,
        block_write_idx=-1,
    ):
        pytest.skip("CUDA AttnRes rejects this storage; nothing to verify")

    attn_res_fwd(
        layer,
        blocks,
        res_weight,
        rms_weight,
        1e-5,
        out_norm_weight=out_norm_weight,
        delta=delta,
        num_valid_blocks=3,
        block_write_idx=-1,
    )
    torch.testing.assert_close(layer, expected)
    torch.testing.assert_close(backing[:, 0, :], expected)
