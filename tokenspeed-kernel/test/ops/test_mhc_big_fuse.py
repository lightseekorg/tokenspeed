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
from tokenspeed_kernel.ops.mhc.triton import (
    _pre_reduce_apply_fuses_norm,
    _pre_reduce_apply_is_supported,
)


def reference(
    projection: torch.Tensor,
    square_sum: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    residual: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = projection.sum(dim=0)
    rstd = torch.rsqrt(square_sum.sum(dim=0) / (4 * residual.shape[-1]) + rms_eps)
    pre = torch.sigmoid(values[:, :4] * rstd[:, None] * scale[0] + base[:4]) + hc_eps
    post = torch.sigmoid(values[:, 4:8] * rstd[:, None] * scale[1] + base[4:8]) * 2
    comb = (values[:, 8:] * rstd[:, None] * scale[2] + base[8:]).view(-1, 4, 4)
    comb = torch.softmax(comb, dim=2) + hc_eps
    comb = comb / (comb.sum(dim=1, keepdim=True) + hc_eps)
    for _ in range(1, sinkhorn_iters):
        comb = comb / (comb.sum(dim=2, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(dim=1, keepdim=True) + hc_eps)
    layer = torch.einsum("mi,mih->mh", pre, residual.float()).bfloat16()
    return layer, post, comb


@pytest.mark.parametrize("num_tokens", [1, 2, 4, 6, 8])
@pytest.mark.parametrize("n_splits", [1, 2, 4, 32, 64])
@pytest.mark.parametrize("block_size", [128, 256, 512])
def test_mhc_big_fuse_matches_reference(
    num_tokens: int, n_splits: int, block_size: int
) -> None:
    if not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip("NVIDIA CUDA required")
    from tokenspeed_kernel.thirdparty.cuda.mhc import mhc_big_fuse

    torch.manual_seed(7)
    device = torch.device("cuda")
    hidden_size = 4096
    projection = torch.randn(
        n_splits, num_tokens, 24, device=device, dtype=torch.float32
    )
    square_sum = torch.rand(
        n_splits, num_tokens, device=device, dtype=torch.float32
    ) * (4 * hidden_size)
    scale = torch.randn(3, device=device, dtype=torch.float32) * 0.1
    base = torch.randn(24, device=device, dtype=torch.float32) * 0.1
    residual = torch.randn(
        num_tokens, 4, hidden_size, device=device, dtype=torch.bfloat16
    )
    layer = torch.empty(num_tokens, hidden_size, device=device, dtype=torch.bfloat16)
    post = torch.empty(num_tokens, 4, device=device, dtype=torch.float32)
    comb = torch.empty(num_tokens, 16, device=device, dtype=torch.float32)
    mhc_big_fuse(
        projection,
        square_sum,
        scale,
        base,
        residual,
        layer,
        post,
        comb,
        hidden_size,
        1e-6,
        1e-6,
        20,
        n_splits,
        num_tokens,
        block_size=block_size,
        enable_pdl=False,
    )
    expected_layer, expected_post, expected_comb = reference(
        projection, square_sum, scale, base, residual, 1e-6, 1e-6, 20
    )
    torch.testing.assert_close(layer, expected_layer, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(post, expected_post, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(
        comb.view(num_tokens, 4, 4), expected_comb, rtol=2e-4, atol=2e-4
    )


@pytest.mark.parametrize("num_tokens", [1, 4, 8])
@pytest.mark.parametrize("n_splits", [32, 64])
@pytest.mark.parametrize("block_size", [128, 256, 512])
def test_mhc_big_fuse_norm_matches_reference(
    num_tokens: int, n_splits: int, block_size: int
) -> None:
    if not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip("NVIDIA CUDA required")
    from tokenspeed_kernel.thirdparty.cuda.mhc import mhc_big_fuse

    torch.manual_seed(17)
    device = torch.device("cuda")
    hidden_size = 4096
    projection = torch.randn(
        n_splits, num_tokens, 24, device=device, dtype=torch.float32
    )
    square_sum = torch.rand(
        n_splits, num_tokens, device=device, dtype=torch.float32
    ) * (4 * hidden_size)
    scale = torch.randn(3, device=device, dtype=torch.float32) * 0.1
    base = torch.randn(24, device=device, dtype=torch.float32) * 0.1
    residual = torch.randn(
        num_tokens, 4, hidden_size, device=device, dtype=torch.bfloat16
    )
    norm_weight = torch.randn(hidden_size, device=device, dtype=torch.bfloat16)
    layer = torch.empty(num_tokens, hidden_size, device=device, dtype=torch.bfloat16)
    post = torch.empty(num_tokens, 4, device=device, dtype=torch.float32)
    comb = torch.empty(num_tokens, 16, device=device, dtype=torch.float32)
    mhc_big_fuse(
        projection,
        square_sum,
        scale,
        base,
        residual,
        layer,
        post,
        comb,
        hidden_size,
        1e-6,
        1e-6,
        20,
        n_splits,
        num_tokens,
        norm_weight=norm_weight,
        norm_eps=1e-6,
        block_size=block_size,
        enable_pdl=False,
    )
    expected_layer, expected_post, expected_comb = reference(
        projection, square_sum, scale, base, residual, 1e-6, 1e-6, 20
    )
    expected_layer = torch.nn.functional.rms_norm(
        expected_layer, (hidden_size,), norm_weight, 1e-6
    )
    torch.testing.assert_close(layer, expected_layer, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(post, expected_post, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(
        comb.view(num_tokens, 4, 4), expected_comb, rtol=2e-4, atol=2e-4
    )


def test_mhc_big_fuse_rejects_unsupported_splits() -> None:
    if not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip("NVIDIA CUDA required")
    from tokenspeed_kernel.thirdparty.cuda.mhc import mhc_big_fuse

    device = torch.device("cuda")
    hidden_size = 4096
    n_splits = 3
    projection = torch.zeros(n_splits, 1, 24, device=device, dtype=torch.float32)
    square_sum = torch.ones(n_splits, 1, device=device, dtype=torch.float32)
    scale = torch.ones(3, device=device, dtype=torch.float32)
    base = torch.zeros(24, device=device, dtype=torch.float32)
    residual = torch.zeros(1, 4, hidden_size, device=device, dtype=torch.bfloat16)
    layer = torch.empty(1, hidden_size, device=device, dtype=torch.bfloat16)
    post = torch.empty(1, 4, device=device, dtype=torch.float32)
    comb = torch.empty(1, 16, device=device, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="unsupported n_splits=3"):
        mhc_big_fuse(
            projection,
            square_sum,
            scale,
            base,
            residual,
            layer,
            post,
            comb,
            hidden_size,
            1e-6,
            1e-6,
            20,
            n_splits,
            1,
            block_size=512,
            enable_pdl=False,
        )


@pytest.mark.parametrize(
    ("supported", "n_splits", "expected"),
    [
        (None, 5, True),
        (frozenset({4, 64}), 64, True),
        (frozenset({4, 64}), 5, False),
    ],
)
def test_pre_reduce_apply_capability_gate(
    supported: frozenset[int] | None, n_splits: int, expected: bool
) -> None:
    def implementation() -> None:
        pass

    if supported is not None:
        implementation.supported_n_splits = supported
    assert _pre_reduce_apply_is_supported(implementation, n_splits) is expected


@pytest.mark.parametrize(
    ("use_pre_reduce_apply", "has_norm_weight", "supports_fused_norm", "expected"),
    [
        (True, True, True, True),
        (False, True, True, False),
        (True, False, True, False),
        (True, True, False, False),
    ],
)
def test_pre_reduce_apply_fused_norm_gate(
    use_pre_reduce_apply: bool,
    has_norm_weight: bool,
    supports_fused_norm: bool,
    expected: bool,
) -> None:
    def implementation() -> None:
        pass

    implementation.supports_fused_norm = supports_fused_norm
    assert (
        _pre_reduce_apply_fuses_norm(
            implementation,
            use_pre_reduce_apply,
            has_norm_weight,
        )
        is expected
    )
