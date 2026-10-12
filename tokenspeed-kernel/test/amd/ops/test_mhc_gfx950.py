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
import torch.nn.functional as F
from tokenspeed_kernel.ops.residual import mhc_pre as kernel_mhc_pre
from utils import assert_no_triton_compile, is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required for the mHC test", allow_module_level=True)


def _reference(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens, hc_mult, hidden_size = residual.shape
    flat = residual.float().view(num_tokens, hc_mult * hidden_size)
    inv_rms = torch.rsqrt(flat.square().mean(dim=-1, keepdim=True) + rms_eps)
    mixes = F.linear(flat, fn) * inv_rms
    pre_raw, post_raw, comb_raw = torch.split(mixes, [4, 4, 16], dim=-1)
    pre = torch.sigmoid(pre_raw * hc_scale[0] + hc_base[:4]) + hc_eps
    post = torch.sigmoid(post_raw * hc_scale[1] + hc_base[4:8]) * 2.0
    comb = torch.softmax(
        comb_raw.view(num_tokens, hc_mult, hc_mult) * hc_scale[2]
        + hc_base[8:].view(1, hc_mult, hc_mult),
        dim=-1,
    )
    comb = comb + hc_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    for _ in range(1, sinkhorn_iters):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    layer_input = (pre.unsqueeze(-1) * residual.float()).sum(dim=1)
    return layer_input.to(torch.bfloat16), post.unsqueeze(-1), comb


@pytest.mark.parametrize("num_tokens", [8, 16, 32, 64])
@pytest.mark.parametrize("hidden_size", [4096, 7168])
def test_gluon_mhc_pre_multitoken_matches_reference(
    num_tokens: int, hidden_size: int
) -> None:
    """The GFX950 fused reduction remains accurate through graph batch 32."""
    generator = torch.Generator(device="cuda").manual_seed(123 + num_tokens)
    residual = torch.randn(
        num_tokens,
        4,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    fn = (
        torch.randn(
            24,
            4 * hidden_size,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
        * 0.01
    )
    hc_scale = torch.tensor([0.7, 1.1, 0.5], device="cuda", dtype=torch.float32)
    hc_base = (
        torch.randn(24, device="cuda", dtype=torch.float32, generator=generator) * 0.01
    )
    args = (residual, fn, hc_scale, hc_base, 1e-6, 1e-6, 20)

    actual = kernel_mhc_pre(*args, norm_weight=None, norm_eps=None)
    expected = _reference(*args)

    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            actual_tensor.float(), expected_tensor.float(), rtol=2e-2, atol=2e-2
        )


def test_glm5_next_mhc_pre_graph_shape_replays_changed_input() -> None:
    generator = torch.Generator(device="cuda").manual_seed(456)
    residual = torch.randn(
        64,
        4,
        4096,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    fn = (
        torch.randn(
            24,
            4 * 4096,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
        * 0.01
    )
    hc_scale = torch.tensor([0.7, 1.1, 0.5], device="cuda", dtype=torch.float32)
    hc_base = (
        torch.randn(24, device="cuda", dtype=torch.float32, generator=generator) * 0.01
    )
    args = (residual, fn, hc_scale, hc_base, 1e-6, 1e-6, 20)

    kernel_mhc_pre(*args, norm_weight=None, norm_eps=None)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = kernel_mhc_pre(*args, norm_weight=None, norm_eps=None)

    residual.copy_(torch.randn(residual.shape, device="cuda", dtype=residual.dtype))
    expected = _reference(*args)
    graph.replay()
    torch.cuda.synchronize()

    for actual_tensor, expected_tensor in zip(graph_output, expected, strict=True):
        torch.testing.assert_close(
            actual_tensor.float(), expected_tensor.float(), rtol=2e-2, atol=2e-2
        )


def _prefill_args(num_tokens: int, hidden_size: int) -> tuple[object, ...]:
    generator = torch.Generator(device="cuda").manual_seed(num_tokens + hidden_size)
    residual = torch.randn(
        num_tokens,
        4,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    fn = (
        torch.randn(
            24,
            4 * hidden_size,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
        * 0.01
    )
    hc_scale = torch.tensor([0.7, 1.1, 0.5], device="cuda", dtype=torch.float32)
    hc_base = torch.zeros(24, device="cuda", dtype=torch.float32)
    return residual, fn, hc_scale, hc_base, 1e-6, 1e-6, 20


@pytest.mark.parametrize(
    ("num_tokens", "hidden_size"),
    [
        (tokens, hidden)
        for tokens in (257, 1024, 1025, 2048, 2049, 4096, 4097, 8192, 8193)
        for hidden in (4096, 7168)
    ],
)
def test_gluon_mhc_large_prefill_matches_reference(
    num_tokens: int, hidden_size: int
) -> None:
    args = _prefill_args(num_tokens, hidden_size)
    actual = kernel_mhc_pre(*args, norm_weight=None, norm_eps=None)
    expected = _reference(*args)
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            actual_tensor.float(), expected_tensor.float(), rtol=2e-2, atol=2e-2
        )


@pytest.mark.parametrize(("hidden_size", "n_splits"), [(4096, 4), (7168, 1)])
def test_gluon_mhc_prefill_projection_split_counts(
    hidden_size: int, n_splits: int
) -> None:
    from tokenspeed_kernel_amd.ops.gfx950 import mhc

    residual, fn, *_ = _prefill_args(65, hidden_size)
    projection = torch.empty(n_splits, 65, 24, device="cuda", dtype=torch.float32)
    square_sum = torch.empty(n_splits, 65, device="cuda", dtype=torch.float32)
    mhc.launch_gluon_mhc_prefill_project_gfx950(
        residual,
        fn,
        projection,
        square_sum,
        n_splits=n_splits,
        block_m=64,
        block_k=256,
    )

    flat = residual.float().view(65, 4 * hidden_size)
    split_k = 4 * hidden_size // n_splits
    for split in range(n_splits):
        part = flat[:, split * split_k : (split + 1) * split_k]
        weight = fn[:, split * split_k : (split + 1) * split_k]
        torch.testing.assert_close(
            projection[split], F.linear(part, weight), rtol=2e-3, atol=2e-3
        )
        torch.testing.assert_close(
            square_sum[split], part.square().sum(dim=-1), rtol=1e-4, atol=1e-3
        )


def test_gluon_mhc_prefill_token_count_does_not_recompile() -> None:
    from tokenspeed_kernel_amd.ops.gfx950 import mhc

    residual, fn, *_ = _prefill_args(320, 4096)

    def launch(num_tokens: int) -> None:
        projection = torch.empty(8, num_tokens, 24, device="cuda", dtype=torch.float32)
        square_sum = torch.empty(8, num_tokens, device="cuda", dtype=torch.float32)
        mhc.launch_gluon_mhc_prefill_project_gfx950(
            residual[:num_tokens],
            fn,
            projection,
            square_sum,
            n_splits=8,
            block_m=32,
            block_k=256,
        )

    launch(257)
    launch(272)
    with assert_no_triton_compile(mhc.gluon_mhc_prefill_project_gfx950):
        for num_tokens in (258, 288, 320):
            launch(num_tokens)


@pytest.mark.parametrize("hidden_size", [4096, 7168])
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("strided_weight", [False, True])
def test_gluon_mhc_prefill_output_normalization(
    hidden_size, weight_dtype, strided_weight
) -> None:
    args = _prefill_args(257, hidden_size)
    norm_weight = torch.randn(
        hidden_size * (2 if strided_weight else 1), dtype=weight_dtype, device="cuda"
    )
    if strided_weight:
        norm_weight = norm_weight[::2]
    actual = kernel_mhc_pre(
        *args,
        norm_weight=norm_weight,
        norm_eps=1e-5,
    )
    layer, post, comb = _reference(*args)
    layer = F.rms_norm(layer, (hidden_size,), norm_weight, 1e-5)
    for result, expected in zip(actual, (layer, post, comb), strict=True):
        torch.testing.assert_close(
            result.float(), expected.float(), rtol=2e-2, atol=2e-2
        )


@pytest.mark.parametrize("normalize", [False, True])
def test_gluon_mhc_prefill_graph_changes_all_inputs(normalize) -> None:
    args = _prefill_args(257, 7168)
    residual, fn, scale, bias, *_ = args
    norm_weight = (
        torch.ones(7168, dtype=torch.bfloat16, device="cuda") if normalize else None
    )
    norm_eps = 1e-5 if normalize else None
    kernel_mhc_pre(*args, norm_weight=norm_weight, norm_eps=norm_eps)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = kernel_mhc_pre(*args, norm_weight=norm_weight, norm_eps=norm_eps)
    residual.copy_(torch.randn_like(residual))
    fn.mul_(0.75)
    scale.mul_(1.2)
    bias.add_(0.1)
    if normalize:
        norm_weight.mul_(0.8)
    expected = list(_reference(*args))
    if normalize:
        expected[0] = F.rms_norm(expected[0], (7168,), norm_weight, norm_eps)
    graph.replay()
    for result, correct in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            result.float(), correct.float(), rtol=2e-2, atol=2e-2
        )


def test_gluon_mhc_prefill_downstream_runtime_splits_do_not_recompile() -> None:
    from tokenspeed_kernel_amd.ops.gfx950 import mhc

    def launch(tokens, splits):
        projection = torch.randn(splits, tokens, 24, device="cuda")
        square_sum = torch.ones(splits, tokens, device="cuda")
        scale = torch.ones(3, device="cuda")
        bias = torch.zeros(24, device="cuda")
        pre = torch.empty(tokens, 4, device="cuda")
        post = torch.empty_like(pre)
        comb = torch.empty(tokens, 16, device="cuda")
        mhc.launch_gluon_mhc_prefill_mix_gfx950(
            projection,
            square_sum,
            scale,
            bias,
            pre,
            post,
            comb,
            hidden_size=4096,
            rms_eps=1e-6,
            hc_eps=1e-6,
            sinkhorn_iters=20,
            n_splits=splits,
            num_tokens=tokens,
        )

    launch(257, 8)
    launch(273, 4)
    with assert_no_triton_compile(mhc.gluon_mhc_prefill_mix_gfx950):
        for tokens, splits in ((258, 1), (289, 2), (320, 4), (301, 8)):
            launch(tokens, splits)
