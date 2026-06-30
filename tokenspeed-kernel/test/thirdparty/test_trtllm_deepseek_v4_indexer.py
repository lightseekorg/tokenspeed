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
from tokenspeed_kernel.ops.attention.triton.deepseek_v4 import (
    deepseek_v4_fused_indexer_q_rope_hadamard_mxfp4,
)
from tokenspeed_kernel.ops.attention.trtllm.deepseek_v4 import (
    has_trtllm_deepseek_v4_indexer_q_prepare,
    supports_trtllm_deepseek_v4_indexer_q_prepare,
    trtllm_deepseek_v4_indexer_q_prepare_mxfp4,
)
from tokenspeed_kernel.thirdparty.cuda.trtllm_deepseek_v4_indexer import (
    _load_module,
    trtllm_fused_cat_fp4,
    trtllm_mla_rope_inplace,
)


def _is_supported() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] == 10
        and has_trtllm_deepseek_v4_indexer_q_prepare()
    )


pytestmark = pytest.mark.skipif(
    not _is_supported(),
    reason="TRT-LLM DeepSeek-V4 indexer kernels require NVIDIA SM100+",
)


def _cos_sin_cache(max_position: int = 64) -> torch.Tensor:
    positions = torch.arange(max_position, device="cuda", dtype=torch.float32)
    frequencies = torch.exp(
        -torch.arange(32, device="cuda", dtype=torch.float32) / 32 * 8.0
    )
    angles = positions[:, None] * frequencies[None, :]
    return torch.cat((angles.cos(), angles.sin()), dim=-1).contiguous()


def _rope_reference(
    data: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> torch.Tensor:
    out = data.float().clone()
    cos_sin = cos_sin_cache[positions.long()]
    cosine = cos_sin[:, None, :32]
    sine = cos_sin[:, None, 32:]
    even = out[..., 64::2].clone()
    odd = out[..., 65::2].clone()
    out[..., 64::2] = even * cosine - odd * sine
    out[..., 65::2] = even * sine + odd * cosine
    return out.to(torch.bfloat16)


def _fp4_reference(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    blocks = rows.float().reshape(-1, 4, 32)
    amax = blocks.abs().amax(dim=-1).clamp_min(1.0e-4)
    exponent = torch.ceil(torch.log2(amax / 6.0))
    scaled = blocks / torch.pow(2.0, exponent).unsqueeze(-1)
    absolute = scaled.abs().clamp_max(6.0)
    code = sum(
        (absolute > boundary).to(torch.uint8)
        for boundary in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
    )
    code |= ((scaled < 0) & (code != 0)).to(torch.uint8) << 3
    packed = (code[..., 0::2] | (code[..., 1::2] << 4)).reshape(-1, 64)
    scales = (exponent + 127).to(torch.uint8).contiguous().view(torch.int32)
    return packed, scales


@pytest.mark.parametrize("enable_pdl", [False, True])
def test_mla_rope_inplace_matches_reference(enable_pdl: bool) -> None:
    torch.manual_seed(101)
    data = torch.randn(7, 64, 128, device="cuda", dtype=torch.bfloat16)
    positions = torch.tensor([0, 1, 3, 7, 15, 31, 62], device="cuda")
    cos_sin_cache = _cos_sin_cache()
    expected = _rope_reference(data, positions, cos_sin_cache)

    results = []
    for position_dtype in (torch.int32, torch.int64):
        actual = data.clone()
        trtllm_mla_rope_inplace(
            actual,
            positions.to(position_dtype),
            cos_sin_cache,
            enable_pdl=enable_pdl,
        )
        torch.testing.assert_close(actual, expected, atol=2.0e-4, rtol=0)
        assert torch.equal(actual[..., :64], data[..., :64])
        results.append(actual)

    assert torch.equal(results[0], results[1])


def test_fused_cat_fp4_matches_deepgemm_layout() -> None:
    torch.manual_seed(102)
    rows = torch.randn(7, 64, 128, device="cuda", dtype=torch.bfloat16)
    rows[0].zero_()
    rows[1].fill_(1.0e-5)
    first, second = rows.split((64, 64), dim=-1)

    packed, scales = trtllm_fused_cat_fp4(first, second)
    expected_packed, expected_scales = _fp4_reference(rows)

    assert torch.equal(packed, expected_packed)
    assert torch.equal(scales, expected_scales)


@pytest.mark.parametrize("invalid_input", ["first", "second"])
def test_fused_cat_fp4_rejects_non_row_linear_input(invalid_input: str) -> None:
    backing = torch.empty(2048, device="cuda", dtype=torch.bfloat16)
    non_row_linear = torch.as_strided(
        backing,
        size=(2, 3, 64),
        stride=(400, 128, 1),
    )
    rows = torch.empty(2, 3, 128, device="cuda", dtype=torch.bfloat16)
    first, second = rows.split((64, 64), dim=-1)
    if invalid_input == "first":
        first = non_row_linear
    else:
        second = non_row_linear

    with pytest.raises(
        ValueError,
        match=rf"{invalid_input} leading dimensions must be row-linear",
    ):
        trtllm_fused_cat_fp4(first, second)


def test_fused_cat_fp4_ffi_rejects_non_row_linear_input() -> None:
    backing = torch.empty(2048, device="cuda", dtype=torch.bfloat16)
    first = torch.as_strided(
        backing,
        size=(2, 3, 64),
        stride=(400, 128, 1),
    )
    rows = torch.empty(2, 3, 128, device="cuda", dtype=torch.bfloat16)
    _, second = rows.split((64, 64), dim=-1)
    packed = torch.empty(6, 64, device="cuda", dtype=torch.uint8)
    scales = torch.empty(6, 1, device="cuda", dtype=torch.int32)

    with pytest.raises(RuntimeError, match="leading dimensions must be row-linear"):
        _load_module().trtllm_deepseek_v4_fused_cat_fp4(
            first,
            second,
            packed,
            scales,
        )


@pytest.mark.parametrize("position_dtype", [torch.int32, torch.int64])
def test_indexer_q_composite_matches_fused_triton(position_dtype: torch.dtype) -> None:
    torch.manual_seed(103)
    index_q = torch.randn(7, 64, 128, device="cuda", dtype=torch.bfloat16)
    positions = torch.tensor([0, 1, 3, 7, 15, 31, 62], device="cuda")
    positions = positions.to(position_dtype)
    cos_sin_cache = _cos_sin_cache()
    weights = torch.randn(7, 64, device="cuda", dtype=torch.float32)
    kwargs = {
        "positions": positions,
        "cos_sin_cache": cos_sin_cache,
        "weights": weights,
        "softmax_scale": 0.25,
        "head_scale": 64**-0.5,
    }

    (expected_q, expected_scales), expected_weights = (
        deepseek_v4_fused_indexer_q_rope_hadamard_mxfp4(
            index_q=index_q,
            **kwargs,
        )
    )
    (actual_q, actual_scales), actual_weights = (
        trtllm_deepseek_v4_indexer_q_prepare_mxfp4(
            index_q=index_q.clone(),
            enable_pdl=True,
            **kwargs,
        )
    )

    assert torch.equal(actual_q, expected_q)
    assert torch.equal(actual_scales, expected_scales)
    assert torch.equal(actual_weights, expected_weights)


def test_indexer_q_composite_cuda_graph_replay() -> None:
    torch.manual_seed(104)
    index_q = torch.randn(32, 64, 128, device="cuda", dtype=torch.bfloat16)
    graph_q = index_q.clone()
    positions = torch.arange(32, device="cuda", dtype=torch.int64)
    cos_sin_cache = _cos_sin_cache()
    weights = torch.randn(32, 64, device="cuda", dtype=torch.float32)
    kwargs = {
        "positions": positions,
        "cos_sin_cache": cos_sin_cache,
        "weights": weights,
        "softmax_scale": 0.25,
        "head_scale": 64**-0.5,
        "enable_pdl": True,
    }

    warmup_stream = torch.cuda.Stream()
    with torch.cuda.stream(warmup_stream):
        trtllm_deepseek_v4_indexer_q_prepare_mxfp4(
            index_q=index_q.clone(),
            **kwargs,
        )
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = trtllm_deepseek_v4_indexer_q_prepare_mxfp4(
            index_q=graph_q,
            **kwargs,
        )

    graph_q.copy_(index_q)
    graph.replay()
    expected = trtllm_deepseek_v4_indexer_q_prepare_mxfp4(
        index_q=index_q.clone(),
        **kwargs,
    )
    torch.cuda.synchronize()

    assert torch.equal(graph_output[0][0], expected[0][0])
    assert torch.equal(graph_output[0][1], expected[0][1])
    assert torch.equal(graph_output[1], expected[1])


def test_indexer_q_composite_empty_and_off_shape_contracts() -> None:
    index_q = torch.empty(0, 64, 128, device="cuda", dtype=torch.bfloat16)
    positions = torch.empty(0, device="cuda", dtype=torch.int64)
    cos_sin_cache = _cos_sin_cache()
    weights = torch.empty(0, 64, device="cuda", dtype=torch.float32)

    assert supports_trtllm_deepseek_v4_indexer_q_prepare(index_q)
    (packed, scales), scaled_weights = trtllm_deepseek_v4_indexer_q_prepare_mxfp4(
        index_q=index_q,
        positions=positions,
        cos_sin_cache=cos_sin_cache,
        weights=weights,
        softmax_scale=0.25,
        head_scale=64**-0.5,
        enable_pdl=True,
    )
    assert packed.shape == (0, 64, 64)
    assert scales.shape == (0, 64)
    assert scaled_weights.shape == (0, 64)

    assert not supports_trtllm_deepseek_v4_indexer_q_prepare(
        torch.empty(1, 63, 128, device="cuda", dtype=torch.bfloat16)
    )
    assert not supports_trtllm_deepseek_v4_indexer_q_prepare(
        torch.empty(1, 64, 128, device="cuda", dtype=torch.float16)
    )
    assert not supports_trtllm_deepseek_v4_indexer_q_prepare(
        torch.empty(1, 64, 256, device="cuda", dtype=torch.bfloat16)[..., ::2]
    )
    misaligned = torch.empty(
        64 * 128 + 1,
        device="cuda",
        dtype=torch.bfloat16,
    )[
        1:
    ].view(1, 64, 128)
    assert misaligned.is_contiguous()
    assert misaligned.data_ptr() % 16 != 0
    assert not supports_trtllm_deepseek_v4_indexer_q_prepare(misaligned)

    assert not supports_trtllm_deepseek_v4_indexer_q_prepare(
        torch.empty(1, 64, 128, device="cuda", dtype=torch.bfloat16),
        torch.empty(1, device="cpu", dtype=torch.int64),
        cos_sin_cache,
    )
