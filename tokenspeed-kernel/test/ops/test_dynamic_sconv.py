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

"""Dynamic head coefficients on the existing short-convolution state path."""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch
from tokenspeed_kernel.ops.conv import inkling_ring_sconv


def _reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    prefix: torch.Tensor,
    dynamic: torch.Tensor | None,
    output_scale: float,
) -> torch.Tensor:
    joined = torch.cat([prefix, x], dim=0).float()
    rows = []
    width = weight.shape[-1]
    for t in range(x.shape[0]):
        acc = torch.zeros(x.shape[1], dtype=torch.float32)
        for tap in range(width):
            lag = tap if dynamic is not None else width - 1 - tap
            value = joined[width - 1 + t - lag]
            coefficient = weight[:, tap].float()
            if dynamic is not None:
                coefficient = coefficient + dynamic[
                    t, :, tap
                ].float().repeat_interleave(x.shape[1] // dynamic.shape[1])
            else:
                coefficient = coefficient.to(x.dtype).float()
            if dynamic is not None and tap == 0:
                acc = value * coefficient
            else:
                # FP64 evaluates the FP32 product and sum before one FP32 round.
                acc = (value.double() * coefficient.double() + acc.double()).float()
        rows.append(((acc + x[t].float()) * output_scale).to(x.dtype))
    return torch.stack(rows)


def _case(device: str, prefill: bool, dynamic: bool, channels: int = 256) -> tuple[
    dict[str, Any],
    torch.Tensor,
    torch.Tensor,
    list[int],
    torch.Tensor | None,
    torch.Tensor,
]:
    generator = torch.Generator(device="cpu").manual_seed(118)
    head_dim, width, ring_size = 128, 4, 9
    lengths = [83, 2, 1] if prefill else [2, 2, 2]
    prefix_lengths = [5, 0, 4]
    total = sum(lengths)
    x_cpu = torch.randn((total, channels), generator=generator).to(torch.bfloat16)
    static_native = torch.randn((width, channels), generator=generator).to(
        torch.bfloat16
    )
    # Dynamic native weights use [lag, channel], exposed without a copy.
    weight = static_native.T if dynamic else static_native.T.contiguous()
    dynamic_cpu = (
        torch.randn((total, channels // head_dim, width), generator=generator).to(
            torch.bfloat16
        )
        if dynamic
        else None
    )
    cache = torch.randn((4, ring_size, channels), generator=generator).to(
        torch.bfloat16
    )
    checkpoint = torch.randn((8, width - 1, channels), generator=generator).to(
        torch.bfloat16
    )
    prefix = checkpoint[3].clone()
    for j in range(width - 1):
        cache[1, (prefix_lengths[0] - (width - 1) + j) % ring_size] = prefix[j]
    cu = torch.tensor(
        [0, lengths[0], lengths[0] + lengths[1], total], dtype=torch.int32
    )
    sequence = torch.repeat_interleave(
        torch.arange(3, dtype=torch.int32), torch.tensor(lengths)
    )
    table = torch.tensor([[3, 4], [0, 5], [6, 7]], dtype=torch.int32)
    arguments = {
        "x": x_cpu.to(device),
        "weight": weight.to(device),
        "conv_cache": cache.to(device),
        "cu_seqlens": cu.to(device),
        "seq_idx": sequence.to(device),
        "cache_indices": torch.tensor([1, 2, -1], dtype=torch.int32, device=device),
        "has_initial_state": torch.tensor([True, False, True], device=device),
        "seq_lens": torch.tensor(
            [a + b for a, b in zip(lengths, prefix_lengths)],
            dtype=torch.int32,
            device=device,
        ),
        "block_table": table.to(device),
        "ckpt_a": checkpoint.to(device),
        "ckpt_b": None,
        "num_extends": 3 if prefill else 0,
        "publish_live": True,
        "dynamic_weights": None if dynamic_cpu is None else dynamic_cpu.to(device),
        "output_scale": 1 / math.sqrt(9 if dynamic else 5),
        "token_range": None,
        "activation": None,
        "use_residual": True,
    }
    return arguments, x_cpu, prefix, lengths, dynamic_cpu, checkpoint


@pytest.mark.parametrize("prefill", [False, True])
@pytest.mark.parametrize("dynamic", [False, True])
# 4608 channels: wide-stream prefill tiles span several heads.
@pytest.mark.parametrize("channels", [256, 4608])
def test_scaled_dynamic_sconv_cuda(prefill: bool, dynamic: bool, channels: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm device unavailable")
    arguments, x_cpu, prefix, lengths, coefficients, checkpoint = _case(
        "cuda", prefill, dynamic, channels
    )
    before_ring = arguments["conv_cache"].clone()
    output = inkling_ring_sconv(**arguments)
    start = 0
    for request, length in enumerate(lengths):
        stop = start + length
        history = prefix if request == 0 else torch.zeros_like(prefix)
        expected = _reference(
            x_cpu[start:stop],
            arguments["weight"].cpu(),
            history,
            None if coefficients is None else coefficients[start:stop],
            arguments["output_scale"],
        )
        torch.testing.assert_close(output[start:stop].cpu(), expected, rtol=0, atol=0)
        if request < 2:
            expected_tail = torch.cat([history, x_cpu[start:stop]], dim=0)[-3:]
            torch.testing.assert_close(
                arguments["ckpt_a"][4 + request].cpu(), expected_tail, rtol=0, atol=0
            )
            final_position = int(arguments["seq_lens"][request]) - 1
            torch.testing.assert_close(
                arguments["conv_cache"][request + 1, final_position % 9].cpu(),
                x_cpu[stop - 1],
                rtol=0,
                atol=0,
            )
        start = stop
    torch.testing.assert_close(
        arguments["conv_cache"][0], before_ring[0], rtol=0, atol=0
    )
    torch.testing.assert_close(
        arguments["conv_cache"][3], before_ring[3], rtol=0, atol=0
    )
    torch.testing.assert_close(
        arguments["ckpt_a"][7].cpu(), checkpoint[7], rtol=0, atol=0
    )


@pytest.mark.parametrize("prefill", [False, True])
@pytest.mark.parametrize("channels", [256, 4608])
def test_fp32_coefficients_act_as_their_input_dtype_rounding(
    prefill: bool, channels: int
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm device unavailable")
    arguments, _, _, _, coefficients, _ = _case("cuda", prefill, True, channels)
    # FP32 coefficients over the same BF16 bits, low bits rounding both ways and to ties.
    generator = torch.Generator(device="cpu").manual_seed(7)
    low = torch.randint(0, 2**16, coefficients.shape, generator=generator)
    low[..., 0] = 2**15
    wide = (
        (coefficients.float().view(torch.int32) + low.to(torch.int32))
        .view(torch.float32)
        .cuda()
    )
    rounded = wide.to(torch.bfloat16)
    assert not torch.equal(rounded.float(), wide)
    # The ring and the published checkpoints must match as well.
    states = {name: arguments[name].clone() for name in ("conv_cache", "ckpt_a")}
    expected = inkling_ring_sconv(**(arguments | {"dynamic_weights": rounded}))
    expected_states = {name: arguments[name].clone() for name in states}
    for name, value in states.items():
        arguments[name].copy_(value)
    output = inkling_ring_sconv(**(arguments | {"dynamic_weights": wide}))
    assert torch.equal(output.view(torch.int16), expected.view(torch.int16))
    for name, value in expected_states.items():
        assert torch.equal(arguments[name].view(torch.int16), value.view(torch.int16))


def test_dynamic_coefficients_change_under_cuda_graph_replay() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm device unavailable")
    arguments, _, _, _, _, _ = _case("cuda", True, True)
    for _ in range(3):
        inkling_ring_sconv(**arguments)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = inkling_ring_sconv(**arguments)
    before = output.clone()
    arguments["dynamic_weights"].add_(0.125)
    graph.replay()
    torch.cuda.synchronize()
    expected = inkling_ring_sconv(**arguments)
    assert not torch.equal(output, before)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


def test_dynamic_metadata_and_empty_rows_cpu() -> None:
    arguments, _, _, _, _, _ = _case("cpu", True, False)
    tokens = arguments["x"].shape[0]
    with pytest.raises(ValueError, match="heads dividing D"):
        inkling_ring_sconv(
            **(arguments | {"dynamic_weights": torch.zeros((tokens, 3, 4))})
        )
    with pytest.raises(ValueError, match="output scale"):
        inkling_ring_sconv(**(arguments | {"output_scale": float("nan")}))
    with pytest.raises(ValueError, match="coefficients"):
        inkling_ring_sconv(**(arguments | {"dynamic_weights": torch.zeros((1, 2, 4))}))
    empty = arguments | {
        "x": torch.empty((0, 256), dtype=torch.bfloat16),
        "dynamic_weights": torch.empty((0, 2, 4), dtype=torch.bfloat16),
    }
    assert inkling_ring_sconv(**empty).shape == (0, 256)
