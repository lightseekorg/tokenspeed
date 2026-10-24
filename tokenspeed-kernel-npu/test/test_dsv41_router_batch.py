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

"""No padding work: independent references, old native path and mutable graphs."""

import pytest
import torch
import torch_npu  # noqa: F401
from tokenspeed_kernel_npu.ops.dsv41_router import (
    _sqrt_softplus_topk_kernel,
    ascend_sqrt_softplus_topk,
)


def original(logits, bias, normalize):
    scores = torch.sqrt(torch.nn.functional.softplus(logits.float()))
    n = logits.shape[0]
    weights = torch.empty(n, 6, device=logits.device, dtype=torch.float32)
    ids = torch.empty(n, 6, device=logits.device, dtype=torch.int32)
    if n:
        _sqrt_softplus_topk_kernel[(n,)](
            scores,
            bias,
            None,
            None,
            weights,
            ids,
            scores,
            *scores.stride(),
            0 if bias is None else bias.stride(0),
            0,
            0,
            1.0,
            EXPERTS=384,
            TOPK=6,
            BLOCK_E=512,
            BLOCK_K=8,
            HASH=False,
            BIAS=bias is not None,
            RENORMALIZE=normalize,
            NEED_SCORES=False,
            PRECOMPUTED=True,
            num_warps=4,
        )
    return weights, ids, scores


@pytest.mark.parametrize("n", [0, 1, 31, 32, 33, 48, 96, 192, 193])
@pytest.mark.parametrize("normalize", [False, True])
def test_batch_matches_original_and_cpu_topk(n, normalize):
    torch.npu.set_device(0)
    torch.manual_seed(917)
    logits = torch.randn(n, 384, device="npu")
    bias = torch.randn(768, device="npu")[::2] * 0.01
    # Keep an actual strided bias after the arithmetic above.
    storage = torch.empty(768, device="npu")
    storage[::2].copy_(bias)
    bias = storage[::2]
    for correction in [None, bias.contiguous(), bias]:
        actual = ascend_sqrt_softplus_topk(
            logits, 6, normalize, correction, None, None, True, 1.0, torch.float32
        )
        expected = original(logits, correction, normalize)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, atol=0, rtol=0, equal_nan=True)
        scores = expected[2].cpu()
        choices = scores if correction is None else scores + correction.cpu()
        cpu_ids = torch.argsort(choices, dim=-1, descending=True, stable=True)[:, :6]
        torch.testing.assert_close(actual[1].cpu().long(), cpu_ids, atol=0, rtol=0)


@pytest.mark.parametrize("n", [1, 6, 33, 192])
def test_mutable_graph_ties_nan_and_extreme_scores(n):
    torch.npu.set_device(0)
    logits = torch.zeros(n, 384, device="npu")
    bias = torch.zeros(384, device="npu")

    def forward():
        return ascend_sqrt_softplus_topk(
            logits, 6, True, bias, None, None, True, 1.0, torch.float32
        )

    for _ in range(3):
        forward()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = forward()
    for kind in ["ties", "random", "nan", "extremes"]:
        x = torch.zeros(n, 384)
        if kind == "random":
            x.normal_()
        if kind == "nan":
            x[:, [7, 23]] = float("nan")
        if kind == "extremes":
            x[:, 0] = float("inf")
            x[:, 1] = float("-inf")
            x[:, 2] = 100.0
            x[:, 3] = -100.0
        logits.copy_(x)
        graph.replay()
        torch.npu.synchronize()
        for a, b in zip(actual, original(logits, bias, True)):
            torch.testing.assert_close(a, b, atol=0, rtol=0, equal_nan=True)


@pytest.mark.parametrize("n", [1, 33])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("routing", ["plain", "bias", "hash"])
def test_registered_route_scaling_and_output_dtype(n, dtype, normalize, routing):
    from tokenspeed_kernel.ops.moe import moe_topk

    torch.npu.set_device(0)
    torch.manual_seed(931)
    logits = torch.randn(n, 384, device="npu")
    bias = torch.randn(384, device="npu") if routing == "bias" else None
    table = (
        torch.arange(48, device="npu", dtype=torch.int32).reshape(8, 6)
        if routing == "hash"
        else None
    )
    inputs = (
        torch.arange(n, device="npu", dtype=torch.int64) % 8
        if table is not None
        else None
    )

    def run():
        return moe_topk(
            logits,
            6,
            "sqrt_softplus",
            "hash" if table is not None else "topk",
            normalize,
            2.5,
            correction_bias=bias,
            hash_indices_table=table,
            input_ids=inputs,
            topk_weights_dtype=dtype,
            override="ascend_sqrt_softplus_topk",
        )

    run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        weights, ids = run()
    for _ in range(2):
        logits.normal_()
        graph.replay()
        torch.npu.synchronize()
        scores = torch.sqrt(torch.nn.functional.softplus(logits.float())).cpu()
        expected_ids = (
            table[inputs].cpu().long()
            if table is not None
            else torch.argsort(
                scores if bias is None else scores + bias.cpu(),
                descending=True,
                stable=True,
            )[:, :6]
        )
        expected = scores.gather(1, expected_ids)
        if normalize:
            expected /= expected.sum(-1, keepdim=True)
        expected = (expected * 2.5).to(dtype)
        assert weights.dtype == dtype
        torch.testing.assert_close(ids.cpu().long(), expected_ids, atol=0, rtol=0)
        torch.testing.assert_close(
            weights.cpu(),
            expected,
            atol=1e-6 if dtype == torch.float32 else 0.008,
            rtol=1e-6 if dtype == torch.float32 else 0.008,
        )


@pytest.mark.parametrize("invalid", [-1, 8])
def test_range_check_rejects_invalid_graph_replay_in_child(invalid):
    # A device assertion poisons its context, so isolate the expected failure.
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(f"""
        import torch
        import torch_npu
        from tokenspeed_kernel_npu.ops.dsv41_router import assert_indices_in_range
        x = torch.tensor([0, 7], device="npu", dtype=torch.int64)
        assert_indices_in_range(x, 8)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            assert_indices_in_range(x, 8)
        graph.replay()
        torch.npu.synchronize()
        print("VALID_REPLAY", flush=True)
        x[0] = {invalid}
        try:
            graph.replay()
            torch.npu.synchronize()
        except RuntimeError:
            print("INVALID_REJECTED", flush=True)
        else:
            raise AssertionError("invalid replay was accepted")
    """)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "VALID_REPLAY" in result.stdout
    assert "INVALID_REJECTED" in result.stdout


@pytest.mark.parametrize("n", [1, 6, 33])
@pytest.mark.parametrize("strided", [False, True])
def test_bias_ordering_across_sign_zero_infinity_and_nan(n, strided):
    torch.npu.set_device(0)
    logits = torch.zeros(n, 384, device="npu")
    scores = torch.sqrt(torch.nn.functional.softplus(logits.float()))
    backing = torch.zeros(768, device="npu")
    bias = backing[::2] if strided else backing[:384]

    def forward():
        return ascend_sqrt_softplus_topk(
            logits, 6, True, bias, None, None, True, 1.0, torch.float32
        )

    forward()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = forward()
    for kind in ("negative", "zero", "infinity", "nan", "subnormal"):
        logits.zero_()
        values = torch.linspace(-20, -2, 384)
        if kind == "subnormal":
            logits.fill_(-1000)
            values = (torch.arange(384) - 192).float() * torch.nextafter(
                torch.tensor(0.0), torch.tensor(1.0)
            )
        elif kind == "zero":
            values.fill_(-scores[0, 0].item())
            values[7] = torch.nextafter(values[7], torch.tensor(float("inf")))
            values[9] = torch.nextafter(values[9], torch.tensor(float("-inf")))
        elif kind == "infinity":
            values.fill_(float("-inf"))
            values[[2, 5]] = float("inf")
        elif kind == "nan":
            values[[3, 11, 12]] = float("nan")
            values[0] = float("inf")
        bias.copy_(values)
        graph.replay()
        torch.npu.synchronize()
        reference_scores = torch.sqrt(
            torch.nn.functional.softplus(logits.float())
        ).cpu()
        choices = reference_scores + values
        expected = torch.argsort(choices, descending=True, stable=True)[:, :6]
        torch.testing.assert_close(actual[1].cpu().long(), expected, atol=0, rtol=0)
        weights = reference_scores.gather(1, expected)
        weights /= weights.sum(-1, keepdim=True).clamp_min(
            torch.finfo(torch.float32).tiny
        )
        torch.testing.assert_close(actual[0].cpu(), weights, atol=0, rtol=0)
