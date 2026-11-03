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

"""CUDA top-p no-truncation normalization and reusable graph state."""

import pytest
import torch
from tokenspeed_kernel.numerics.torch.sampling import torch_topk_topp_renorm
from tokenspeed_kernel.ops.sampling.cuda import (
    chain_speculative_sampling_target_only,
    fused_topk_topp_prepare,
    fused_topk_topp_renorm,
    fused_topk_topp_workspace_size,
)
from tokenspeed_kernel.ops.sampling.flashinfer import softmax
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform().is_nvidia, reason="CUDA sampling kernels require NVIDIA"
)

_TOP_K_DISABLED = 1 << 30


@pytest.mark.parametrize("rows,vocab", [(4, 8191), (128, 248320), (192, 248320)])
@pytest.mark.parametrize("enable_pdl", [False, True])
def test_uncut_rows_still_renormalize(rows: int, vocab: int, enable_pdl: bool) -> None:
    # This small FP32 mass error must be removed even when no tokens are cut.
    # All additions are exact, so the reference also pins the output rounding.
    probs = torch.zeros((rows, vocab), device="cuda", dtype=torch.float32)
    values = torch.tensor([0.5, 0.25, 0.125, 0.125 + 2**-20], device="cuda")
    probs[:, [0, 1, vocab // 2, vocab - 1]] = values
    original = probs.clone()
    inv = torch.tensor(1.0 / (1.0 + 2**-20), device="cuda", dtype=torch.float32)
    expected = probs * inv
    out = torch.full_like(probs, float("nan"))
    ks = torch.full((rows,), _TOP_K_DISABLED, device="cuda", dtype=torch.int32)
    ps = torch.ones(rows, device="cuda", dtype=torch.float32)
    got = fused_topk_topp_renorm(
        probs, ks, ps, out=out, enable_pdl=enable_pdl, in_place_skip=False
    )
    assert got.data_ptr() == out.data_ptr()
    assert torch.equal(probs, original)
    assert not torch.equal(got, original)
    torch.testing.assert_close(got, expected, atol=0, rtol=0)


@pytest.mark.parametrize("vocab,offset", [(8191, 0), (248320, 1)])
@pytest.mark.parametrize("enable_pdl", [False, True])
def test_mixed_top_p_graph_reuses_workspace(
    vocab: int, offset: int, enable_pdl: bool
) -> None:
    torch.manual_seed(35)
    rows = 8
    storage = torch.empty(rows * vocab + offset, device="cuda")
    probs = storage[offset:].view(rows, vocab)
    probs.copy_(torch.softmax(torch.randn_like(probs) * 3, dim=-1))
    original = probs.clone()
    ks = torch.full((rows,), _TOP_K_DISABLED, device="cuda", dtype=torch.int32)
    ps = torch.full((rows,), 0.8, device="cuda", dtype=torch.float32)
    out = torch.empty_like(probs)
    workspace = torch.full(
        (fused_topk_topp_workspace_size(rows, vocab),),
        0xA5,
        dtype=torch.uint8,
        device="cuda",
    )
    fused_topk_topp_prepare(probs.device)

    def run() -> torch.Tensor:
        return fused_topk_topp_renorm(
            probs,
            ks,
            ps,
            workspace=workspace,
            out=out,
            enable_pdl=enable_pdl,
            in_place_skip=False,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            run()
    stream.synchronize()

    # The same captured pointers transition from active radix to no radix,
    # mixed top-k/top-p/greedy rows, then back to active radix.
    for k_values, p_values in [
        ([_TOP_K_DISABLED] * rows, [0.8] * rows),
        ([_TOP_K_DISABLED] * rows, [1.0] * rows),
        (
            [1, _TOP_K_DISABLED, 64, _TOP_K_DISABLED, 128, 1, 2, _TOP_K_DISABLED],
            [1.0, 1.0, 1.0, 0.9, 0.7, 0.5, 0.8, 0.5],
        ),
        ([_TOP_K_DISABLED] * rows, [0.8] * rows),
    ]:
        ks.copy_(torch.tensor(k_values, dtype=torch.int32, device="cuda"))
        ps.copy_(torch.tensor(p_values, dtype=torch.float32, device="cuda"))
        out.fill_(float("nan"))
        graph.replay()
        actual = out.clone()
        eager = fused_topk_topp_renorm(
            probs, ks, ps, enable_pdl=enable_pdl, in_place_skip=False
        )
        reference = torch_topk_topp_renorm(probs, ks, ps)
        assert torch.equal(actual, eager)
        assert torch.equal(probs, original)
        torch.testing.assert_close(
            actual.sum(-1), torch.ones(rows, device="cuda"), atol=2e-6, rtol=0
        )
        assert ((actual > 0) != (reference > 0)).sum(-1).max().item() <= 1
        torch.testing.assert_close(actual, reference, atol=1e-5, rtol=1e-4)
        uncut = (ks == _TOP_K_DISABLED) & (ps == 1)
        normalized = (
            probs[uncut].double() / probs[uncut].double().sum(-1, keepdim=True)
        ).float()
        torch.testing.assert_close(actual[uncut], normalized, atol=0, rtol=2e-6)


@pytest.mark.parametrize("bs", [1, 32, 48])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("enable_pdl", [False, True])
def test_uncut_graph_producer_and_verifier(
    bs: int, dtype: torch.dtype, enable_pdl: bool
) -> None:
    """Changing inputs must reach the verifier through the shortened radix path."""
    vocab, width = 248320, 4
    source = torch.full((bs * width, vocab), -torch.inf, dtype=dtype, device="cuda")
    source[:, [17, vocab - 5]] = 0
    logits = torch.empty_like(source)
    probs = torch.empty_like(source, dtype=torch.float32)
    normalized = torch.empty_like(probs)
    top_ks = torch.full(
        (bs * width,), _TOP_K_DISABLED, dtype=torch.int32, device="cuda"
    )
    top_ps = torch.ones(bs * width, device="cuda")
    candidates = torch.full((bs, width), 17, dtype=torch.int32, device="cuda")
    coins = torch.full((bs, width), 0.25, device="cuda")
    final_coins = torch.full((bs,), 0.75, device="cuda")
    predicts = torch.empty(bs * width, dtype=torch.int32, device="cuda")
    indices = torch.empty((bs, width), dtype=torch.int32, device="cuda")
    lengths = torch.empty(bs, dtype=torch.int32, device="cuda")
    workspace = torch.empty(
        fused_topk_topp_workspace_size(bs * width, vocab),
        dtype=torch.uint8,
        device="cuda",
    )
    fused_topk_topp_prepare(source.device)

    def run() -> None:
        predicts.fill_(-1)
        indices.fill_(-1)
        logits.copy_(source)
        softmax(logits, out=probs, enable_pdl=enable_pdl)
        fused_topk_topp_renorm(
            probs,
            top_ks,
            top_ps,
            workspace=workspace,
            out=normalized,
            enable_pdl=enable_pdl,
            in_place_skip=False,
        )
        # No intervening operation may serialize away the consumer's PDL wait.
        chain_speculative_sampling_target_only(
            predicts=predicts,
            accept_index=indices,
            accept_token_num=lengths,
            candidates=candidates,
            uniform_samples=coins,
            uniform_samples_for_final_sampling=final_coins,
            target_probs=normalized.view(bs, width, vocab),
            draft_probs=None,
            threshold_single=1.0,
            threshold_acc=1.0,
            deterministic=True,
            enable_pdl=enable_pdl,
            use_draft_prob=False,
            reject_draft_prob_threshold=1.0,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            run()
    stream.synchronize()

    failures = []
    for step in range(16):
        first, second = 17 + step % 3, vocab - 5 - step % 3
        source.fill_(-torch.inf)
        source[:, [first, second]] = 0
        candidates.fill_(first)
        coins.fill_(0.25 if step % 2 == 0 else 0.75)
        expected_probs = torch.zeros_like(normalized)
        expected_probs[:, [first, second]] = 0.5
        expected_tokens = torch.full_like(predicts, -1).view(bs, width)
        expected_indices = torch.full_like(indices, -1)
        if step % 2 == 0:
            expected_tokens[:, :3] = first
            expected_tokens[:, 3] = second
            expected_indices.copy_(
                torch.arange(bs * width, device="cuda").view(bs, width)
            )
            expected_length = 3
        else:
            expected_tokens[:, 0] = second
            expected_indices[:, 0] = torch.arange(bs, device="cuda") * width
            expected_length = 0
        graph.replay()
        failures.append(
            torch.stack(
                [
                    (normalized != expected_probs).any(),
                    (predicts.view(bs, width) != expected_tokens).any(),
                    (indices != expected_indices).any(),
                    (lengths != expected_length).any(),
                ]
            )
        )
    assert not torch.stack(failures).any().item()
