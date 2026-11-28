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

"""GPU selection preserves the original Torch/AOK row arithmetic."""

from test.utils import assert_no_triton_compile

import pytest
import torch
from tokenspeed_kernel.ops.sampling.full_vocab import (
    FullVocabScores,
    FullVocabTopK,
    full_vocab_active_rows,
)
from tokenspeed_kernel.ops.sampling.torch import torch_full_vocab_active_rows
from tokenspeed_kernel.ops.sampling.triton.full_vocab import _full_vocab_active_rows
from tokenspeed_kernel.thirdparty.cuda import full_vocab_scoring

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not full_vocab_scoring.available(),
    reason="requires the built Torch 2.9.0 selected-row CUDA library",
)


def _bits(actual, expected):
    assert torch.equal(
        actual.contiguous().view(torch.int32), expected.contiguous().view(torch.int32)
    )


def _scorer():
    # The tests run with the configured batch-invariant provider. The optional
    # AOK registration is supplied by the deployment test environment.
    return FullVocabTopK(
        k=128, vocab_block=32768, form="negative-cross-entropy", solution=None
    )


def _reference(logits, scorer):
    values, ids = torch.topk(logits, scorer.k, dim=-1)
    maximum = logits.max(dim=-1).values
    shifted = logits - maximum[:, None]
    partials = scorer.plan.kernel(shifted, block_size=scorer.vocab_block)
    total = partials[:, 0]
    for block in range(1, partials.shape[1]):
        total = total + partials[:, block]
    log_z = total.log()
    head = -(log_z[:, None] - shifted.gather(1, ids))
    tail = torch.logsumexp(logits.scatter(1, ids, -torch.inf), dim=-1) - (
        maximum + log_z
    )
    return FullVocabScores(
        shifted,
        log_z,
        ids,
        head,
        tail,
        scorer.form,
        scorer.vocab_block,
        scorer.plan.kernel.name,
    )


@pytest.mark.parametrize("style", ["fp32", "bf16", "ties", "signed_zero"])
def test_empty_radix_prefix_blocks_refresh_in_graph(style):
    torch.manual_seed(37)
    logits = torch.full((8, 163840), -20.0, device="cuda")
    band = torch.randn((8, 1024), device="cuda") + 4
    if style == "bf16":
        band = band.bfloat16().float()
    elif style == "ties":
        band.fill_(2)
    elif style == "signed_zero":
        band.zero_()
        band[:, ::2] = -0.0
    active = torch.arange(8, device="cuda") % 2 == 0
    logits[:, :1024].copy_(band)
    full_vocab_scoring.topk(logits, active, 128)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        values, ids = full_vocab_scoring.topk(logits, active, 128)
    for offset in [0, 65536, 162816]:
        logits.fill_(-20)
        logits[:, offset : offset + 1024].copy_(band)
        active.logical_not_()
        expected_values, expected_ids = torch.topk(logits, 128, dim=-1)
        graph.replay()
        assert torch.equal(ids[active], expected_ids[active])
        _bits(values[active], expected_values[active])
        assert torch.count_nonzero(ids[~active]) == 0
        assert torch.count_nonzero(values[~active]) == 0


@pytest.mark.parametrize(
    ("rows", "vocab"),
    [(8, 163840), (8, 163841), (8, 164863), (224, 167935), (224, 167936)],
)
@pytest.mark.parametrize("ties", [False, True])
def test_radix_vector_loads_preserve_partial_chunks_on_graph_replay(rows, vocab, ties):
    torch.manual_seed(71)
    logits = torch.randn((rows, vocab), device="cuda")
    if ties:
        logits.round_()
    original = logits.clone()
    active = torch.arange(rows, device="cuda") % 3 == 0
    full_vocab_scoring.topk(logits, active, 128)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        values, ids = full_vocab_scoring.topk(logits, active, 128)
    for shift in [0, 3, 1031]:
        logits.copy_(original.roll(shift, dims=1))
        active.logical_not_()
        expected_values, expected_ids = torch.topk(logits, 128, dim=-1)
        graph.replay()
        assert torch.equal(ids[active], expected_ids[active])
        _bits(values[active], expected_values[active])
        assert torch.count_nonzero(ids[~active]) == 0
        assert torch.count_nonzero(values[~active]) == 0


@pytest.mark.parametrize("rows", [1, 8, 32, 64, 128, 256])
@pytest.mark.parametrize(
    "style", ["random", "bf16", "ties", "signed_zero", "large_offset", "extreme"]
)
def test_selected_rows_match_original_bits(rows, style):
    torch.manual_seed(37)
    logits = torch.randn(rows, 163840, device="cuda")
    if style == "bf16":
        logits = logits.bfloat16().float()
    elif style == "ties":
        logits = logits.round()
    elif style == "large_offset":
        logits = (logits + 1e6).bfloat16().float()
    elif style == "extreme":
        logits.mul_(30)
    elif style == "signed_zero":
        logits.zero_()
        logits[:, ::2] = -0.0
    scorer = _scorer()
    expected = _reference(logits, scorer)
    complete = scorer.prepare(logits)
    assert torch.equal(complete.ids, expected.ids)
    _bits(complete.shifted, expected.shifted)
    _bits(complete.log_z, expected.log_z)
    _bits(complete.logprobs, expected.logprobs)
    _bits(complete.log_tail, expected.log_tail)
    active = torch.arange(rows, device="cuda") % 3 == 0
    pending = scorer.begin(logits, defer_head=True)
    # A sampler may overwrite its original logits after begin. Retaining an
    # alias here would change selected token IDs and scores after verification.
    logits.add_(19.0)
    selected = pending.complete(active)
    assert torch.equal(selected.ids[active], expected.ids[active])
    _bits(selected.logprobs[active], expected.logprobs[active])
    _bits(selected.log_tail[active], expected.log_tail[active])
    targets = torch.arange(rows, device="cuda")
    for actual, reference in zip(
        selected.pack(targets, prediction_rows=None),
        expected.pack(targets, prediction_rows=None),
        strict=True,
    ):
        _bits(actual[active], reference[active])


def test_maximum_reuse_preserves_first_zero_and_nonfinite_values():
    logits = torch.full((5, 163840), -1.0, device="cuda")
    logits[0, 0] = -0.0
    logits[0, 1] = 0.0
    logits[1, 0] = 0.0
    logits[1, 1] = -0.0
    logits[2, 1] = torch.inf
    logits[3, 1] = torch.nan
    logits[4].fill_(-torch.inf)
    values = torch.topk(logits, 128, dim=-1).values
    _bits(full_vocab_scoring.maximum(logits, values), logits.max(dim=-1).values)


@pytest.mark.parametrize("rows", [8, 64, 256])
@pytest.mark.parametrize(
    "style", ["random", "signed_zero", "underflow", "nan", "large_offset"]
)
def test_tail_final_projection_keeps_torch_bits_on_graph_replay(rows, style):
    torch.manual_seed(41)
    shifted = -torch.rand(rows, 163840, device="cuda") * 20
    if style == "signed_zero":
        shifted.zero_()
        shifted[:, ::2] = -0.0
    elif style == "underflow":
        shifted.fill_(-torch.inf)
        shifted[:, :128] = 0
    elif style == "nan":
        shifted[:, -1] = torch.nan
    elif style == "large_offset":
        shifted.add_(1e6)
    ids = torch.arange(128, device="cuda").expand(rows, -1).contiguous()
    log_z = torch.linspace(0, 20, rows, device="cuda")
    row_maximum = shifted.max(dim=-1).values
    active = torch.ones(rows, device="cuda", dtype=torch.bool)

    def run():
        return full_vocab_scoring.tail(shifted, ids, log_z, row_maximum, active)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    for remainder in [0, 1, 2]:
        active.copy_(torch.arange(rows, device="cuda") % 3 != remainder)
        log_z.add_(0.5)
        graph.replay()
        expected = torch.logsumexp(shifted.scatter(1, ids, -torch.inf), dim=-1)
        expected = expected - (row_maximum + log_z)
        _bits(actual[active], expected[active])


def test_graph_replay_refreshes_acceptance_without_host_sync():
    scorer = _scorer()
    rows, width = 32, 8
    logits = torch.randn(rows, 163840, device="cuda").bfloat16().float()
    targets = torch.arange(rows, device="cuda")
    lengths = torch.tensor([2, 3, 1, 8], dtype=torch.int32, device="cuda")
    pools = torch.tensor([1, 2, 3, 99], dtype=torch.int64, device="cuda")
    kwargs = dict(
        num_prefill_outputs=0,
        num_extends=0,
        decode_width=width,
        padding_index=99,
        prediction_rows=None,
    )

    def run():
        pending = scorer.begin(logits, defer_head=True)
        ordinary = pending.sampled_logprobs(targets, prediction_rows=None)
        active = full_vocab_active_rows(lengths, pools, **kwargs)
        complete = pending.complete_accepted(lengths, pools, **kwargs)
        return complete.pack(targets, prediction_rows=None), ordinary, active

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        run()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.cuda.set_sync_debug_mode("error")
        try:
            packed, ordinary, active = run()
        finally:
            torch.cuda.set_sync_debug_mode("default")
    for accepted in [[2, 3, 1, 8], [0, 0, 0, 0], [8, 1, 4, 0]]:
        lengths.copy_(torch.tensor(accepted, dtype=torch.int32, device="cuda"))
        logits.copy_(torch.randn_like(logits).bfloat16().float())
        with assert_no_triton_compile(_full_vocab_active_rows):
            graph.replay()
        reference = _reference(logits, scorer)
        expected_active = torch_full_vocab_active_rows(lengths, pools, **kwargs)
        assert torch.equal(active, expected_active)
        for actual, expected in zip(
            packed, reference.pack(targets, prediction_rows=None), strict=True
        ):
            _bits(actual[active], expected[active])
        _bits(ordinary, reference.sampled_logprobs(targets, prediction_rows=None))


def test_empty_selected_batch_and_normalizer_slices():
    scorer = _scorer()
    logits = torch.empty((0, 163840), device="cuda")
    result = scorer.prepare(logits)
    assert result.ids.shape == (0, 128)
    logits = torch.randn((8, 163840), device="cuda")
    pending = scorer.begin(logits, defer_head=True)
    tokens = torch.arange(8, device="cuda")
    _bits(
        pending.rows(slice(2, 5)).sampled_logprobs(tokens[2:5], prediction_rows=None),
        pending.sampled_logprobs(tokens, prediction_rows=None)[2:5],
    )
    sliced = pending.rows(slice(2, 5)).complete(
        torch.ones(3, dtype=torch.bool, device="cuda")
    )
    _bits(sliced.log_tail, _reference(logits[2:5], scorer).log_tail)


def test_deferred_tree_rows_keep_packed_output_order():
    scorer = _scorer()
    logits = torch.randn((32, 163840), device="cuda").bfloat16().float()
    expected = _reference(logits, scorer)
    lengths = torch.tensor([3, 2, 0, 8], dtype=torch.int32, device="cuda")
    pools = torch.tensor([1, 2, 3, 99], device="cuda")
    local = torch.tensor([0, 4, 2, 3, 4, 5, 6, 7], device="cuda")
    path = (local + torch.arange(4, device="cuda")[:, None] * 8).flatten()
    tokens = torch.arange(32, device="cuda")
    result = scorer.begin(logits, defer_head=True).complete_accepted(
        lengths,
        pools,
        num_prefill_outputs=0,
        num_extends=0,
        decode_width=8,
        padding_index=99,
        prediction_rows=path,
    )
    output_mask = (
        (torch.arange(8, device="cuda")[None] < lengths[:, None])
        & (pools[:, None] != 99)
    ).flatten()
    for actual, reference in zip(
        result.pack(tokens, prediction_rows=path),
        expected.pack(tokens, prediction_rows=path),
        strict=True,
    ):
        _bits(actual[output_mask], reference[output_mask])


@pytest.mark.parametrize(
    "form,zero_bits", [("direct", 0), ("negative-cross-entropy", -2147483648)]
)
def test_native_selected_rows_preserve_zero_logprob_bits(form, zero_bits):
    logits = torch.full((8, 163840), -200.0, device="cuda")
    logits[:, 0].zero_()
    scorer = FullVocabTopK(k=128, vocab_block=32768, form=form, solution=None)
    active = torch.ones(8, dtype=torch.bool, device="cuda")
    targets = torch.zeros(8, dtype=torch.int64, device="cuda")
    for scores in [
        scorer.prepare(logits),
        scorer.begin(logits, defer_head=True).complete(active),
    ]:
        packed, sampled = scores.pack(targets, prediction_rows=None)
        expected = torch.full((8,), zero_bits, dtype=torch.int32, device="cuda")
        assert torch.equal(sampled.view(torch.int32), expected)
        assert torch.equal(
            scores.logprobs[:, 0].contiguous().view(torch.int32), expected
        )
        assert torch.equal(packed[:, 130], expected)
