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

"""Fused full-vocabulary helpers retain the original FP32 bit patterns."""

from test.utils import assert_no_triton_compile

import pytest
import torch
from tokenspeed_kernel.ops.sampling import FullVocabTopK
from tokenspeed_kernel.ops.sampling.full_vocab import (
    FullVocabScores,
    full_vocab_active_rows,
)
from tokenspeed_kernel.ops.sampling.torch import (
    torch_block_sumexp,
    torch_full_vocab_active_rows,
    torch_full_vocab_head,
    torch_full_vocab_pack,
)
from tokenspeed_kernel.ops.sampling.triton.full_vocab import (
    _full_vocab_active_rows,
    _full_vocab_head,
    _full_vocab_pack,
    _full_vocab_sum,
    triton_full_vocab_head,
    triton_full_vocab_pack,
    triton_full_vocab_sum,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")


def _bits_equal(actual, expected):
    assert torch.equal(
        actual.contiguous().view(torch.int32), expected.contiguous().view(torch.int32)
    )


def _reference(logits, *, k, vocab_block, form):
    # The pre-fusion arithmetic: keep the block provider, torch log, masked
    # logsumexp and subtraction order, including torch's own Top-K tie order.
    logits = logits.float()
    ids = torch.topk(logits, k, dim=-1).indices
    maximum = logits.max(dim=-1).values
    shifted = logits - maximum[:, None]
    partials = torch_block_sumexp(shifted, block_size=vocab_block)
    total = partials[:, 0]
    for block in range(1, partials.shape[1]):
        total = total + partials[:, block]
    log_z = total.log()
    selected = shifted.gather(1, ids)
    head = (
        selected - log_z[:, None] if form == "direct" else -(log_z[:, None] - selected)
    )
    tail = torch.logsumexp(logits.scatter(1, ids, -torch.inf), dim=-1) - (
        maximum + log_z
    )
    return shifted, log_z, ids, head, tail


@pytest.mark.parametrize("blocks", [1, 4, 7, 16])
def test_fold_retains_left_to_right_rounding(blocks):
    values = torch.rand((129, blocks * 3), device="cuda")[:, ::3]
    if blocks >= 4:
        # A balanced or reordered sum loses a different set of low bits.
        values[:, :4] = torch.tensor([1.0, 2**-24, 2**-24, 2**-24], device="cuda")
    values[0].zero_()
    values[1].fill_(-0.0)
    values[2].fill_(torch.finfo(torch.float32).tiny / 2)
    expected = values[:, 0]
    for column in range(1, blocks):
        expected = expected + values[:, column]
    _bits_equal(triton_full_vocab_sum(values), expected)
    with assert_no_triton_compile(_full_vocab_sum):
        for rows in (1, 3, 16, 17, 65):
            _bits_equal(triton_full_vocab_sum(values[:rows]), expected[:rows])


@pytest.mark.parametrize("form", ["direct", "negative-cross-entropy"])
@pytest.mark.parametrize("kind", ["random", "ties", "offset", "signed_zero", "bf16"])
def test_scoring_and_packing_match_original_bits(form, kind):
    torch.manual_seed(29)
    logits = torch.randn((17, 32768), device="cuda")
    if kind == "ties":
        logits.zero_()
    elif kind == "offset":
        logits.fill_(1e8)
    elif kind == "signed_zero":
        logits.fill_(-200)
        logits[:, 0] = 0
    elif kind == "bf16":
        logits = logits.bfloat16()
    original = logits.clone()
    shifted, log_z, ids, head, tail = _reference(
        logits, k=128, vocab_block=8192, form=form
    )
    scorer = FullVocabTopK(k=128, vocab_block=8192, form=form, solution="torch")
    scores = scorer.prepare(logits)
    assert torch.equal(scores.ids, ids)
    for actual, expected in (
        (scores.shifted, shifted),
        (scores.log_z, log_z),
        (scores.logprobs, head),
        (scores.log_tail, tail),
    ):
        _bits_equal(actual, expected)
    _bits_equal(logits.float(), original.float())
    tokens = torch.arange(17, device="cuda")
    tokens[-2:] = torch.tensor([-1, 40000], device="cuda")
    # Repeated tree rows and invalid padded token IDs retain their old bytes.
    path = torch.arange(16, -1, -1, device="cuda") // 2
    for mapping in (None, path):
        expected = torch_full_vocab_pack(
            shifted,
            log_z,
            ids,
            head,
            tail,
            tokens,
            prediction_rows=mapping,
            form=form,
        )
        packed, sampled = scores.pack(tokens, prediction_rows=mapping)
        assert torch.equal(packed, expected[:, :-1])
        _bits_equal(sampled, expected[:, -1].view(torch.float32))


@pytest.mark.parametrize("form", ["direct", "negative-cross-entropy"])
def test_replay_reads_changed_logits_and_tree_rows(form):
    torch.manual_seed(31)
    logits = torch.randn((17, 1024), device="cuda")
    tokens = torch.arange(17, device="cuda")
    path = torch.arange(17, device="cuda")
    scorer = FullVocabTopK(k=8, vocab_block=256, form=form, solution="torch")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            scorer.prepare(logits).pack(tokens, prediction_rows=path)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        scores = scorer.prepare(logits)
        packed, sampled = scores.pack(tokens, prediction_rows=path)
    for _ in range(3):
        logits.normal_()
        path.copy_(torch.randperm(17, device="cuda"))
        tokens.random_(0, 1024)
        graph.replay()
        reference = _reference(logits, k=8, vocab_block=256, form=form)
        expected = torch_full_vocab_pack(
            *reference,
            tokens,
            prediction_rows=path,
            form=form,
        )
        assert torch.equal(packed, expected[:, :-1])
        _bits_equal(sampled, expected[:, -1].view(torch.float32))


@pytest.mark.parametrize("materialized_head", [False, True])
def test_pack_reuses_kernel_for_counts_and_strides(materialized_head):
    values = torch.randn((33, 64), device="cuda")
    reference = _reference(values, k=4, vocab_block=16, form="direct")
    if not materialized_head:
        reference = (*reference[:3], None, reference[4])
    tokens = torch.arange(33, device="cuda")
    paths = torch.arange(33, device="cuda")
    for has_path in (False, True):
        mapping = paths if has_path else None
        triton_full_vocab_pack(
            *reference, tokens, prediction_rows=mapping, form="direct"
        )
        with assert_no_triton_compile(_full_vocab_pack):
            for count, stride in ((1, 3), (3, 2), (16, 1), (17, 2)):
                strided = torch.empty_strided(
                    (count,), (stride,), device="cuda", dtype=torch.int64
                )
                strided.copy_(tokens[:count])
                args = tuple(
                    None if value is None else value[:count] for value in reference
                )
                mapping = strided if has_path else None
                actual = triton_full_vocab_pack(
                    *args, strided, prediction_rows=mapping, form="direct"
                )
                expected = torch_full_vocab_pack(
                    *args, strided, prediction_rows=mapping, form="direct"
                )
                assert torch.equal(actual, expected)


def test_far_rows_do_not_overflow_int32_offsets():
    stride = 2**30 + 1
    if torch.cuda.mem_get_info()[0] < (2 * stride + 64) * 4 + 2**30:
        pytest.skip("requires an 8 GiB strided tensor")
    compact = torch.randn((3, 64), device="cuda")
    shifted = torch.empty_strided((3, 64), (stride, 1), device="cuda")
    shifted.copy_(compact)
    partials = shifted[:, :4]
    expected = compact[:, 0] + compact[:, 1]
    expected = expected + compact[:, 2]
    expected = expected + compact[:, 3]
    _bits_equal(triton_full_vocab_sum(partials), expected)
    _, log_z, ids, head, tail = _reference(compact, k=4, vocab_block=16, form="direct")
    tokens = torch.tensor([0, 17, 63], device="cuda")
    _bits_equal(
        triton_full_vocab_head(shifted, ids, log_z, form="direct"),
        torch_full_vocab_head(compact, ids, log_z, form="direct"),
    )
    actual = triton_full_vocab_pack(
        shifted, log_z, ids, head, tail, tokens, prediction_rows=None, form="direct"
    )
    expected = torch_full_vocab_pack(
        compact, log_z, ids, head, tail, tokens, prediction_rows=None, form="direct"
    )
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("form", ["direct", "negative-cross-entropy"])
def test_fused_head_preserves_bits_and_does_not_specialize_runtime_strides(form):
    torch.manual_seed(43)
    values = torch.randn((17, 64), device="cuda")
    values[:, :2] = torch.tensor([0.0, -0.0], device="cuda")
    values[:, 2] = torch.inf
    values[:, 3] = torch.nan
    ids = torch.arange(8, device="cuda").expand(17, -1).contiguous()
    log_z = torch.zeros(17, device="cuda")
    triton_full_vocab_head(values, ids, log_z, form=form)
    with assert_no_triton_compile(_full_vocab_head):
        for rows, stride in [(1, 1), (3, 2), (16, 3), (17, 1)]:
            shifted = torch.empty_strided(
                (rows, 64), (64 * stride, stride), device="cuda"
            )
            shifted.copy_(values[:rows])
            actual = triton_full_vocab_head(
                shifted, ids[:rows], log_z[:rows], form=form
            )
            expected = torch_full_vocab_head(
                shifted, ids[:rows], log_z[:rows], form=form
            )
            _bits_equal(actual, expected)


@pytest.mark.parametrize("form", ["direct", "negative-cross-entropy"])
@pytest.mark.parametrize("materialized_head", [False, True])
def test_head_and_pack_preserve_exceptional_bits_on_graph_replay(
    form, materialized_head
):
    raw = torch.tensor(
        [
            0,
            -2147483648,
            0x7F800000,
            -0x800000,
            0x7FC01234,
            0xFFC01234 - 2**32,
            0x7F801234,
            0xFF801234 - 2**32,
        ],
        dtype=torch.int32,
        device="cuda",
    )
    values = raw.view(torch.float32).repeat(4, 1)
    ids = torch.arange(8, device="cuda").expand(4, -1).contiguous()
    log_z = torch.tensor([0.0, torch.inf, -torch.inf, torch.nan], device="cuda")
    tokens = torch.tensor([4, 2, 3, 0], device="cuda")
    tail = torch.zeros(4, device="cuda")

    def run():
        scores = FullVocabScores(
            shifted=values,
            log_z=log_z,
            ids=ids,
            _head=None,
            log_tail=tail,
            form=form,
            vocab_block=8,
            kernel_name="torch",
        )
        head = scores.logprobs
        if materialized_head:
            from dataclasses import replace

            scores = replace(scores, _head=head)
        packed, sampled = scores.pack(tokens, prediction_rows=None)
        return head, packed, sampled

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        head, packed, sampled = run()
    for _ in range(2):
        graph.replay()
        expected_head = torch_full_vocab_head(values, ids, log_z, form=form)
        expected_pack = torch_full_vocab_pack(
            values,
            log_z,
            ids,
            expected_head if materialized_head else None,
            tail,
            tokens,
            prediction_rows=None,
            form=form,
        )
        _bits_equal(head, expected_head)
        assert torch.equal(packed, expected_pack[:, :-1])
        _bits_equal(sampled, expected_pack[:, -1].view(torch.float32))
        log_z.copy_(log_z.roll(1))


@pytest.mark.parametrize("rows,k,vocab", [(0, 1, 8), (1, 1, 8), (3, 8, 8)])
@pytest.mark.parametrize("form", ["direct", "negative-cross-entropy"])
def test_empty_rows_and_full_head_preserve_bits(rows, k, vocab, form):
    logits = torch.randn((rows, vocab), device="cuda")
    tokens = torch.zeros(rows, dtype=torch.int64, device="cuda")
    expected = torch_full_vocab_pack(
        *_reference(logits, k=k, vocab_block=vocab, form=form),
        tokens,
        prediction_rows=None,
        form=form,
    )
    scorer = FullVocabTopK(k=k, vocab_block=vocab, form=form, solution="torch")
    scores = scorer.prepare(logits)
    actual, sampled = scores.pack(tokens, prediction_rows=None)
    assert torch.equal(actual, expected[:, :-1])
    _bits_equal(sampled, expected[:, -1].view(torch.float32))
    if k == vocab:
        assert torch.isneginf(scores.log_tail).all()


@pytest.mark.parametrize("tree", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_active_mask_handles_stops_padding_and_tree_aliases(tree, strided):
    lengths = torch.tensor([1, 0, 3, 8], dtype=torch.int32, device="cuda")
    pools = torch.tensor([7, 8, 9, 99], dtype=torch.int64, device="cuda")
    path = None
    prefill, extends = (2, 2) if not tree else (0, 0)
    width = 8
    if tree:
        # Unaccepted identity rows alias node 4, which is on the accepted path.
        local = torch.tensor([0, 4, 2, 3, 4, 5, 6, 7], device="cuda")
        path = (local + torch.arange(4, device="cuda")[:, None] * width).flatten()
    kwargs = dict(
        num_prefill_outputs=prefill,
        num_extends=extends,
        decode_width=width,
        padding_index=99,
        prediction_rows=path,
    )
    if strided:
        lengths = torch.stack((torch.zeros_like(lengths), lengths), dim=-1).flatten()[
            1::2
        ]
        pools = torch.stack((torch.zeros_like(pools), pools), dim=-1).flatten()[1::2]
        if path is not None:
            path = torch.stack((torch.zeros_like(path), path), dim=-1).flatten()[1::2]
            kwargs["prediction_rows"] = path
    expected = torch_full_vocab_active_rows(lengths, pools, **kwargs)
    actual = full_vocab_active_rows(lengths, pools, **kwargs)
    assert torch.equal(actual, expected)
    with assert_no_triton_compile(_full_vocab_active_rows):
        lengths.zero_()
        assert not bool(full_vocab_active_rows(lengths, pools, **kwargs).any())


def test_source_shift_large_row_offset_and_runtime_strides():
    import gc

    from tokenspeed_kernel.ops.sampling.triton.full_vocab import (
        _full_vocab_source_shift,
        triton_full_vocab_source_shift,
    )

    for stride in [2048, 2053]:
        logits = torch.randn((3, stride), device="cuda")[:, :1024]
        maximum = logits.max(dim=-1).values
        actual_source, actual_shifted = triton_full_vocab_source_shift(logits, maximum)
        _bits_equal(actual_source, logits)
        _bits_equal(actual_shifted, logits - maximum[:, None])
    with assert_no_triton_compile(_full_vocab_source_shift):
        logits = torch.randn((7, 2061), device="cuda")[:, :1024]
        maximum = logits.max(dim=-1).values
        source, shifted = triton_full_vocab_source_shift(logits, maximum)
        _bits_equal(source, logits)
        _bits_equal(shifted, logits - maximum[:, None])
    gc.collect()
    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info()
    if free < 9 * 2**30:
        pytest.skip("requires 9 GiB free for a >2^31-element row offset")
    logits = torch.empty_strided((2, 1024), (2**31, 1), device="cuda")
    logits[0].fill_(1.25)
    logits[1].fill_(-0.0)
    maximum = torch.tensor([0.25, -0.0], device="cuda")
    source, shifted = triton_full_vocab_source_shift(logits, maximum)
    _bits_equal(source, logits)
    _bits_equal(shifted, logits - maximum[:, None])


def test_fold_large_block_stride_does_not_wrap_int32():
    stride = 2**30 + 1
    required = (2 * stride + 2) * 4
    if torch.cuda.mem_get_info()[0] < required + 2**30:
        pytest.skip("large block-stride test needs 9 GiB free")
    values = torch.empty_strided((2, 3), (1, stride), device="cuda")
    compact = torch.tensor([[1.0, 2**-24, 2**-24], [0.25, 0.5, 1.0]], device="cuda")
    values.copy_(compact)
    expected = compact[:, 0] + compact[:, 1]
    expected = expected + compact[:, 2]
    _bits_equal(triton_full_vocab_sum(values), expected)
