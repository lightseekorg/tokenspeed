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

"""The vocab-parallel top-k primitive: shard projection, packing, selection."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=15, suite="runtime-1gpu")

from tokenspeed.runtime.layers import vocab_parallel_topk as topk_module  # noqa: E402
from tokenspeed.runtime.layers.vocab_parallel_topk import (  # noqa: E402
    VocabParallelTopK,
)


def _selector(lm_head, tp_size: int, top_k: int, max_rows: int, vocab_size: int):
    """A planned selector, with the radix probe left out."""
    selector = VocabParallelTopK.__new__(VocabParallelTopK)
    selector.lm_head = lm_head
    selector.tp_size = tp_size
    selector.tp_group = None
    selector.vocab_size = vocab_size
    selector.top_k = top_k
    selector.max_rows = max_rows
    selector.logit_scale = None
    selector.softcapping = None
    selector._enabled = True
    selector._radix_topk = None
    selector._shard_seq_lens = None
    selector._gather_buffers = None
    return selector


#: The real quant method refuses to construct off Blackwell.
_needs_nvfp4 = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10),
    reason="NVFP4 W4A16 requires SM100/SM103 and compatible FlashInfer",
)


def _nvfp4_head(shard: int, hidden: int, logits: torch.Tensor):
    """A head the production predicate accepts as genuinely quantized.

    Same shape as the NextN fixture: a packed uint8 weight plus the runtime
    attributes ``should_apply_lm_head_quant_method`` requires. Only ``apply``
    is stood in for, so the dispatch is checked against the real quant method.
    """
    from torch import nn

    from tokenspeed.runtime.layers.dense.nvfp4 import Nvfp4W4A16LinearMethod

    head = nn.Module()
    head.register_parameter(
        "weight",
        nn.Parameter(
            torch.empty((shard, hidden // 2), dtype=torch.uint8), requires_grad=False
        ),
    )
    head.register_parameter(
        "weight_scale",
        nn.Parameter(torch.ones((1,), dtype=torch.float32), requires_grad=False),
    )
    head.alpha = torch.ones((1,), dtype=torch.float32)
    head.input_size_per_partition = hidden
    head.output_size_per_partition = shard
    head.quant_method = Nvfp4W4A16LinearMethod(SimpleNamespace(group_size=16))
    head.quant_method.apply = mock.Mock(return_value=logits)
    return head


def test_a_padded_or_offset_shard_declines_the_shard_local_topk() -> None:
    """A shard-local index is a global token id only without padding."""
    head = SimpleNamespace(weight=torch.zeros(8, 4))
    exact = SimpleNamespace(
        num_org_elements=8,
        num_org_elements_padded=8,
        num_added_elements=0,
        org_vocab_start_index=8,
    )
    head.shard_indices = exact
    assert VocabParallelTopK(
        head,
        tp_size=2,
        tp_rank=1,
        tp_group=None,
        vocab_size=16,
        top_k=2,
        max_rows=4,
        logit_scale=None,
        softcapping=None,
        skip_all_gather=False,
        dp_sampling_enabled=False,
    ).enabled

    for broken in (
        SimpleNamespace(**{**vars(exact), "num_added_elements": 1}),
        SimpleNamespace(**{**vars(exact), "num_org_elements_padded": 12}),
        SimpleNamespace(**{**vars(exact), "org_vocab_start_index": 0}),
    ):
        head.shard_indices = broken
        assert not VocabParallelTopK(
            head,
            tp_size=2,
            tp_rank=1,
            tp_group=None,
            vocab_size=16,
            top_k=2,
            max_rows=4,
            logit_scale=None,
            softcapping=None,
            skip_all_gather=False,
            dp_sampling_enabled=False,
        ).enabled


@_needs_nvfp4
def test_a_quantized_head_reaches_its_quant_method_not_a_matmul() -> None:
    """A packed weight must never be matmul'd, on the fast path either."""
    from tokenspeed.runtime.layers.logits_processor import (
        should_apply_lm_head_quant_method,
    )

    torch.manual_seed(11)
    rows, hidden, shard = 3, 6, 20
    hidden_states = torch.randn(rows, hidden)
    dense = torch.randn(rows, shard)

    head = _nvfp4_head(shard, hidden, dense)
    # The production predicate, not our own opinion, is what must say "quantized".
    assert should_apply_lm_head_quant_method(head, head.quant_method)

    logits = _selector(head, 2, 4, 4, 40)._shard_logits(hidden_states)

    head.quant_method.apply.assert_called_once()
    torch.testing.assert_close(logits, dense)


def test_shard_logits_matmuls_an_unquantized_head() -> None:
    """The dense head keeps the plain matmul."""
    torch.manual_seed(12)
    rows, hidden, shard = 3, 6, 20
    weight = torch.randn(shard, hidden)
    hidden_states = torch.randn(rows, hidden)

    head = SimpleNamespace(weight=weight, quant_method=None)
    torch.testing.assert_close(
        _selector(head, 2, 4, 4, 40)._shard_logits(hidden_states),
        hidden_states @ weight.T,
    )


@_needs_nvfp4
def test_a_quantized_head_picks_candidates_through_the_padding_slice(
    monkeypatch,
) -> None:
    """The whole path on a quantized head: apply, slice, gather, select."""
    torch.manual_seed(13)
    rows, hidden, shard, pad, top_k, tp_size = 3, 6, 20, 4, 4, 2
    hidden_states = torch.randn(rows, hidden)
    # apply() answers over the padded partition; only the first `shard`
    # columns are real token ids, so the slice is what keeps ids meaningful.
    padded = torch.randn(rows, shard + pad)
    padded[:, shard:] = 1e4  # would win every slot if the slice were dropped

    head = _nvfp4_head(shard, hidden, padded)
    head.shard_indices = SimpleNamespace(
        num_org_elements=shard, org_vocab_start_index=0
    )
    selector = _selector(head, tp_size, top_k, 2 * rows, shard * tp_size)

    peer = torch.full((rows, shard), -1e4)
    peer_values, peer_ids = torch.topk(peer, top_k, dim=-1, sorted=False)
    peer_packed = torch.cat((peer_values, (peer_ids + shard).float()), dim=-1)

    def fake_all_gather(out, src, group):
        out[: src.shape[0]].copy_(src)
        out[src.shape[0] :].copy_(peer_packed)

    monkeypatch.setattr(topk_module, "all_gather_into_tensor", fake_all_gather)
    candidate_ids, _ = selector(hidden_states)

    # Every winner is a real token of this shard, never one of the pad columns
    # and never the peer's deliberately-losing rows.
    assert candidate_ids.max().item() < shard
    want = torch.topk(padded[:, :shard], top_k, dim=-1).values
    got = torch.gather(padded[:, :shard], 1, candidate_ids)
    assert sorted(got.flatten().tolist()) == sorted(want.flatten().tolist())


def test_shard_local_topk_picks_what_a_whole_vocabulary_topk_would(
    monkeypatch,
) -> None:
    """Two shard-local top-ks must agree with one top-k over the vocabulary."""
    torch.manual_seed(7)
    rows, hidden, vocab, top_k, tp_size = 3, 6, 40, 4, 2
    weight = torch.randn(vocab, hidden)
    hidden_states = torch.randn(rows, hidden)
    shard = vocab // tp_size

    head = SimpleNamespace(
        weight=weight[:shard],
        quant_method=None,
        shard_indices=SimpleNamespace(num_org_elements=shard, org_vocab_start_index=0),
    )
    selector = _selector(head, tp_size, top_k, 2 * rows, vocab)

    # Stand in for rank 1: its own shard-local top-k, packed as this rank
    # packs its own -- values then global ids, one fp32 row.
    peer = torch.matmul(hidden_states, weight[shard:].T)
    peer_values, peer_ids = torch.topk(peer, top_k, dim=-1, sorted=False)
    peer_packed = torch.cat((peer_values, (peer_ids + shard).float()), dim=-1)

    calls = []

    def fake_all_gather(out, src, group):
        calls.append(src.shape)
        out[: src.shape[0]].copy_(src)
        out[src.shape[0] :].copy_(peer_packed)

    monkeypatch.setattr(topk_module, "all_gather_into_tensor", fake_all_gather)
    candidate_ids, values = selector(hidden_states)

    expected = torch.topk(torch.matmul(hidden_states, weight.T), top_k, dim=-1)
    assert candidate_ids.dtype == torch.int64
    assert candidate_ids.tolist() == expected.indices.tolist()
    torch.testing.assert_close(values, expected.values.float())
    # One collective carrying both halves, not one per half.
    assert calls == [(rows, 2 * top_k)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_radix_shard_topk_picks_the_same_candidates_as_torch() -> None:
    """The shard top-k on the vendored single-pass radix kernel.

    ``sorted=False`` either way, so only the candidate set has to match -- but
    it has to match exactly, since these are the tokens the caller proposes.
    """
    head = SimpleNamespace(weight=torch.zeros(1, device="cuda", dtype=torch.bfloat16))
    selector = _selector(head, 8, 16, 56, 20480 * 8)
    selector._radix_topk = selector._probe_radix_topk(20480)
    if selector._radix_topk is None:
        pytest.skip("vendored radix top-k is not built here")

    torch.manual_seed(0)
    for rows in (7, 56):
        logits = torch.randn(rows, 20480, device="cuda", dtype=torch.bfloat16)
        values, ids = selector._shard_topk(logits)
        want = torch.topk(logits.float(), 16, dim=-1).values
        got = torch.gather(logits.float(), 1, ids.long())
        assert sorted(got.flatten().tolist()) == sorted(want.flatten().tolist())
        torch.testing.assert_close(values.float(), got, atol=1e-2, rtol=0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
